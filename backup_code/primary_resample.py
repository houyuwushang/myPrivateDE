import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
from functools import partial
import threading
import queue
import numpy as np
from typing import Optional

# --- PyTorch and JAX Management ---
# 确保 PyTorch 可以使用 GPU
# Ensure PyTorch can use the GPU
import torch
import torch.nn.functional as F


# 将 JAX 强制置于 CPU 模式，为 PyTorch 释放 GPU
# Force JAX to CPU mode to free up GPU for PyTorch
os.environ['JAX_PLATFORMS'] = 'cpu'

# 将 src 目录添加到 Python 搜索路径中
# Add the src directory to the Python search path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 仅在需要时导入 JAX
# Import JAX only when needed
try:
    import jax
    import jax.numpy as jnp
    print("JAX 已成功导入，将用于统计计算 (CPU)。")
    print("JAX 检测到的设备:", jax.devices())
except ImportError:
    print("警告: JAX 未安装。统计计算将回退到 NumPy。")
from genetic_sd.genetic_sd import GSDSynthesizer
from genetic_sd.utils.dataset_jax import Dataset
from genetic_sd.utils.domain import Domain, DataType, ColumnAttribute
from snsynth.transform.type_map import TypeMap
from genetic_sd.adaptive_statistics import AdaptiveChainedStatistics, Marginals

class PrivateDEGeneratorPT:
    def __init__(self, domain: Domain, preprocessor, device_spec: str = "auto"):
        self.domain = domain
        self.num_columns = len(self.domain.attrs)
        self.preprocessor = preprocessor

        mins, maxs, col_types = [], [], []
        # 使用 zip 安全地遍历列和转换器
        for col, t in zip(self.domain.attrs, self.preprocessor._transformer.transformers):
            if self.domain.is_continuous(col):
                col_types.append(0) # 0 for continuous
                mins.append(t.lower)
                maxs.append(t.upper)
            else: # Ordinal/Categorical
                col_types.append(1) # 1 for categorical/ordinal
                mins.append(t.fit_lower)
                maxs.append(t.fit_upper)

        self.col_types_list = col_types

        self._init_devices(device_spec)

        self.base_mins = torch.tensor(mins, dtype=torch.float32)
        self.base_scales = torch.tensor(maxs, dtype=torch.float32) - self.base_mins
        self.base_scales[self.base_scales == 0] = 1.0
        self.base_maxs = self.base_mins + self.base_scales
        self.base_discrete_mask = torch.tensor([ct != 0 for ct in col_types], dtype=torch.bool)

        self._device_buffers: dict[torch.device, dict[str, torch.Tensor]] = {}
        for dev in self.compute_devices:
            self._ensure_device_buffers(dev)

        print("生成器已初始化 (V50 - L∞ 定向重采样版)。")

    def _init_devices(self, device_spec: str) -> None:
        def _parse_spec(spec: str) -> list[torch.device]:
            spec = spec.strip().lower()
            if spec in {"auto", "cuda", "gpu"}:
                if torch.cuda.is_available():
                    count = torch.cuda.device_count()
                    return [torch.device(f"cuda:{idx}") for idx in range(count)]
                return [torch.device("cpu")]
            if spec in {"cpu", "cpu-only"}:
                return [torch.device("cpu")]
            devices: list[torch.device] = []
            for token in spec.split(','):
                token = token.strip()
                if not token:
                    continue
                if token.startswith('cuda'):
                    try:
                        device = torch.device(token)
                    except RuntimeError as exc:  # pragma: no cover - guard against invalid spec
                        raise ValueError(f"无法解析 GPU 设备标识 '{token}': {exc}") from exc
                    devices.append(device)
                elif token in {'cpu', 'cpu-only'}:
                    devices.append(torch.device('cpu'))
                elif token.isdigit():
                    devices.append(torch.device(f"cuda:{token}"))
                else:
                    raise ValueError(f"无法解析设备标识 '{token}'。支持 'cpu'、'auto' 或逗号分隔的 cuda:id 列表。")
            if not devices:
                return [torch.device("cpu")]
            gpu_devices = [dev for dev in devices if dev.type == 'cuda']
            if gpu_devices and not torch.cuda.is_available():
                raise ValueError("当前环境未检测到可用 GPU，但指定了 GPU 设备。")
            if gpu_devices:
                max_idx = torch.cuda.device_count() - 1
                for dev in gpu_devices:
                    if dev.index is None or dev.index > max_idx:
                        raise ValueError(f"请求的 GPU {dev} 超出可用范围 [0, {max_idx}]。")
            return devices

        parsed_devices = _parse_spec(device_spec)
        if not parsed_devices:
            parsed_devices = [torch.device("cpu")]

        # 过滤重复项同时保留顺序
        seen: set[str] = set()
        unique_devices: list[torch.device] = []
        for dev in parsed_devices:
            key = str(dev)
            if key not in seen:
                seen.add(key)
                unique_devices.append(dev)

        self.compute_devices = unique_devices
        self.primary_device = self.compute_devices[0]

        device_desc = ', '.join(str(dev) for dev in self.compute_devices)
        print(f"PyTorch 可用设备: {device_desc}")
        if len(self.compute_devices) > 1 and not torch.cuda.is_available():
            print("警告: 多 GPU 已指定但当前 PyTorch 未启用 CUDA。将退回到 CPU。")

    def _ensure_device_buffers(self, device: torch.device) -> dict[str, torch.Tensor]:
        if device not in self._device_buffers:
            mins = self.base_mins.to(device)
            scales = self.base_scales.to(device)
            maxs = self.base_maxs.to(device)
            discrete_mask = self.base_discrete_mask.to(device)
            self._device_buffers[device] = {
                'mins': mins,
                'scales': scales,
                'maxs': maxs,
                'discrete_mask': discrete_mask,
            }
        return self._device_buffers[device]

    def _split_sizes(self, total: int) -> list[int]:
        num_devices = len(self.compute_devices)
        if num_devices == 0:
            return [total]
        base = total // num_devices
        remainder = total % num_devices
        return [base + (1 if idx < remainder else 0) for idx in range(num_devices)]

    def _apply_per_device(self, primary_tensor: torch.Tensor, worker, paired_tensors: Optional[list[torch.Tensor]] = None
                          ) -> torch.Tensor:
        if paired_tensors is None:
            paired_tensors = []

        if len(self.compute_devices) == 1:
            device = self.compute_devices[0]
            chunk = primary_tensor.to(device)
            paired_chunks = [tensor.to(device) for tensor in paired_tensors]
            result = worker(chunk, device, *paired_chunks)
            return result.to(self.primary_device, non_blocking=True)

        sizes = self._split_sizes(primary_tensor.shape[0])
        offset = 0
        chunk_specs: list[tuple[int, torch.device, int]] = []
        for device, size in zip(self.compute_devices, sizes):
            start = offset
            if size > 0:
                chunk_specs.append((start, device, size))
            offset += size

        if not chunk_specs:
            return torch.empty(0, device=self.primary_device, dtype=primary_tensor.dtype)

        results: list[tuple[int, torch.Tensor]] = []
        results_lock = threading.Lock()
        exceptions: list[BaseException] = []

        def run_worker(start: int, device: torch.device, size: int) -> None:
            nonlocal results, exceptions
            try:
                chunk = primary_tensor.narrow(0, start, size).to(device, non_blocking=True)
                paired_chunks = [
                    tensor.narrow(0, start, size).to(device, non_blocking=True)
                    for tensor in paired_tensors
                ]
                result = worker(chunk, device, *paired_chunks)
                result_primary = result.to(self.primary_device, non_blocking=True)
                with results_lock:
                    results.append((start, result_primary))
            except BaseException as exc:  # pragma: no cover - defensive guard for worker execution
                with results_lock:
                    exceptions.append(exc)

        threads: list[threading.Thread] = []
        for start, device, size in chunk_specs:
            thread = threading.Thread(target=run_worker, args=(start, device, size))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if exceptions:
            raise exceptions[0]

        results.sort(key=lambda item: item[0])
        ordered = [res for _, res in results]
        if not ordered:
            return torch.empty(0, device=self.primary_device, dtype=primary_tensor.dtype)
        return torch.cat(ordered, dim=0)

    @staticmethod
    def _get_queries_and_k_from_stat_module(stat_module: 'AdaptiveChainedStatistics') -> tuple[np.ndarray, int]:
        first_marginal_module = next((m for m in stat_module.stat_modules if isinstance(m, Marginals)), None)
        k = first_marginal_module.k if first_marginal_module else 0
        all_queries_list = [np.array(m.queries) for m in stat_module.stat_modules if hasattr(m, 'queries') and m.queries.shape[0] > 0]
        if not all_queries_list: return np.array([]), 0
        return np.concatenate(all_queries_list), k

    def _decode_pt(self, data_norm_pt: torch.Tensor) -> torch.Tensor:
        """将标准化的 [-1, 1] 数据解码回原始范围"""
        device = data_norm_pt.device
        buffers = self._ensure_device_buffers(device)
        mins = buffers['mins']
        scales = buffers['scales']
        maxs = buffers['maxs']
        discrete_mask = buffers['discrete_mask']

        data_01 = (data_norm_pt + 1.0) / 2.0
        data_original = data_01 * scales + mins
        if discrete_mask.any():
            data_original[..., discrete_mask] = torch.round(data_original[..., discrete_mask])
        return torch.clamp(data_original, mins, maxs)

    def _encode_pt(self, data_original_pt: torch.Tensor) -> torch.Tensor:
        """将原始数据重新编码回 [-1, 1] 范围"""
        device = data_original_pt.device
        buffers = self._ensure_device_buffers(device)
        mins = buffers['mins']
        scales = buffers['scales']
        maxs = buffers['maxs']

        data_clamped = torch.clamp(data_original_pt, mins, maxs)
        data_01 = (data_clamped - mins) / scales
        data_norm = data_01 * 2.0 - 1.0
        return torch.clamp(data_norm, -1.0, 1.0)

    def _calculate_answers_single_device(self, data_pt: torch.Tensor, q_I_pt: torch.Tensor, q_U_pt: torch.Tensor,
                                         q_L_pt: torch.Tensor, batch_size_records: int, query_chunk_size: int,
                                         normalize_by: Optional[int] = None) -> torch.Tensor:
        batch_P, num_records, _ = data_pt.shape
        num_queries = q_I_pt.shape[0]
        device = data_pt.device
        total_counts = torch.zeros(batch_P, num_queries, dtype=torch.float32, device=device)

        query_chunk_size = max(1, int(query_chunk_size))

        for record_start in range(0, num_records, batch_size_records):
            record_end = min(record_start + batch_size_records, num_records)
            data_batch = data_pt[:, record_start:record_end, :]
            rec_batch = record_end - record_start

            for query_start in range(0, num_queries, query_chunk_size):
                query_end = min(query_start + query_chunk_size, num_queries)
                q_block = q_I_pt[query_start:query_end]
                flat_ids = q_block.reshape(-1).long()
                if flat_ids.numel() == 0:
                    total_counts[:, query_start:query_end] += float(rec_batch)
                    continue
                gathered = torch.index_select(data_batch, 2, flat_ids)
                gathered = gathered.view(batch_P, rec_batch, query_end - query_start, -1)

                lowers = q_L_pt[query_start:query_end].view(1, 1, query_end - query_start, -1)
                uppers = q_U_pt[query_start:query_end].view(1, 1, query_end - query_start, -1)

                range_cond = (gathered >= lowers) & (gathered < uppers)
                phi_results = torch.all(range_cond, dim=3)
                total_counts[:, query_start:query_end] += torch.sum(phi_results, dim=1)

        if normalize_by is not None and normalize_by > 0:
            total_counts = total_counts / float(normalize_by)
        return total_counts

    def _calculate_answers_batched(self, data_pt: torch.Tensor, query_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                                   batch_size_records: int, query_chunk_size: int,
                                   normalize_by: Optional[int] = None) -> torch.Tensor:
        def worker(chunk: torch.Tensor, device: torch.device) -> torch.Tensor:
            q_I_pt, q_U_pt, q_L_pt = query_cache[device]
            return self._calculate_answers_single_device(
                chunk, q_I_pt, q_U_pt, q_L_pt, batch_size_records, query_chunk_size, normalize_by)

        return self._apply_per_device(data_pt, worker)

    def _evaluate_query_mask_rows(self, rows: torch.Tensor, col_ids: torch.Tensor,
                                  lowers: torch.Tensor, uppers: torch.Tensor) -> torch.Tensor:
        """给定若干记录行，判断它们是否满足某个查询的区间约束"""
        device = rows.device
        if rows.numel() == 0:
            return torch.zeros(rows.shape[0], dtype=torch.bool, device=device)

        valid_mask = col_ids >= 0
        if not torch.any(valid_mask):
            return torch.ones(rows.shape[0], dtype=torch.bool, device=device)

        valid_cols = col_ids[valid_mask].long()
        selected = rows.index_select(1, valid_cols)
        lower_bounds = lowers[valid_mask].unsqueeze(0)
        upper_bounds = uppers[valid_mask].unsqueeze(0)
        cond = (selected >= lower_bounds) & (selected < upper_bounds)
        return torch.all(cond, dim=1)

    def _calculate_linf_fitness_single_device(self, data_pt: torch.Tensor, error_pt: torch.Tensor, q_I_pt: torch.Tensor,
                                              q_L_pt: torch.Tensor, q_U_pt: torch.Tensor, batch_size_records: int,
                                              query_chunk_size: int) -> torch.Tensor:
        """计算每条记录的 L∞ 适应度 (即行参与的最大查询误差)"""
        batch_P, num_records, _ = data_pt.shape
        num_queries = q_I_pt.shape[0]
        device = data_pt.device
        linf_values = torch.zeros(batch_P, num_records, dtype=torch.float32, device=device)
        abs_error = torch.abs(error_pt)
        query_chunk_size = max(1, int(query_chunk_size))

        for record_start in range(0, num_records, batch_size_records):
            record_end = min(record_start + batch_size_records, num_records)
            data_batch = data_pt[:, record_start:record_end, :]
            rec_batch = record_end - record_start
            batch_linf = torch.zeros(batch_P, rec_batch, dtype=torch.float32, device=device)

            for query_start in range(0, num_queries, query_chunk_size):
                query_end = min(query_start + query_chunk_size, num_queries)
                q_block = q_I_pt[query_start:query_end]
                flat_ids = q_block.reshape(-1).long()
                if flat_ids.numel() == 0:
                    max_err = abs_error[:, query_start:query_end].amax(dim=1, keepdim=True)
                    batch_linf = torch.maximum(batch_linf, max_err.expand(-1, rec_batch))
                    continue
                gathered = torch.index_select(data_batch, 2, flat_ids)
                gathered = gathered.view(batch_P, rec_batch, query_end - query_start, -1)

                lowers = q_L_pt[query_start:query_end].view(1, 1, query_end - query_start, -1)
                uppers = q_U_pt[query_start:query_end].view(1, 1, query_end - query_start, -1)

                phi_results = torch.all((gathered >= lowers) & (gathered < uppers), dim=3)
                err_block = abs_error[:, query_start:query_end].unsqueeze(1)
                contribution = phi_results.float() * err_block
                batch_linf = torch.maximum(batch_linf, contribution.max(dim=2).values)

            linf_values[:, record_start:record_end] = batch_linf

        return linf_values

    def _calculate_linf_fitness(self, data_pt: torch.Tensor, error_pt: torch.Tensor,
                                query_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                                batch_size_records: int, query_chunk_size: int) -> torch.Tensor:
        def worker(chunk: torch.Tensor, device: torch.device, error_chunk: torch.Tensor) -> torch.Tensor:
            q_I_pt, q_U_pt, q_L_pt = query_cache[device]
            return self._calculate_linf_fitness_single_device(
                chunk, error_chunk, q_I_pt, q_L_pt, q_U_pt, batch_size_records, query_chunk_size)

        return self._apply_per_device(data_pt, worker, paired_tensors=[error_pt])

    def _sample_values_inside(self, col_idx: int, low: float, up: float, num_samples: int,
                              device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.primary_device
        min_val = float(self.base_mins[col_idx].item())
        max_val = float(self.base_maxs[col_idx].item())
        if self.col_types_list[col_idx] == 0:
            low_clamped = max(low, min_val)
            up_clamped = min(up, max_val)
            if up_clamped <= low_clamped + 1e-8:
                return torch.full((num_samples,), low_clamped, device=device)
            return torch.empty(num_samples, device=device).uniform_(low_clamped, up_clamped)
        domain_min = int(round(min_val))
        domain_max = int(round(max_val))
        low_int = max(domain_min, int(np.ceil(low)))
        high_int = min(domain_max, int(np.floor(up - 1e-6)))
        if high_int < low_int:
            low_int, high_int = domain_min, domain_max
        possible = torch.arange(low_int, high_int + 1, device=device, dtype=torch.float32)
        if possible.numel() == 1:
            return possible.repeat(num_samples)
        idx = torch.randint(0, possible.numel(), (num_samples,), device=device)
        return possible[idx]

    def _sample_values_outside(self, col_idx: int, low: float, up: float, num_samples: int,
                               device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.primary_device
        min_val = float(self.base_mins[col_idx].item())
        max_val = float(self.base_maxs[col_idx].item())
        if self.col_types_list[col_idx] == 0:
            low_clamped = max(low, min_val)
            up_clamped = min(up, max_val)
            ranges = []
            if low_clamped - min_val > 1e-6:
                ranges.append((min_val, low_clamped))
            if max_val - up_clamped > 1e-6:
                ranges.append((up_clamped, max_val))
            if not ranges:
                return torch.empty(num_samples, device=device).uniform_(min_val, max_val)
            if len(ranges) == 1:
                start, end = ranges[0]
                if end <= start + 1e-8:
                    return torch.full((num_samples,), start, device=device)
                return torch.empty(num_samples, device=device).uniform_(start, end)
            choice = torch.randint(0, len(ranges), (num_samples,), device=device)
            samples = torch.empty(num_samples, device=device)
            for idx, (start, end) in enumerate(ranges):
                mask = choice == idx
                if mask.any():
                    if end <= start + 1e-8:
                        samples[mask] = start
                    else:
                        samples[mask] = torch.empty(int(mask.sum().item()), device=device).uniform_(start, end)
            return samples
        domain_min = int(round(min_val))
        domain_max = int(round(max_val))
        low_int = int(np.ceil(low))
        high_int = int(np.floor(up - 1e-6))
        all_values = torch.arange(domain_min, domain_max + 1, device=device, dtype=torch.float32)
        mask = (all_values < low_int) | (all_values > high_int)
        possible = all_values[mask]
        if possible.numel() == 0:
            possible = all_values
        idx = torch.randint(0, possible.numel(), (num_samples,), device=device)
        return possible[idx]

    def _sample_value_inside(self, col_idx: int, low: float, up: float,
                             device: Optional[torch.device] = None) -> torch.Tensor:
        return self._sample_values_inside(col_idx, low, up, 1, device=device).squeeze(0)

    def _sample_value_outside(self, col_idx: int, low: float, up: float,
                              device: Optional[torch.device] = None) -> torch.Tensor:
        return self._sample_values_outside(col_idx, low, up, 1, device=device).squeeze(0)

    def _sample_values_inside_aggressive(self, col_idx: int, low: float, up: float, num_samples: int,
                                         device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.primary_device
        min_val = float(self.base_mins[col_idx].item())
        max_val = float(self.base_maxs[col_idx].item())
        if self.col_types_list[col_idx] == 0:
            low_clamped = max(low, min_val)
            up_clamped = min(up, max_val)
            if up_clamped <= low_clamped + 1e-8:
                return torch.full((num_samples,), low_clamped, device=device)
            midpoint = 0.5 * (low_clamped + up_clamped)
            width = max(1e-6, (up_clamped - low_clamped) * 0.1)
            samples = torch.empty(num_samples, device=device).normal_(mean=midpoint, std=width)
            return torch.clamp(samples, low_clamped, up_clamped)
        domain_min = int(round(min_val))
        domain_max = int(round(max_val))
        low_int = max(domain_min, int(np.ceil(low)))
        high_int = min(domain_max, int(np.floor(up - 1e-6)))
        if high_int < low_int:
            low_int, high_int = domain_min, domain_max
        mid_val = (low_int + high_int) // 2
        window_radius = max(0, (high_int - low_int) // 10)
        start = max(low_int, mid_val - window_radius)
        end = min(high_int, mid_val + window_radius)
        possible = torch.arange(start, end + 1, device=device, dtype=torch.float32)
        if possible.numel() == 0:
            possible = torch.tensor([float(mid_val)], device=device)
        idx = torch.randint(0, possible.numel(), (num_samples,), device=device)
        return possible[idx]

    def _sample_values_outside_aggressive(self, col_idx: int, low: float, up: float, num_samples: int,
                                          device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.primary_device
        min_val = float(self.base_mins[col_idx].item())
        max_val = float(self.base_maxs[col_idx].item())
        if self.col_types_list[col_idx] == 0:
            low_clamped = max(low, min_val)
            up_clamped = min(up, max_val)
            below_span = max(0.0, low_clamped - min_val)
            above_span = max(0.0, max_val - up_clamped)
            if below_span <= 1e-8 and above_span <= 1e-8:
                return self._sample_values_outside(col_idx, low, up, num_samples, device=device)
            if below_span >= above_span:
                return torch.full((num_samples,), min_val, device=device)
            return torch.full((num_samples,), max_val, device=device)
        domain_min = int(round(min_val))
        domain_max = int(round(max_val))
        low_int = int(np.ceil(low))
        high_int = int(np.floor(up - 1e-6))
        below_count = max(0, low_int - domain_min)
        above_count = max(0, domain_max - high_int)
        if below_count == 0 and above_count == 0:
            return self._sample_values_outside(col_idx, low, up, num_samples, device=device)
        if below_count >= above_count:
            start = domain_min
            end = max(domain_min, low_int - 1)
            possible = torch.arange(start, end + 1, device=device, dtype=torch.float32)
        else:
            start = min(domain_max, high_int + 1)
            end = domain_max
            possible = torch.arange(start, end + 1, device=device, dtype=torch.float32)
        if possible.numel() == 0:
            possible = torch.tensor([float(domain_min), float(domain_max)], device=device)
        idx = torch.randint(0, possible.numel(), (num_samples,), device=device)
        return possible[idx]

    def _mutate_rows_to_interval_batch(self, rows: torch.Tensor, col_ids: torch.Tensor, lowers: torch.Tensor,
                                       uppers: torch.Tensor, make_inside: bool, aggressive: bool = False) -> None:
        if rows.numel() == 0:
            return
        num_rows = rows.shape[0]
        device = rows.device
        for pos in range(col_ids.shape[0]):
            col_idx = int(col_ids[pos].item())
            if col_idx < 0 or col_idx >= self.num_columns:
                continue
            low = float(lowers[pos].item())
            up = float(uppers[pos].item())
            if make_inside:
                if aggressive:
                    new_values = self._sample_values_inside_aggressive(col_idx, low, up, num_rows, device=device)
                else:
                    new_values = self._sample_values_inside(col_idx, low, up, num_rows, device=device)
            else:
                if aggressive:
                    new_values = self._sample_values_outside_aggressive(col_idx, low, up, num_rows, device=device)
                else:
                    new_values = self._sample_values_outside(col_idx, low, up, num_rows, device=device)
            rows[:, col_idx] = new_values

    def _apply_targeted_mutations(self, decoded_population: torch.Tensor, global_error: torch.Tensor,
                                  q_I_pt: torch.Tensor, q_L_pt: torch.Tensor,
                                  q_U_pt: torch.Tensor, replacement_batch_size: int, num_records: int,
                                  aggressive_queries_mask: torch.Tensor, aggressive_multiplier: float) -> torch.Tensor:
        batch_P, _, _ = decoded_population.shape
        num_queries = global_error.shape[1]
        for pop_idx in range(batch_P):
            errors = global_error[pop_idx]
            _, sorted_queries = torch.sort(torch.abs(errors), descending=True)
            rows_changed = 0
            for q_rank in range(num_queries):
                q_idx = int(sorted_queries[q_rank].item())
                err_val = float(errors[q_idx].item())
                if abs(err_val) < 1e-8:
                    continue
                col_ids = q_I_pt[q_idx]
                lowers = q_L_pt[q_idx]
                uppers = q_U_pt[q_idx]
                satisfied_mask = self._evaluate_query_mask_rows(decoded_population[pop_idx], col_ids, lowers, uppers)
                candidate_mask = (~satisfied_mask) if err_val > 0 else satisfied_mask
                candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
                if candidate_indices.numel() == 0:
                    continue
                desired = max(1, int(abs(err_val) * num_records))
                if aggressive_queries_mask[q_idx].item():
                    desired = max(desired, int(desired * aggressive_multiplier))
                remaining = replacement_batch_size - rows_changed
                if remaining <= 0:
                    break
                num_updates = min(desired, remaining, candidate_indices.numel())
                perm = torch.randperm(candidate_indices.numel(), device=candidate_indices.device)[:num_updates]
                chosen_indices = candidate_indices[perm]
                rows_to_mutate = decoded_population[pop_idx].index_select(0, chosen_indices)
                self._mutate_rows_to_interval_batch(rows_to_mutate, col_ids, lowers, uppers,
                                                    make_inside=(err_val > 0),
                                                    aggressive=bool(aggressive_queries_mask[q_idx].item()))
                decoded_population[pop_idx].index_copy_(0, chosen_indices, rows_to_mutate)
                rows_changed += num_updates
                if rows_changed >= replacement_batch_size:
                    break
        return decoded_population

    def _inject_diversity(self, population_norm: torch.Tensor, best_population_norm: torch.Tensor,
                          worst_indices: torch.Tensor, jitter_scale: float) -> torch.Tensor:
        if worst_indices.numel() == 0:
            return population_norm
        jitter = torch.empty_like(population_norm[worst_indices]).uniform_(-jitter_scale, jitter_scale)
        blended_best = best_population_norm.unsqueeze(0).expand_as(population_norm[worst_indices])
        mixed = 0.5 * population_norm[worst_indices] + 0.5 * blended_best + jitter
        population_norm[worst_indices] = torch.clamp(mixed, -1.0, 1.0)
        return population_norm

    def _run_evolution(self, key, initial_population, noised_answers, queries, k, num_records, G, P,
                       fitness_batch_size, query_chunk_size, replacement_batch_size, crossover_rate, crossover_num_rows,
                       stagnation_patience: int, diversity_fraction: float, diversity_jitter: float,
                       mutation_growth: float, min_effective_replacement: int, improvement_tolerance: float,
                       aggressive_query_patience: int, aggressive_query_multiplier: float,
                       query_improvement_tolerance: float, aggressive_replacement_boost: float):
        """包含L∞适应度重采样和精英选择的主循环"""

        q_I_cpu = torch.tensor(queries[:, :k], dtype=torch.long)
        q_U_cpu = torch.tensor(queries[:, k:2*k], dtype=torch.float32)
        q_L_cpu = torch.tensor(queries[:, 2*k:3*k], dtype=torch.float32)
        query_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for dev in self.compute_devices:
            query_cache[dev] = (q_I_cpu.to(dev), q_U_cpu.to(dev), q_L_cpu.to(dev))
        q_I_primary = query_cache[self.primary_device][0]
        q_U_primary = query_cache[self.primary_device][1]
        q_L_primary = query_cache[self.primary_device][2]

        noised_answers_pt = torch.tensor(noised_answers, dtype=torch.float32, device=self.primary_device).unsqueeze(0)

        population_norm = initial_population.to(self.primary_device)

        decoded_pop = self._decode_pt(population_norm)
        current_answers = self._calculate_answers_batched(decoded_pop, query_cache,
                                                          fitness_batch_size, query_chunk_size,
                                                          normalize_by=num_records)
        global_error = noised_answers_pt - current_answers
        global_errors_per_pop = torch.linalg.norm(global_error, dim=1)
        prev_query_abs_error = torch.abs(global_error).mean(dim=0)
        query_stagnation = torch.zeros(prev_query_abs_error.shape[0], dtype=torch.int32, device=self.primary_device)
        aggressive_queries_mask = torch.zeros_like(prev_query_abs_error, dtype=torch.bool, device=self.primary_device)
        prev_aggressive_count = 0

        best_idx = torch.argmin(global_errors_per_pop)
        best_overall_score = global_errors_per_pop[best_idx].item()
        best_overall_population = population_norm[best_idx].clone()

        print(f"初始最佳误差 (L2): {best_overall_score:.4f} | 原始尺度 {best_overall_score * num_records:.2f}")
        print("说明: 误差是使用带噪 (差分隐私) 查询答案作为目标计算的。")

        stagnation_counter = 0

        for g in tqdm(range(G), desc="世代 (Generations)"):

            # 1. 解码并计算全局误差
            decoded_pop = self._decode_pt(population_norm)
            current_answers = self._calculate_answers_batched(decoded_pop, query_cache,
                                                              fitness_batch_size, query_chunk_size,
                                                              normalize_by=num_records)
            global_error = noised_answers_pt - current_answers

            # 2. 基于误差执行针对性突变
            effective_replacement = int(replacement_batch_size * (1.0 + stagnation_counter * mutation_growth))
            effective_replacement = max(min_effective_replacement, effective_replacement)
            effective_replacement = min(effective_replacement, num_records)
            if aggressive_queries_mask.any():
                effective_replacement = min(
                    num_records,
                    max(min_effective_replacement,
                        int(effective_replacement * (1.0 + aggressive_replacement_boost)))
                )
                new_count = int(aggressive_queries_mask.sum().item())
                if new_count > prev_aggressive_count:
                    tqdm.write(f"激活 {new_count} 个查询进入强化突变模式。")
                prev_aggressive_count = new_count
            else:
                prev_aggressive_count = 0
            evolved_decoded_pop = self._apply_targeted_mutations(
                decoded_pop.clone(), global_error, q_I_primary,
                q_L_primary, q_U_primary,
                effective_replacement, num_records, aggressive_queries_mask, aggressive_query_multiplier)

            population_norm = self._encode_pt(evolved_decoded_pop)

            # 3. 评估并选择精英
            updated_answers = self._calculate_answers_batched(evolved_decoded_pop, query_cache,
                                                              fitness_batch_size, query_chunk_size,
                                                              normalize_by=num_records)
            updated_global_error = noised_answers_pt - updated_answers
            linf_fitness = self._calculate_linf_fitness(
                evolved_decoded_pop, updated_global_error, query_cache,
                fitness_batch_size, query_chunk_size)
            current_global_errors = torch.linalg.norm(updated_global_error, dim=1)

            best_current_idx = torch.argmin(current_global_errors)
            best_current_score = current_global_errors[best_current_idx].item()

            if best_current_score < best_overall_score - improvement_tolerance:
                best_overall_score = best_current_score
                best_overall_population = population_norm[best_current_idx].clone()
                stagnation_counter = 0
                tqdm.write(
                    f"世代 {g+1} 结束。发现新的最佳误差 (L2): {best_overall_score:.4f} | 原始尺度 {best_overall_score * num_records:.2f}")
            else:
                stagnation_counter += 1
                tqdm.write(
                    f"世代 {g+1} 结束。当前世代最佳: {best_current_score:.4f} (原始尺度 {best_current_score * num_records:.2f}), "
                    f"维持全局最佳: {best_overall_score:.4f} (原始尺度 {best_overall_score * num_records:.2f})")

            # 4. L∞适应度驱动的交叉
            if crossover_rate > 0 and P > 1:
                decoded_best = self._decode_pt(best_overall_population.unsqueeze(0))
                best_answers = self._calculate_answers_batched(decoded_best, query_cache,
                                                               fitness_batch_size, query_chunk_size,
                                                               normalize_by=num_records)
                best_global_error = noised_answers_pt - best_answers
                best_linf_fitness = self._calculate_linf_fitness(
                    decoded_best, best_global_error, query_cache,
                    fitness_batch_size, query_chunk_size).squeeze(0)
                _, elite_indices = torch.topk(best_linf_fitness, crossover_num_rows, largest=False)

                num_to_crossover = int(P * crossover_rate)
                _, worst_pop_indices = torch.topk(current_global_errors, num_to_crossover, largest=True)

                for pop_idx in worst_pop_indices:
                    if pop_idx == best_current_idx:
                        continue
                    _, worst_record_indices = torch.topk(linf_fitness[pop_idx], crossover_num_rows, largest=True)
                    population_norm[pop_idx, worst_record_indices, :] = best_overall_population[elite_indices, :]

            if stagnation_counter >= stagnation_patience:
                num_diverse = max(1, int(P * diversity_fraction))
                _, worst_indices = torch.topk(current_global_errors, num_diverse, largest=True)
                population_norm = self._inject_diversity(population_norm, best_overall_population, worst_indices, diversity_jitter)
                stagnation_counter = 0
                tqdm.write(
                    f"触发多样性注入: 重置 {num_diverse} 个个体, 当前全局最佳 L2 {best_overall_score:.4f} | 原始尺度 {best_overall_score * num_records:.2f}")

            current_query_abs_error = torch.abs(updated_global_error).mean(dim=0)
            improvement = prev_query_abs_error - current_query_abs_error
            query_stagnation = torch.where(
                improvement > query_improvement_tolerance,
                torch.zeros_like(query_stagnation),
                query_stagnation + 1
            )
            aggressive_queries_mask = query_stagnation >= aggressive_query_patience
            prev_query_abs_error = current_query_abs_error

        return best_overall_population

    def generate(self, stat_module: 'AdaptiveChainedStatistics', noised_answers: np.ndarray, num_records: int, G: int, P: int, fitness_batch_size: int, replacement_batch_size: int, crossover_rate: float, crossover_num_rows: int, **kwargs) -> 'Dataset':
        queries_np, k = self._get_queries_and_k_from_stat_module(stat_module)
        if k == 0 or queries_np.shape[0] == 0:
            raise ValueError("未能从统计模块中提取有效的查询。")

        initial_population = torch.rand(P, num_records, self.num_columns, device=self.primary_device) * 2 - 1

        stagnation_patience = kwargs.get('stagnation_patience', 40)
        diversity_fraction = kwargs.get('diversity_fraction', 0.2)
        diversity_jitter = kwargs.get('diversity_jitter', 0.15)
        mutation_growth = kwargs.get('mutation_growth', 0.3)
        min_effective_replacement = kwargs.get('min_effective_replacement', max(1, replacement_batch_size // 2))
        improvement_tolerance = kwargs.get('improvement_tolerance', 1e-4)
        aggressive_query_patience = max(1, int(kwargs.get('aggressive_query_patience', 3)))
        aggressive_query_multiplier = float(kwargs.get('aggressive_query_multiplier', 2.0))
        query_improvement_tolerance = float(kwargs.get('query_improvement_tolerance', 1e-4))
        aggressive_replacement_boost = float(kwargs.get('aggressive_replacement_boost', 0.5))
        query_chunk_size = int(kwargs.get('query_chunk_size', 256))

        final_population_norm = self._run_evolution(
            None, initial_population, noised_answers, queries_np, k, num_records, G, P,
            fitness_batch_size, query_chunk_size, replacement_batch_size, crossover_rate, crossover_num_rows,
            stagnation_patience, diversity_fraction, diversity_jitter,
            mutation_growth, min_effective_replacement, improvement_tolerance,
            aggressive_query_patience, aggressive_query_multiplier,
            query_improvement_tolerance, aggressive_replacement_boost
        )

        final_decoded_pt = self._decode_pt(final_population_norm.unsqueeze(0)).squeeze(0)
        final_df = pd.DataFrame(final_decoded_pt.cpu().numpy(), columns=self.domain.attrs)

        return Dataset(final_df, self.domain)

def main():
    parser = argparse.ArgumentParser(description="Private-DE: V50 - L∞ 定向重采样版 (PyTorch)。")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epsilon', type=float, default=10.0)
    parser.add_argument('-P', '--population_size', type=int, default=100)
    parser.add_argument('-G', type=int, default=50000, help="演化世代数。")
    parser.add_argument('--fitness_batch_size', type=int, default=1024, help="计算适应度时的记录批处理大小。")
    parser.add_argument('--replacement_batch_size', type=int, default=256, help="每代中被替换的记录数量。")
    parser.add_argument('--crossover_rate', type=float, default=0.5)
    parser.add_argument('--crossover_num_rows', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stagnation_patience', type=int, default=5, help="在没有取得新的全局最优前允许的停滞代数。")
    parser.add_argument('--diversity_fraction', type=float, default=0.2, help="触发多样性注入时需要重采样的人口比例。")
    parser.add_argument('--diversity_jitter', type=float, default=1, help="多样性注入时在 [-1,1] 空间施加的随机扰动幅度。")
    parser.add_argument('--mutation_growth', type=float, default=0.3, help="停滞时放大 replacement_batch_size 的增长系数。")
    parser.add_argument('--min_effective_replacement', type=int, default=128, help="针对性突变时的最小替换行数下限。")
    parser.add_argument('--improvement_tolerance', type=float, default=1e-4, help="判定出现有效改进所需的 L2 阈值。")
    parser.add_argument('--aggressive_query_patience', type=int, default=3, help="单个查询在连续多少代没有明显改善后进入强化突变模式。")
    parser.add_argument('--aggressive_query_multiplier', type=float, default=2.0, help="强化突变模式下单个查询的替换行数放大倍率。")
    parser.add_argument('--query_improvement_tolerance', type=float, default=1e-4, help="判定单个查询误差得到改善所需的最小幅度。")
    parser.add_argument('--aggressive_replacement_boost', type=float, default=0.5, help="存在强化突变查询时额外放大的替换行数比例。")
    parser.add_argument('--query_chunk_size', type=int, default=32000, help="按查询分块进行批处理时的块大小，较小值可显著降低显存占用。")
    parser.add_argument('--device', type=str, default='auto',
                        help="要使用的计算设备。可选: 'auto' (默认, 自动使用所有 GPU), 'cpu', 或逗号分隔的 cuda:id 列表。")

    args = parser.parse_args(args=['--input', '/home/qianqiu/experiment-trade/private_gsd/generate_script/source/acs.csv',
                                   '--output', './synthetic_acs.csv',
                                   '--population_size', '1'])

    if not os.path.exists(args.input):
        print(f"错误: 输入文件 '{args.input}' 不存在。"); return

    print("--- 步骤 1: 加载数据并预处理 ---")
    raw_df = pd.read_csv(args.input, skipinitialspace=True, na_values='?')
    
    types = TypeMap.infer_column_types(raw_df)
    meta_data = {}
    print("正在为数值列和序数列推断公共边界 (min/max)...")
    for col in types['ordinal_columns'] + types['continuous_columns']:
        min_val, max_val = raw_df[col].min(), raw_df[col].max()
        meta_data[col] = {'type': 'int' if col in types['ordinal_columns'] else 'float', 'lower': min_val, 'upper': max_val}
    for col in types['categorical_columns']:
        meta_data[col] = {'type': 'string'}

    temp_synthesizer = GSDSynthesizer(epsilon=args.epsilon)
    real_dataset = temp_synthesizer._get_data(
        raw_df, 
        meta_data=meta_data,
        categorical_columns=types['categorical_columns'],
        ordinal_columns=types['ordinal_columns'],
        continuous_columns=types['continuous_columns']
    )
    num_records = len(raw_df)
    print(f"\n数据加载和预处理完成！合成数据集大小将为: {num_records}")

    print("\n--- 步骤 2: 设置并测量带噪统计查询 (DP) ---")
    stat_module = AdaptiveChainedStatistics(real_dataset)
    marginals_2way = Marginals.get_all_kway_combinations(real_dataset.domain, k=2)
    stat_module.add_stat_module_and_fit(marginals_2way)
    
    key = jax.random.PRNGKey(args.seed)
    rho = temp_synthesizer.rho 
    print(f"使用 rho={rho:.4f} 为所有查询添加噪音...")
    stat_module.private_measure_all_statistics(key, rho)
    
    noised_answers = stat_module.get_selected_noised_statistics()
    print(f"\n统计模块设置完成，已测量 {len(noised_answers)} 个带噪查询。")
    
    print("\n--- 步骤 3: 启动 PyTorch 生成器 ---")
    generator = PrivateDEGeneratorPT(real_dataset.domain, temp_synthesizer, device_spec=args.device)
    
    gen_params = {
        'G': args.G, 'P': args.population_size,
        'fitness_batch_size': args.fitness_batch_size,
        'replacement_batch_size': args.replacement_batch_size,
        'crossover_rate': args.crossover_rate,
        'crossover_num_rows': args.crossover_num_rows,
        'stagnation_patience': args.stagnation_patience,
        'diversity_fraction': args.diversity_fraction,
        'diversity_jitter': args.diversity_jitter,
        'mutation_growth': args.mutation_growth,
        'min_effective_replacement': args.min_effective_replacement,
        'improvement_tolerance': args.improvement_tolerance,
        'aggressive_query_patience': args.aggressive_query_patience,
        'aggressive_query_multiplier': args.aggressive_query_multiplier,
        'query_improvement_tolerance': args.query_improvement_tolerance,
        'aggressive_replacement_boost': args.aggressive_replacement_boost,
        'query_chunk_size': args.query_chunk_size
    }

    final_synthetic_dataset = generator.generate(stat_module, np.array(noised_answers), num_records, **gen_params)

    print(f"\n--- 步骤 4: 后处理并保存合成数据 ---")
    data_list = temp_synthesizer.get_values_as_list(final_synthetic_dataset.domain, final_synthetic_dataset.df)
    output_df = temp_synthesizer._transformer.inverse_transform(data_list)
    output_df.to_csv(args.output, index=False)
    print("保存完成。")

if __name__ == '__main__':
    main()

