import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import threading
from typing import Optional

from genetic_sd.utils.dataset_jax import Dataset
from genetic_sd.utils.domain import Domain
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
            elif self.domain.is_ordinal(col): # 显式处理有序变量
                col_types.append(1) # 1 for ordinal
                mins.append(t.fit_lower)
                maxs.append(t.fit_upper)
            else: # 剩下的是分类变量
                col_types.append(1) # 1 for categorical
                mins.append(0)  # 分类变量编码后下限通常是 0
                # 从 transformer 的 cardinality 属性获取类别数量
                maxs.append(t.cardinality[0] - 1)

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

        self.nudge_prob = 0.5           # 以 50% 概率对选中的行采用“最小改动”而非完全重采样
        self.nudge_margin = 0.05        # inside 时靠近边界的安全边距占区间宽度的比例
        self.outside_nudge_prob = 0.6    # outside 也进行最小越界的概率
        self._epsilon_small = 1e-6      # 半开区间的数值安全裕度

        # 查询加权调度的强度（EMA → 权重），可调大些在平台期更激进
        self.query_weight_beta = 0.6
        self.raking_topk = 256           # 每次 raking 覆盖的高残差查询数
        self.raking_iters = 2            # 每次触发 raking 的迭代轮数
        self.raking_step = 0.5 

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
        """
        计算每条记录的“符号占优和”行级适应度（签名不变）：
        - 对每条记录，只累计其命中的查询：
            pos_sum = Σ max(err_q, 0)
            neg_sum = Σ min(err_q, 0)  (<= 0)
        - 若 pos_sum >= |-neg_sum|，返回 -pos_sum   （负号，便于 topk(largest=False) 选出“正向最强”的行）
        否则返回  |-neg_sum|        （正号，便于 topk(largest=True) 选出“负向最强”的行）
        这样不改调用代码也能实现：用“最大正向”去指导“最大负向”。
        """
        batch_P, num_records, _ = data_pt.shape
        num_queries = q_I_pt.shape[0]
        device = data_pt.device

        # 分块累计（避免显存爆）
        query_chunk_size = max(1, int(query_chunk_size))
        fitness_values = torch.zeros(batch_P, num_records, dtype=torch.float32, device=device)

        for record_start in range(0, num_records, batch_size_records):
            record_end = min(record_start + batch_size_records, num_records)
            data_batch = data_pt[:, record_start:record_end, :]         # (P, Rb, C)
            rec_batch = record_end - record_start

            # 对当前记录批的累计器
            pos_accum = torch.zeros(batch_P, rec_batch, dtype=torch.float32, device=device)
            neg_accum = torch.zeros(batch_P, rec_batch, dtype=torch.float32, device=device)

            for query_start in range(0, num_queries, query_chunk_size):
                query_end = min(query_start + query_chunk_size, num_queries)

                # 当前查询块的列索引与边界
                q_block = q_I_pt[query_start:query_end]                  # (Qb, k)
                flat_ids = q_block.reshape(-1).long()

                # 当前查询块的误差（每个个体各一份）
                err_block = error_pt[:, query_start:query_end]           # (P, Qb)

                if flat_ids.numel() == 0:
                    # 空查询（对所有行都命中），把该块的正/负误差和加给所有行
                    pos_block = torch.clamp_min(err_block, 0.0).sum(dim=1, keepdim=True)  # (P,1)
                    neg_block = torch.clamp_max(err_block, 0.0).sum(dim=1, keepdim=True)  # (P,1)
                    pos_accum += pos_block.expand(-1, rec_batch)
                    neg_accum += neg_block.expand(-1, rec_batch)
                    continue

                # 选出该块涉及的所有列，重排成 (P, Rb, Qb, k)
                gathered = torch.index_select(data_batch, 2, flat_ids)   # (P, Rb, Qb*k)
                gathered = gathered.view(batch_P, rec_batch, query_end - query_start, -1)

                lowers = q_L_pt[query_start:query_end].view(1, 1, query_end - query_start, -1)  # (1,1,Qb,k)
                uppers = q_U_pt[query_start:query_end].view(1, 1, query_end - query_start, -1)  # (1,1,Qb,k)

                # φ：记录是否命中每个查询（AND over k）
                phi_results = torch.all((gathered >= lowers) & (gathered < uppers), dim=3)      # (P, Rb, Qb)

                # 把正/负误差只加到命中的查询上
                pos_block = torch.clamp_min(err_block.unsqueeze(1), 0.0)                       # (P,1,Qb)
                neg_block = torch.clamp_max(err_block.unsqueeze(1), 0.0)                       # (P,1,Qb)

                pos_accum += (phi_results.float() * pos_block).sum(dim=2)                      # (P,Rb)
                neg_accum += (phi_results.float() * neg_block).sum(dim=2)                      # (P,Rb)

            # 主导侧选择与符号编码
            neg_mag = -neg_accum                                # (P,Rb)  >=0
            dominant_is_pos = pos_accum >= neg_mag              # (P,Rb)  True→正向占优
            dominant_mag = torch.where(dominant_is_pos, pos_accum, neg_mag)
            # 返回值编码：正向占优→ -pos_sum（≤0）， 负向占优→ +|neg_sum|（≥0）
            fitness_values[:, record_start:record_end] = torch.where(
                dominant_is_pos, -dominant_mag, dominant_mag
            )

        return fitness_values

    def _calculate_linf_fitness(self, data_pt: torch.Tensor, error_pt: torch.Tensor,
                                query_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                                batch_size_records: int, query_chunk_size: int) -> torch.Tensor:
        def worker(chunk: torch.Tensor, device: torch.device, error_chunk: torch.Tensor) -> torch.Tensor:
            q_I_pt, q_U_pt, q_L_pt = query_cache[device]
            # 注意：单设备函数的签名仍是 (I, L, U) 顺序
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

    def _inside_distance_per_row(self, rows: torch.Tensor, col_ids: torch.Tensor,
                                lowers: torch.Tensor, uppers: torch.Tensor) -> torch.Tensor:
        """每行到“满足查询”的最小总改变量（对违规列的距离求和）。行数 R → 返回 (R,)"""
        if rows.numel() == 0:
            return torch.zeros(0, device=rows.device, dtype=torch.float32)
        device = rows.device
        valid = col_ids >= 0
        if not torch.any(valid):
            return torch.zeros(rows.shape[0], device=device, dtype=torch.float32)
        cols = col_ids[valid].long()
        sel = rows.index_select(1, cols)                       # (R, k)
        low = lowers[valid].unsqueeze(0).expand_as(sel)        # (R, k)
        up  = uppers[valid].unsqueeze(0).expand_as(sel)        # (R, k)
        below = torch.clamp(low - sel, min=0.0)                # sel < low
        above = torch.clamp(sel - (up - self._epsilon_small), min=0.0)  # sel >= up
        dist = below + above                                   # 对每列需要的最小位移
        return dist.sum(dim=1)                                 # (R,)

    def _exit_distance_per_row(self, rows: torch.Tensor, col_ids: torch.Tensor,
                            lowers: torch.Tensor, uppers: torch.Tensor) -> torch.Tensor:
        """每行到“跳出查询”的最小改变量（任选一列越界的最小距离）。R→(R,)"""
        if rows.numel() == 0:
            return torch.zeros(0, device=rows.device, dtype=torch.float32)
        device = rows.device
        valid = col_ids >= 0
        if not torch.any(valid):
            return torch.zeros(rows.shape[0], device=device, dtype=torch.float32)
        cols = col_ids[valid].long()
        sel = rows.index_select(1, cols)                       # (R, k)
        low = lowers[valid].unsqueeze(0).expand_as(sel)        # (R, k)
        up  = uppers[valid].unsqueeze(0).expand_as(sel)        # (R, k)
        # 假设当前在区间内，离两侧边界的距离（再加一点点 ε 就能越界）
        to_low = torch.clamp(sel - low, min=0.0) + self._epsilon_small
        to_up  = torch.clamp(up - sel,  min=0.0) + self._epsilon_small
        per_col = torch.minimum(to_low, to_up)                 # 每列最小改变量
        return per_col.min(dim=1).values                       # 任选一列越界 → 取列最小

    def _nudge_rows_to_interval_batch(self, rows: torch.Tensor, col_ids: torch.Tensor,
                                    lowers: torch.Tensor, uppers: torch.Tensor,
                                    make_inside: bool) -> None:
        """
        “最小改动”的微调：inside 时仅改违规列，向边界内侧推进一个 margin；
        outside（当前实现）仍然建议用采样，避免反复踩边界。
        """
        if rows.numel() == 0:
            return
        device = rows.device
        valid = col_ids >= 0
        if not torch.any(valid):
            return
        cols = col_ids[valid].long()
        low = lowers[valid]
        up  = uppers[valid]

        for j, col_idx in enumerate(cols.tolist()):
            l = float(low[j].item()); u = float(up[j].item())
            # 列域上下界
            min_v = float(self.base_mins[col_idx].item())
            max_v = float(self.base_maxs[col_idx].item())
            if make_inside:
                # 只对违规的行做微调
                cur = rows[:, col_idx]
                width = max(self._epsilon_small, u - l)
                margin = self.nudge_margin * width
                below = cur < l
                above = cur >= u
                if below.any():
                    rows[below, col_idx] = min(max(l + margin, min_v), max_v)
                if above.any():
                    rows[above, col_idx] = min(max(u - margin, min_v), max_v)
            else:
                # outside 的微调容易与 inside 冲突，这里仍建议用采样版
                pass

    def _nudge_rows_to_outside_batch(self, rows: torch.Tensor, col_ids: torch.Tensor,
                                    lowers: torch.Tensor, uppers: torch.Tensor) -> None:
        """
        对“当前命中查询”的行做最小越界：选一列离边界最近的方向轻推出区间。
        连续列：向 l- 或 u+ 推一个 margin；离散列：取 l-1 或 ceil(u) 的最近可用整数。
        """
        if rows.numel() == 0:
            return
        device = rows.device
        valid = (col_ids >= 0)
        if not torch.any(valid):
            return

        cols = col_ids[valid].long()
        lowv = lowers[valid]
        upv  = uppers[valid]

        for j, col_idx in enumerate(cols.tolist()):
            l = float(lowv[j].item())
            u = float(upv[j].item())

            cur = rows[:, col_idx]
            inside_mask = (cur >= l) & (cur < u)
            if not inside_mask.any():
                continue

            # 连续 vs 离散
            is_discrete = bool(self.base_discrete_mask[col_idx].item())
            min_v = float(self.base_mins[col_idx].item())
            max_v = float(self.base_maxs[col_idx].item())

            if not is_discrete:
                # 连续：谁离边界近就往哪边出（+一个 margin）
                width = max(self._epsilon_small, u - l)
                margin = self.nudge_margin * width
                cur_in = cur[inside_mask]
                to_low = torch.clamp(cur_in - l, min=0.0) + self._epsilon_small
                to_up  = torch.clamp(u - cur_in,  min=0.0) + self._epsilon_small
                choose_low = (to_low <= to_up)

                newvals = cur_in.clone()
                # 向区间外侧推进
                newvals[choose_low]  = max(min_v, l - margin)
                newvals[~choose_low] = min(max_v, u + margin)
                rows[inside_mask, col_idx] = newvals
            else:
                # 离散：优先选 l-1 或 ceil(u) 中“更近”的一侧
                domain_min = int(round(min_v))
                domain_max = int(round(max_v))
                left_out  = max(domain_min, int(np.ceil(l)) - 1)
                right_out = min(domain_max, int(np.ceil(u)))

                cur_in = torch.round(cur[inside_mask])
                # 若两侧都不可行，退回到“采样 outside”
                if left_out < domain_min and right_out > domain_max:
                    fallback = self._sample_values_outside(col_idx, l, u, int(cur_in.numel()), device=device)
                    rows[inside_mask, col_idx] = fallback
                    continue

                # 选距离较近的一侧
                dl = torch.abs(cur_in - left_out)
                dr = torch.abs(cur_in - right_out)
                choose_left = (dl <= dr)

                newvals = torch.empty_like(cur_in)
                newvals[choose_left]  = left_out
                newvals[~choose_left] = right_out
                rows[inside_mask, col_idx] = newvals.float()

    def _apply_targeted_mutations(self, decoded_population: torch.Tensor, global_error: torch.Tensor,
                                q_I_pt: torch.Tensor, q_L_pt: torch.Tensor,
                                q_U_pt: torch.Tensor, replacement_batch_size: int, num_records: int,
                                aggressive_queries_mask: torch.Tensor, aggressive_multiplier: float) -> torch.Tensor:
        """
        供需转移 + 距离感知 + 微调（nudge）：
        - 正向误差 q+：优先从负向误差查询 q- 的命中行中抽“最容易被翻转”的样本（inside距离最小），
        先 inside(q+)；对 q- 的 outside 仍与之前相同（仅在非重叠列）。
        若供给不足，再从“不满足 q+ 的行”中按 inside 距离升序补齐。
        对于一部分行（nudge_prob），采用 _nudge_rows_to_interval_batch 以最小改动推进边界。
        - 负向误差 q-：从“满足 q- 的行”中按 exit 距离升序挑选，做 outside 采样。
        - 查询优先级：若存在 self._q_weights，则按 |err| * weight 排序，否则按 |err|。
        """
        batch_P, _, _ = decoded_population.shape
        num_queries = global_error.shape[1]

        for pop_idx in range(batch_P):
            errors = global_error[pop_idx]                 # (Q,)
            abs_errors = torch.abs(errors)
            # —— 查询优先级（EMA 权重，如果可用）——
            if hasattr(self, "_q_weights") and self._q_weights is not None and self._q_weights.shape[0] == num_queries:
                prio = abs_errors * self._q_weights.to(errors.device)
            else:
                prio = abs_errors
            _, sorted_queries = torch.sort(prio, descending=True)

            # 负向查询（供给端）按 |err| 降序
            neg_mask = errors < 0
            if torch.any(neg_mask):
                neg_indices = torch.nonzero(neg_mask, as_tuple=False).squeeze(1)
                neg_sorted = neg_indices[torch.argsort(torch.abs(errors[neg_indices]), descending=True)]
            else:
                neg_sorted = torch.empty(0, dtype=torch.long, device=errors.device)

            # 控制每个 q- 的“用量”
            supply_used = torch.zeros(num_queries, dtype=torch.int32, device=errors.device)

            rows_changed = 0

            for q_rank in range(num_queries):
                if rows_changed >= replacement_batch_size:
                    break

                q_idx = int(sorted_queries[q_rank].item())
                err_val = float(errors[q_idx].item())
                if abs(err_val) < 1e-8:
                    continue

                col_ids_q = q_I_pt[q_idx]; lowers_q = q_L_pt[q_idx]; uppers_q = q_U_pt[q_idx]
                is_aggr = bool(aggressive_queries_mask[q_idx].item())

                desired = max(1, int(abs(err_val) * num_records))
                if is_aggr:
                    desired = max(desired, int(desired * aggressive_multiplier))
                remaining_global = replacement_batch_size - rows_changed
                if remaining_global <= 0:
                    break
                desired = min(desired, remaining_global)

                if err_val > 0:
                    # ------------- 正向：先从负向供给里“抽最容易inside的行” -------------
                    assigned = 0
                    need_mask = ~self._evaluate_query_mask_rows(decoded_population[pop_idx], col_ids_q, lowers_q, uppers_q)

                    for neg_q in neg_sorted:
                        if assigned >= desired:
                            break
                        neg_qi = int(neg_q.item())

                        neg_err_mag = int(abs(float(errors[neg_qi].item())) * num_records)
                        neg_left = max(0, neg_err_mag - int(supply_used[neg_qi].item()))
                        if neg_left <= 0:
                            continue

                        col_ids_n = q_I_pt[neg_qi]; lowers_n = q_L_pt[neg_qi]; uppers_n = q_U_pt[neg_qi]
                        is_aggr_n = bool(aggressive_queries_mask[neg_qi].item())

                        sat_neg = self._evaluate_query_mask_rows(decoded_population[pop_idx], col_ids_n, lowers_n, uppers_n)
                        cand_mask = need_mask & sat_neg
                        candidate_indices = torch.nonzero(cand_mask, as_tuple=False).squeeze(1)
                        if candidate_indices.numel() == 0:
                            continue

                        take = min(desired - assigned, neg_left, int(candidate_indices.numel()))
                        if take <= 0:
                            continue

                        # —— 距离感知：inside距离最小的行优先 —— 
                        cand_rows = decoded_population[pop_idx].index_select(0, candidate_indices)
                        d_in = self._inside_distance_per_row(cand_rows, col_ids_q, lowers_q, uppers_q)  # (M,)
                        order = torch.argsort(d_in)[:take]
                        chosen_indices = candidate_indices[order]

                        rows_to_mutate = decoded_population[pop_idx].index_select(0, chosen_indices)

                        # inside(q+)：nudge 或 采样
                        if torch.rand(1, device=rows_to_mutate.device).item() < self.nudge_prob:
                            self._nudge_rows_to_interval_batch(rows_to_mutate, col_ids_q, lowers_q, uppers_q, make_inside=True)
                        else:
                            self._mutate_rows_to_interval_batch(rows_to_mutate, col_ids_q, lowers_q, uppers_q,
                                                                make_inside=True, aggressive=is_aggr)

                        # outside(q- 非重叠列)
                        with torch.no_grad():
                            valid_n = (col_ids_n >= 0); valid_q = (col_ids_q >= 0)
                            if valid_n.any():
                                if valid_q.any():
                                    membership = (col_ids_n.view(-1, 1) == col_ids_q.view(1, -1))
                                    overlap_any = membership.any(dim=1)
                                    neg_only_mask = valid_n & (~overlap_any)
                                else:
                                    neg_only_mask = valid_n
                            else:
                                neg_only_mask = torch.zeros_like(col_ids_n, dtype=torch.bool)
                        if neg_only_mask.any():
                            col_ids_n_masked = col_ids_n.clone()
                            col_ids_n_masked[~neg_only_mask] = -1
                            self._mutate_rows_to_interval_batch(rows_to_mutate, col_ids_n_masked, lowers_n, uppers_n,
                                                                make_inside=False, aggressive=is_aggr_n)

                        decoded_population[pop_idx].index_copy_(0, chosen_indices, rows_to_mutate)
                        assigned += int(take); rows_changed += int(take); supply_used[neg_qi] += int(take)

                    # 从“不满足 q+ 的行”补齐，仍按 inside 距离升序
                    if assigned < desired:
                        fallback_indices = torch.nonzero(need_mask, as_tuple=False).squeeze(1)
                        if fallback_indices.numel() > 0:
                            take = min(desired - assigned, int(fallback_indices.numel()))
                            cand_rows = decoded_population[pop_idx].index_select(0, fallback_indices)
                            d_in = self._inside_distance_per_row(cand_rows, col_ids_q, lowers_q, uppers_q)
                            order = torch.argsort(d_in)[:take]
                            chosen_indices = fallback_indices[order]
                            rows_to_mutate = decoded_population[pop_idx].index_select(0, chosen_indices)
                            if torch.rand(1, device=rows_to_mutate.device).item() < self.nudge_prob:
                                self._nudge_rows_to_interval_batch(rows_to_mutate, col_ids_q, lowers_q, uppers_q, make_inside=True)
                            else:
                                self._mutate_rows_to_interval_batch(rows_to_mutate, col_ids_q, lowers_q, uppers_q,
                                                                    make_inside=True, aggressive=is_aggr)
                            decoded_population[pop_idx].index_copy_(0, chosen_indices, rows_to_mutate)
                            rows_changed += int(take)

                else:
                    # ------------- 负向：从“满足 q- 的行”中选 exit 距离最小的，优先 nudge 越界 -------------
                    sat_neg = self._evaluate_query_mask_rows(decoded_population[pop_idx], col_ids_q, lowers_q, uppers_q)
                    candidate_indices = torch.nonzero(sat_neg, as_tuple=False).squeeze(1)
                    if candidate_indices.numel() == 0:
                        continue

                    take = min(desired, int(candidate_indices.numel()))
                    cand_rows = decoded_population[pop_idx].index_select(0, candidate_indices)
                    d_exit = self._exit_distance_per_row(cand_rows, col_ids_q, lowers_q, uppers_q)
                    order = torch.argsort(d_exit)[:take]  # 越容易“跳出”的行越优先
                    chosen_indices = candidate_indices[order]
                    rows_to_mutate = decoded_population[pop_idx].index_select(0, chosen_indices)

                    if torch.rand(1, device=rows_to_mutate.device).item() < self.outside_nudge_prob:
                        # 最小越界
                        self._nudge_rows_to_outside_batch(rows_to_mutate, col_ids_q, lowers_q, uppers_q)
                    else:
                        # 采样 outside（你现有的逻辑）
                        self._mutate_rows_to_interval_batch(rows_to_mutate, col_ids_q, lowers_q, uppers_q,
                                                            make_inside=False, aggressive=is_aggr)

                    decoded_population[pop_idx].index_copy_(0, chosen_indices, rows_to_mutate)
                    rows_changed += int(take)

        return decoded_population

    def _raking_resample_population(self, rows: torch.Tensor, per_pop_error: torch.Tensor,
                                    q_I_pt: torch.Tensor, q_L_pt: torch.Tensor, q_U_pt: torch.Tensor,
                                    noised_answers_pt: torch.Tensor, num_records: int,
                                    topk: int, iters: int, step: float) -> torch.Tensor:
        """
        对单个个体 rows(N,C) 做 raking：
        - 选择 |误差| 最大的 topk 个查询
        - 迭代 iters 次：对每个查询 j，计算“加权命中频率”并按比率 (target/current)^step
            乘到命中行的权重上；每次归一化权重，使均值为 1
        - 按权重多项式重采样 N 行，返回新的 rows
        """
        device = rows.device
        N = rows.shape[0]
        eps = 1e-8

        abs_err = torch.abs(per_pop_error)  # (Q,)
        k = min(topk, abs_err.numel())
        if k <= 0:
            return rows

        top_idx = torch.topk(abs_err, k, largest=True).indices

        w = torch.ones(N, device=device, dtype=torch.float32)

        for _ in range(max(1, iters)):
            for q_idx in top_idx.tolist():
                q_idx = int(q_idx)
                mask = self._evaluate_query_mask_rows(rows, q_I_pt[q_idx], q_L_pt[q_idx], q_U_pt[q_idx])  # (N,)
                if not mask.any():
                    continue
                current = (w[mask].sum() / (w.sum() + eps)).clamp(min=eps, max=1.0 - eps)
                target = float(noised_answers_pt[0, q_idx].item())  # 目标频率
                target = min(max(target, eps), 1.0 - eps)

                ratio = (target / current) ** step
                w[mask] = w[mask] * ratio

            # 归一：避免数值爆炸/塌缩
            w = w / (w.mean() + eps)

        # 按权重重采样 N 行
        probs = (w / (w.sum() + eps)).clamp(min=eps)
        idx = torch.multinomial(probs, num_samples=N, replacement=True)
        new_rows = rows.index_select(0, idx)
        return new_rows

    def _inject_diversity(self, population_norm: torch.Tensor, best_population_norm: torch.Tensor,
                          worst_indices: torch.Tensor, jitter_scale: float) -> torch.Tensor:
        if worst_indices.numel() == 0:
            return population_norm
        jitter = torch.empty_like(population_norm[worst_indices]).uniform_(-jitter_scale, jitter_scale)
        blended_best = best_population_norm.unsqueeze(0).expand_as(population_norm[worst_indices])
        mixed = 0.5 * population_norm[worst_indices] + 0.5 * blended_best + jitter
        population_norm[worst_indices] = torch.clamp(mixed, -1.0, 1.0)
        return population_norm

    def _row_weight_grad_single_device(self, data_pt: torch.Tensor, error_pt: torch.Tensor,
                                    q_I_pt: torch.Tensor, q_L_pt: torch.Tensor, q_U_pt: torch.Tensor,
                                    batch_size_records: int, query_chunk_size: int,
                                    normalize_by: int) -> torch.Tensor:
        """
        计算每条记录的梯度 g_r = -(1/N) * Σ_q φ_{r,q} * e_q
        返回形状 (P, num_records)
        """
        batch_P, num_records, _ = data_pt.shape
        num_queries = q_I_pt.shape[0]
        device = data_pt.device
        grad = torch.zeros(batch_P, num_records, dtype=torch.float32, device=device)
        scale = -1.0 / float(max(1, int(normalize_by)))

        query_chunk_size = max(1, int(query_chunk_size))

        for record_start in range(0, num_records, batch_size_records):
            record_end = min(record_start + batch_size_records, num_records)
            data_batch = data_pt[:, record_start:record_end, :]  # (P, Rb, C)
            rec_batch = record_end - record_start
            accum = torch.zeros(batch_P, rec_batch, dtype=torch.float32, device=device)

            for query_start in range(0, num_queries, query_chunk_size):
                query_end = min(query_start + query_chunk_size, num_queries)
                q_block = q_I_pt[query_start:query_end]                   # (Qb, k)
                flat_ids = q_block.reshape(-1).long()
                e_block = error_pt[:, query_start:query_end]              # (P, Qb)

                if flat_ids.numel() == 0:
                    # 空查询：对所有行贡献相同
                    accum += (e_block.sum(dim=1, keepdim=True) * scale).expand(-1, rec_batch)
                    continue

                gathered = torch.index_select(data_batch, 2, flat_ids)   # (P, Rb, Qb*k)
                gathered = gathered.view(batch_P, rec_batch, query_end - query_start, -1)
                lowers = q_L_pt[query_start:query_end].view(1, 1, -1, q_L_pt.shape[1])
                uppers = q_U_pt[query_start:query_end].view(1, 1, -1, q_U_pt.shape[1])
                phi = torch.all((gathered >= lowers) & (gathered < uppers), dim=3)  # (P, Rb, Qb)

                accum += scale * (phi.float() * e_block.unsqueeze(1)).sum(dim=2)    # (P, Rb)

            grad[:, record_start:record_end] = accum

        return grad


    def _row_weight_grad(self, data_pt: torch.Tensor, error_pt: torch.Tensor,
                        query_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                        batch_size_records: int, query_chunk_size: int,
                        normalize_by: int) -> torch.Tensor:
        """
        多设备封装，返回 (P, N) 的行级梯度
        """
        def worker(chunk: torch.Tensor, device: torch.device, error_chunk: torch.Tensor) -> torch.Tensor:
            q_I_pt, q_U_pt, q_L_pt = query_cache[device]
            return self._row_weight_grad_single_device(
                chunk, error_chunk, q_I_pt, q_L_pt, q_U_pt, batch_size_records, query_chunk_size, normalize_by
            )
        return self._apply_per_device(data_pt, worker, paired_tensors=[error_pt])

    def _mwu_weight_fit_resample_population(self, rows: torch.Tensor, error_vec: torch.Tensor,
                                            query_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                                            num_records: int, fitness_batch_size: int, query_chunk_size: int,
                                            steps: int = 5, eta: float = 0.8,
                                            wmin: float = 0.1, wmax: float = 10.0, l2_reg: float = 0.0) -> torch.Tensor:
        """
        对单个体 rows(N,C) 做 MWU：
        - 多轮：w ← w * exp(-η * (g + λ(w-1)))，保持 mean(w)=1，裁剪到 [wmin,wmax]
        - 按 w 重采样 N 行，返回新 rows
        error_vec: 形状 (Q,) 的 e = (noised_answers - current_answers)
        """
        device = rows.device
        N = rows.shape[0]
        eps = 1e-8
        # 把 rows/e/error 封装成 (P=1, ...) 复用多设备代码
        data_pt = rows.unsqueeze(0)
        error_pt = error_vec.unsqueeze(0)

        w = torch.ones(N, device=device, dtype=torch.float32)

        for _ in range(max(1, int(steps))):
            g = self._row_weight_grad(data_pt, error_pt, query_cache,
                                    fitness_batch_size, query_chunk_size, normalize_by=num_records).squeeze(0)  # (N,)
            if l2_reg > 0:
                g = g + l2_reg * (w - 1.0)

            # 乘法权重更新 + 归一 + 裁剪
            w = w * torch.exp(-eta * g)
            w = w / (w.mean() + eps)
            w = torch.clamp(w, wmin, wmax)

        # 重采样 N 行
        probs = (w / (w.sum() + eps)).clamp(min=eps)
        idx = torch.multinomial(probs, num_samples=N, replacement=True)
        new_rows = rows.index_select(0, idx)
        return new_rows

    def _row_pos_neg_sums_single_device(self, data_pt: torch.Tensor, error_pt: torch.Tensor, q_I_pt: torch.Tensor,
                                        q_L_pt: torch.Tensor, q_U_pt: torch.Tensor,
                                        batch_size_records: int, query_chunk_size: int) -> torch.Tensor:
        """
        返回形状 (batch_P, num_records, 2) 的张量：
        [..., 0] = pos_sum = Σ 命中查询的 max(err, 0)
        [..., 1] = neg_mag = Σ 命中查询的 max(-err, 0)
        """
        batch_P, num_records, _ = data_pt.shape
        num_queries = q_I_pt.shape[0]
        device = data_pt.device

        query_chunk_size = max(1, int(query_chunk_size))
        # 累计器
        pos_accum = torch.zeros(batch_P, num_records, dtype=torch.float32, device=device)
        neg_accum = torch.zeros(batch_P, num_records, dtype=torch.float32, device=device)

        for record_start in range(0, num_records, batch_size_records):
            record_end = min(record_start + batch_size_records, num_records)
            data_batch = data_pt[:, record_start:record_end, :]  # (P, Rb, C)
            rec_batch = record_end - record_start

            pos_local = torch.zeros(batch_P, rec_batch, dtype=torch.float32, device=device)
            neg_local = torch.zeros(batch_P, rec_batch, dtype=torch.float32, device=device)

            for query_start in range(0, num_queries, query_chunk_size):
                query_end = min(query_start + query_chunk_size, num_queries)

                q_block = q_I_pt[query_start:query_end]   # (Qb, k)
                flat_ids = q_block.reshape(-1).long()

                err_block = error_pt[:, query_start:query_end]              # (P, Qb)
                pos_block = torch.clamp_min(err_block, 0.0).unsqueeze(1)    # (P,1,Qb)
                neg_block = torch.clamp_min(-err_block, 0.0).unsqueeze(1)   # (P,1,Qb)

                if flat_ids.numel() == 0:
                    pos_local += pos_block.sum(dim=2).expand(-1, rec_batch)
                    neg_local += neg_block.sum(dim=2).expand(-1, rec_batch)
                    continue

                gathered = torch.index_select(data_batch, 2, flat_ids)      # (P, Rb, Qb*k)
                gathered = gathered.view(batch_P, rec_batch, query_end - query_start, -1)

                lowers = q_L_pt[query_start:query_end].view(1, 1, -1, q_L_pt.shape[1])
                uppers = q_U_pt[query_start:query_end].view(1, 1, -1, q_U_pt.shape[1])

                phi = torch.all((gathered >= lowers) & (gathered < uppers), dim=3)  # (P, Rb, Qb)

                pos_local += (phi.float() * pos_block).sum(dim=2)
                neg_local += (phi.float() * neg_block).sum(dim=2)

            pos_accum[:, record_start:record_end] = pos_local
            neg_accum[:, record_start:record_end] = neg_local

        # 打包为 (P, N, 2)
        return torch.stack([pos_accum, neg_accum], dim=2)


    def _row_pos_neg_sums(self, data_pt: torch.Tensor, error_pt: torch.Tensor,
                        query_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                        batch_size_records: int, query_chunk_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        多设备封装，返回：
        pos_sums: (P, N) ，neg_mags: (P, N)
        """
        def worker(chunk: torch.Tensor, device: torch.device, error_chunk: torch.Tensor) -> torch.Tensor:
            q_I_pt, q_U_pt, q_L_pt = query_cache[device]
            return self._row_pos_neg_sums_single_device(
                chunk, error_chunk, q_I_pt, q_L_pt, q_U_pt, batch_size_records, query_chunk_size
            )  # (Pk, Rb, 2)

        packed = self._apply_per_device(data_pt, worker, paired_tensors=[error_pt])  # (P, N, 2)
        return packed[..., 0], packed[..., 1]

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

            abs_err_mean = torch.abs(updated_global_error).mean(dim=0)  # (Q,)
            if not hasattr(self, "_q_ema"):
                self._q_ema = torch.zeros_like(abs_err_mean)
            self._q_ema = 0.9 * self._q_ema + 0.1 * abs_err_mean

            # 权重：1 + beta * (EMA / 均值)
            mean_ema = self._q_ema.mean().clamp_min(1e-8)
            self._q_weights = (1.0 + self.query_weight_beta * (self._q_ema / mean_ema)).detach()

            # 4. L∞适应度驱动的交叉
            if crossover_rate > 0 and P > 1:
                decoded_best = self._decode_pt(best_overall_population.unsqueeze(0))
                best_answers = self._calculate_answers_batched(decoded_best, query_cache,
                                                            fitness_batch_size, query_chunk_size,
                                                            normalize_by=num_records)
                best_global_error = noised_answers_pt - best_answers

                # 行级分解：pos_sum / neg_mag
                best_pos, best_neg = self._row_pos_neg_sums(decoded_best, best_global_error, query_cache,
                                                            fitness_batch_size, query_chunk_size)
                pop_pos,  pop_neg  = self._row_pos_neg_sums(evolved_decoded_pop, updated_global_error, query_cache,
                                                            fitness_batch_size, query_chunk_size)

                fallback_gamma = 1e-8
                neg_dom_any = torch.any(pop_neg > pop_pos + fallback_gamma) or torch.any(best_neg > best_pos + fallback_gamma)

                num_to_crossover = int(P * crossover_rate)
                _, worst_pop_indices = torch.topk(current_global_errors, num_to_crossover, largest=True)

                if neg_dom_any:
                    # —— 原策略：最大负向 ← 最大正向（利用你当前的行级“符号占优和”适应度）
                    best_row_fit = self._calculate_linf_fitness(
                        decoded_best, best_global_error, query_cache, fitness_batch_size, query_chunk_size
                    ).squeeze(0)  # (N,)
                    _, elite_indices = torch.topk(best_row_fit, crossover_num_rows, largest=False)  # 正向最强（值更小）

                    for pop_idx in worst_pop_indices:
                        if pop_idx == best_current_idx:
                            continue
                        # 受体：该个体里“负向最强”的行
                        _, worst_record_indices = torch.topk(linf_fitness[pop_idx], crossover_num_rows, largest=True)
                        population_norm[pop_idx, worst_record_indices, :] = best_overall_population[elite_indices, :]

                else:
                    # —— 兜底策略：全体正向占优
                    # 供体：best 个体里 “命中正缺口最多”的行（pos_sum 最大）
                    _, elite_indices = torch.topk(best_pos.squeeze(0), crossover_num_rows, largest=True)

                    for pop_idx in worst_pop_indices:
                        if pop_idx == best_current_idx:
                            continue
                        # 受体：该个体里 “命中正缺口最少”的行（pos_sum 最小）
                        _, worst_record_indices = torch.topk(pop_pos[pop_idx], crossover_num_rows, largest=False)
                        population_norm[pop_idx, worst_record_indices, :] = best_overall_population[elite_indices, :]

            if stagnation_counter >= stagnation_patience:
                num_diverse = max(1, int(P * diversity_fraction))
                _, worst_indices = torch.topk(current_global_errors, num_diverse, largest=True)
                population_norm = self._inject_diversity(population_norm, best_overall_population, worst_indices, diversity_jitter)
                stagnation_counter = 0
                tqdm.write(
                    f"触发多样性注入: 重置 {num_diverse} 个个体, 当前全局最佳 L2 {best_overall_score:.4f} | 原始尺度 {best_overall_score * num_records:.2f}")

            if stagnation_counter >= stagnation_patience:
                # —— 在当前最佳个体上做一次“最后一公里”raking —— 
                decoded_best = self._decode_pt(best_overall_population.unsqueeze(0)).squeeze(0)  # (N,C)
                best_answers = self._calculate_answers_batched(decoded_best.unsqueeze(0), query_cache,
                                                            fitness_batch_size, query_chunk_size,
                                                            normalize_by=num_records)  # (1,Q)
                best_err = (noised_answers_pt - best_answers).squeeze(0)  # (Q,)

                raked_decoded = self._raking_resample_population(
                    decoded_best, best_err, q_I_primary, q_L_primary, q_U_primary,
                    noised_answers_pt, num_records,
                    topk=self.raking_topk, iters=self.raking_iters, step=self.raking_step
                )
                raked_norm = self._encode_pt(raked_decoded)

                # 用“raking 后的最佳个体”覆盖一部分最差个体
                num_inject = max(1, int(P * 0.2))  # 注入 20% 种群，可按需调小
                _, worst_indices = torch.topk(current_global_errors, num_inject, largest=True)
                population_norm[worst_indices] = raked_norm.unsqueeze(0).expand_as(population_norm[worst_indices])
                decoded_best = self._decode_pt(best_overall_population.unsqueeze(0)).squeeze(0)  # (N,C)
                best_answers = self._calculate_answers_batched(decoded_best.unsqueeze(0), query_cache,
                                                            fitness_batch_size, query_chunk_size,
                                                            normalize_by=num_records)  # (1,Q)
                best_err = (noised_answers_pt - best_answers).squeeze(0)  # (Q,)

                mwu_decoded = self._mwu_weight_fit_resample_population(
                    decoded_best, best_err, query_cache, num_records,
                    fitness_batch_size, query_chunk_size,
                    steps=5, eta=0.8, wmin=0.1, wmax=10.0, l2_reg=0.0
                )
                mwu_norm = self._encode_pt(mwu_decoded)
                num_inject_mwu = max(1, int(P * 0.2))
                _, worst_indices_mwu = torch.topk(current_global_errors, num_inject_mwu, largest=True)
                population_norm[worst_indices_mwu] = mwu_norm.unsqueeze(0).expand_as(population_norm[worst_indices_mwu])


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
