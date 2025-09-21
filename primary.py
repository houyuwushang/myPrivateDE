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
    def __init__(self, domain: Domain, preprocessor):
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PyTorch 将使用设备: {self.device}")
        
        self.mins_pt = torch.tensor(mins, dtype=torch.float32, device=self.device)
        self.scales_pt = torch.tensor(maxs, dtype=torch.float32, device=self.device) - self.mins_pt
        self.scales_pt[self.scales_pt == 0] = 1.0
        self.maxs_pt = self.mins_pt + self.scales_pt
        self.col_types_list = col_types
        self.discrete_mask = torch.tensor([ct != 0 for ct in col_types], dtype=torch.bool, device=self.device)
        
        print("生成器已初始化 (V50 - L∞ 定向重采样版)。")

    @staticmethod
    def _get_queries_and_k_from_stat_module(stat_module: 'AdaptiveChainedStatistics') -> tuple[np.ndarray, int]:
        first_marginal_module = next((m for m in stat_module.stat_modules if isinstance(m, Marginals)), None)
        k = first_marginal_module.k if first_marginal_module else 0
        all_queries_list = [np.array(m.queries) for m in stat_module.stat_modules if hasattr(m, 'queries') and m.queries.shape[0] > 0]
        if not all_queries_list: return np.array([]), 0
        return np.concatenate(all_queries_list), k

    def _decode_pt(self, data_norm_pt: torch.Tensor) -> torch.Tensor:
        """将标准化的 [-1, 1] 数据解码回原始范围"""
        data_01 = (data_norm_pt + 1.0) / 2.0
        data_original = data_01 * self.scales_pt + self.mins_pt
        data_original[..., self.discrete_mask] = torch.round(data_original[..., self.discrete_mask])
        return torch.clamp(data_original, self.mins_pt, self.maxs_pt)

    def _encode_pt(self, data_original_pt: torch.Tensor) -> torch.Tensor:
        """将原始数据重新编码回 [-1, 1] 范围"""
        data_clamped = torch.clamp(data_original_pt, self.mins_pt, self.maxs_pt)
        data_01 = (data_clamped - self.mins_pt) / self.scales_pt
        data_norm = data_01 * 2.0 - 1.0
        return torch.clamp(data_norm, -1.0, 1.0)

    def _calculate_answers_batched(self, data_pt: torch.Tensor, q_I_pt, q_U_pt, q_L_pt,
                                   batch_size_records: int, normalize_by: Optional[int] = None) -> torch.Tensor:
        """在 PyTorch 中分批计算查询答案"""
        batch_P, num_records, _ = data_pt.shape
        num_queries = q_I_pt.shape[0]
        total_counts = torch.zeros(batch_P, num_queries, dtype=torch.float32, device=self.device)

        for i in range(0, num_records, batch_size_records):
            data_batch = data_pt[:, i:i+batch_size_records, :]
            data_subset = data_batch[:, :, q_I_pt]
            range_cond = (data_subset >= q_L_pt) & (data_subset < q_U_pt)
            phi_results = torch.all(range_cond, dim=3)
            total_counts += torch.sum(phi_results, dim=1)
        if normalize_by is not None and normalize_by > 0:
            total_counts = total_counts / float(normalize_by)
        return total_counts

    def _calculate_record_query_errors_batched(self, data_pt: torch.Tensor, error_pt: torch.Tensor, q_I_pt, q_U_pt, q_L_pt,
                                               batch_size_records: int) -> tuple[torch.Tensor, torch.Tensor]:
        """计算每个记录对每个查询的误差贡献"""
        batch_P, num_records, _ = data_pt.shape
        num_queries = q_I_pt.shape[0]
        record_query_errors = torch.zeros(batch_P, num_records, num_queries, device=self.device)
        phi_matrix = torch.zeros(batch_P, num_records, num_queries, device=self.device)

        for i in range(0, num_records, batch_size_records):
            data_batch = data_pt[:, i:i+batch_size_records, :]
            data_subset = data_batch[:, :, q_I_pt]
            range_cond = (data_subset >= q_L_pt) & (data_subset < q_U_pt)
            phi_matrix_batch = torch.all(range_cond, dim=3).float() # Shape: [P, batch_size, Q]
            # 广播 error_pt: [P, 1, Q]
            record_query_errors[:, i:i+batch_size_records, :] = phi_matrix_batch * error_pt.unsqueeze(1)
            phi_matrix[:, i:i+batch_size_records, :] = phi_matrix_batch

        return record_query_errors, phi_matrix

    def _sample_values_inside(self, col_idx: int, low: float, up: float, num_samples: int) -> torch.Tensor:
        min_val = float(self.mins_pt[col_idx].item())
        max_val = float(self.maxs_pt[col_idx].item())
        if self.col_types_list[col_idx] == 0:
            low_clamped = max(low, min_val)
            up_clamped = min(up, max_val)
            if up_clamped <= low_clamped + 1e-8:
                return torch.full((num_samples,), low_clamped, device=self.device)
            return torch.empty(num_samples, device=self.device).uniform_(low_clamped, up_clamped)
        domain_min = int(round(min_val))
        domain_max = int(round(max_val))
        low_int = max(domain_min, int(np.ceil(low)))
        high_int = min(domain_max, int(np.floor(up - 1e-6)))
        if high_int < low_int:
            low_int, high_int = domain_min, domain_max
        possible = torch.arange(low_int, high_int + 1, device=self.device, dtype=torch.float32)
        if possible.numel() == 1:
            return possible.repeat(num_samples)
        idx = torch.randint(0, possible.numel(), (num_samples,), device=self.device)
        return possible[idx]

    def _sample_values_outside(self, col_idx: int, low: float, up: float, num_samples: int) -> torch.Tensor:
        min_val = float(self.mins_pt[col_idx].item())
        max_val = float(self.maxs_pt[col_idx].item())
        if self.col_types_list[col_idx] == 0:
            low_clamped = max(low, min_val)
            up_clamped = min(up, max_val)
            ranges = []
            if low_clamped - min_val > 1e-6:
                ranges.append((min_val, low_clamped))
            if max_val - up_clamped > 1e-6:
                ranges.append((up_clamped, max_val))
            if not ranges:
                return torch.empty(num_samples, device=self.device).uniform_(min_val, max_val)
            if len(ranges) == 1:
                start, end = ranges[0]
                if end <= start + 1e-8:
                    return torch.full((num_samples,), start, device=self.device)
                return torch.empty(num_samples, device=self.device).uniform_(start, end)
            choice = torch.randint(0, len(ranges), (num_samples,), device=self.device)
            samples = torch.empty(num_samples, device=self.device)
            for idx, (start, end) in enumerate(ranges):
                mask = choice == idx
                if mask.any():
                    if end <= start + 1e-8:
                        samples[mask] = start
                    else:
                        samples[mask] = torch.empty(int(mask.sum().item()), device=self.device).uniform_(start, end)
            return samples
        domain_min = int(round(min_val))
        domain_max = int(round(max_val))
        low_int = int(np.ceil(low))
        high_int = int(np.floor(up - 1e-6))
        all_values = torch.arange(domain_min, domain_max + 1, device=self.device, dtype=torch.float32)
        mask = (all_values < low_int) | (all_values > high_int)
        possible = all_values[mask]
        if possible.numel() == 0:
            possible = all_values
        idx = torch.randint(0, possible.numel(), (num_samples,), device=self.device)
        return possible[idx]

    def _sample_value_inside(self, col_idx: int, low: float, up: float) -> torch.Tensor:
        return self._sample_values_inside(col_idx, low, up, 1).squeeze(0)

    def _sample_value_outside(self, col_idx: int, low: float, up: float) -> torch.Tensor:
        return self._sample_values_outside(col_idx, low, up, 1).squeeze(0)

    def _mutate_rows_to_interval_batch(self, rows: torch.Tensor, col_ids: torch.Tensor, lowers: torch.Tensor,
                                       uppers: torch.Tensor, make_inside: bool) -> None:
        if rows.numel() == 0:
            return
        num_rows = rows.shape[0]
        for pos in range(col_ids.shape[0]):
            col_idx = int(col_ids[pos].item())
            if col_idx < 0 or col_idx >= self.num_columns:
                continue
            low = float(lowers[pos].item())
            up = float(uppers[pos].item())
            if make_inside:
                new_values = self._sample_values_inside(col_idx, low, up, num_rows)
            else:
                new_values = self._sample_values_outside(col_idx, low, up, num_rows)
            rows[:, col_idx] = new_values

    def _apply_targeted_mutations(self, decoded_population: torch.Tensor, global_error: torch.Tensor,
                                  phi_matrix: torch.Tensor, q_I_pt: torch.Tensor, q_L_pt: torch.Tensor,
                                  q_U_pt: torch.Tensor, replacement_batch_size: int, num_records: int) -> torch.Tensor:
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
                if err_val > 0:
                    candidate_mask = phi_matrix[pop_idx, :, q_idx] < 0.5
                else:
                    candidate_mask = phi_matrix[pop_idx, :, q_idx] >= 0.5
                candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
                if candidate_indices.numel() == 0:
                    continue
                desired = max(1, int(abs(err_val) * num_records))
                remaining = replacement_batch_size - rows_changed
                if remaining <= 0:
                    break
                num_updates = min(desired, remaining, candidate_indices.numel())
                perm = torch.randperm(candidate_indices.numel(), device=self.device)[:num_updates]
                chosen_indices = candidate_indices[perm]
                col_ids = q_I_pt[q_idx]
                lowers = q_L_pt[q_idx]
                uppers = q_U_pt[q_idx]
                rows_to_mutate = decoded_population[pop_idx].index_select(0, chosen_indices)
                self._mutate_rows_to_interval_batch(rows_to_mutate, col_ids, lowers, uppers,
                                                    make_inside=(err_val > 0))
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

    def _run_evolution(self, key, initial_population, target_answers, queries, k, num_records, G, P,
                       fitness_batch_size, replacement_batch_size, crossover_rate, crossover_num_rows,
                       stagnation_patience: int, diversity_fraction: float, diversity_jitter: float,
                       mutation_growth: float, min_effective_replacement: int, improvement_tolerance: float):
        """包含L∞适应度重采样和精英选择的主循环"""
        
        q_I_pt = torch.tensor(queries[:, :k], dtype=torch.long, device=self.device)
        q_U_pt = torch.tensor(queries[:, k:2*k], dtype=torch.float32, device=self.device)
        q_L_pt = torch.tensor(queries[:, 2*k:3*k], dtype=torch.float32, device=self.device)
        target_answers_pt = torch.tensor(target_answers, dtype=torch.float32, device=self.device).unsqueeze(0)

        population_norm = initial_population.to(self.device)
        
        decoded_pop = self._decode_pt(population_norm)
        current_answers = self._calculate_answers_batched(decoded_pop, q_I_pt, q_U_pt, q_L_pt,
                                                          fitness_batch_size, normalize_by=num_records)
        global_errors_per_pop = torch.linalg.norm(target_answers_pt - current_answers, dim=1)
        
        best_idx = torch.argmin(global_errors_per_pop)
        best_overall_score = global_errors_per_pop[best_idx].item()
        best_overall_population = population_norm[best_idx].clone()
        
        print(f"初始最佳误差 (L2): {best_overall_score:.4f} | 原始尺度 {best_overall_score * num_records:.2f}")

        stagnation_counter = 0

        for g in tqdm(range(G), desc="世代 (Generations)"):

            # 1. 解码并计算全局误差
            decoded_pop = self._decode_pt(population_norm)
            current_answers = self._calculate_answers_batched(decoded_pop, q_I_pt, q_U_pt, q_L_pt,
                                                              fitness_batch_size, normalize_by=num_records)
            global_error = target_answers_pt - current_answers

            # 2. 基于误差执行针对性突变
            record_query_errors, phi_matrix = self._calculate_record_query_errors_batched(
                decoded_pop, global_error, q_I_pt, q_U_pt, q_L_pt, fitness_batch_size)
            effective_replacement = int(replacement_batch_size * (1.0 + stagnation_counter * mutation_growth))
            effective_replacement = max(min_effective_replacement, effective_replacement)
            effective_replacement = min(effective_replacement, num_records)
            evolved_decoded_pop = self._apply_targeted_mutations(
                decoded_pop.clone(), global_error, phi_matrix, q_I_pt, q_L_pt, q_U_pt,
                effective_replacement, num_records)

            population_norm = self._encode_pt(evolved_decoded_pop)

            # 3. 评估并选择精英
            updated_answers = self._calculate_answers_batched(evolved_decoded_pop, q_I_pt, q_U_pt, q_L_pt,
                                                              fitness_batch_size, normalize_by=num_records)
            updated_global_error = target_answers_pt - updated_answers
            updated_record_query_errors, _ = self._calculate_record_query_errors_batched(
                evolved_decoded_pop, updated_global_error, q_I_pt, q_U_pt, q_L_pt, fitness_batch_size)
            linf_fitness = torch.max(torch.abs(updated_record_query_errors), dim=2).values
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
                best_answers = self._calculate_answers_batched(decoded_best, q_I_pt, q_U_pt, q_L_pt,
                                                               fitness_batch_size, normalize_by=num_records)
                best_global_error = target_answers_pt - best_answers
                best_record_query_errors, _ = self._calculate_record_query_errors_batched(
                    decoded_best, best_global_error, q_I_pt, q_U_pt, q_L_pt, fitness_batch_size)
                best_linf_fitness = torch.max(torch.abs(best_record_query_errors), dim=2).values.squeeze(0)
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

        return best_overall_population

    def generate(self, stat_module: 'AdaptiveChainedStatistics', answers: np.ndarray, num_records: int, G: int, P: int, fitness_batch_size: int, replacement_batch_size: int, crossover_rate: float, crossover_num_rows: int, **kwargs) -> 'Dataset':
        queries_np, k = self._get_queries_and_k_from_stat_module(stat_module)
        if k == 0 or queries_np.shape[0] == 0:
            raise ValueError("未能从统计模块中提取有效的查询。")

        initial_population = torch.rand(P, num_records, self.num_columns, device=self.device) * 2 - 1

        stagnation_patience = kwargs.get('stagnation_patience', 40)
        diversity_fraction = kwargs.get('diversity_fraction', 0.2)
        diversity_jitter = kwargs.get('diversity_jitter', 0.15)
        mutation_growth = kwargs.get('mutation_growth', 0.3)
        min_effective_replacement = kwargs.get('min_effective_replacement', max(1, replacement_batch_size // 2))
        improvement_tolerance = kwargs.get('improvement_tolerance', 1e-4)

        final_population_norm = self._run_evolution(
            None, initial_population, answers, queries_np, k, num_records, G, P,
            fitness_batch_size, replacement_batch_size, crossover_rate, crossover_num_rows,
            stagnation_patience, diversity_fraction, diversity_jitter,
            mutation_growth, min_effective_replacement, improvement_tolerance
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
    parser.add_argument('-G', type=int, default=5000, help="演化世代数。")
    parser.add_argument('--fitness_batch_size', type=int, default=1024, help="计算适应度时的记录批处理大小。")
    parser.add_argument('--replacement_batch_size', type=int, default=256, help="每代中被替换的记录数量。")
    parser.add_argument('--crossover_rate', type=float, default=0.5)
    parser.add_argument('--crossover_num_rows', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stagnation_patience', type=int, default=40, help="在没有取得新的全局最优前允许的停滞代数。")
    parser.add_argument('--diversity_fraction', type=float, default=0.2, help="触发多样性注入时需要重采样的人口比例。")
    parser.add_argument('--diversity_jitter', type=float, default=0.15, help="多样性注入时在 [-1,1] 空间施加的随机扰动幅度。")
    parser.add_argument('--mutation_growth', type=float, default=0.3, help="停滞时放大 replacement_batch_size 的增长系数。")
    parser.add_argument('--min_effective_replacement', type=int, default=128, help="针对性突变时的最小替换行数下限。")
    parser.add_argument('--improvement_tolerance', type=float, default=1e-4, help="判定出现有效改进所需的 L2 阈值。")

    args = parser.parse_args(args=['--input', '/home/qianqiu/experiment-trade/private_gsd/generate_script/source/acs.csv',  
                                   '--output', './synthetic_acs.csv',
                                   '--population_size', '4'])

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
    
    target_answers = stat_module.get_selected_noised_statistics()
    print(f"\n统计模块设置完成，已测量 {len(target_answers)} 个带噪查询。")
    
    print("\n--- 步骤 3: 启动 PyTorch 生成器 ---")
    generator = PrivateDEGeneratorPT(real_dataset.domain, temp_synthesizer)
    
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
        'improvement_tolerance': args.improvement_tolerance
    }
    
    final_synthetic_dataset = generator.generate(stat_module, np.array(target_answers), num_records, **gen_params)

    print(f"\n--- 步骤 4: 后处理并保存合成数据 ---")
    data_list = temp_synthesizer.get_values_as_list(final_synthetic_dataset.domain, final_synthetic_dataset.df)
    output_df = temp_synthesizer._transformer.inverse_transform(data_list)
    output_df.to_csv(args.output, index=False)
    print("保存完成。")

if __name__ == '__main__':
    main()