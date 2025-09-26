import sys
import os
import argparse
import pandas as pd
from tqdm import tqdm
from functools import partial
import threading
import queue
import numpy as np

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
    jnp = np

# --- 从 genetic_sd 和 diffevo 库导入核心组件 ---
# --- Core classes from genetic_sd and diffevo libraries ---
from genetic_sd.genetic_sd import GSDSynthesizer
from genetic_sd.utils.dataset_jax import Dataset
from genetic_sd.utils.domain import Domain, DataType, ColumnAttribute
from snsynth.transform.type_map import TypeMap
from genetic_sd.adaptive_statistics import AdaptiveChainedStatistics, Marginals

# ==============================================================================
# V50: L∞ 定向重采样版 (PyTorch)
# V50: L-Infinity Directed Resampling Edition (PyTorch)
# ==============================================================================

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
        self.col_types_list = col_types
        
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
        is_cat = torch.tensor(self.col_types_list, device=self.device, dtype=torch.bool)
        data_original[..., is_cat] = torch.round(data_original[..., is_cat])
        return torch.clamp(data_original, self.mins_pt, self.mins_pt + self.scales_pt)

    def _calculate_answers_batched(self, data_pt: torch.Tensor, q_I_pt, q_U_pt, q_L_pt, batch_size_records: int) -> torch.Tensor:
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
        return total_counts

    def _calculate_record_query_errors_batched(self, data_pt: torch.Tensor, error_pt: torch.Tensor, q_I_pt, q_U_pt, q_L_pt, batch_size_records: int) -> torch.Tensor:
        """计算每个记录对每个查询的误差贡献"""
        batch_P, num_records, _ = data_pt.shape
        num_queries = q_I_pt.shape[0]
        record_query_errors = torch.zeros(batch_P, num_records, num_queries, device=self.device)

        for i in range(0, num_records, batch_size_records):
            data_batch = data_pt[:, i:i+batch_size_records, :]
            data_subset = data_batch[:, :, q_I_pt]
            range_cond = (data_subset >= q_L_pt) & (data_subset < q_U_pt)
            phi_matrix_batch = torch.all(range_cond, dim=3).float() # Shape: [P, batch_size, Q]
            # 广播 error_pt: [P, 1, Q]
            record_query_errors[:, i:i+batch_size_records, :] = phi_matrix_batch * error_pt.unsqueeze(1)
            
        return record_query_errors

    def _run_evolution(self, key, initial_population, target_answers, queries, k, G, P, fitness_batch_size, replacement_batch_size, crossover_rate, crossover_num_rows):
        """包含L∞适应度重采样和精英选择的主循环"""
        
        q_I_pt = torch.tensor(queries[:, :k], dtype=torch.long, device=self.device)
        q_U_pt = torch.tensor(queries[:, k:2*k], dtype=torch.float32, device=self.device)
        q_L_pt = torch.tensor(queries[:, 2*k:3*k], dtype=torch.float32, device=self.device)
        target_answers_pt = torch.tensor(target_answers, dtype=torch.float32, device=self.device).unsqueeze(0)

        population_norm = initial_population.to(self.device)
        
        decoded_pop = self._decode_pt(population_norm)
        current_answers = self._calculate_answers_batched(decoded_pop, q_I_pt, q_U_pt, q_L_pt, fitness_batch_size)
        global_errors_per_pop = torch.linalg.norm(target_answers_pt - current_answers, dim=1)
        
        best_idx = torch.argmin(global_errors_per_pop)
        best_overall_score = global_errors_per_pop[best_idx].item()
        best_overall_population = population_norm[best_idx].clone()
        
        print(f"初始最佳误差 (L2): {best_overall_score:.4f}")

        for g in tqdm(range(G), desc="世代 (Generations)"):
            
            # 1. 解码并计算全局误差
            decoded_pop = self._decode_pt(population_norm)
            current_answers = self._calculate_answers_batched(decoded_pop, q_I_pt, q_U_pt, q_L_pt, fitness_batch_size)
            global_error = target_answers_pt - current_answers

            # 2. 计算L∞适应度
            record_query_errors = self._calculate_record_query_errors_batched(decoded_pop, global_error, q_I_pt, q_U_pt, q_L_pt, fitness_batch_size)
            linf_fitness = torch.max(torch.abs(record_query_errors), dim=2).values # Shape: [P, N]

            # 3. 执行L∞适应度重采样演化
            evolved_population = population_norm.clone()
            for i in range(P):
                num_to_replace = replacement_batch_size
                
                # 找出L∞适应度最高（最差）和最低（最好）的记录
                _, top_worst_indices = torch.topk(linf_fitness[i], num_to_replace, largest=True)
                _, top_best_indices = torch.topk(linf_fitness[i], num_to_replace, largest=False)
                
                # 执行替换
                evolved_population[i, top_worst_indices, :] = evolved_population[i, top_best_indices, :]
            
            # 4. 评估并选择精英
            decoded_evolved_pop = self._decode_pt(evolved_population)
            current_global_errors = torch.linalg.norm(target_answers_pt - self._calculate_answers_batched(decoded_evolved_pop, q_I_pt, q_U_pt, q_L_pt, fitness_batch_size), dim=1)
            
            best_current_idx = torch.argmin(current_global_errors)
            best_current_score = current_global_errors[best_current_idx].item()

            if best_current_score < best_overall_score:
                best_overall_score = best_current_score
                best_overall_population = evolved_population[best_current_idx].clone()
                tqdm.write(f"世代 {g+1} 结束。发现新的最佳误差 (L2): {best_overall_score:.4f}")
            else:
                tqdm.write(f"世代 {g+1} 结束。当前世代最佳: {best_current_score:.4f}, 维持全局最佳: {best_overall_score:.4f}")

            # 5. L∞适应度驱动的交叉
            if crossover_rate > 0 and P > 1:
                # 找出全局最优解中的精英基因 (L∞最低)
                decoded_best = self._decode_pt(best_overall_population.unsqueeze(0))
                best_answers = self._calculate_answers_batched(decoded_best, q_I_pt, q_U_pt, q_L_pt, fitness_batch_size)
                best_global_error = target_answers_pt - best_answers
                best_record_query_errors = self._calculate_record_query_errors_batched(decoded_best, best_global_error, q_I_pt, q_U_pt, q_L_pt, fitness_batch_size)
                best_linf_fitness = torch.max(torch.abs(best_record_query_errors), dim=2).values.squeeze(0)
                _, elite_indices = torch.topk(best_linf_fitness, crossover_num_rows, largest=False)
                
                # 替换种群中其他个体的最差基因 (L∞最高)
                num_to_crossover = int(P * crossover_rate)
                _, worst_pop_indices = torch.topk(current_global_errors, num_to_crossover, largest=True)

                for pop_idx in worst_pop_indices:
                    if pop_idx == best_current_idx: continue
                    _, worst_record_indices = torch.topk(linf_fitness[pop_idx], crossover_num_rows, largest=True)
                    evolved_population[pop_idx, worst_record_indices, :] = best_overall_population[elite_indices, :]

            population_norm = evolved_population

        return best_overall_population

    def generate(self, stat_module: 'AdaptiveChainedStatistics', answers: np.ndarray, num_records: int, G: int, P: int, fitness_batch_size: int, replacement_batch_size: int, crossover_rate: float, crossover_num_rows: int, **kwargs) -> 'Dataset':
        queries_np, k = self._get_queries_and_k_from_stat_module(stat_module)
        if k == 0 or queries_np.shape[0] == 0:
            raise ValueError("未能从统计模块中提取有效的查询。")

        initial_population = torch.rand(P, num_records, self.num_columns, device=self.device) * 2 - 1

        final_population_norm = self._run_evolution(
            None, initial_population, answers, queries_np, k, G, P, fitness_batch_size, replacement_batch_size, crossover_rate, crossover_num_rows
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
    parser.add_argument('-G', type=int, default=500, help="演化世代数。")
    parser.add_argument('--fitness_batch_size', type=int, default=1024, help="计算适应度时的记录批处理大小。")
    parser.add_argument('--replacement_batch_size', type=int, default=256, help="每代中被替换的记录数量。")
    parser.add_argument('--crossover_rate', type=float, default=0.5)
    parser.add_argument('--crossover_num_rows', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args(args=['--input', '/home/qianqiu/experiment-trade/private_gsd/generate_script/source/acs.csv',  
                                   '--output', './synthetic_acs.csv',
                                   '--population_size', '10'])

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
    
    target_answers = stat_module.get_selected_noised_statistics()
    print(f"\n统计模块设置完成，已测量 {len(target_answers)} 个带噪查询。")
    
    print("\n--- 步骤 3: 启动 PyTorch 生成器 ---")
    generator = PrivateDEGeneratorPT(real_dataset.domain, temp_synthesizer)
    
    gen_params = {
        'G': args.G, 'P': args.population_size,
        'fitness_batch_size': args.fitness_batch_size,
        'replacement_batch_size': args.replacement_batch_size,
        'crossover_rate': args.crossover_rate, 
        'crossover_num_rows': args.crossover_num_rows
    }
    
    final_synthetic_dataset = generator.generate(stat_module, np.array(target_answers), num_records, **gen_params)

    print(f"\n--- 步骤 4: 后处理并保存合成数据 ---")
    data_list = temp_synthesizer.get_values_as_list(final_synthetic_dataset.domain, final_synthetic_dataset.df)
    output_df = temp_synthesizer._transformer.inverse_transform(data_list)
    output_df.to_csv(args.output, index=False)
    print("保存完成。")

if __name__ == '__main__':
    main()

