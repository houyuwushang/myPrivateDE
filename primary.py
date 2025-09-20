import sys
import os

# --- 智能设备管理：强制 JAX 使用 CPU，为 PyTorch 释放 GPU ---
# --- Smart Device Management: Force JAX to use CPU, freeing GPUs for PyTorch ---
# 必须在 JAX 导入之前设置此环境变量
# This environment variable must be set BEFORE importing JAX
os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse
import itertools
from collections import defaultdict
from enum import Enum
from typing import Any, Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
import threading
import queue

# 将 src 目录添加到 Python 搜索路径中
# Add the src directory to the Python search path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    import jax
    import jax.numpy as jnp
    from jax import lax # 导入 lax 模块
    JAX_AVAILABLE = True
    print("JAX 已成功导入。根据配置，将使用 CPU 进行统计计算。")
    print("JAX检测到的设备:", jax.devices())
except ImportError:
    JAX_AVAILABLE = False
    print("警告: JAX 未安装。将仅使用 NumPy 后端。")
    jnp = np
    lax = None # 如果 JAX 未安装，提供一个占位符

# --- 核心类从 genetic_sd 库导入 ---
# --- Core classes from genetic_sd library ---
from genetic_sd.genetic_sd import GSDSynthesizer
from genetic_sd.utils.dataset_jax import Dataset
from genetic_sd.utils.domain import Domain, DataType, ColumnAttribute
from snsynth.transform.type_map import TypeMap
# --- 新增导入: 统计管理模块 ---
# --- New import: Statistics management module ---
from genetic_sd.adaptive_statistics import AdaptiveChainedStatistics, Marginals
import torch
import torch.nn.functional as F

# ==============================================================================
# 主要生成器类
# Main Generator Class
# ==============================================================================

class PrivateDEGeneratorPT:
    """
    【最终稳定版 V31 - 质心引导修复版】
    此版本使用“质心引导”替代随机目标选择，并修复了L2误差报告的bug。
    """
    def __init__(self, real_dataset: 'Dataset'):
        self.domain = real_dataset.domain
        self.num_columns = len(self.domain.attrs)
        
        mins, maxs, col_types = [], [], []
        for col_name in self.domain.attrs:
            config = self.domain.config[col_name]
            col_type_str = config.get('type', config.get(ColumnAttribute.TYPE))
            if col_type_str in ('float', DataType.CONTINUOUS.value):
                col_types.append(0)
                min_val, max_val = real_dataset.df[col_name].min(), real_dataset.df[col_name].max()
                mins.append(min_val); maxs.append(max_val)
            else:
                col_types.append(1)
                mins.append(0.0)
                size = config.get('size', config.get(ColumnAttribute.SIZE))
                maxs.append(float(size - 1))

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.devices = [torch.device("cuda:0"), torch.device("cuda:1")]
            self.main_device = self.devices[1]
            print(f"检测到多个 GPU。将使用 {self.devices[0]} 和 {self.devices[1]} 进行并行计算。主设备: {self.main_device}")
        elif torch.cuda.is_available():
            self.devices = [torch.device("cuda:0")]
            self.main_device = self.devices[0]
            print(f"检测到单个 GPU。将使用设备: {self.main_device}")
        else:
            self.devices = [torch.device("cpu")]
            self.main_device = self.devices[0]
            print("未检测到 GPU。将使用 CPU。")
        
        self.mins_pt = torch.tensor(mins, dtype=torch.float32, device=self.main_device)
        self.scales_pt = torch.tensor(maxs, dtype=torch.float32, device=self.main_device) - self.mins_pt
        self.scales_pt[self.scales_pt == 0] = 1.0
        self.col_types_list = col_types
        
        print("生成器已初始化 (最终稳定版 V31 - 质心引导修复版)。")
    
    @staticmethod
    def _get_queries_and_k_from_stat_module(stat_module: 'AdaptiveChainedStatistics') -> Tuple[np.ndarray, int]:
        first_marginal_module = next((m for m in stat_module.stat_modules if isinstance(m, Marginals)), None)
        k = first_marginal_module.k if first_marginal_module else 0
        all_queries_list = [np.array(m.queries) for m in stat_module.stat_modules if hasattr(m, 'queries') and m.queries.shape[0] > 0]
        if not all_queries_list: return np.array([]), 0
        return np.concatenate(all_queries_list), k

    @staticmethod
    def _decode_pt(data_scaled_pt: torch.Tensor, mins_pt: torch.Tensor, scales_pt: torch.Tensor, col_types: List[int]) -> torch.Tensor:
        data_original = data_scaled_pt * scales_pt + mins_pt
        is_cat = torch.tensor(col_types, device=data_original.device, dtype=torch.bool)
        data_original[..., is_cat] = torch.round(data_original[..., is_cat])
        return torch.clamp(data_original, mins_pt, mins_pt + scales_pt)

    def _calculate_answers_batched(self, data_pt: torch.Tensor, q_I_pt, q_U_pt, q_L_pt, batch_size_records: int, query_batch_size: int) -> torch.Tensor:
        batch_P, num_records, _ = data_pt.shape
        num_queries = q_I_pt.shape[0]
        total_counts = torch.zeros(batch_P, num_queries, dtype=torch.int32, device=data_pt.device)
        
        for q_start in range(0, num_queries, query_batch_size):
            q_end = min(q_start + query_batch_size, num_queries)
            q_I_batch = q_I_pt[q_start:q_end]
            q_U_batch = q_U_pt[q_start:q_end]
            q_L_batch = q_L_pt[q_start:q_end]

            for i in range(0, num_records, batch_size_records):
                data_batch = data_pt[:, i:i+batch_size_records, :]
                data_subset = data_batch[:, :, q_I_batch]
                range_cond = (data_subset >= q_L_batch) & (data_subset < q_U_batch)
                phi_results = torch.all(range_cond, dim=3)
                total_counts[:, q_start:q_end] += torch.sum(phi_results, dim=1)
                
        return total_counts

    def _calculate_fitness_batched(self, data_pt: torch.Tensor, error_pt: torch.Tensor, q_I_pt, q_U_pt, q_L_pt, batch_size_records: int, query_batch_size: int) -> torch.Tensor:
        batch_P, num_records, _ = data_pt.shape
        num_queries = q_I_pt.shape[0]
        fitness = torch.zeros(batch_P, num_records, device=data_pt.device)

        for q_start in range(0, num_queries, query_batch_size):
            q_end = min(q_start + query_batch_size, num_queries)
            q_I_batch = q_I_pt[q_start:q_end]
            q_U_batch = q_U_pt[q_start:q_end]
            q_L_batch = q_L_pt[q_start:q_end]
            error_batch = error_pt[:, q_start:q_end]

            for i in range(0, num_records, batch_size_records):
                data_batch = data_pt[:, i:i+batch_size_records, :]
                data_subset = data_batch[:, :, q_I_batch]
                range_cond = (data_subset >= q_L_batch) & (data_subset < q_U_batch)
                phi_matrix_batch = torch.all(range_cond, dim=3).float()
                fitness[:, i:i+batch_size_records] += torch.einsum('brq,bq->br', phi_matrix_batch, error_batch)
        
        return fitness
    
    def _parallel_evolution_step(self, result_queue, pop_chunk, device, answers_pt, q_I_pt, q_U_pt, q_L_pt, S, lr, diff_scale, batch_size_records, num_records, query_batch_size):
        x_t = pop_chunk.to(device)
        batch_P_chunk = x_t.shape[0]

        for _ in range(S):
            x_01_scaled = (x_t + 1) / 2.0
            decoded_population = self._decode_pt(x_01_scaled, self.mins_pt.to(device), self.scales_pt.to(device), self.col_types_list)
            
            current_answers_pop = self._calculate_answers_batched(decoded_population, q_I_pt.to(device), q_U_pt.to(device), q_L_pt.to(device), batch_size_records, query_batch_size)
            error_pop = answers_pt.to(device) - current_answers_pop
            fitness_scores = self._calculate_fitness_batched(decoded_population, error_pop, q_I_pt.to(device), q_U_pt.to(device), q_L_pt.to(device), batch_size_records, query_batch_size)
            
            # --- V31: 质心引导演化逻辑 ---
            positive_mask = fitness_scores > 0
            negative_mask = fitness_scores < 0
            
            # 默认目标是记录自身（保持不变）
            x_target_base = x_t.clone() 
            
            for i in range(batch_P_chunk):
                pos_indices = torch.where(positive_mask[i])[0]
                neg_indices = torch.where(negative_mask[i])[0]

                # 只有当同时存在正负适应度记录时，才进行定向演化
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    # 计算所有“好”记录的质心
                    mean_of_positives = torch.mean(x_t[i, pos_indices], dim=0)
                    # 将这个质心作为所有“坏”记录的学习目标
                    x_target_base[i, neg_indices] = mean_of_positives.unsqueeze(0)
            
            # 引入差分向量来增强探索
            rand_indices_1 = torch.randint(0, num_records, (batch_P_chunk, num_records), device=device)
            rand_indices_2 = torch.randint(0, num_records, (batch_P_chunk, num_records), device=device)
            
            x_r1 = torch.gather(x_t, 1, rand_indices_1.unsqueeze(-1).expand(-1, -1, self.num_columns))
            x_r2 = torch.gather(x_t, 1, rand_indices_2.unsqueeze(-1).expand(-1, -1, self.num_columns))
            
            diff_vec = x_r1 - x_r2
            x_target = x_target_base + diff_scale * diff_vec
            
            # 向增强后的目标进行插值更新
            x_t = x_t * (1.0 - lr) + x_target * lr
        
        result_queue.put(x_t.to(self.main_device))

    def generate(self, stat_module: 'AdaptiveChainedStatistics', answers: np.ndarray, num_records: int, G: int, P: int, S: int, m_size: int, g_swap: int, lr: float, diff_scale: float, batch_size: int, batch_size_records: int, **kwargs) -> 'Dataset':
        
        queries_np, k = self._get_queries_and_k_from_stat_module(stat_module)
        if k == 0 or queries_np.shape[0] == 0:
            raise ValueError("未能从统计模块中提取有效的查询。")

        q_I, q_U, q_L = queries_np[:, :k], queries_np[:, k:2*k], queries_np[:, 2*k:3*k]
        answers_pt = torch.tensor(np.copy(answers), dtype=torch.float32, device=self.main_device)
        q_I_pt = torch.tensor(q_I, dtype=torch.long, device=self.main_device)
        q_U_pt = torch.tensor(q_U, dtype=torch.float32, device=self.main_device)
        q_L_pt = torch.tensor(q_L, dtype=torch.float32, device=self.main_device)
        
        query_batch_size = 512

        print(f"启动高性能生成器 (V31): G={G}, P={P}, S={S}, lr={lr}, diff_scale={diff_scale}, Pop Batch={batch_size}, Record Batch={batch_size_records}, Query Batch={query_batch_size}")
        
        population_neg1_1 = torch.rand(P, num_records, self.num_columns, device=self.main_device) * 2 - 1
        global_best_score = float('inf')
        best_overall_individual = population_neg1_1[0].clone()

        for g in tqdm(range(G), desc="世代 (Generations)"):
            
            if g > 0 and (g + 1) % g_swap == 0 and P > 1:
                rows_to_swap = torch.randperm(num_records, device=self.main_device)[:m_size]
                population_neg1_1[1:, rows_to_swap, :] = best_overall_individual[rows_to_swap, :]

            threads = []
            result_queue = queue.Queue()
            pop_chunks = torch.chunk(population_neg1_1, len(self.devices), dim=0)
            
            for i, device in enumerate(self.devices):
                thread = threading.Thread(target=self._parallel_evolution_step, args=(
                    result_queue, pop_chunks[i], device, answers_pt, q_I_pt, q_U_pt, q_L_pt,
                    S, lr, diff_scale, batch_size_records, num_records, query_batch_size
                ))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()

            evolved_chunks = []
            while not result_queue.empty():
                evolved_chunks.append(result_queue.get())
            
            evolved_population_neg1_1 = torch.cat(evolved_chunks, dim=0)
            
            pool_neg1_1 = torch.cat([population_neg1_1, evolved_population_neg1_1])
            
            all_scores = []
            for i in range(0, pool_neg1_1.shape[0], batch_size):
                pool_batch_01 = (pool_neg1_1[i:i+batch_size] + 1) / 2.0
                decoded_pool_batch = self._decode_pt(pool_batch_01, self.mins_pt, self.scales_pt, self.col_types_list)
                current_answers_batch = self._calculate_answers_batched(decoded_pool_batch, q_I_pt, q_U_pt, q_L_pt, batch_size_records, query_batch_size)
                error_batch = answers_pt - current_answers_batch
                all_scores.append(torch.linalg.norm(error_batch.float(), dim=1))

            scores_tensor = torch.cat(all_scores)
            _, top_indices = torch.topk(-scores_tensor, k=P)
            
            population_neg1_1 = pool_neg1_1[top_indices]
            
            # --- V31: 修复 L2 误差报告 Bug ---
            current_best_score = scores_tensor[top_indices[0]].item()

            if current_best_score < global_best_score:
                global_best_score = current_best_score
                best_overall_individual = population_neg1_1[0].clone()
            
            tqdm.write(f"世代 {g+1} 结束。最佳误差 (L2): {global_best_score:.4f}")

        final_decoded_pt = self._decode_pt((best_overall_individual + 1) / 2.0, self.mins_pt, self.scales_pt, self.col_types_list)
        final_df = pd.DataFrame(final_decoded_pt.cpu().numpy(), columns=self.domain.attrs)
        
        return Dataset(final_df, self.domain)


class PrivateDE:
    def __init__(self, real_dataset: 'Dataset', stat_module: 'AdaptiveChainedStatistics'):
        self.real_dataset = real_dataset
        self.stat_module = stat_module
        print("\n--- 3. 初始化生成器 ---")
        self.generator = PrivateDEGeneratorPT(real_dataset)

    def run(self, **gen_params) -> 'Dataset':
        print("\n--- 算法 run 方法 ---")
        true_answers = self.stat_module.get_all_true_statistics()
        print(f"从统计模块中获取了 {len(true_answers)} 个查询的真实答案。")
        return self.generator.generate(self.stat_module, true_answers, len(self.real_dataset), **gen_params)

def main():
    parser = argparse.ArgumentParser(description="Private-DE: Differentially Private Data Synthesis via Diffusion Evolution.")
    parser.add_argument('--input', type=str, required=True, help="输入真实数据 CSV 文件的路径。")
    parser.add_argument('--output', type=str, required=True, help="保存输出的合成数据 CSV 文件的路径。")
    parser.add_argument('-G', type=int, default=10, help="演化世代数。")
    parser.add_argument('-P', type=int, default=100, help="种群大小。")
    parser.add_argument('-S', type=int, default=50, help="内层优化循环的步数。")
    parser.add_argument('-msize', type=int, default=1000, help="交叉互换的记录行数。")
    parser.add_argument('-gswap', type=int, default=2, help="种群间个体迁移的频率（每 N 个世代）。")
    parser.add_argument('--lr', type=float, default=0.1, help="内层优化循环的学习率。")
    parser.add_argument('--diff_scale', type=float, default=0.5, help="差分变异的缩放因子 (F)。")
    parser.add_argument('--batch_size', type=int, default=4, help="处理种群的批次大小。")
    parser.add_argument('--batch_size_records', type=int, default=2048, help="处理单个数据集中记录的批次大小，以控制显存。")

    
    args = parser.parse_args(args=['--input', '/home/qianqiu/experiment-trade/private_gsd/generate_script/source/acs.csv',  
                                   '--output', './synthetic_acs.csv',
                                  '-G', '10', '-P', '100', '-S', '50', '-msize', '1000', '-gswap', '2',
                                   '--lr', '0.1', '--diff_scale', '0.5',
                                   '--batch_size', '2', '--batch_size_records', '20480']) # 使用您提供的较大值

    if not os.path.exists(args.input):
        print(f"错误: 输入文件 '{args.input}' 不存在。"); return

    print("--- 步骤 1: 加载数据并推断元数据 ---")
    raw_df = pd.read_csv(args.input, skipinitialspace=True, na_values='?')
    types = TypeMap.infer_column_types(raw_df)
    
    meta_data = {}
    print("正在为数值列推断公共边界 (min/max)...")
    for col in types['ordinal_columns'] + types['continuous_columns']:
        min_val, max_val = raw_df[col].min(), raw_df[col].max()
        meta_data[col] = {'type': 'int' if col in types['ordinal_columns'] else 'float', 'lower': min_val, 'upper': max_val}
    for col in types['categorical_columns']:
        meta_data[col] = {'type': 'string'}

    temp_synthesizer = GSDSynthesizer(epsilon=1.0)
    real_dataset = temp_synthesizer._get_data(
        raw_df, 
        meta_data=meta_data, 
        categorical_columns=types['categorical_columns'], 
        ordinal_columns=types['ordinal_columns'], 
        continuous_columns=types['continuous_columns']
    )
    
    print(f"\n数据加载和预处理完成！Domain 中的属性: {real_dataset.domain.attrs}, 记录数: {len(real_dataset)}")
    
    print("\n--- 步骤 2: 设置统计查询 ---")
    stat_module = AdaptiveChainedStatistics(real_dataset)
    marginals_2way = Marginals.get_all_kway_combinations(real_dataset.domain, k=2)
    stat_module.add_stat_module_and_fit(marginals_2way)
    print("\n统计模块设置完成。")
    
    private_de = PrivateDE(real_dataset, stat_module)
    
    gen_params = {
        'G': args.G, 'P': args.P, 'S': args.S,
        'm_size': args.msize, 'g_swap': args.gswap,
        'batch_size': args.batch_size,
        'batch_size_records': args.batch_size_records,
        'lr': args.lr,
        'diff_scale': args.diff_scale
    }
    
    final_synthetic_dataset = private_de.run(**gen_params)

    print(f"\n--- 保存合成数据到 '{args.output}' ---")
    final_df_numeric = final_synthetic_dataset.df
    data_list = temp_synthesizer.get_values_as_list(final_synthetic_dataset.domain, final_df_numeric)
    output_df = temp_synthesizer._transformer.inverse_transform(data_list)
    output_df.to_csv(args.output, index=False)
    print("保存完成。")

if __name__ == '__main__':
    main()

