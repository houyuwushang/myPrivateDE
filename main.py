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
from mygenerator import PrivateDEGeneratorPT


def main():
    parser = argparse.ArgumentParser(description="Private-DE: V50 - L∞ 定向重采样版 (PyTorch)。")
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--epsilon', type=float, default=10.0)
    parser.add_argument('-P', '--population_size', type=int, default=100)
    parser.add_argument('-G', type=int, default=20000, help="演化世代数。")
    parser.add_argument('--fitness_batch_size', type=int, default=1024, help="计算适应度时的记录批处理大小。")
    parser.add_argument('--replacement_batch_size', type=int, default=100, help="每代中被替换的记录数量。")
    parser.add_argument('--crossover_rate', type=float, default=0.5)
    parser.add_argument('--crossover_num_rows', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stagnation_patience', type=int, default=5, help="在没有取得新的全局最优前允许的停滞代数。")
    parser.add_argument('--diversity_fraction', type=float, default=0.2, help="触发多样性注入时需要重采样的人口比例。")
    parser.add_argument('--diversity_jitter', type=float, default=1, help="多样性注入时在 [-1,1] 空间施加的随机扰动幅度。")
    parser.add_argument('--mutation_growth', type=float, default=0.3, help="停滞时放大 replacement_batch_size 的增长系数。")
    parser.add_argument('--min_effective_replacement', type=int, default=10, help="针对性突变时的最小替换行数下限。")
    parser.add_argument('--improvement_tolerance', type=float, default=1e-4, help="判定出现有效改进所需的 L2 阈值。")
    parser.add_argument('--aggressive_query_patience', type=int, default=3, help="单个查询在连续多少代没有明显改善后进入强化突变模式。")
    parser.add_argument('--aggressive_query_multiplier', type=float, default=2.0, help="强化突变模式下单个查询的替换行数放大倍率。")
    parser.add_argument('--query_improvement_tolerance', type=float, default=1e-4, help="判定单个查询误差得到改善所需的最小幅度。")
    parser.add_argument('--aggressive_replacement_boost', type=float, default=0.5, help="存在强化突变查询时额外放大的替换行数比例。")
    parser.add_argument('--query_chunk_size', type=int, default=320000, help="按查询分块进行批处理时的块大小，较小值可显著降低显存占用。")
    parser.add_argument('--device', type=str, default='auto',
                        help="要使用的计算设备。可选: 'auto' (默认, 自动使用所有 GPU), 'cpu', 或逗号分隔的 cuda:id 列表。")

    args = parser.parse_args(args=['--input', '/home/qianqiu/experiment-trade/private_gsd/generate_script/source/acs.csv',
                                   '--output', './synthetic_acs.csv',
                                   '--population_size', '2'])

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
    #marginals_2way = Marginals.get_all_kway_combinations(real_dataset.domain, k=2)
    marginals_1way = Marginals.get_all_kway_combinations(real_dataset.domain, k=3)
    #stat_module.add_stat_module_and_fit(marginals_2way)
    stat_module.add_stat_module_and_fit(marginals_1way)
    
    key = jax.random.PRNGKey(args.seed)
    rho = temp_synthesizer.rho 
    print(f"使用 rho={rho:.4f} 为所有查询添加噪音...")
    stat_module.private_measure_all_statistics(key, rho)
    
    noised_answers = stat_module.get_selected_noised_statistics()
    print(f"\n统计模块设置完成，已测量 {len(noised_answers)} 个带噪查询。")

    noised_answers = np.asarray(noised_answers, dtype=np.float64)
    proj_answers, l2_gap = project_2way_to_consistent_1way(stat_module, noised_answers)
    print(f"[一致性投影] 与原带噪目标的 L2 距离 ≈ {l2_gap:.6f}（这是“不可行性缺口”的估计）")
    
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

    final_synthetic_dataset = generator.generate(stat_module, np.array(proj_answers), num_records, **gen_params)

    print(f"\n--- 步骤 4: 后处理并保存合成数据 ---")
    data_list = temp_synthesizer.get_values_as_list(final_synthetic_dataset.domain, final_synthetic_dataset.df)
    output_df = temp_synthesizer._transformer.inverse_transform(data_list)
    output_df.to_csv(args.output, index=False)
    print("保存完成。")

if __name__ == '__main__':
    main()