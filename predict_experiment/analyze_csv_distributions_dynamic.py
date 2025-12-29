#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python analyze_csv_distributions_dynamic.py --csv_paths ./dataset_result/benchmarkserving_result/llama/results_20251110_073554_llama.csv ./dataset_result/benchmarkserving_result/qwen/results_20251111_025525_qwen.csv ./dataset_result/benchmarkserving_result/deepseek/results_20251112_144147_deepseek.csv --output_dir results_test/test1
python analyze_csv_distributions_dynamic.py --csv_paths ./dataset_result/benchmarkserving_result/llama/merge/llama_20251114T064238_30.csv ./dataset_result/benchmarkserving_result/qwen/merge/qwen_20251114T064707_30.csv  ./dataset_result/benchmarkserving_result/deepseek/merge/deepseek_20251116T022356_30.csv --output_dir results_test/test5
"""
"""
analyze_csv_distributions_dynamic.py
（更新版：支持在文件名任意位置识别 llama/qwen/deepseek；mode=1 显示所有图例并在图中画均值虚线）
"""
from typing import List, Dict, Tuple
from datetime import datetime
import os
import re
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity


def _find_output_len_cols(columns: List[str]) -> List[str]:
    pattern = re.compile(r"^output_len(?:_?)(\d+)$", re.IGNORECASE)
    cols = []
    for c in columns:
        if not isinstance(c, str):
            continue
        m = pattern.match(c.strip())
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort(key=lambda x: x[0])
    return [c for _, c in cols]


def calculate_distribution_from_csv(csv_path: str) -> Tuple[pd.DataFrame, Tuple[float, float, float, float]]:
    df = pd.read_csv(csv_path)

    if 'req_id' not in df.columns:
        raise ValueError("CSV 文件必须包含 'req_id' 列")
    if 'prompt' not in df.columns:
        raise ValueError("CSV 文件必须包含 'prompt' 列")

    output_len_columns = _find_output_len_cols(list(df.columns))
    if not output_len_columns:
        raise ValueError("CSV 文件中未找到任何 output_len* 列 (例如 output_len1)")

    all_lengths = []
    for col in output_len_columns:
        valid_lengths = pd.to_numeric(df[col], errors='coerce').dropna()
        all_lengths.extend(valid_lengths.tolist())

    if len(all_lengths) == 0:
        raise ValueError("没有在 output_len 列中找到任何有效的数值数据")

    all_lengths = np.array(all_lengths, dtype=float)
    percentiles = [20, 40, 60, 80]
    quantiles = np.percentile(all_lengths, percentiles)
    a, b, c, d = quantiles

    categories = [
        (0, a),
        (a, b),
        (b, c),
        (c, d),
        (d, float('inf'))
    ]

    results = []
    for idx, row in df.iterrows():
        req_id = row['req_id']
        prompt = row['prompt']

        request_lengths = []
        for col in output_len_columns:
            v = row.get(col)
            if pd.isna(v):
                continue
            try:
                val = float(v)
            except (ValueError, TypeError):
                continue
            if np.isfinite(val) and val >= 0:
                request_lengths.append(val)

        if not request_lengths:
            continue

        category_counts = [0] * len(categories)
        for length in request_lengths:
            placed = False
            for cat_idx, (low, high) in enumerate(categories):
                if low <= length < high:
                    category_counts[cat_idx] += 1
                    placed = True
                    break
            if not placed:
                category_counts[-1] += 1

        total_count = len(request_lengths)
        probabilities = [cnt / total_count for cnt in category_counts]

        result_row = {
            'req_id': req_id,
            'prompt': prompt,
            'total_outputs': total_count,
            'category_counts': ','.join(map(str, category_counts)),
            'raw_lengths': ','.join(map(str, request_lengths))
        }
        for i, p in enumerate(probabilities):
            result_row[f'category_{i}_prob'] = float(p)

        results.append(result_row)

    result_df = pd.DataFrame(results)
    for i in range(5):
        col = f'category_{i}_prob'
        if col not in result_df.columns:
            result_df[col] = 0.0
        result_df[col] = result_df[col].astype(float)

    return result_df, (a, b, c, d)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("向量长度必须相同")
    v1 = np.array(vec1, dtype=float).reshape(1, -1)
    v2 = np.array(vec2, dtype=float).reshape(1, -1)
    sim = sk_cosine_similarity(v1, v2)[0, 0]
    return float(max(0.0, min(1.0, sim)))


def process_multiple_csv_files(csv_paths: List[str], output_dir: str) -> Dict[str, pd.DataFrame]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: Dict[str, pd.DataFrame] = {}
    quantiles_info: Dict[str, Tuple[float, float, float, float]] = {}

    for csv_path in csv_paths:
        filename = os.path.basename(csv_path)
        print(f"处理文件: {filename}")
        try:
            dist_df, quantiles = calculate_distribution_from_csv(csv_path)
            results[filename] = dist_df
            quantiles_info[filename] = quantiles

            output_filename = f"{os.path.splitext(filename)[0]}_distribution_{timestamp}.csv"
            output_path = os.path.join(output_dir, output_filename)
            dist_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  保存分布结果到: {output_path}")

            quantile_filename = f"{os.path.splitext(filename)[0]}_quantiles_{timestamp}.txt"
            quantile_path = os.path.join(output_dir, quantile_filename)
            a, b, c, d = quantiles
            with open(quantile_path, 'w', encoding='utf-8') as f:
                f.write("5分位数分界点:\n")
                f.write(f"a (20%分位数): {a:.6f}\n")
                f.write(f"b (40%分位数): {b:.6f}\n")
                f.write(f"c (60%分位数): {c:.6f}\n")
                f.write(f"d (80%分位数): {d:.6f}\n")
            print(f"  保存分位点信息到: {quantile_path}")

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {str(e)}")
            traceback.print_exc()
            raise

    return results


def calculate_cosine_similarities(distributions: Dict[str, pd.DataFrame], output_dir: str) -> Tuple[Dict[Tuple[str, str], List[float]], List[Dict]]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    similarity_results: Dict[Tuple[str, str], List[float]] = {}
    stats_list: List[Dict] = []

    filenames = list(distributions.keys())

    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            file1 = filenames[i]
            file2 = filenames[j]

            print(f"计算 {file1} 和 {file2} 之间的余弦相似度...")

            df1 = distributions[file1].sort_values('req_id')
            df2 = distributions[file2].sort_values('req_id')

            common_req_ids = set(df1['req_id']).intersection(set(df2['req_id']))
            print(f"  共同的req_id数量: {len(common_req_ids)}")

            similarities: List[float] = []
            req_ids: List[str] = []

            for req_id in sorted(common_req_ids):
                row1 = df1[df1['req_id'] == req_id].iloc[0]
                row2 = df2[df2['req_id'] == req_id].iloc[0]

                vec1 = [float(row1[f'category_{k}_prob']) for k in range(5)]
                vec2 = [float(row2[f'category_{k}_prob']) for k in range(5)]

                sim = cosine_similarity(vec1, vec2)
                similarities.append(sim)
                req_ids.append(req_id)

            similarity_df = pd.DataFrame({'req_id': req_ids, 'cosine_similarity': similarities})
            output_filename = f"similarity_{os.path.splitext(file1)[0]}_vs_{os.path.splitext(file2)[0]}_{timestamp}.csv"
            output_path = os.path.join(output_dir, output_filename)
            similarity_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  保存余弦相似度结果到: {output_path}")

            if similarities:
                mean_sim = float(np.mean(similarities))
                std_sim = float(np.std(similarities))
                min_sim = float(np.min(similarities))
                max_sim = float(np.max(similarities))
                count = len(similarities)
            else:
                mean_sim = std_sim = min_sim = max_sim = 0.0
                count = 0

            stats_list.append({
                'file1': file1,
                'file2': file2,
                'mean': mean_sim,
                'std': std_sim,
                'min': min_sim,
                'max': max_sim,
                'count': count
            })

            similarity_results[(file1, file2)] = similarities

    stats_df = pd.DataFrame(stats_list)
    stats_output = os.path.join(output_dir, f"cosine_similarity_stats_{timestamp}.csv")
    stats_df.to_csv(stats_output, index=False, encoding='utf-8-sig')
    print(f"已保存所有对比的统计信息到: {stats_output}")

    return similarity_results, stats_list


def _short_label(filename: str) -> str:
    """
    在文件名任意位置匹配 llama|qwen|deepseek（不区分大小写），
    匹配不到则返回文件名前缀（去掉扩展名）。
    """
    name = os.path.basename(filename)
    m = re.search(r"(llama|qwen|deepseek)", name, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return os.path.splitext(name)[0]


def _find_stat(stats_list: List[Dict], f1: str, f2: str) -> Tuple[float, float]:
    if not stats_list:
        return 0.0, 0.0
    for s in stats_list:
        if (s['file1'] == f1 and s['file2'] == f2) or (s['file1'] == f2 and s['file2'] == f1):
            return float(s.get('mean', 0.0)), float(s.get('std', 0.0))
    return 0.0, 0.0


def plot_cosine_similarities(similarities: Dict[Tuple[str, str], List[float]],
                             output_dir: str,
                             mode: int = 1,
                             stats_list: List[Dict] = None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(7, 4.5), dpi=300)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    color_idx = 0

    # 选择要绘制的对
    pairs_to_plot = []
    if mode == 1:
        pairs_to_plot = list(similarities.keys())
    elif mode == 2:
        # llama vs qwen
        pairs_to_plot = [p for p in similarities.keys() if {'llama', 'qwen'} == { _short_label(p[0]), _short_label(p[1]) }]
    elif mode == 3:
        pairs_to_plot = [p for p in similarities.keys() if {'llama', 'deepseek'} == { _short_label(p[0]), _short_label(p[1]) }]
    elif mode == 4:
        pairs_to_plot = [p for p in similarities.keys() if {'qwen', 'deepseek'} == { _short_label(p[0]), _short_label(p[1]) }]
    else:
        print(f"未知的 mode={mode}，使用 mode=1")
        pairs_to_plot = list(similarities.keys())

    if not pairs_to_plot:
        print("没有找到满足条件的文件对要绘制，退出绘图。")
        return

    all_sims_flat = [sim for sim_list in similarities.values() for sim in sim_list]  # 全集用于 y 轴范围

    for (file1, file2) in pairs_to_plot:
        sim_list = similarities.get((file1, file2), [])
        if not sim_list:
            color_idx += 1
            continue
        x_indices = list(range(1, len(sim_list) + 1))
        color = colors[color_idx % len(colors)]

        mean_val, std_val = _find_stat(stats_list, file1, file2)
        # 文本形式 avg ± std 两位小数
        avgstd_text = f"avg:{mean_val:.2f}±{std_val:.2f}"

        # label 仅在 mode==1 或如果该对是被选中的（mode 2/3/4）显示；其它对在图例中不显示
        if mode == 1:
            label_text = f"{_short_label(file1)} vs {_short_label(file2)} ({avgstd_text})"
        else:
            # 对于 mode 2/3/4 只绘制选中对，label 也可以显示 avgstd
            label_text = f"{_short_label(file1)} vs {_short_label(file2)} ({avgstd_text})"

        # 平滑曲线（主线）
        if len(sim_list) > 10:
            smoothed = gaussian_filter1d(sim_list, sigma=3.0)
        else:
            smoothed = sim_list

        # 绘制主线并在图例中显示（如果 mode==1 则所有对都显示；mode 2/3/4 则仅选中对存在）
        plt.plot(x_indices, smoothed, linewidth=1.6, color=color, label=label_text, alpha=0.95)

        # 若非 mode=1 且你只想小点也一起画：在所有 mode 都绘制散点（但当 mode==1 你可以选择不画散点）
        if mode != 1:
            plt.scatter(x_indices, sim_list, s=10, color=color, alpha=0.6, zorder=3)

        # 在图中用相同颜色的虚线标出均值（不重复添加到图例）
        plt.axhline(y=mean_val, color=color, linestyle='--', linewidth=0.9, alpha=0.8, label="_nolegend_")

        color_idx += 1

    # 坐标与网格
    plt.xlabel('Request index (sorted)', fontsize=12)
    plt.ylabel('cosine similarity', fontsize=12)
    plt.title('Comparison of request distribution cosine similarity', fontsize=12)

    if all_sims_flat:
        min_sim = max(0.0, min(all_sims_flat) - 0.1)
        max_sim = min(1.0, max(all_sims_flat) + 0.1)
        plt.ylim(min_sim, max_sim)

    total_len = max((len(sim) for sim in similarities.values()), default=0)
    if total_len > 20:
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))

    plt.grid(True, alpha=0.35, linestyle='--', linewidth=0.7)
    plt.legend(loc='best', fontsize=9, framealpha=0.9)
    plt.tight_layout(pad=3.0)

    output_filename = f"cosine_similarity_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.5)
    print(f"保存余弦相似度图表到: {output_path}")
    plt.close()


def main(csv_paths: List[str], output_dir: str, mode: int = 1):
    print("开始处理多个CSV文件...")
    print(f"CSV文件列表: {csv_paths}")
    print(f"输出目录: {output_dir}")

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")

    try:
        distributions = process_multiple_csv_files(csv_paths, output_dir)
        similarities, stats_list = calculate_cosine_similarities(distributions, output_dir)

        print('\n余弦相似度统计摘要:')
        for s in stats_list:
            print(f"{s['file1']} vs {s['file2']}: mean={s['mean']:.4f}, std={s['std']:.4f}, n={s['count']}")

        plot_cosine_similarities(similarities, output_dir, mode=mode, stats_list=stats_list)
        print("✅ 所有处理完成!")

    except Exception as e:
        print(f"❌ 处理过程中出错: {str(e)}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='分析多个CSV文件中的输出长度分布并计算相似度')
    parser.add_argument('--csv_paths', type=str, nargs='+', required=True, help='CSV文件路径列表，用空格分隔')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录 (默认: output)')
    parser.add_argument('--mode', type=int, choices=[1, 2, 3, 4], default=1,
                        help='绘图模式：1=全部（显示所有图例并画均值虚线），2=llama vs qwen，3=llama vs deepseek，4=qwen vs deepseek')
    args = parser.parse_args()

    main(args.csv_paths, args.output_dir, mode=args.mode)
