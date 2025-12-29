#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_csv_distributions_dynamic.py
（更新版：支持在文件名任意位置识别 llama/qwen/deepseek；mode=1 显示所有图例并在图中画均值虚线）
已修改：不再计算余弦相似度；改为统计相同 req_id 下最大概率类别是否相同（两两比较，可受 mode 控制）
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
# from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity  # 不再需要

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
    # 保留以防将来需要（但当前主流程不再使用）
    raise NotImplementedError("余弦相似度在当前版本中已停用；请使用 argmax 匹配统计。")


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


# ---------- 新增函数：统计 argmax 匹配数 ----------
def calculate_argmax_matches(distributions: Dict[str, pd.DataFrame], output_dir: str, mode: int = 1) -> Tuple[Dict[Tuple[str, str], int], List[Dict]]:
    """
    对 distributions 中的每对文件（或根据 mode 选择的对）进行比较：
    - 统计相同 req_id 下两者的最大概率类别索引是否相同（match）。
    返回 (matches_dict, stats_list)
    matches_dict: key=(file1,file2) -> match_count (int)
    stats_list: 每对的字典包含 file1,file2,match_count,common_count,match_ratio
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filenames = list(distributions.keys())
    matches: Dict[Tuple[str, str], int] = {}
    stats_list: List[Dict] = []

    # 选择需要比较的 pairs（与绘图 mode 对应）
    all_pairs = []
    if mode == 1:
        # 所有两两组合
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                all_pairs.append((filenames[i], filenames[j]))
    else:
        # 选择指定 pair
        def pair_matches_short(fa, fb, s1, s2):
            return { _short_label(fa), _short_label(fb) } == { s1, s2 }

        if mode == 2:
            target = ('llama', 'qwen')
        elif mode == 3:
            target = ('llama', 'deepseek')
        elif mode == 4:
            target = ('qwen', 'deepseek')
        else:
            print(f"未知 mode={mode}，将使用 mode=1（所有两两组合）")
            for i in range(len(filenames)):
                for j in range(i + 1, len(filenames)):
                    all_pairs.append((filenames[i], filenames[j]))
            target = None

        if target is not None:
            s1, s2 = target
            for i in range(len(filenames)):
                for j in range(i + 1, len(filenames)):
                    if { _short_label(filenames[i]), _short_label(filenames[j]) } == { s1, s2 }:
                        all_pairs.append((filenames[i], filenames[j]))

    if not all_pairs:
        print("没有要比较的文件对（可能 mode 筛选后为空）。")
        return matches, stats_list

    for (f1, f2) in all_pairs:
        df1 = distributions[f1].set_index('req_id', drop=False)
        df2 = distributions[f2].set_index('req_id', drop=False)

        common_req_ids = sorted(set(df1.index).intersection(set(df2.index)))
        common_count = len(common_req_ids)
        match_count = 0

        rows_for_output = []
        for req_id in common_req_ids:
            row1 = df1.loc[req_id]
            row2 = df2.loc[req_id]

            # 取 category_0_prob ... category_4_prob 的最大概率索引
            vec1 = [float(row1.get(f'category_{k}_prob', 0.0)) for k in range(5)]
            vec2 = [float(row2.get(f'category_{k}_prob', 0.0)) for k in range(5)]

            # tie-break: np.argmax 会返回第一个最大值索引（即最小索引）
            idx1 = int(np.argmax(vec1))
            idx2 = int(np.argmax(vec2))

            match = 1 if idx1 == idx2 else 0
            match_count += match

            rows_for_output.append({'req_id': req_id, f'{os.path.splitext(f1)[0]}_argmax': idx1,
                                    f'{os.path.splitext(f2)[0]}_argmax': idx2, 'match': match})

        # 保存每对的详细 csv
        out_filename = f"argmax_match_{os.path.splitext(f1)[0]}_vs_{os.path.splitext(f2)[0]}_{timestamp}.csv"
        out_path = os.path.join(output_dir, out_filename)
        if rows_for_output:
            pd.DataFrame(rows_for_output).to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"  已保存对比明细到: {out_path}")
        else:
            # 写空文件表头
            pd.DataFrame(columns=['req_id', f'{os.path.splitext(f1)[0]}_argmax', f'{os.path.splitext(f2)[0]}_argmax', 'match']).to_csv(out_path, index=False, encoding='utf-8-sig')
            print(f"  对比明细（空）已保存: {out_path}")

        ratio = (match_count / common_count) if common_count > 0 else 0.0
        stats_list.append({
            'file1': f1,
            'file2': f2,
            'match_count': int(match_count),
            'common_count': int(common_count),
            'match_ratio': float(ratio)
        })

        matches[(f1, f2)] = int(match_count)
        print(f"Pair: {_short_label(f1)} vs {_short_label(f2)} -> match {match_count}/{common_count} ({ratio:.4f})")

    # 保存汇总统计
    stats_df = pd.DataFrame(stats_list)
    stats_out = os.path.join(output_dir, f"argmax_match_stats_{timestamp}.csv")
    stats_df.to_csv(stats_out, index=False, encoding='utf-8-sig')
    print(f"已保存汇总统计到: {stats_out}")

    return matches, stats_list


def plot_cosine_similarities(similarities: Dict[Tuple[str, str], List[float]],
                             output_dir: str,
                             mode: int = 1,
                             stats_list: List[Dict] = None):
    # 保留原绘图函数（基于余弦相似度），若以后需要可再次激活
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.figure(figsize=(7, 4.5), dpi=300)

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    color_idx = 0

    pairs_to_plot = list(similarities.keys())

    if not pairs_to_plot:
        print("没有找到满足条件的文件对要绘制，退出绘图。")
        return

    all_sims_flat = [sim for sim_list in similarities.values() for sim in sim_list]

    for (file1, file2) in pairs_to_plot:
        sim_list = similarities.get((file1, file2), [])
        if not sim_list:
            color_idx += 1
            continue
        x_indices = list(range(1, len(sim_list) + 1))
        color = colors[color_idx % len(colors)]

        mean_val, std_val = _find_stat(stats_list or [], file1, file2)
        avgstd_text = f"avg:{mean_val:.2f}±{std_val:.2f}"

        if len(sim_list) > 10:
            smoothed = gaussian_filter1d(sim_list, sigma=3.0)
        else:
            smoothed = sim_list

        plt.plot(x_indices, smoothed, linewidth=1.6, color=color, label=f"{_short_label(file1)} vs {_short_label(file2)} ({avgstd_text})", alpha=0.95)

        plt.scatter(x_indices, sim_list, s=10, color=color, alpha=0.6, zorder=3)
        plt.axhline(y=mean_val, color=color, linestyle='--', linewidth=0.9, alpha=0.8, label="_nolegend_")
        color_idx += 1

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

    output_filename = f"cosine_similarity_comparison_{timestamp}.png"
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

        # ---------- 改为计算 argmax 匹配，而不是余弦相似度 ----------
        matches, stats_list = calculate_argmax_matches(distributions, output_dir, mode=mode)

        print('\nargmax 匹配统计摘要:')
        for s in stats_list:
            ratio_pct = s['match_ratio'] * 100.0
            print(f"{_short_label(s['file1'])} vs {_short_label(s['file2'])}: match={s['match_count']}/{s['common_count']} ({ratio_pct:.2f}%)")

        # 如果后续仍需绘图或其它统计，这里可继续调用
        print("✅ 所有处理完成!")

    except Exception as e:
        print(f"❌ 处理过程中出错: {str(e)}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='分析多个CSV文件中的输出长度分布并统计最大概率类别匹配')
    parser.add_argument('--csv_paths', type=str, nargs='+', required=True, help='CSV文件路径列表，用空格分隔')
    parser.add_argument('--output_dir', type=str, default='output', help='输出目录 (默认: output)')
    parser.add_argument('--mode', type=int, choices=[1, 2, 3, 4], default=1,
                        help='统计模式：1=全部两两比较，2=llama vs qwen，3=llama vs deepseek，4=qwen vs deepseek')
    args = parser.parse_args()

    main(args.csv_paths, args.output_dir, mode=args.mode)
