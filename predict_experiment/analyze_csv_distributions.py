# 指定输出目录
# python analyze_csv_distributions.py --csv_paths file1.csv file2.csv --output_dir results
import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import gaussian_filter1d
import argparse
from typing import List, Dict, Tuple
import math
import traceback

def calculate_distribution_from_csv(csv_path: str) -> Tuple[pd.DataFrame, Tuple[float, float, float, float]]:
    """
    从CSV文件计算每个请求的输出长度分布
    
    参数:
        csv_path (str): CSV文件路径
    
    返回:
        Tuple[pd.DataFrame, Tuple[float, float, float, float]]: 
        - 包含每个请求分布的DataFrame
        - 5分位数分界点 (a, b, c, d)
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 检查必要的列是否存在
    output_len_columns = [f'output_len{i}' for i in range(1, 11)]
    required_columns = ['req_id', 'prompt'] + output_len_columns
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必需的列: {col}")
    
    # 提取所有10个输出长度的值
    all_lengths = []
    for col in output_len_columns:
        # 过滤掉NaN和非数值数据
        valid_lengths = df[col].dropna()
        # 确保是数值类型
        valid_lengths = pd.to_numeric(valid_lengths, errors='coerce').dropna()
        all_lengths.extend(valid_lengths.tolist())
    
    # 计算5分位数（4个分位点）
    all_lengths = np.array(all_lengths, dtype=float)
    percentiles = [20, 40, 60, 80]
    quantiles = np.percentile(all_lengths, percentiles)
    a, b, c, d = quantiles
    
    # 定义类别区间
    categories = [
        (0, a),      # 类别0: (0, a)
        (a, b),      # 类别1: (a, b)
        (b, c),      # 类别2: (b, c)
        (c, d),      # 类别3: (c, d)
        (d, float('inf'))  # 类别4: (d, +∞)
    ]
    
    # 为每个请求计算概率分布
    results = []
    
    for idx, row in df.iterrows():
        # 获取当前请求的10个输出长度
        request_lengths = []
        for col in output_len_columns:
            length = row[col]
            if pd.notna(length):
                try:
                    length_val = float(length)
                    if np.isfinite(length_val) and length_val >= 0:
                        request_lengths.append(length_val)
                except (ValueError, TypeError):
                    continue
        
        # 如果没有有效的长度数据，跳过
        if not request_lengths:
            continue
        
        # 统计每个类别中的数量
        category_counts = [0, 0, 0, 0, 0]
        
        for length in request_lengths:
            for cat_idx, (low, high) in enumerate(categories):
                if low <= length < high:
                    category_counts[cat_idx] += 1
                    break
                # 处理边界情况，特别是最后一个区间
                if cat_idx == len(categories) - 1 and length >= low:
                    category_counts[cat_idx] += 1
                    break
        
        # 计算概率分布
        total_count = len(request_lengths)
        if total_count > 0:
            probabilities = [count / total_count for count in category_counts]
        else:
            probabilities = [0.2] * 5  # 如果没有数据，均匀分布
        
        # 保存结果
        result_row = {
            'req_id': row['req_id'],
            'prompt': row['prompt'],
            'total_outputs': total_count,
            'category_0_prob': probabilities[0],
            'category_1_prob': probabilities[1],
            'category_2_prob': probabilities[2],
            'category_3_prob': probabilities[3],
            'category_4_prob': probabilities[4],
            'category_counts': ','.join(map(str, category_counts)),
            'raw_lengths': ','.join(map(str, request_lengths))
        }
        results.append(result_row)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(results)
    
    return result_df, (a, b, c, d)

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """
    计算两个向量的余弦相似度
    
    参数:
        vec1 (List[float]): 第一个向量
        vec2 (List[float]): 第二个向量
    
    返回:
        float: 余弦相似度 (0-1之间)
    """
    if len(vec1) != len(vec2):
        raise ValueError("向量长度必须相同")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    # 确保结果在0-1之间
    return max(0.0, min(1.0, similarity))

def process_multiple_csv_files(csv_paths: List[str], output_dir: str) -> Dict[str, pd.DataFrame]:
    """
    处理多个CSV文件，计算每个文件的分布
    
    参数:
        csv_paths (List[str]): CSV文件路径列表
        output_dir (str): 输出目录
    
    返回:
        Dict[str, pd.DataFrame]: 文件名到分布DataFrame的映射
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}
    quantiles_info = {}
    
    for csv_path in csv_paths:
        filename = os.path.basename(csv_path)
        print(f"处理文件: {filename}")
        
        try:
            # 计算分布
            dist_df, quantiles = calculate_distribution_from_csv(csv_path)
            results[filename] = dist_df
            quantiles_info[filename] = quantiles
            
            # 保存结果
            output_filename = f"{os.path.splitext(filename)[0]}_distribution_{timestamp}.csv"
            output_path = os.path.join(output_dir, output_filename)
            dist_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  保存分布结果到: {output_path}")
            
            # 保存分位点信息
            quantile_filename = f"{os.path.splitext(filename)[0]}_quantiles_{timestamp}.txt"
            quantile_path = os.path.join(output_dir, quantile_filename)
            a, b, c, d = quantiles
            with open(quantile_path, 'w', encoding='utf-8') as f:
                f.write(f"5分位数分界点:\n")
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

def calculate_cosine_similarities(distributions: Dict[str, pd.DataFrame], 
                                 output_dir: str) -> Dict[Tuple[str, str], List[float]]:
    """
    计算不同CSV文件之间相同req_id的分布的余弦相似度
    
    参数:
        distributions (Dict[str, pd.DataFrame]): 文件名到分布DataFrame的映射
        output_dir (str): 输出目录
    
    返回:
        Dict[Tuple[str, str], List[float]]: (file1, file2)到余弦相似度列表的映射
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    similarity_results = {}
    
    # 获取所有文件名
    filenames = list(distributions.keys())
    
    # 两两比较
    for i in range(len(filenames)):
        for j in range(i + 1, len(filenames)):
            file1 = filenames[i]
            file2 = filenames[j]
            
            print(f"计算 {file1} 和 {file2} 之间的余弦相似度...")
            
            df1 = distributions[file1]
            df2 = distributions[file2]
            
            # 按req_id排序
            df1 = df1.sort_values('req_id')
            df2 = df2.sort_values('req_id')
            
            # 确保req_id对齐
            common_req_ids = set(df1['req_id']).intersection(set(df2['req_id']))
            print(f"  共同的req_id数量: {len(common_req_ids)}")
            
            similarities = []
            req_ids = []
            
            for req_id in sorted(common_req_ids):
                row1 = df1[df1['req_id'] == req_id].iloc[0]
                row2 = df2[df2['req_id'] == req_id].iloc[0]
                
                # 获取概率分布向量
                vec1 = [
                    row1['category_0_prob'],
                    row1['category_1_prob'],
                    row1['category_2_prob'],
                    row1['category_3_prob'],
                    row1['category_4_prob']
                ]
                
                vec2 = [
                    row2['category_0_prob'],
                    row2['category_1_prob'],
                    row2['category_2_prob'],
                    row2['category_3_prob'],
                    row2['category_4_prob']
                ]
                
                # 计算余弦相似度
                similarity = cosine_similarity(vec1, vec2)
                similarities.append(similarity)
                req_ids.append(req_id)
            
            # 保存相似度结果
            similarity_df = pd.DataFrame({
                'req_id': req_ids,
                'cosine_similarity': similarities
            })
            
            output_filename = f"similarity_{os.path.splitext(file1)[0]}_vs_{os.path.splitext(file2)[0]}_{timestamp}.csv"
            output_path = os.path.join(output_dir, output_filename)
            similarity_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"  保存余弦相似度结果到: {output_path}")
            
            similarity_results[(file1, file2)] = similarities
    
    return similarity_results

def plot_cosine_similarities(similarities: Dict[Tuple[str, str], List[float]], 
                           output_dir: str):
    """
    绘制余弦相似度曲线图
    
    参数:
        similarities (Dict[Tuple[str, str], List[float]]): (file1, file2)到相似度列表的映射
        output_dir (str): 输出目录
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.figure(figsize=(5, 3), dpi=300)  # 增大图像尺寸和分辨率
    
    # 定义颜色
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    color_idx = 0
    import re  # 需要导入re模块

    # 在循环开始前定义正则表达式模式
    pattern = r'(llama|qwen|deepseek)'
    # 为每对文件绘制曲线
    for (file1, file2), sim_list in similarities.items():
        # 生成索引ID (1, 2, 3, ...)
        x_indices = list(range(1, len(sim_list) + 1))
        
        # 使用高斯滤波进行平滑处理，避免边界骤变
        if len(sim_list) > 10:
            smoothed_sim = gaussian_filter1d(sim_list, sigma=3.0)  # 增大sigma值使曲线更平滑
        else:
            smoothed_sim = sim_list
        # 使用正则匹配llama,qwen,deepseek三个关键词作为label


        plt.plot(x_indices, smoothed_sim, 
                linewidth=1.2,  # 增加线宽
                color=colors[color_idx % len(colors)],
                label=f"{re.search(pattern, file1, re.IGNORECASE).group(1) if re.search(pattern, file1, re.IGNORECASE) else os.path.splitext(file1)[0]} vs {re.search(pattern, file2, re.IGNORECASE).group(1) if re.search(pattern, file2, re.IGNORECASE) else os.path.splitext(file2)[0]}",
                alpha=0.95)
        
        color_idx += 1
    
    plt.xlabel('Request_id', fontsize=12)
    plt.ylabel('cosine similarity', fontsize=12)
    plt.title('Comparison of request distribution cosine similarity', fontsize=12)
    
    # 设置y轴范围，确保不会骤降
    all_similarities = [sim for sim_list in similarities.values() for sim in sim_list]
    if all_similarities:
        min_sim = max(0.0, min(all_similarities) - 0.1)
        max_sim = min(1.0, max(all_similarities) + 0.1)
        plt.ylim(min_sim, max_sim)
    
    # 设置x轴刻度，避免过多刻度
    if len(x_indices) > 20:
        plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    
    # 添加网格线
    plt.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    
    # 设置图例位置
    plt.legend(loc='best', fontsize=4, framealpha=0.9)
    
    # 调整布局
    plt.tight_layout(pad=3.0)
    
    # 保存图像
    output_filename = f"cosine_similarity_comparison_{timestamp}.png"
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.5)  # 高分辨率保存
    print(f"保存余弦相似度图表到: {output_path}")
    
    plt.close()
# def plot_cosine_similarities(similarities: Dict[Tuple[str, str], List[float]], 
#                            output_dir: str):
#     """
#     绘制余弦相似度曲线图（包含散点）并计算平均余弦相似度和标准差
    
#     参数:
#         similarities (Dict[Tuple[str, str], List[float]]): (file1, file2)到相似度列表的映射
#         output_dir (str): 输出目录
#     """
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     plt.figure(figsize=(5, 3), dpi=300)
    
#     # 定义颜色
#     colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
#     color_idx = 0
    
#     # 存储统计结果
#     stats_results = []
    
#     # 为每对文件绘制曲线和散点
#     for (file1, file2), sim_list in similarities.items():
#         # 生成索引ID (1, 2, 3, ...)
#         x_indices = list(range(1, len(sim_list) + 1))
        
#         # 计算统计指标
#         avg_similarity = np.mean(sim_list)
#         std_similarity = np.std(sim_list)
#         min_similarity = np.min(sim_list)
#         max_similarity = np.max(sim_list)
        
#         stats_results.append({
#             'file1': file1,
#             'file2': file2,
#             'avg_cosine_similarity': avg_similarity,
#             'std_cosine_similarity': std_similarity,
#             'min_cosine_similarity': min_similarity,
#             'max_cosine_similarity': max_similarity,
#             'sample_count': len(sim_list)
#         })
        
#         print(f"  {file1} vs {file2}:")
#         print(f"    平均余弦相似度: {avg_similarity:.4f}")
#         print(f"    标准差: {std_similarity:.4f}")
#         print(f"    范围: [{min_similarity:.4f}, {max_similarity:.4f}]")
        
#         # 使用高斯滤波进行平滑处理，避免边界骤变
#         if len(sim_list) > 10:
#             smoothed_sim = gaussian_filter1d(sim_list, sigma=3.0)
#         else:
#             smoothed_sim = sim_list
        
#         # 首先绘制散点图
#         plt.scatter(x_indices, sim_list, 
#                    s=8,
#                    color=colors[color_idx % len(colors)],
#                    alpha=0.6,
#                    marker='o',
#                    edgecolors='none',
#                    zorder=2)
        
#         # 然后绘制平滑曲线
#         line, = plt.plot(x_indices, smoothed_sim, 
#                 linewidth=1.2,
#                 color=colors[color_idx % len(colors)],
#                 alpha=0.95,
#                 zorder=3)
        
#         # 添加标准差阴影区域
#         if len(sim_list) > 1:
#             plt.fill_between(x_indices, 
#                            np.maximum(0, smoothed_sim - std_similarity),
#                            np.minimum(1, smoothed_sim + std_similarity),
#                            color=colors[color_idx % len(colors)],
#                            alpha=0.2,
#                            zorder=1)
        
#         # 绘制平均线（虚线）
#         plt.axhline(y=avg_similarity, 
#                    color=colors[color_idx % len(colors)],
#                    linestyle='--',
#                    alpha=0.7,
#                    linewidth=1.0,
#                    zorder=1,
#                    label=f'{os.path.splitext(file1)[0]} vs {os.path.splitext(file2)[0]}\n(avg: {avg_similarity:.4f} ± {std_similarity:.4f})')
        
#         color_idx += 1
    
#     plt.xlabel('Request_id', fontsize=12)
#     plt.ylabel('cosine similarity', fontsize=12)
#     plt.title('Comparison of request distribution cosine similarity', fontsize=12)
    
#     # 设置y轴范围，确保不会骤降
#     all_similarities = [sim for sim_list in similarities.values() for sim in sim_list]
#     if all_similarities:
#         min_sim = max(0.0, min(all_similarities) - 0.1)
#         max_sim = min(1.0, max(all_similarities) + 0.1)
#         plt.ylim(min_sim, max_sim)
    
#     # 设置x轴刻度，避免过多刻度
#     if len(x_indices) > 20:
#         plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    
#     # 添加网格线
#     plt.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
    
#     # 设置图例位置
#     plt.legend(loc='best', fontsize=4, framealpha=0.9)
    
#     # 调整布局
#     plt.tight_layout(pad=3.0)
    
#     # 保存图像
#     output_filename = f"cosine_similarity_comparison_{timestamp}.png"
#     output_path = os.path.join(output_dir, output_filename)
#     plt.savefig(output_path, dpi=600, bbox_inches='tight', pad_inches=0.5)
#     print(f"保存余弦相似度图表到: {output_path}")
    
#     # 保存统计结果到CSV文件
#     stats_df = pd.DataFrame(stats_results)
#     stats_filename = f"cosine_similarity_statistics_{timestamp}.csv"
#     stats_path = os.path.join(output_dir, stats_filename)
#     stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
#     print(f"保存余弦相似度统计结果到: {stats_path}")
    
#     # 打印总体统计信息
#     print(f"\n{'='*50}")
#     print("总体统计摘要:")
#     print(f"{'='*50}")
    
#     overall_avg = np.mean([res['avg_cosine_similarity'] for res in stats_results])
#     overall_std = np.mean([res['std_cosine_similarity'] for res in stats_results])
    
#     print(f"总体平均余弦相似度: {overall_avg:.4f}")
#     print(f"平均标准差: {overall_std:.4f}")
    
#     # 按平均相似度排序
#     sorted_stats = sorted(stats_results, key=lambda x: x['avg_cosine_similarity'], reverse=True)
#     print(f"\n按平均相似度排序:")
#     for i, res in enumerate(sorted_stats, 1):
#         print(f"{i}. {res['file1']} vs {res['file2']}:")
#         print(f"   平均: {res['avg_cosine_similarity']:.4f}, 标准差: {res['std_cosine_similarity']:.4f}")
#         print(f"   范围: [{res['min_cosine_similarity']:.4f}, {res['max_cosine_similarity']:.4f}]")
    
#     plt.close()

def main(csv_paths: List[str], output_dir: str):
    """
    主函数：处理多个CSV文件，计算分布，计算相似度，绘制图表
    
    参数:
        csv_paths (List[str]): CSV文件路径列表
        output_dir (str): 输出目录
    """
    print("开始处理多个CSV文件...")
    print(f"CSV文件列表: {csv_paths}")
    print(f"输出目录: {output_dir}")
    
    # 验证文件是否存在
    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    try:
        # 1. 计算每个CSV文件的分布
        distributions = process_multiple_csv_files(csv_paths, output_dir)
        
        # 2. 计算余弦相似度
        similarities = calculate_cosine_similarities(distributions, output_dir)
        
        # 3. 绘制图表
        plot_cosine_similarities(similarities, output_dir)
        
        print("✅ 所有处理完成!")
        
    except Exception as e:
        print(f"❌ 处理过程中出错: {str(e)}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='分析多个CSV文件中的输出长度分布并计算相似度')
    parser.add_argument('--csv_paths', type=str, nargs='+', required=True, 
                        help='CSV文件路径列表，用空格分隔')
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='输出目录 (默认: output)')
    
    args = parser.parse_args()
    
    main(args.csv_paths, args.output_dir)