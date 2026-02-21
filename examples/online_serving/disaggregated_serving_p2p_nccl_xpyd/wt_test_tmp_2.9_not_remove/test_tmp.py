import torch
import numpy as np
from datasets import load_from_disk, Dataset

# 固定随机种子（numpy 和 torch 都设置）
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 加载你的数据集（替换成你的实际路径）
dataset = load_from_disk("/root/myshare/predict_project/act_predictor_train/qwen_dataset/qwen_4_sample_range_50_lens_compute_means_and_probabilities_20250716_131530")

def add_selected_len(example):
    # example['lens'] 是 torch.Tensor，shape=(50,)
    lens_tensor = example['lens']
    
    # 方法1：用 torch 随机选（推荐，与深度学习生态更一致）
    idx = torch.randint(0, 50, (1,)).item()          # 随机整数 0~49
    selected = lens_tensor[idx].item()               # 取出标量并转成 python int/float
    
    # 方法2：用 numpy（也可以，但需要先转成 numpy）
    # lens_np = lens_tensor.numpy()
    # selected = np.random.choice(lens_np)
    
    example['ares_len'] = selected
    return example

# 应用转换
# 如果数据集很大，建议加上 batched=True + batch_size 来加速
new_dataset= dataset.add_column('ares_len', [0]*len(dataset))  # 预先创建列以避免警告
new_dataset = new_dataset.map(
    add_selected_len,
    num_proc=4,           # 可选：多进程加速（根据你的机器核心数调整）
    desc="Adding random selected length"
)

# 保存到指定目录
new_dataset.save_to_disk("/root/myshare/predict_project/act_predictor_train/qwen_dataset/ares_qwen_4_sample")

print("已完成：添加 selected_len 列，并保存到磁盘")
print(f"示例数据查看：\n{new_dataset[0]}")

# def compute_bucket_averages(dataset_path):
#     """
#     计算基于分位数划分的各个bucket的平均长度
    
#     Args:
#         dataset_path: Hugging Face dataset 路径
        
#     Returns:
#         各个bucket的平均长度
#     """
#     # 加载数据集
#     dataset = load_from_disk(dataset_path)
#     print(f"数据集总样本数: {len(dataset)}")
    
#     # 给定的分位数边界
#     bucket_boundaries = [0, 281.00, 704.00, 26232.60, 32730.00, 32768]
#     bucket_names = [
#         '[0, 281]',
#         '[281, 704]', 
#         '[704, 26232.60]',
#         '[26232.60, 32730.00]',
#         '[32730.00, 32768]'
#     ]
    
#     # 初始化bucket容器
#     buckets = [[] for _ in range(len(bucket_boundaries) - 1)]
    
#     # 收集所有lens值并分配到对应的bucket
#     total_values = 0
#     for example in tqdm(dataset, desc="处理数据集"):
#         lens_tensor = example['lens']
        
#         # 确保是tensor格式并转换为numpy
#         if isinstance(lens_tensor, torch.Tensor):
#             lens_values = lens_tensor.cpu().numpy()
#         elif isinstance(lens_tensor, np.ndarray):
#             lens_values = lens_tensor
#         else:
#             lens_values = np.array(lens_tensor)
        
#         # 将每个值分配到对应的bucket
#         for value in lens_values.flatten():
#             total_values += 1
#             for i in range(len(buckets)):
#                 if bucket_boundaries[i] <= value < bucket_boundaries[i + 1]:
#                     buckets[i].append(value)
#                     break
    
#     print(f"\n总共处理了 {total_values} 个lens值")
    
#     # 计算每个bucket的平均值
#     results = {}
#     bucket_counts = []
    
#     for i, (bucket, name) in enumerate(zip(buckets, bucket_names)):
#         if len(bucket) > 0:
#             avg_value = np.mean(bucket)
#             count = len(bucket)
#             percentage = (count / total_values) * 100
#             results[name] = {
#                 'average': avg_value,
#                 'count': count,
#                 'percentage': percentage
#             }
#             bucket_counts.append(count)
#             print(f"Bucket {name}:")
#             print(f"  平均长度: {avg_value:.2f}")
#             print(f"  样本数量: {count} ({percentage:.2f}%)")
#         else:
#             results[name] = {
#                 'average': 0,
#                 'count': 0,
#                 'percentage': 0
#             }
#             print(f"Bucket {name}: 无数据")
    
#     # 验证数据完整性
#     total_bucket_count = sum(bucket_counts)
#     if total_bucket_count != total_values:
#         print(f"\n警告: bucket总数 ({total_bucket_count}) != 总样本数 ({total_values})")
#         missing_count = total_values - total_bucket_count
#         print(f"  未分配的样本数: {missing_count}")
    
#     return results

# # 使用示例
# if __name__ == "__main__":
#     # 替换为你的实际数据集路径
#     dataset_path = "/root/myshare/predict_project/act_predictor_train/qwen_dataset/qwen_4_sample_range_50_lens_compute_means_and_probabilities_20250716_131530"
    
#     try:
#         bucket_averages = compute_bucket_averages(dataset_path)
        
#         # 打印最终结果
#         print("\n" + "="*50)
#         print("最终结果 - 各bucket平均长度:")
#         print("="*50)
        
#         for bucket_name, stats in bucket_averages.items():
#             if stats['count'] > 0:
#                 print(f"{bucket_name}: {stats['average']:.2f} (n={stats['count']}, {stats['percentage']:.2f}%)")
        
#     except Exception as e:
#         print(f"处理数据集时出错: {str(e)}")

# from datasets import load_from_disk, Dataset
# import numpy as np
# import torch

# def compute_lens_percentiles(dataset_path):
#     """
#     从Hugging Face dataset中提取lens列并计算分位数
    
#     Args:
#         dataset_path: 数据集路径
        
#     Returns:
#         包含20%, 40%, 60%, 80%分位数的字典
#     """
#     # 加载数据集
#     dataset = load_from_disk(dataset_path)
    
#     print(f"数据集总样本数: {len(dataset)}")
    
#     # 提取所有lens数据
#     all_lens_values = []
    
#     # 遍历数据集，收集所有lens值
#     for example in dataset:
#         lens_tensor = example['lens']
        
#         # 确保是tensor格式
#         if isinstance(lens_tensor, torch.Tensor):
#             lens_values = lens_tensor.cpu().numpy()
#         elif isinstance(lens_tensor, np.ndarray):
#             lens_values = lens_tensor
#         else:
#             # 如果是其他格式，尝试转换
#             lens_values = np.array(lens_tensor)
        
#         # 将当前样本的lens值添加到总列表中
#         all_lens_values.extend(lens_values.flatten().tolist())
    
#     print(f"总共收集到 {len(all_lens_values)} 个lens值")
    
#     # 转换为numpy数组以便计算分位数
#     all_lens_array = np.array(all_lens_values)
    
#     # 计算分位数
#     percentiles = [20, 40, 60, 80]
#     results = {}
    
#     for p in percentiles:
#         percentile_value = np.percentile(all_lens_array, p)
#         results[f"{p}%"] = percentile_value
#         print(f"{p}% 分位数: {percentile_value:.2f}")
    
#     # 也可以计算一些额外的统计信息
#     print("\n额外统计信息:")
#     print(f"最小值: {np.min(all_lens_array):.2f}")
#     print(f"最大值: {np.max(all_lens_array):.2f}")
#     print(f"平均值: {np.mean(all_lens_array):.2f}")
#     print(f"中位数 (50%): {np.median(all_lens_array):.2f}")
    
#     return results

# # 使用示例
# if __name__ == "__main__":
#     # 替换为你的数据集路径
#     dataset_path = "/root/myshare/predict_project/act_predictor_train/qwen_dataset/qwen_4_sample_range_50_lens_compute_means_and_probabilities_20250716_131530"
    
#     try:
#         percentiles = compute_lens_percentiles(dataset_path)
        
#         # 打印结果
#         print("\n最终结果:")
#         for percentile, value in percentiles.items():
#             print(f"{percentile} 分位数: {value:.2f}")
            
#     except Exception as e:
#         print(f"处理数据集时出错: {str(e)}")