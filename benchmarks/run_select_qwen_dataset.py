import datasets
import torch # 如果dataset格式是torch，需要用到，虽然.item()是通用的

# 1. 加载数据集
# 假设 dataset 的路径是 path1, dataset1 的路径是 path2
path1 = '/root/myshare/predict_project/act_predictor_train/llama_dataset/llama_4_sample_range_50_lens_compute_means_and_probabilities_20250603_121645'
path2 = '/root/myshare/predict_project/act_predictor_train/qwen_dataset/qwen_4_sample_range_50_lens_compute_means_and_probabilities_20250716_131530'
save_path = '/root/myshare/predict_project/act_predictor_train/qwen_dataset/qwen_remain_dataset' # 请修改为你想要保存的路径

ds = datasets.load_from_disk(path1)
ds1 = datasets.load_from_disk(path2)

print(f"原始 Dataset 总行数: {len(ds)}")
print(f"Dataset1 (子集) 总行数: {len(ds1)}")

# 2. 提取 dataset1 中的所有 ID 并转换为 Python 整数存入集合
# 注意：这里假设 load_from_disk 后，列数据已经是 Tensor 或者可以通过 item() 转换
# 使用 set 集合可以极大提高后续的查找速度
ids_to_exclude = set()

# 遍历 ds1 获取需要排除的 ID
# 这一步会自动处理 dataset1 中每个 ID 重复出现4次的情况，set会自动去重
for item in ds1['new_ids']:
    # 如果 item 是 tensor，使用 .item() 转为 python int
    # 如果 item 已经是 int，直接使用
    if hasattr(item, 'item'):
        ids_to_exclude.add(item.item())
    else:
        ids_to_exclude.add(int(item))

print(f"需要排除的唯一 ID 数量: {len(ids_to_exclude)}")

# 3. 定义过滤函数
def filter_not_in_subset(example):
    # 获取当前行的 id
    current_id = example['id']
    
    # 同样将其转换为 int
    if hasattr(current_id, 'item'):
        current_id_val = current_id.item()
    else:
        current_id_val = int(current_id)
        
    # 如果 id 不在排除列表中，则保留（返回 True）
    return current_id_val not in ids_to_exclude

# 4. 执行过滤生成 dataset2
dataset2 = ds.filter(filter_not_in_subset)

# 5. 验证数据量
# 理论上 dataset2 数量应为: 10400 - 5848 = 4552
print(f"生成的 Dataset2 总行数: {len(dataset2)}")

if len(dataset2) == (len(ds) - len(ds1)):
    print("数据校验成功！数量符合预期。")
else:
    print("注意：数据数量与简单的减法不一致，请检查是否存在ID跨数据集不匹配的情况。")

# 6. 保存新的数据集
dataset2.save_to_disk(save_path)
print(f"Dataset2 已保存至: {save_path}")