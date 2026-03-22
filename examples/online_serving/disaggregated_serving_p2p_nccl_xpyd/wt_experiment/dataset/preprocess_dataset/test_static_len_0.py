import pandas as pd
import os
import re
dataset_path1 = "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/qwen_dataset/lmsys-chat/split"
dataset_path2 = "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/qwen_dataset/lmsys-chat/static_len_0/static_len_0.csv"
save_path1=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/qwen_dataset/lmsys-chat/static_len_0/split'
os.makedirs(save_path1, exist_ok=True)
# req_id,prompt,prompt_len,output_tokens,temperature,top_p,top_k,repetition_penalty,ttft,itl,latency
# 读取指定路径下所有csv文件mysharegpt_preprocessed_prompts_part_0.csv，按照part后面的数字顺序读取
all_files = [f for f in os.listdir(dataset_path1) if f.endswith(".csv")]

def extract_part_index(filename: str) -> int:
    match = re.search(r"part_(\d+)", filename)
    if match:
        return int(match.group(1))
    return -1

all_files.sort(key=extract_part_index)  # 按照part后面的数字顺序排序
# 将所有csv文件合并成一个DataFrame
df_list = []
for file in all_files:
    file_path = os.path.join(dataset_path1, file)
    df = pd.read_csv(file_path)
    df_list.append(df)
df1 = pd.concat(df_list, ignore_index=True)
df2 = pd.read_csv(dataset_path2)
# 根据df2中的id从df1中筛选出对应的行，保存到新的csv文件中
filtered_df = df1[df1['id'].isin(df2['id'])]
# 每1000条数据分割成一个新的csv文件，用索引命名
num_rows_per_file = 1000
num_files = (len(filtered_df) + num_rows_per_file - 1) // num_rows_per_file
for i in range(num_files):
    start_index = i * num_rows_per_file
    end_index = min((i + 1) * num_rows_per_file, len(filtered_df))
    df_subset = filtered_df.iloc[start_index:end_index]
    df_subset.to_csv(f'{save_path1}/lmsys-chat_static_len_0_part_{i}.csv', index=False)