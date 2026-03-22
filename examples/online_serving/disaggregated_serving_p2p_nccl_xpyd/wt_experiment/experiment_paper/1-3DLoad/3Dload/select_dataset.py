# 选择访存负载最大数据集
import os
import pandas as pd

dataset_path = f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/llama_dataset/lmsys-chat/processed_output/llama-lmsys-chat.csv'
save_path = f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/3DLoad/3Dtestdataset/selected_dataset_memory.csv'
target_sample_size = 5000

df = pd.read_csv(dataset_path)
print(df[['prompt_len', 'output_tokens']].describe())
# 从中根据prompt_len和output_tokens指定值随机选择5000条数据，保存到新的csv文件中
filtered_df = df[
	(df['prompt_len'] >= 50)
	& (df['prompt_len'] <= 2000)
	& (df['output_tokens'] >= 200)
	& (df['output_tokens'] <= 2000)
]

available_size = len(filtered_df)
if available_size == 0:
	raise ValueError('筛选条件下没有可用数据，请放宽 prompt_len/output_tokens 范围。')

sample_size = min(target_sample_size, available_size)
if sample_size < target_sample_size:
	print(f'Warning: 筛选后仅有 {available_size} 条数据，已自动采样 {sample_size} 条。')

selected_df = filtered_df.sample(n=sample_size, random_state=42)
selected_df.to_csv(save_path, index=False)
# 输出选择的数据集的prompt_len和output_tokens的统计信息
print("Selected dataset statistics:")
print(selected_df[['prompt_len', 'output_tokens']].describe())
# 选择计算负载最大的数据集
# import os
# import pandas as pd
# dataset_path = f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/llama_dataset/lmsys-chat/processed_output'
# save_path = f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/3DLoad/3Dtestdataset/selected_dataset.csv'
# # 读取目录下所有csv文件，并合并为一个DataFrame
# df_list = []
# for file in os.listdir(dataset_path):
#     if file.endswith('.csv'):
#         df_list.append(pd.read_csv(os.path.join(dataset_path, file)))
# df = pd.concat(df_list, ignore_index=True)
# # 从中根据prompt_len和output_tokens指定值随机选择5000条数据，保存到新的csv文件中
# selected_df = df[(df['prompt_len'] <= 256) & (df['output_tokens'] <= 256)].sample(n=5000, random_state=42)
# selected_df.to_csv(save_path, index=False)
