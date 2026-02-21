import pandas as pd
import os
dataset_path1=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/qwen_dataset/lmsys-chat/processed_output/qwen-lmsys-chat.csv'
dataset_path2=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/qwen_dataset/lmsys-chat/static_len_0/processed_output/qwen-lmsys-chat.csv'
# 读取两个csv文件
df1 = pd.read_csv(dataset_path1)
df2 = pd.read_csv(dataset_path2)
# 用df2中与df1相同req_id的行替换df1中的对应行
df1.set_index('req_id', inplace=True)
df2.set_index('req_id', inplace=True)
df1.update(df2)
# 将更新后的DataFrame保存为新的csv文件
output_path = f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/qwen_dataset/lmsys-chat/processed_output/qwen-lmsys-chat-updated.csv'
df1.reset_index(inplace=True)
df1.to_csv(output_path, index=False)