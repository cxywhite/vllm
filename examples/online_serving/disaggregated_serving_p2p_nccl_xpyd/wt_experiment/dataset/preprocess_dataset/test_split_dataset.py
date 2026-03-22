import pandas as pd
import os
# 读取csv文件
df=pd.read_csv('/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/qwen_dataset/mysharegpt/static_len_0/static_len_0.csv')
# 每1000条数据分割成一个新的csv文件，用索引命名
num_rows_per_file=1000
os.makedirs('/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/qwen_dataset/mysharegpt/static_len_0/split',exist_ok=True)
num_files=(len(df) + num_rows_per_file - 1) // num_rows_per_file
for i in range(num_files):
    start_index=i * num_rows_per_file
    end_index=min((i + 1) * num_rows_per_file, len(df))
    df_subset=df.iloc[start_index:end_index]
    df_subset.to_csv(f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/qwen_dataset/mysharegpt/static_len_0/split/mysharegpt_static_len_0_part_{i}.csv',index=False)