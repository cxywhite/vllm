import pandas as pd
dataset_path1=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/Loadbalance/dataset/qwen-mysharegpt-loadbalance_1p3d_good_3.csv'
dataset_path2=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/llama_dataset/mysharegpt/processed_output/llama-mysharegpt.csv'
save_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/Loadbalance/dataset/llama-mysharegpt-loadbalance_1p3d.csv'
df1 = pd.read_csv(dataset_path1)
df2 = pd.read_csv(dataset_path2)
# 从df2中提取df1中相同req_id的数据
matching_req_ids = df1['req_id'].unique()
filtered_df2 = df2[df2['req_id'].isin(matching_req_ids)]
# 按照df1的req_id顺序对filtered_df2进行排序
filtered_df2 = filtered_df2.set_index('req_id').loc[df1['req_id']].reset_index()
# 保存filtered_df2
# 打印df1 prompt_len和output_tokens的统计信息
print("df1 prompt_len statistics:")
print(df1[['prompt_len','output_tokens']].describe())
# 打印filtered_df2 prompt_len和output_tokens的统计信息
print("filtered_df2 prompt_len statistics:")
print(filtered_df2[['prompt_len','output_tokens']].describe())
# 打印df1和filtered_df2的前6行数据
print("df1 head:")
print(df1.head(6))
print("filtered_df2 head:")
print(filtered_df2.head(6))
filtered_df2.to_csv(save_path, index=False)