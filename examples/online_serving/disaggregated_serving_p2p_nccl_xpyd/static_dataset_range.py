# import pandas as pd
# dataset_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/3DLoad/old/3Dtestdataset/selected_dataset_compute.csv'
# # 统计数据集中output_tokens最小值、最大值、平均值、中位数不需要保存
# df=pd.read_csv(dataset_path)
# # 统计数据集中output_tokens最小值、最大值、平均值
# min_output_tokens = df['output_tokens'].min()
# max_output_tokens = df['output_tokens'].max()
# mean_output_tokens = df['output_tokens'].mean()
# median_output_tokens = df['output_tokens'].median()
# print(f"Output tokens statistics:")
# print(f"Min: {min_output_tokens}")
# print(f"Max: {max_output_tokens}")
# print(f"Mean: {mean_output_tokens}")
# print(f"Median: {median_output_tokens}")
import pandas as pd
dataset_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/3DLoad/3Dload/benchmark_np5000_rr100_20260212_160904_1p1d/dataset_result/test_results_20260212_161106.csv'
# save_path根据dataset_path自动生成，去掉最后的文件名，改成total_tokens_statistics.txt
save_path = dataset_path.rsplit('/', 1)[0] + '/total_tokens_statistics.txt'
df=pd.read_csv(dataset_path)
# 统计数据集中prompt_len+output_tokens最小值、最大值、平均值、中位数
df['total_tokens'] = df['prompt_len'] + df['output_tokens']
min_total_tokens = df['total_tokens'].min()
max_total_tokens = df['total_tokens'].max()
mean_total_tokens = df['total_tokens'].mean()
median_total_tokens = df['total_tokens'].median()
print(f"Total tokens (prompt_len + output_tokens) statistics:")
print(f"Min: {min_total_tokens}")
print(f"Max: {max_total_tokens}")
print(f"Mean: {mean_total_tokens}")
print(f"Median: {median_total_tokens}")
# 保存到save_path的txt文件中
with open(save_path, "w") as f:
    f.write(f"Total tokens (prompt_len + output_tokens) statistics:\n")
    f.write(f"Min: {min_total_tokens}\n")
    f.write(f"Max: {max_total_tokens}\n")
    f.write(f"Mean: {mean_total_tokens}\n")
    f.write(f"Median: {median_total_tokens}\n")