import datasets
import transformers
from transformers import AutoTokenizer,AutoConfig
import random
import os
from datasets import load_from_disk
import pandas as pd
df=pd.read_csv("/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/qwen_baseline_127/benchmark_np500_rr16_mt32768_20260127_185637_2p4d/dataset_result/results_20260127_194544.csv")
# 按照id排序
df_sorted=df.sort_values(by="req_id")
df1=pd.read_csv("/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/qwen_dataset/sharegpt_preprocessed_prompts.csv")
df1_sorted=df1.sort_values(by="id")
# 取出df_sorted的output_tokens列，添加到df1_sorted中
df1_sorted["output_tokens"]=df_sorted["output_len1"].values
# 各自打印前5行进行对比
print("First 5 rows of df1_sorted:")
print(df1_sorted.head())
print("First 5 rows of df_sorted:")
print(df_sorted.head())
output_csv_path="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/qwen_dataset/sharegpt_preprocessed_prompts_with_output_tokens.csv"
df1_sorted.to_csv(output_csv_path,index=False)
print(f"Saved merged dataset to {output_csv_path}")