import pandas as pd
import os
dataset_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/qwen_dataset/lmsys-chat/static_len_0/processed_output'

# 按顺序读取该目录下的所有csv文件，并合并成一个DataFrame
all_files = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.csv')])
df_list = [pd.read_csv(file) for file in all_files]
merged_df = pd.concat(df_list, ignore_index=True)
# 将合并后的DataFrame保存为新的csv文件
import pdb
print(merged_df.tail())
# pdb.set_trace()
merged_df.to_csv(os.path.join(dataset_path, 'qwen-lmsys-chat.csv'), index=False)