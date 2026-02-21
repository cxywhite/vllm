import os
import pandas as pd
dataset_path = f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/llama_dataset/lmsys-chat/processed_output'
save_path = f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/3DLoad/3Dtestdataset/selected_dataset.csv'
# 读取目录下所有csv文件，并合并为一个DataFrame
df_list = []
for file in os.listdir(dataset_path):
    if file.endswith('.csv'):
        df_list.append(pd.read_csv(os.path.join(dataset_path, file)))
df = pd.concat(df_list, ignore_index=True)
# 从中根据prompt_len和output_tokens指定值随机选择5000条数据，保存到新的csv文件中
selected_df = df[(df['prompt_len'] <= 256) & (df['output_tokens'] <= 256)].sample(n=5000, random_state=42)
selected_df.to_csv(save_path, index=False)
