import datasets
from transformers import AutoTokenizer
import pandas as pd
import random
import pdb
dataset_path=f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/llama_dataset/mysharegpt/benchmark_np103415_rr4_mt8192_20260117_163601_1p4d/dataset_result/results_20260119_035802.csv'
model_path="/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct"
df = pd.read_csv(dataset_path)
# 随机选择50k条数据
random.seed(42)
df = df.sample(n=50000, random_state=42).reset_index(drop=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
ids=[]
prompts=[]
input_tokens=[]
input_lengths=[]
custom_max_tokens=[]
temperatures=[]
top_ps=[]
top_ks=[]
repetitions_penaltys=[]
for i in range(len(df)):
    ids.append(i)
    prompt = df.loc[i, 'prompt']
    prompts.append(str(prompt))
    message = [{"role": "user", "content": str(prompt)}]
    input = tokenizer.apply_chat_template(message, return_tensors="pt", add_generation_prompt=True)
    input_token = input[0].tolist()
    input_tokens.append(input_token)
    input_lengths.append(len(input_token))
    custom_max_tokens.append(8192 - len(input_token))
    # temperature 在 0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8 中随机选择
    temperature = random.choice([0.2, 0.5, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    temperatures.append(temperature)
    # top_p 在 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 中随机选择
    top_p = random.choice([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    top_ps.append(top_p)
    # top_k 在 10, 50, 500, 10000, 50000, -1中随机选择
    top_k = random.choice([10, 50, 500, 10000, 50000, -1])
    top_ks.append(top_k)
    # repetition_penalty 在 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 中随机选择
    repetition_penalty = random.choice([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    repetitions_penaltys.append(repetition_penalty)
df = pd.DataFrame({
    'id': ids,
    'prompt': prompts,
    'input_tokens': input_tokens,
    'input_length': input_lengths,
    'custom_max_tokens': custom_max_tokens,
    'temperature': temperatures,
    'top_p': top_ps,
    'top_k': top_ks,
    'repetition_penalty': repetitions_penaltys
})
# 每16K条数据保存为一个csv文件
for i in range(0, len(df), 16000):
    df_part = df.iloc[i:i+16000]
    # 打印前5行数据进行检查
    print(df_part.head())
    pdb.set_trace()
    df_part.to_csv(f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset/dataset/llama_dataset/mysharegpt/split/mysharegpt_preprocessed_prompts_part_{i//16000}.csv', index=False)
