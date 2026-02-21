import datasets
import transformers
from transformers import AutoTokenizer,AutoConfig
import random
import os
from datasets import load_from_disk
import pandas as pd
dataset_name="lmsys"
model='qwen'
num_prompts=500
output_csv_path="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/qwen_dataset/lmsys_preprocessed_prompts.csv"
# 确保输出目录存在
output_dir=os.path.dirname(output_csv_path)
os.makedirs(output_dir,exist_ok=True)
global tokenizer
if dataset_name=="lmsys" and model == 'qwen':
    dataset=datasets.load_from_disk("/root/myshare/predictor/lmsys_meta-llama-3-8b-instruct_multi-cls_50K/lmsys-50k")
    dataset=dataset.remove_columns(['id', 'input_tokens', 'input_length', 'output', 'output_tokens', 'output_length', 'class'])
    # huggingface datasets 转成 dataframe
    df=pd.DataFrame(dataset)
    tokenizer=None
    if model=='qwen':
        tokenizer = AutoTokenizer.from_pretrained('/root/.cache/huggingface/hub/Qwen2.5-7B-Instruct',local_files_only=True)
    def tokenize_len(text:str) -> int:
        global tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        tokens = tokenizer(text, add_special_tokens=False)
        return len(tokens['input_ids'])
    df['prompt_tokens']=df['prompt'].apply(lambda x: tokenize_len(x))
    
    before_filter_counts=len(df)
    df = df[
        (df['prompt_tokens'] >= 1) &
        (df['prompt_tokens'] <= 8192)
    ].reset_index(drop=True)
    after_filter_counts=len(df)
    print(f"Valid samples before filtering: {before_filter_counts}, after filtering: {after_filter_counts}")
    df=df.sample(frac=1,random_state=42).reset_index(drop=True)
    sampled_df=df.sample(n=500,random_state=42).reset_index(drop=True)
    print(f"Sampled {len(sampled_df)} prompts for preprocessing.")

    sampled_df['custom_max_tokens']=sampled_df['prompt_tokens'].apply(lambda input_len: max(1,min(32768-int(input_len),32768)))
    # id 从0开始重新编号
    sampled_df['id']=range(len(sampled_df))
    # 遍历sampled_df添加temperature,top_p,top_k,repetition_penalty列
    for index, row in sampled_df.iterrows():
        # d_temp在[0.8,1.0]中随机取值
        d_temp = random.choice([0.80, 1.00])
        # d_top_p在[0.7,0.9,1.0]中随机取值
        d_top_p = random.choice([0.70, 0.90, 1.00])
        # d_top_k在[-1,500,10000]中随机取值
        d_top_k = random.choice([-1, 500, 10000])
        # d_rep_pen在[1.0,1.1]
        d_rep_pen = random.choice([1.00, 1.10])
        sampled_df.at[index, 'temperature'] = d_temp
        sampled_df.at[index, 'top_p'] = d_top_p
        sampled_df.at[index, 'top_k'] = d_top_k
        sampled_df.at[index, 'repetition_penalty'] = d_rep_pen
    # 显示前5行
    print(sampled_df.head())
    # 保存成指定路径csv
    
    sampled_df.to_csv(output_csv_path,index=False)
    print(f"Preprocessed dataset saved to {output_csv_path}")
elif dataset_name=="sharegpt" and model == 'llama':
    df=pd.read_csv("/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/qwen_dataset/sharegpt_preprocessed_prompts.csv")
    
    tokenizer=None
    if model=='llama':
        tokenizer = AutoTokenizer.from_pretrained('/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct',local_files_only=True)
    def tokenize_len(text:str) -> int:
        global tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        tokens = tokenizer(text, add_special_tokens=False)
        return len(tokens['input_ids'])
    df['prompt_tokens']=df['prompt'].apply(lambda x: tokenize_len(x))
    df['custom_max_tokens']=df['prompt_tokens'].apply(lambda input_len: max(1,min(8192-int(input_len),8192)))
    print(df.head())
    # 保存成指定路径csv
    
    df.to_csv(output_csv_path,index=False)
    print(f"Preprocessed dataset saved to {output_csv_path}")
elif dataset_name=="sharegpt" and model == 'qwen':
    dataset_path="/root/.cache/huggingface/hub/datasets--shibing624--sharegpt_gpt4/snapshots/3fb53354e02a931777556fb1da37e931d73af48a"
    files=[
            "sharegpt_gpt4.jsonl",
            "sharegpt_V3_format.jsonl",
            "sharegpt_zh_38K_format.jsonl",
        ]
    target_file=[]
    for fname in files:
        fpath=os.path.join(dataset_path,fname)
        if os.path.exists(fpath):
            target_file.append(fpath)
    total_ds=[]
    for f in target_file:
        ds=datasets.load_dataset("json",data_files=f)
        total_ds.append(ds["train"])
        print(f"Loaded {f} with {len(ds['train'])} samples")
    ds_f=datasets.concatenate_datasets(total_ds)
    print(f"Total dataset size: {len(ds_f)} samples")

    parsed_rows = []
    for row in ds_f:
        
        convs = row.get("conversations", [])
        if not convs:
            continue
        prompt = ""
        for turn in convs:
            if turn.get('from') in ['human','user']:
                prompt = turn.get('value', '')
                break
        if not prompt.strip():
            continue

         # d_temp在[0.8,1.0]中随机取值
        d_temp = random.choice([0.80, 1.00])
        # d_top_p在[0.7,0.9,1.0]中随机取值
        d_top_p = random.choice([0.70, 0.90, 1.00])
        # d_top_k在[-1,500,10000]中随机取值
        d_top_k = random.choice([-1, 500, 10000])
        # d_rep_pen在[1.0,1.1]
        d_rep_pen = random.choice([1.00, 1.10])

        parsed_rows.append({
            "id": row.get('id', 0),
            "prompt": prompt,
            "prompt_tokens": -1,
            "temperature": d_temp,
            "top_p": d_top_p,
            "top_k": d_top_k,
            "repetition_penalty": d_rep_pen
        })
    df=pd.DataFrame(parsed_rows)
    valid_counts=len(df)
    
    tokenizer=None
    if model=='qwen':
        tokenizer = AutoTokenizer.from_pretrained('/root/.cache/huggingface/hub/Qwen2.5-7B-Instruct',local_files_only=True)
    def tokenize_len(text:str) -> int:
        global tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer is not initialized.")
        tokens = tokenizer(text, add_special_tokens=False)
        return len(tokens['input_ids'])
    df['prompt_tokens']=df['prompt'].apply(lambda x: tokenize_len(x))
    
    before_filter_counts=len(df)
    df = df[
        (df['prompt_tokens'] >= 1) &
        (df['prompt_tokens'] <= 8192)
    ].reset_index(drop=True)
    after_filter_counts=len(df)
    print(f"Valid samples before filtering: {before_filter_counts}, after filtering: {after_filter_counts}")
    df=df.sample(frac=1,random_state=42).reset_index(drop=True)
    sampled_df=df.sample(n=500,random_state=42).reset_index(drop=True)
    print(f"Sampled {len(sampled_df)} prompts for preprocessing.")

    sampled_df['custom_max_tokens']=sampled_df['prompt_tokens'].apply(lambda input_len: max(1,min(32768-int(input_len),32768)))
    # id 从0开始重新编号
    sampled_df['id']=range(len(sampled_df))
    # 显示前5行
    print(sampled_df.head())
    # 保存成指定路径csv
    
    sampled_df.to_csv(output_csv_path,index=False)
    print(f"Preprocessed dataset saved to {output_csv_path}")
    