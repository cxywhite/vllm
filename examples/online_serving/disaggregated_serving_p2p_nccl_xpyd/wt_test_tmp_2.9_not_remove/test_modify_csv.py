import pandas as pd
from transformers import AutoTokenizer
import torch
import os
from typing import Optional, Union

def get_tokenizer(model_path: str) -> AutoTokenizer:
    """
    加载Llama3-8B-Instruct的tokenizer
    """
    try:
        # Llama3模型通常需要设置trust_remote_code=True和use_fast=False
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            use_fast=False, 
            trust_remote_code=True
        )
        print(f"成功加载tokenizer: {model_path}")
        return tokenizer
    except Exception as e:
        print(f"加载tokenizer失败: {e}")
        raise

def tokenize_len(text: str, tokenizer: Optional[AutoTokenizer] = None, tokenizer_id: Optional[str] = None) -> int:
    """
    计算文本的token长度，参考您提供的函数逻辑
    """
    if text is None or str(text).strip() == "":
        return 0
    
    text = str(text)
    
    try:
        if tokenizer is not None:
            # 尝试使用tokenizer直接调用
            maybe = tokenizer(text, add_special_tokens=True)
            if isinstance(maybe, dict) and 'input_ids' in maybe:
                return len(maybe['input_ids'])
            # 尝试使用encode方法
            if hasattr(tokenizer, 'encode'):
                ids = tokenizer.encode(text, add_special_tokens=True)
                return len(ids)
    except Exception as e:
        print(f"使用tokenizer计算失败: {e}")
        pass

    try:
        # 回退机制：尝试使用AutoTokenizer重新加载
        if tokenizer_id is not None:
            local_tok = AutoTokenizer.from_pretrained(
                tokenizer_id, 
                use_fast=False, 
                trust_remote_code=True
            )
            maybe = local_tok(text, add_special_tokens=True)
            if isinstance(maybe, dict) and 'input_ids' in maybe:
                return len(maybe['input_ids'])
    except Exception as e:
        print(f"回退加载tokenizer失败: {e}")
        pass

    # 最后的回退：返回字符长度（不准确，但保证有值）
    print(f"所有tokenizer方法失败，使用字符长度作为回退")
    return len(text)

def process_csv_with_tokenizer(csv_path: str, output_path: str, model_path: str):
    """
    处理CSV文件，添加input_len列
    """
    # 1. 读取CSV文件
    print(f"读取CSV文件: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"文件包含 {len(df)} 条记录")
    print(f"列名: {df.columns.tolist()}")
    
    # 2. 验证必需的列是否存在
    required_columns = ['req_id', 'prompt', 'max_tokens', 'output_len1']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV文件缺少必需的列: '{col}'")
    
    # 3. 加载tokenizer
    print("加载Llama3-8B-Instruct tokenizer...")
    tokenizer = get_tokenizer(model_path)
    
    # 4. 计算input_len
    print("计算每个prompt的输入长度...")
    
    # 使用apply方法，传入tokenizer和model_path作为参数
    df['input_len'] = df['prompt'].apply(
        lambda x: tokenize_len(x, tokenizer=tokenizer, tokenizer_id=model_path)
    )
    
    # 5. 显示统计信息
    print("\n输入长度统计:")
    print(f"最小长度: {df['input_len'].min()}")
    print(f"最大长度: {df['input_len'].max()}")
    print(f"平均长度: {df['input_len'].mean():.2f}")
    print(f"空值数量: {df['prompt'].isna().sum()}")
    
    # 6. 保存结果
    print(f"\n保存结果到: {output_path}")
    df.to_csv(output_path, index=False)
    print("保存成功！")
    
    # 7. 显示预览
    print("\n结果预览 (前5行):")
    preview_cols = ['req_id', 'input_len', 'max_tokens', 'output_len1']
    preview_cols = [col for col in preview_cols if col in df.columns]
    print(df.head()[preview_cols])

if __name__ == "__main__":
    # 配置参数
    CSV_FILE_PATH = "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/baseline_117/benchmark_np103415_rr4_mt8192_20260117_163601_1p4d/dataset_result/results_20260119_035802.csv"  # 替换为您的CSV文件路径
    OUTPUT_FILE_PATH = "/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/experiment_result/experiment_paper/tmp/baseline_117/benchmark_np103415_rr4_mt8192_20260117_163601_1p4d/dataset_result/results.csv"  # 输出文件路径
    
    # 模型路径配置（选择一种方式）
    # 方式1: 使用Hugging Face Hub上的模型 (需要登录和权限)
    # MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # 方式2: 使用本地模型路径 (推荐)
    MODEL_PATH = "/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct"  # 替换为您的本地模型路径
    
    # 如果使用Hugging Face Hub，需要先登录
    if "meta-llama" in MODEL_PATH:
        from huggingface_hub import login
        HF_TOKEN = "your_hf_token_here"  # 从环境变量获取更安全
        login(token=HF_TOKEN)
        print("已登录Hugging Face Hub")
    
    # 执行处理
    process_csv_with_tokenizer(
        csv_path=CSV_FILE_PATH,
        output_path=OUTPUT_FILE_PATH,
        model_path=MODEL_PATH
    )