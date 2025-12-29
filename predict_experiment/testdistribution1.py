#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM 批量请求脚本（修改版）
- 每个请求只尝试发送一次（内部不进行多次重试）
- 去掉了每次调用的 max_retries 参数
- 初次发送完成后会收集失败请求并按 --ensure-success 配置进行多轮重传
"""

import os
import sys
import json
import time
import random
import argparse
import asyncio
from typing import Optional, Any, Dict, List, Tuple

import requests
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

try:
    from transformers import AutoTokenizer, AutoConfig
except Exception:
    AutoTokenizer = None
    AutoConfig = None

# ========== 简易异步进度条封装 ==========
class _AsyncTqdm:
    def __init__(self, total: int, desc: str = '', disable: bool = False):
        self._pbar = tqdm(total=total, desc=desc, disable=disable)
    def update(self, n: int = 1):
        self._pbar.update(n)
    def close(self):
        self._pbar.close()

def async_tqdm(total: int, desc: str = '', disable: bool = False):
    return _AsyncTqdm(total=total, desc=desc, disable=disable)

# ========== 数据加载与 ID 处理 ==========
def ensure_id_column_from_df(df: pd.DataFrame) -> pd.DataFrame:
    if 'id' in df.columns:
        return df
    id_candidates = ['_id', 'uid', 'idx', 'index', 'id_str']
    for col in id_candidates:
        if col in df.columns:
            print(f"找到ID候选列 '{col}'，重命名为 'id'")
            return df.rename(columns={col: 'id'})
    print("未找到ID列，创建新的索引列作为id")
    return df.reset_index().rename(columns={'index': 'id'})


def load_and_prepare_dataset(dataset_path: str, split: Optional[str] = None) -> pd.DataFrame:
    print(f"尝试加载数据集: {dataset_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集文件不存在: {dataset_path}")
    try:
        if dataset_path.endswith('.jsonl'):
            print("检测到.jsonl文件，使用逐行读取...")
            records = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line.strip())
                            records.append(record)
                        except json.JSONDecodeError as e:
                            print(f"警告: 跳过无效JSON行: {e}")
            if not records:
                raise ValueError("没有读取到有效的JSON记录")
            df = pd.DataFrame(records)
            print(f"成功读取 {len(df)} 条记录")
        elif dataset_path.endswith('.json'):
            print("检测到.json文件，使用pandas直接读取...")
            try:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    df = pd.DataFrame(data)
                elif isinstance(data, dict) and 'data' in data:
                    df = pd.DataFrame(data['data'])
                else:
                    raise ValueError("不支持的JSON格式")
                print(f"成功读取 {len(df)} 条记录")
            except Exception as e:
                print(f"尝试JSON数组格式失败: {e}")
                print("尝试逐行读取JSON...")
                records = []
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            try:
                                record = json.loads(line.strip())
                                records.append(record)
                            except json.JSONDecodeError:
                                continue
                if not records:
                    raise ValueError("无法解析JSON文件格式")
                df = pd.DataFrame(records)
                print(f"成功读取 {len(df)} 条记录")
        else:
            raise ValueError("仅支持.json或.jsonl格式的本地文件")
        return ensure_id_column_from_df(df)
    except Exception as e:
        print(f"加载数据集失败: {e}")
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            print(f"磁盘空间: 总计 {total//2**30}GB, 已用 {used//2**30}GB, 空闲 {free//2**30}GB")
        except Exception:
            pass
        print("建议: 对于大JSONL文件，逐行读取比一次性load更可靠")
        sys.exit(1)

# ========== 提取第一个 human prompt ==========
def extract_first_human_prompt(conversations: Any) -> str:
    if not isinstance(conversations, list):
        return ''
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get('from', '')).strip().lower()
        if role in ['human', 'user'] and 'value' in turn:
            value = turn['value']
            if isinstance(value, str):
                return value.strip()
            elif isinstance(value, list) and value:
                return str(value[0]).strip() if value else ''
    return ''

# ========== vLLM 调用封装（只尝试一次） ==========
async def call_vllm_api_async(
    prompt: str,
    base_url: str,
    model_name: str,
    sampling_params: Dict[str, Any],
    api_key: Optional[str] = None,
    timeout: int = 120,
) -> Optional[Dict]:
    """只发送一次请求；出现异常或非200状态码则视为失败并返回 None"""
    url = f"{base_url.rstrip('/')}/v1/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": sampling_params.get("max_tokens", 1024),
        "temperature": sampling_params.get("temperature", 1.0),
        "top_p": sampling_params.get("top_p", 1.0),
        "top_k": sampling_params.get("top_k", -1),
        "repetition_penalty": sampling_params.get("repetition_penalty", 1.0),
        "echo": False,
        "stream": False,
    }
    try:
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: requests.post(url, headers=headers, json=payload, timeout=timeout),
        )
        if response.status_code == 200:
            try:
                return response.json()
            except Exception:
                return None
        else:
            # 非200视为失败，返回None；调用方会记录失败并在批量完成后重试
            if response.status_code == 429:
                # 打印限流提示（不进行内部重试）
                print(f"请求被限流: HTTP 429")
            else:
                error_msg = response.text[:200] if response.text else f"HTTP {response.status_code}"
                print(f"API错误: HTTP {response.status_code}, {error_msg}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"网络错误: {e}")
        return None
    except Exception as e:
        print(f"未知错误: {e}")
        return None

# ========== 速率受控请求发送器（每个请求只发送一次） ==========
async def send_requests_rate_limited_async(
    base_url: str,
    model_name: str,
    prompts: List[str],
    sampling_params_list: List[Dict[str, Any]],
    api_key: Optional[str],
    request_rate: float,
    max_concurrency: Optional[int] = None,
    timeout: int = 120,
    verbose: bool = False,
    burstiness: float = 1.0,
) -> List[Optional[Dict]]:
    total_requests = len(prompts)
    if len(sampling_params_list) != total_requests:
        raise ValueError("sampling_params_list 长度必须与 prompts 长度相同")

    results: List[Optional[Dict]] = [None] * total_requests
    if verbose:
        print(f"{'='*50}")
        print(f"准备发送 {total_requests} 个请求（每个请求仅发送一次）")
        print(f"请求速率: {request_rate} RPS")
        print(f"最大并发: {max_concurrency if max_concurrency else '无限制'}")
        print(f"流量突发性: {burstiness} ({'泊松分布' if burstiness == 1.0 else '伽马分布'})")
        print(f"{'='*50}")
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request(prompt: str, idx: int, sampling_params: Dict[str, Any]) -> Tuple[int, Optional[Dict]]:
        if semaphore:
            async with semaphore:
                return idx, await call_vllm_api_async(
                    prompt, base_url, model_name, sampling_params, api_key, timeout
                )
        else:
            return idx, await call_vllm_api_async(
                prompt, base_url, model_name, sampling_params, api_key, timeout
            )

    async def request_generator():
        if request_rate <= 0 or request_rate == float('inf'):
            for idx, prompt in enumerate(prompts):
                yield idx, prompt, sampling_params_list[idx], time.time()
            return
        current_time = time.time()
        distribution_name = "泊松分布" if burstiness == 1.0 else "伽马分布"
        if verbose:
            print(f"使用 {distribution_name} 生成请求间隔")
        for idx, prompt in enumerate(prompts):
            if burstiness == 1.0:
                interval = random.expovariate(request_rate)
            else:
                shape = 1.0 / burstiness
                scale = burstiness / request_rate
                interval = random.gammavariate(shape, scale)
            current_time += interval
            yield_time = current_time
            now = time.time()
            if yield_time > now:
                await asyncio.sleep(yield_time - now)
            yield idx, prompt, sampling_params_list[idx], yield_time

    pbar = async_tqdm(total=total_requests, desc="处理请求", disable=not verbose)
    tasks = []
    async for idx, prompt, sampling_params, timestamp in request_generator():
        task = asyncio.create_task(limited_request(prompt, idx, sampling_params))
        tasks.append(task)
    for coro in asyncio.as_completed(tasks):
        idx, result = await coro
        results[idx] = result
        pbar.update(1)
    pbar.close()
    return results

# ========== 失败请求的二次/多次重传（在初次发送后进行） ==========
async def retry_failed_requests_async(
    base_url: str,
    model_name: str,
    prompts: List[str],
    sampling_params_list: List[Dict[str, Any]],
    api_key: Optional[str],
    initial_responses: List[Optional[Dict]],
    request_rate: float,
    max_concurrency: Optional[int],
    timeout: int,
    verbose: bool,
    burstiness: float,
    ensure_retries: int = 5,
    ensure_backoff: float = 2.0,
) -> List[Optional[Dict]]:
    """
    对 initial_responses 中视为失败的请求进行重传，直到成功或达到 ensure_retries。
    成功判定逻辑: resp is dict and has non-empty choices[0].get('text','').
    """
    responses = list(initial_responses)
    def is_success(resp: Optional[Dict]) -> bool:
        if not resp or not isinstance(resp, dict):
            return False
        if 'choices' not in resp or len(resp['choices']) == 0:
            return False
        txt = resp['choices'][0].get('text', '')
        return bool(str(txt).strip())

    # 计算初始失败列表
    failed_indices = [i for i, r in enumerate(responses) if not is_success(r)]
    if not failed_indices:
        if verbose:
            print("所有请求初次已成功，无需重传。")
        return responses

    if verbose:
        print(f"初次发送完成：发现 {len(failed_indices)} 个失败请求，准备进行最多 {ensure_retries} 轮重传。")

    for attempt in range(ensure_retries):
        if not failed_indices:
            break
        sleep_time = ensure_backoff * (2 ** attempt) + random.random()
        if verbose:
            print(f"[重传轮 {attempt+1}/{ensure_retries}] 将重试 {len(failed_indices)} 个请求，等待 {sleep_time:.2f}s 后开始...")
        await asyncio.sleep(sleep_time)

        # 构建需要重传的 prompts/params 列表，并保留原始 index 映射
        retry_prompts = [prompts[i] for i in failed_indices]
        retry_params = [sampling_params_list[i] for i in failed_indices]
        # 发送这些请求（每个请求仍然只发送一次）
        retry_results = await send_requests_rate_limited_async(
            base_url=base_url,
            model_name=model_name,
            prompts=retry_prompts,
            sampling_params_list=retry_params,
            api_key=api_key,
            request_rate=request_rate,
            max_concurrency=max_concurrency,
            timeout=timeout,
            verbose=verbose,
            burstiness=burstiness,
        )

        # 将 retry_results 写回 responses 的相应位置
        new_failed = []
        for j, idx in enumerate(failed_indices):
            resp = retry_results[j]
            responses[idx] = resp
            if not is_success(resp):
                new_failed.append(idx)
        failed_indices = new_failed
        if verbose:
            print(f"[重传轮 {attempt+1}] 完成，剩余失败: {len(failed_indices)}")
    if failed_indices and verbose:
        print(f"重传完成后仍有 {len(failed_indices)} 个请求失败（达到重试上限）。")
    return responses

# ========== 主逻辑：加载、过滤、采样、展开请求、发送并收集 ==========
async def main_async():
    parser = argparse.ArgumentParser(description='使用vLLM在线服务批量生成文本样本（不截断prompt，过滤后采样，定制化采样参数）')
    parser.add_argument('--base-url', required=True, help='vLLM服务的基础URL，例如 http://localhost:8000')
    parser.add_argument('--model-name', required=True, help='在线服务中注册的模型名称')
    parser.add_argument('--api-key', default=None, help='API密钥（如果需要）')
    parser.add_argument('--dataset', required=True, help='数据集路径或HF名称')
    parser.add_argument('--split', default=None, help='数据集split（如train/validation）')
    parser.add_argument('--n', type=int, required=True, help='抽样数量 (从过滤后的数据集中随机抽取 n 条 prompt)')
    parser.add_argument('--m', type=int, required=True, help='每个 prompt 重复的请求次数（相邻 m 个请求相同）')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output', required=True, help='输出CSV基础路径（会自动添加时间戳）')
    parser.add_argument('--max-tokens-custom', type=int, default=None, help='自定义最大生成token数（每个请求会根据输入长度动态调整）')
    parser.add_argument('--temperature', type=float, default=1.0, help='采样温度')
    parser.add_argument('--topk', type=int, default=-1, help='topk采样')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='重复惩罚')
    parser.add_argument('--top-p', type=float, default=1.0, help='top-p采样')
    parser.add_argument('--batch-size', type=int, default=8, help='（已忽略）')
    parser.add_argument('--max-concurrency', type=int, default=8, help='最大并发请求数')
    parser.add_argument('--rps', type=float, default=0.0, help='发送速率 (requests per second)。0 表示无速率限制')
    parser.add_argument('--request-timeout', type=int, default=120, help='单个请求超时时间（秒）')
    parser.add_argument('--burstiness', type=float, default=1.0, help='流量突发性因子 (1.0=泊松分布, >1.0=突发流量)')
    parser.add_argument('--verbose', action='store_true', help='详细输出')

    # 新增用于保证成功的参数（针对初次发送后按批重传）
    parser.add_argument('--ensure-success', action='store_true', help='开启对失败请求的重传直到成功或达到 ensure-retries 上限')
    parser.add_argument('--ensure-retries', type=int, default=5, help='当 --ensure-success 打开时的最大重传轮数')
    parser.add_argument('--ensure-backoff', type=float, default=2.0, help='重传指数退避基准秒数（sleep = backoff * 2**attempt + jitter）')

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"【阶段1】加载和准备数据集...")
    try:
        df = load_and_prepare_dataset(args.dataset, args.split)
        print(f"✓ 数据集加载成功，共 {len(df)} 条记录")
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        sys.exit(1)

    if 'conversations' not in df.columns:
        conversation_candidates = ['messages', 'dialogue', 'turns', 'conversation']
        found = False
        for candidate in conversation_candidates:
            if candidate in df.columns:
                print(f"未找到'conversations'字段，但找到'{candidate}'字段，将使用此字段")
                df = df.rename(columns={candidate: 'conversations'})
                found = True
                break
        if not found:
            print("错误: 数据集缺少对话字段。需要包含以下任一字段: conversations, messages, dialogue, turns")
            print(f"可用字段: {', '.join(df.columns.tolist())}")
            sys.exit(1)

    print("提取人类prompt...")
    df['prompt'] = df['conversations'].apply(extract_first_human_prompt)
    original_count = len(df)
    df = df[df['prompt'].str.strip() != ''].reset_index(drop=True)
    valid_count = len(df)
    print(f"过滤空prompt: {original_count} -> {valid_count} 条有效记录")
    if len(df) == 0:
        print("错误: 没有找到有效的prompt")
        sys.exit(1)

    print(f"【阶段2】加载tokenizer和模型配置...")
    tokenizer = None
    max_pos = 2048  # 默认值

    try:
        if AutoTokenizer is None:
            raise RuntimeError('transformers 未安装或导入失败')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
        print(f"✓ 成功加载tokenizer: {args.model_name}")
    except Exception as e:
        print(f"✗ 加载tokenizer失败: {e}")
        print("错误: 必须加载tokenizer来过滤长prompt和计算token长度")
        sys.exit(1)

    try:
        if AutoConfig is None:
            raise RuntimeError('transformers 未安装或导入失败')
        config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        max_pos = getattr(config, 'max_position_embeddings', 2048)
        print(f"✓ 成功加载模型配置，最大位置: {max_pos}")
    except Exception as e:
        if args.verbose:
            print(f"✗ 加载模型配置失败，使用默认最大长度: {e}")
        max_pos = 2048

    max_pos = min(max_pos, 131072)  # 设置一个合理的上限
    print(f"最终使用最大位置: {max_pos}")

    print(f"【阶段3】计算prompt长度并过滤过长的prompt...")
    df['prompt_tokens'] = df['prompt'].apply(
        lambda x: len(tokenizer(x, add_special_tokens=False)['input_ids']) if tokenizer else len(x)
    )

    # 过滤掉超过max_pos的prompt（保留小于max_pos的）
    filtered_df = df[df['prompt_tokens'] < max_pos].reset_index(drop=True)
    filtered_count = len(filtered_df)

    print(f"过滤长prompt: {valid_count} -> {filtered_count} 条记录 (max_pos={max_pos})")

    if filtered_count == 0:
        print("错误: 过滤后没有可用的prompt。所有prompt都超过了模型的最大长度限制。")
        print(f"建议: 增加max_pos限制或使用更短的prompt数据集")
        sys.exit(1)

    if filtered_count < args.n:
        print(f"警告: 过滤后只有 {filtered_count} 条记录，少于请求的 {args.n} 条")
        print(f"将使用所有 {filtered_count} 条记录")
        sample_size = filtered_count
    else:
        sample_size = args.n

    sample_df = filtered_df.sample(n=sample_size, random_state=args.seed).reset_index(drop=True)
    print(f"抽样完成: {sample_size} 个样本 (从 {filtered_count} 个过滤后的样本中抽取)")

    # 为每个样本计算定制化的max_tokens（根据 input_len 计算）
    max_tokens_custom = args.max_tokens_custom or 1024
    sample_df['custom_max_tokens'] = sample_df['prompt_tokens'].apply(
        lambda input_len: max(1, min(max_pos - input_len, max_tokens_custom))
    )

    if args.verbose:
        print("\n样本长度统计:")
        print(f"  - 最小prompt长度: {sample_df['prompt_tokens'].min()}")
        print(f"  - 最大prompt长度: {sample_df['prompt_tokens'].max()}")
        print(f"  - 平均prompt长度: {sample_df['prompt_tokens'].mean():.1f}")
        print(f"  - 最小定制max_tokens: {sample_df['custom_max_tokens'].min()}")
        print(f"  - 最大定制max_tokens: {sample_df['custom_max_tokens'].max()}")
        print(f"  - 平均定制max_tokens: {sample_df['custom_max_tokens'].mean():.1f}")

    print(f"【阶段4】准备发送请求...")
    expanded_prompts: List[str] = []
    expanded_sampling_params: List[Dict[str, Any]] = []  # 为每个请求准备定制参数
    request_map: List[int] = []

    for sample_idx in range(len(sample_df)):
        row = sample_df.iloc[sample_idx]
        prompt = row['prompt']
        input_len = row['prompt_tokens']
        custom_max_tokens = int(row['custom_max_tokens'])

        # 为这个样本创建基础采样参数（使用根据 input_len 计算的 custom_max_tokens）
        base_sampling_params = {
            "max_tokens": custom_max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.topk,
            "repetition_penalty": args.repetition_penalty,
        }

        # 为m个重复请求创建相同的prompt和参数
        for _ in range(args.m):
            expanded_prompts.append(prompt)
            expanded_sampling_params.append(base_sampling_params.copy())  # 确保每个请求有自己的参数副本
            request_map.append(sample_idx)

    total_requests = len(expanded_prompts)
    print(f"总请求数 = n * m = {len(sample_df)} * {args.m} = {total_requests}")

    if args.verbose:
        print(f"最大并发 (信号量): {args.max_concurrency}")
        print(f"目标速率 (RPS): {args.rps if args.rps>0 else '无'}")
        # 显示前几个请求的参数示例
        print("\n前3个请求的定制参数示例:")
        for i in range(min(3, len(expanded_sampling_params))):
            print(f"  请求 {i}: max_tokens={expanded_sampling_params[i]['max_tokens']}, "
                  f"temp={expanded_sampling_params[i]['temperature']}")

    # 第一次批量发送（每个请求仅发出一次）
    responses = await send_requests_rate_limited_async(
        base_url=args.base_url,
        model_name=args.model_name,
        prompts=expanded_prompts,
        sampling_params_list=expanded_sampling_params,
        api_key=args.api_key,
        request_rate=args.rps if args.rps > 0 else float('inf'),
        max_concurrency=args.max_concurrency,
        timeout=args.request_timeout,
        verbose=args.verbose,
        burstiness=args.burstiness,
    )

    # 如果用户要求确保成功，尝试对视为失败的请求进行多轮重传
    if args.ensure_success:
        responses = await retry_failed_requests_async(
            base_url=args.base_url,
            model_name=args.model_name,
            prompts=expanded_prompts,
            sampling_params_list=expanded_sampling_params,
            api_key=args.api_key,
            initial_responses=responses,
            request_rate=args.rps if args.rps > 0 else float('inf'),
            max_concurrency=args.max_concurrency,
            timeout=args.request_timeout,
            verbose=args.verbose,
            burstiness=args.burstiness,
            ensure_retries=args.ensure_retries,
            ensure_backoff=args.ensure_backoff,
        )

    # ===== 后处理：把 responses 写回 sample_slots 并存盘（原逻辑不变） =====
    results: List[Dict[str, Any]] = []
    sample_slots: Dict[int, Dict[str, Any]] = {}

    for sample_idx in range(len(sample_df)):
        row = sample_df.iloc[sample_idx]
        sample_slots[sample_idx] = {
            'id': row['id'],
            'prompt': row['prompt'],
            'prompt_length': row['prompt_tokens'],
            'max_tokens_custom': int(row['custom_max_tokens']),
            'sampling_params': {
                'temperature': args.temperature,
                'top_p': args.top_p,
                'top_k': args.topk,
                'repetition_penalty': args.repetition_penalty,
            },
            'outputs': []
        }

    for i, resp in enumerate(responses):
        sample_idx = request_map[i]
        text = ''
        if resp and isinstance(resp, dict) and 'choices' in resp and len(resp['choices'])>0:
            text = resp['choices'][0].get('text', '')
        tok_len = 0
        if text and text.strip():
            try:
                tok_len = len(tokenizer(text, add_special_tokens=False)['input_ids'])
            except Exception as e:
                if args.verbose:
                    print(f"计算token长度失败: {e}")
                tok_len = 0

        sample_slots[sample_idx]['outputs'].append({
            'text': text[:200] + '...' if len(text) > 200 else text,
            'length': tok_len,
            'full_text': text
        })

    for sample_idx in range(len(sample_df)):
        slot = sample_slots[sample_idx]
        outputs = slot['outputs'] + [{}] * max(0, args.m - len(slot['outputs']))
        outputs = outputs[:args.m]

        row = {
            'id': slot['id'],
            'prompt': slot['prompt'],
            'prompt_length': slot['prompt_length'],
            'max_tokens_requested': slot['max_tokens_custom'],
            'temperature': slot['sampling_params']['temperature'],
            'top_p': slot['sampling_params']['top_p'],
            'top_k': slot['sampling_params']['top_k'],
            'repetition_penalty': slot['sampling_params']['repetition_penalty'],
        }

        for k in range(args.m):
            if k < len(outputs) and outputs[k]:
                row[f'output_text_{k+1}'] = outputs[k]['text']
                row[f'output_len_{k+1}'] = outputs[k]['length']
            else:
                row[f'output_text_{k+1}'] = ''
                row[f'output_len_{k+1}'] = 0

        results.append(row)

    print(f"{'='*50}")
    print(f"【阶段5】结果保存和统计")
    print(f"{'='*50}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{os.path.splitext(args.output)[0]}_{timestamp}.csv"

    result_df = pd.DataFrame(results)

    # 确保所有必要的列都存在
    base_columns = ['id', 'prompt', 'prompt_length', 'max_tokens_requested', 
                   'temperature', 'top_p', 'top_k', 'repetition_penalty']
    output_columns = [f'output_text_{i+1}' for i in range(args.m)] + [f'output_len_{i+1}' for i in range(args.m)]
    columns = base_columns + output_columns

    for col in columns:
        if col not in result_df.columns:
            result_df[col] = ''

    result_df = result_df[columns]
    result_df.to_csv(output_path, index=False)
    print(f"✓ 完成! 结果已保存到: {output_path}")

    total_requests = len(results) * args.m
    successful_requests = sum(1 for r in results for i in range(args.m) 
                            if r.get(f'output_len_{i+1}', 0) > 0)
    success_rate = successful_requests / total_requests * 100 if total_requests > 0 else 0

    print(f"📊 总请求数: {total_requests}")
    print(f"📊 成功请求数: {successful_requests}")
    print(f"📊 成功率: {success_rate:.2f}%")

    all_lengths = [r[f'output_len_{i+1}'] for r in results for i in range(args.m) 
                  if r.get(f'output_len_{i+1}', 0) > 0]

    if all_lengths:
        print(f"📊 生成长度统计:")
        print(f"   - 平均长度: {np.mean(all_lengths):.1f}")
        print(f"   - 中位数: {np.median(all_lengths):.1f}")
        print(f"   - 最小值: {np.min(all_lengths)}")
        print(f"   - 最大值: {np.max(all_lengths)}")
        print(f"   - 标准差: {np.std(all_lengths):.1f}")
        
        # 显示实际使用的max_tokens分布
        actual_max_tokens = [r['max_tokens_requested'] for r in results]
        print(f"\n📊 请求参数统计:")
        print(f"   - 平均请求max_tokens: {np.mean(actual_max_tokens):.1f}")
        print(f"   - 最小请求max_tokens: {np.min(actual_max_tokens)}")
        print(f"   - 最大请求max_tokens: {np.max(actual_max_tokens)}")

    # 保存完整的生成文本到单独文件
    full_texts = []
    for r in results:
        for i in range(args.m):
            sid = r['id']
            matches = sample_df.index[sample_df['id'] == sid].tolist()
            if not matches:
                continue
            sidx = matches[0]
            outputs = sample_slots[sidx]['outputs'] if sidx in sample_slots else []
            if i < len(outputs):
                full_text = outputs[i].get('full_text', '')
                if full_text:
                    full_texts.append({
                        'id': r['id'],
                        'prompt': r['prompt'],
                        'output_index': i+1,
                        'text': full_text
                    })

    if full_texts:
        full_text_df = pd.DataFrame(full_texts)
        full_text_path = f"{os.path.splitext(args.output)[0]}_{timestamp}_full_texts.csv"
        full_text_df.to_csv(full_text_path, index=False)
        print(f"✓ 完整生成文本已保存到: {full_text_path}")

    config_path = f"{os.path.splitext(args.output)[0]}_{timestamp}_config.json"
    config_info = {
        "command": " ".join(sys.argv),
        "args": vars(args),
        "timestamp": timestamp,
        "model_max_position": max_pos,
        "total_prompts": len(results),
        "requests_per_prompt": args.m,
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "success_rate": success_rate,
        "filtered_dataset_size": filtered_count,
        "original_dataset_size": valid_count,
    }

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    print(f"✓ 配置信息已保存到: {config_path}")


def main():
    asyncio.run(main_async())

if __name__ == '__main__':
    main()
