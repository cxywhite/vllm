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
import csv
import json
import os
import sys
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from typing import List, Dict, Any, Optional, Callable, Union
from pathlib import Path
import csv
import json
import os
from datetime import datetime
import re
'''
paths = save_outputs(
    outputs=benchmark_results,
    out_path="./results/perf",
    fields=["req_id", "metrics.latency", "metrics.tps", "prompt"],
    header=["ID", "Latency (ms)", "TPS", "Prompt"],
    default_values={"metrics.latency": -1, "metrics.tps": 0},
    field_transforms={
        "metrics.latency": lambda x: f"{x:.2f}",
        "prompt": lambda x: x[:50] + "..." if x and len(x) > 50 else x
    },
    save_json=False
)
'''
def save_outputs(
    outputs: List[Any],
    out_path: Union[str, Path],
    fields: List[str],
    header: Optional[List[str]] = None,
    default_values: Optional[Dict[str, Any]] = None,
    timestamp_fmt: str = "%Y%m%d_%H%M%S",
    save_json: bool = False,
    field_transforms: Optional[Dict[str, Callable[[Any], Any]]] = None,
    dataset_name: Optional[str] = None
) -> Dict[str, str]:
    """
    通用数据保存函数：将任意结构的输出对象保存为 CSV（可选 JSON）。

    Args:
        outputs: 输出对象列表，每个对象应包含 fields 指定的属性
        out_path: 输出目录或文件前缀（不含扩展名）
        fields: 要提取的属性名列表（支持嵌套属性，如 "metadata.latency"）
        header: 可选，CSV 表头显示名称。若为 None 则使用 fields
        default_values: 可选，属性缺失时的默认值字典，如 {"latency": 0}
        timestamp_fmt: 时间戳格式，默认 "%Y%m%d_%H%M%S"
        save_json: 是否同时保存 JSON 副本（包含原始对象结构）
        field_transforms: 可选，字段值转换函数，如 {"tokens": lambda x: x[:10]}

    Returns:
        Dict[str, str]: 保存的文件路径，如 {"csv": ".../file.csv", "json": ".../file.json"}
    """
    print(f'dataset_name: {dataset_name}')
    # === 路径解析 ===
    out_path = Path(out_path)
    if out_path.is_dir() or str(out_path).endswith(os.sep):
        base_dir = out_path
        base_name = "results"
    else:
        base_dir = out_path.parent if out_path.parent != Path("") else Path(".")
        base_name = out_path.stem

    base_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime(timestamp_fmt)

    # === 文件路径生成 ===
    if dataset_name:
        match = re.search(r'_part_(\d+)\.csv$', dataset_name)
        if match:
            number = int(match.group(1))
            base_name = f"{base_name}_part_{number:03d}"
    csv_filename = f"{base_name}_{ts}.csv"
    csv_path = base_dir / csv_filename
    result_paths = {"csv": str(csv_path)}

    # === 表头准备 ===
    if header is None:
        header = fields.copy()
    if len(header) != len(fields):
        raise ValueError(f"header 长度 ({len(header)}) 必须与 fields 长度 ({len(fields)}) 一致")

    # === 默认值准备 ===
    default_values = default_values or {}
    for field in fields:
        default_values.setdefault(field, "N/A")

    # === 嵌套属性安全访问 ===
    def get_nested_attr(obj: Any, attr_path: str, default: Any = None) -> Any:
        """安全获取嵌套属性，如 obj.metadata.latency"""
        try:
            for part in attr_path.split("."):
                if isinstance(obj, dict):
                    obj = obj.get(part, default)
                else:
                    obj = getattr(obj, part, default)
                if obj is None:
                    return default
            return obj
        except (AttributeError, KeyError, TypeError):
            return default

    # === 写入 CSV ===
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for out in outputs:
            row = []
            for field in fields:
                # 获取原始值
                value = get_nested_attr(out, field, default_values.get(field, "N/A"))
                # 应用转换函数（如有）
                if field_transforms and field in field_transforms:
                    try:
                        value = field_transforms[field](value)
                    except Exception as e:
                        value = f"[ERR:{e}]"
                # 转为字符串（避免 CSV 写入异常）
                row.append(str(value) if value is not None else str(default_values.get(field, "N/A")))
            writer.writerow(row)

    # === 可选：保存 JSON 副本 ===
    if save_json:
        json_filename = f"{base_name}_{ts}.json"
        json_path = base_dir / json_filename
        result_paths["json"] = str(json_path)

        serializable_outputs = []
        for out in outputs:
            item = {}
            for field in fields:
                val = get_nested_attr(out, field, default_values.get(field, "N/A"))
                item[field] = val
            serializable_outputs.append(item)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "saved_at": datetime.now().isoformat(),
                    "total_records": len(outputs),
                    "fields": fields,
                    "header": header
                },
                "data": serializable_outputs
            }, f, ensure_ascii=False, indent=2)

    return result_paths
def save_sample(
    input_requests: List[Any],           # list[RequestFuncOutput]
    out_path: str | Path,        # directory or file prefix (no ext needed)
    timestamp_fmt: str = "%Y%m%d_%H%M%S",
) -> str:
    """
    将 input_requests 保存为 CSV 文件。

    """
    out_path = Path(out_path)
    if out_path.is_dir() or str(out_path).endswith(os.sep):
        base_dir = out_path
        base_name = "sampled_requests"
    else:
        base_dir = out_path.parent if out_path.parent != Path("") else Path(".")
        base_name = out_path.stem

    base_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime(timestamp_fmt)
    csv_filename = f"{base_name}_{ts}.csv"
    csv_path = base_dir / csv_filename

    # CSV header
    header = ["req_id", "prompt", "temperature", "top_p", "top_k", "repetition_penalty"]

    # 写 CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for req in input_requests:
            req_id = getattr(req, "req_id", "unknown")
            prompt = getattr(req, "prompt", "")
            temperature = getattr(req, "temperature", 0.0)
            top_p = getattr(req, "top_p", 0.0)
            top_k = getattr(req, "top_k", 0)
            repetition_penalty = getattr(req, "repetition_penalty", 1.0)
            writer.writerow([req_id, prompt, temperature, top_p, top_k, repetition_penalty])
    return str(csv_path)
def save_test_outputs(
    outputs: List[Any],           # list[RequestFuncOutput]
    out_path: str | Path,        # directory or file prefix (no ext needed)
    timestamp_fmt: str = "%Y%m%d_%H%M%S",
    run_config: Optional[Dict[str, Any]] = None,
    prompt_truncate: Optional[int] = None,  # 可选：截断 prompt 到多少字符
) -> Dict[str, str]:
    """
    将 outputs 保存为 CSV 文件，包含每个 output 的req_id和对应output_tokens。

    """
    out_path = Path(out_path)
    if out_path.is_dir() or str(out_path).endswith(os.sep):
        base_dir = out_path
        base_name = "test_results"
    else:
        base_dir = out_path.parent if out_path.parent != Path("") else Path(".")
        base_name = out_path.stem

    base_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime(timestamp_fmt)
    csv_filename = f"{base_name}_{ts}.csv"
    json_filename = f"{base_name}_{ts}.json"
    csv_path = base_dir / csv_filename
    json_path = base_dir / json_filename

    # CSV header
    header = ["req_id","prompt","prompt_len","output_tokens","expect_output_len","temperature","top_p","top_k","repetition_penalty","latency","ttft","tpot","itl"]

    # 写 CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for out in outputs:
            req_id = getattr(out, "req_id", "unknown")
            prompt = getattr(out, "prompt", "")
            prompt_len = getattr(out, "prompt_len", 0)
            output_tokens = getattr(out, "output_tokens", 0)
            expect_output_len = getattr(out, "expect_output_len", 0)
            temperature = getattr(out, "temperature", 0.0)
            top_p = getattr(out, "top_p", 0.0)
            top_k = getattr(out, "top_k", 0)
            repetition_penalty = getattr(out, "repetition_penalty", 1.0)
            latency = getattr(out, "latency", 0.0)
            ttft = getattr(out, "ttft", 0.0)
            tpot = getattr(out, "tpot", 0.0)
            itl = getattr(out, "itl", [])
            writer.writerow([req_id, prompt, prompt_len, output_tokens, expect_output_len, temperature, top_p, top_k, repetition_penalty, latency, ttft, tpot, itl])
    #只保存csv文件并返回路径
    return str(csv_path)

def save_full_outputs(
    outputs: List[Any],           # list[RequestFuncOutput]
    out_path: str | Path,        # directory or file prefix (no ext needed)
    timestamp_fmt: str = "%Y%m%d_%H%M%S",
    run_config: Optional[Dict[str, Any]] = None,
    prompt_truncate: Optional[int] = None,  # 可选：截断 prompt 到多少字符
) -> Dict[str, str]:
    """
    将 outputs 保存为 CSV 文件，包含每个 output 的req_id和对应output_tokens。

    """
    out_path = Path(out_path)
    if out_path.is_dir() or str(out_path).endswith(os.sep):
        base_dir = out_path
        base_name = "results"
    else:
        base_dir = out_path.parent if out_path.parent != Path("") else Path(".")
        base_name = out_path.stem

    base_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime(timestamp_fmt)
    csv_filename = f"{base_name}_{ts}.csv"
    json_filename = f"{base_name}_{ts}.json"
    csv_path = base_dir / csv_filename
    json_path = base_dir / json_filename
    # CSV header
    header = ["req_id", "output_tokens","expect_output_len","prompt","prompt_len","generated_text","success","latency","ttft","itl","tpot","error"]

    # 写 CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for out in outputs:
            req_id = getattr(out, "req_id", "unknown")
            output_tokens = getattr(out, "output_tokens", 0)
            expect_output_len = getattr(out, "expect_output_len", 0)
            prompt = getattr(out, "prompt", "")
            prompt_len = getattr(out, "prompt_len", 0)
            generated_text = getattr(out, "generated_text", "")
            success = getattr(out, "success", False)
            latency = getattr(out, "latency", 0.0)
            ttft = getattr(out, "ttft", 0.0)
            itl = getattr(out, "itl", [])
            tpot = getattr(out, "tpot", 0.0)
            error = getattr(out, "error", "")
            writer.writerow([req_id, output_tokens])
    #只保存csv文件并返回路径
    return str(csv_path)
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


_re_suffix = re.compile(r"^(.*?)(?:_r(\d+))?$")

def _safe_get_attr(obj, *attrs, default=None):
    for a in attrs:
        if hasattr(obj, a):
            return getattr(obj, a)
    return default

def save_outputs_grouped_csv_by_base_reqid(
    outputs: List[Any],           # list[RequestFuncOutput]
    out_path: str | Path,        # directory or file prefix (no ext needed)
    run_config: Optional[Dict[str, Any]] = None,
    timestamp_fmt: str = "%Y%m%d_%H%M%S",
    prompt_truncate: Optional[int] = None,  # 可选：截断 prompt 到多少字符
) -> Dict[str, str]:
    """
    将 outputs 按 base req_id 聚合（去掉尾部 _rN），并把相同 base 的多个 output_tokens 展平成一行：
    req_id, prompt, max_tokens, output_len1, output_len2, ... output_lenM

    返回 {'csv': csv_path, 'json': json_path}
    """
    out_path = Path(out_path)
    if out_path.is_dir() or str(out_path).endswith(os.sep):
        base_dir = out_path
        base_name = "results"
    else:
        base_dir = out_path.parent if out_path.parent != Path("") else Path(".")
        base_name = out_path.stem

    base_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime(timestamp_fmt)
    csv_filename = f"{base_name}_{ts}.csv"
    json_filename = f"{base_name}_{ts}.json"
    csv_path = base_dir / csv_filename
    json_path = base_dir / json_filename

    # groups: base_id -> {"prompt": first_nonempty, "max_tokens": first_nonempty, "outputs": {rep_index: token}}
    groups: Dict[str, Dict[str, Any]] = {}

    for idx, out in enumerate(outputs):
        raw_req_id = _safe_get_attr(out, "req_id", "reqid", "id", default=None)
        if not raw_req_id:
            raw_req_id = f"{idx}"

        m = _re_suffix.match(str(raw_req_id))
        if m:
            base_id = m.group(1)
            rep_idx = int(m.group(2)) if m.group(2) is not None else None
        else:
            base_id = str(raw_req_id)
            rep_idx = None

        if base_id not in groups:
            groups[base_id] = {"prompt": None, "max_tokens": None, "outputs": {}, "order_idxs": []}

        g = groups[base_id]

        # prompt
        prompt = _safe_get_attr(out, "prompt", "generated_text", default=None)
        if prompt is not None and (g["prompt"] in (None, "")):
            g["prompt"] = prompt

        # expected output length keys compatibility
        expect = _safe_get_attr(out, "expect_output_len", "expected_output_len", "expect_len", "expected_len", default=None)
        if expect is not None and g["max_tokens"] in (None, ""):
            try:
                g["max_tokens"] = int(expect)
            except Exception:
                g["max_tokens"] = expect

        # output tokens
        output_tokens = _safe_get_attr(out, "output_tokens", "output_len", "tokens", "completion_tokens", default=None)
        try:
            output_tokens = int(output_tokens) if output_tokens is not None else 0
        except Exception:
            output_tokens = 0

        # 存储到 outputs dict，可根据 rep_idx 放在对应位置，否则使用发现顺序追加
        if rep_idx is not None:
            g["outputs"][rep_idx] = output_tokens
        else:
            # 找到下一个可用索引（避免覆盖已有 rep_idx）
            next_idx = 0
            while next_idx in g["outputs"]:
                next_idx += 1
            g["outputs"][next_idx] = output_tokens

    # 计算全局最大复制数（用于列头）
    max_copies = 0
    for base_id, g in groups.items():
        if g["outputs"]:
            max_copies = max(max_copies, max(g["outputs"].keys()) + 1)  # rep_idx 是 0-based
    # 处理当 groups 里某些 base 没有任何 outputs 的情况
    max_copies = max(max_copies, 0)

    # CSV header
    header = ["req_id", "prompt", "max_tokens"] + [f"output_len{i+1}" for i in range(max_copies)]

    # 写 CSV - 修改这里：设置 escapechar 和 quoting 策略
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        # 关键修改：设置 escapechar 和 quoting
        writer = csv.writer(f, escapechar='\\', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        # 按 base_id 排序输出，保证可重复性
        for base_id in sorted(groups.keys()):
            g = groups[base_id]
            prompt_val = g["prompt"] if g["prompt"] is not None else ""
            if prompt_truncate and isinstance(prompt_val, str):
                prompt_val = prompt_val[:prompt_truncate]
            max_tokens_val = g["max_tokens"] if g["max_tokens"] is not None else ""
            # 构造 output 列，按 index 顺序 0..max_copies-1 填写，缺的以空字符串占位
            row_outputs = []
            for i in range(max_copies):
                if i in g["outputs"]:
                    row_outputs.append(str(g["outputs"][i]))
                else:
                    row_outputs.append("")
            row = [base_id, prompt_val, max_tokens_val] + row_outputs
            writer.writerow(row)

    # 写 JSON run config（若无则记录 sys.argv）
    saved_config = run_config if run_config is not None else {"cmd": " ".join(sys.argv)}
    meta = {
        "generated_at": datetime.now().isoformat(),
        "csv_file": str(csv_path),
        "entries": len(groups),
        "max_copies": max_copies,
        "run_config": saved_config,
    }
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(meta, jf, indent=2, ensure_ascii=False)
    # 打印保存的文件路径
    # print(f"保存聚合结果到 CSV: {csv_path}")
    # print(f"保存运行配置到 JSON: {json_path}")
    return {"csv": str(csv_path), "json": str(json_path)}
