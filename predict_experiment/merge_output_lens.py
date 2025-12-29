#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_output_lens.py

用法示例：
# 指定多个文件（按给定顺序合并）
python merge_output_lens.py --out-dir ./merged_out file1.csv file2.csv file3.csv

# 或者指定输入目录，脚本会读取该目录下所有 .csv（按文件名排序）
python merge_output_lens.py --input-dir ./dataset_result/benchmarkserving_result/deepseek/single --out-dir ./dataset_result/benchmarkserving_result/deepseek/merge

输出文件示例： llama_20251114T153045.csv
"""
import argparse
import os
import re
import sys
from datetime import datetime

import pandas as pd

OUTPUT_PREFIX = "merged"

def find_output_len_cols(columns):
    """返回按数字后缀排序的 output_len 列名列表，支持 output_len1 或 output_len_1"""
    pattern = re.compile(r"^output_len(?:_?)(\d+)$", re.IGNORECASE)
    cols = []
    for c in columns:
        if not isinstance(c, str):
            continue
        m = pattern.match(c.strip())
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort(key=lambda x: x[0])
    return [c for _, c in cols]

def read_csv_safe(path):
    """尝试用 utf-8-sig 读取 CSV，若失败再尝试默认编码"""
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, dtype=str)

def main():
    parser = argparse.ArgumentParser(description="把多个 CSV 中相同 req_id 的 output_len 列按文件顺序拼接到一起")
    parser.add_argument("files", nargs="*", help="要合并的 csv 文件（可选）")
    parser.add_argument("--input-dir", "-d", help="如果提供，则读取该目录下所有 .csv 文件（按文件名排序）")
    parser.add_argument("--out-dir", "-o", required=True, help="结果保存目录（会创建如果不存在）")
    parser.add_argument("--prefix", default=OUTPUT_PREFIX, help="输出文件名前缀，默认：merged")
    parser.add_argument("--verbose", "-v", action="store_true", help="打印过程信息和警告")
    args = parser.parse_args()

    # 构建输入文件列表（按顺序）
    input_files = []
    if args.input_dir:
        if not os.path.isdir(args.input_dir):
            print(f"ERROR: 指定的目录不存在: {args.input_dir}", file=sys.stderr)
            sys.exit(1)
        found = [os.path.join(args.input_dir, f) for f in sorted(os.listdir(args.input_dir)) if f.lower().endswith(".csv")]
        input_files.extend(found)
    if args.files:
        input_files.extend(args.files)

    if not input_files:
        print("ERROR: 没有找到输入文件（请指定文件或 --input-dir）。", file=sys.stderr)
        sys.exit(1)

    # 去重并保持顺序
    seen = set()
    ordered_files = []
    for f in input_files:
        if f not in seen:
            seen.add(f)
            ordered_files.append(f)
    input_files = ordered_files

    if args.verbose:
        print(f"将按以下顺序处理 {len(input_files)} 个文件：")
        for i, f in enumerate(input_files, 1):
            print(f"  {i}. {f}")

    # 主数据结构： req_id -> dict(prompt='', max_tokens='', lengths=[...])
    data = {}
    max_len_count = 0

    for file_idx, file_path in enumerate(input_files, start=1):
        if args.verbose:
            print(f"\n读取文件 {file_idx}: {file_path}")
        if not os.path.isfile(file_path):
            print(f"WARNING: 文件不存在，跳过: {file_path}", file=sys.stderr)
            continue
        try:
            df = read_csv_safe(file_path)
        except Exception as e:
            print(f"ERROR: 无法读取文件 {file_path}: {e}", file=sys.stderr)
            sys.exit(1)

        # 规范化列名（strip 空格）
        df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

        # 把 prompt / max_tokens 统一填充并转为字符串，避免 pd.NA 导致布尔判断出错
        if 'prompt' in df.columns:
            df['prompt'] = df['prompt'].fillna('').astype(str)
        if 'max_tokens' in df.columns:
            df['max_tokens'] = df['max_tokens'].fillna('').astype(str)

        # 必需列检查
        if 'req_id' not in df.columns:
            print(f"ERROR: 文件 {file_path} 中缺少 req_id 列。", file=sys.stderr)
            sys.exit(1)

        output_len_cols = find_output_len_cols(df.columns)
        if not output_len_cols:
            if args.verbose:
                print(f"警告: 在文件 {file_path} 中未找到任何 output_len* 列，跳过该文件。")
            continue

        if args.verbose:
            print(f"  发现 {len(output_len_cols)} 个 output_len 列（按序）: {output_len_cols}")

        # 遍历行，合并到 data 中
        for _, row in df.iterrows():
            req_id = str(row.get('req_id', '')).strip()
            if req_id == '':
                continue
            if req_id not in data:
                data[req_id] = {
                    'prompt': '',
                    'max_tokens': '',
                    'lengths': []
                }

            # prompt / max_tokens 只在当前为空时填充（你已确认不同文件相同 req_id 的 prompt/max_tokens 是相同）
            if data[req_id]['prompt'] == '' and 'prompt' in df.columns:
                v = row.get('prompt', '')
                if v is not None and str(v).strip() != '':
                    data[req_id]['prompt'] = str(v).strip()

            if data[req_id]['max_tokens'] == '' and 'max_tokens' in df.columns:
                v = row.get('max_tokens', '')
                if v is not None and str(v).strip() != '':
                    data[req_id]['max_tokens'] = str(v).strip()

            # 取出 output_len 列值并按顺序 append（保留原始字符串，空值保 ''）
            for c in output_len_cols:
                val = row.get(c)
                if pd.notna(val) and str(val).strip() != '':
                    data[req_id]['lengths'].append(str(val).strip())
                else:
                    data[req_id]['lengths'].append('')

            # 更新最大长度计数
            if len(data[req_id]['lengths']) > max_len_count:
                max_len_count = len(data[req_id]['lengths'])

    if not data:
        print("没有合并到任何 req_id（可能所有文件都没有 output_len 列或都为空）。", file=sys.stderr)
        sys.exit(1)

    # 统一列名： output_len1 ... output_len{max_len_count}
    out_columns = ['req_id', 'prompt', 'max_tokens']
    out_columns += [f"output_len{i}" for i in range(1, max_len_count + 1)]

    # 构造输出数据行
    rows = []
    for req_id, info in data.items():
        lengths = info['lengths']
        # pad to max_len_count
        if len(lengths) < max_len_count:
            lengths = lengths + [''] * (max_len_count - len(lengths))
        row = [req_id, info.get('prompt') or '', info.get('max_tokens') or ''] + lengths
        rows.append(row)

    out_df = pd.DataFrame(rows, columns=out_columns)

    # 创建输出目录
    # ---- 自动前缀替换：如果来自 llama 或 qwen 路径，则自动改 prefix ----
    auto_prefix = args.prefix  # 默认使用命令行指定的 merged

    if args.input_dir:
        # 获取路径片段，用于匹配模型名
        parts = args.input_dir.replace("\\", "/").split("/")
        # 常见模型关键字列表，可自行扩展
        model_keys = ["llama", "qwen", "deepseek", "mistral"]

        for key in model_keys:
            for p in parts:
                if key.lower() == p.lower():
                    auto_prefix = key   # 例如 "llama"
                    break

    # ------------------------------------------------------------
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_name = f"{auto_prefix}_{ts}.csv"

    out_path = os.path.join(args.out_dir, out_name)
    os.makedirs(args.out_dir, exist_ok=True)
    # ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    # out_name = f"{args.prefix}_{ts}.csv"
    # out_path = os.path.join(args.out_dir, out_name)

    # 保存为带 BOM 的 UTF-8（方便 Excel 在 Windows 下直接打开）
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\n已合并 {len(input_files)} 个文件（实际处理 {len(ordered_files) if 'ordered_files' in locals() else len(input_files)} 个），生成 {len(out_df)} 条唯一 req_id 的记录。")
    print(f"输出文件： {out_path}")
    if args.verbose:
        print(f"包含列： {out_columns}")

if __name__ == "__main__":
    main()
