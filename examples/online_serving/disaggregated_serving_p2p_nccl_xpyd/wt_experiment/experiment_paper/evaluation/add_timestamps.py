#!/usr/bin/env python3
"""
为 evaluation_dataset 中的评估数据集添加 timestamp 列。

读取 select/ 目录下各 trace 类型的原始时间戳，按照 trace_utils.py 中的逻辑
统一转换为毫秒，然后写入 evaluation_dataset/ 下对应 trace 类型子目录中的
所有 CSV 数据集（llama 和 qwen）。

使用方式:
    python add_timestamps.py [--dry-run]
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path


# ── 时间戳解析逻辑（与 trace_utils.py 保持一致） ──────────────────────

def _parse_float_like(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _infer_numeric_timestamp_scale_to_ms(values: list[float]) -> float:
    finite_nonzero = [abs(v) for v in values if math.isfinite(v) and v != 0.0]
    if not finite_nonzero:
        return 1.0
    finite_nonzero.sort()
    probe = finite_nonzero[len(finite_nonzero) // 2]

    if probe >= 1e17:
        return 1.0 / 1_000_000.0   # ns → ms
    if probe >= 1e14:
        return 1.0 / 1_000.0       # μs → ms
    if probe >= 1e11:
        return 1.0                  # already ms
    if probe >= 1e8:
        return 1_000.0             # epoch seconds → ms
    return 1_000.0 if probe < 1e4 else 1.0


def parse_azure_timestamp(value: str) -> float:
    """Azure ISO 8601 字符串 → 毫秒"""
    text = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp() * 1000.0


def parse_burstgpt_timestamp(value) -> float:
    """BurstGPT 秒数 → 毫秒"""
    seconds = _parse_float_like(value)
    if seconds is None:
        raise ValueError(f"invalid burstgpt timestamp: {value}")
    return seconds * 1000.0


# ── 各 trace 类型的时间戳提取 ─────────────────────────────────────────

def load_azure_timestamps(trace_path: Path) -> list[float]:
    """从 Azure CSV 中提取 TIMESTAMP 列，转为毫秒"""
    timestamps_ms = []
    with trace_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_ms = parse_azure_timestamp(row["TIMESTAMP"])
            timestamps_ms.append(ts_ms)
    return timestamps_ms


def load_burstgpt_timestamps(trace_path: Path) -> list[float]:
    """从 BurstGPT CSV 中提取 Timestamp 列，转为毫秒"""
    timestamps_ms = []
    with trace_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_ms = parse_burstgpt_timestamp(row["Timestamp"])
            timestamps_ms.append(ts_ms)
    return timestamps_ms


def load_qwen_timestamps(trace_path: Path) -> list[float]:
    """从 qwen JSONL 中提取 timestamp，自动推断单位并转为毫秒"""
    raw_timestamps = []
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ts = _parse_float_like(obj.get("timestamp"))
            if ts is not None:
                raw_timestamps.append(ts)

    scale = _infer_numeric_timestamp_scale_to_ms(raw_timestamps)
    return [ts * scale for ts in raw_timestamps]


# ── 格式化输出（与 trace_utils._format_scaled_timestamp 保持一致） ────

def format_timestamp_for_output(trace_type: str, timestamp_ms: float):
    """将毫秒时间戳统一输出为整数毫秒。

    所有 trace 类型统一使用毫秒作为输出单位，确保
    benchmark_serving_trace.py 的 trace 模式 (trace_timestamp_scale=1e-3)
    可以正确处理。
    """
    return int(round(timestamp_ms))


# ── 给 CSV 添加 timestamp 列 ─────────────────────────────────────────

def add_timestamp_to_csv(csv_path: Path, timestamps_ms: list[float], trace_type: str, dry_run: bool = False):
    """读取 CSV，添加 timestamp 列后覆盖写回"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    n_rows = len(df)
    n_ts = len(timestamps_ms)

    if n_rows != n_ts:
        raise ValueError(
            f"行数不匹配: {csv_path.name} 有 {n_rows} 行, "
            f"但 trace 有 {n_ts} 个时间戳"
        )

    # 格式化时间戳
    formatted = [format_timestamp_for_output(trace_type, ts) for ts in timestamps_ms]

    # 将 timestamp 插入为第二列 (req_id 之后)
    if "timestamp" in df.columns:
        df["timestamp"] = formatted
        print(f"  [更新] {csv_path.name}: 覆盖已有 timestamp 列")
    else:
        df.insert(1, "timestamp", formatted)
        print(f"  [添加] {csv_path.name}: 插入 timestamp 列 ({n_rows} 行)")

    if not dry_run:
        df.to_csv(csv_path, index=False)


# ── 主逻辑 ────────────────────────────────────────────────────────────

BASE = Path("/root/predict-schedule/vllm/examples/online_serving/"
            "disaggregated_serving_p2p_nccl_xpyd/wt_experiment/dataset")

SELECT_DIR = BASE / "select"
EVAL_DIR = BASE / "evaluation_dataset"

# trace 类型 → (trace 文件路径, 时间戳加载函数)
TRACE_CONFIG = {
    "azure": {
        "trace_file": SELECT_DIR / "azure" / "AzureLLMInferenceTrace_code_1week_ts20240511T064500009930Z_20240511T065000009930Z.csv",
        "loader": load_azure_timestamps,
    },
    "burstgpt": {
        "trace_file": SELECT_DIR / "burstgpt" / "BurstGPT_without_fails_1_ts2978278_2979278.csv",
        "loader": load_burstgpt_timestamps,
    },
    "qwen": {
        "trace_file": SELECT_DIR / "qwen" / "qwen_traceB_blksz_16_x0p25_ts7600000_7900000.jsonl",
        "loader": load_qwen_timestamps,
    },
}


def main():
    parser = argparse.ArgumentParser(description="为评估数据集添加 timestamp 列")
    parser.add_argument("--dry-run", action="store_true", help="仅打印，不实际写入")
    args = parser.parse_args()

    for trace_type, config in TRACE_CONFIG.items():
        trace_file = config["trace_file"]
        loader = config["loader"]

        print(f"\n{'='*60}")
        print(f"处理 trace 类型: {trace_type}")
        print(f"  trace 文件: {trace_file.name}")

        if not trace_file.exists():
            print(f"  [跳过] trace 文件不存在: {trace_file}")
            continue

        timestamps_ms = loader(trace_file)
        print(f"  加载 {len(timestamps_ms)} 个时间戳")
        if timestamps_ms:
            print(f"  时间戳范围: {min(timestamps_ms):.3f} ~ {max(timestamps_ms):.3f} ms")

        # 遍历 llama 和 qwen 两个模型目录下的对应 trace 子目录
        for model_dir_name in ("llama", "qwen"):
            target_dir = EVAL_DIR / model_dir_name / trace_type
            if not target_dir.exists():
                print(f"  [跳过] 目标目录不存在: {target_dir}")
                continue

            csv_files = sorted(target_dir.glob("*.csv"))
            if not csv_files:
                print(f"  [跳过] {model_dir_name}/{trace_type} 无 CSV 文件")
                continue

            print(f"\n  目标: {model_dir_name}/{trace_type}/ ({len(csv_files)} 个文件)")
            for csv_path in csv_files:
                try:
                    add_timestamp_to_csv(csv_path, timestamps_ms, trace_type, dry_run=args.dry_run)
                except Exception as e:
                    print(f"  [错误] {csv_path.name}: {e}", file=sys.stderr)

    print(f"\n{'='*60}")
    print("完成!" + (" (dry-run 模式，未实际写入)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
