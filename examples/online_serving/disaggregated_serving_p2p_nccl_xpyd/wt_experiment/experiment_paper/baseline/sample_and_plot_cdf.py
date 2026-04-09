import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT_CSV = (
    "/root/.cache/huggingface/hub/datasets/wt_predictor/dataset/"
    "llama_dataset/lmsys-chat/processed_output/llama-lmsys-chat.csv"
)
DEFAULT_BASELINE_DIR = (
    "/root/predict-schedule/vllm/examples/online_serving/"
    "disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/baseline"
)

INPUT_LEN_CANDIDATES = [
    "prompt_len",
    "input_len",
    "input_length",
    "ContextTokens",
    "Request tokens",
    "PromptTokens",
]
OUTPUT_LEN_CANDIDATES = [
    "output_tokens",
    "output_len",
    "output_length",
    "GeneratedTokens",
    "Response tokens",
    "CompletionTokens",
]
REQ_ID_CANDIDATES = [
    "req_id",
    "request_id",
    "ReqId",
    "RequestId",
    "id",
]

LLAMA_TO_QWEN_UPDATED_PAIR = {
    "/root/.cache/huggingface/hub/datasets/wt_predictor/dataset/llama_dataset/lmsys-chat/processed_output/llama-lmsys-chat-updated.csv":
    "/root/.cache/huggingface/hub/datasets/wt_predictor/dataset/qwen_dataset/lmsys-chat/processed_output/qwen-lmsys-chat-updated.csv",
    "/root/.cache/huggingface/hub/datasets/wt_predictor/dataset/llama_dataset/mysharegpt/processed_output/llama-mysharegpt-updated.csv":
    "/root/.cache/huggingface/hub/datasets/wt_predictor/dataset/qwen_dataset/mysharegpt/processed_output/qwen-mysharegpt-updated.csv",
}


def _count_csv_rows(file_path: Path, chunksize: int) -> int:
    # Count logical CSV records via pandas parser to handle embedded newlines in quoted fields.
    total_rows = 0
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        total_rows += len(chunk)
    return total_rows


def _pick_column(columns, preferred, user_value, kind):
    if user_value:
        if user_value not in columns:
            raise ValueError(f"指定的{kind}列不存在: {user_value}")
        return user_value

    for name in preferred:
        if name in columns:
            return name

    raise ValueError(
        f"无法自动识别{kind}列。可用列: {columns}，请通过参数显式指定。"
    )


def _normalize_req_id_series(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    numeric = pd.to_numeric(series, errors="coerce")
    int_like_mask = numeric.notna() & (numeric % 1 == 0)
    int_like_text = numeric.astype("Int64").astype("string")
    text = text.mask(int_like_mask, int_like_text)
    return text.fillna("")


def _infer_paired_qwen_csv(input_csv: Path):
    input_str = str(input_csv)
    if input_str in LLAMA_TO_QWEN_UPDATED_PAIR:
        return Path(LLAMA_TO_QWEN_UPDATED_PAIR[input_str])

    if ("/llama_dataset/" not in input_str) or (not input_csv.name.endswith("-updated.csv")):
        return None

    candidate_str = input_str.replace("/llama_dataset/", "/qwen_dataset/")
    candidate = Path(candidate_str)
    if candidate.name.startswith("llama-"):
        candidate = candidate.with_name("qwen-" + candidate.name[len("llama-"):])

    if candidate.exists():
        return candidate
    return None


def _sample_rows_by_req_ids(
    file_path: Path,
    req_id_col: str,
    req_id_set,
    chunksize: int,
):
    matched_chunks = []
    matched_req_ids = set()

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        req_ids_norm = _normalize_req_id_series(chunk[req_id_col])
        mask = req_ids_norm.isin(req_id_set)
        if mask.any():
            matched_chunks.append(chunk.loc[mask].copy())
            matched_req_ids.update(req_ids_norm.loc[mask].tolist())

    if not matched_chunks:
        return pd.DataFrame(), matched_req_ids
    return pd.concat(matched_chunks, ignore_index=True), matched_req_ids


def _extract_valid_lengths(df: pd.DataFrame, input_len_col: str, output_len_col: str):
    input_lengths = pd.to_numeric(df[input_len_col], errors="coerce")
    output_lengths = pd.to_numeric(df[output_len_col], errors="coerce")
    valid_mask = (~input_lengths.isna()) & (~output_lengths.isna())

    valid_input = input_lengths[valid_mask].to_numpy(dtype=float, copy=False)
    valid_output = output_lengths[valid_mask].to_numpy(dtype=float, copy=False)
    return valid_input, valid_output, int(valid_mask.sum())


def _sample_rows_from_csv(file_path: Path, selected_indices: np.ndarray, chunksize: int) -> pd.DataFrame:
    sampled_chunks = []
    selected_sorted = np.sort(selected_indices)
    global_start = 0

    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        global_end = global_start + len(chunk)
        left = np.searchsorted(selected_sorted, global_start, side="left")
        right = np.searchsorted(selected_sorted, global_end, side="left")

        if right > left:
            local_idx = selected_sorted[left:right] - global_start
            sampled_chunks.append(chunk.iloc[local_idx])

        global_start = global_end
        if right >= selected_sorted.size:
            break

    if not sampled_chunks:
        return pd.DataFrame()
    return pd.concat(sampled_chunks, ignore_index=True)


def _build_stats_lines(values, title):
    arr = np.asarray(values, dtype=float)
    quantiles = [0.5, 0.9, 0.95, 0.99]
    lines = [
        f"[{title}]",
        f"count: {arr.size}",
        f"min: {arr.min():.4f}",
        f"max: {arr.max():.4f}",
        f"mean: {arr.mean():.4f}",
    ]
    for q in quantiles:
        lines.append(f"p{int(q * 100)}: {np.quantile(arr, q):.4f}")
    lines.append("")
    return lines


def _build_cdf_curve(values):
    sorted_vals = np.sort(np.asarray(values, dtype=float))
    n = sorted_vals.size
    if n == 1:
        return sorted_vals, np.array([1.0])

    empirical_cdf = np.arange(1, n + 1, dtype=float) / n
    x_dense = np.linspace(sorted_vals[0], sorted_vals[-1], num=min(max(200, n * 10), 5000))
    y_dense = np.interp(x_dense, sorted_vals, empirical_cdf, left=0.0, right=1.0)
    return x_dense, y_dense


def main():
    parser = argparse.ArgumentParser(
        description="随机抽样CSV并输出输入/输出长度统计与CDF图（随机种子默认42）"
    )
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="输入CSV路径")
    parser.add_argument(
        "--sample-size",
        type=int,
        required=True,
        help="随机抽样数量（不放回）",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(DEFAULT_BASELINE_DIR) / "output"),
        help="输出目录（保存抽样CSV、统计txt、CDF图）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认42）",
    )
    parser.add_argument(
        "--input-len-col",
        default='prompt_len',
        help="输入长度列名，不传则自动识别",
    )
    parser.add_argument(
        "--output-len-col",
        default='output_tokens',
        help="输出长度列名，不传则自动识别",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=200000,
        help="分块读取CSV时的块大小",
    )
    parser.add_argument(
        "--paired-qwen-csv",
        default=None,
        help="可选：qwen侧CSV路径。不传则对 llama-*-updated.csv 自动推断",
    )
    parser.add_argument(
        "--req-id-col",
        default="req_id",
        help="llama侧请求ID列名（用于匹配qwen）",
    )
    parser.add_argument(
        "--qwen-req-id-col",
        default=None,
        help="qwen侧请求ID列名（默认与 --req-id-col 相同）",
    )
    parser.add_argument(
        "--qwen-input-len-col",
        default=None,
        help="qwen侧输入长度列名，不传则自动识别",
    )
    parser.add_argument(
        "--qwen-output-len-col",
        default=None,
        help="qwen侧输出长度列名，不传则自动识别",
    )
    parser.add_argument(
        "--disable-reqid-pairing",
        action="store_true",
        help="关闭按req_id联动提取qwen对应请求",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_size <= 0:
        raise ValueError("sample-size 必须为正整数")

    columns = pd.read_csv(input_csv, nrows=0).columns.tolist()
    input_len_col = _pick_column(columns, INPUT_LEN_CANDIDATES, args.input_len_col, "输入长度")
    output_len_col = _pick_column(columns, OUTPUT_LEN_CANDIDATES, args.output_len_col, "输出长度")

    total_rows = _count_csv_rows(input_csv, args.chunksize)
    if total_rows == 0:
        raise ValueError("输入CSV没有数据行")
    if args.sample_size > total_rows:
        raise ValueError(
            f"sample-size={args.sample_size} 大于数据集行数 {total_rows}，无法无放回抽样"
        )

    rng = np.random.default_rng(args.seed)
    selected_indices = rng.choice(total_rows, size=args.sample_size, replace=False)

    sampled_df = _sample_rows_from_csv(input_csv, selected_indices, args.chunksize)
    if sampled_df.empty:
        raise RuntimeError("抽样失败，未得到任何数据")

    valid_input, valid_output, valid_rows_for_stats = _extract_valid_lengths(
        sampled_df, input_len_col, output_len_col
    )

    if valid_input.size == 0 or valid_output.size == 0:
        raise ValueError("抽样数据中未找到有效的输入/输出长度数值")

    now_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = input_csv.stem
    sample_csv_path = output_dir / f"sampled_{stem}_n{args.sample_size}_seed{args.seed}_{now_tag}.csv"
    stats_path = output_dir / f"length_stats_{stem}_n{args.sample_size}_seed{args.seed}_{now_tag}.txt"
    cdf_path = output_dir / f"length_cdf_{stem}_n{args.sample_size}_seed{args.seed}_{now_tag}.png"

    sampled_df.to_csv(sample_csv_path, index=False)

    stats_lines = [
        f"input_csv: {input_csv}",
        f"sample_size: {args.sample_size}",
        f"total_rows: {total_rows}",
        f"seed: {args.seed}",
        f"input_len_col: {input_len_col}",
        f"output_len_col: {output_len_col}",
        f"valid_rows_for_stats: {valid_rows_for_stats}",
        "",
    ]
    stats_lines.extend(_build_stats_lines(valid_input, "input_len"))
    stats_lines.extend(_build_stats_lines(valid_output, "output_len"))
    stats_path.write_text("\n".join(stats_lines), encoding="utf-8")

    x_in, y_in = _build_cdf_curve(valid_input)
    x_out, y_out = _build_cdf_curve(valid_output)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5.5))
    ax.plot(x_in, y_in, label="input_len", color="#1f77b4", linewidth=2.2)
    ax.plot(x_out, y_out, label="output_len", color="#ff7f0e", linewidth=2.2)
    ax.set_xlabel("Length", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(cdf_path, dpi=300)

    print(f"抽样CSV已保存: {sample_csv_path}")
    print(f"统计TXT已保存: {stats_path}")
    print(f"CDF图已保存: {cdf_path}")

    if args.disable_reqid_pairing:
        return

    paired_qwen_csv = Path(args.paired_qwen_csv) if args.paired_qwen_csv else _infer_paired_qwen_csv(input_csv)
    if paired_qwen_csv is None:
        print("未找到自动配对的qwen CSV，已跳过req_id联动统计")
        return
    if not paired_qwen_csv.exists():
        raise FileNotFoundError(f"qwen CSV不存在: {paired_qwen_csv}")

    req_id_col = _pick_column(columns, REQ_ID_CANDIDATES, args.req_id_col, "llama请求ID")
    sampled_req_ids = _normalize_req_id_series(sampled_df[req_id_col])
    sampled_req_id_set = set(sampled_req_ids[sampled_req_ids != ""].tolist())
    if not sampled_req_id_set:
        raise ValueError("llama抽样结果中没有可用req_id，无法进行qwen联动统计")

    qwen_columns = pd.read_csv(paired_qwen_csv, nrows=0).columns.tolist()
    qwen_req_id_col = _pick_column(
        qwen_columns,
        REQ_ID_CANDIDATES,
        args.qwen_req_id_col if args.qwen_req_id_col else args.req_id_col,
        "qwen请求ID",
    )
    qwen_input_len_col = _pick_column(
        qwen_columns,
        INPUT_LEN_CANDIDATES,
        args.qwen_input_len_col,
        "qwen输入长度",
    )
    qwen_output_len_col = _pick_column(
        qwen_columns,
        OUTPUT_LEN_CANDIDATES,
        args.qwen_output_len_col,
        "qwen输出长度",
    )

    qwen_matched_df, matched_req_ids = _sample_rows_by_req_ids(
        paired_qwen_csv,
        qwen_req_id_col,
        sampled_req_id_set,
        args.chunksize,
    )
    if qwen_matched_df.empty:
        raise RuntimeError("qwen侧没有匹配到任何req_id")

    qwen_valid_input, qwen_valid_output, qwen_valid_rows_for_stats = _extract_valid_lengths(
        qwen_matched_df,
        qwen_input_len_col,
        qwen_output_len_col,
    )
    if qwen_valid_input.size == 0 or qwen_valid_output.size == 0:
        raise ValueError("qwen匹配数据中未找到有效的输入/输出长度数值")

    qwen_stem = paired_qwen_csv.stem
    qwen_sample_csv_path = output_dir / (
        f"matched_{qwen_stem}_by_reqid_from_{stem}_n{args.sample_size}_seed{args.seed}_{now_tag}.csv"
    )
    qwen_stats_path = output_dir / (
        f"length_stats_{qwen_stem}_by_reqid_from_{stem}_n{args.sample_size}_seed{args.seed}_{now_tag}.txt"
    )
    qwen_cdf_path = output_dir / (
        f"length_cdf_{qwen_stem}_by_reqid_from_{stem}_n{args.sample_size}_seed{args.seed}_{now_tag}.png"
    )

    qwen_matched_df.to_csv(qwen_sample_csv_path, index=False)

    missing_req_id_count = len(sampled_req_id_set - matched_req_ids)
    qwen_stats_lines = [
        f"qwen_csv: {paired_qwen_csv}",
        f"source_llama_csv: {input_csv}",
        f"sample_size_on_llama: {args.sample_size}",
        f"seed: {args.seed}",
        f"llama_req_id_col: {req_id_col}",
        f"qwen_req_id_col: {qwen_req_id_col}",
        f"qwen_input_len_col: {qwen_input_len_col}",
        f"qwen_output_len_col: {qwen_output_len_col}",
        f"sampled_unique_req_ids_on_llama: {len(sampled_req_id_set)}",
        f"matched_unique_req_ids_on_qwen: {len(matched_req_ids)}",
        f"missing_req_ids_on_qwen: {missing_req_id_count}",
        f"matched_rows_on_qwen: {len(qwen_matched_df)}",
        f"valid_rows_for_stats: {qwen_valid_rows_for_stats}",
        "",
    ]
    qwen_stats_lines.extend(_build_stats_lines(qwen_valid_input, "input_len"))
    qwen_stats_lines.extend(_build_stats_lines(qwen_valid_output, "output_len"))
    qwen_stats_path.write_text("\n".join(qwen_stats_lines), encoding="utf-8")

    qx_in, qy_in = _build_cdf_curve(qwen_valid_input)
    qx_out, qy_out = _build_cdf_curve(qwen_valid_output)
    qfig, qax = plt.subplots(1, 1, figsize=(7, 5.5))
    qax.plot(qx_in, qy_in, label="input_len", color="#1f77b4", linewidth=2.2)
    qax.plot(qx_out, qy_out, label="output_len", color="#ff7f0e", linewidth=2.2)
    qax.set_xlabel("Length", fontsize=12)
    qax.set_ylabel("CDF", fontsize=12)
    qax.set_ylim(0, 1.0)
    qax.grid(True, linestyle="--", alpha=0.6)
    qax.legend(loc="upper right")
    qfig.tight_layout()
    qfig.savefig(qwen_cdf_path, dpi=300)

    print(f"qwen匹配CSV已保存: {qwen_sample_csv_path}")
    print(f"qwen统计TXT已保存: {qwen_stats_path}")
    print(f"qwen CDF图已保存: {qwen_cdf_path}")


if __name__ == "__main__":
    main()