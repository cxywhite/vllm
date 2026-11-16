#!/usr/bin/env python3
"""Summarize existing RR, Ours, and Dynamo experiment results."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOTS = {
    "result_0630": Path("/root/.cache/huggingface/hub/wt_predictor/0630/result_0630"),
    "dynamo_0630": Path("/root/.cache/huggingface/hub/wt_predictor/0630/dynamo_0630"),
    "dynamo_retry3": Path("/root/myshare/tmp_dynamo_result/retry3"),
    "rr_ours_retry3": Path("/root/predict-schedule/experiment/zrecent_use/0630/retry_test/retry3"),
    "organized_llama_rr": Path("/root/myshare/0802_wty_result/4080_result_llama/rr"),
    "organized_llama_ours": Path("/root/myshare/0802_wty_result/4080_result_llama/ours"),
    "organized_llama_dynamo": Path("/root/myshare/0802_wty_result/4080_result_llama/dynamo"),
}
OUTPUT_DIR = Path("/root/predict-schedule/experiment/zrecent_use/0630")
CSV_PATH = OUTPUT_DIR / "existing_experiment_matrix.csv"
MD_PATH = OUTPUT_DIR / "existing_experiment_summary.md"
ATTEMPTS_CSV_PATH = OUTPUT_DIR / "failed_experiment_attempts.csv"
TOKEN_TOLERANCE = 0.01
MAIN_TPOT_SLO_MS = 30
TRACES = ("burstgpt", "mooncake", "qwen")
MODELS = ("llama", "qwen")
DATASETS = ("lmsys", "sharegpt")
PD_RATIOS = ("2p6d", "4p4d")
METHODS = ("RR", "Ours", "Dynamo")
REQUESTS_BY_TRACE = {"burstgpt": 673, "mooncake": 1790, "qwen": 4080}
ATTEMPT_SUMMARIES = (
    OUTPUT_DIR / "1tmp/summary_20260727_161914.csv",
    OUTPUT_DIR / "1tmp/summary_20260727_162919.csv",
    OUTPUT_DIR / "experiment_result_2/summary_20260728_235006.csv",
)
SUPPLEMENTARY_CSV_PATH = OUTPUT_DIR / "supplementary_experiment_matrix.csv"
SUPPLEMENTARY_SOURCES = (
    ("1p1d_routing", OUTPUT_DIR / "experiment_1p1d_compare/success"),
    ("1p1d_routing", Path("/root/predict-schedule/experiment/wt_experiment/experiment_paper/evaluation")),
    ("scheduling_ablation", Path("/root/predict-schedule/experiment/wt_experiment/experiment_paper/evaluation/experiment_result_2")),
)
EXTRA_FAILED_DIRS = (OUTPUT_DIR / "retry_fail",)
SUPP_1P1D_RE = re.compile(
    r"^llama_(?P<strategy>[a-z_]+?)_llama_(?:lmsys_)?(?:1p1d_)?"
    r"(?P<workload>1k|best500_predict|400)_pd(?P<pd>\d+p\d+d)_(?P<timestamp>\d{8}_\d{6})$"
)
SUPP_2P6D_RE = re.compile(
    r"^llama_llama-lmsys-chat_n(?P<requests>\d+)_pd(?P<pd>\d+p\d+d)_tpot(?P<slo>\d+)ms"
    r"_rep(?P<rep>\d+)_(?P<timestamp>\d{8}_\d{6})_(?P<strategy>.+)$"
)


@dataclass
class Result:
    source: str
    path: Path
    model: str
    dataset: str
    trace: str
    requests_expected: int
    pd_ratio: str
    tpot_slo_ms: int
    rep: int
    method: str
    timestamp: str
    status: str
    successful_requests: int = 0
    output_tokens: int = 0
    duration_s: float = 0.0
    request_goodput: float = 0.0
    output_goodput: float = 0.0
    output_throughput: float = 0.0
    mean_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    mean_tpot_ms: float = 0.0
    p99_tpot_ms: float = 0.0

    @property
    def key(self) -> tuple[str, str, str, int, str, int]:
        return (self.trace, self.model, self.dataset, self.requests_expected,
                self.pd_ratio, self.tpot_slo_ms)


NAME_RE = re.compile(
    r"^(llama|qwen)_(?:llama|qwen)-(lmsys|mysharegpt)(?:-chat)?-(.+?)"
    r"_n(\d+)_pd(\d+p\d+d)_tpot(\d+)ms_trace(.+?)_rep(\d+)"
    r"(?:_(\d{8}_\d{6}))?_(roundrobin|scheme4_gwf|dynamo)$"
)


def metric(text: str, label: str) -> float:
    matches = re.findall(rf"{re.escape(label)}\s*([0-9]+(?:\.[0-9]+)?)", text)
    return float(matches[-1]) if matches else 0.0


def benchmark_log(path: Path) -> Path | None:
    candidates = list((path / "log").glob("bench*.log")) if (path / "log").is_dir() else []
    candidates.extend(path.glob("benchmark.log"))
    for candidate in sorted(candidates):
        text = candidate.read_text(errors="ignore")
        if "Serving Benchmark Result" in text:
            return candidate
    return None


def parse_result(source: str, path: Path) -> Result | None:
    match = NAME_RE.match(path.name)
    if not match:
        return None
    model, dataset, _, requests, pd_ratio, slo, trace, rep, timestamp, suffix = match.groups()
    requests_expected = int(requests)
    dataset = "sharegpt" if dataset == "mysharegpt" else dataset
    if requests_expected == 4080 or "aggressive" in trace.lower():
        trace = "qwen"
    method = {"roundrobin": "RR", "scheme4_gwf": "Ours", "dynamo": "Dynamo"}[suffix]
    log = benchmark_log(path)
    if log is None:
        return Result(source, path, model, dataset, trace, requests_expected, pd_ratio,
                      int(slo), int(rep), method, timestamp or "", "failed/incomplete")
    text = log.read_text(errors="ignore")
    successful = int(metric(text, "Successful requests:"))
    return Result(
        source=source,
        path=path,
        model=model,
        dataset=dataset,
        trace=trace,
        requests_expected=requests_expected,
        pd_ratio=pd_ratio,
        tpot_slo_ms=int(slo),
        rep=int(rep),
        method=method,
        timestamp=timestamp or "",
        status="success" if successful else "failed/incomplete",
        successful_requests=successful,
        output_tokens=int(metric(text, "Total generated tokens:")),
        duration_s=metric(text, "Benchmark duration (s):"),
        request_goodput=metric(text, "Request goodput (req/s):"),
        output_goodput=metric(text, "Output token goodput (tok/s):"),
        output_throughput=metric(text, "Output token throughput (tok/s):"),
        mean_ttft_ms=metric(text, "Mean TTFT (ms):"),
        p99_ttft_ms=metric(text, "P99 TTFT (ms):"),
        mean_tpot_ms=metric(text, "Mean TPOT (ms):"),
        p99_tpot_ms=metric(text, "P99 TPOT (ms):"),
    )


def collect() -> list[Result]:
    results: list[Result] = []
    for source, root in ROOTS.items():
        if not root.is_dir():
            continue
        if source in {"result_0630", "dynamo_0630"}:
            candidates = [
                candidate
                for slo_dir in root.glob("tpot*") if slo_dir.is_dir()
                for candidate in slo_dir.iterdir() if candidate.is_dir()
            ]
            candidates += [
                candidate
                for extra_dir in root.glob("dynamo_tmp") if extra_dir.is_dir()
                for candidate in extra_dir.iterdir() if candidate.is_dir()
            ]
        elif source in {"dynamo_retry3", "organized_llama_dynamo"}:
            candidates = [
                candidate
                for run_dir in root.glob("run_*") if run_dir.is_dir()
                for candidate in run_dir.iterdir() if candidate.is_dir()
            ]
        else:
            candidates = [candidate for candidate in root.iterdir() if candidate.is_dir()]
        for candidate in candidates:
            result = parse_result(source, candidate)
            if result:
                results.append(result)
    return results


def rank(result: Result) -> tuple[int, str, int, float]:
    return (result.status == "success", result.timestamp, result.rep,
            result.path.stat().st_mtime)


def select_latest(results: list[Result]) -> dict[tuple, Result]:
    selected: dict[tuple, Result] = {}
    for result in results:
        key = (*result.key, result.method)
        if key not in selected or rank(result) > rank(selected[key]):
            selected[key] = result
    return selected


def pct(new: float, old: float) -> float | None:
    return (new / old - 1.0) * 100.0 if old else None


def fmt(value: float | None, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def token_match(reference: Result, other: Result) -> bool:
    if not reference.output_tokens or not other.output_tokens:
        return False
    return abs(other.output_tokens / reference.output_tokens - 1.0) <= TOKEN_TOLERANCE


def write_csv(results: list[Result]) -> None:
    fields = [
        "source", "trace", "model", "dataset", "requests_expected", "pd_ratio",
        "tpot_slo_ms", "method", "rep", "timestamp", "status",
        "successful_requests", "output_tokens", "duration_s", "request_goodput_req_s",
        "output_goodput_tok_s", "output_throughput_tok_s", "mean_ttft_ms",
        "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms", "result_path",
    ]
    with CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in sorted(results, key=lambda item: (item.key, item.method, item.source, rank(item))):
            writer.writerow({
                "source": result.source,
                "trace": result.trace,
                "model": result.model,
                "dataset": result.dataset,
                "requests_expected": result.requests_expected,
                "pd_ratio": result.pd_ratio,
                "tpot_slo_ms": result.tpot_slo_ms,
                "method": result.method,
                "rep": result.rep,
                "timestamp": result.timestamp,
                "status": result.status,
                "successful_requests": result.successful_requests,
                "output_tokens": result.output_tokens,
                "duration_s": result.duration_s,
                "request_goodput_req_s": result.request_goodput,
                "output_goodput_tok_s": result.output_goodput,
                "output_throughput_tok_s": result.output_throughput,
                "mean_ttft_ms": result.mean_ttft_ms,
                "p99_ttft_ms": result.p99_ttft_ms,
                "mean_tpot_ms": result.mean_tpot_ms,
                "p99_tpot_ms": result.p99_tpot_ms,
                "result_path": str(result.path),
            })


def write_attempts_csv() -> list[dict[str, str]]:
    attempts: list[dict[str, str]] = []
    for summary_path in ATTEMPT_SUMMARIES:
        if not summary_path.is_file():
            continue
        with summary_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                row["summary_path"] = str(summary_path)
                attempts.append(row)
    fields = ["summary_path"]
    fields.extend(key for row in attempts for key in row if key not in fields)
    with ATTEMPTS_CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(attempts)
    return attempts


def supplementary_metrics(log: Path) -> dict[str, float]:
    text = log.read_text(errors="ignore")
    return {
        "successful_requests": metric(text, "Successful requests:"),
        "output_tokens": metric(text, "Total generated tokens:"),
        "output_goodput": metric(text, "Output token goodput (tok/s):"),
        "output_throughput": metric(text, "Output token throughput (tok/s):"),
        "mean_ttft_ms": metric(text, "Mean TTFT (ms):"),
        "mean_tpot_ms": metric(text, "Mean TPOT (ms):"),
    }


def collect_supplementary() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for group, root in SUPPLEMENTARY_SOURCES:
        if not root.is_dir():
            continue
        for candidate in sorted(root.rglob("*")):
            if not candidate.is_dir() or candidate.name in seen:
                continue
            log = benchmark_log(candidate)
            if log is None:
                continue
            match = SUPP_1P1D_RE.match(candidate.name)
            if match:
                workflow, label = "1p1d", match.group("workload")
                pd_ratio, slo, strategy = match.group("pd"), 30, match.group("strategy")
                requests = 0
            else:
                match = SUPP_2P6D_RE.match(candidate.name)
                if not match:
                    continue
                workflow, label = "2p6d", "lmsys-n1790"
                pd_ratio = match.group("pd")
                slo = int(match.group("slo"))
                strategy = match.group("strategy")
                requests = int(match.group("requests"))
            seen.add(candidate.name)
            metrics = supplementary_metrics(log)
            if not metrics["successful_requests"]:
                continue
            records.append({
                "group": "scheduling_ablation" if workflow == "2p6d" else "1p1d_routing",
                "workflow": workflow,
                "strategy": strategy,
                "label": label,
                "pd_ratio": pd_ratio,
                "tpot_slo_ms": slo,
                "requests_expected": requests or int(metrics["successful_requests"]),
                **metrics,
                "result_path": str(candidate),
                "log_path": str(log),
            })
    return records


def write_supplementary_csv(records: list[dict[str, object]]) -> None:
    fields = [
        "group", "workflow", "strategy", "label", "pd_ratio", "tpot_slo_ms",
        "requests_expected", "successful_requests", "output_tokens", "output_goodput",
        "output_throughput", "mean_ttft_ms", "mean_tpot_ms", "result_path", "log_path",
    ]
    with SUPPLEMENTARY_CSV_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (item["workflow"], item["strategy"], item["label"])):
            writer.writerow({key: record.get(key, "") for key in fields})


def extra_failed_count() -> int:
    skip = {"config", "log", "dataset_result", "__pycache__"}
    return sum(
        1
        for root in EXTRA_FAILED_DIRS if root.is_dir()
        for candidate in root.iterdir()
        if candidate.is_dir() and candidate.name not in skip
    )


def supplementary_rows(records: list[dict[str, object]], workflow: str) -> list[str]:
    return [
        f"| {record['strategy']} | {record['label']} | {record['pd_ratio']} | {record['tpot_slo_ms']} | "
        f"{int(record['successful_requests'])} | {int(record['output_tokens'])} | "
        f"{fmt(float(record['output_goodput']))} | {fmt(float(record['output_throughput']))} | "
        f"{fmt(float(record['mean_ttft_ms']))} | {fmt(float(record['mean_tpot_ms']))} | {record['group']} |"
        for record in sorted(records, key=lambda item: (item["label"], item["strategy"]))
        if record["workflow"] == workflow
    ]


def write_markdown(results: list[Result], attempts: list[dict[str, str]],
                   supplementary: list[dict[str, object]]) -> None:
    selected = select_latest(results)
    methods_by_key: dict[tuple, dict[str, Result]] = defaultdict(dict)
    for (*key, method), result in selected.items():
        methods_by_key[tuple(key)][method] = result

    expected_keys = [
        (trace, model, dataset, REQUESTS_BY_TRACE[trace], pd_ratio, MAIN_TPOT_SLO_MS)
        for trace in TRACES
        for dataset in DATASETS
        for model in MODELS
        for pd_ratio in PD_RATIOS
    ]
    achieved: list[tuple] = []
    partial: list[tuple] = []
    missing: list[tuple] = []
    incomparable: list[tuple] = []
    rows: list[str] = []
    for key in expected_keys:
        methods = methods_by_key.get(key, {})
        rr, ours, dynamo = (methods.get(name) for name in METHODS)
        present = ",".join(methods)
        comparable = bool(rr and all(token_match(rr, item) for item in (ours, dynamo) if item))
        verdict = "incomplete"
        if rr and ours and dynamo:
            if not comparable:
                verdict = "incomparable-token-mismatch"
                incomparable.append(key)
            elif ours.output_goodput > rr.output_goodput and ours.output_goodput > dynamo.output_goodput:
                verdict = "achieved"
                achieved.append(key)
            else:
                verdict = "not-achieved"
        elif rr and ours:
            if not token_match(rr, ours):
                verdict = "incomparable-token-mismatch"
                incomparable.append(key)
            elif ours.output_goodput > rr.output_goodput:
                verdict = "partial-ours>rr"
                partial.append(key)
            missing.append((key, tuple(name for name in METHODS if name not in methods)))
        else:
            missing.append((key, tuple(name for name in METHODS if name not in methods)))
        trace, model, dataset, requests, pd_ratio, slo = key
        rows.append(
            f"| {trace} | {model} | {dataset} | {requests} | {pd_ratio} | {slo} | {present} | "
            f"{fmt(rr.output_goodput if rr else None)} | {fmt(ours.output_goodput if ours else None)} | "
            f"{fmt(dynamo.output_goodput if dynamo else None)} | "
            f"{fmt(pct(ours.output_goodput, rr.output_goodput) if ours and rr else None)}% | "
            f"{fmt(pct(ours.output_goodput, dynamo.output_goodput) if ours and dynamo else None)}% | {verdict} |"
        )

    source_counts = Counter((result.source, result.status) for result in results)
    total_success = sum(result.status == "success" for result in results)
    attempt_statuses = Counter(row.get("status", "unknown") for row in attempts)
    covered_slots = sum(
        method in methods_by_key.get(key, {})
        for key in expected_keys
        for method in METHODS
    )
    extra_slos = sorted({result.tpot_slo_ms for result in results if result.tpot_slo_ms != MAIN_TPOT_SLO_MS})
    lines = [
        "# 现有实验覆盖与结果总结",
        "",
        "## 范围与判据",
        "",
        "- 数据集类别统一为 `burstgpt`、`mooncake`、`qwen`；其中 `qwen` 类使用 `qwentrace`，所有请求数为 4080 或名称含 `aggressive` 的结果均归入该类。",
        "- workload 统一为 `lmsys`、`sharegpt`（原目录名 `mysharegpt` 归一为 `sharegpt`）。",
        "- 主矩阵固定为 TPOT=30 ms：3 类数据集 x 2 workload x 2 模型 x 2 PD x 3 方法，共 72 个方法槽位。",
        f"- 其他 TPOT SLO（{', '.join(map(str, extra_slos)) if extra_slos else '无'} ms）作为额外实验单独统计，不计入 72 个主矩阵槽位。",
        "- 核心比较指标为满足 TTFT <= 1000 ms 和指定 TPOT SLO 的 output-token goodput。",
        "- 仅当 RR、Ours、Dynamo 均成功，生成 token 与 RR 相差不超过 1%，且 Ours 同时超过两个基线时，判定为 `achieved`。",
        "- 重复实验在比较矩阵中选最新成功结果；所有 run 均保留在 CSV 明细中。",
        "",
        "## 数据盘点",
        "",
        f"- 共解析 {len(results)} 次实验；成功 {total_success} 次；失败或不完整 {len(results) - total_success} 次。",
        f"- TPOT=30 ms 主矩阵已覆盖 {covered_slots}/72 个方法槽位，尚缺 {72 - covered_slots} 个。",
    ]
    for source in ROOTS:
        lines.append(
            f"- `{source}`：成功 {source_counts[(source, 'success')]} 次，"
            f"失败或不完整 {source_counts[(source, 'failed/incomplete')]} 次。"
        )
    lines += [
        f"- 另发现 2026-07-27 至 2026-07-29 的 {len(attempts)} 次未纳入尝试："
        f"start_failed {attempt_statuses['start_failed']} 次，"
        f"benchmark_failed {attempt_statuses['benchmark_failed']} 次。",
        "- 这些尝试没有完整 benchmark 指标，不计入 57/72 的有效方法覆盖；详细记录单列在失败尝试 CSV。",
        f"- 另有 2026-08-25 的 {extra_failed_count()} 次失败重跑（`retry_fail`），同样不产生可比较指标。",
        f"- 2026-07-29 与 2026-07-05 的 {len(supplementary)} 组成功补充实验不属于 72 槽位主矩阵，单列在下方补充实验章节。",
    ]
    lines += [
        "",
        "## 对比矩阵",
        "",
        "| 数据集类别 | 模型 | Workload | 请求数 | PD | TPOT ms | 已有方法 | RR goodput | Ours goodput | Dynamo goodput | Ours vs RR | Ours vs Dynamo | 结论 |",
        "|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|",
        *rows,
        "",
        "## 补充实验结果",
        "",
        "以下实验在 2026-07-29 与 2026-07-05 产生，不属于 TPOT=30 ms 的 72 槽位主矩阵，单独列出以免与主矩阵混淆。goodput 口径为主矩阵一致：TTFT <= 1000 ms 且 TPOT 30 ms。",
        "",
    ]
    one_p_one_d = [record for record in supplementary if record["workflow"] == "1p1d"]
    scheduling = [record for record in supplementary if record["workflow"] == "2p6d"]
    if one_p_one_d:
        lines += [
            f"### 1p1d 路由策略消融（2026-07-29，成功 {len(one_p_one_d)} 组）",
            "",
            "| 策略 | Workload | PD | TPOT ms | 请求数 | 生成 token | goodput | throughput | Mean TTFT | Mean TPOT | 来源 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            *supplementary_rows(one_p_one_d, "1p1d"),
            "",
            "- 该批次为 1p1d 部署下的路由策略对比，与 2p6d/4p4d 主矩阵不是同一部署形态，因此不能直接填入 72 槽位。",
            "- `aggressive` 三个档位与 `default` 400 档位因 TTFT 超过 1000 ms，goodput 记为 0，仅保留 throughput 作为参考。",
            "- `predict_gwf` 在三个档位上均未被 TTFT 约束归零，是这批 1p1d 实验中最稳的策略。",
            "",
        ]
    if scheduling:
        lines += [
            f"### 调度策略消融（2026-07-05，mooncake-lmsys n1790 / 2p6d / 30 ms，成功 {len(scheduling)} 组）",
            "",
            "| 策略 | Workload | PD | TPOT ms | 请求数 | 生成 token | goodput | throughput | Mean TTFT | Mean TPOT | 来源 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
            *supplementary_rows(scheduling, "2p6d"),
            "",
            "- 这 4 组生成 token 均为 2767957，与主矩阵 `mooncake / llama / lmsys / 1790 / 2p6d / 30` 同源，可作为该配置下更早策略版本的对照。",
            "- 其中 `decode_predict_3Dload` 表现最好，但仍低于主矩阵中较新的 Ours 与 RR 结果。",
            "",
        ]
    lines += [
        "## 目标达成情况",
        "",
        f"- 完全达成：{len(achieved)} 组。",
        f"- 阶段性达成（Ours > RR，但缺 Dynamo）：{len(partial)} 组。",
        f"- token 不匹配、不能直接比较：{len(incomparable)} 组。",
        "",
    ]
    if achieved:
        lines += ["### 完全达成", ""]
        for key in achieved:
            lines.append(f"- `{' / '.join(map(str, key))}`")
        lines.append("")
    if partial:
        lines += ["### 阶段性达成", ""]
        for key in partial:
            lines.append(f"- `{' / '.join(map(str, key))}`：补跑 Dynamo 后才能形成完整结论。")
        lines.append("")
    lines += ["## 仍需补充的实验", ""]
    if missing:
        for key, absent in missing:
            lines.append(f"- `{' / '.join(map(str, key))}`：补充 {', '.join(absent)}。")
    else:
        lines.append("- 当前配置并集中没有缺失实验。")
    lines += [
        "",
        "## 补实验优先级",
        "",
        "1. 优先补齐上方 72 槽位主矩阵中缺失的方法。",
        "2. TPOT=50 ms 属于额外矩阵，待 30 ms 主矩阵完整后再统一补齐。",
        "3. 对 token 不匹配或失败的配置重跑，完成前不做性能结论。",
        "4. 最终论文使用的配置至少补到两次干净重复实验。",
        "",
        "## 产物",
        "",
        f"- 完整逐 run 表格：`{CSV_PATH}`",
        f"- 七月底失败尝试表格：`{ATTEMPTS_CSV_PATH}`",
        f"- 补充实验表格：`{SUPPLEMENTARY_CSV_PATH}`",
        f"- 本报告：`{MD_PATH}`",
    ]
    MD_PATH.write_text("\n".join(lines) + "\n")


def main() -> None:
    results = collect()
    write_csv(results)
    attempts = write_attempts_csv()
    supplementary = collect_supplementary()
    write_supplementary_csv(supplementary)
    write_markdown(results, attempts, supplementary)
    print(f"parsed={len(results)} attempts={len(attempts)} "
          f"supplementary={len(supplementary)} csv={CSV_PATH} md={MD_PATH}")


if __name__ == "__main__":
    main()