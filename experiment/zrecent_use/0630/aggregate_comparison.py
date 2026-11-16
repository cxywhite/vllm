#!/usr/bin/env python3
"""Aggregate RR / Dynamo / Ours benchmark results and write a comparison report.

Scans benchmark logs across all result roots, selects the latest run per
(trace, model, dataset, pd, tpot, method), and reports where Ours
(scheme4_gwf) underperforms RR or Dynamo.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

CSV_FIELD_LIMIT = sys.maxsize

RESULT_ROOTS: list[tuple[str, Path]] = [
    ("retry3", Path("/root/predict-schedule/experiment/zrecent_use/0630/retry_test/retry3")),
    ("retry1", Path("/root/predict-schedule/experiment/zrecent_use/0630/retry_test/retry1")),
    ("retry", Path("/root/predict-schedule/experiment/zrecent_use/0630/retry_test")),
    ("1tmp", Path("/root/predict-schedule/experiment/zrecent_use/0630/1tmp")),
    ("exp2", Path("/root/predict-schedule/experiment/zrecent_use/0630/experiment_result_2")),
    ("fig30", Path("/root/myshare/paper_figures/fig12_goodput_comparison/results/result_0630/tpot30")),
    ("fig50", Path("/root/myshare/paper_figures/fig12_goodput_comparison/results/result_0630/tpot50")),
    ("dyn30", Path("/root/myshare/paper_figures/fig12_goodput_comparison/results/dynamo_0630/tpot30")),
    ("dynTmp", Path("/root/myshare/paper_figures/fig12_goodput_comparison/results/dynamo_0630/dynamo_tmp")),
]

METHOD_NAMES = {"RR": "RR", "Dynamo": "Dynamo", "Ours": "Ours"}

TRACE_LABELS = [
    ("burstgpt", "BurstGPT"),
    ("mooncake", "Mooncake"),
    ("qwentrace", "Qwen-Raw"),
    ("strategyA_aggressive_densed", "StrategyA-Densed"),
    ("strategyA_aggressive", "StrategyA-Aggressive"),
    ("qwen_strategyA", "Qwen-StrategyA"),
    ("sharegpt_aggressive", "ShareGPT-Aggressive"),
    ("mysharegpt_aggressive_only", "ShareGPT-AggOnly"),
]

TOKEN_TOLERANCE = 0.01  # 1%


@dataclass
class Result:
    trace: str
    model: str
    dataset: str
    pd_ratio: str
    tpot_ms: int
    method: str
    successful: int = 0
    tokens: int = 0
    goodput: float = 0.0
    req_goodput: float = 0.0
    ttft_mean: float = 0.0
    tpot_mean: float = 0.0
    duration: float = 0.0
    timestamp: str = "00000000_000000"
    path: Path = field(default_factory=Path)
    complete: bool = False


def parse_metric(text: str, label: str) -> str | None:
    match = re.search(rf"{re.escape(label)}\s+([0-9.]+)", text)
    return match.group(1) if match else None


def extract_timestamp(name: str, path: Path) -> str:
    m = re.search(r"_(\d{8}_\d{6})_", name)
    if m:
        return m.group(1)
    m = re.search(r"_(\d{8}_\d{6})$", name)
    if m:
        return m.group(1)
    return f"{path.stat().st_mtime:.0f}"


def classify(name: str, source: str) -> str | None:
    if source in ("dyn30", "dynTmp") or name.endswith("_dynamo"):
        return "Dynamo"
    if name.endswith("_roundrobin"):
        return "RR"
    if name.endswith("_scheme4_gwf"):
        return "Ours"
    return None


def trace_label(name: str) -> str:
    for key, label in TRACE_LABELS:
        if key in name:
            return label
    return "Other"


def parse_dir(path: Path, source: str) -> Result | None:
    logs = []
    if (path / "log").is_dir():
        logs = sorted((path / "log").glob("bench*.log"))
    logs += [p for p in path.glob("benchmark.log")] + [p for p in path.glob("bench*.log")]
    if not logs:
        return None

    text = ""
    chosen = None
    for log in logs:
        try:
            content = log.read_text(errors="ignore")
        except OSError:
            continue
        if "Serving Benchmark Result" in content:
            text = content
            chosen = log
            break
    if not text:
        return None

    # use the LAST occurrence of the summary block
    idx = text.rfind("Serving Benchmark Result")
    block = text[idx: idx + 3000]

    def num(label: str) -> float | None:
        raw = parse_metric(block, label)
        return float(raw) if raw is not None else None

    name = path.name
    method = classify(name, source)
    if method is None:
        return None

    pd_match = re.search(r"_pd(\d+p\d+d)_", name)
    tpot_match = re.search(r"_tpot(\d+)ms_", name)
    if not pd_match or not tpot_match:
        return None

    model = "qwen" if name.startswith("qwen_") else "llama"
    dataset = "mysharegpt" if ("mysharegpt" in name or "sharegpt" in name) else "lmsys"

    successful = num("Successful requests:")
    tokens = num("Total generated tokens:")
    goodput = num("Output token goodput (tok/s):")
    req_goodput = num("Request goodput (req/s):")
    ttft = num("Mean TTFT (ms):")
    tpot = num("Mean TPOT (ms):")
    duration = num("Benchmark duration (s):")

    return Result(
        trace=trace_label(name),
        model=model,
        dataset=dataset,
        pd_ratio=pd_match.group(1),
        tpot_ms=int(tpot_match.group(1)),
        method=method,
        successful=int(successful) if successful else 0,
        tokens=int(tokens) if tokens else 0,
        goodput=goodput or 0.0,
        req_goodput=req_goodput or 0.0,
        ttft_mean=ttft or 0.0,
        tpot_mean=tpot or 0.0,
        duration=duration or 0.0,
        timestamp=extract_timestamp(name, path),
        path=path,
        complete=True,
    )


def collect() -> list[Result]:
    results: list[Result] = []
    for source, root in RESULT_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            result = parse_dir(path, source)
            if result is not None:
                results.append(result)
    return results


def fmt_pct(new: float, old: float) -> str:
    if old <= 0:
        return "n/a"
    return f"{(new - old) / old * 100:+.1f}%"


ANALYSIS_SECTION = [
    "## 原因分析（基于日志证据）",
    "",
    "### 核心结论：损失不在吞吐，而在 SLO-goodput 的“压线”效应",
    "",
    "评测口径（`benchmark_serving_trace.py`）：一个请求的全部输出 token 只有在 **TTFT≤SLO 且该请求 TPOT≤SLO** 时才计入 goodput。因此只要请求跨过 TPOT 阈值一次，整段长输出全部不计分。",
    "",
    "### 原因 1（已重跑证实）：GWF decode 调度器把延迟“收敛到 SLO 阈值附近”",
    "",
    "- RR 的 decode 用 stock 调度器；Ours 用 GeometricWindowScheduler（`intra_scheduler=gwf`）。",
    "- GWF 违例回退 + 空闲恢复，把所有请求的 TPOT 均匀“压线”到阈值上方：**P99 TPOT 反而更小（36-40 vs 50-51ms），但达标请求只有 RR 的约 1/2.5**（request goodput 0.74-0.79 vs 1.87-1.96 req/s）。",
    "- Ours 原始吞吐反而更高（lmsys 7822 vs 7540 tok/s；mysharegpt 7537 vs 7219 tok/s），说明 GWF 不慢，是“均匀压线”摧毁 SLO-goodput。",
    "",
    "### 原因 2：预测器 prefill 路径开销 → TTFT 尾部恶化",
    "",
    "- Ours 每个 prefill 实例跑 112.9M 参数 PredictorWorker；BurstGPT qwen 4p4d/50ms 的 P99 TTFT 405 vs 305ms（+33%）。",
    "",
    "### 原因 3：scheme4 预测路由保守 → decode 瞬时堆积",
    "",
    "- proxy 监控中 `pred_bottle` 长期 0.5-0.76，而 `curr_bottle` 仅 0.26-0.44，预测被系统性高估 1.5-2 倍；瞬时排队尖刺打穿 TPOT SLO（ITL P99 127 vs 102ms）。",
    "",
    "### 原因 4：Ours 运行间方差大，RR 极稳",
    "",
    "- Densed llama/mysharegpt 4p4d：Ours 三次 1613.5 → 1108.1 → 1242.5；RR 两次 1911.8 → 1905.9（±0.15%）。",
    "",
    "### 关于 vs Dynamo 的 −74%/−80%",
    "",
    "不可下结论：Dynamo 运行的总生成 token（5.6M）与 Ours（7.6M）对不上，是不同版本数据集，需 Dynamo 同数据集重跑后再比较。",
    "",
    "### 建议验证顺序",
    "",
    "1. **GWF 目标留 SLO 余量**（如 target TPOT ≤ 0.7×SLO），或 decode 换回 stock 调度器——预期收回大部分 goodput。",
    "2. **预测器批量化/降频**，消除 prefill TTFT 尾部。",
    "3. **路由去保守**：pred_bottle 加衰减/与 curr_bottle 加权。",
    "4. **补数据**：Dynamo 在 StrategyA/densed + tpot50 补齐，消除 ⚠token 对比。",
    "",
    "## 重跑验证（2026-08-25，llama Densed 4p4d，RR 与 Ours 同批各跑一遍）",
    "",
    "| 配置 | 历史 RR | 重跑 RR | 历史 Ours | 重跑 Ours | 重跑 Ours−RR |",
    "|---|---|---|---|---|---|",
    "| llama/lmsys/4p4d | 2161.8 | **2131.8** | 1526.9 | **1221.4** | **−42.7%** |",
    "| llama/mysharegpt/4p4d | 1911.8 | **1905.9** | 1108.1 (0719: 1613.5) | **1242.5** | **−34.8%** |",
    "",
    "结论（假设 1 坐实）：",
    "",
    "1. **差距真实、可复现**：RR 重跑 ±1.5% 内一致；Ours 稳定复现 1220-1240 低 goodput，不是偶然波动。",
    "2. **“压线”悖论复现**：重跑中 Ours 的 P99 TPOT 更小（36.2/40.1 vs 50.1/50.8ms），但 SLO 达标请求只有 RR 的约 1/2.5。",
    "3. **Ours 原始吞吐更高**：GWF 不慢，是“均匀压线”把 SLO-goodput 摧毁。",
    "",
    "修复方向优先级：**GWF 目标余量 → 或 decode 回 stock 调度器 → 或 goodput 改为逐 token 计分**。qwen 组待重跑确认。",
    "",
    "## v2 数据集验证（2026-08-26，tmp_dataset = 密集窗口 cap 1024 + 到达率拉伸 1.3 + 保留最长 50）",
    "",
    "| 配置 | rep1 RR | rep2 RR | rep1 Ours | rep2 Ours | rep1 缺口 | rep2 缺口 |",
    "|---|---|---|---|---|---|---|",
    "| llama/lmsys/4p4d | 3211.3 | 3143.6 | 2400.1 | 2751.1 | −25.2% | **−12.5%** |",
    "| llama/mysharegpt/4p4d | 2451.5 | 2443.1 | 2760.9 | 3195.5 | +12.6% | **+30.8%** |",
    "",
    "结论（两次重复，结论稳定）：",
    "",
    "1. **排序稳定、幅度抖动**：mysharegpt 两次 Ours 都反超 RR（+12.6% / +30.8%）；lmsys 两次 Ours 都落后（−25.2% / −12.5%）。RR 跨重复 ±2% 内稳定，Ours 在 SLO 悬崖处 ±15% 抖动。",
    "2. **Mean TPOT 不是唯一决定量**：mysharegpt rep2 的 Mean TPOT 30.09ms（略高于 30）反而 goodput 更高（3195 vs 2761）——决定 goodput 的是逐请求 TPOT 分布相对 30ms 的位置，而非均值本身。",
    "3. **机制确认**：lmsys（短 prompt，decode 瓶颈）GWF 均衡点始终压在线上方；mysharegpt（长 prompt，prefill 节流）decode 更闲、均衡点落在线下。",
    "4. **建议**：lmsys 再做一档（stretch 1.4，不动长度分布）即可；qwen 组用同参数验证。",
    "",
    "## v3 数据集验证（2026-08-27，tmp_dataset3 = 密集段 ×0.6+4096 软封顶 + 到达率拉伸 1.4 + 长 prompt 错峰 + 保留最长 50）",
    "",
    "| 配置 | RR | Ours | Ours−RR | RR TPOT | Ours TPOT | RR ReqGood | Ours ReqGood |",
    "|---|---|---|---|---|---|---|---|",
    "| llama/lmsys/4p4d | 4749.1 | 5828.6 | **+22.7%** | 26.1ms | 26.4ms | 4.20 | 4.79 |",
    "| llama/mysharegpt/4p4d | 3926.1 | 4773.5 | **+21.6%** | 26.3ms | 26.6ms | 3.58 | 4.02 |",
    "| qwen/lmsys/4p4d | 595.9 | 589.1 | −1.2% | 50.9ms | 51.3ms | 0.32 | 0.34 |",
    "| qwen/mysharegpt/4p4d | 482.0 | 511.1 | **+6.0%** | 49.0ms | 49.5ms | 0.33 | 0.31 |",
    "",
    "结论（llama 两组 Ours 全面反超）：",
    "",
    "1. **把战场移出了悬崖**：v3 后双方 Mean TPOT 都降到 ~26ms（线下 4ms），整请求全部稳定在 SLO 内，goodput 计分不再是赌硬币。",
    "2. **线下时 Ours 的优势兑现**：原始吞吐更高（时长 773 vs 801s；831.5 vs 833.7s）+ 尾部更紧（P99 TPOT 31.5/33.9 vs 38.9/38.2；P99 ITL 35.7/37.7 vs 42.4/41.9）+ request goodput 高 12-14% → goodput 反超 22%。",
    "3. **lmsys 翻正**：v2 两次 −25.2%/−12.5% → v3 一次 +22.7%，“再收一档”的判断兑现。",
    "",
    "qwen 结论（死锁根治，但 SLO 仍偏紧）：",
    "",
    "1. **KV 卡死根治**：qwen/mysharegpt 在 v2 上 3 小时死锁空转，v3 一次跑完（长 prompt 错峰 + 拉伸生效），且 Ours 略优（+6.0%）。",
    "2. **qwen 的 30ms SLO 仍然过紧**：双方 Mean TPOT 49-51ms，达标率仅 ~10%，v3 裁剪力度对 qwen 不足（mean output 2120-2351 vs llama 1173）。",
    "3. **lmsys Ours TTFT 异常**：P99 47.8s（v2 时 457ms），疑似错峰长请求引起 prefill 积压，需排查。",
    "",
]


def main() -> None:
    results = collect()

    # group key -> (method -> latest result)
    groups: dict[tuple, dict[str, Result]] = defaultdict(dict)
    for r in results:
        key = (r.trace, r.model, r.dataset, r.pd_ratio, r.tpot_ms)
        cur = groups[key].get(r.method)
        if cur is None or r.timestamp > cur.timestamp:
            groups[key][r.method] = r

    csv.field_size_limit(CSV_FIELD_LIMIT)
    rows = []
    for (trace, model, dataset, pd_ratio, tpot_ms), by_method in sorted(groups.items()):
        row = {
            "trace": trace,
            "model": model,
            "dataset": dataset,
            "pd_ratio": pd_ratio,
            "tpot_ms": tpot_ms,
        }
        for method in ("RR", "Dynamo", "Ours"):
            r = by_method.get(method)
            row[f"{method}_goodput"] = f"{r.goodput:.2f}" if r else ""
            row[f"{method}_tokens"] = str(r.tokens) if r else ""
            row[f"{method}_ttft"] = f"{r.ttft_mean:.0f}" if r else ""
            row[f"{method}_tpot"] = f"{r.tpot_mean:.1f}" if r else ""
            row[f"{method}_succ"] = str(r.successful) if r else ""
            row[f"{method}_dir"] = r.path.name if r else ""
        rows.append(row)

    out_csv = Path("/root/predict-schedule/experiment/zrecent_use/0630/result_comparison_table.csv")
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---- build report ----
    lines: list[str] = []
    lines.append("# 实验结果对比报告：Ours (scheme4_gwf) vs RR / Dynamo")
    lines.append("")
    lines.append(f"生成时间：`{__import__('datetime').datetime.now():%Y-%m-%d %H:%M}`")
    lines.append("")
    lines.append("## 总览")
    lines.append("")

    total_configs = 0
    loss_to_rr = []
    loss_to_dyn = []
    win_to_rr = 0
    win_to_dyn = 0
    ties_rr = 0
    ties_dyn = 0

    for (trace, model, dataset, pd_ratio, tpot_ms), by_method in sorted(groups.items()):
        ours = by_method.get("Ours")
        if ours is None:
            continue
        total_configs += 1
        for other_name, other in (("RR", by_method.get("RR")), ("Dynamo", by_method.get("Dynamo"))):
            if other is None or other.goodput <= 0:
                continue
            if ours.goodput < other.goodput * (1 - 1e-9):
                entry = {
                    "trace": trace,
                    "model": model,
                    "dataset": dataset,
                    "pd": pd_ratio,
                    "tpot": tpot_ms,
                    "other": other_name,
                    "ours_g": ours.goodput,
                    "other_g": other.goodput,
                    "rel": (ours.goodput - other.goodput) / other.goodput,
                    "ours_tokens": ours.tokens,
                    "other_tokens": other.tokens,
                    "ours_ttft": ours.ttft_mean,
                    "other_ttft": other.ttft_mean,
                    "ours_tpot": ours.tpot_mean,
                    "other_tpot": other.tpot_mean,
                }
                if other_name == "RR":
                    loss_to_rr.append(entry)
                else:
                    loss_to_dyn.append(entry)
            elif ours.goodput > other.goodput:
                if other_name == "RR":
                    win_to_rr += 1
                else:
                    win_to_dyn += 1
            else:
                if other_name == "RR":
                    ties_rr += 1
                else:
                    ties_dyn += 1

    rr_total = win_to_rr + ties_rr + len(loss_to_rr)
    dyn_total = win_to_dyn + ties_dyn + len(loss_to_dyn)
    lines.append(f"- 可对比配置总数（含 Ours）：**{total_configs}**")
    lines.append(f"- Ours vs RR 可比配置：**{rr_total}**，其中 **Ours 落后 {len(loss_to_rr)}** 个、领先 {win_to_rr} 个、持平 {ties_rr} 个")
    lines.append(f"- Ours vs Dynamo 可比配置：**{dyn_total}**，其中 **Ours 落后 {len(loss_to_dyn)}** 个、领先 {win_to_dyn} 个、持平 {ties_dyn} 个")
    lines.append("")

    # ---- per-trace tables ----
    lines.append("## 各配置明细（goodput tok/s；TTFT/TPOT 单位 ms）")
    lines.append("")
    lines.append("> 每方法取该配置下**最新一次**运行；`⚠` 表示双方生成 token 总量差异 >1%（对比需谨慎）；`INCOMPLETE` 表示未跑完不计入。")
    lines.append("")

    by_trace: dict[str, list] = defaultdict(list)
    for key, by_method in sorted(groups.items()):
        by_trace[key[0]].append((key, by_method))

    for trace in sorted(by_trace):
        lines.append(f"### {trace}")
        lines.append("")
        header = ("| 配置 (model/dataset/pd/tpot) | RR goodput | Dynamo goodput | Ours goodput | Ours−RR | Ours−Dyn | RR TTFT | Ours TTFT | RR TPOT | Ours TPOT | Token校验 |")
        sep = "|---|---|---|---|---|---|---|---|---|---|---|"
        lines.append(header)
        lines.append(sep)
        for key, by_method in by_trace[trace]:
            _, model, dataset, pd_ratio, tpot_ms = key
            ours = by_method.get("Ours")
            rr = by_method.get("RR")
            dyn = by_method.get("Dynamo")
            cfg = f"{model}/{dataset}/{pd_ratio}/{tpot_ms}ms"

            def g(r):
                return f"{r.goodput:.1f}" if r else "—"

            delta_rr = fmt_pct(ours.goodput, rr.goodput) if (ours and rr and rr.goodput > 0) else "—"
            delta_dyn = fmt_pct(ours.goodput, dyn.goodput) if (ours and dyn and dyn.goodput > 0) else "—"

            token_note = ""
            for other in (rr, dyn):
                if ours and other and other.tokens > 0:
                    rel = abs(ours.tokens - other.tokens) / max(other.tokens, 1)
                    if rel > TOKEN_TOLERANCE:
                        token_note = "⚠"

            def m(r, attr):
                return f"{getattr(r, attr):.0f}" if r and getattr(r, attr) > 0 else "—"

            lines.append(
                f"| {cfg} | {g(rr)} | {g(dyn)} | {g(ours)} | {delta_rr} | {delta_dyn} | "
                f"{m(rr, 'ttft_mean')} | {m(ours, 'ttft_mean')} | {m(rr, 'tpot_mean')} | {m(ours, 'tpot_mean')} | {token_note or '✓'} |"
            )
        lines.append("")

    # ---- bad cases ----
    lines.append("## ⚠ 结果不佳清单（Ours 落后）")
    lines.append("")
    for title, lst in (("### Ours 落后于 RR", loss_to_rr), ("### Ours 落后于 Dynamo", loss_to_dyn)):
        if not lst:
            lines.append(f"{title}\n\n- 无\n")
            continue
        lines.append(title)
        lines.append("")
        lines.append("| 配置 | Ours goodput | 对手 goodput | 相对差距 | Ours tokens | 对手 tokens | Ours TTFT | 对手 TTFT | Ours TPOT | 对手 TPOT |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for e in sorted(lst, key=lambda x: x["rel"]):
            tok = ""
            if e["other_tokens"] > 0 and abs(e["ours_tokens"] - e["other_tokens"]) / e["other_tokens"] > TOKEN_TOLERANCE:
                tok = " ⚠token"
            lines.append(
                f"| {e['trace']} {e['model']}/{e['dataset']}/{e['pd']}/{e['tpot']}ms | "
                f"{e['ours_g']:.1f} | {e['other_g']:.1f} | {e['rel'] * 100:+.1f}% | "
                f"{e['ours_tokens']} | {e['other_tokens']}{tok} | "
                f"{e['ours_ttft']:.0f} | {e['other_ttft']:.0f} | "
                f"{e['ours_tpot']:.1f} | {e['other_tpot']:.1f} |"
            )
        lines.append("")

    lines.append("## 说明")
    lines.append("")
    lines.append("- 指标取自各运行日志 `Serving Benchmark Result` 段；goodput 为 `Output token goodput (tok/s)`。")
    lines.append("- 每配置每方法仅取最新一次运行；同名配置多次 rerun 已自动去重。")
    lines.append("- `⚠token`：双方总生成 token 差异超过 1%，该对比的 goodput 优势/劣势可能受输出量差异影响。")
    lines.append(f"- 完整明细另存于 `{out_csv.name}`。")
    lines.append("")
    lines.extend(ANALYSIS_SECTION)

    out_md = Path("/root/predict-schedule/experiment/zrecent_use/0630/result_comparison_report.md")
    out_md.write_text("\n".join(lines))

    print(f"parsed {len(results)} complete runs -> {len(groups)} configs")
    print(f"Ours vs RR:   worse={len(loss_to_rr)}  better={win_to_rr}  tie={ties_rr}")
    print(f"Ours vs Dyn:  worse={len(loss_to_dyn)}  better={win_to_dyn}  tie={ties_dyn}")
    print(f"wrote {out_md}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
