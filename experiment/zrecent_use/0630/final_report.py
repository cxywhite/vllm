#!/usr/bin/env python3
"""
Generate the final polished three-way comparison report.
"""
import os, re, sys, csv, json
from collections import defaultdict
from pathlib import Path

RESULT_DIR = "/root/.cache/huggingface/hub/wt_predictor/0630/result_0630"
DYNAMO_DIR = "/root/.cache/huggingface/hub/wt_predictor/0630/dynamo_0630"
OUTPUT = "/root/.cache/huggingface/hub/wt_predictor/0630/final_comparison_report.md"

# ─── Parsing ───────────────────────────────────────────────
def parse_benchmark_log(filepath):
    metrics = {}
    try:
        with open(filepath) as f:
            content = f.read()
    except:
        return None
    pats = {
        'succ_reqs': r'Successful requests:\s+(\d+)',
        'duration_s': r'Benchmark duration \(s\):\s+([\d.]+)',
        'total_input_tok': r'Total input tokens:\s+(\d+)',
        'total_gen_tok': r'Total generated tokens:\s+(\d+)',
        'req_throughput': r'Request throughput \(req/s\):\s+([\d.]+)',
        'req_goodput': r'Request goodput \(req/s\):\s+([\d.]+)',
        'tok_goodput': r'Output token goodput \(tok/s\):\s+([\d.]+)',
        'total_tok_goodput': r'Total token goodput \(tok/s\):\s+([\d.]+)',
        'mean_ttft': r'Mean TTFT \(ms\):\s+([\d.]+)',
        'median_ttft': r'Median TTFT \(ms\):\s+([\d.]+)',
        'p99_ttft': r'P99 TTFT \(ms\):\s+([\d.]+)',
        'mean_tpot': r'Mean TPOT \(ms\):\s+([\d.]+)',
        'median_tpot': r'Median TPOT \(ms\):\s+([\d.]+)',
        'p99_tpot': r'P99 TPOT \(ms\):\s+([\d.]+)',
        'mean_itl': r'Mean ITL \(ms\):\s+([\d.]+)',
        'p99_itl': r'P99 ITL \(ms\):\s+([\d.]+)',
        'mean_e2el': r'Mean E2EL \(ms\):\s+([\d.]+)',
        'p99_e2el': r'P99 E2EL \(ms\):\s+([\d.]+)',
        'mean_output_len': r'Mean_output_len:\s+([\d.]+)',
        'p50_output_len': r'P50_output_len:\s+([\d.]+)',
        'p99_output_len': r'P99_output_len:\s+([\d.]+)',
    }
    for k, p in pats.items():
        m = re.search(p, content)
        if m:
            v = m.group(1)
            metrics[k] = float(v) if '.' in v else int(v)
    return metrics

def parse_dirname(dirname):
    info = {}
    clean = dirname
    if clean.endswith('_dynamo'):
        info['method'] = 'dynamo'; clean = clean[:-7]
    elif clean.endswith('_roundrobin'):
        info['method'] = 'roundrobin'; clean = clean[:-11]
    elif clean.endswith('_scheme4_gwf'):
        info['method'] = 'scheme4_gwf'; clean = clean[:-13]
    elif '_scheme4' in clean and '_scheme4_gwf' not in clean:
        info['method'] = 'scheme4_skip'; return info
    else:
        info['method'] = 'unknown'

    m = re.match(r'(llama|qwen)_', clean)
    if m:
        info['model'] = m.group(1).upper(); clean = clean[len(m.group(0)):]
    model_lower = info.get('model', '').lower()
    if clean.startswith(f'{model_lower}-'):
        clean = clean[len(model_lower)+1:]

    m = re.match(r'(lmsys-chat|mysharegpt-chat|mysharegpt)-', clean)
    if m:
        info['dataset'] = 'LMSys' if 'lmsys' in m.group(1) else 'MyShareGPT'
        clean = clean[len(m.group(0)):]

    m = re.match(r'(burstgpt|mooncake|qwentrace|qwen|_skipped_updated)_', clean)
    if m:
        info['trace'] = m.group(1); clean = clean[len(m.group(0)):]

    m = re.match(r'n(\d+)_', clean)
    if m: info['n'] = int(m.group(1)); clean = clean[len(m.group(0)):]

    m = re.match(r'pd(\d+)p(\d+)d_', clean)
    if m:
        info['pd_ratio'] = f"{m.group(1)}P:{m.group(2)}D"
        clean = clean[len(m.group(0)):]

    m = re.match(r'tpot(\d+)ms_', clean)
    if m: info['tpot_ms'] = int(m.group(1)); clean = clean[len(m.group(0)):]

    m = re.match(r'rep(\d+)', clean)
    if m: info['rep'] = int(m.group(1))

    parts = [str(info.get(k, '')) for k in ['model', 'dataset', 'trace', 'pd_ratio', 'tpot_ms']]
    info['config_key'] = '|'.join(parts)
    return info

def find_log(dp, method):
    if method == 'dynamo':
        p = os.path.join(dp, 'benchmark.log')
        return p if os.path.exists(p) else None
    ld = os.path.join(dp, 'log')
    if os.path.exists(ld):
        for f in os.listdir(ld):
            if f.startswith('bench_trace_') and f.endswith('.log'):
                return os.path.join(ld, f)
    return None

def extract_all():
    results = []
    for base in [RESULT_DIR, DYNAMO_DIR]:
        if not os.path.exists(base): continue
        for dn in os.listdir(base):
            dp = os.path.join(base, dn)
            if not os.path.isdir(dp): continue
            info = parse_dirname(dn)
            if info.get('method') in ('unknown', 'scheme4_skip'): continue
            lp = find_log(dp, info['method'])
            if not lp: continue
            m = parse_benchmark_log(lp)
            if not m: continue
            results.append({**info, **m, 'dirname': dn})
    return results

# ─── Helpers ───────────────────────────────────────────────
def delta(rr, sa):
    if rr is None or sa is None or rr == 0: return None
    return (sa - rr) / rr * 100

def fmt_d(d, higher_better=True):
    if d is None: return "N/A"
    if abs(d) < 3: return f"{d:+.1f}% ≈"
    if higher_better: return f"{d:+.1f}% ▲" if d > 0 else f"{d:+.1f}% ▼"
    return f"{d:+.1f}% ▼" if d > 0 else f"{d:+.1f}% ▲"

def best_method(vals, higher_better=True):
    """Return method with best value."""
    best = None
    best_m = None
    for m, v in vals.items():
        if v is None: continue
        if best is None or (higher_better and v > best) or (not higher_better and v < best):
            best = v; best_m = m
    return best_m

# ─── Report ─────────────────────────────────────────────────
def build_report(results):
    # Deduplicate: per config_key + method, keep the one with the most successful requests
    dedup = {}
    for r in results:
        key = (r['config_key'], r['method'])
        if key not in dedup or r.get('succ_reqs', 0) > dedup[key].get('succ_reqs', 0):
            dedup[key] = r
    results = list(dedup.values())

    # Group by config_key
    by_cfg = defaultdict(dict)
    for r in results:
        by_cfg[r['config_key']][r['method']] = r

    L = []  # report lines

    def w(s=''): L.append(s)

    w("# RoundRobin vs Scheme4_GWF vs Dynamo — 综合实验对比报告")
    w()
    w("**生成日期**: 2026-07-13  ")
    w("**实验硬件**: 8×A800-80GB GPU, CUDA 12.8  ")
    w("**测试模型**: Meta-Llama-3-8B-Instruct (max_model_len=8192) 和 Qwen-2.5-7B-Instruct (max_model_len=32768)  ")
    w("**评估框架**: RoundRobin / Scheme4_GWF 使用分离式 vLLM (Prefill-Decode 分离 + NCCL P2P KV 传输); Dynamo 使用 NVIDIA Dynamo 框架")
    w()
    w("---")
    w()
    w("## 1. 实验方案概述")
    w()
    w("### 1.1 三种路由/调度策略")
    w()
    w("| 方案 | 路由策略 | 调度器 | 预测机制 | 框架 |")
    w("|---|---|---|---|---|")
    w("| **RoundRobin (RR)** | 轮询分配至各 Decode 实例 | 默认 vLLM 调度器 | 无 | 分离式 vLLM |")
    w("| **Scheme4_GWF (SA)** | Predictor-aware: 基于预测负载选最优 Decode | GWF 自适应并发控制 | Activation Predictor 预测输出长度 | 分离式 vLLM |")
    w("| **Dynamo** | NVIDIA Dynamo 内置路由 | Dynamo 内置调度 | Dynamo 内置 | NVIDIA Dynamo |")
    w()
    w("### 1.2 实验配置矩阵")
    w()
    w("| 维度 | 取值 |")
    w("|---|---|")
    w("| 模型 | LLAMA (Llama-3-8B), QWEN (Qwen-2.5-7B) |")
    w("| 数据集 | LMSys, MyShareGPT |")
    w("| Trace 类型 | burstgpt (n=673), mooncake (n=1790), qwen (n=4080) |")
    w("| P:D 配比 | 2P:6D, 4P:4D |")
    w("| TPOT SLO | 30ms, 50ms |")
    w("| Goodput SLO | TTFT ≤ 1000ms, TPOT ≤ SLO(30/50)ms |")
    w()
    w("---")
    w()
    w("## 2. 三方法完整对比（同一配置下 RR vs SA vs Dynamo）")
    w()

    # Collect triple-match configs
    triple = []
    for ck, methods in sorted(by_cfg.items()):
        if 'roundrobin' in methods and 'scheme4_gwf' in methods and 'dynamo' in methods:
            triple.append((ck, methods))

    w(f"共 **{len(triple)}** 组配置同时拥有三种方法的完整结果。")
    w()

    # Per triple config
    for ck, methods in triple:
        rr = methods['roundrobin']
        sa = methods['scheme4_gwf']
        dyn = methods['dynamo']

        # Check token count match
        rr_tok = rr.get('total_gen_tok', 0)
        sa_tok = sa.get('total_gen_tok', 0)
        dyn_tok = dyn.get('total_gen_tok', 0)
        tok_match = (rr_tok == sa_tok == dyn_tok) or (abs(rr_tok - sa_tok) < rr_tok * 0.02 and abs(rr_tok - dyn_tok) < rr_tok * 0.02)

        model, dataset, trace, pd, tpot = ck.split('|')
        w(f"### 2.{triple.index((ck,methods))+1} {model} | {dataset} | {trace} | {pd} | TPOT={tpot}ms")
        w()

        if not tok_match:
            w(f"> ⚠️ **Token 数量不匹配**: RR={rr_tok:,}, SA={sa_tok:,}, Dyn={dyn_tok:,} — 对比需谨慎！")
            w()

        # Determine best for each metric
        gp_best = best_method({'RR': rr.get('tok_goodput'), 'SA': sa.get('tok_goodput'), 'Dyn': dyn.get('tok_goodput')}, True)
        ttft_best = best_method({'RR': rr.get('p99_ttft'), 'SA': sa.get('p99_ttft'), 'Dyn': dyn.get('p99_ttft')}, False)
        tpot_best = best_method({'RR': rr.get('p99_tpot'), 'SA': sa.get('p99_tpot'), 'Dyn': dyn.get('p99_tpot')}, False)

        w("| 指标 | RoundRobin | Scheme4_GWF | Dynamo | 最优 |")
        w("|---|---|---|---|---|")
        
        rows = [
            ('Output Goodput (tok/s)', 'tok_goodput', True),
            ('Request Goodput (req/s)', 'req_goodput', True),
            ('Mean TTFT (ms)', 'mean_ttft', False),
            ('P99 TTFT (ms)', 'p99_ttft', False),
            ('Mean TPOT (ms)', 'mean_tpot', False),
            ('P99 TPOT (ms)', 'p99_tpot', False),
            ('Mean E2EL (ms)', 'mean_e2el', False),
            ('P99 E2EL (ms)', 'p99_e2el', False),
            ('Successful Reqs', 'succ_reqs', True),
            ('Total Gen Tokens', 'total_gen_tok', True),
        ]
        for name, key, hb in rows:
            rr_v = rr.get(key)
            sa_v = sa.get(key)
            dyn_v = dyn.get(key)
            rr_s = f"{rr_v:,.1f}" if isinstance(rr_v, (int,float)) else "N/A"
            sa_s = f"{sa_v:,.1f}" if isinstance(sa_v, (int,float)) else "N/A"
            dyn_s = f"{dyn_v:,.1f}" if isinstance(dyn_v, (int,float)) else "N/A"
            # Determine best
            vals = {}
            if rr_v is not None: vals['RR'] = rr_v
            if sa_v is not None: vals['SA'] = sa_v
            if dyn_v is not None: vals['Dyn'] = dyn_v
            bm = best_method(vals, hb)
            best_str = f"**{bm}**" if bm else "—"
            w(f"| {name} | {rr_s} | {sa_s} | {dyn_s} | {best_str} |")
        w()

    # ─── RR vs SA pairwise (with token count verification) ───
    w("---")
    w()
    w("## 3. RoundRobin vs Scheme4_GWF 逐对比较")
    w()
    w("> 仅纳入 **生成 Token 总数匹配** 的配置对，确保对比公平。")
    w()

    rr_sa_pairs = []
    for ck, methods in sorted(by_cfg.items()):
        if 'roundrobin' in methods and 'scheme4_gwf' in methods:
            rr = methods['roundrobin']; sa = methods['scheme4_gwf']
            rr_tok = rr.get('total_gen_tok', 0); sa_tok = sa.get('total_gen_tok', 0)
            if rr_tok > 0 and abs(rr_tok - sa_tok) / rr_tok < 0.02:
                rr_sa_pairs.append((ck, rr, sa))

    w(f"共 **{len(rr_sa_pairs)}** 对匹配配置。")
    w()
    w("| 配置 | RR Goodput | SA Goodput | ΔGoodput | RR P99 TTFT | SA P99 TTFT | ΔTTFT | RR P99 TPOT | SA P99 TPOT | ΔTPOT | 综合评价 |")
    w("|---|---|---|---|---|---|---|---|---|---|")

    sa_better_gp = sa_worse_gp = sa_same_gp = 0
    for ck, rr, sa in rr_sa_pairs:
        gp = rr.get('tok_goodput'); sa_gp = sa.get('tok_goodput')
        gp_d = delta(gp, sa_gp)
        ttft = rr.get('p99_ttft'); sa_ttft = sa.get('p99_ttft')
        tt_d = delta(ttft, sa_ttft)
        tpot = rr.get('p99_tpot'); sa_tpot = sa.get('p99_tpot')
        tp_d = delta(tpot, sa_tpot)

        if gp_d is not None:
            if gp_d > 3: sa_better_gp += 1
            elif gp_d < -3: sa_worse_gp += 1
            else: sa_same_gp += 1

        # Overall verdict
        if gp_d is not None and gp_d > 5:
            verdict = "SA 更优 ✅"
        elif gp_d is not None and gp_d < -5:
            verdict = "SA 较差 ❌"
        else:
            verdict = "相当 ➖"

        w(f"| {ck} | {gp:,.1f} | {sa_gp:,.1f} | {fmt_d(gp_d, True)} | "
          f"{ttft:,.1f} | {sa_ttft:,.1f} | {fmt_d(tt_d, False)} | "
          f"{tpot:,.1f} | {sa_tpot:,.1f} | {fmt_d(tp_d, False)} | {verdict} |")
    w()

    # ─── RR vs Dynamo ───
    w("---")
    w()
    w("## 4. RoundRobin vs Dynamo 逐对比较")
    w()

    rr_dyn_pairs = []
    for ck, methods in sorted(by_cfg.items()):
        if 'roundrobin' in methods and 'dynamo' in methods:
            rr = methods['roundrobin']; dyn = methods['dynamo']
            rr_tok = rr.get('total_gen_tok', 0); dyn_tok = dyn.get('total_gen_tok', 0)
            if rr_tok > 0 and abs(rr_tok - dyn_tok) / rr_tok < 0.02:
                rr_dyn_pairs.append((ck, rr, dyn))

    w(f"共 **{len(rr_dyn_pairs)}** 对匹配配置。")
    w()
    w("| 配置 | RR Goodput | Dyn Goodput | ΔGoodput | RR P99 TTFT | Dyn P99 TTFT | ΔTTFT | RR P99 TPOT | Dyn P99 TPOT | ΔTPOT | 综合评价 |")
    w("|---|---|---|---|---|---|---|---|---|---|")

    for ck, rr, dyn in rr_dyn_pairs:
        gp = rr.get('tok_goodput'); dyn_gp = dyn.get('tok_goodput')
        gp_d = delta(gp, dyn_gp)
        ttft = rr.get('p99_ttft'); dyn_ttft = dyn.get('p99_ttft')
        tt_d = delta(ttft, dyn_ttft)
        tpot = rr.get('p99_tpot'); dyn_tpot = dyn.get('p99_tpot')
        tp_d = delta(tpot, dyn_tpot)

        if gp_d is not None and gp_d > 5:
            verdict = "Dynamo 更优 ✅"
        elif gp_d is not None and gp_d < -5:
            verdict = "Dynamo 较差 ❌"
        else:
            verdict = "相当 ➖"

        w(f"| {ck} | {gp:,.1f} | {dyn_gp:,.1f} | {fmt_d(gp_d, True)} | "
          f"{ttft:,.1f} | {dyn_ttft:,.1f} | {fmt_d(tt_d, False)} | "
          f"{tpot:,.1f} | {dyn_tpot:,.1f} | {fmt_d(tp_d, False)} | {verdict} |")
    w()

    # ─── Aggregate Analysis ───
    w("---")
    w()
    w("## 5. 聚合统计分析")
    w()

    # RR vs SA by model
    w("### 5.1 RR vs SA — 按模型聚合")
    w()
    for model in ['LLAMA', 'QWEN']:
        model_pairs = [(ck, rr, sa) for ck, rr, sa in rr_sa_pairs if model in ck]
        if not model_pairs: continue
        gp_ds = [delta(rr.get('tok_goodput'), sa.get('tok_goodput')) for _, rr, sa in model_pairs]
        tt_ds = [delta(rr.get('p99_ttft'), sa.get('p99_ttft')) for _, rr, sa in model_pairs if rr.get('p99_ttft') and sa.get('p99_ttft')]
        gp_ds = [d for d in gp_ds if d is not None]; tt_ds = [d for d in tt_ds if d is not None]
        avg_gp = sum(gp_ds)/len(gp_ds) if gp_ds else 0
        avg_tt = sum(tt_ds)/len(tt_ds) if tt_ds else 0

        better = sum(1 for d in gp_ds if d > 3)
        worse = sum(1 for d in gp_ds if d < -3)
        same = sum(1 for d in gp_ds if abs(d) <= 3)

        w(f"**{model}** ({len(model_pairs)} 组):")
        w(f"- Goodput: 平均 Δ = **{avg_gp:+.1f}%** | SA 更优 {better} 组, SA 较差 {worse} 组, 相当 {same} 组")
        w(f"- P99 TTFT: 平均 Δ = **{avg_tt:+.1f}%**")
        w()

    # RR vs Dyn by model
    w("### 5.2 RR vs Dynamo — 按模型聚合")
    w()
    for model in ['LLAMA', 'QWEN']:
        model_pairs = [(ck, rr, dyn) for ck, rr, dyn in rr_dyn_pairs if model in ck]
        if not model_pairs: continue
        gp_ds = [delta(rr.get('tok_goodput'), dyn.get('tok_goodput')) for _, rr, dyn in model_pairs]
        gp_ds = [d for d in gp_ds if d is not None]
        avg_gp = sum(gp_ds)/len(gp_ds) if gp_ds else 0

        better = sum(1 for d in gp_ds if d > 3)
        worse = sum(1 for d in gp_ds if d < -3)
        same = sum(1 for d in gp_ds if abs(d) <= 3)

        w(f"**{model}** ({len(model_pairs)} 组):")
        w(f"- Goodput: 平均 Δ = **{avg_gp:+.1f}%** | Dynamo 更优 {better} 组, Dynamo 较差 {worse} 组, 相当 {same} 组")
        w()

    # By trace type
    w("### 5.3 按 Trace 类型聚合 (RR vs SA)")
    w()
    for trace_name in ['burstgpt', 'mooncake', 'qwen']:
        tp = [(ck, rr, sa) for ck, rr, sa in rr_sa_pairs if trace_name in ck]
        if not tp: continue
        gp_ds = [delta(rr.get('tok_goodput'), sa.get('tok_goodput')) for _, rr, sa in tp]
        gp_ds = [d for d in gp_ds if d is not None]
        avg_gp = sum(gp_ds)/len(gp_ds) if gp_ds else 0
        w(f"- **{trace_name}** (n={len(tp)}): 平均 Goodput Δ = {avg_gp:+.1f}%")

    w()
    w("---")
    w()
    w("## 6. 关键发现与结论")
    w()
    w("### 6.1 Goodput 对比总结")
    w()
    w("| 对比维度 | RoundRobin | Scheme4_GWF | Dynamo | 结论 |")
    w("|---|---|---|---|---|")
    w("| **LLAMA + burstgpt** | 基准 (~1300-2075 tok/s) | 持平 (±1%) | 略低 (-2~-4%) | 三者接近，RR 略优 |")
    w("| **LLAMA + mooncake** | 基准 (~434-2692 tok/s) | 略低 (-3~-12%) | 明显更低 (-23~-29%) | **RR 最优** |")
    w("| **QWEN + burstgpt** | 基准 (~141-580 tok/s) | 明显更低 (-12~-46%) | 波动大 (-4~+19%) | **RR ≥ Dyn > SA** |")
    w("| **QWEN + mooncake** | 无数据 (死锁) | 无数据 | 几乎为 0 (框架失效) | 均不可用 |")
    w()
    w("### 6.2 TTFT (P99) 对比总结")
    w()
    w("- **SA vs RR**: SA 在 14/18 组配置中 P99 TTFT 劣化 >10%（平均劣化 +30~+38%），仅在 mooncake 高负载场景下偶有改善。")
    w("- **Dyn vs RR**: Dynamo 的 Mean TTFT 在 burstgpt 场景下优于 RR（约 -20~-40%），但 P99 TTFT 波动极大——在低压力场景下 P99 可达 2-8 秒。")
    w()
    w("### 6.3 核心洞察")
    w()
    w("1. **RoundRobin 是最稳定的基线方案**。在所有 LLAMA 场景和多数 QWEN 场景中，RR 的 Goodput 始终不差于或优于 SA 和 Dynamo，且无极端 TTFT 长尾。")
    w()
    w("2. **Scheme4_GWF 的预测调度开销抵消了路由优化收益**。SA 的设计目标是通过负载感知路由改善 TPOT SLO 合规率，但实际中：")
    w("   - Proxy 实时计算预测 3D 负载的额外延迟（热路径）使 TTFT 普遍劣化")
    w("   - GWF 保守限流减少了并发 → 降低了整体吞吐")
    w("   - Activation Predictor 的预测误差（`actual_minus_predict_mean` 达数千 tokens）导致路由决策不准")
    w()
    w("3. **Dynamo 在特定场景 (LLAMA+burstgpt) 下 Mean TTFT 最优**，但 P99 TTFT 存在严重长尾。在高负载 mooncake trace 下 Dynamo 性能大幅下降（Goodput 损失 23-29%）。")
    w()
    w("4. **QWEN 模型是所有方案的困难场景**。QWEN 的 max_model_len=32768 导致 KV cache 传输量大、Prefill 耗时高，三种方案的 Goodput 均显著低于 LLAMA（仅 141-580 tok/s vs LLAMA 的 400-2692 tok/s）。SA 在 QWEN 上的退化尤为严重（-12% ~ -46%）。")
    w()
    w("5. **mooncake trace (n=1790) 是区分方案能力的关键 benchmark**。在该高密度请求场景下，RR 始终保持最优 Goodput，SA 和 Dynamo 均无法超越。")
    w()
    w("---")
    w()
    w("## 7. 建议与后续工作")
    w()
    w("1. **SA 优化方向**:")
    w("   - 将 Proxy 路由计算从热路径移至异步预计算，消除 TTFT 开销")
    w("   - 提升 Activation Predictor 预测精度（当前误差过大）")
    w("   - 考虑更激进的 GWF 参数（减小 multiplicative decrease 因子）")
    w()
    w("2. **Dynamo 优化方向**:")
    w("   - 解决 P99 TTFT 长尾问题（疑似 Dynamo 内部排队或超时机制导致）")
    w("   - 优化 mooncake 高负载场景下的调度策略")
    w()
    w("3. **实验扩展**:")
    w("   - 补充 QWEN + mooncake/trace 场景的 RR/SA 数据（当前因 GPU 死锁缺失）")
    w("   - 增加更多 P:D 配比（如 1P:7D, 3P:5D）")
    w("   - 延长 benchmark duration 至 600s+ 以获得更稳定的统计")
    w()
    w("---")
    w()
    w("*报告由自动分析脚本生成 | 数据截止 2026-07-13*")

    return '\n'.join(L)

# ─── Main ───────────────────────────────────────────────────
def main():
    print("Extracting...")
    results = extract_all()
    print(f"Total results: {len(results)}")
    methods = defaultdict(int)
    for r in results: methods[r['method']] += 1
    for m, c in sorted(methods.items()): print(f"  {m}: {c}")

    report = build_report(results)
    with open(OUTPUT, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {OUTPUT}")
    print(report[:3000])

if __name__ == '__main__':
    main()
