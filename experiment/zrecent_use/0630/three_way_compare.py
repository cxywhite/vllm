#!/usr/bin/env python3
"""
Extract metrics from RoundRobin, Scheme4_GWF, and Dynamo experiment results
and create a comprehensive comparison report.
"""
import os
import re
import json
import csv
import sys
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

# Base paths
RESULT_DIR = "/root/.cache/huggingface/hub/wt_predictor/0630/result_0630"
TMP_RESULT_DIR = "/root/.cache/huggingface/hub/wt_predictor/0630/tmp_result_0630"
DYNAMO_DIR = "/root/.cache/huggingface/hub/wt_predictor/0630/dynamo_0630"
OUTPUT_DIR = "/root/.cache/huggingface/hub/wt_predictor/0630"

# ---- Config parsing from directory name ----
def parse_dirname(dirname: str):
    """Parse experiment directory name to extract configuration.
    
    Two formats:
    1. RoundRobin/Scheme4: {model}_{model}-{dataset}-{trace}_n{n}_pd{p}d{d}d_tpot{tpot}ms_trace{tracename}_rep{rep}_{timestamp}_{method}
       e.g. llama_llama-lmsys-chat-burstgpt_n673_pd2p6d_tpot30ms_traceburstgpt_rep1_20260709_155459_roundrobin
    2. Dynamo: {model}_{model}-{dataset}-{trace}_n{n}_pd{p}d{d}d_tpot{tpot}ms_trace{tracename}_rep{rep}_dynamo
       e.g. llama_llama-lmsys-chat-burstgpt_n673_pd2p6d_tpot30ms_traceburstgpt_rep1_dynamo
    """
    info = {}
    clean = dirname
    
    # Remove method suffix first
    if clean.endswith('_dynamo'):
        info['method'] = 'dynamo'
        clean = clean[:-7]
    elif clean.endswith('_roundrobin'):
        info['method'] = 'roundrobin'
        clean = clean[:-11]
    elif clean.endswith('_scheme4_gwf'):
        info['method'] = 'scheme4_gwf'
        clean = clean[:-13]
    elif '_scheme4' in clean and '_scheme4_gwf' not in clean:
        # Skip non-GWF scheme4 results (tmp/partial)
        info['method'] = 'scheme4_skip'
        return info
    else:
        info['method'] = 'unknown'
    
    # Parse model: {model}_{model}-...
    # First part is the model name (llama or qwen)
    m = re.match(r'(llama|qwen)_', clean)
    if m:
        info['model'] = m.group(1).upper()
        clean = clean[len(m.group(0)):]
    
    # Skip repeated model prefix if present (e.g., "llama-" or "qwen-")
    model_lower = info['model'].lower()
    if clean.startswith(f'{model_lower}-'):
        clean = clean[len(model_lower) + 1:]
    
    # Parse dataset: lmsys-chat or mysharegpt or mysharegpt-chat
    m = re.match(r'(lmsys-chat|mysharegpt-chat|mysharegpt)-', clean)
    if m:
        ds = m.group(1)
        info['dataset'] = 'LMSys' if 'lmsys' in ds else 'MyShareGPT'
        clean = clean[len(m.group(0)):]
    
    # Parse trace name: burstgpt, mooncake, qwen, qwentrace, _skipped_updated
    m = re.match(r'(burstgpt|mooncake|qwentrace|qwen|_skipped_updated)_', clean)
    if m:
        info['trace'] = m.group(1)
        clean = clean[len(m.group(0)):]
    
    # Parse n
    m = re.match(r'n(\d+)_', clean)
    if m:
        info['n'] = int(m.group(1))
        clean = clean[len(m.group(0)):]
    
    # Parse pd ratio
    m = re.match(r'pd(\d+)p(\d+)d_', clean)
    if m:
        info['pd_ratio'] = f"{m.group(1)}P:{m.group(2)}D"
        info['num_prefill'] = int(m.group(1))
        info['num_decode'] = int(m.group(2))
        clean = clean[len(m.group(0)):]
    
    # Parse tpot
    m = re.match(r'tpot(\d+)ms_', clean)
    if m:
        info['tpot_ms'] = int(m.group(1))
        clean = clean[len(m.group(0)):]
    
    # Parse trace name after tpot
    m = re.match(r'trace(\w+)_', clean)
    if m:
        info['trace2'] = m.group(1)
        clean = clean[len(m.group(0)):]
    
    # Parse rep
    m = re.match(r'rep(\d+)', clean)
    if m:
        info['rep'] = int(m.group(1))
    
    # Build a config key for matching (exclude n, timestamp, rep)
    parts = []
    for k in ['model', 'dataset', 'trace', 'pd_ratio', 'tpot_ms']:
        if k in info:
            parts.append(str(info[k]))
    info['config_key'] = '|'.join(parts)
    
    return info


def parse_benchmark_log(filepath: str):
    """Parse benchmark log to extract summary metrics."""
    metrics = {}
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        return None
    
    # Patterns to extract
    patterns = {
        'successful_requests': r'Successful requests:\s+(\d+)',
        'benchmark_duration_s': r'Benchmark duration \(s\):\s+([\d.]+)',
        'total_input_tokens': r'Total input tokens:\s+(\d+)',
        'total_generated_tokens': r'Total generated tokens:\s+(\d+)',
        'request_throughput': r'Request throughput \(req/s\):\s+([\d.]+)',
        'request_goodput': r'Request goodput \(req/s\):\s+([\d.]+)',
        'output_token_goodput': r'Output token goodput \(tok/s\):\s+([\d.]+)',
        'total_token_goodput': r'Total token goodput \(tok/s\):\s+([\d.]+)',
        'output_token_throughput': r'Output token throughput \(tok/s\):\s+([\d.]+)',
        'total_token_throughput': r'Total Token throughput \(tok/s\):\s+([\d.]+)',
        'mean_ttft': r'Mean TTFT \(ms\):\s+([\d.]+)',
        'median_ttft': r'Median TTFT \(ms\):\s+([\d.]+)',
        'p99_ttft': r'P99 TTFT \(ms\):\s+([\d.]+)',
        'mean_tpot': r'Mean TPOT \(ms\):\s+([\d.]+)',
        'median_tpot': r'Median TPOT \(ms\):\s+([\d.]+)',
        'p99_tpot': r'P99 TPOT \(ms\):\s+([\d.]+)',
        'mean_itl': r'Mean ITL \(ms\):\s+([\d.]+)',
        'median_itl': r'Median ITL \(ms\):\s+([\d.]+)',
        'p99_itl': r'P99 ITL \(ms\):\s+([\d.]+)',
        'mean_e2el': r'Mean E2EL \(ms\):\s+([\d.]+)',
        'median_e2el': r'Median E2EL \(ms\):\s+([\d.]+)',
        'p99_e2el': r'P99 E2EL \(ms\):\s+([\d.]+)',
        'mean_output_len': r'Mean_output_len:\s+([\d.]+)',
        'p50_output_len': r'P50_output_len:\s+([\d.]+)',
        'p99_output_len': r'P99_output_len:\s+([\d.]+)',
    }
    
    for key, pattern in patterns.items():
        m = re.search(pattern, content)
        if m:
            val = m.group(1)
            try:
                if '.' in val:
                    metrics[key] = float(val)
                else:
                    metrics[key] = int(val)
            except ValueError:
                metrics[key] = val
    
    return metrics


def find_benchmark_log(result_path: str, method: str):
    """Find the benchmark log file in a result directory."""
    if method == 'dynamo':
        # dynamo results have benchmark.log in root
        log_path = os.path.join(result_path, 'benchmark.log')
        if os.path.exists(log_path):
            return log_path
    else:
        # RoundRobin/Scheme4 have logs in log/ subdirectory
        log_dir = os.path.join(result_path, 'log')
        if os.path.exists(log_dir):
            for f in os.listdir(log_dir):
                if f.startswith('bench_trace_') and f.endswith('.log'):
                    return os.path.join(log_dir, f)
    return None


def find_dataset_csv(result_path: str, method: str):
    """Find the dataset CSV to count actual entries."""
    if method == 'dynamo':
        for f in os.listdir(result_path):
            if f.startswith('results_') and f.endswith('.csv'):
                return os.path.join(result_path, f)
    else:
        ds_dir = os.path.join(result_path, 'dataset_result')
        if os.path.exists(ds_dir):
            for f in os.listdir(ds_dir):
                if f.startswith('test_results_') and f.endswith('.csv'):
                    return os.path.join(ds_dir, f)
    return None


def count_csv_rows(csv_path: str):
    """Count rows in a CSV file."""
    try:
        with open(csv_path, 'r') as f:
            # Use csv.reader for proper parsing (handles multiline fields)
            csv.field_size_limit(sys.maxsize)
            reader = csv.reader(f)
            next(reader)  # skip header
            count = sum(1 for _ in reader)
        return count
    except Exception:
        return None


def extract_all_results():
    """Extract metrics from all result directories."""
    all_results = []
    
    # Sources to scan
    sources = [
        (RESULT_DIR, 'result_0630'),
        (DYNAMO_DIR, 'dynamo_0630'),
    ]
    
    for base_dir, source_name in sources:
        if not os.path.exists(base_dir):
            continue
        
        for dirname in os.listdir(base_dir):
            dirpath = os.path.join(base_dir, dirname)
            if not os.path.isdir(dirpath):
                continue
            
            info = parse_dirname(dirname)
            method = info.get('method', 'unknown')
            
            # Find benchmark log
            log_path = find_benchmark_log(dirpath, method)
            if not log_path:
                continue
            
            metrics = parse_benchmark_log(log_path)
            if not metrics:
                continue
            
            # Count entries
            csv_path = find_dataset_csv(dirpath, method)
            entry_count = count_csv_rows(csv_path) if csv_path else None
            
            result = {
                'source': source_name,
                'dirname': dirname,
                'dirpath': dirpath,
                **info,
                **metrics,
                'csv_entry_count': entry_count,
            }
            all_results.append(result)
    
    return all_results


def calculate_delta(rr_val, sa_val):
    """Calculate percentage delta."""
    if rr_val is None or sa_val is None or rr_val == 0:
        return None
    return (sa_val - rr_val) / rr_val * 100


def format_delta(delta, higher_is_better=True):
    """Format delta with color indicators."""
    if delta is None:
        return "N/A"
    if abs(delta) < 3:
        return f"{delta:+.1f}% ➖"
    if higher_is_better:
        if delta > 0:
            return f"{delta:+.1f}% ✅"
        else:
            return f"{delta:+.1f}% ❌"
    else:
        if delta < 0:
            return f"{delta:+.1f}% ✅"
        else:
            return f"{delta:+.1f}% ❌"


def generate_report(all_results):
    """Generate comprehensive comparison report."""
    
    # Group results by config_key
    by_config = defaultdict(dict)
    for r in all_results:
        key = r.get('config_key', 'unknown')
        method = r.get('method', 'unknown')
        by_config[key][method] = r
    
    # Output report
    report_lines = []
    
    report_lines.append("# RoundRobin vs Scheme4_GWF vs Dynamo 综合实验对比报告")
    report_lines.append("")
    report_lines.append(f"**生成时间**: 2026-07-13")
    report_lines.append(f"**数据来源**:")
    report_lines.append(f"- RoundRobin / Scheme4_GWF: `result_0630/` (39 个结果)")
    report_lines.append(f"- Dynamo: `dynamo_0630/` (18 个结果)")
    report_lines.append("")
    
    # Summary stats
    rr_count = sum(1 for r in all_results if r.get('method') == 'roundrobin')
    sa_count = sum(1 for r in all_results if r.get('method') == 'scheme4_gwf')
    dyn_count = sum(1 for r in all_results if r.get('method') == 'dynamo')
    
    report_lines.append("## 数据概览")
    report_lines.append("")
    report_lines.append(f"| 方法 | 结果数 |")
    report_lines.append(f"|---|---|")
    report_lines.append(f"| RoundRobin | {rr_count} |")
    report_lines.append(f"| Scheme4_GWF | {sa_count} |")
    report_lines.append(f"| Dynamo | {dyn_count} |")
    report_lines.append("")
    
    # Find configs that have all three methods
    report_lines.append("## 三方法对比（同一配置）")
    report_lines.append("")
    
    # Collect triple-matched configs
    triple_matches = []
    pair_matches_rr_sa = []
    pair_matches_rr_dyn = []
    pair_matches_sa_dyn = []
    
    for key, methods in sorted(by_config.items()):
        has_rr = 'roundrobin' in methods
        has_sa = 'scheme4_gwf' in methods
        has_dyn = 'dynamo' in methods
        
        if has_rr and has_sa and has_dyn:
            triple_matches.append((key, methods))
        elif has_rr and has_sa:
            pair_matches_rr_sa.append((key, methods))
        elif has_rr and has_dyn:
            pair_matches_rr_dyn.append((key, methods))
        elif has_sa and has_dyn:
            pair_matches_sa_dyn.append((key, methods))
    
    report_lines.append(f"### 完整三方法对比: {len(triple_matches)} 组配置")
    report_lines.append("")
    
    if triple_matches:
        # Detailed triple comparison table
        for key, methods in triple_matches:
            rr = methods.get('roundrobin', {})
            sa = methods.get('scheme4_gwf', {})
            dyn = methods.get('dynamo', {})
            
            report_lines.append(f"#### {key}")
            report_lines.append("")
            
            # Main metrics table
            report_lines.append("| 指标 | RoundRobin | Scheme4_GWF | Dynamo | SA vs RR | Dyn vs RR |")
            report_lines.append("|---|---|---|---|---|---|")
            
            metrics_to_compare = [
                ('output_token_goodput', 'Output Goodput (tok/s)', True),
                ('request_goodput', 'Request Goodput (req/s)', True),
                ('mean_ttft', 'Mean TTFT (ms)', False),
                ('p99_ttft', 'P99 TTFT (ms)', False),
                ('mean_tpot', 'Mean TPOT (ms)', False),
                ('p99_tpot', 'P99 TPOT (ms)', False),
                ('mean_e2el', 'Mean E2EL (ms)', False),
                ('total_generated_tokens', 'Total Gen Tokens', True),
                ('successful_requests', 'Successful Reqs', True),
            ]
            
            for metric_key, metric_name, higher_better in metrics_to_compare:
                rr_val = rr.get(metric_key)
                sa_val = sa.get(metric_key)
                dyn_val = dyn.get(metric_key)
                
                rr_str = f"{rr_val:,.1f}" if isinstance(rr_val, (int, float)) else "N/A"
                sa_str = f"{sa_val:,.1f}" if isinstance(sa_val, (int, float)) else "N/A"
                dyn_str = f"{dyn_val:,.1f}" if isinstance(dyn_val, (int, float)) else "N/A"
                
                sa_delta = calculate_delta(rr_val, sa_val)
                dyn_delta = calculate_delta(rr_val, dyn_val)
                
                sa_delta_str = format_delta(sa_delta, higher_better) if sa_delta is not None else "N/A"
                dyn_delta_str = format_delta(dyn_delta, higher_better) if dyn_delta is not None else "N/A"
                
                report_lines.append(f"| {metric_name} | {rr_str} | {sa_str} | {dyn_str} | {sa_delta_str} | {dyn_delta_str} |")
            
            report_lines.append("")
    
    # RR vs SA pairs (without Dynamo)
    report_lines.append(f"### RoundRobin vs Scheme4_GWF 对比: {len(pair_matches_rr_sa)} 组配置（无 Dynamo）")
    report_lines.append("")
    
    if pair_matches_rr_sa:
        report_lines.append("| 配置 | RR Goodput | SA Goodput | ΔGoodput | RR P99 TTFT | SA P99 TTFT | ΔTTFT | RR P99 TPOT | SA P99 TPOT | ΔTPOT |")
        report_lines.append("|---|---|---|---|---|---|---|---|---|---|")
        
        for key, methods in pair_matches_rr_sa:
            rr = methods.get('roundrobin', {})
            sa = methods.get('scheme4_gwf', {})
            
            rr_gp = rr.get('output_token_goodput', 'N/A')
            sa_gp = sa.get('output_token_goodput', 'N/A')
            gp_delta = calculate_delta(rr_gp, sa_gp)
            
            rr_ttft = rr.get('p99_ttft', 'N/A')
            sa_ttft = sa.get('p99_ttft', 'N/A')
            ttft_delta = calculate_delta(rr_ttft, sa_ttft)
            
            rr_tpot = rr.get('p99_tpot', 'N/A')
            sa_tpot = sa.get('p99_tpot', 'N/A')
            tpot_delta = calculate_delta(rr_tpot, sa_tpot)
            
            report_lines.append(
                f"| {key} | {rr_gp} | {sa_gp} | {format_delta(gp_delta, True)} | "
                f"{rr_ttft} | {sa_ttft} | {format_delta(ttft_delta, False)} | "
                f"{rr_tpot} | {sa_tpot} | {format_delta(tpot_delta, False)} |"
            )
        report_lines.append("")
    
    # RR vs Dynamo pairs
    report_lines.append(f"### RoundRobin vs Dynamo 对比: {len(pair_matches_rr_dyn)} 组配置")
    report_lines.append("")
    
    if pair_matches_rr_dyn:
        report_lines.append("| 配置 | RR Goodput | Dyn Goodput | ΔGoodput | RR P99 TTFT | Dyn P99 TTFT | ΔTTFT | RR P99 TPOT | Dyn P99 TPOT | ΔTPOT |")
        report_lines.append("|---|---|---|---|---|---|---|---|---|---|")
        
        for key, methods in pair_matches_rr_dyn:
            rr = methods.get('roundrobin', {})
            dyn = methods.get('dynamo', {})
            
            rr_gp = rr.get('output_token_goodput', 'N/A')
            dyn_gp = dyn.get('output_token_goodput', 'N/A')
            gp_delta = calculate_delta(rr_gp, dyn_gp)
            
            rr_ttft = rr.get('p99_ttft', 'N/A')
            dyn_ttft = dyn.get('p99_ttft', 'N/A')
            ttft_delta = calculate_delta(rr_ttft, dyn_ttft)
            
            rr_tpot = rr.get('p99_tpot', 'N/A')
            dyn_tpot = dyn.get('p99_tpot', 'N/A')
            tpot_delta = calculate_delta(rr_tpot, dyn_tpot)
            
            report_lines.append(
                f"| {key} | {rr_gp} | {dyn_gp} | {format_delta(gp_delta, True)} | "
                f"{rr_ttft} | {dyn_ttft} | {format_delta(ttft_delta, False)} | "
                f"{rr_tpot} | {dyn_tpot} | {format_delta(tpot_delta, False)} |"
            )
        report_lines.append("")
    
    # SA vs Dynamo pairs
    report_lines.append(f"### Scheme4_GWF vs Dynamo 对比: {len(pair_matches_sa_dyn)} 组配置")
    report_lines.append("")
    
    if pair_matches_sa_dyn:
        report_lines.append("| 配置 | SA Goodput | Dyn Goodput | ΔGoodput | SA P99 TTFT | Dyn P99 TTFT | ΔTTFT | SA P99 TPOT | Dyn P99 TPOT | ΔTPOT |")
        report_lines.append("|---|---|---|---|---|---|---|---|---|---|")
        
        for key, methods in pair_matches_sa_dyn:
            sa = methods.get('scheme4_gwf', {})
            dyn = methods.get('dynamo', {})
            
            sa_gp = sa.get('output_token_goodput', 'N/A')
            dyn_gp = dyn.get('output_token_goodput', 'N/A')
            gp_delta = calculate_delta(sa_gp, dyn_gp)
            
            sa_ttft = sa.get('p99_ttft', 'N/A')
            dyn_ttft = dyn.get('p99_ttft', 'N/A')
            ttft_delta = calculate_delta(sa_ttft, dyn_ttft)
            
            sa_tpot = sa.get('p99_tpot', 'N/A')
            dyn_tpot = dyn.get('p99_tpot', 'N/A')
            tpot_delta = calculate_delta(sa_tpot, dyn_tpot)
            
            report_lines.append(
                f"| {key} | {sa_gp} | {dyn_gp} | {format_delta(gp_delta, True)} | "
                f"{sa_ttft} | {dyn_ttft} | {format_delta(ttft_delta, False)} | "
                f"{sa_tpot} | {dyn_tpot} | {format_delta(tpot_delta, False)} |"
            )
        report_lines.append("")
    
    # ---- Aggregate analysis ----
    report_lines.append("## 聚合分析")
    report_lines.append("")
    
    # RR vs SA aggregate
    rr_sa_deltas = []
    for key, methods in by_config.items():
        if 'roundrobin' in methods and 'scheme4_gwf' in methods:
            rr = methods['roundrobin']
            sa = methods['scheme4_gwf']
            # Only compare if token counts match
            rr_tok = rr.get('total_generated_tokens', 0)
            sa_tok = sa.get('total_generated_tokens', 0)
            if rr_tok and sa_tok:
                gp_delta = calculate_delta(rr.get('output_token_goodput'), sa.get('output_token_goodput'))
                ttft_delta = calculate_delta(rr.get('p99_ttft'), sa.get('p99_ttft'))
                tpot_delta = calculate_delta(rr.get('p99_tpot'), sa.get('p99_tpot'))
                if gp_delta is not None:
                    rr_sa_deltas.append({
                        'key': key,
                        'gp_delta': gp_delta,
                        'ttft_delta': ttft_delta,
                        'tpot_delta': tpot_delta,
                        'rr_tokens': rr_tok,
                        'sa_tokens': sa_tok,
                    })
    
    if rr_sa_deltas:
        report_lines.append("### RR vs SA 聚合统计")
        report_lines.append("")
        
        sa_better_gp = sum(1 for d in rr_sa_deltas if d['gp_delta'] > 3)
        sa_worse_gp = sum(1 for d in rr_sa_deltas if d['gp_delta'] < -3)
        sa_same_gp = sum(1 for d in rr_sa_deltas if abs(d['gp_delta']) <= 3)
        
        sa_better_ttft = sum(1 for d in rr_sa_deltas if d['ttft_delta'] is not None and d['ttft_delta'] < -10)
        sa_worse_ttft = sum(1 for d in rr_sa_deltas if d['ttft_delta'] is not None and d['ttft_delta'] > 10)
        sa_same_ttft = sum(1 for d in rr_sa_deltas if d['ttft_delta'] is not None and abs(d['ttft_delta']) <= 10)
        
        report_lines.append(f"- 配置对数: {len(rr_sa_deltas)}")
        report_lines.append(f"- **Goodput**: SA 更优 {sa_better_gp}, SA 较差 {sa_worse_gp}, 相当 {sa_same_gp}")
        report_lines.append(f"- **P99 TTFT**: SA 改善 {sa_better_ttft}, SA 劣化 {sa_worse_ttft}, 相当 {sa_same_ttft}")
        report_lines.append("")
        
        # By model breakdown
        for model in ['LLAMA', 'QWEN']:
            model_deltas = [d for d in rr_sa_deltas if model in d['key']]
            if model_deltas:
                avg_gp = sum(d['gp_delta'] for d in model_deltas) / len(model_deltas)
                avg_ttft = sum(d['ttft_delta'] for d in model_deltas if d['ttft_delta'] is not None) / max(1, sum(1 for d in model_deltas if d['ttft_delta'] is not None))
                report_lines.append(f"**{model}** ({len(model_deltas)} 组): 平均 Goodput Δ = {avg_gp:+.1f}%, 平均 TTFT Δ = {avg_ttft:+.1f}%")
        report_lines.append("")
    
    # RR vs Dynamo aggregate
    rr_dyn_deltas = []
    for key, methods in by_config.items():
        if 'roundrobin' in methods and 'dynamo' in methods:
            rr = methods['roundrobin']
            dyn = methods['dynamo']
            gp_delta = calculate_delta(rr.get('output_token_goodput'), dyn.get('output_token_goodput'))
            ttft_delta = calculate_delta(rr.get('p99_ttft'), dyn.get('p99_ttft'))
            tpot_delta = calculate_delta(rr.get('p99_tpot'), dyn.get('p99_tpot'))
            if gp_delta is not None:
                rr_dyn_deltas.append({
                    'key': key,
                    'gp_delta': gp_delta,
                    'ttft_delta': ttft_delta,
                    'tpot_delta': tpot_delta,
                })
    
    if rr_dyn_deltas:
        report_lines.append("### RR vs Dynamo 聚合统计")
        report_lines.append("")
        
        dyn_better_gp = sum(1 for d in rr_dyn_deltas if d['gp_delta'] > 3)
        dyn_worse_gp = sum(1 for d in rr_dyn_deltas if d['gp_delta'] < -3)
        dyn_same_gp = sum(1 for d in rr_dyn_deltas if abs(d['gp_delta']) <= 3)
        
        report_lines.append(f"- 配置对数: {len(rr_dyn_deltas)}")
        report_lines.append(f"- **Goodput**: Dynamo 更优 {dyn_better_gp}, Dynamo 较差 {dyn_worse_gp}, 相当 {dyn_same_gp}")
        report_lines.append("")
        
        for model in ['LLAMA', 'QWEN']:
            model_deltas = [d for d in rr_dyn_deltas if model in d['key']]
            if model_deltas:
                avg_gp = sum(d['gp_delta'] for d in model_deltas) / len(model_deltas)
                avg_ttft = sum(d['ttft_delta'] for d in model_deltas if d['ttft_delta'] is not None) / max(1, sum(1 for d in model_deltas if d['ttft_delta'] is not None))
                report_lines.append(f"**{model}** ({len(model_deltas)} 组): 平均 Goodput Δ = {avg_gp:+.1f}%, 平均 TTFT Δ = {avg_ttft:+.1f}%")
        report_lines.append("")
    
    # ---- Detailed raw data ----
    report_lines.append("## 完整原始数据")
    report_lines.append("")
    report_lines.append("| 来源 | 方法 | 模型 | 数据集 | Trace | P:D | TPOT | n | Req Goodput | Tok Goodput | Mean TTFT | P99 TTFT | Mean TPOT | P99 TPOT | Succ Reqs | Gen Tokens |")
    report_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    
    for r in sorted(all_results, key=lambda x: (x.get('config_key', ''), x.get('method', ''))):
        report_lines.append(
            f"| {r.get('source','')} | {r.get('method','')} | {r.get('model','')} | "
            f"{r.get('dataset','')} | {r.get('trace','')} | {r.get('pd_ratio','')} | "
            f"{r.get('tpot_ms','')}ms | {r.get('n','')} | "
            f"{r.get('request_goodput','N/A')} | {r.get('output_token_goodput','N/A')} | "
            f"{r.get('mean_ttft','N/A')} | {r.get('p99_ttft','N/A')} | "
            f"{r.get('mean_tpot','N/A')} | {r.get('p99_tpot','N/A')} | "
            f"{r.get('successful_requests','N/A')} | {r.get('total_generated_tokens','N/A')} |"
        )
    
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("*报告由自动分析脚本生成*")
    
    return "\n".join(report_lines)


def main():
    print("Extracting experiment results...")
    all_results = extract_all_results()
    print(f"Found {len(all_results)} total results")
    
    # Print summary
    methods_count = defaultdict(int)
    for r in all_results:
        methods_count[r.get('method', 'unknown')] += 1
    for m, c in sorted(methods_count.items()):
        print(f"  {m}: {c}")
    
    print("\nGenerating report...")
    report = generate_report(all_results)
    
    output_path = os.path.join(OUTPUT_DIR, "three_way_comparison_report.md")
    with open(output_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")
    
    # Also save CSV for further analysis
    csv_path = os.path.join(OUTPUT_DIR, "three_way_comparison.csv")
    if all_results:
        # Collect all keys
        all_keys = set()
        for r in all_results:
            all_keys.update(r.keys())
        all_keys = sorted(all_keys)
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            for r in all_results:
                # Filter out complex types
                clean_r = {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
                writer.writerow(clean_r)
        print(f"CSV data saved to: {csv_path}")
    
    # Print report to stdout too
    print("\n" + "="*80)
    print(report)


if __name__ == '__main__':
    main()
