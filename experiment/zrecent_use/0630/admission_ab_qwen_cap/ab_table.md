# Admission A/B — `/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_qwen_cap`

## scheme4 (greedy)
- goodput **nan req/s** (throughput nan req/s)
- TTFT p50/p99 nan / nan ms
- TPOT p50/p99 nan / nan ms
- good requests 0/0 = 0.0 %
- running max 455, kv_pct max 100.0
- **preempted_recompute_tokens total 74,439** (per instance: decode0=22,993, decode1=0, decode2=51,446, decode3=0)
- demoted Σ 0, stop_window Σ 0, window min n/a

## scheme4_gwf (GWF)
- goodput **nan req/s** (throughput nan req/s)
- TTFT p50/p99 nan / nan ms
- TPOT p50/p99 nan / nan ms
- good requests 0/0 = 0.0 %
- running max 105, kv_pct max 13.8
- **preempted_recompute_tokens total 0** (per instance: decode0=0, decode1=0, decode2=0, decode3=0)
- demoted Σ 0, stop_window Σ 0, window min 256

## Δ (GWF vs greedy)
- P99 TTFT +nan ms, P99 TPOT +nan ms
- recompute tokens 0 vs 74,439 → -100.0 %
