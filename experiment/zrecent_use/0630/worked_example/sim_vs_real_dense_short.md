# Simulator vs real — `qwen_lmsys_bimodal_burst_600s.csv`

curve: `measured` (scale 0.7), imbalance 1.0, kv cap 881,616 tok

| metric | real greedy | real GWF | sim greedy | sim GWF |
|---|---|---|---|---|
| goodput (req/s, cluster) | 13.25 | 13.73 | 13.61 | 11.47 |
| Δ goodput GWF vs greedy | — | +3.6% | — | -15.7% |
| max_running (per instance) | 338 | 256 | 296 | 256 |
| max_waiting | 18 | 20 | 0 | 44 |
| demoted (total) | 0 | 1525 | 0 | 615 |
| stop_window (total) | 0 | 179 | 0 | 221 |
| preempted (total) | 0 | 0 | 0 | 0 |
| P50 TTFT (ms) | 823 | 701 | n/a | n/a |
| P99 TTFT (ms) | 5412 | 4334 | n/a | n/a |
| median TPOT (ms) | 28.9 | 29.1 | n/a | n/a |

## Verdict
- **SIGN MISMATCH**: real +3.6% vs sim -15.7% — the sim must not be used to rank workloads until this is fixed.
- demoted count ratio sim/real = 0.4x (615 vs 1525); target 0.7-1.5x — the `is_hopeless()` predicate is too eager when >2x.
- stop_window count ratio sim/real = 1.2x (221 vs 179).
- the sim does not emit TTFT/TPOT percentiles yet — add them before using it to rank workloads on SLO attainment.
