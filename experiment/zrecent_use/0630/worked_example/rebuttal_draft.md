# Rebuttal draft — R1: concrete example of how predicted length / eV / 120ms vs 30ms drive admission

> Reviewer question: "能否举个具体的例子，说明预测长度、驱逐价值以及 120 毫秒/30 毫秒的目标
> 是如何影响准入决策的？……将有助于阐明第二阶段（准入阶段）所带来的具体增益。"

## 1. The three quantities enter three *different* decisions

The GWF intra-instance scheduler (Algorithm 2) has three independent levers, and it is a mistake to
read them as one:

| Quantity | Enters via | Decides |
|---|---|---|
| θ = 120 ms | window feedback `every 1s: if τ>θ ∧ \|R\|≥A then A←max(32,A/2) else A←min(256,A+A/8)` | **how many** requests to admit (window A) |
| 30 ms | `hopeless(r) = (observed TPOT > 30 ms, ≥4 samples) ∨ (TTFT > 1000 ms)` | **which** requests are demoted/evicted first (a strictly-lower class) |
| predicted length | `eV(r) = gen_ratio − slack − 0.5·rem_ratio`, `gen_ratio = generated/predicted`, `rem_ratio = (predicted−generated)/max_len` | **which** running request is preempted when KV allocation fails |

Crucially, **θ and hopelessness are driven by *measured* TPOT, not by the predictor**. Predicted
length enters only the eviction *tie-break* (eV). This is why the admission stage retains value even
when the length predictor is imperfect — its core is measurement-driven, not prediction-driven.

## 2. Concrete example (θ = 120 ms vs 30 ms)

At measured TPOT `τ = 98 ms` (the flat regime, 65–128 running requests, quoted from the scheduler's
own calibration log) with `|R| = 256 ≥ A`:

| target | window A over 4 s |
|---|---|
| θ = 120 ms (paper) | 256 → 256 → 256 → 256 → 256 |
| θ = 30 ms (= SLO, naive) | 256 → 128 → 64 → 32 → 32 |

With θ = 30 ms the window halves every second and collapses to `A_min = 32`, destroying throughput;
with θ = 120 ms it stays at the cap. **This is why 120 ms is a "reachable operating point" and 30 ms
is enforced separately by the eviction/demotion logic** — not by the window.

## 3. Concrete example (predicted length → eV → victim)

Snapshot of five running requests (real `predict_output_mean`/`actual_output_len` from the 4080-request
qwen trace; `g` = tokens generated so far, `τ_i` = observed per-token time):

| req | prompt | predicted | actual | g | τ (ms) | hopeless | gen_ratio | rem_ratio | slack | eV |
|---|---|---|---|---|---|---|---|---|---|---|
| 5498 | 124 | 31718 | 1024 | 512 | 40.0 | **yes** | 0.016 | 0.952 | 0.00 | −0.460 |
| 45328 | 45 | 27864 | 1225 | 512 | 45.0 | **yes** | 0.018 | 0.835 | 0.00 | −0.399 |
| 14312 | 40 | 13650 | 149 | 74 | 20.0 | no | 0.005 | 0.414 | 0.33 | −0.535 |
| 16315 | 32 | 28307 | 1231 | 615 | 22.0 | no | 0.022 | 0.845 | 0.27 | −0.667 |
| 25457 | 53 | 22411 | 1024 | 610 | 18.0 | no | 0.027 | 0.665 | 0.40 | **−0.705** |

- The two `τ > 30 ms` requests (5498, 45328) are **hopeless**: they already missed the SLO, so they
  form a strictly-lower eviction class and are reclaimed **before** any live request.
- Among the three live requests, the lowest `eV` is **req 25457 (−0.705)** → it is the preemption
  victim. It has the highest slack-credit (0.40) yet the largest `rem_ratio` (0.665): its remaining
  predicted work is the most expensive relative to progress already realized.

## 4. Admission-stage gain (isolated A/B, router held fixed)

We fill the gap noted in §6.4 ("ablates routing without isolating the admission controller") by
running **scheme4 (predictor router + stock greedy admission) vs scheme4_gwf (same router + GWF
admission)** on the 4080-request qwen trace, 4P4D, 30 ms TPOT, median of 3 runs.

<!-- PLACEHOLDER: filled from analyze_admission_ab.py when the 6 runs finish -->
| metric (median) | scheme4 (greedy) | scheme4_gwf (GWF) | Δ |
|---|---|---|---|
| output goodput (tok/s) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩% |
| P99 TPOT (ms) | ⟨…⟩ | ⟨…⟩ | ⟨…⟩ |
| preemptions / evicted_hopeless | — | ⟨…⟩ | — |

## 5. Honest caveat (prediction calibration)

The raw `predict_output_mean` is systematically miscalibrated across datasets and serving runs:
`pred/actual` median = 10.96 (offline predictor-friendly-4080), 8.23 (live qwen trace), 12.87 (live
burstgpt trace); 93–100% over-predict >2×. So `gen_ratio` collapses toward 0 and `rem_ratio` saturates —
`eV` degenerates toward `−slack − 0.5`, dominated by the prediction-*independent* slack term. The admission
controller's window + hopeless logic (§2) are unaffected; only the eV tie-break (§3) is weakened. This is a
workload/calibration issue, not an algorithm issue, and we report the gain with this caveat rather than hiding it.
