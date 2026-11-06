# Worked example — how predicted length, eV, and θ=120ms/30ms drive the GWF admission decision

Grounding: real data from `qwen_lmsys_predictor_friendly_4080_600s_strategyA.csv` (n=4080)
and its `predict_request_decode*.csv` predictor output; the formulas are copied verbatim from
`vllm/vllm/v1/core/sched/geometric_window_scheduler.py` (Algorithm 2). Reproduce with
`python3 worked_example.py`.

## 1. The three quantities and where each enters the decision

| Quantity | Symbol | Where it enters | Effect |
|---|---|---|---|
| Predicted output length | `predicted_output_len` | eviction value `eV(r)` via `gen_ratio = generated/predicted` and `rem_ratio = (predicted − generated)/max_model_len` | decides *which* running request is preempted when KV allocation fails |
| Eviction value | `eV(r) = gen_ratio − slack − 0.5·rem_ratio` | running requests are sorted by `(hopeless, eV)` **descending**; the tail (`pop()`) is the preemption victim | keeps high-progress / low-future-cost requests, sacrifices low-progress / high-remaining ones |
| θ = 120 ms | window-feedback target | `every 1s: if τ > θ ∧ \|R\| ≥ A then A ← max(32, A/2) else A ← min(256, A + A/8)` | decides *how many* requests to admit (the window `A`), not the SLO |
| 30 ms | TPOT SLO | `hopeless(r) = (observed TPOT > 30 ms, ≥4 samples) or (TTFT > 1000 ms)` | splits requests into a strictly-lower "hopeless" class, demoted/evicted first |

The two targets are deliberately different: **120 ms bounds the admission window, 30 ms bounds the SLO**.
The scheduler never tries to reach 30 ms with the window — it reaches 120 ms, and lets the
eviction/demotion logic enforce 30 ms.

## 2. Concrete numbers (θ=120 ms vs 30 ms)

At `τ = 98 ms` (the measured per-token time in the flat regime, 65–128 running requests, from the
scheduler's own calibration log) with `|R| = 256 ≥ A`:

| target | window `A` over 4 s |
|---|---|
| θ = 120 ms (paper) | 256 → 256 → 256 → 256 → 256 |
| θ = 30 ms (= SLO, naive) | 256 → 128 → 64 → 32 → 32 |

With θ = 30 ms the window halves every second and collapses to `A_min = 32`, destroying throughput;
with θ = 120 ms it stays at the cap. **This is exactly why the paper calls 120 ms "a reachable operating
point" rather than the 30 ms SLO.**

## 3. Concrete numbers (predicted length → eV → victim)

Snapshot of five running requests (real predicted/actual lengths; `g` = tokens generated so far,
`τ_i` = observed per-token time):

| req | prompt | predicted | actual | g | τ (ms) | hopeless | gen_ratio | rem_ratio | slack | eV |
|---|---|---|---|---|---|---|---|---|---|---|
| 14312 | 40 | 13650 | 149 | 74 | 20.0 | no | 0.005 | 0.414 | 0.33 | −0.535 |
| 5498 | 124 | 31718 | 1024 | 512 | 40.0 | **yes** | 0.016 | 0.952 | 0.00 | −0.460 |
| 16315 | 32 | 28307 | 1231 | 615 | 22.0 | no | 0.022 | 0.845 | 0.27 | −0.667 |
| 45328 | 45 | 27864 | 1225 | 512 | 45.0 | **yes** | 0.018 | 0.835 | 0.00 | −0.399 |
| 25457 | 53 | 22411 | 1024 | 610 | 18.0 | no | 0.027 | 0.665 | 0.40 | −0.705 |

- The two requests with τ > 30 ms (5498, 45328) are **hopeless**: they already missed the TPOT SLO,
  so they form a strictly-lower eviction class and are preempted/demoted **before** any live request.
- Among the three live requests, the lowest `eV` is **req 25457 (−0.705)** → it is the preemption victim.
  It has the highest slack-credit (0.40) yet the largest `rem_ratio` (0.665): its remaining predicted work
  is the most expensive relative to progress already made.

## 4. Caveat we must state honestly (and the fix)

On this run the raw `predicted_output_len` is **~11× the actual length** (median `pred/actual = 10.96`,
97.4% over-predict >2×). That collapses `gen_ratio` toward 0 and inflates `rem_ratio` (e.g. req 14312:
`gen_ratio = 0.005`, `rem_ratio = 0.414`), so `eV` is dominated by the prediction-independent `slack` term.
The fix is a workload/config change, not an algorithm change: re-select requests whose predicted length is
calibrated, or apply the existing `WT_PREDICT_OUTPUT_SCALE` calibration before re-testing the admission gain.
