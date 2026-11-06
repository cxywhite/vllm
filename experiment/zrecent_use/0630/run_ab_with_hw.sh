#!/usr/bin/env bash
# Attach low-overhead hardware sampling to an admission A/B run.
#
# The eval script itself is NOT modified: this wrapper
#   1. pre-flight GPU check (refuses to start on a busy box -- GPU memory held by a
#      foreign job silently corrupts the A/B, see experiment_log.md pitfall #6);
#   2. starts `hw_metrics_sampler_v2.py` under `nice -n 19`, detached with `setsid`
#      so the eval script's cleanup trap (which kills its own process groups) cannot
#      take the sampler down with it;
#   3. runs `test_evaluation_scheme4_rr_compare.sh` with the caller's env;
#   4. stops the sampler and prints the hardware summary + CSV path.
#
# Sampler cost: ~0.05 % of one core (single persistent `nvidia-smi dmon`, 5 s poll,
# measured utime+stime == 0 ticks over 58 s). It is read-only.
#
# Usage (env is the *only* difference from calling the eval script directly):
#   BASE_RESULT_DIR=$PWD/admission_ab_confirm \
#   COMPARE_EXPERIMENT_VARIANTS="scheme4 scheme4_gwf" PD_RATIO_LIST="4:4" \
#   TPOT_MS_LIST="30" RUN_REPEAT_PER_CONFIG=3 TRACE_NAME=bimodal_burst \
#   DATASET_BASE_DIR=$PWD/admission_ab_base \
#   bash run_ab_with_hw.sh
#
# Knobs of this wrapper itself:
#   HW_INTERVAL_S=5        sampling period (seconds)
#   HW_GPU_BUSY_MIB=2000   pre-flight: used GPU memory above this = box busy
#   HW_ALLOW_BUSY_GPU=1    skip the pre-flight refusal (for deliberate co-runs)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${EVAL_SCRIPT:-$HERE/test_evaluation_scheme4_rr_compare.sh}"
SAMPLER="${SAMPLER:-$HERE/hw_metrics_sampler_v2.py}"
HW_INTERVAL_S="${HW_INTERVAL_S:-5}"
HW_GPU_BUSY_MIB="${HW_GPU_BUSY_MIB:-2000}"
HW_ALLOW_BUSY_GPU="${HW_ALLOW_BUSY_GPU:-0}"
OUT_DIR="${BASE_RESULT_DIR:-$HERE/admission_ab_hw}"
mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
CSV="$OUT_DIR/hw_metrics_${STAMP}.csv"

echo "[hw] wrapper: eval=$EVAL_SCRIPT"
echo "[hw] output csv: $CSV"

# ---------------- 1. pre-flight: is the box ours? ----------------
max_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
  | awk 'BEGIN{m=0} {if ($1+0 > m) m=$1+0} END{print m+0}')
echo "[hw] pre-flight: max GPU memory in use = ${max_used_mib} MiB (threshold ${HW_GPU_BUSY_MIB} MiB)"
if (( max_used_mib > HW_GPU_BUSY_MIB )) && [[ "$HW_ALLOW_BUSY_GPU" != "1" ]]; then
  echo "[hw] REFUSING to start: GPUs already hold ${max_used_mib} MiB." >&2
  echo "[hw] Foreign compute apps (host view):" >&2
  nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv 2>/dev/null | head -12 >&2
  echo "[hw] A/B results would be contaminated. Free the GPUs, or set HW_ALLOW_BUSY_GPU=1 to override." >&2
  exit 2
fi

# ---------------- 2. start sampler ----------------
setsid nice -n 19 python3 "$SAMPLER" --out "$CSV" --interval "$HW_INTERVAL_S" \
  > "${CSV%.csv}.sampler.log" 2>&1 &
SAMPLER_PID=$!
sleep 2
if ! kill -0 "$SAMPLER_PID" 2>/dev/null; then
  echo "[hw] ERROR: sampler died at startup, see ${CSV%.csv}.sampler.log" >&2
  exit 3
fi
echo "[hw] sampler pid=$SAMPLER_PID interval=${HW_INTERVAL_S}s"

stop_sampler() {
  if kill -0 "$SAMPLER_PID" 2>/dev/null; then
    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$SAMPLER_PID" 2>/dev/null || break
      sleep 0.25
    done
    kill -KILL "$SAMPLER_PID" 2>/dev/null || true
  fi
}
trap 'stop_sampler' EXIT INT TERM

# ---------------- 3. run the A/B ----------------
start_s=$(date +%s)
bash "$EVAL_SCRIPT" "$@"
rc=$?
end_s=$(date +%s)

# ---------------- 4. stop + summary ----------------
stop_sampler
trap - EXIT INT TERM

echo
echo "[hw] eval exit code: $rc   wall: $((end_s - start_s)) s"
python3 - "$CSV" <<'PY'
import sys, csv, statistics
path = sys.argv[1]
try:
    rows = list(csv.DictReader(open(path)))
except FileNotFoundError:
    print(f"[hw] no csv written: {path}")
    sys.exit(0)
print(f"[hw] samples: {len(rows)}  ({path})")
for prefix, label in (("gpu", "per-GPU")):
    for i in (0, 4):
        fb = [float(r[f"gpu{i}_fb_mb"]) for r in rows if r.get(f"gpu{i}_fb_mb")]
        sm = [float(r[f"gpu{i}_sm_pct"]) for r in rows if r.get(f"gpu{i}_sm_pct")]
        if fb:
            print(f"[hw] {label} gpu{i}: fb {min(fb):.0f}..{max(fb):.0f} MiB "
                  f"(peak {max(fb):.0f}), sm mean {statistics.mean(sm):.1f}% max {max(sm):.0f}%")
load = [float(r["load1"]) for r in rows if r.get("load1")]
if load:
    print(f"[hw] load1 mean {statistics.mean(load):.1f} max {max(load):.1f}")
PY
exit "$rc"
