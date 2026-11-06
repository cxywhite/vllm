#!/usr/bin/env bash
# Continuously re-simulate the in-flight experiment from its live data.
# Stops when the GWF variant produces results, or the runner exits, or after N cycles.
set -u
cd /root/predict-schedule/experiment/zrecent_use/0630/worked_example || exit 1
BASE=${1:?base result dir}
TRACE=${2:?trace csv}
PID=${3:-0}
MAXC=${4:-60}
OUT=${5:-/root/predict-schedule/experiment/zrecent_use/0630/live_trace.log}

for i in $(seq 1 "$MAXC"); do
  D=$(ls -dt "$BASE"/qwen_* 2>/dev/null | head -1)
  if [ -z "$D" ]; then sleep 30; continue; fi
  {
    echo "===== cycle $i  $(date +%H:%M:%S)  dir=$(basename "$D") ====="
    python3 live_sim.py --dir "$D" --trace "$TRACE" 2>/dev/null | sed -n '1,40p'
  } >> "$OUT"
  # stop when GWF results appear
  if ls "$BASE"/qwen_*_scheme4_gwf/dataset_result/test_results_*.csv >/dev/null 2>&1; then
    echo "GWF_RESULTS_READY" >> "$OUT"; break
  fi
  # stop when runner gone
  if [ "$PID" != "0" ] && ! ps -p "$PID" >/dev/null 2>&1; then
    echo "RUNNER_GONE" >> "$OUT"; break
  fi
  sleep 40
done
echo "LIVE_LOOP_DONE" >> "$OUT"
