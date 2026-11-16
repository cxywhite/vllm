#!/usr/bin/env python3
"""Find requests in the original dataset that did NOT complete in a benchmark run.

Usage:
  python3 find_stuck_requests.py <original.csv> <test_results_*.csv>

It prints:
  - total / completed / stuck counts
  - stuck requests grouped by trace time window
  - stuck requests grouped by output token length
  - first N and last N stuck requests with their CSV line numbers
"""

import csv
import sys
from collections import Counter
from pathlib import Path

csv.field_size_limit(sys.maxsize)

ORIG = Path(sys.argv[1])
RESULT = Path(sys.argv[2])

with ORIG.open(newline="", encoding="utf-8") as f:
    orig = list(csv.DictReader(f))
with RESULT.open(newline="", encoding="utf-8") as f:
    done_ids = {r["req_id"] for r in csv.DictReader(f)}

t0 = int(orig[0]["timestamp"])
stuck = []
for i, r in enumerate(orig):
    if r["req_id"] not in done_ids:
        ts = (int(r["timestamp"]) - t0) / 1000
        stuck.append((i, r["req_id"], ts, int(r["output_tokens"]), int(r["prompt_len"])))

if not stuck:
    print(f"All {len(orig)} requests completed. No stuck requests.")
    sys.exit(0)

print(f"Total: {len(orig)}  Completed: {len(done_ids)}  Stuck: {len(stuck)}")

# Time distribution (1-minute bins)
bins = Counter()
for _, _, ts, _, _ in stuck:
    bin_start = int(ts) // 60 * 60
    bins[f"{bin_start}-{bin_start + 60}s"] += 1
print("\nStuck by trace time window:")
for k in sorted(bins, key=lambda x: int(x.split("-")[0])):
    print(f"  {k}: {bins[k]}")

# Output length distribution
out_bins = Counter()
for _, _, _, out, _ in stuck:
    if out <= 1024:
        out_bins["<1K"] += 1
    elif out <= 2048:
        out_bins["1K-2K"] += 1
    elif out <= 3072:
        out_bins["2K-3K"] += 1
    elif out <= 4096:
        out_bins["3K-4K"] += 1
    elif out <= 6144:
        out_bins["4K-6K"] += 1
    else:
        out_bins["6K-8K"] += 1
print("\nStuck output length distribution:")
for k in ["<1K", "1K-2K", "2K-3K", "3K-4K", "4K-6K", "6K-8K"]:
    print(f"  {k}: {out_bins.get(k, 0)}")

# First & last stuck with CSV line numbers
N = 20
print(f"\nFirst {N} stuck (line = CSV line number):")
for idx, rid, ts, out, plen in stuck[:N]:
    print(f"  line {idx+2}: req_id={rid}, trace={ts:.1f}s, output={out}, prompt_len={plen}")

print(f"\nLast {N} stuck:")
for idx, rid, ts, out, plen in stuck[-N:]:
    print(f"  line {idx+2}: req_id={rid}, trace={ts:.1f}s, output={out}, prompt_len={plen}")
