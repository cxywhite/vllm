#!/usr/bin/env python3
"""Turn a hardware CSV (+ optional scheduler logs) into workload-construction advice.

Two modes:

  report (default): post-hoc analysis of one run
      python3 hw_workload_report.py --hw-csv hw_metrics_x.csv \
          --sched-glob 'admission_ab_*/log/decode*.log' [--per-req-csv ...] \
          [--out report.md]

  watch: live projection while a run is in progress (read-only)
      python3 hw_workload_report.py --hw-csv hw_metrics_x.csv --watch --interval 10

IMPORTANT modelling fact (measured, see experiment_log.md #7): vLLM **preallocates**
the KV cache at `gpu_memory_utilization`, so `fb_mb` (framebuffer used) does NOT grow
with concurrency -- a run at running=5 and a run at running=300 show the same fb.
Therefore:
  * concurrency / KV fill  -> use `kv_usage_pct` from the `[WT][NATIVE]`/`[WT][GWF]`
    scheduler log lines (added 2026-11-07), not the GPU CSV;
  * `fb_mb` is still valuable for (a) detecting a foreign GPU tenant before a run,
    (b) detecting allocator growth / fragmentation, (c) confirming the KV pool size;
  * hardware signals that DO track load: `sm_pct` (compute), `mem_pct` (KV read
    bandwidth), `pwr_w`, host `load1`.
KV transfer (P2pNcclConnector) runs over NVLink, so `rxpci/txpci` stay ~0; account
for it analytically instead: 57,344 B/token (Qwen2.5-7B, bf16, 28 layers, 4 KV heads).

Units are suffix-labelled per AGENTS.md (mib / mb_s / pct / ms / tokens).
Read-only: nothing is deleted or moved; only --out is written.
"""
from __future__ import annotations

import argparse
import csv
import glob
import statistics as st
import sys

MiB = 1024.0 * 1024.0

# Qwen2.5-7B: 28 layers, 4 KV heads (GQA), head_dim 128, bf16 -> 2 bytes/elt.
# 2 (K and V) * 28 * 4 * 128 * 2 = 57,344 bytes/token == 56 KiB/token.
QWEN_KV_BYTES_PER_TOKEN = 2 * 28 * 4 * 128 * 2
KV_CAP_TOKENS = 881_616      # decode0 log, kv_cache_utils, 4P4D qwen-7b
MAX_NUM_SEQS = 1024          # --max-num-seqs default in the eval script
GWF_DEMOTE_GATE = 128        # A_max/2, Algorithm 2 line 12
GWF_WINDOW_GATE = 256        # A_max


def _f(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key)
    if v in (None, ""):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def load_hw(path: str) -> tuple[list[dict], bool]:
    rows = list(csv.DictReader(open(path)))
    keys = list(rows[0].keys()) if rows else []
    return rows, any(k.endswith("_sm_pct") for k in keys)


def gpu_ids(rows: list[dict]) -> list[int]:
    ids = set()
    for k in rows[0]:
        if k.startswith("gpu") and "_" in k[3:]:
            try:
                ids.add(int(k[3:k.index("_", 3)]))
            except ValueError:
                pass
    return sorted(ids)


def col(row: dict, is_v2: bool, i: int, kind: str) -> float:
    """kind in {sm, mem, fb, pwr} -- v1 only has util/mem, no fb/pwr."""
    if is_v2:
        m = {"sm": f"gpu{i}_sm_pct", "mem": f"gpu{i}_mem_pct",
             "fb": f"gpu{i}_fb_mb", "pwr": f"gpu{i}_pwr_w"}
        return _f(row, m[kind])
    if kind == "sm":
        return _f(row, f"gpu{i}_util_pct")
    if kind == "fb":
        return _f(row, f"gpu{i}_mem_mib")
    return 0.0


def parse_sched(log_paths: list[str]) -> list[dict]:
    """[WT][NATIVE]/[WT][GWF] key=value lines -> records (t_ms absent on old builds)."""
    recs: list[dict] = []
    for path in log_paths:
        for line in open(path, errors="ignore"):
            if "[WT][" not in line:
                continue
            tail = line[line.index("[WT]["):]
            kv: dict[str, str] = {}
            for tok in tail.split():
                if "=" in tok:
                    k, _, v = tok.partition("=")
                    kv[k] = v
            if "running" not in kv:
                continue
            try:
                recs.append({
                    "t_ms": int(kv.get("t_ms", -1)),
                    "running": int(kv["running"]),
                    "waiting": int(kv.get("waiting", -1)),
                    "kv_pct": float(kv.get("kv_usage_pct", "nan")),
                    "tpot_ms": float(kv.get("meas_tpot_ms", "nan")),
                    "window": int(kv.get("window", -1)),
                    "demoted": int(kv.get("demoted", 0)),
                    "stop_window": int(kv.get("stop_window", 0)),
                    "preempted": int(kv.get("preempted", 0)),
                    "src": "gwf" if "[WT][GWF]" in line else "native",
                })
            except ValueError:
                continue
    recs.sort(key=lambda r: r["t_ms"])
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hw-csv", required=True)
    ap.add_argument("--sched-glob", default="", help="comma-separated globs, e.g. 'dir/log/decode*.log'")
    ap.add_argument("--per-req-csv", default="")
    ap.add_argument("--kv-bytes-per-token", type=float, default=QWEN_KV_BYTES_PER_TOKEN)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.watch:
        return _watch(args)

    rows, is_v2 = load_hw(args.hw_csv)
    if not rows:
        print(f"empty hw csv: {args.hw_csv}")
        return 1
    ids = gpu_ids(rows)
    n_half = len(ids) // 2
    prefill_ids, decode_ids = ids[:n_half], ids[n_half:]   # eval script convention: P GPUs then D GPUs
    dur_s = _f(rows[-1], "ts_s") - _f(rows[0], "ts_s")

    out: list[str] = []
    p = out.append
    p(f"# Hardware report: `{args.hw_csv}`")
    p(f"\n- samples: {len(rows)}, span: {dur_s:.0f} s, schema: {'v2 (dmon)' if is_v2 else 'v1 (query-gpu)'}")
    p(f"- roles by launch convention: prefill GPUs {prefill_ids}, decode GPUs {decode_ids}")

    def gstat(kind: str, gids: list[int]) -> tuple[float, float]:
        vals = [col(r, is_v2, i, kind) for r in rows for i in gids]
        return (st.mean(vals), max(vals)) if vals else (0.0, 0.0)

    # ---- 1. hardware load by role ----------------------------------------
    print("## 1. Hardware load by role")
    print("| role | sm mean/max (%) | mem_pct mean/max (%) | pwr mean (W) | fb min..max (MiB) |")
    print("|---|---|---|---|---|")
    for label, gids in (("prefill", prefill_ids), ("decode", decode_ids)):
        sm_m, sm_x = gstat("sm", gids)
        me_m, me_x = gstat("mem", gids)
        pw_m, _ = gstat("pwr", gids)
        fb = [col(r, is_v2, i, "fb") for r in rows for i in gids]
        print(f"| {label} | {sm_m:.0f}/{sm_x:.0f} | {me_m:.0f}/{me_x:.0f} | {pw_m:.0f} | "
              f"{min(fb):.0f}..{max(fb):.0f} |")
    fb_span = max(col(r, is_v2, i, "fb") for r in rows for i in ids) - \
              min(col(r, is_v2, i, "fb") for r in rows for i in ids)
    print(f"\n- fb span across all GPUs: {fb_span:.0f} MiB (KV cache is preallocated; "
          f"a small span is EXPECTED and does not mean concurrency was low)")

    # ---- 2. load phases ---------------------------------------------------
    print("\n## 2. Load phases (mean sm across all GPUs)")
    tot_sm = [st.mean([col(r, is_v2, i, "sm") for i in ids]) for r in rows]
    thr = max(20.0, 0.5 * max(tot_sm))
    above = [v >= thr for v in tot_sm]
    phases: list[tuple[float, float]] = []
    t0 = None
    for r, a in zip(rows, above):
        t = _f(r, "ts_s")
        if a and t0 is None:
            t0 = t
        elif not a and t0 is not None:
            phases.append((t0, t))
            t0 = None
    if t0 is not None:
        phases.append((t0, _f(rows[-1], "ts_s")))
    print(f"- mean sm {st.mean(tot_sm):.0f}%, peak {max(tot_sm):.0f}%, threshold {thr:.0f}%")
    print(f"- {len(phases)} phase(s) above threshold (first 8):")
    for a, b in phases[:8]:
        print(f"    - {a:6.0f}s .. {b:6.0f}s ({b - a:5.0f} s)")

    # ---- 3. scheduler counters -------------------------------------------
    recs: list[dict] = []
    if args.sched_glob:
        paths: list[str] = []
        for g in args.sched_glob.split(","):
            if g.strip():
                paths += glob.glob(g.strip())
        recs = parse_sched(paths)
    print("\n## 3. Scheduler counters / concurrency vs GWF gates")
    if recs:
        by_src: dict[str, list[dict]] = {}
        for r in recs:
            by_src.setdefault(r["src"], []).append(r)
        print("| src | samples | running max/p50 | waiting max | kv_pct max | meas_tpot p50 (ms) | window min | demoted max/Σ | stop_window max/Σ | preempted max/Σ |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for src, rs in by_src.items():
            run = [r["running"] for r in rs]
            wait = [r["waiting"] for r in rs if r["waiting"] >= 0]
            kvp = [r["kv_pct"] for r in rs if r["kv_pct"] == r["kv_pct"]]
            tp = [r["tpot_ms"] for r in rs if r["tpot_ms"] == r["tpot_ms"] and r["tpot_ms"] > 0]
            win = [r["window"] for r in rs if r["window"] > 0]
            kv_txt = f"{max(kvp):.1f}" if kvp else "n/a"
            tp_txt = f"{st.median(tp):.1f}" if tp else "n/a"
            print(f"| {src} | {len(rs)} | {max(run)}/{st.median(run):.0f} | {max(wait) if wait else -1} | "
                  f"{kv_txt} | {tp_txt} | {min(win) if win else -1} | "
                  f"{max(r['demoted'] for r in rs)}/{sum(r['demoted'] for r in rs)} | "
                  f"{max(r['stop_window'] for r in rs)}/{sum(r['stop_window'] for r in rs)} | "
                  f"{max(r['preempted'] for r in rs)}/{sum(r['preempted'] for r in rs)} |")
        run = [r["running"] for r in recs]
        mx = max(run)
        kvp = [r["kv_pct"] for r in recs if r["kv_pct"] == r["kv_pct"]]
        print(f"\n- peak running {mx} vs demote gate {GWF_DEMOTE_GATE} / window gate {GWF_WINDOW_GATE}")
        if kvp:
            seq_tokens = 256  # 56 prompt + ~200 output, dense-short profile
            print(f"- peak kv_usage {max(kvp):.1f}% → ~{max(kvp)/100*KV_CAP_TOKENS/1000:.0f}k tokens in KV; "
                  f"at {seq_tokens} tok/request that is ~{max(kvp)/100*KV_CAP_TOKENS/seq_tokens:.0f} concurrent slots "
                  f"(cap by --max-num-seqs is {MAX_NUM_SEQS})")
            print(f"- KV cap {KV_CAP_TOKENS:,} tokens ≈ {KV_CAP_TOKENS*QWEN_KV_BYTES_PER_TOKEN/MiB/1024:.1f} GiB "
                  f"per decode instance (this is what fb shows as preallocated)")
    else:
        print("- no scheduler logs matched (`--sched-glob`), skipping concurrency analysis")

    # ---- 4. request mix ---------------------------------------------------
    if args.per_req_csv:
        print("\n## 4. Request mix")
        outs, ins = [], []
        for path in args.per_req_csv.split(","):
            for r in csv.DictReader(open(path.strip(), errors="ignore")):
                try:
                    outs.append(int(float(r["output_tokens"])))
                    ins.append(int(float(r["prompt_len"])))
                except (KeyError, ValueError):
                    continue
        if outs:
            outs.sort()
            ins.sort()
            print(f"- n={len(outs)}, prompt_len p50={ins[len(ins)//2]}, output_tokens "
                  f"p50={outs[len(outs)//2]} p90={outs[int(len(outs)*0.9)]} max={outs[-1]}")
            print(f"- mean output {st.mean(outs):.0f} tok → KV per completed request "
                  f"≈ {st.mean(outs) * QWEN_KV_BYTES_PER_TOKEN / MiB:.1f} MiB")

    # ---- 5. advice --------------------------------------------------------
    print("\n## 5. Next workload knob (rules fired)")
    adv: list[str] = []
    if recs:
        mx = max(r["running"] for r in recs)
        if mx >= GWF_WINDOW_GATE:
            adv.append(f"peak running {mx} ≥ {GWF_WINDOW_GATE}: both demote and window-shrink can fire — "
                       "this is the regime where GWF has a chance to win.")
        elif mx >= GWF_DEMOTE_GATE:
            adv.append(f"peak running {mx} ∈ [{GWF_DEMOTE_GATE},{GWF_WINDOW_GATE}): demote fires, window never shrinks; "
                       f"push ~{GWF_WINDOW_GATE/max(1,mx):.1f}x more concurrency (rate × service time) to reach the full gate.")
        else:
            adv.append(f"peak running {mx} < {GWF_DEMOTE_GATE}: below the demote gate — GWF is pure overhead here. "
                       f"Need ~{GWF_DEMOTE_GATE/max(1,mx):.1f}x more concurrency.")
    sm_pre, sm_pre_x = gstat("sm", prefill_ids)
    sm_dec, sm_dec_x = gstat("sm", decode_ids)
    if sm_pre_x < 50 and sm_dec < 70:
        adv.append(f"neither role is saturated (prefill peak {sm_pre_x:.0f}%, decode mean {sm_dec:.0f}%): "
                   "the box has headroom → raise arrival rate before touching GWF params.")
    if sm_dec > 90:
        adv.append(f"decode sm mean {sm_dec:.0f}% ≈ saturated: more concurrency will only inflate TPOT "
                   "(the regime where GWF's hopless-demote should help).")
    if not is_v2:
        adv.append("this CSV is from the v1 sampler (no sm/mem/power split, no fb): "
                   "use hw_metrics_sampler_v2.py to get the dmon columns.")
    load1 = [_f(r, "load1") for r in rows]
    mem = [_f(r, "mem_avail_mib") for r in rows]
    adv.append(f"host load1 mean {st.mean(load1):.1f} max {max(load1):.1f}; "
               f"mem_avail min {min(mem):.0f} MiB (swap-space 0 → no KV is swapped to CPU)")
    for a in adv:
        print(f"- {a}")

    text = "\n".join(out) + "\n"
    if args.out:
        open(args.out, "w").write(text)
        print(f"\nwrote {args.out}")
    return 0


def _watch(args) -> int:
    import time
    rows, is_v2 = load_hw(args.hw_csv)
    ids = gpu_ids(rows) if rows else list(range(8))
    decode = ids[len(ids) // 2:]
    print(f"[watch] {args.hw_csv}  decode GPUs {decode}  every {args.interval:.0f}s", flush=True)
    seen = 0
    try:
        while True:
            try:
                rows, is_v2 = load_hw(args.hw_csv)
            except Exception:
                rows = []
            if len(rows) > seen:
                new, seen = rows[seen:], len(rows)
                sm_dec = [col(r, is_v2, i, "sm") for r in new for i in decode]
                sm_pre = [col(r, is_v2, i, "sm") for r in new for i in ids[:len(ids)//2]]
                mem_dec = [col(r, is_v2, i, "mem") for r in new for i in decode]
                print(f"[watch] t={_f(new[-1],'ts_s'):7.0f}s  "
                      f"sm decode {st.mean(sm_dec):5.0f}% / prefill {st.mean(sm_pre):5.0f}%  "
                      f"mem_pct decode {st.mean(mem_dec):5.0f}%  load1 {_f(new[-1],'load1'):.1f}", flush=True)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
