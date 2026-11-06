#!/usr/bin/env python3
"""Concurrency probe: measure CLEAN-LOAD decode per-token time vs batch size.

Why this exists (the missing calibration for workload search):
  The scheduler's "cliff" table (74/84/98/97/5264 ms @ running<=8/32/128/256/257+)
  was measured on the OLD 4080 trace that contained 30k-token extreme requests.
  Those monsters thrash the KV cache and inflate the decode step time, so the
  "97 ms @ 129-256" and "257+ cliff" numbers are NOT trustworthy for clean load.
  On clean bimodal8000 we measured decode-side meas_tpot ~23 ms at running<=96.

  This script sweeps batch size on ONE card (same config as the online decode
  instance: enforce_eager, max_model_len 32768, gpu_util 0.8, max_num_seqs 1024)
  and reports the pure decode ms/token (slope over two generation lengths, which
  cancels the fixed TTFT / launch overhead). That gives the real
  TPOT-vs-concurrency curve and locates the actual cliff / goodput-kill point.

Read-only: never deletes or moves anything; writes one JSON under --out.
Units suffix-labelled per AGENTS.md (ms / tokens / s).

Usage (after the A/B run frees the GPUs):
  CUDA_VISIBLE_DEVICES=<free_gpu> python3 wt_concurrency_probe.py \
      --batch 1,8,32,64,96,128,192,256,320,384 --lengths 128,512
"""
from __future__ import annotations
import argparse
import json
import os
import time

os.environ.setdefault("VLLM_USE_V1", "1")
MODEL_DEFAULT = "/root/myshare/models/Qwen-2.5-7b-Instruct"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="1,8,32,64,96,128,192,256,320,384")
    ap.add_argument("--lengths", default="128,512")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-util", type=float, default=0.80)
    ap.add_argument("--max-num-seqs", type=int, default=1024)
    ap.add_argument("--out", default="/root/myshare/wty_tmp_result/env_fingerprint/"
                   "concurrency_probe_qwen.json")
    a = ap.parse_args()

    batches = [int(x) for x in a.batch.split(",")]
    lengths = [int(x) for x in a.lengths.split(",")]

    from vllm import LLM, SamplingParams
    print("=" * 72)
    print("[WT][PROBE] clean-load decode TPOT vs batch  "
          f"max_model_len={a.max_model_len} gpu_util={a.gpu_util} "
          f"max_num_seqs={a.max_num_seqs} enforce_eager=True")
    print("=" * 72)
    t0 = time.perf_counter()
    llm = LLM(model=a.model, dtype="bfloat16", enforce_eager=True,
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
              max_num_seqs=a.max_num_seqs, disable_log_stats=True)
    print(f"engine up in {time.perf_counter()-t0:.1f} s")

    prompt = ("Explain the architecture of a transformer model in detail, "
              "covering attention, feed-forward layers, and normalization. " * 4)

    results = []
    for bs in batches:
        prompts = [prompt] * bs
        # warm up
        llm.generate(prompts, SamplingParams(max_tokens=16, ignore_eos=True,
                                             temperature=0.0), use_tqdm=False)
        recs = []
        for L in lengths:
            sp = SamplingParams(max_tokens=L, ignore_eos=True, temperature=0.0)
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            dt_s = time.perf_counter() - t0
            ntoks = sum(len(o.outputs[0].token_ids) for o in outs)
            recs.append((L, ntoks, dt_s))
        # slope of (dt_s vs L) = pure per-step decode time at this batch
        n = len(recs)
        sx = sum(r[0] for r in recs)
        sy = sum(r[2] for r in recs)
        sxx = sum(r[0] * r[0] for r in recs)
        sxy = sum(r[0] * r[2] for r in recs)
        denom = n * sxx - sx * sx
        slope_s = (n * sxy - sx * sy) / denom if denom else float("nan")
        tpot_ms = slope_s * 1000.0
        results.append({"batch": bs, "tpot_ms_per_token": round(tpot_ms, 3),
                        "samples": recs})
        print(f"  batch={bs:<4} -> decode {tpot_ms:.2f} ms/token")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"model": a.model, "max_model_len": a.max_model_len,
                   "gpu_util": a.gpu_util, "max_num_seqs": a.max_num_seqs,
                   "results": results}, f, indent=2)
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
