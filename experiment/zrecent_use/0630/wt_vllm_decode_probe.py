#!/usr/bin/env python3
# [WT] 真实推理语义验证：单卡 vLLM 跑 Qwen-2.5-7B-Instruct，测 batch=1 decode 每步耗时，
#      与「权重字节 / 实测 HBM 带宽」的 roofline 下限对比。
#
# 关键设计：
#   1. 用「不同生成长度求斜率」得到纯 decode 每步时间，抵消掉固定开销
#      （TTFT、进程间通信、Python 侧启动），比直接 (总时间/N) 干净得多。
#   2. 同一模型分别用 enforce_eager=True / False 各跑一遍。
#      True  = 我们线上服务的配置（无 CUDA Graph）
#      False = 有 CUDA Graph
#      两者之差 = 纯 CPU 侧每步 launch 开销 —— 这是判断「慢在 CPU 还是慢在 GPU」的决定性对照。
#   3. 反推「等效权重带宽」= 权重字节 / 每步时间，直接和 HBM 理论/实测比。
#
# 用法：
#   python3 wt_vllm_decode_probe.py --eager 1 --device 1
#   python3 wt_vllm_decode_probe.py --eager 0 --device 1

import argparse
import json
import os
import time

os.environ.setdefault("VLLM_USE_V1", "1")

MODEL_DEFAULT = "/root/myshare/models/Qwen-2.5-7b-Instruct"
# 由 config.json 精确算得（见 wt_gpu_hw_audit 报告）：7.615B 参数，bf16 = 15.23 GB
WEIGHT_GB_DEFAULT = 15.23


def count_params(model_dir):
    """用 transformers 的 config 精确算 bf16 权重字节（含未 tied 的 lm_head）。"""
    import json as _j
    c = _j.load(open(os.path.join(model_dir, "config.json")))
    H = c["hidden_size"]; L = c["num_hidden_layers"]; I = c["intermediate_size"]
    V = c.get("vocab_size"); KVH = c["num_key_value_heads"]; NH = c["num_attention_heads"]
    hd = c.get("head_dim", H // NH)
    attn = H * H + 2 * H * (KVH * hd) + H * H
    mlp = 3 * H * I
    total = L * (attn + mlp) + V * H
    if not c.get("tie_word_embeddings", True):
        total += V * H
    total += L * 2 * H
    return total, total * 2 / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eager", type=int, default=1, choices=[0, 1])
    ap.add_argument("--device", type=int, default=1)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--lengths", default="64,256,448")
    ap.add_argument("--batch", default="1,8")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-util", type=float, default=0.80)
    a = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.device)

    n_params, weight_gb = count_params(a.model)
    lengths = [int(x) for x in a.lengths.split(",")]
    batches = [int(x) for x in a.batch.split(",")]

    from vllm import LLM, SamplingParams

    print("=" * 78)
    print(f"[WT][INFER] 单卡 vLLM decode 探测   device={a.device}  "
          f"enforce_eager={bool(a.eager)}")
    print(f"  模型参数 {n_params/1e9:.3f} B  ->  bf16 权重 {weight_gb:.2f} GB")
    print(f"  max_model_len={a.max_model_len} gpu_mem_util={a.gpu_util} "
          f"max_num_seqs=1024  (与线上服务一致)")
    print("=" * 78)

    t0 = time.perf_counter()
    llm = LLM(model=a.model, dtype="bfloat16", enforce_eager=bool(a.eager),
              max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_util,
              max_num_seqs=1024, disable_log_stats=True)
    print(f"  引擎启动耗时 {time.perf_counter()-t0:.1f} s")

    # 固定的一条 prompt（长度 64 左右），保证每次都是同一份计算
    prompt = ("Explain the architecture of a transformer model in detail, "
              "covering attention, feed-forward layers, and normalization. " * 4)

    results = []
    for bs in batches:
        prompts = [prompt] * bs
        # 预热
        llm.generate(prompts, SamplingParams(max_tokens=16, ignore_eos=True,
                                             temperature=0.0), use_tqdm=False)
        recs = []
        for L in lengths:
            sp = SamplingParams(max_tokens=L, ignore_eos=True, temperature=0.0)
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp, use_tqdm=False)
            dt = time.perf_counter() - t0
            ntoks = sum(len(o.outputs[0].token_ids) for o in outs)
            recs.append((L, ntoks, dt))
            print(f"  batch={bs:<3} max_tokens={L:<4} 实际生成={ntoks:<5} "
                  f"墙钟 {dt:.3f} s  ->  {dt/ (L/bs) *1000 if bs else 0:.2f} ms/token(粗)")
        # 线性拟合：dt = a + b * L  (b = 纯 decode 每步时间)
        n = len(recs)
        sx = sum(r[0] for r in recs); sy = sum(r[2] for r in recs)
        sxx = sum(r[0] * r[0] for r in recs); sxy = sum(r[0] * r[2] for r in recs)
        denom = n * sxx - sx * sx
        slope = (n * sxy - sx * sy) / denom if denom else float("nan")
        inter = (sy - slope * sx) / n
        tpot_ms = slope * 1000
        eff_bw = weight_gb / slope if slope else float("nan")
        print(f"  => batch={bs}: 斜率 {slope*1000:.3f} ms/token  截距 {inter*1000:.1f} ms")
        print(f"     等效权重带宽 = {weight_gb:.2f} GB / {slope*1000:.2f} ms "
              f"= {eff_bw:.0f} GB/s   (HBM 理论 2039, 实测最好 1798)")
        results.append({"batch": bs, "slope_ms": tpot_ms, "intercept_ms": inter * 1000,
                        "eff_weight_gbs": eff_bw, "samples": recs})

    out = (f"/root/myshare/wty_tmp_result/env_fingerprint/"
           f"vllm_decode_eager{a.eager}_dev{a.device}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"enforce_eager": bool(a.eager), "weight_gb": weight_gb,
                   "n_params": n_params, "results": results}, f, indent=2)
    print(f"\n原始结果: {out}")


if __name__ == "__main__":
    main()
