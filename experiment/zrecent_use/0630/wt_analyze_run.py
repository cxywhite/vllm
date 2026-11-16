#!/usr/bin/env python3
# [WT] 单个 run 结果分析：从 dataset_result/test_results_*.csv 重算达标率与分位数。
#
# 为什么要自己算：bench 汇总只给 mean/median，而 goodput 的达标条件
# （TTFT<=1000ms 且 TPOT<=30ms）由分布**左尾**决定，mean/median 完全看不出来。
#
# ⚠️ CSV 里 latency / ttft / tpot 单位是**秒**，要 *1000 换 ms。
# ⚠️ 必须用 csv.reader（文件里有内嵌换行），不能用 wc -l。
#
# 用法：python3 wt_analyze_run.py <结果目录> [TTFT_SLO_ms] [TPOT_SLO_ms]

import csv
import glob
import os
import sys

csv.field_size_limit(sys.maxsize)


def pct(v, p):
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(len(v) - 1, max(0, int(len(v) * p)))]


def main():
    d = sys.argv[1]
    ttft_slo = float(sys.argv[2]) if len(sys.argv) > 2 else 1000.0
    tpot_slo = float(sys.argv[3]) if len(sys.argv) > 3 else 30.0

    fs = glob.glob(os.path.join(d, "dataset_result", "test_results_*.csv"))
    if not fs:
        print(f"找不到 test_results_*.csv: {d}/dataset_result/")
        return 1
    f = sorted(fs)[-1]
    rows = list(csv.DictReader(open(f, newline="")))
    print(f"[WT] {os.path.basename(d)}")
    print(f"     结果文件 {os.path.basename(f)}  记录 {len(rows)} 条")

    ttft = [float(r["ttft"]) * 1000 for r in rows if r.get("ttft") not in (None, "")]
    tpot = [float(r["tpot"]) * 1000 for r in rows
            if r.get("tpot") not in (None, "") and r["tpot"] != ""]
    # tpot 为空 = 只生成 1 个 token，无 TPOT
    out = [float(r["output_tokens"]) for r in rows if r.get("output_tokens")]
    exp = [float(r["expect_output_len"]) for r in rows if r.get("expect_output_len")]

    short = [r for r in rows if r.get("output_tokens") and r.get("expect_output_len")
             and float(r["output_tokens"]) < float(r["expect_output_len"])]

    print(f"     TTFT(ms)  p5={pct(ttft,.05):8.1f} p50={pct(ttft,.5):8.1f} "
          f"p90={pct(ttft,.9):8.1f} p99={pct(ttft,.99):8.1f}")
    print(f"     TPOT(ms)  p5={pct(tpot,.05):8.2f} p50={pct(tpot,.5):8.2f} "
          f"p90={pct(tpot,.9):8.2f} p99={pct(tpot,.99):8.2f}   (n={len(tpot)})")
    print(f"     输出长度  p50={pct(out,.5):8.0f} p90={pct(out,.9):8.0f} "
          f"p99={pct(out,.99):8.0f} 合计={int(sum(out))} 期望合计={int(sum(exp))}")

    n = len(rows)
    ok_ttft = sum(1 for v in ttft if v <= ttft_slo)
    ok_tpot = sum(1 for v in tpot if v <= tpot_slo)
    # 只生成 1 token 的请求没有 tpot，按 TPOT 达标算（它不违反 TPOT 约束）
    def is_ok(r):
        if r.get("ttft") in (None, ""):
            return False
        if float(r["ttft"]) * 1000 > ttft_slo:
            return False
        return r.get("tpot") in (None, "") or float(r["tpot"]) * 1000 <= tpot_slo
    ok_both = sum(1 for r in rows if is_ok(r))
    print(f"\n     达标 TTFT<={ttft_slo:.0f}ms : {ok_ttft}/{n} = {ok_ttft/n*100:.2f}%")
    print(f"     达标 TPOT<={tpot_slo:.0f}ms : {ok_tpot}/{len(tpot)} "
          f"= {ok_tpot/max(1,len(tpot))*100:.2f}%")
    print(f"     双达标（按请求）        : {ok_both}/{n} = {ok_both/n*100:.2f}%")

    # ---- 按 token 加权的 goodput：复现 bench 汇总里的 "Output token goodput" ----
    # 关键：bench 的 goodput 是 token 加权，不是请求加权。长请求（3 万+ token）
    # 几乎必然 TPOT 超标，所以它们贡献了大量 throughput 但对 goodput 零贡献。
    def ntok(r):
        try:
            return float(r["output_tokens"])
        except (TypeError, ValueError):
            return 0.0

    def ms(r, k):
        try:
            return float(r[k])
        except (TypeError, ValueError):
            return None

    t0 = min(x for x in (ms(r, "send_timestamp_ms") for r in rows) if x is not None)
    t1 = max((ms(r, "send_timestamp_ms") or 0) + (ms(r, "latency") or 0) * 1000
             for r in rows)
    dur = (t1 - t0) / 1000.0
    gp_tok = sum(ntok(r) for r in rows if is_ok(r))
    tot_tok = sum(ntok(r) for r in rows)
    print(f"\n     时长(自算)              : {dur:.1f} s")
    print(f"     throughput              : {tot_tok/dur:9.1f} tok/s")
    print(f"     goodput（token 加权）   : {gp_tok/dur:9.1f} tok/s"
          f"   <- 与 bench 汇总对比")
    print(f"     达标请求贡献的 token 占比: {gp_tok/max(1,tot_tok)*100:.2f}%")
    print(f"     达标请求的平均输出长度   : {gp_tok/max(1,ok_both):8.0f}")
    print(f"     全部请求的平均输出长度   : {tot_tok/max(1,n):8.0f}")


    if short:
        print(f"\n     ⚠️ 未跑满期望长度的请求 {len(short)} 个，"
              f"缺口 {int(sum(exp)-sum(out))} tokens")
        for r in sorted(short, key=lambda x: -float(x["output_tokens"]))[:5]:
            print(f"        req {r['req_id']}: 实际 {r['output_tokens']} / 期望 "
                  f"{r['expect_output_len']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
