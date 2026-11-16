# 四个关键实验详细记录 + 统一命名方案

生成：2026-11-12。口径：goodput = Output token goodput (tok/s)，SLO = TTFT≤1000ms ∧ TPOT≤30ms。
公共：模型 Qwen-2.5-7b-Instruct、vLLM 0.11.0、A800-80GB、scheme4 预测路由（完美预测）、P2P-NCCL KV、prefill gmu=0.8、请求 4080。

---

## 一、统一命名方案

### 1.1 旧命名的问题

| 旧名 | 问题 |
|---|---|
| `o1600` / `o2000` | "o" 看不出来是「输出均值」还是「封顶」；不含尾部（p90）和到达率 |
| `kv1p6` | "1p6" 是 ×1.6 的加密写法，看不懂 |
| `×1.2×8` | 两个乘数哪个是输出、哪个是到达率，完全不可读 |

### 1.2 新命名规则（trace 目录，统一格式）

```
qwen_out{均值}k_p90{p90}k_r{到达率}
```

| 字段 | 含义 | 说明 |
|---|---|---|
| `out{均值}k` | 输出 token 均值 | k=千，保留 1 位小数 |
| `p90{p90}k` | 输出 token P90 | 尾部，决定 KV 打满程度 |
| `r{到达率}` | 到达率 req/s | 决定过载程度 |
| 追加 `_cap{max}k` | 输出封顶 | 仅当 max 明显低于自然尾部时（如 o4296 封 8k、ln2k 封 4k）|
| prompt | 不重复写 | lmsys 系列恒 p50=57/p90=203；非 lmsys（burstgpt）单独标注 |

### 1.3 映射表（旧名 → 新名，实测聚合值）

| 旧名 | 输出均值 | 输出 P90 | 输出 max | 到达率 | 新名 |
|---|---|---|---|---|---|
| qwen_o400 | 400 | 1401 | 1663 | 6.82 | `qwen_out0.4k_p90_1.4k_r6.8` |
| qwen_o1200 | 1200 | 4203 | 4990 | 6.82 | `qwen_out1.2k_p90_4.2k_r6.8` |
| qwen_o1600 | 1600 | 5605 | 6653 | 6.82 | `qwen_out1.6k_p90_5.6k_r6.8` |
| qwen_o2000 | 2000 | 7006 | 8316 | 6.82 | `qwen_out2.0k_p90_7.0k_r6.8` |
| qwen_o4296 | 4296 | 8000 | 8000 | 6.82 | `qwen_out4.3k_p90_8.0k_cap8k_r6.8` |
| qwen_o1500 | 1500 | 5254 | 6237 | 6.82 | `qwen_out1.5k_p90_5.3k_r6.8` |
| qwen_o2000_kv1p6 | 3006 | 11209 | 13305 | 6.82 | `qwen_out3.0k_p90_11.2k_r6.8` |
| qwen_o2000_kv1p2_1p1d_x8 | 2335 | 8407 | 9979 | 0.85 | `qwen_out2.3k_p90_8.4k_r0.85` |
| bimodal_burst | 200 | 200 | 200 | 40 | `qwen_out0.2k_p90_0.2k_r40` |
| qwen_ln2k_cap4k | 1940 | 3338 | 4000 | 6.82 | `qwen_out1.9k_p90_3.3k_cap4k_r6.8` |
| burstgpt_real | 9269 | 32704 | 32736 | 2.24 | `qwen_out9.3k_p90_32.7k_r2.2`（非 lmsys，prompt p50=51）|

> **注意**：本方案只改「trace 标签」，不物理重命名已有目录（会破坏脚本/日志/config 里的 `dataset_path` 引用）。若需物理重命名，另开任务并同步改所有引用。

---

## 二、四个关键实验详细记录

### 实验 1：`qwen_out1.6k_p90_5.6k_r6.8`（旧 o1600）—— GWF margin 甜点

**trace**：qwen_o1600，n=4080，输出均值 1600 / p50 551 / p90 5605 / p99 6650 / max 6653；prompt p50 57；到达率 6.82 req/s。
**config**：4p4d，gmu 0.8/0.8，tpot 30ms；GWF = sum θ=120，margin ∈ {0,100,250}。

| 臂 | goodput | 相对 native |
|---|---|---|
| 贪心 (scheme4) | 524.6 | — |
| GWF margin=0 | 513.9 | −2% |
| GWF margin=100 | **556.0** | **+6%** |
| GWF margin=250 | 425.0 | −19% |

**结论**：margin 倒 U，甜点 100ms。margin=250 抬 TPOT（40→77ms）净亏；margin=0 空转；100 恰好 TPOT≈native 且 TTFT 不爆。
**目录/日志**：日志 #30/#32/#33（#26 的 per-token τ 结果已作废）。
**caveat**：native TPOT median 54ms 已 >30ms SLO，GWF 只能小胜。

### 实验 2：`qwen_out3.0k_p90_11.2k_r6.8`（旧 kv1p6）—— KV 打满，GWF 唯一略赢

**trace**：qwen_o2000_kv1p6（= o2000 顶部 30% 输出 ×1.6），n=4080，输出均值 3006 / p90 11209 / p99 13299 / max 13305；到达率 6.82。
**config**：4p4d，gmu 0.8/0.8，tpot 30ms；GWF = sum θ=120 + margin250；UniBoost = 固定γ=1e-2（admit_preempt off）/ γ-Ada wall。

| 臂 | goodput | recompute | mTTFT | p99TPOT |
|---|---|---|---|---|
| 贪心 | 134.2 | 3.0M (25%) | 16.3s | 1024ms |
| GWF | **135.1** | 32k (0.26%) | 36.6s | 186ms |
| UniBoost 固定γ | 130.7 | 14.6M (119%) | 57.6s | 424ms |
| UniBoost γ-Ada | 128.4 | 3.0M (25%) | 51s | 172ms |

**结论**：KV 打满 regime，GWF 零重算准入略赢（+0.7%，方差内）；UniBoost 固定γ 重算风暴、γ-Ada 强 SJF 过度优待短请求，都输。
**目录**：`wl_o2000kv1p6_native/`、`wl_o2000kv1p6_gwf/`、`wl_o2000kv1p6_ub_fix/`。
**caveat**：UniBoost 两臂都用 victim_longest（已证偏离论文 SelectVictim(π,swap-cost)+MemGuard+hysteresis）。

### 实验 3：`qwen_out2.0k_p90_7.0k_r6.8`（旧 o2000）—— UniBoost γ 扫描最佳

**trace**：qwen_o2000，n=4080，输出均值 2000 / p90 7006 / max 8316；到达率 6.82。
**config**：4p4d，gmu 0.8/0.8，tpot 30ms；GWF = sum θ=120，margin ∈ {0,250}；UniBoost = γ ∈ {1e-4,1e-3,1e-2,1e-1}。

| 臂 | goodput |
|---|---|
| 贪心 | 250.3 |
| GWF margin=0 | 246.8 (−1.4%) |
| GWF margin=250 | 360.8 (+46%) |
| UniBoost γ=1e-4 | 326.0 |
| UniBoost γ=1e-3 | 355.4 |
| UniBoost γ=1e-2 | **392.1** |
| UniBoost γ=1e-1 | 386.2 |

**结论**：UniBoost γ=1e-2 甜点 392.1，压过 GWF(margin250) 360.8。γ 曲线非单调（γ→∞ 收敛回 FCFS=250）。
**目录**：`wl_ub_o2000_g1e-4/` `_g1e-3/` `_g1e-2/` `_g1e-1/`；GWF 见日志 #29/#31。
**caveat**：GWF 靠 margin250 从 −1.4% 拉到 +46%，主因 TTFT 饿死被 margin 救回；TPOT 仍 80ms≈2.7×SLO。

### 实验 4：`qwen_out2.3k_p90_8.4k_r0.85`（旧 ×1.2×8，1p1d）—— GWF +8%（污染）

**trace**：qwen_o2000_kv1p2_1p1d_x8（o2000 输出 ×1.2 + 时间戳 ×8），n=4080，输出均值 2335 / p90 8407 / max 9979；到达率 0.85 req/s（demand ~1990 tok/s ≈1.3–1.6× 单 decode 吞吐）。
**config**：**1p1d**（prefill GPU0 / decode GPU4），gmu 0.8/0.8，tpot 30ms；GWF = θ=120 + margin250；UniBoost = wall+γ-Ada（`WT_UNIBOOST_CLOCK=wall GAMMA0_S=0.1 MAX_JUMP_S=60` + admit_preempt off，当时默认 victim_longest）。

| 臂 | goodput |
|---|---|
| 贪心 | 未跑 |
| GWF | **144.17** |
| UniBoost | 133.51（−7.6%）|

**结论**：GWF +8%，但 UniBoost 臂用 victim_longest → 10,912 次抢占、5.45M 重算（base 计数）/1.55M（UniBoost 计数），属污染；victim_newest 重跑未做。
**目录**：`wl_1p1d_x1p2_x8_ab/`。
**caveat**：1p1d 纯实例内调度（绕开 inter-instance 路由污染）；GWF 臂 pre-bug 有效，UniBoost 臂需 victim_newest 重验。

---

## 三、四个实验一句话对照

| 实验（新名） | 核心结论 |
|---|---|
| out1.6k（o1600）| GWF margin=100 小胜 +6%，margin 倒 U |
| out3.0k_p90_11.2k（kv1p6）| KV 打满，GWF 零重算略赢 +0.7%，UniBoost 重算风暴 |
| out2.0k（o2000）| UniBoost γ=1e-2 最佳 392 > GWF 361 > 贪心 250 |
| out2.3k_r0.85（×1.2×8）| GWF +8%，但 UniBoost 受 victim_longest 污染 |
