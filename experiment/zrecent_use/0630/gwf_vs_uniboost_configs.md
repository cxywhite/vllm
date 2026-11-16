# GWF vs 贪心 vs UniBoost 完整实验数据汇总表（含全部配置）

生成：2026-11-12。口径：goodput = "Output token goodput (tok/s)"（SLO = TTFT≤1000ms ∧ TPOT≤30ms）；早期 sweep（11-06/11-07）部分只有 req/s 口径，单独列在 §5。
来源：各结果目录 `config/config_*.json` + driver `*.sh` + `experiment_log.md`。

## 1. 公共配置（所有场景一致）

| 项 | 值 |
|---|---|
| 模型 | Qwen-2.5-7b-Instruct（`/root/myshare/models/Qwen-2.5-7b-Instruct`）|
| vLLM | 0.11.0 原生（只加 scheduler 子类，不动核心）|
| 硬件 | A800-SXM4-80GB |
| SLO / 指标 | TTFT≤1000ms ∧ TPOT≤30ms；goodput = Output token goodput (tok/s) |
| 路由 | scheme4 预测路由；完美预测 `WT_SIMPLIFY_PREDICT=1` |
| KV 传输 | P2pNcclConnector，PUT_ASYNC，8 channel，prefill/decode tensor pool 4G |
| 请求数 | 4080（dense_short=12000，burstgpt_real=673）|
| prefill gmu | 0.8（全部）|
| 评估脚本 | `test_evaluation_scheme4_rr_compare.sh` |
| 臂别名 | 贪心 = `scheme4`（预测路由+贪心准入+FCFS）；GWF = `scheme4_gwf`；UniBoost = `scheme4_uniboost` |

## 2. Trace 定义（负载清单，`admission_ab_base/` 下）

| trace 目录 | 生成规则 |
|---|---|
| `qwen_o400` / `o800` / `o1200` / `o1600` / `o1800` / `o2000` / `o2140` / `o2471` / `o2900` / `o4296` | lmsys-chat 4080 条，输出长度封顶到 N（N=目录后缀）|
| `qwen_o1500` | 输出 ~1500 |
| `qwen_o2000_kv1p6` | o2000 顶部 30% 输出 ×1.6 → 怪物 8000–16000 token（KV 打满）|
| `qwen_o2000_kv1p2_1p1d_x8` | o2000 输出 ×1.2 + 时间戳 ×8（rate 0.85/s，demand ~1990 tok/s）|
| `qwen_o2000_kv1p8_1p1d` | o2000 输出 ×1.8 + 1p1d（严重过载）|
| `bimodal_burst` | 12000 × 输出 ~200 均匀短 |
| `qwen_ln2k_cap4k` | 输出 p50 1752 / p90 3342 / max 4000 |
| `burstgpt_real` | 673 条真实 BurstGPT |

## 3. 三臂完整对比（output-token goodput, tok/s）

| 场景 | PD | gmu_p/d | GWF 配置 | UniBoost 配置 | 贪心 | GWF | UniBoost | 结果目录 / 备注 |
|---|---|---|---|---|---|---|---|---|
| o1200 | 4:4 | 0.8/0.8 | τ=EMA per-token θ=120 | — | **3007.0** | 2198.9 (−27%) | — | 日志 #27；native TPOT 29.9ms 恰好压线 |
| o1600 | 4:4 | 0.8/0.8 | sum θ=120，margin∈{0,100,250} | — | 524.6 | 513.9 / **556.0** / 425.0 | — | 日志 #26/#30/#32/#33；margin 甜点 100 |
| o2000 | 4:4 | 0.8/0.8 | sum θ=120，margin∈{0,250} | γ∈{1e-4,1e-3,1e-2,1e-1} | 250.3 | 246.8 / 360.8 | 326 / 355 / **392.1** / 386 | `wl_ub_o2000_g{γ}/`；UniBoost γ=1e-2 最佳 |
| dense_short | 4:4 | 0.8/0.8 | θ=120 + margin 250 | γ=1e-2 | 2276.0 | 2158.6 | **3223.7** (+42%) | `wl_dense_ab/` |
| kv1p6 | 4:4 | 0.8/0.8 | θ=120 + margin 250 | 固定γ=1e-2 / γ-Ada wall | 134.2 | **135.1** (+0.7%) | 130.7 / 128.4 | `wl_o2000kv1p6_{native,gwf,ub_fix}/`；UniBoost 臂 victim_longest |
| burstgpt_real | 4:4 | 0.8/0.8 | θ=120 + margin 250 | — | 56.40 | 56.18 (−0.4%) | — | 日志 #34；KV 受限，GWF 机制够不到 |
| o1500 | 3:4 | 0.8/**0.5** | θ=120 + margin 250 | γ=1e-2（崩溃）| 274.03 | **577.63 (+110%)** | 崩溃 | `wl_util05_o1500/`、`wl_util05_uniboost_ab5/`；**gmu_d=0.5 违反固定约束** |
| ln2k_cap4k | 3:4 | 0.8/0.7 | θ=120 | — | **0.37** req/s | 0.30 req/s (−19%) | — | 日志 #17；KV 富余，GWF 负 |
| ×1.2×8 | **1:1** | 0.8/0.8 | θ=120 + margin 250 | wall+γ-Ada（当时默认 victim_longest）| — | **144.17** | 133.51 (−7.6%，污染) | `wl_1p1d_x1p2_x8_ab/` |
| ×1.8 | 1:1 | 0.8/0.8 | θ=120 + margin 250 | wall+γ-Ada | — | 13.03 (≈0) | 崩溃 | 日志 #607；两臂都不可用 |

## 4. GWF 配置细节（θ / margin / 窗口）

| 参数 | env | 取值 |
|---|---|---|
| 窗口上限 A | `WT_GWF_WINDOW_CAP` | 256（默认）；220 变体（`run_step2_amax220.sh`）|
| TPOT 目标 θ | `WT_GWF_WINDOW_TPOT_TARGET_MS` | 120（sum 单位 = 每 token 30ms；早期 per-token 口径 = 4×SLO 已作废）|
| TTFT 余量 margin | `WT_GWF_TTFT_MARGIN_MS` | 0 / 100 / 250（倒 U：o1600 甜点 100，o2000 甜点 250）|

## 5. UniBoost 配置细节

| 参数 | env | 取值 |
|---|---|---|
| 固定 γ | `WT_UNIBOOST_GAMMA` | 1e-4 / 1e-3 / 1e-2（甜点）/ 1e-1 |
| 墙钟自适应 γ | `WT_UNIBOOST_CLOCK=wall` + `WT_UNIBOOST_GAMMA0_S=0.1` + `WT_UNIBOOST_MAX_JUMP_S=60` | γ-Ada 臂 |
| 准入驱逐 | `WT_UNIBOOST_ADMIT_PREEMPT` | 0（kv1p6/×1.2×8 已关，修 crash）|
| victim 选择 | `WT_UNIBOOST_VICTIM` | longest（旧默认，已证偏离论文）→ **newest（当前默认，修活锁）** |

## 6. 早期 sweep（11-06/11-07，req/s 口径，预测非完美）

| 场景 | 贪心 | GWF | Δ | 备注 |
|---|---|---|---|---|
| burstgpt cap8192 | 1956.75 | 1851.73 | −5.4% | 无悬崖，GWF 空转 |
| bimodal(300短+300长) | 1502.81 | 951.02 | −36.7% | 并发<128，准入没触发 |
| dense_short(#6) | 2566.19 | 2637.21 | **+2.8%** | 第一次 GWF 赢 |
| o1500@gmu0.5(#15) | 0.48 | 0.81 | **+68.8%** | 对方跑，rep 复现 +14.3%，中位数 +41.5% |

## 7. 结论与 caveat（必须同表一起读）

1. **GWF 对贪心正收益只在两处**：`o1500@gmu0.5`（+110%，但 gmu_d=0.5 违反约束）和 `o2000+margin250`（+46%，靠 TTFT-margin 救饿死）。其余 o1200/o1600/dense_short/burstgpt/ln2k 全为负或平。
2. **三臂同框只有 3 处**：o2000（UniBoost 392 > GWF 361 > 贪心 250）、dense_short（UniBoost 3224 > 贪心 2276 > GWF 2159）、kv1p6（GWF 135 > 贪心 134 > UniBoost 131/128）。
3. **污染项**：×1.2×8 的 UniBoost(133.51) 和 kv1p6 的 UniBoost 两臂(130.7/128.4) 都用 victim_longest（已证偏离论文 §3.2/§3.3 的 SelectVictim(π,swap-cost)+MemGuard+hysteresis）；o1500 的 UniBoost 与 ×1.8 两臂都崩溃。
4. **核心未解决**：GWF 至今没有在"干净、无污染、幅度可观"的意义上同时赢过贪心和 UniBoost；唯一大赢（+110%）违反 gmu0.8 约束且 UniBoost 崩了。
