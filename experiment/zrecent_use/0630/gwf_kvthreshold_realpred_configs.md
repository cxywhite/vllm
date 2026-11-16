# GWF vs KV-threshold 真实预测实验数据汇总（含全部配置）

生成：2026-11-15。口径：goodput = "Output token goodput (tok/s)"，SLO = TTFT≤1000ms ∧ TPOT≤30ms。
关键区别：本组**路由用真实 BERT 预测**（`WT_SIMPLIFY_PREDICT=0` + β=0.30），不是早先 `gwf_vs_uniboost_configs.md` 里的 oracle（`WT_SIMPLIFY_PREDICT=1`）。
来源：各结果目录 `config/config_*.json` + `run_*.sh` + `experiment_log.md`。

## 1. 公共配置（所有臂一致）

| 项 | 值 |
|---|---|
| 模型 | Qwen-2.5-7b-Instruct（`/root/myshare/models/Qwen-2.5-7b-Instruct`）|
| vLLM | 0.11.0 + 自研 scheduler 子类（`--scheduler-cls` 注入，默认关）|
| 硬件 | A800-SXM4-80GB × 8 |
| PD | 4:4（prefill GPU 0-3，decode GPU 4-7）|
| gmu | 0.8 / 0.8 |
| SLO / 指标 | TTFT≤1000ms ∧ TPOT≤30ms；goodput = Output token goodput (tok/s) |
| 路由 | scheme4 预测路由；**真实 BERT** `WT_SIMPLIFY_PREDICT=0` + `WT_PREDICT_OUTPUT_SCALE=0.30`（β 只进路由）|
| KV 传输 | P2pNcclConnector，PUT_ASYNC，8 channel，prefill/decode tensor pool 4G |
| 请求数 | 4080 |
| trace | `qwen_o2000_kv1p6`（output p50=688 / p90=11209 / max=13305，monster 长尾）|
| 评估脚本 | `test_evaluation_scheme4_rr_compare.sh` |

## 2. 臂定义与结果（output-token goodput, tok/s）

| 臂 | variant | 实例内调度 | goodput | 成功 | Median TTFT | Mean TTFT | P99 TTFT | Median TPOT | Mean TPOT | P99 TPOT | 吞吐(tok/s) | 目录 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FCFS（贪心）| `scheme4` | stock | **119.72** | 4080/4080 | 463 | 28.1s | 495s | 110 | 131.16 | 973 | 4344 | `wl_o2000kv1p6_fcfs_real/...152814_scheme4` |
| **GWF margin=100** | `scheme4_gwf` | 窗口准入 | **118.80** | 4080/4080 | 618 | 32.7s | 559s | 110.30 | 107.21 | 239.57 | 4647 | `wl_o2000kv1p6_gwf_m100/...164300_scheme4_gwf` |
| PKAS（忠实）| `scheme4_pkas` | shadow 模拟 | 112.65 | 4080/4080 | 456 | 24.5s | 547s | 112.74 | 132.19 | 1027 | 4113 | `wl_o2000kv1p6_pkas/...014630_scheme4_pkas` |
| GWF margin=250 | `scheme4_gwf` | 窗口准入 | 110.52 | 4080/4080 | 733 | 37.2s | 609s | 122 | 115.67 | 249 | 4364 | `wl_o2000kv1p6_gwf_real/...130747_scheme4_gwf` |
| conservative | `scheme4_progress` | 进度预留 1.5× | 91.88 | 4080/4080 | 705 | 254s | 1215s | 186.93 | — | — | 3444 | `wl_o2000kv1p6_conservative_real/...142704_scheme4_progress` |
| **KV-threshold** | `scheme4_kvthreshold` | 阈值准入 0.8 | **90.56** | 4080/4080 | 482 | 68.5s | 903s | 120.50 | 117.87 | 332.73 | 3535 | `wl_o2000kv1p6_kvthreshold/...025216_scheme4_kvthreshold` |

（TTFT 单位 ms，Mean/P99 TTFT 是 s；PKAS 和 conservative 的部分 TPOT 未记录。）

## 3. GWF 配置细节

| 参数 | env | 取值 |
|---|---|---|
| 窗口上限 A | `WT_GWF_WINDOW_CAP` | 256 |
| 窗口下限 Amin | `WT_GWF_WINDOW_FLOOR` | 32 |
| 窗口初值 | `WT_GWF_WINDOW_INIT` | 32（AIMD 慢启动）|
| TPOT 目标 θ | `WT_GWF_WINDOW_TPOT_TARGET_MS` | 120（sum 单位 = 每 token 30ms）|
| TTFT 预算 | `WT_GWF_TTFT_BUDGET_MS` | 1000 |
| TPOT 预算 | `WT_GWF_TPOT_BUDGET_MS` | 30 |
| **TTFT 余量 margin** | `WT_GWF_TTFT_MARGIN_MS` | **250（默认）/ 100（调优）** |
| 绝望请求上限 | `WT_GWF_HOPELESS_MAX_DEFER_MS` | 2000 |

margin 语义：`waited ≥ ttft_budget − margin` 时强制准入（绕过窗口闸）。margin 越大强制准入越早。

## 4. KV-threshold 配置细节

| 参数 | env | 取值 |
|---|---|---|
| KV 占用阈值 | `WT_KV_ADMIT_THRESHOLD` | **0.8**（KV 用到 80% 停准入，留 20% buffer）|

语义：`schedule()` 里若 `kv_cache_manager.usage ≥ 0.8`，把 `max_num_running_reqs` 封到当前 running 数（停止从 waiting 加请求）；低于阈值恢复原 cap。**预测无关、无模拟、无自定义抢占**——审稿人"track KV usage + keep a buffer"的直译版。

## 5. 关键逐请求数据（FCFS vs GWF margin=100，解释 0.8% 差距）

| 指标 | FCFS | GWF(m100) |
|---|---|---|
| SLO 请求数 | 570 (14.0%) | 523 (12.8%) |
| TPOT≤30ms | 578 | 533 |
| TTFT≤1s | 3279 (80.4%) | 2263 (55.5%) |
| SLO 输出 token | 338745 | 313533 |
| 墙钟时间 | 2822s | 2638s |

goodput 拆解：`FCFS/GWF = (338745/313533) / (2822/2638) = 1.0804/1.0698 = +1.0%`。FCFS 多 8% SLO token、GWF 快 6.5%，几乎抵消。

TPOT 直方图（TPOT≤30ms 差来自"超快桶"）：

| TPOT 桶 | FCFS | GWF(m100) | 差 |
|---|---|---|---|
| 10–20 ms | 264 | 184 | **+80** |
| 20–30 ms | 308 | 345 | −37 |
| 0–10 ms | 6 | 4 | +2 |
| **≤30ms 合计** | **578** | **533** | **+45** |

## 6. 结论

1. **GWF(margin=100)=118.80 与 FCFS=119.72 基本打平（−0.8%）**；差距由"10–20ms 超快桶少 80 个请求"驱动（贪婪准入的 batch 波动制造小 batch 快跑窗口，GWF 窗口抹平 batch 丢了它），被 GWF 更快吞吐（+7%）部分抵消。
2. **GWF 默认 margin=250（110.52）反而输 PKAS（112.65）**；调 margin 到 100 才反超。margin 是 GWF 在 kv1p6 上的主控旋钮。
3. **KV-threshold（90.56）是六臂最差**：20% buffer 被 monster 输出增长吃光 → KV 实际顶到 97–99% → 频繁抢占 → Mean TTFT 68.5s。证明"naive keep-a-buffer"在长尾负载上失效。
4. **排序**：FCFS ≈ GWF(m100) > PKAS > GWF(m250) > conservative ≈ KV-threshold。
5. **预测式准入 > 朴素 buffer**：PKAS(112.65) 比 KV-threshold(90.56) 高 24%；GWF(m100) 比 KV-threshold 高 31%。
