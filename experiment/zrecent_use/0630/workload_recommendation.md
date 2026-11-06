# 负载建议（数据驱动，给实测侧）

> 角色分工：本文件由"分析侧"维护（拟合/模拟），给"实测侧"下具体实验参数。
> 依据：`dense_short` 两轮真实 A/B + 拟合出的干净负载 TPOT 曲线。所有数字可追溯到
> `admission_ab_dense_short/*/dataset_result/test_results_*.csv` 与 `log/decode*.log`。

## 1. 现状：第一个正收益负载是 dense_short

| 轮次 | greedy goodput | GWF goodput | Δ | 备注 |
|---|---|---|---|---|
| #6 | 2566.19 | 2637.21 | **+2.8%** | tpot p99 −14.1% |
| #7 | 2326.8 | 2442.6 | **+5.0%** | tpot p99 −13.4%、ttft p99 −20.2%、max −30% |

两轮同向为正，但**绝对值 run-to-run 方差 ~7–9%**（greedy 2566 vs 2327），单轮不可作为结论。

## 2. 拟合结论（决定后续负载设计）

1. **单 decode 实例 GPU KV = 881,616 tokens** → 并发上限 `881616/(prompt+output)`。
   - output=200 → 上限 ~3443；output=2000 → ~428；output=3000 → ~289。
2. **干净负载 decode meas_tpot ≈ 20 + 0.11×running（ms）**，近似线性，**无 5264ms 悬崖**。
   分桶 p50：0-31→19.7、64-95→29.7、128-159→38.1、224-255→47.9、256-287→49.6。
3. **θ=120ms 不可达**：meas_tpot 最大 ~61ms << 120ms → window 永不收缩、恒 256。
   GWF 实际退化为"固定 256 上限 + 淘汰 hopeless"。
4. **goodput 阈值 ≈ running 64**：30ms SLO 对应客户端 tpot≤30ms，而 tpot=30ms 时 running≈64。
   窗口上限 256 是该阈值的 **4 倍** → 这就是增益只有 +5% 的根本原因（封顶无法把请求拉回 SLO 区内）。
5. **负载集中 2–3×**：predictor 路由把请求压到部分 decode 实例（dense_short 实测 decode2/3 卡 256，
   decode0/1 仅 105/115），反而降低触发门槛。

## 3. 建议（按优先级）

### R1（必做）：dense_short 补 3 rep 确认
拿到稳定中位数，才能写进论文。参数：

```bash
export BASE_RESULT_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_dense_short_3rep
export MODEL_TYPE=qwen
export DATASET_BASE_DIR=/root/predict-schedule/experiment/zrecent_use/0630/admission_ab_base
export TRACE_NAME=bimodal_burst
export PD_RATIO_LIST=4:4 TPOT_MS_LIST=30
export COMPARE_EXPERIMENT_VARIANTS="scheme4 scheme4_gwf"
export RUN_REPEAT_PER_CONFIG=3
```

### R2（推荐试，免复现直接迭）：短+中长双峰密集负载（放大 demote）
纯短请求里所有请求"同样 hopeless"，demote 无区分度，增益只来自尾部；
**给一部分中长请求当"该被淘汰的"**，demote 才有真区分度。模拟对比（都摊在 300s）：

| 负载 | greedy 峰值并发 | GWF 峰值 | demote 次数 |
|---|---|---|---|
| 纯短 200 | 154 | 256 | 0 |
| **短 200 + 长 1000（30%）** | **834** | **256** | **13040** |
| 短 200 + 长 2000（30%） | 428 | 256 | 3440 |

长 1000 的区分度最大（greedy 834 vs GWF 256）。建议先试 `LF=0.3, long=1000, N=12000/300s`。

构造（**不要覆盖现有 trace**，用新目录）：
```bash
TRACE_SPAN_S=300 BURST_PERIOD_S=300 BURST_DURATION_S=300 \
REQS_PER_BURST=12000 LONG_FRACTION=0.3 \
OUTPUT_SHORT_TOKENS=200 OUTPUT_LONG_TOKENS=2000 \
OUT_PATH=<新目录>/qwen_lmsys_bimodal_burst_600s.csv \
python3 build_bimodal_burst_trace.py
```
然后用 R1 同样的 env，把 `DATASET_BASE_DIR`/`TRACE_NAME` 指向新目录，先跑 1 rep 看信号。
预期：greedy 的长尾被中长请求拖死 → shorts 饿死；GWF demote 后 shorts 拿回 SLO → Δ 应大于 +5%。

### R3（最高杠杆，需授权）：让窗口上限对齐 SLO
**核心发现**：goodput 的 30ms SLO 对应 running ≈ 64（tpot=30ms），而窗口上限是 256——
**4× 错配**。所以封顶 256 无法把请求拉回 SLO 区，增益被锁在 +5%。两个等效旋钮：

- `WT_GWF_WINDOW_CAP=64`（或 96）：直接把窗口上限压到 SLO 最优点；
- `WT_GWF_WINDOW_TPOT_TARGET_MS=35`：让收缩阈值≈SLO，窗口在 running≈64 收敛。

两者都**只是配置旋钮、不是算法改动**，但属于"调参数赢"，需用户明确同意后才做。默认不做。
若允许，预期增益远大于 +5%（GWF 把并发压在 64 的 SLO 最优区，greedy 冲到数百）。

## 4. 不建议的方向（已验证为坑）

- **超长 output（8000/20000）**：撞 KV 上限，running 被卡在 ~107/~44，GWF 门根本不触发。
- **冲 KV 悬崖（output≈3000 且高密度）**：greedy 会 thrash，单轮 15min+ 且可能卡死（exit 137）。
- **只靠 1 rep 下结论**：方差 7–9%，会误判方向。
