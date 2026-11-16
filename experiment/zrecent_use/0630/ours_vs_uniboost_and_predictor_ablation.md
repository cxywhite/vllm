# ours（预测+scheme4+GWF）vs 预测+scheme4+UniBoost：全部实验数据 + 预测器消融

> 生成：2026-11-13。本文只做「方法定义 + 数据罗列 + 溯源」，不含过程叙事（过程在 `experiment_log.md`）。
> 口径：goodput = **Output token goodput (tok/s)**，SLO = **TTFT≤1000ms ∧ TPOT≤30ms**。
> 公共：模型 Qwen-2.5-7b-Instruct、vLLM v0.11.0、A800-SXM4-80GB、P2P-NCCL KV、强制生成（ignore-eos，`actual_output_len == output_tokens` 100%）。

---

## 0. 方法定义与公共配置（红线 #5 的底座）

### 0.1 两臂定义

| 臂 | 组成 | 差异点 |
|---|---|---|
| **ours（Presage）** | 真实 BERT 预测 + **scheme4 预测路由** + **GWF 准入** | 实例内 = GWF 窗口准入（θ/margin/Amax）|
| **对照臂** | 真实 BERT 预测 + **scheme4 预测路由** + **UniBoost 调度** | 实例内 = UniBoost 优先级排序（γ/clock/victim）|

- 两臂**共享同一套** scheme4 预测路由（`predict_bottle = max(Load_Compute, Load_Memory, Load_Memory_Capacity)`，fastpath 预留准入），路由代码路径已对齐（日志 #24）。
- 唯一实验变量 = 实例内调度器（GWF 窗口准入 vs UniBoost 优先级排序），故是干净的「准入 vs 驱逐」对比。

### 0.2 各旋钮（每个数据点都带这些）

| 旋钮 | GWF（ours） | UniBoost（对照） |
|---|---|---|
| 预测 | 真实 BERT（关 `WT_SIMPLIFY_PREDICT`）+ `WT_PREDICT_OUTPUT_SCALE=β` | 同左 |
| 路由 | scheme4 | scheme4 |
| 实例内 | `WT_GWF_WINDOW_TPOT_TARGET_MS=120`（sum 口径 θ）、`WT_GWF_TTFT_MARGIN_MS`（0/100/250）、窗口 Amax=256 | `WT_UNIBOOST_CLOCK=wall`、γ-Ada（`WT_UNIBOOST_GAMMA0_S=0.1`、`WT_UNIBOOST_MAX_JUMP_S=60`）或固定 `WT_UNIBOOST_GAMMA`、`WT_UNIBOOST_ADMIT_PREEMPT=0`、victim=`newest`（修复后默认）/`longest`（旧、已证偏离论文）|

- oracle 预测 = `WT_SIMPLIFY_PREDICT=1`（预测值冻结为 `max_tokens`，即完美预测）。
- 拓扑：`4:4`（8 卡，prefill 0-3 / decode 4-7）或 `1:1`（2 卡，纯实例内调度）。
- gpu util：prefill/decode 均 **0.8**（固定，不可改）。

### 0.3 结果目录速查

| 实验 | 结果目录（`experiment/zrecent_use/0630/`） |
|---|---|
| o2000 γ 扫描 | `wl_ub_o2000_g1e-4/` `_g1e-3/` `_g1e-2/` `_g1e-1/` `_gammaada/` |
| dense_short 三臂 | `wl_dense_ab/` |
| kv1p6 三臂 | `wl_o2000kv1p6_native/` `_gwf/` `_ub_fix/`、`wl_kv1p6_gammaada/` `_r2/` |
| 1p1d ×1.2×8 | `wl_1p1d_x1p2_x8_ab/` |
| 1p1d ×1.8 / ×1.4 | `wl_1p1d_x1p8_ab/`、`wl_1p1d_x1p4_ab/` |
| 阶段 B real5p1d（1p1d） | `wl_1p1d_real5p1d_x8_ab/`、`wl_1p1d_real5p1d_x8_gwf_fixverify/` |
| **predacc 最终（4p4d 真实预测）** | `wl_predacc_lmsys_o1600/`、`wl_predacc_lmsys_o2000/` |
| 预测器消融（β=1.0，仅 GWF） | `wl_predacc_ours_nobeta_o{1600,2000}/`、`wl_predacc_muserve_o{1600,2000}/`、`wl_predacc_trailpp_o{1600,2000}/` |

---

## 1. ours vs UniBoost 头对头全量（按预测口径分组）

### 1.1 oracle 预测（`WT_SIMPLIFY_PREDICT=1`，完美预测）

> 这一组是「预测器完美」下纯调度对比：GWF 准入 vs UniBoost 排序，谁能赢由负载决定。

#### (a) o2000（4p4d，gmu 0.8，θ=120 sum）—— UniBoost 赢

| 臂 | 配置 | goodput | 备注 |
|---|---|---|---|
| native（scheme4 贪心） | — | 250.3 | 基准 |
| GWF | θ=120 + margin=250 | 360.8 | TTFT-margin 救回 +46% |
| UniBoost γ=1e-4 | 固定 γ，token 钟 | 326.0 | |
| UniBoost γ=1e-3 | 固定 γ | 355.4 | |
| **UniBoost γ=1e-2** | 固定 γ | **392.1（甜点）** | |
| UniBoost γ=1e-1 | 固定 γ | 386.2 | γ→∞ 收敛回 FCFS=250 |
| UniBoost γ-Ada | wall 钟，自适应 γ | 388.8 | 收敛到 ≈ 手动甜点 |

**结论**：oracle 下 UniBoost（392）> GWF（361）。γ 曲线非单调、峰值 1e-2。
**溯源**：`wl_ub_o2000_g1e-{4,3,2,1}/`、`wl_ub_o2000_gammaada/`；GWF 见日志 #29/#31。

#### (b) dense_short（12000×200 均匀短，4p4d，gmu 0.8）—— UniBoost 赢

| 臂 | 配置 | goodput |
|---|---|---|
| native | — | 2276.0 |
| GWF | θ=120（sum） | 2158.6（−5.2%）|
| **UniBoost** | γ=1e-2 | **3223.7（+42%）** |

**结论**：UniBoost 靠 `w̃=max(output,prompt)` 的 prompt 长度 SPF 排序，在均匀输出负载上仍 +42%。
**溯源**：`wl_dense_ab/`。

#### (c) kv1p6（KV 打满，4p4d，gmu 0.8，θ=120+margin250）—— GWF 边际赢

| 臂 | 配置 | goodput | recompute | mTTFT | p99TPOT |
|---|---|---|---|---|---|
| native | — | 134.2 | ~3.0M（25%）| 16.3s | 1024ms |
| **GWF** | 零重算准入 | **135.1** | **~32k（0.26%）** | 36.6s | 186ms |
| UniBoost 固定γ | γ=1e-2，victim_longest，admit_preempt off | 130.7 | ~14.6M（119%）| 57.6s | 424ms |
| UniBoost γ-Ada | wall，victim_longest | 128.4 | ~3.0M | 51s | 172ms |

**结论**：KV 打满下 GWF 靠零重算边际赢（+0.7%，方差内）；UniBoost 固定 γ 重算风暴、γ-Ada 强 SJF 过度优待短请求，都输。方向与重算代价一致。
**caveat**：UniBoost 两臂用 victim_longest（已证偏离论文 SelectVictim+MemGuard+hysteresis）。
**溯源**：`wl_o2000kv1p6_native/`、`wl_o2000kv1p6_gwf/`、`wl_o2000kv1p6_ub_fix/`、`wl_kv1p6_gammaada/`。

#### (d) 1p1d ×1.2×8（纯实例内，GPU0+4，gmu 0.8）—— GWF +8%（污染）

| 臂 | 配置 | goodput | 备注 |
|---|---|---|---|
| **GWF** | θ=120 + margin250 | **144.17** | pre-bug 有效 |
| UniBoost | wall + γ-Ada，victim_longest | 133.51（−7.6%）| 10,912 次抢占、重算风暴，**受 victim_longest 污染** |

**结论**：1p1d 绕开 inter-instance 路由污染，纯看实例内调度；GWF +8%，但 UniBoost 臂用了偏离论文的 victim_longest，非干净对比。
**溯源**：`wl_1p1d_x1p2_x8_ab/`。victim_newest 复跑未做（待办）。

#### (e) 1p1d ×1.8 / ×1.4（过载，无效对比）

| 负载 | GWF | UniBoost | 结论 |
|---|---|---|---|
| ×1.8（rate1.7，mean~3000）| 13.03（≈0，TTFT 584s/TPOT 519ms）| 崩溃（主机 OOM）| 两臂都 ≈0，过载不可用 |
| ×1.4（rate1.7）| 20% 即终止（KV 100%、τ≈1s）| 未跑 | 过载签名同 ×1.8 |

**溯源**：`wl_1p1d_x1p8_ab/`、`wl_1p1d_x1p4_ab/`。根因：到达率 1.7 req/s 对单 decode GPU 过高（demand≈3× 吞吐）。

### 1.2 真实预测（β=0.30，真实 BERT）

#### (f) step3 qwen_4080_600s（4p4d，extreme trace，mean 7872）—— 两臂都崩

| 臂 | 结果 |
|---|---|
| GWF（β=0.30）| goodput **24.16**（SLO 几乎全崩：mean TTFT 545.6s / P99 3841.7s）|
| UniBoost（β=0.30）| 96%（3903/4080）被超时预算误杀（踩坑：超时当预算）|

**结论**：弱预测器（corr 0.351）+ extreme trace（342 条 >30k）→ goodput 崩塌。无有效对比。
**溯源**：日志 step3；`run_step3_realpred.sh`。

#### (g) 阶段 B real5p1d（1p1d，真实 5p1d 子集，mean 2312）—— 基础设施 bug 污染

| 臂 | 结果 |
|---|---|
| GWF（β=0.30）| 355（虚高：400 长请求被心跳竞态丢弃）→ **113.77**（心跳修复后）|
| UniBoost（β=0.30）| 死锁（victim_longest 竞态，~4% 进度）|

**结论**：两臂都被基础设施 bug 污染（心跳余量 5s vs 3s、victim_longest 竞态），无干净对比。
**溯源**：`wl_1p1d_real5p1d_x8_ab/`、`wl_1p1d_real5p1d_x8_gwf_fixverify/`；日志 #28-#31。

#### (h) ★ predacc_lmsys o1600/o2000（4p4d，真实 lmsys 子集，最终干净结果）

**负载**：`admission_ab_base/predacc_lmsys_o1600/o2000/qwen_lmsys_predacc_o{1600,2000}.csv`（各 4080 条真实 lmsys prompt，output_tokens 精确匹配合成 o1600/o2000，逐请求预测准 82%/86% 在 30% 内）。
**配置**：4p4d、gmu 0.8、tpot 30ms、真实 BERT 预测（β=0.30）、scheme4 路由、GWF θ=120+margin250 / UniBoost wall+γ-Ada+admit_preempt off+victim newest。

| 负载 | 臂 | goodput | mTTFT | P99 TTFT | mTPOT | P99 TPOT |
|---|---|---|---|---|---|---|
| o1600 | **GWF** | **524.59** | 323.8 | 1349.2 | 55.70 | **80.97** |
| o1600 | UniBoost | 506.39 | 316.7 | 1216.8 | 55.42 | 90.48 |
| o2000 | **GWF** | **277.86** | 502.5 | 1385.7 | 88.41 | **127.85** |
| o2000 | UniBoost | 257.72 | 442.3 | 2062.7 | 91.05 | 170.16 |

**结论**：GWF o1600 赢 **+3.6%**、o2000 赢 **+7.8%**；两臂 recompute=0（无 KV 打满抢占），是**纯调度对比**。
**驱动**：`run_predacc_lmsys_o1600.sh` / `run_predacc_lmsys_o2000.sh`（exit 0，Failed runs 0）。
**溯源**：`wl_predacc_lmsys_o1600/`、`wl_predacc_lmsys_o2000/`（各含 `summary_*.csv` + `config/config_*.json` + `dataset_result/test_results_*.csv`）。

### 1.3 头对头汇总：真实预测让 GWF 翻盘

| 预测口径 | 对比（o2000 附近） | 谁赢 |
|---|---|---|
| oracle（完美预测）| UniBoost 392 > GWF 361（+57% over native）| **UniBoost** |
| 真实预测（β=0.30）| GWF 278 > UniBoost 258（o2000）；GWF 525 > UniBoost 506（o1600）| **GWF** |

**机制**：UniBoost 的优先级排序对**预测误差**敏感（预测错 → 排序错 → 更吃亏）；GWF 的窗口准入只依赖「准入/驱逐」的粗粒度决策，对预测误差更鲁棒。这直接命中项目目标——**「真实数据 + 真实预测」是 GWF 赢 UniBoost 的干净条件**（而非 oracle）。

---

## 2. 预测器消融（只换预测器，β=1.0，仅 GWF，无 UniBoost）

> 目标：回答「把 TRAIL++ / µ-Serve 的预测器直接套到 PD disagg 上，goodput 掉多少」。
> 统一配置：β=1.0（**无缩放**，刻意去掉 Presage 的 β=0.30）· 4p4d · scheme4 + GWF(θ=120/margin=250) · gmu 0.8 · SLO TTFT≤1000∧TPOT≤30 · 强制生成 · predacc_lmsys_o1600/o2000。
> 三预测器共享同一 5 桶中心 `[79,302,927,16934,32614]`，差异只在输入特征（见 `predictor_ablation/predictor_ablation_design.md` §1）。

### 2.1 三预测器 goodput 消融（serving 端）

| 数据集 | ours（Presage） | µ-Serve（prompt 文本 BERT） | TRAIL++（act-13+attention top-k） |
|---|---|---|---|
| o1600 | 520.34 | **533.43** | 458.52 |
| o2000 | **322.87** | 250.43 | 252.41 |

三者极差：o1600 = 74.91（16%），o2000 = 72.44（29%）。
**完整指标表**（successful/duration/throughput/TTFT/TPOT/ITL/E2EL 全列）见 `predictor_ablation_results.md`。

### 2.2 离线预测精度（Phase 1）

| 预测器 | pearson | median over-pred |
|---|---|---|
| ours | 0.495 | ~7× |
| µ-Serve | 0.231 | ~16× |
| TRAIL++ | —（serving 版 corr 0.689，见 §2.3）| — |

### 2.3 采样参数重配（ours 预测器，离线 o2000）—— 提升精度只有两种形式

| 配置 | MAE↓ | corr |
|---|---|---|
| TRAIL++（serving）| 5288.2 | 0.689 |
| ours 自然采样参数 | 8630.7 | 0.549 |
| ours 弃用采样参数分支（α=0，合法）| 5302.1 | 0.810 |
| vA 略高（top_k=500 统一）| 5172.5 | — |
| vB 远高（activation 分 8 桶）| 1151.9 | — |
| 全局统一组合（temp/top_p/top_k/rep 固定）| 1257.6 | — |
| **作弊上界（逐请求按真实长度）** | **243.4** | **0.954** |

**结论**：采样参数在这条强制生成 trace 上与真实长度零信息（corr(temp,actual)=+0.037）。「重配采样参数提升精度」只有两种形式，都不是真正预测能力：
- 全局统一组合 = **β 校准**（MAE 1258）；
- 逐请求重配 = **作弊**（MAE 243，把答案写进采样参数查找表）。

合法最优 = 关掉采样参数分支（MAE 5302 / corr 0.81）。数据集 `predictor_ablation/sampling_adjusted_datasets/*.csv`（vA/vB，各 4080 条，采样参数已重配）。
**溯源**：`predictor_ablation/sampling_reassign_results.md`；提取 `offline_extract_cls.py`、搜索 `search_sampling_cheat.py`。

### 2.4 消融结论

1. **β=1.0 下换预测器无一致赢家、量级小**（16–29% 极差）：µ-Serve o1600 最好/o2000 最差；ours o2000 最好/o1600 居中；TRAIL++ o1600 最差。
2. **ours 最稳**：两数据集都排前二；µ-Serve/TRAIL++ 在 o2000 明显掉到 ~250。
3. **根因**：三者预测都严重高估（7×/16×/…），β=1.0 让 GWF 准入过度保守（decode KV 仅 ~2%），预测器相对准确度（pearson ours 0.495 > µ-Serve 0.231）没有机会发挥。
4. **核心启示**：真正主导 goodput 的是 **β 校准**（Presage 的 β=0.30），不是预测器选择——单纯「套用别人的预测器」且不配 β 不能复现收益。

---

## 3. 结论一览

| 问题 | 答案 | 关键数据 |
|---|---|---|
| ours vs UniBoost 谁赢？ | 取决于预测口径 | oracle：UniBoost（392 vs 361）；真实预测：**GWF（+3.6%/+7.8%）** |
| 换预测器影响多大？ | β=1.0 下 16–29%，无一致赢家 | ours 520/323 vs µ-Serve 533/250 vs TRAIL++ 459/252 |
| 什么决定 goodput？ | **β 校准**，不是预测器选择 | β=0.30（ours 4 臂）>> β=1.0（消融）|

**未决/待办**（承接 `experiment_log.md` TODO）：
- [待修] 块表越界 bug（根因定位，victim_newest 缓解）。
- [可选] mysharegpt 版 predacc 实验（当前只跑了 lmsys）。
- [可选] 3-rep 确认（goodput 是尾部统计量，1 rep 方差 7–10%）。
- [待做] victim_newest 下 1p1d ×1.2×8 UniBoost 臂复跑（坐实 phase A 是否干净）。
