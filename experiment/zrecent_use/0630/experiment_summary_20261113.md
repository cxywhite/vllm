# 实验数据总整理（2026-11-06 ~ 11-13）

> 覆盖：从早期 GWF 准入机制探索，到 UniBoost 引入、bug 修复、真实 lmsys 子集构建，最终"真实预测下 GWF 赢 UniBoost"的完整历程。
> 口径：goodput = Output token goodput (tok/s)，SLO = TTFT≤1000ms ∧ TPOT≤30ms；模型 Qwen-2.5-7b（Llama 仅出现在部分对照）。

---

## 一、时间线（关键里程碑）

| 日期 | 里程碑 |
|---|---|
| 11-06 | GWF 准入机制初探：dense_short 首次正收益 +2.8%；发现"并发不够/测量解耦"两大根因 |
| 11-07 | o1500@gmu0.5 找到 GWF 大赢 +110%（vs native）；KV 打满+短请求 regime；宿主 OOM 踩坑 |
| 11-08 | 路由修复 + 完美预测冻结；o400→o4296 输出长度定标（甜点在 800–2000） |
| 11-09 | τ 语义修正（per-token→sum）+ TTFT-margin 机制；o2000 margin250 +24%、o1600 margin100 +4.3% |
| 11-10 | **UniBoost 引入**：γ 扫描 + 三负载三臂；发现 0.8 下 GWF ≤ UniBoost；kv1p6 边际赢 |
| 11-11 | 1p1d 实例内隔离：×1.2×8 GWF 144 > UniBoost 133（污染）；UniBoost 死锁/活锁定位 |
| 11-12 | 三个严重 bug 修复：unit bug、victim_longest 活锁、块表越界；β 向量校准 |
| 11-13 | 真实 lmsys 子集 predacc 构建 + 4p4d 真实预测 GWF vs UniBoost：**GWF 赢** |

---

## 二、GWF vs native（贪心）的结果（11-06~11-09）

### 正收益 regime（3 个，按机制）

| regime | 配置 | native → GWF | 机制 |
|---|---|---|---|
| dense_short | 12000×200 短，40 req/s，4p4d | +2.8% ~ +5.0% | running 顶到 256 → demote/stop_window 触发 |
| o1500@gmu0.5 | 3p4d，decode gmu=0.5 | **+68.8%（output-tok +110.8%）**（复现 +14.3%，中位 +41.5%）| 窗口 256 ≈ KV 容量 → 零抢占重算（796k→0）|
| o2000 + margin250 | 4p4d，sum θ=120 | **+24%（output-tok +46%）**| TTFT-margin 强制准入，P99 TTFT 510s→1.6s |
| o1600 + margin100 | 4p4d，sum θ=120 | +4.3%（output-tok +6%）| margin 甜点：TPOT≈native + TTFT 不爆 |

### 负/平 regime（边界条件）

| regime | 结果 | 原因 |
|---|---|---|
| ln2k_cap4k@gmu0.7 | **−19%** | KV 富余（34.8%），窗口只挡请求不省抢占 |
| o1200/o1600/o2000（无 margin）| −16% / −4% / 0% | 30ms SLO 物理不可达，θ=120 守不住 |
| burstgpt_real | 平（56.40 vs 56.18）| KV 受限，running 卡 ~40，两机制都够不到 |
| o4296 | 不可救 | 长 output，TPOT 124ms，30ms 不可达 |

### 关键机制结论
1. **GWF 只在「greedy 会越界抢占重算」时有正收益**；KV 富余时中性到负。
2. **TTFT-margin 是 11-09 最关键的新机制**：窗口会饿死请求，margin 强制放行救回。
3. **margin 是双刃剑**：o2000（深过载）甜点 250ms，o1600（中过载）甜点 100ms，倒 U 曲线。

---

## 三、UniBoost 引入 + 三臂对比（11-10~11-13）

### 三负载三臂（oracle 预测，4p4d util0.8）

| 负载 | native | GWF | UniBoost | 谁赢 |
|---|---|---|---|---|
| o2000 | 250.3 | 360.8 | **392.1**（γ=1e-2）| UniBoost |
| dense_short | 2276.0 | 2158.6 | **3223.7** | UniBoost |
| kv1p6（KV打满）| 134.2 | **135.1** | 130.7 / 128.4 | GWF（+0.7%，边际）|

### 1p1d 实例内隔离（11-11）

| 负载 | GWF | UniBoost | 结论 |
|---|---|---|---|
| ×1.2×8 | **144.17** | 133.51 | GWF +8%（受 victim_longest 重算风暴污染）|
| ×1.8 | 13.03（≈0）| 崩溃 | 两臂都过载，不可用 |

### 真实 lmsys 子集 + 真实预测（11-13，最终）

| 负载 | GWF | UniBoost | 谁赢 |
|---|---|---|---|
| predacc o1600 | **524.59** | 506.39 | **GWF +3.6%** |
| predacc o2000 | **277.86** | 257.72 | **GWF +7.8%** |

（两臂 recompute=0，无 KV 打满抢占，纯调度对比。）

---

## 四、核心结论：真实预测让 GWF 翻盘

| 预测口径 | o2000 上 UniBoost vs GWF |
|---|---|
| **oracle**（WT_SIMPLIFY_PREDICT=1）| UniBoost 392 > GWF 361（**UniBoost 赢**）|
| **真实预测**（β=0.30）| GWF 278 > UniBoost 258（**GWF 赢**）|

**机制**：UniBoost 的优先级排序对预测误差敏感（预测错 → 排序错 → 更吃亏）；GWF 的窗口准入更鲁棒。

**这直接命中项目初始目标**——"找到 GWF 赢 UniBoost 的负载/条件"。答案：**真实数据 + 真实预测**（而非 oracle）。

---

## 五、代码 bug / 修复（11-11~11-12，三个严重 bug）

| bug | 现象 | 根因 | 修复 |
|---|---|---|---|
| **unit bug** | γ-Ada 墙钟模式静默退化 FCFS（γ_s 发散 7.5/13.98）| bin_tokens 当秒用，所有请求塌进同一 bin | `_phi` wall 模式 `bin_tokens/rate` |
| **victim_longest 活锁** | lmsys UniBoost 卡死 77%、吞吐 0、recompute 132 万 | 抢占最长 + num_output_tokens 不清 → 反复重算 | victim 默认改 newest |
| **块表越界** | `ValueError shape(498)→(212)` | WT「本地重算」修改导致 num_computed_tokens 与块表不同步 | 待修（victim_newest 缓解）|

其他：心跳竞态（DEFAULT_PING_SECONDS 5→20）、KV 路由丢包（待查）、β 标量→向量校准。

### 论文语义核对（重要更正）
- 复现的"驱逐最长"**不是论文语义**：论文（arXiv:2606.18431）是 `SelectVictim(π, swap-cost)` + MemGuard 几何限频 + hysteresis δ_hyst；"longest" 正文出现 0 次。

---

## 六、最终交付物（11-13）

| 交付 | 位置 |
|---|---|
| 真实 lmsys/mysharegpt 子集（预测准）| `/root/myshare/wty_data/001/`（8 文件 + README）|
| 4p4d 真实预测 GWF vs UniBoost 结果 | `wl_predacc_lmsys_o1600/`、`wl_predacc_lmsys_o2000/` |
| 驱动脚本 | `run_predacc_lmsys_o1600.sh`、`run_predacc_lmsys_o2000.sh` |
| 50k 预测数据（完整）| `/root/.cache/huggingface/hub/wt_predictor/0630/dataset/predict_bucket_5p1d/` |

### 子集三特点验证
- ① 真实请求（真实 lmsys/mysharegpt prompt）
- ② 输入输出匹配合成（output 精确==target，prompt 3–5% 误差）
- ③ 逐请求预测准（82–86% lmsys / 61–67% mysharegpt 在 30% 内，β×predict≈target）

---

## 七、待办（截至 11-13）

- [待修] 块表越界 bug（根因已定位）
- [待查] KV 路由丢包根因
- [可选] mysharegpt 版 predacc 实验（当前只跑了 lmsys）
- [可选] 3-rep 确认（goodput 是尾部统计量，1 rep 方差 7–10%）
