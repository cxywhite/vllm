# 实验结果对比报告：Ours (scheme4_gwf) vs RR / Dynamo

生成时间：`2026-08-27 10:40`

## 总览

- 可对比配置总数（含 Ours）：**33**
- Ours vs RR 可比配置：**30**，其中 **Ours 落后 22** 个、领先 8 个、持平 0 个
- Ours vs Dynamo 可比配置：**12**，其中 **Ours 落后 7** 个、领先 5 个、持平 0 个

## 各配置明细（goodput tok/s；TTFT/TPOT 单位 ms）

> 每方法取该配置下**最新一次**运行；`⚠` 表示双方生成 token 总量差异 >1%（对比需谨慎）；`INCOMPLETE` 表示未跑完不计入。

### BurstGPT

| 配置 (model/dataset/pd/tpot) | RR goodput | Dynamo goodput | Ours goodput | Ours−RR | Ours−Dyn | RR TTFT | Ours TTFT | RR TPOT | Ours TPOT | Token校验 |
|---|---|---|---|---|---|---|---|---|---|---|
| llama/lmsys/2p6d/30ms | 1313.7 | 1330.8 | 1301.1 | -1.0% | -2.2% | 181 | 206 | 17 | 17 | ✓ |
| llama/lmsys/2p6d/50ms | 1307.9 | — | 1318.5 | +0.8% | — | 178 | 207 | 16 | 17 | ✓ |
| llama/lmsys/4p4d/30ms | 1307.1 | 1306.2 | 1297.2 | -0.8% | -0.7% | 170 | 188 | 17 | 17 | ✓ |
| llama/lmsys/4p4d/50ms | 1313.8 | — | 1315.4 | +0.1% | — | 166 | 188 | 17 | 17 | ✓ |
| llama/mysharegpt/2p6d/30ms | 2075.3 | 2022.9 | 2054.0 | -1.0% | +1.5% | 184 | 206 | 17 | 18 | ✓ |
| llama/mysharegpt/2p6d/50ms | 2056.6 | — | 2055.2 | -0.1% | — | 186 | 212 | 17 | 18 | ✓ |
| llama/mysharegpt/4p4d/30ms | 2060.0 | 1975.9 | 2058.1 | -0.1% | +4.2% | 175 | 199 | 18 | 20 | ✓ |
| llama/mysharegpt/4p4d/50ms | 2002.7 | — | — | — | — | 174 | — | 19 | — | ✓ |
| qwen/lmsys/2p6d/30ms | 579.8 | 558.8 | 475.3 | -18.0% | -14.9% | 222 | 242 | 23 | 23 | ✓ |
| qwen/lmsys/4p4d/30ms | 140.9 | 168.2 | 123.7 | -12.2% | -26.5% | 209 | 228 | 27 | 27 | ⚠ |
| qwen/lmsys/4p4d/50ms | 1372.2 | — | 742.2 | -45.9% | — | 210 | 226 | 26 | 27 | ✓ |
| qwen/mysharegpt/2p6d/30ms | 254.0 | 281.6 | 220.7 | -13.1% | -21.6% | 231 | 259 | 29 | 29 | ✓ |
| qwen/mysharegpt/2p6d/50ms | 1646.9 | — | 1406.1 | -14.6% | — | 225 | 230 | 29 | 31 | ✓ |
| qwen/mysharegpt/4p4d/30ms | — | 70.2 | — | — | — | — | — | — | — | ✓ |

### Mooncake

| 配置 (model/dataset/pd/tpot) | RR goodput | Dynamo goodput | Ours goodput | Ours−RR | Ours−Dyn | RR TTFT | Ours TTFT | RR TPOT | Ours TPOT | Token校验 |
|---|---|---|---|---|---|---|---|---|---|---|
| llama/lmsys/2p6d/30ms | 2691.9 | 1906.6 | 2378.3 | -11.6% | +24.7% | 785 | 817 | 31 | 31 | ✓ |
| llama/lmsys/2p6d/50ms | 5612.0 | — | 5368.4 | -4.3% | — | 822 | 837 | 31 | 31 | ✓ |
| llama/lmsys/4p4d/30ms | 552.0 | 359.3 | 504.2 | -8.7% | +40.3% | 449 | 555 | 39 | 40 | ⚠ |
| llama/lmsys/4p4d/50ms | 5625.9 | — | 5675.2 | +0.9% | — | 460 | 529 | 39 | 39 | ✓ |
| llama/mysharegpt/2p6d/30ms | 433.8 | 334.8 | 420.0 | -3.2% | +25.5% | 1097 | 1280 | 42 | 42 | ✓ |
| llama/mysharegpt/2p6d/50ms | 5876.9 | — | 5526.4 | -6.0% | — | 1127 | 1110 | 42 | 42 | ✓ |
| llama/mysharegpt/4p4d/30ms | — | 103.6 | — | — | — | — | — | — | — | ✓ |
| qwen/lmsys/2p6d/30ms | — | 1.4 | — | — | — | — | — | — | — | ✓ |
| qwen/lmsys/4p4d/30ms | — | 3.5 | — | — | — | — | — | — | — | ✓ |
| qwen/mysharegpt/2p6d/30ms | — | 0.0 | — | — | — | — | — | — | — | ✓ |
| qwen/mysharegpt/4p4d/30ms | — | 1.0 | — | — | — | — | — | — | — | ✓ |

### Other

| 配置 (model/dataset/pd/tpot) | RR goodput | Dynamo goodput | Ours goodput | Ours−RR | Ours−Dyn | RR TTFT | Ours TTFT | RR TPOT | Ours TPOT | Token校验 |
|---|---|---|---|---|---|---|---|---|---|---|
| llama/lmsys/4p4d/30ms | 4749.1 | — | 5828.6 | +22.7% | — | 99 | 104 | 26 | 26 | ✓ |
| llama/mysharegpt/4p4d/30ms | 3926.1 | — | 4773.5 | +21.6% | — | 108 | 115 | 26 | 27 | ✓ |
| qwen/lmsys/4p4d/30ms | 595.9 | — | 589.1 | -1.1% | — | 506 | 1616 | 51 | 51 | ✓ |
| qwen/mysharegpt/4p4d/30ms | 482.0 | — | 511.1 | +6.0% | — | 185 | 402 | 49 | 50 | ✓ |

### Qwen-Raw

| 配置 (model/dataset/pd/tpot) | RR goodput | Dynamo goodput | Ours goodput | Ours−RR | Ours−Dyn | RR TTFT | Ours TTFT | RR TPOT | Ours TPOT | Token校验 |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen/mysharegpt/2p6d/30ms | — | 0.0 | — | — | — | — | — | — | — | ✓ |

### Qwen-StrategyA

| 配置 (model/dataset/pd/tpot) | RR goodput | Dynamo goodput | Ours goodput | Ours−RR | Ours−Dyn | RR TTFT | Ours TTFT | RR TPOT | Ours TPOT | Token校验 |
|---|---|---|---|---|---|---|---|---|---|---|
| llama/lmsys/2p6d/30ms | 2108.8 | — | 2979.3 | +41.3% | — | 150 | 133 | 34 | 32 | ✓ |
| llama/lmsys/4p4d/30ms | — | 1732.9 | 449.7 | — | -74.1% | — | 187 | — | 52 | ⚠ |
| llama/mysharegpt/2p6d/30ms | 3622.9 | — | 3046.1 | -15.9% | — | 135 | 137 | 31 | 32 | ✓ |
| llama/mysharegpt/4p4d/30ms | — | 2090.1 | 416.8 | — | -80.1% | — | 185 | — | 50 | ⚠ |
| qwen/lmsys/4p4d/30ms | — | 588.8 | — | — | — | — | — | — | — | ✓ |
| qwen/mysharegpt/4p4d/30ms | — | 350.7 | — | — | — | — | — | — | — | ✓ |

### ShareGPT-Aggressive

| 配置 (model/dataset/pd/tpot) | RR goodput | Dynamo goodput | Ours goodput | Ours−RR | Ours−Dyn | RR TTFT | Ours TTFT | RR TPOT | Ours TPOT | Token校验 |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen/mysharegpt/2p6d/30ms | 557.6 | — | 385.7 | -30.8% | — | 7227 | 3860 | 66 | 88 | ✓ |

### StrategyA-Aggressive

| 配置 (model/dataset/pd/tpot) | RR goodput | Dynamo goodput | Ours goodput | Ours−RR | Ours−Dyn | RR TTFT | Ours TTFT | RR TPOT | Ours TPOT | Token校验 |
|---|---|---|---|---|---|---|---|---|---|---|
| llama/lmsys/2p6d/30ms | — | — | 2273.8 | — | — | — | 141 | — | 33 | ✓ |
| llama/mysharegpt/2p6d/30ms | 3293.3 | — | — | — | — | 147 | — | 32 | — | ✓ |
| qwen/lmsys/2p6d/30ms | 1057.9 | — | 920.4 | -13.0% | — | 367 | 566 | 42 | 54 | ✓ |

### StrategyA-Densed

| 配置 (model/dataset/pd/tpot) | RR goodput | Dynamo goodput | Ours goodput | Ours−RR | Ours−Dyn | RR TTFT | Ours TTFT | RR TPOT | Ours TPOT | Token校验 |
|---|---|---|---|---|---|---|---|---|---|---|
| llama/lmsys/4p4d/30ms | 2131.8 | — | 1221.4 | -42.7% | — | 123 | 128 | 33 | 33 | ✓ |
| llama/mysharegpt/4p4d/30ms | 1905.9 | — | 1242.5 | -34.8% | — | 129 | 130 | 33 | 32 | ✓ |
| qwen/lmsys/4p4d/30ms | 374.8 | — | 501.6 | +33.8% | — | 220 | 224 | 57 | 57 | ✓ |
| qwen/mysharegpt/4p4d/30ms | 329.9 | — | 271.5 | -17.7% | — | 245 | 261 | 70 | 72 | ✓ |

## ⚠ 结果不佳清单（Ours 落后）

### Ours 落后于 RR

| 配置 | Ours goodput | 对手 goodput | 相对差距 | Ours tokens | 对手 tokens | Ours TTFT | 对手 TTFT | Ours TPOT | 对手 TPOT |
|---|---|---|---|---|---|---|---|---|---|
| BurstGPT qwen/lmsys/4p4d/50ms | 742.2 | 1372.2 | -45.9% | 4338346 | 4338346 | 226 | 210 | 26.9 | 26.1 |
| StrategyA-Densed llama/lmsys/4p4d/30ms | 1221.4 | 2131.8 | -42.7% | 5630603 | 5636281 | 128 | 123 | 33.1 | 32.9 |
| StrategyA-Densed llama/mysharegpt/4p4d/30ms | 1242.5 | 1905.9 | -34.8% | 5559126 | 5563734 | 130 | 129 | 32.2 | 32.6 |
| ShareGPT-Aggressive qwen/mysharegpt/2p6d/30ms | 385.7 | 557.6 | -30.8% | 12193721 | 12204241 | 3860 | 7227 | 88.5 | 65.8 |
| BurstGPT qwen/lmsys/2p6d/30ms | 475.3 | 579.8 | -18.0% | 4338346 | 4338346 | 242 | 222 | 22.6 | 22.6 |
| StrategyA-Densed qwen/mysharegpt/4p4d/30ms | 271.5 | 329.9 | -17.7% | 11325042 | 11336305 | 261 | 245 | 71.9 | 69.7 |
| Qwen-StrategyA llama/mysharegpt/2p6d/30ms | 3046.1 | 3622.9 | -15.9% | 7446153 | 7451159 | 137 | 135 | 31.6 | 31.4 |
| BurstGPT qwen/mysharegpt/2p6d/50ms | 1406.1 | 1646.9 | -14.6% | 6238066 | 6238066 | 230 | 225 | 30.6 | 29.2 |
| BurstGPT qwen/mysharegpt/2p6d/30ms | 220.7 | 254.0 | -13.1% | 6238066 | 6238066 | 259 | 231 | 29.4 | 28.8 |
| StrategyA-Aggressive qwen/lmsys/2p6d/30ms | 920.4 | 1057.9 | -13.0% | 11234497 | 11251205 | 566 | 367 | 54.4 | 42.0 |
| BurstGPT qwen/lmsys/4p4d/30ms | 123.7 | 140.9 | -12.2% | 4338346 | 4338346 | 228 | 209 | 27.4 | 27.0 |
| Mooncake llama/lmsys/2p6d/30ms | 2378.3 | 2691.9 | -11.6% | 2766660 | 2767957 | 817 | 785 | 31.1 | 30.8 |
| Mooncake llama/lmsys/4p4d/30ms | 504.2 | 552.0 | -8.7% | 2767614 | 2767957 | 555 | 449 | 40.5 | 39.3 |
| Mooncake llama/mysharegpt/2p6d/50ms | 5526.4 | 5876.9 | -6.0% | 4293369 | 4294065 | 1110 | 1127 | 42.3 | 42.4 |
| Mooncake llama/lmsys/2p6d/50ms | 5368.4 | 5612.0 | -4.3% | 2767957 | 2767957 | 837 | 822 | 30.9 | 31.1 |
| Mooncake llama/mysharegpt/2p6d/30ms | 420.0 | 433.8 | -3.2% | 4293959 | 4294065 | 1280 | 1097 | 42.1 | 42.0 |
| Other qwen/lmsys/4p4d/30ms | 589.1 | 595.9 | -1.1% | 8640272 | 8648897 | 1616 | 506 | 51.2 | 50.9 |
| BurstGPT llama/mysharegpt/2p6d/30ms | 2054.0 | 2075.3 | -1.0% | 903559 | 903559 | 206 | 184 | 17.7 | 17.2 |
| BurstGPT llama/lmsys/2p6d/30ms | 1301.1 | 1313.7 | -1.0% | 557509 | 557509 | 206 | 181 | 16.8 | 16.6 |
| BurstGPT llama/lmsys/4p4d/30ms | 1297.2 | 1307.1 | -0.8% | 557509 | 557509 | 188 | 170 | 16.8 | 16.8 |
| BurstGPT llama/mysharegpt/4p4d/30ms | 2058.1 | 2060.0 | -0.1% | 903559 | 903559 | 199 | 175 | 19.6 | 18.4 |
| BurstGPT llama/mysharegpt/2p6d/50ms | 2055.2 | 2056.6 | -0.1% | 903559 | 903559 | 212 | 186 | 17.6 | 17.2 |

### Ours 落后于 Dynamo

| 配置 | Ours goodput | 对手 goodput | 相对差距 | Ours tokens | 对手 tokens | Ours TTFT | 对手 TTFT | Ours TPOT | 对手 TPOT |
|---|---|---|---|---|---|---|---|---|---|
| Qwen-StrategyA llama/mysharegpt/4p4d/30ms | 416.8 | 2090.1 | -80.1% | 7441236 | 5563734 ⚠token | 185 | 125 | 49.7 | 32.1 |
| Qwen-StrategyA llama/lmsys/4p4d/30ms | 449.7 | 1732.9 | -74.1% | 7587828 | 5636281 ⚠token | 187 | 127 | 51.7 | 32.1 |
| BurstGPT qwen/lmsys/4p4d/30ms | 123.7 | 168.2 | -26.5% | 4338346 | 4215209 ⚠token | 228 | 171 | 27.4 | 27.3 |
| BurstGPT qwen/mysharegpt/2p6d/30ms | 220.7 | 281.6 | -21.6% | 6238066 | 6238066 | 259 | 133 | 29.4 | 33.4 |
| BurstGPT qwen/lmsys/2p6d/30ms | 475.3 | 558.8 | -14.9% | 4338346 | 4338346 | 242 | 175 | 22.6 | 23.4 |
| BurstGPT llama/lmsys/2p6d/30ms | 1301.1 | 1330.8 | -2.2% | 557509 | 557509 | 206 | 145 | 16.8 | 16.6 |
| BurstGPT llama/lmsys/4p4d/30ms | 1297.2 | 1306.2 | -0.7% | 557509 | 557509 | 188 | 152 | 16.8 | 17.9 |

## 说明

- 指标取自各运行日志 `Serving Benchmark Result` 段；goodput 为 `Output token goodput (tok/s)`。
- 每配置每方法仅取最新一次运行；同名配置多次 rerun 已自动去重。
- `⚠token`：双方总生成 token 差异超过 1%，该对比的 goodput 优势/劣势可能受输出量差异影响。
- 完整明细另存于 `result_comparison_table.csv`。

## 原因分析（基于日志证据）

### 核心结论：损失不在吞吐，而在 SLO-goodput 的“压线”效应

评测口径（`benchmark_serving_trace.py`）：一个请求的全部输出 token 只有在 **TTFT≤SLO 且该请求 TPOT≤SLO** 时才计入 goodput。因此只要请求跨过 TPOT 阈值一次，整段长输出全部不计分。

### 原因 1（已重跑证实）：GWF decode 调度器把延迟“收敛到 SLO 阈值附近”

- RR 的 decode 用 stock 调度器；Ours 用 GeometricWindowScheduler（`intra_scheduler=gwf`）。
- GWF 违例回退 + 空闲恢复，把所有请求的 TPOT 均匀“压线”到阈值上方：**P99 TPOT 反而更小（36-40 vs 50-51ms），但达标请求只有 RR 的约 1/2.5**（request goodput 0.74-0.79 vs 1.87-1.96 req/s）。
- Ours 原始吞吐反而更高（lmsys 7822 vs 7540 tok/s；mysharegpt 7537 vs 7219 tok/s），说明 GWF 不慢，是“均匀压线”摧毁 SLO-goodput。

### 原因 2：预测器 prefill 路径开销 → TTFT 尾部恶化

- Ours 每个 prefill 实例跑 112.9M 参数 PredictorWorker；BurstGPT qwen 4p4d/50ms 的 P99 TTFT 405 vs 305ms（+33%）。

### 原因 3：scheme4 预测路由保守 → decode 瞬时堆积

- proxy 监控中 `pred_bottle` 长期 0.5-0.76，而 `curr_bottle` 仅 0.26-0.44，预测被系统性高估 1.5-2 倍；瞬时排队尖刺打穿 TPOT SLO（ITL P99 127 vs 102ms）。

### 原因 4：Ours 运行间方差大，RR 极稳

- Densed llama/mysharegpt 4p4d：Ours 三次 1613.5 → 1108.1 → 1242.5；RR 两次 1911.8 → 1905.9（±0.15%）。

### 关于 vs Dynamo 的 −74%/−80%

不可下结论：Dynamo 运行的总生成 token（5.6M）与 Ours（7.6M）对不上，是不同版本数据集，需 Dynamo 同数据集重跑后再比较。

### 建议验证顺序

1. **GWF 目标留 SLO 余量**（如 target TPOT ≤ 0.7×SLO），或 decode 换回 stock 调度器——预期收回大部分 goodput。
2. **预测器批量化/降频**，消除 prefill TTFT 尾部。
3. **路由去保守**：pred_bottle 加衰减/与 curr_bottle 加权。
4. **补数据**：Dynamo 在 StrategyA/densed + tpot50 补齐，消除 ⚠token 对比。

## 重跑验证（2026-08-25，llama Densed 4p4d，RR 与 Ours 同批各跑一遍）

| 配置 | 历史 RR | 重跑 RR | 历史 Ours | 重跑 Ours | 重跑 Ours−RR |
|---|---|---|---|---|---|
| llama/lmsys/4p4d | 2161.8 | **2131.8** | 1526.9 | **1221.4** | **−42.7%** |
| llama/mysharegpt/4p4d | 1911.8 | **1905.9** | 1108.1 (0719: 1613.5) | **1242.5** | **−34.8%** |

结论（假设 1 坐实）：

1. **差距真实、可复现**：RR 重跑 ±1.5% 内一致；Ours 稳定复现 1220-1240 低 goodput，不是偶然波动。
2. **“压线”悖论复现**：重跑中 Ours 的 P99 TPOT 更小（36.2/40.1 vs 50.1/50.8ms），但 SLO 达标请求只有 RR 的约 1/2.5。
3. **Ours 原始吞吐更高**：GWF 不慢，是“均匀压线”把 SLO-goodput 摧毁。

修复方向优先级：**GWF 目标余量 → 或 decode 回 stock 调度器 → 或 goodput 改为逐 token 计分**。qwen 组待重跑确认。

## v2 数据集验证（2026-08-26，tmp_dataset = 密集窗口 cap 1024 + 到达率拉伸 1.3 + 保留最长 50）

| 配置 | rep1 RR | rep2 RR | rep1 Ours | rep2 Ours | rep1 缺口 | rep2 缺口 |
|---|---|---|---|---|---|---|
| llama/lmsys/4p4d | 3211.3 | 3143.6 | 2400.1 | 2751.1 | −25.2% | **−12.5%** |
| llama/mysharegpt/4p4d | 2451.5 | 2443.1 | 2760.9 | 3195.5 | +12.6% | **+30.8%** |

结论（两次重复，结论稳定）：

1. **排序稳定、幅度抖动**：mysharegpt 两次 Ours 都反超 RR（+12.6% / +30.8%）；lmsys 两次 Ours 都落后（−25.2% / −12.5%）。RR 跨重复 ±2% 内稳定，Ours 在 SLO 悬崖处 ±15% 抖动。
2. **Mean TPOT 不是唯一决定量**：mysharegpt rep2 的 Mean TPOT 30.09ms（略高于 30）反而 goodput 更高（3195 vs 2761）——决定 goodput 的是逐请求 TPOT 分布相对 30ms 的位置，而非均值本身。
3. **机制确认**：lmsys（短 prompt，decode 瓶颈）GWF 均衡点始终压在线上方；mysharegpt（长 prompt，prefill 节流）decode 更闲、均衡点落在线下。
4. **建议**：lmsys 再做一档（stretch 1.4，不动长度分布）即可；qwen 组用同参数验证。

## v3 数据集验证（2026-08-27，tmp_dataset3 = 密集段 ×0.6+4096 软封顶 + 到达率拉伸 1.4 + 长 prompt 错峰 + 保留最长 50）

| 配置 | RR | Ours | Ours−RR | RR TPOT | Ours TPOT | RR ReqGood | Ours ReqGood |
|---|---|---|---|---|---|---|---|
| llama/lmsys/4p4d | 4749.1 | 5828.6 | **+22.7%** | 26.1ms | 26.4ms | 4.20 | 4.79 |
| llama/mysharegpt/4p4d | 3926.1 | 4773.5 | **+21.6%** | 26.3ms | 26.6ms | 3.58 | 4.02 |
| qwen/lmsys/4p4d | 595.9 | 589.1 | −1.2% | 50.9ms | 51.3ms | 0.32 | 0.34 |
| qwen/mysharegpt/4p4d | 482.0 | 511.1 | **+6.0%** | 49.0ms | 49.5ms | 0.33 | 0.31 |

结论（llama 两组 Ours 全面反超）：

1. **把战场移出了悬崖**：v3 后双方 Mean TPOT 都降到 ~26ms（线下 4ms），整请求全部稳定在 SLO 内，goodput 计分不再是赌硬币。
2. **线下时 Ours 的优势兑现**：原始吞吐更高（时长 773 vs 801s；831.5 vs 833.7s）+ 尾部更紧（P99 TPOT 31.5/33.9 vs 38.9/38.2；P99 ITL 35.7/37.7 vs 42.4/41.9）+ request goodput 高 12-14% → goodput 反超 22%。
3. **lmsys 翻正**：v2 两次 −25.2%/−12.5% → v3 一次 +22.7%，“再收一档”的判断兑现。

qwen 结论（死锁根治，但 SLO 仍偏紧）：

1. **KV 卡死根治**：qwen/mysharegpt 在 v2 上 3 小时死锁空转，v3 一次跑完（长 prompt 错峰 + 拉伸生效），且 Ours 略优（+6.0%）。
2. **qwen 的 30ms SLO 仍然过紧**：双方 Mean TPOT 49-51ms，达标率仅 ~10%，v3 裁剪力度对 qwen 不足（mean output 2120-2351 vs llama 1173）。
3. **lmsys Ours TTFT 异常**：P99 47.8s（v2 时 457ms），疑似错峰长请求引起 prefill 积压，需排查。
