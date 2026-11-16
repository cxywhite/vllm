# 预测器消融完整结果（β=1.0，4p4d，scheme4+GWF，无 UniBoost）

统一配置：Qwen-2.5-7b · β=1.0（无缩放）· 4p4d · scheme4 路由 + GWF(θ=120ms/margin=250ms) · gmu 0.8 · SLO TTFT≤1000ms ∧ TPOT≤30ms · 强制生成(ignore-eos) · predacc_lmsys_o1600/o2000（各 4080 条真实 lmsys prompt）。

## o1600

| 指标 | ours | µ-Serve | TRAIL++ |
|---|---|---|---|
| Successful requests | 4080 | 4080 | 4080 |
| Benchmark duration (s) | 891.69 | 857.43 | 927.83 |
| Total input tokens | 424244 | 424244 | 424244 |
| Total generated tokens | 6516479 | 6490558 | 6519548 |
| Request throughput (req/s) | 4.58 | 4.76 | 4.40 |
| Request goodput (req/s) | 0.69 | 0.71 | 0.64 |
| **Output token goodput (tok/s)** | 520.34 | **533.43** | 458.52 |
| Total token goodput (tok/s) | 599.19 | 616.83 | 532.25 |
| Output token throughput (tok/s) | 7307.98 | 7569.74 | 7026.64 |
| Total token throughput (tok/s) | 7783.76 | 8064.53 | 7483.89 |
| Mean / Median / P99 TTFT (ms) | 550.36 / 422.59 / 1344.90 | 478.48 / 332.70 / 1561.26 | 431.30 / 338.85 / 1271.86 |
| Mean / Median / P99 TPOT (ms) | 53.57 / 54.69 / 84.14 | 53.60 / 56.32 / 79.62 | 56.20 / 58.16 / 89.79 |
| Mean / Median / P99 ITL (ms) | 54.84 / 54.35 / 116.16 | 54.78 / 54.35 / 124.01 | 58.37 / 58.19 / 123.92 |
| Mean / Median / P99 E2EL (ms) | 88091.72 / 27397.97 / 389981.26 | 87577.20 / 26706.80 / 390953.23 | 93646.39 / 28574.67 / 424655.67 |

## o2000

| 指标 | ours | µ-Serve | TRAIL++ |
|---|---|---|---|
| Successful requests | 4080 | 4080 | 4080 |
| Benchmark duration (s) | 1153.65 | 1239.63 | 1202.93 |
| Total input tokens | 418865 | 418865 | 418865 |
| Total generated tokens | 8140536 | 8144725 | 8158788 |
| Request throughput (req/s) | 3.54 | 3.29 | 3.39 |
| Request goodput (req/s) | 0.49 | 0.42 | 0.42 |
| **Output token goodput (tok/s)** | **322.87** | 250.43 | 252.41 |
| Total token goodput (tok/s) | 380.12 | 299.09 | 301.30 |
| Output token throughput (tok/s) | 7056.36 | 6570.28 | 6782.43 |
| Total token throughput (tok/s) | 7419.44 | 6908.17 | 7130.63 |
| Mean / Median / P99 TTFT (ms) | 686.96 / 639.50 / 2121.75 | 777.61 / 936.79 / 2009.28 | 745.33 / 934.93 / 1902.42 |
| Mean / Median / P99 TPOT (ms) | 76.01 / 84.63 / 126.63 | 84.26 / 94.41 / 134.17 | 79.92 / 89.94 / 114.79 |
| Mean / Median / P99 ITL (ms) | 76.71 / 79.14 / 149.96 | 85.03 / 88.72 / 181.66 | 81.75 / 84.82 / 158.71 |
| Mean / Median / P99 E2EL (ms) | 153685.98 / 44659.48 / 751478.70 | 170444.26 / 53660.63 / 808707.48 | 164162.48 / 50463.09 / 782419.91 |

## 结论（一句话）
β=1.0 下换预测器无一致赢家、量级小（o1600 极差 16%、o2000 极差 29%）；P99 TTFT/TPOT 全线超标（TTFT 1272–2121ms、TPOT 79–134ms，SLO 要 ≤1000/≤30），所以 goodput 只占吞吐 4–7%。真正主导 goodput 的是 β 校准而非预测器选择。
