# 现有实验覆盖与结果总结

## 范围与判据

- 数据集类别统一为 `burstgpt`、`mooncake`、`qwen`；其中 `qwen` 类使用 `qwentrace`，所有请求数为 4080 或名称含 `aggressive` 的结果均归入该类。
- workload 统一为 `lmsys`、`sharegpt`（原目录名 `mysharegpt` 归一为 `sharegpt`）。
- 主矩阵固定为 TPOT=30 ms：3 类数据集 x 2 workload x 2 模型 x 2 PD x 3 方法，共 72 个方法槽位。
- 其他 TPOT SLO（50 ms）作为额外实验单独统计，不计入 72 个主矩阵槽位。
- 核心比较指标为满足 TTFT <= 1000 ms 和指定 TPOT SLO 的 output-token goodput。
- 仅当 RR、Ours、Dynamo 均成功，生成 token 与 RR 相差不超过 1%，且 Ours 同时超过两个基线时，判定为 `achieved`。
- 重复实验在比较矩阵中选最新成功结果；所有 run 均保留在 CSV 明细中。

## 数据盘点

- 共解析 108 次实验；成功 106 次；失败或不完整 2 次。
- TPOT=30 ms 主矩阵已覆盖 57/72 个方法槽位，尚缺 15 个。
- `result_0630`：成功 65 次，失败或不完整 1 次。
- `dynamo_0630`：成功 26 次，失败或不完整 0 次。
- `dynamo_retry3`：成功 3 次，失败或不完整 1 次。
- `rr_ours_retry3`：成功 4 次，失败或不完整 0 次。
- `organized_llama_rr`：成功 2 次，失败或不完整 0 次。
- `organized_llama_ours`：成功 2 次，失败或不完整 0 次。
- `organized_llama_dynamo`：成功 4 次，失败或不完整 0 次。
- 另发现 2026-07-27 至 2026-07-29 的 25 次未纳入尝试：start_failed 1 次，benchmark_failed 24 次。
- 这些尝试没有完整 benchmark 指标，不计入 57/72 的有效方法覆盖；详细记录单列在失败尝试 CSV。
- 另有 2026-08-25 的 12 次失败重跑（`retry_fail`），同样不产生可比较指标。
- 2026-07-29 与 2026-07-05 的 12 组成功补充实验不属于 72 槽位主矩阵，单列在下方补充实验章节。

## 对比矩阵

| 数据集类别 | 模型 | Workload | 请求数 | PD | TPOT ms | 已有方法 | RR goodput | Ours goodput | Dynamo goodput | Ours vs RR | Ours vs Dynamo | 结论 |
|---|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| burstgpt | llama | lmsys | 673 | 2p6d | 30 | RR,Ours,Dynamo | 1313.7 | 1301.1 | 1330.8 | -1.0% | -2.2% | not-achieved |
| burstgpt | llama | lmsys | 673 | 4p4d | 30 | RR,Ours,Dynamo | 1307.1 | 1297.2 | 1302.1 | -0.8% | -0.4% | not-achieved |
| burstgpt | qwen | lmsys | 673 | 2p6d | 30 | Ours,RR,Dynamo | 579.8 | 475.3 | 558.8 | -18.0% | -14.9% | not-achieved |
| burstgpt | qwen | lmsys | 673 | 4p4d | 30 | Ours,RR,Dynamo | 140.9 | 123.7 | 168.2 | -12.2% | -26.5% | incomparable-token-mismatch |
| burstgpt | llama | sharegpt | 673 | 2p6d | 30 | RR,Ours,Dynamo | 2075.3 | 2054.0 | 2022.9 | -1.0% | 1.5% | not-achieved |
| burstgpt | llama | sharegpt | 673 | 4p4d | 30 | RR,Ours,Dynamo | 2060.0 | 2058.1 | 1975.9 | -0.1% | 4.2% | not-achieved |
| burstgpt | qwen | sharegpt | 673 | 2p6d | 30 | RR,Ours,Dynamo | 254.0 | 220.7 | 281.6 | -13.1% | -21.6% | not-achieved |
| burstgpt | qwen | sharegpt | 673 | 4p4d | 30 | Dynamo | - | - | 70.2 | -% | -% | incomplete |
| mooncake | llama | lmsys | 1790 | 2p6d | 30 | Ours,RR,Dynamo | 2691.9 | 2378.3 | 1906.6 | -11.6% | 24.7% | not-achieved |
| mooncake | llama | lmsys | 1790 | 4p4d | 30 | Ours,RR,Dynamo | 552.0 | 504.2 | 359.3 | -8.7% | 40.3% | incomparable-token-mismatch |
| mooncake | qwen | lmsys | 1790 | 2p6d | 30 | Dynamo | - | - | 1.4 | -% | -% | incomplete |
| mooncake | qwen | lmsys | 1790 | 4p4d | 30 | Dynamo | - | - | 0.0 | -% | -% | incomplete |
| mooncake | llama | sharegpt | 1790 | 2p6d | 30 | Ours,RR,Dynamo | 433.8 | 420.0 | 334.8 | -3.2% | 25.5% | not-achieved |
| mooncake | llama | sharegpt | 1790 | 4p4d | 30 | Dynamo | - | - | 103.6 | -% | -% | incomplete |
| mooncake | qwen | sharegpt | 1790 | 2p6d | 30 | Dynamo | - | - | 0.0 | -% | -% | incomplete |
| mooncake | qwen | sharegpt | 1790 | 4p4d | 30 | Dynamo | - | - | 1.0 | -% | -% | incomplete |
| qwen | llama | lmsys | 4080 | 2p6d | 30 | RR,Ours | 2108.8 | 2273.8 | - | 7.8% | -% | partial-ours>rr |
| qwen | llama | lmsys | 4080 | 4p4d | 30 | RR,Ours,Dynamo | 4749.1 | 5828.6 | 5735.7 | 22.7% | 1.6% | achieved |
| qwen | qwen | lmsys | 4080 | 2p6d | 30 | Ours,RR | 1057.9 | 920.4 | - | -13.0% | -% | incomplete |
| qwen | qwen | lmsys | 4080 | 4p4d | 30 | Ours,RR,Dynamo | 595.9 | 589.1 | 833.5 | -1.1% | -29.3% | not-achieved |
| qwen | llama | sharegpt | 4080 | 2p6d | 30 | RR,Ours | 3293.3 | 3046.1 | - | -7.5% | -% | incomplete |
| qwen | llama | sharegpt | 4080 | 4p4d | 30 | Ours,RR,Dynamo | 3926.1 | 4773.5 | 4859.4 | 21.6% | -1.8% | not-achieved |
| qwen | qwen | sharegpt | 4080 | 2p6d | 30 | Ours,RR,Dynamo | 557.6 | 385.7 | 0.0 | -30.8% | -% | incomparable-token-mismatch |
| qwen | qwen | sharegpt | 4080 | 4p4d | 30 | RR,Ours,Dynamo | 482.0 | 511.1 | 654.1 | 6.0% | -21.9% | not-achieved |

## 补充实验结果

以下实验在 2026-07-29 与 2026-07-05 产生，不属于 TPOT=30 ms 的 72 槽位主矩阵，单独列出以免与主矩阵混淆。goodput 口径为主矩阵一致：TTFT <= 1000 ms 且 TPOT 30 ms。

### 1p1d 路由策略消融（2026-07-29，成功 8 组）

| 策略 | Workload | PD | TPOT ms | 请求数 | 生成 token | goodput | throughput | Mean TTFT | Mean TPOT | 来源 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| aggressive | 1k | 1p1d | 30 | 562 | 392301 | 0.0 | 424.7 | 300077.6 | 16.9 | 1p1d_routing |
| conservative | 1k | 1p1d | 30 | 562 | 392301 | 627.1 | 628.7 | 83.8 | 17.0 | 1p1d_routing |
| predict_gwf | 1k | 1p1d | 30 | 562 | 392301 | 627.3 | 629.0 | 82.4 | 16.8 | 1p1d_routing |
| gwf | 400 | 1p1d | 30 | 400 | 222708 | 368.8 | 370.5 | 82.4 | 16.1 | 1p1d_routing |
| conservative | 400 | 1p1d | 30 | 400 | 222708 | 368.8 | 370.5 | 80.4 | 16.1 | 1p1d_routing |
| default | 400 | 1p1d | 30 | 400 | 222708 | 0.0 | 247.1 | 300077.2 | 16.1 | 1p1d_routing |
| aggressive | best500_predict | 1p1d | 30 | 500 | 1009331 | 0.0 | 1062.7 | 300097.3 | 28.1 | 1p1d_routing |
| predict_gwf | best500_predict | 1p1d | 30 | 500 | 1009331 | 763.8 | 1554.5 | 102.6 | 28.3 | 1p1d_routing |

- 该批次为 1p1d 部署下的路由策略对比，与 2p6d/4p4d 主矩阵不是同一部署形态，因此不能直接填入 72 槽位。
- `aggressive` 三个档位与 `default` 400 档位因 TTFT 超过 1000 ms，goodput 记为 0，仅保留 throughput 作为参考。
- `predict_gwf` 在三个档位上均未被 TTFT 约束归零，是这批 1p1d 实验中最稳的策略。

### 调度策略消融（2026-07-05，mooncake-lmsys n1790 / 2p6d / 30 ms，成功 4 组）

| 策略 | Workload | PD | TPOT ms | 请求数 | 生成 token | goodput | throughput | Mean TTFT | Mean TPOT | 来源 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| decode_predict_3Dload | lmsys-n1790 | 2p6d | 30 | 1790 | 2767957 | 2560.5 | 6383.9 | 774.4 | 31.1 | scheduling_ablation |
| p2d_current_3Dload | lmsys-n1790 | 2p6d | 30 | 1790 | 2767957 | 2154.8 | 6490.8 | 760.5 | 31.1 | scheduling_ablation |
| p2d_current_capacity | lmsys-n1790 | 2p6d | 30 | 1790 | 2767957 | 2146.9 | 6564.2 | 753.0 | 31.2 | scheduling_ablation |
| roundrobin | lmsys-n1790 | 2p6d | 30 | 1790 | 2767957 | 1861.8 | 6304.3 | 792.6 | 31.4 | scheduling_ablation |

- 这 4 组生成 token 均为 2767957，与主矩阵 `mooncake / llama / lmsys / 1790 / 2p6d / 30` 同源，可作为该配置下更早策略版本的对照。
- 其中 `decode_predict_3Dload` 表现最好，但仍低于主矩阵中较新的 Ours 与 RR 结果。

## 目标达成情况

- 完全达成：1 组。
- 阶段性达成（Ours > RR，但缺 Dynamo）：1 组。
- token 不匹配、不能直接比较：3 组。

### 完全达成

- `qwen / llama / lmsys / 4080 / 4p4d / 30`

### 阶段性达成

- `qwen / llama / lmsys / 4080 / 2p6d / 30`：补跑 Dynamo 后才能形成完整结论。

## 仍需补充的实验

- `burstgpt / qwen / sharegpt / 673 / 4p4d / 30`：补充 RR, Ours。
- `mooncake / qwen / lmsys / 1790 / 2p6d / 30`：补充 RR, Ours。
- `mooncake / qwen / lmsys / 1790 / 4p4d / 30`：补充 RR, Ours。
- `mooncake / llama / sharegpt / 1790 / 4p4d / 30`：补充 RR, Ours。
- `mooncake / qwen / sharegpt / 1790 / 2p6d / 30`：补充 RR, Ours。
- `mooncake / qwen / sharegpt / 1790 / 4p4d / 30`：补充 RR, Ours。
- `qwen / llama / lmsys / 4080 / 2p6d / 30`：补充 Dynamo。
- `qwen / qwen / lmsys / 4080 / 2p6d / 30`：补充 Dynamo。
- `qwen / llama / sharegpt / 4080 / 2p6d / 30`：补充 Dynamo。

## 补实验优先级

1. 优先补齐上方 72 槽位主矩阵中缺失的方法。
2. TPOT=50 ms 属于额外矩阵，待 30 ms 主矩阵完整后再统一补齐。
3. 对 token 不匹配或失败的配置重跑，完成前不做性能结论。
4. 最终论文使用的配置至少补到两次干净重复实验。

## 产物

- 完整逐 run 表格：`/root/predict-schedule/experiment/zrecent_use/0630/existing_experiment_matrix.csv`
- 七月底失败尝试表格：`/root/predict-schedule/experiment/zrecent_use/0630/failed_experiment_attempts.csv`
- 补充实验表格：`/root/predict-schedule/experiment/zrecent_use/0630/supplementary_experiment_matrix.csv`
- 本报告：`/root/predict-schedule/experiment/zrecent_use/0630/existing_experiment_summary.md`
