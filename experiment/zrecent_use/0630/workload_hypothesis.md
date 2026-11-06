# Workload 画像与迭代测试协议（GWF 准入 vs vLLM 贪心）

> 目标：找到一类**有特点的负载**，使论文的实例内调度（GWF 准入：θ=120ms 窗口 + 30ms hopeless 驱逐 + eV 抢占）
> 在真实硬件上稳定优于 vLLM 贪心准入（`scheme4`）。本文件是假设、方法、判据与迭代循环的单一来源。

## 1. 假设（当前 workload 画像 v1）

GWF 的三个部件各对应一个负载特征，凑齐即赢：

| 维度 | 画像 | 对应 GWF 部件 | 机制 |
|---|---|---|---|
| **输入长度** | 短–中等（30–2000 token） | 间接（次要） | 让 KV/计算压力来自*输出*而非输入，使准入决策聚焦"未来还要生多少" |
| **实际输出长度** | **双峰**：短(≈200，能达标) + 长(≈8000+，必超 SLO) | 30ms hopeless 驱逐 | 长请求驻留→TPOT 超 30ms→变 hopeless；贪心 FIFO 让长请求堵死短请求，GWF 甩队尾保短请求 |
| **预测输出长度** | **校准**（predict≈actual） | eV 抢占受害者 | 正确挑"谁付重算代价"；当前实测 8–13× 过预测使 `gen_ratio→0`、`rem_ratio→1`，eV 失效 |
| **到达模式**（第 4 维，关键） | **突发**（周期性高 QPS） | θ=120ms 窗口 | 突发把并发推到 cliff（≈257）之上；贪心全收→TPOT 98ms 跳 5264ms→拖垮短请求；GWF 封顶 A=256 停在平坦区 |

**核心一句话**：GWF 的价值是"让短请求活下来"，贪心的失败是"让长请求把短请求一起拖死"。
前两维（突发→θ 窗口、长尾→hopeless）**完全靠实测 TPOT，不依赖预测器**；预测校准只是第三维（eV）的加分项。

### 已证事实（支撑画像）

- 预测器系统性过预测：pred/actual 中位数 10.96（离线）/ 8.23（live qwen）/ 12.87（live burstgpt），93–100% 过预测 >2×。
- decode 会真过载：burstgpt 上 `curr_bottle=1.09–1.13 > 1.0`。
- 贪心失控：qwen 稠密 trace P99 TPOT=156ms（5× SLO）；封顶 8192 burstgpt P99 TPOT=37ms（紧贴 SLO，贪心尚可）。
- TPOT 悬崖（代码实测）：并发 ≤256 时 p50≈74–98ms，>257 时 p50=5264ms。

## 2. 被测环境

- **硬件**：8× NVIDIA A800-80GB SXM4（NVLink/NVSwitch）、CPU/SSD 见运行日志。
- **软件**：vLLM v0.11.0、PyTorch 2.8.0、CUDA 12.8、NCCL 2.27.3、Python 3.10（openrlhf env）。
- **评估**：分离式 vLLM，4P4D，Qwen-2.5-7B-Instruct，TTFT SLO 1000ms、TPOT SLO 30ms。
- **A/B 隔离**：`scheme4`（预测路由+贪心准入）vs `scheme4_gwf`（同路由+GWF 准入），唯一差别是准入控制器。

## 3. 每个请求要记录的指标（打点）

| 指标 | 来源 | 用途（判断哪类负载 GWF 更有利） |
|---|---|---|
| `prompt_len` | predict CSV | 输入长度维度 |
| `actual_output_len` | predict CSV | 实际输出长度维度 |
| `predict_output_mean` / `bucket_probs_json` | predict CSV | 预测长度维度 + 校准误差 |
| per-request TPOT / TTFT / E2EL | benchmark 输出 | 是否达标（SLO 判定） |
| is_good（是否满足 SLO） | benchmark goodput 逻辑 | goodput 归因 |
| 是否被抢占 / demote / 驱逐 | decode 日志（GWF 计数器） | GWF 部件生效证据 |
| decode 实例 + `curr_bottle`/`pred_bottle` | proxy MONITOR | 负载区间/悬崖定位 |

> 判据：把 per-request 数据按（input_len、actual_len、pred/actual 比值、TPOT）分箱，
> 看 `scheme4_gwf` 相对 `scheme4` 的 goodput 增益集中在哪个箱 → 反推"有利特征"。

## 4. 迭代循环

```
构造 trace（可配：输入长度/输出双峰/预测/突发强度）
   → 跑 scheme4 vs scheme4_gwf（各 ≥3 次取中位数）
   → 分析 goodput Δ%、P99 TPOT、抢占/demote 计数
   → 判据：GWF goodput 显著高于贪心（Δ% > 噪声）且抢占/长尾 SLO 违规更少？
       ├─ 满足 → 锁定该负载画像，写结论
       └─ 不满足 → 按 per-request 分箱结果更新画像（改哪个维度、朝哪个方向），回到构造
```

### 判据（诚实、不事后编准则）

- **主判据**：output goodput（tok/s，30ms TPOT + 1000ms TTFT SLO），`scheme4_gwf` vs `scheme4` 中位数差。
- **辅证**：P99 TPOT、抢占次数、demote/hopeless 计数、分箱后的短请求 goodput。
- **不预设方向**：如果某维度（如预测校准）无效，就照实记录并降权该维度；只靠实测 TPOT 的 θ 窗口 + hopeless 能赢，也算成功。

## 5. 当前状态与下一步

- 已完成：worked example（`worked_example/`）、封顶 8192 burstgpt A/B（在跑，rep1 scheme4 goodput=1956.75、P99 TPOT=37.32ms）。
- 已就绪：双峰+突发 trace（`admission_ab_base/bimodal_burst/`，300 长×8000 + 300 短×200，10 突发/600s）。
- 下一步：等封顶 A/B 出 `scheme4_gwf` 结果 → 若增益小（封顶版并发 <257 不触发悬崖），换双峰 trace（300 长请求 → 并发 ~300 > 257）重跑。
