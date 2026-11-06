# 硬件指标 → 工作负载构造 / 离线模拟 校准指南

> 适用：`experiment/zrecent_use/0630/test_evaluation_scheme4_rr_compare.sh` 的准入隔离 A/B
> （`scheme4` = 贪心准入，`scheme4_gwf` = GWF 准入）。
> 本文所有数字都来自本仓库的真实运行，出处见每节末尾的「证据」。

---

## 0. 先记住三条结论（踩过的坑）

1. **GPU framebuffer 不能当并发代理**：vLLM 按 `gpu_memory_utilization` 预分配 KV cache，
   `running=5` 和 `running=300` 的 `fb_mb` 几乎一样（实测 4P4D Qwen-7B：prefill 64,326–64,360 MiB，
   decode 66,348–66,416 MiB，全程只差几十 MiB）。并发只能看调度器日志里的
   `kv_usage_pct`（2026-11-07 起 `[WT][NATIVE]` / `[WT][GWF]` 都打印）。
2. **`meas_tpot_ms` 不是瞬时步长**：它是每个请求的「首 token 到现在」的**生命周期平均**
   （`_observed_tpot_s = elapsed / gen_delta`）。所以在负载爬升期它滞后，
   按 running 分桶得到的曲线会**高估**瞬时 TPOT（实测约 1.4–1.6×）。离线模拟要乘 `--curve-scale≈0.7`。
3. **P2pNccl 的 KV 传输走 NVLink，不走 PCIe**：`dmon` 的 `rxpci/txpci` 全程 ~0，
   别拿它当 KV 流量指标；KV 体积按 token 记账（Qwen-2.5-7B：**57,344 B/token**）。

---

## 1. 采集：零扰动的硬件采样

| 组件 | 作用 | 开销 |
|---|---|---|
| `hw_metrics_sampler_v2.py` | 单进程 `nvidia-smi dmon` + `/proc` 读数，5 s 一次，CSV | 实测 58 s 内 `utime+stime = 0` tick（<10 ms），约 0.05 % 单核 |
| `run_ab_with_hw.sh` | 包住评测脚本：**开跑前查 GPU 是否被别人占用** → 起采样器（`nice -n 19` + `setsid`）→ 跑 A/B → 停采样器并打摘要 | 与不包一层相同 |
| `hw_workload_report.py` | 事后报告 + `--watch` 实时投影 | 纯离线读 CSV |

采样列（全部带单位）：`load1/5/15`、`mem_avail_mib`、每卡 `pwr_w / sm_pct / mem_pct / fb_mb / rx_mb_s / tx_mb_s`，
外加 `ts_epoch_s` 供与 per-request CSV 对齐。

**为什么不会影响实验**
- 只有一个常驻 `nvidia-smi dmon`，不 spawn 子进程；v1 版本每 5 s spawn 一次 `nvidia-smi`（每次 40–60 ms），
  GPU 数量多时可能到 1 % 单核，故改用 v2。
- 采样器用 `setsid` 独立进程组，评测脚本 cleanup 里的进程组 kill 不会误杀它；
  `pkill -f "vllm serve"` 也匹配不到它。
- 只读：不写 GPU、不改 sysctl、不碰文件系统（除自己的 CSV）。

**必做的前置检查**（wrapper 已内置）：开跑前 `nvidia-smi --query-gpu=memory.used`，
>2000 MiB 就拒绝启动。2026-11-07 06:05 实测过一次：8 卡被另一个容器的作业占了 63 GB，
>此时起 A/B 只会得到被污染的 goodput，浪费 40 分钟。

证据：`/root/myshare/hw_sampler_selftest/selftest_60s.csv`（10 个样本、5.7 s 间隔、`utime=0 stime=0`）。

---

## 2. 用硬件指标构造工作负载（本轮实测规律）

### 2.1 决定峰值并发的是「突发密度 × 服务时间」，不是平均 QPS

实测（`admission_ab_dense_short`，12000 条 × 200 token，10 个突发、每突发 1200 条 / 5 s）：

| 指标 | 值 |
|---|---|
| 平均到达率 | 43.6 req/s（275 s 跨度） |
| 峰值 1 s / 5 s 窗口 | 280 / 1200 条（= 240 req/s cluster，60 req/s/实例） |
| 贪心准入实测峰值 running | **338** |
| GWF 实测峰值 running | **256**（撞上 `A_max=256` 窗口） |
| 实测 `kv_usage_pct` 峰值 | < 60 %（KV 远未打满） |

拟合出的构造规则：

```
peak_running_per_instance ≈ λ_instance × min(B, S)
    λ_instance = 每突发请求数 / B / 4        # 4 个 decode 实例
    B          = 突发持续时间 (s)
    S          = 服务时间 ≈ output_tokens × tpot_ms(running)/1000
```

对本轮：`λ_inst = 1200/5/4 = 60`，`B=5`，`S≈9.6 s` → `peak ≈ 60×5 = 300`，实测 338（+13 %，够用）。

**要跨过 GWF 的两道门槛**（`demote: running ≥ A/2 = 128`，`窗口收缩: running ≥ A = 256`）：

| 目标 | 需要（200 token 请求、B=5 s） | 等价做法 |
|---|---|---|
| `running ≥ 128` | `λ_inst ≥ 25.6/s` → 每突发 ≥ 512 条 | 1200 条/5 s 已 2.3× 富余 |
| `running ≥ 256` | `λ_inst ≥ 51/s` → 每突发 ≥ 1024 条 | 1200 条/5 s 刚好够 |

想更省事就**拉长 B**：突发时长翻倍（10 s）而请求数不变，`λ_inst` 减半，峰值相对更低但持续更久；
由于 `B < S` 时峰值只由 `λ×B` 决定，**`1200 条/10 s` 峰值只有 150**，会掉到 128 门槛下面。
所以「短而密」比「长而疏」更容易触发 GWF。

### 2.2 用 sm / mem_pct / power 判断瓶颈在哪

| 现象 | 含义 | 该调的旋钮 |
|---|---|---|
| decode `sm_pct` 均值 50–70 %、`mem_pct` 高 | 正常解码态（memory-bound） | 想加并发就加突发密度 |
| decode `sm_pct` > 90 % | 解码算力打满 | 再加负载只会推高 TPOT（这正是 GWF 该赢的区域） |
| prefill `sm_pct` 尖峰 ~100 %、decode 反而空闲 | prefill 瓶颈 | 缩短 input 长度；或改 2P6D（不是加 rate） |
| 两边都 < 60 % | 机器有空闲 | 先加 rate，别动 GWF 参数 |
| `load1` 接近核数 | proxy/scheduler CPU 成瓶颈 | 降 rate；这类抖动会污染 A/B |

实测基线（bimodal20k，长请求 8000 token）：prefill sm 均值 0 %（突发太短、5 s 采样抓不到尖峰）、
decode sm 均值 53 %、`load1` 54 → 说明当时是「decode 常年忙碌 + prefill 偶发尖峰」的形态，
这也是 `fb` 看起来「两边都满」的原因（预分配）。

### 2.3 每次跑完的固定动作

```bash
python3 hw_workload_report.py --hw-csv <result>/hw_metrics_*.csv \
    --sched-glob '<result>/*/log/decode*.log' \
    --per-req-csv <result>/*/dataset_result/test_results_*.csv \
    --out <result>/hw_report.md
```

报告会直接回答：峰值 running 离 128/256 多远、瓶颈在 prefill 还是 decode、下一次该动哪个旋钮。

---

## 3. 辅助离线模拟（`worked_example/sim_admission.py`）

### 3.1 模拟器需要什么、从哪来

| 输入 | 来源 | 脚本 |
|---|---|---|
| `tpot(running)` 曲线 | 调度器 1 s 日志的 `(running, meas_tpot_ms)` 分桶 p50 | `extract_service_model.py` |
| KV 容量 | decode 引擎日志 `GPU KV cache size: 881,616 tokens` | 同上 |
| 真实 goodput / TTFT / TPOT 分位 | bench 日志 + per-request CSV | 同上 |
| 路由不均衡度 | 4 个 decode 实例的峰值 running 之比 | 同上（`routing_imbalance`） |
| 突发结构 | trace CSV 的到达时间 | `load_trace()` |

```bash
python3 extract_service_model.py --run-dir <...scheme4_gwf> --out service_models/dense_short_gwf.json
python3 worked_example/sim_admission.py --trace <trace.csv> \
    --curve-json service_models/dense_short_gwf.json --curve-scale 0.7 --imbalance 1.0
```

### 3.2 已经发现的三个模拟器缺陷（本轮修复/待修）

1. **跨策略状态污染（已修）**：`run_one()` 用同一批 `Req` 对象先后跑 greedy 和 GWF，
   第一遍跑完所有请求都是 `finished=True`，于是 GWF 那一遍从一个空队列起跑
   （`max_running=11` 而不是 ~300）。**此前任何「模拟显示 GWF 输/赢 X%」的结论作废。**
   修复：`simulate()` 入口重置请求运行时字段。
2. **曲线尺度（已加 `--curve-scale`）**：分桶曲线是生命周期平均，需乘 ~0.7
   才能复现实测 goodput 水平（模拟 34 % good 请求 vs 实测 33.6 %）。
   模型 JSON 里有 `curve_scale_suggested` 作为初值。
3. **模拟器还不会输出 TTFT/TPOT 分位数（待补）**：现在的 goodput 只用
   `tpot_of()` + 固定 1000 ms TTFT 判据，无法与实测 P99 对齐。

### 3.3 目前模拟器还**不能**用来排序工作负载

`sim_vs_real.py` 的校验结果（`worked_example/sim_vs_real_dense_short.md`）：

| 指标 | 实测贪心 | 实测 GWF | 模拟贪心 | 模拟 GWF |
|---|---|---|---|---|
| goodput (req/s) | 13.25 | 13.73 | 13.61 | 11.47 |
| Δ GWF vs 贪心 | — | **+3.6 %** | — | **−15.7 %** |
| 峰值 running | 338 | 256 | 296 | 256 |
| demoted 总数 | 0 | 1525 | 0 | 615 |
| stop_window 总数 | 0 | 179 | 0 | 221 |

**符号都不对**（实测 GWF 小胜，模拟判 GWF 输 16 %）。在修好之前：

- 模拟器只用来**筛掉**明显不可能触发门槛的工作负载（峰值 running < 128 的那类），
  这是它现在唯一可信的用途；
- 不要用它给出「GWF 提升 X %」的数字；
- 工作负载排序仍然要真实 A/B（1 rep 快速试错，胜出后 3 rep 确认）。

下一步修模拟器的优先级：
1. `is_hopeless()` 用「排队等待而非服务时间」判据（现在把大量排队请求判成 hopeless，
   造成 615 次 demote 的无效轮转）；
2. 加入 TTFT 分位数输出，与实测 P99 对齐；
3. 把 `--imbalance` 从「到达率倍数」改成「按预测长请求的集中路由」（实测 imbalance 1.0–1.3，
   默认 1.6 偏大）。

---

## 4. 本轮真实 A/B 结果（首个 GWF 正收益的负载）

`admission_ab_dense_short`，12000 × 200 token，4P4D，`WT_SIMPLIFY_PREDICT=1`，1 rep：

| 指标 | scheme4（贪心） | scheme4_gwf | Δ |
|---|---|---|---|
| Request goodput | 13.25 req/s | **13.73 req/s** | **+3.6 %** |
| Output token goodput | 2566 tok/s | 2637 tok/s | +2.8 % |
| Mean TTFT | 1230 ms | **1140 ms** | −7.3 % |
| P99 TTFT | 5412 ms | **4334 ms** | −19.9 % |
| P99 TPOT | 55.2 ms | **47.4 ms** | −14.1 % |
| P99 E2EL | 11791 ms | **10620 ms** | −9.9 % |
| 峰值 running | 338 | 256 | 窗口生效 |
| demoted / stop_window | 0 / 0 | 1525 / 179 | 首次真正触发 |

注意：1 rep，尚未做 3 rep 中位数确认；`WT_SIMPLIFY_PREDICT=1` 是完美预测，
不代表带 BERT 预测器的端到端效果。
