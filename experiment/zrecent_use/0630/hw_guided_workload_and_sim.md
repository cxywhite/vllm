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

---

## 5. 下一轮建议（按优先级）

### 5.1 放大 GWF 的分离度：更密的突发，而不是更多的长请求

第 6/7 轮（12000×200、突发 1200 条/5 s）已经让 GWF 赢 2.8 %/5.0 %，但贪心只到 running 338、
GWF 撞 256 上限；把突发密度再提一档可以让分离度更明显，同时保持「40 req/s 平均」这个现实画像：

```
TRACE_SPAN_S=275 BURST_PERIOD_S=27.5 BURST_DURATION_S=5 \
REQS_PER_BURST=2000 LONG_FRACTION=0 OUTPUT_SHORT_TOKENS=200 \
  python3 build_bimodal_burst_trace.py        # 6 个突发 × 2000 条 = 12000 条
```

预期：`λ_inst = 2000/5/4 = 100/s`，`peak ≈ 100×5 = 500`（贪心）vs 256（GWF）——
贪心的 P99 TTFT 会明显恶化，GWF 的限流窗口收益放大。KV 用量仍 < 15 %（500×256 token）。

**不要**用「加长输出到 8000/20000」的方式加大压力：那会先撞 KV 容量
（output 8000 → 每实例并发上限 ~107，到不了 128 门槛；output 20000 → 只有 ~44）。

### 5.2 确认用的启动命令（带硬件采样）

```bash
cd /root/predict-schedule/experiment/zrecent_use/0630
VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct \
WT_SIMPLIFY_PREDICT=1 WT_NATIVE_LOG_INTERVAL_MS=1000 \
COMPARE_EXPERIMENT_VARIANTS="scheme4 scheme4_gwf" PD_RATIO_LIST="4:4" \
TPOT_MS_LIST="30" RUN_REPEAT_PER_CONFIG=3 TRACE_NAME=bimodal_burst \
DATASET_BASE_DIR=$PWD/admission_ab_base \
BASE_RESULT_DIR=$PWD/admission_ab_confirm \
  bash run_ab_with_hw.sh
```

6 次运行 ≈ 45 min；产物里会多出 `hw_metrics_<stamp>.csv`（5 s 粒度）与
`hw_metrics_<stamp>.sampler.log`。跑完立刻：

```bash
python3 hw_workload_report.py --hw-csv admission_ab_confirm/hw_metrics_*.csv \
  --sched-glob 'admission_ab_confirm/*/log/decode*.log' --out admission_ab_confirm/hw_report.md
```

### 5.3 模拟器修复顺序（在用它排序工作负载之前）

1. 让 `is_hopeless()` 只对「还没拿到首 token 且排队超 SLO」的请求生效，
   并把 demote 的每步上限与真实值对齐（实测 1525 次/轮，模拟 615 次/轮 → 需按实例数归一）；
2. 输出 TTFT/TPOT 分位数，与 bench 的 P50/P99 对齐；
3. `--imbalance` 改为按「预测长度路由」建模（实测 1.0–1.3，默认 1.6 偏大）。

---

## 2.4 第三种regime：KV 受限区（论文自己的 mooncake trace 就在这里）

之前我把「长输出」一棍子打死（output 8000/20000 → 并发被 KV 卡在 107/44，触发不了 128 门槛），
**这个结论只对极端长输出成立**。准确的门槛是「运行点的平均序列长度」：

```
KV 上限并发 slots = 881,616 tokens / avg_seq     (avg_seq = prompt + output，运行时的均值)
demote 门可达  <=>  slots >= 128   <=>  avg_seq <= 6,887
窗口门可达    <=>  slots >= 256   <=>  avg_seq <= 3,444
```

实测（`admission_ab_mooncake`，论文 mooncake trace n=1790，4P4D，6.0 req/s）：

| 指标 | 实测 |
|---|---|
| output p50 / p90 / max | 515 / 12,730 / 30,944 |
| seq(prompt+out) p50 / p90 | 619 / 12,768 |
| **kv_usage_pct** | p50 62.5 %，p90 95.6 %，**max 100 %（1805 个采样里 80 个 ≥99.5 %）** |
| running | max **257**（略超 A_max=256） |
| waiting | max 18 |
| **preempted_recompute_tokens** | **33,633**（累计，单调递增）→ 真实发生了「抢占 + 整段重算」 |
| 日志里 preempted 事件 | 1,822 次；watchdog `stalled` 警告 17 次 |

**这是比 dense_short 更贴论文的场景**：KV 打满 → 原生调度器被迫抢占正在跑的请求、丢弃其 KV、
从 0 重算（`--swap-space 0`，没有 CPU 换出）。GWF 的 eV（淘汰价值）正是为这里设计的。
`preempted_recompute_tokens` 就是「白干的工作量」，GWF 若有效，这个数应该显著变小。

**构造公式补全**：

```
peak_running ≈ min( λ_inst × min(B, S),  881,616 / avg_seq,  max_num_seqs )
```

注意 KV 项：`avg_seq` 用的是**运行中的均值**，不是 trace 里的最大值——
mooncake 的 p90 是 12.7k token，但运行时均值只有 ~4.6k（长请求少、短请求多），所以 slots 还有 ~190。

**逐实例看（2026-11-07 实测，同一轮 mooncake 的 4 个 decode）**：

| 实例 | running max/p50 | kv_pct p50/max | 抢占重算 tokens |
|---|---|---|---|
| decode0 | 187 / 144 | 62.2 / 100.0 | 21,269 |
| decode1 | 209 / 167 | 67.9 / 100.0 | 87,905 |
| decode2 | 237 / 176 | 62.2 / 100.0 | 15,504 |
| decode3 | 257 / 181 | 66.6 / 100.0 | 96,506 |

**方法教训（我踩过）**：把 4 个 decode 实例的 `running`/`kv_pct` 混在一起算平均，
会得到「running 200–300 的桶 kv_pct 反而比 128–200 低」这种自相矛盾的结论——
因为每个实例的请求长度分布不同（decode0 均值 ~4000 token，decode3 ~2800）。
**必须逐实例看**；`hw_workload_report.py` 现在会打 per-instance 表。

**风险**：KV 打到 100 % 时出现过 `GPU deadlock` 看门狗警告（17 次 stalled），
极端长输出（20k）曾直接把 benchmark 打挂（exit 137）。所以往 KV 受限区推要**小步**：
先 1.1–1.2× 到达率，不要一次把 output 拉到 20k。

已备好一个 1.2× 到达率的派生 trace（不改原数据集，新建文件）：
`derived_datasets/mooncake_rate1p2x/qwen_lmsys_mooncake_1790_rate1p2x.csv`
（1790 条不变、时间戳压缩到 247 s、7.3 req/s）。用途：把贪心再往 KV 悬崖里推 5–10 %，
让 `preempted_recompute_tokens` 明显大于 0，从而让 GWF 的 eV 有东西可省。

---

## 2.5 为什么窗口一直不收缩——以及什么样的负载才能让它收缩

GWF 的收缩条件是 `meas_tpot > θ = 120 ms` **且** `running ≥ A`（Algorithm 2）。
用实测干净负载曲线 `meas_tpot ≈ 19.7 + 0.117×running`（dense_short，32→288 桶）反解：

```
120 ms  ->  running ≈ 860
```

而我们实测的 running 峰值只有 **338**（贪心，dense_short 最密突发），
所以 `window` 从未低于 256，GWF 退化成「固定 256 上限 + 淘汰 hopeless」。

**这意味着 dense_short 的 +3.6 % 只是「尾部清理」的收益，不是论文窗口机制的收益。**
要让窗口真正从 256→128→64→32 收下来，必须让 `meas_tpot` 远超 120 ms。
唯一现实途径是 **KV 打满后的抢占重算抖动**——也就是 2026-09-24 那次 4080 运行里
`running ≥ 257 → meas_tpot 5264 ms` 的那个悬崖（当时由 30k-token extreme + KV thrash 造成）。

所以工作负载的正确画像是：

```
running 卡在 200–300（跨过 demote/窗口门）
  + KV 长期 ~100 %（迫使调度器不断抢占/重算，步长暴涨）
  + 平均序列长度 2,000–4,000 token（slots = 881,616/avg_seq ≈ 220–440，刚好卡在这个区间）
```

mooncake（未截断）实测就是这个形态：4 个实例 kv 全部 100 %、running p50 144–181、
单实例重算 15k–101k token。**截断到 8192 的版本会削弱它**（mean seq 3,301 → slots ≈ 267，
压力变小），观测时要对比这两版。

**判别方法（跑的时候就能看）**：GWF 变体的 `[WT][GWF]` 日志里如果出现
`window=128/64/32`，说明窗口机制真的激活了；如果 `window` 恒 256，那这轮赢的仍然只是尾部清理。

---

## 2.6 关键约束：KV 打满 → 抢占重算 → **引擎 deadlock**（两轮实测）

两轮 mooncake（未截断 / 截断 8192）都以同样的方式死掉，时间线完全一致：

| 事件 | 未截断 mooncake | 截断 mooncake |
|---|---|---|
| kv_usage 触顶 | 99.8–100 % | 99.8–100 % |
| **首次抢占** | t ≈ 447–457 s（running 164–180） | t ≈ 446–463 s（running 156–170） |
| 抢占后 20–70 s 内 | 重算 token 从 0 跳到 21k–101k，引擎停摆 | 从 0 跳到 8k–90k，引擎停摆 |
| 结束方式 | `[WARN] GPU deadlock detected (decode GPUs all mem>0 util=0)` → 300 s 倒计时 → kill → **benchmark exit 137** | 同上 |

对照：dense_short（kv < 60 %、recompute 恒 0）从未 deadlock。

**结论（写进工作负载约束）**：

1. 这个 build（PD 分离 + `P2pNcclConnector` + `--swap-space 0`）在 decode 侧做
   **本地抢占重算会在几十秒内 deadlock**。因此「KV 打满」的负载不可用——
   不是慢，是直接报废整轮。
2. 所以 **GWF 的 preemption / eV 路径在本环境无法测量**；
   能演示的只有「准入窗口 + 淘汰 hopeless」这条路径（dense_short：+3.6 %/+5.0 %）。
3. 安全设计线：**kv_usage 峰值 < 95 %**（留 5 % 余量），避免触发抢占。
   即 `running 的构造要让 KV 有富余`：`avg_seq × peak_running ≲ 830k`。
4. 由此推论：**窗口收缩（`meas_tpot > 120 ms`）在本环境也基本不可达**，
   因为唯一能把 TPOT 顶过 120 ms 的机制就是 KV 抖动，而它会 deadlock。
   → 若要用到论文的窗口机制，只能调 θ（`WT_GWF_WINDOW_TPOT_TARGET_MS`，
   当前 120 ms = 4×SLO；实测 TPOT 在 running≈220 时才 ~45 ms）。
   这是一个**参数敏感性实验**，不是工作负载问题，需要单独决定是否做。

---

## 2.7 收益从哪来、上限是多少（dense_short 两轮逐请求归因）

把两轮 12000 条请求按 SLO 归因（`test_results_*.csv`，ttft/tpot 单位是秒，已 ×1000）：

| | 贪心 | GWF | Δ |
|---|---|---|---|
| good（TTFT≤1 s 且 TPOT≤30 ms） | 3690（30.8 %） | 3896（**32.5 %**） | +206 条 |
| TTFT 超（>1 s） | 5009（41.7 %） | 4289（**35.7 %**） | **−720 条** |
| TPOT 超（>30 ms） | 6165（51.4 %） | 6380（53.2 %） | +215 条 |
| 只有 TTFT 超（TPOT 达标） | 2145 | 1724 | −421 |
| 只有 TPOT 超（TTFT 达标） | 3301 | 3815 | +514 |

**结论**：GWF 是在**用 TPOT 换 TTFT**——队列侧的违规减少 720 条，解码侧反而多 215 条，
净赚 206 条。这不是 bug，而是「窗口 256 仍高于 TPOT≈30 ms 对应的 running≈64」的必然结果。

**收益上限**：能救的池子 = 「只有 TTFT 超」的请求 = 2145/12000 ≈ 17.9 %；
实测救回 421 条，约 **20 % 的池子**。所以同一负载下 goodput 理论上限 ≈ +17.9 %，
目前 +3.6 % 还有空间，但空间在「让 GWF 更精准地挑出 TTFT 会超、而 TPOT 不会超的请求」。

**由此得到的负载设计原则**（和之前反过来了）：

- 不要以为「越挤越好」：挤到 running 300+ 时几乎人人都 TPOT 超，GWF 无处施力；
- 目标画像是 **TTFT 违规占主导、TPOT 违规占比小**：突发把队列顶起来（TTFT 超），
  但突发总工作量小、能在 TPOT 恶化前排空（running 峰值 150–250 后快速回落到 <100）；
- 这类 trace 的判别：跑出来 `tpot_bad` 占比 < 40 %、`ttft_bad` 占比 > 50 %。

---

## 2.8 修正：持续负载下「TTFT 主导、TPOT 干净」不可能——改用振荡负载

§2.7 我写的「让 TTFT 违规占主导、TPOT 违规占比小」在**持续负载**下自相矛盾，
因为 demote 门和 TPOT 违规区重叠：

| running | decode tpot (20+0.11r) | 客户端 tpot (+≈6 ms) | 对 30 ms SLO |
|---|---|---|---|
| 36 | 24.0 ms | 30 ms | 刚好 |
| 64 | 27.0 ms | 33 ms | 已超 |
| **128（demote 门）** | **34.1 ms** | **40 ms** | **必然超** |
| 160 | 37.6 ms | 44 ms | 必然超 |
| 338（dense_short 峰值） | 57.2 ms | 63 ms | 必然超 |

也就是说 rate600 / rate900 / dense_short 三档**全在 TPOT 违规区**，
单纯降低平均密度不会让「可救池（只有 TTFT 超）」变大。

**唯一能同时做到「突发期 TTFT 爆」+「有低并发期把 TPOT 洗回来」的是振荡负载**：
短突发（把队列和 running 顶到 130–200）→ 长静默（并发回落到 <64，deferred 请求在这里被服务）。

### 关键设计约束：请求必须「跨过突发尾部」

请求的 TPOT 是**生命周期平均**。如果它在突发内就解码完（`S(r) < B`），
它的平均 TPOT 就是突发期的高值 → 仍然 TPOT-bad，静默期白留。

```
S(r) = out_tokens × (20 + 0.11·r) / 1000   (s)
要求 S(r) > B - 到达偏移   →  out_tokens > B×1000 / (20+0.11r)
r=160, B=5 s  →  out > 136 tok   (200 tok ✓, S=7.5 s)
r=160, B=10 s →  out > 272 tok   (200 tok ✗，会整段落在突发内)
```

### 参数表（由实测曲线反解，`r = λ·S(r)` 稳态）

| B (s) | G (s) | 目标峰值 r | λ (req/s/实例) | 每突发 N | 平均 (req/s) | KV 峰值 | 预填耗时 |
|---|---|---|---|---|---|---|---|
| 5 | 60 | 130 | 18.8 | 600 | 9.2 | 4.4 % | 0.84 s |
| 5 | 60 | 160 | 21.3 | 600 | 9.2 | 5.4 % | 0.84 s |
| 5 | 60 | 200 | 23.8 | 600 | 9.2 | 6.8 % | 0.84 s |
| 5 | 30 | 160 | 21.3 | 600 | 17.1 | 5.4 % | 0.84 s |
| 10 | 60 | 160 | 21.3 | 851 | 12.2 | 5.4 % | 1.19 s |

注意 `λ` 对 r 不敏感（r 越大 S 越长，反馈自我平衡），所以「每突发 N」几乎只由 B 决定：
`N ≈ 4 × λ × B`，B=5、λ≈21 → 600 条/突发就是 r≈160 的设计点。
KV 全在 7 % 以下 → 离 §2.6 的 deadlock 线极远，安全。

### 已生成的振荡 trace

- `admission_ab_base/osc_slow/`：B=5 s、周期 65 s（G=60）、600 条/突发、5400 条、平均 10.3 req/s
- `admission_ab_base/osc_fast/`：B=5 s、周期 35 s（G=30）、600 条/突发、10200 条、平均 18.1 req/s
- 两条的峰密度都是 **120 req/s cluster（30/s/实例）→ 设计峰值 running≈150**

跑完用 `ab_compare.py` 看归因：期望 `ttft_bad` 占比高、`tpot_bad` 占比明显低于 dense_short 的 51 %。

---

## 2.9 三组实验画出的边界：GWF 只在「会越界抢占」时为正

| 负载 | decode gmu | greedy 峰值 running | greedy kv% | greedy recompute | GWF Δ goodput |
|---|---|---|---|---|---|
| qwen_o1500（3P4D） | 0.5 | 366 / 494 | **100 %** | 426k / 426k / 259k | **+68.8 % / +14.3 %** |
| qwen_ln2k_cap4k（3P4D） | 0.7 | 247 | **34.8 %** | 0 | **−18.9 %** |
| dense_short（4P4D） | 0.8 | 294–338 | **5.5 %** | 0 | +2.8 % / +5.0 % / — |

规律（用可测的量表达，不用推算）：

```
GWF 显著为正  <=>  greedy 把 kv_usage 顶到 ~100%（即会抢占重算）
GWF 明显为负  <=>  kv 富余 但 running 已接近 256（窗口空转，只是拒绝工作）
其余情形      <=>  |Δ| 几个百分点，落在 run-to-run 噪声里（同负载 greedy goodput 实测 13.25/14.56，差 10 %）
```

为什么：窗口 256 是常数，而「KV 能装多少并发」= KV 容量 / 单请求瞬时占用，
单请求瞬时占用 ≈ prompt + 已生成的一半。o1500@gmu0.5 的容量只够 ~490 并发 →
greedy 冲到 494 就必然越界、开始重算；ln2k@gmu0.7 的容量够 ~700 并发 →
greedy 只到 247，窗口 256 只是把 3 个请求挡回去，没有任何收益。

**选负载的判据（跑之前就能估）**：
`峰值需求 ≈ 到达率 × 服务时间`，`KV 可容纳并发 ≈ (gmu×80GB − 15GB) / 57344B / (prompt + output/2)`；
要让 `GWF_window(256) ≤ KV 可容纳并发 < 峰值需求`。

---

## 2.10 如果 decode `gmu` 提到 0.8，什么负载还能让 GWF 有效

gmu=0.8、4P4D 时单 decode 实例的 KV 容量是**实测值 881,616 token**（引擎日志 `GPU KV cache size`）。
GWF 的窗口是常数 256，所以第一条是**硬约束**：

```
256 × (prompt + output/2) ≤ 881,616      # 窗口自身不能越界，否则 GWF 自己也要抢占
=> output ≤ ~6,700 token                 # prompt≈60
```

第二条是**压力条件**：greedy 必须把 KV 打满才会出现「抢占重算」，GWF 才有东西可救。
按实测可持续吞吐 ~6.0–7.5k tok/s（o1500：6.1M/1023s=5.98k；dense_short：2.4M/320s=7.5k），
并按 o1500 那组的成功比例（offered ≈1.7× 可持续）外推：

| output L | GWF@256 的 KV 占用 | 要 greedy 越界需要的 offered | 判定 |
|---|---|---|---|
| 2,000 | 31 % | ~18k tok/s（不可达） | 无效（就是 ln2k_cap4k 的 −19 %） |
| 4,000 | 60 % | ~9k tok/s（勉强） | 临界，值得一试 |
| **5,000** | **74 %** | **~12k tok/s ≈ 2.5 req/s** | **推荐** |
| 6,000 | 89 % | ~12k tok/s ≈ 2.0 req/s | 更猛，但逼近 89 %，有抢占/死锁风险 |
| ≥7,000 | >100 % | — | 窗口自己越界，禁用 |

已生成两条候选 trace（600 s、均匀长输出、10 个突发）：

```
admission_ab_base/u08_o5000/  n=1500  L=5000  2.5 req/s  7.5M tok  (约 20 min/run)
admission_ab_base/u08_o6000/  n=1200  L=6000  2.0 req/s  7.2M tok
```

**跑的时候看三个数就能判成败**：
1. greedy 的 `kv_usage_pct` 是否打到 ~100 % 且 `preempted_recompute_tokens > 0`（压力成立）；
2. GWF 的 `kv_usage_pct` 是否 ≤ 85 % 且 `window=256`（窗口没被冲破）；
3. GWF 的 `preempted_recompute_tokens` 是否 = 0（没有越界）。
若 GWF 的 kv 也到 100 % → L 太大，降到 4,500。

**注意**：这个 SLO（TPOT ≤30 ms）在 running≈256 时必然不满足（decode tpot ≈48 ms），
所以绝对 goodput 会很低（0.3–0.8 req/s），比较的是**相对增益**与 **P99 TPOT / 重算量**，
不是绝对值。若审稿人只认 goodput 绝对值，那 gmu=0.8 + 30 ms SLO 这一档本身就不友好。
