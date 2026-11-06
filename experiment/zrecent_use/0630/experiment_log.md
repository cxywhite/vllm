# 实验日志（workload 迭代 + 踩坑记录）

> 目标：每次实验跑完，记录对"调整负载"有用的结论 + 踩过的坑，驱动下一轮负载构造。
> 与 `workload_hypothesis.md`（静态假设）配套；本文件是动态流水账，按时间追加。

## 迭代记录（每次实验一行）

| # | 时间 | trace | 预测 | 单次时长 | 关键结果 | 对负载调整的启示 |
|---|---|---|---|---|---|---|
| 1 | 11-06 | qwen 4080（uncapped，50 条 30k extreme） | BERT（坏） | ~1.7h | greedy p99 TPOT=156ms（5× SLO），25% 负载 goodput=317 tok/s | extreme 请求既拖慢又制造压力；greedy 在稠密 trace 上确实失控 |
| 2 | 11-06 | burstgpt 673（uncapped，150 条 >8192） | BERT（坏） | ~55min | live pred/actual 中位数 12.87×（93% 过预测>2×） | 预测 miscalibration 是系统性的，与 trace 无关 |
| 3 | 11-06 | burstgpt cap8192 | BERT（坏） | ~11min | greedy goodput=1956.75、p99 TPOT=37.32ms；GWF goodput=1851.73（**Δ=−5.4%**）、计数器全 0 | **封顶后负载太容易**：并发 <257 无悬崖、无 hopeless → GWF 准入门一次没触发，纯开销反而亏 5%。**结论：GWF 在"没悬崖/没 hopeless"时是负收益，必须把负载推到悬崖/长尾才轮到它赢** |
| 4 | 11-06 | bimodal（300 短×200 + 300 长×8000，10 burst） | **完美预测**（WT_SIMPLIFY_PREDICT=1） | ~9min | greedy goodput=1502.81、p99=65.82ms；GWF goodput=951.02（**Δ=−36.7%**）、计数器仍全 0、meas_tpot_p50=23.4ms | **决定性发现**：双峰负载确实更难（greedy p99 从 37→65.8ms），但 GWF 准入仍一次没触发——两个根因：① **并发不够**：`running` 峰值仅 61（< A/2=128），demote 分支 `\|running\|≥A/2` 永远进不去，window 恒 256、waiting 恒 0；② **测量口径解耦**：meas_tpot（decode 侧 23.4ms）比 SLO（client 侧 median 32.85ms）低 ~10ms。per-request：短请求 good 148/300（49%）、长请求 good 105/300（35%） |
| 5 | 11-06 | bimodal（300 短×200 + 300 长×**20000**，10 burst） | 完美预测 | ~12min 后崩溃 | benchmark exit 137（SIGKILL）+ "GPU deadlock"；未到 scheme4_gwf 就死 | **假设错了**：加长到 20000 反而**撞 KV 容量**。实测单 decode 实例 GPU KV=**881,616 tokens**；20000+prompt≈20056 → 最多 ~44/实例并发，远 < 128。触发 demote/悬崖要 output ≤ ~3400（悬崖）/~6800（demote） |
| 6 | 11-06 | dense_short（12000 短×200，300s 内 ~40 req/s） | 完美预测 | ~7min | **正向结果**：greedy goodput=2566.19、P99 TPOT=55.17ms；GWF goodput=**2637.21（+2.8%）**、P99 TPOT=**47.39ms（−14.1%）**；GWF running 峰值 **256**、demoted=**1525**、stop_window=**179**、preempted=0 | **第一次 GWF 真正赢**。高并发短请求（200 token、40 req/s）把 running 推到 256（窗口上限）、meas_tpot 到 46–48ms（>30ms）→ hopeless 驱逐 + 窗口限流真正生效。**注意：离线模拟曾预测"两者 goodput≈0"，与实际不符——因为旧 4080 TPOT 曲线（256 并发 97ms）不适用于短请求（实际 ~29ms），短请求 goodput 非零** |

| 7 | 11-06 | dense_short 第二轮（同 12000×200） | 完美预测 | ~6min/轮 | greedy 2326.8 / GWF **2442.6（+5.0%）**；good 3690→3896；tpot p99 55.3→47.8(−13.4%)、max 58.8→49.4；ttft p50 823→701、p99 5435→4336(−20.2%)、max 7784→5446(−30%)；GWF：decode2/3 卡 256、demoted=194、stop_window=115、preempted=0 | 与第 6 轮同向（+2.8% / +5.0%），**两轮都正、方向一致，但幅度差 2pp → run-to-run 方差约 7–9%，必须 3 rep 取中位数确认**。同时：greedy 两轮 goodput 差 9%（2566 vs 2327），单轮不可信 |

## 结论快照（随迭代更新）

- **KV 容量是并发上限（关键新认知）**：单 decode 实例 GPU KV=**881,616 tokens**（decode0 日志 kv_cache_utils）。并发 running ≤ 881616/(prompt+output)。所以：
  - 要触发 demote（running≥A/2=128）→ output ≤ ~6800；
  - 要触发悬崖/stop_window（running≥A=256）→ output ≤ ~3400。
  - 之前的 8000/20000 长请求分别把并发卡在 ~107/~44，**根本到不了触发阈值**——这就是 GWF 计数器全 0 的根因。
- **GWF 只在两个过载区出手**：并发过载（running≥256→stop_window + demote）和 KV 过载（KV 满→preempt+eV）。当前 8000/20000 trace 是"KV 受限"但并发不到 128，两者都不触发。
- **真实干净负载 TPOT 曲线（2026-11-06 拟合，1136 个 (running,meas_tpot) 样本）**：decode 侧 meas_tpot ≈ **20 + 0.11×running** ms，近似线性，**没有 5264ms 悬崖**（旧 4080 的悬崖是 30k extreme + KV thrash 造成的假象）。分桶 p50：0-31→19.7、64-95→29.7、128-159→38.1、224-255→47.9、256-287→49.6ms。
- **θ=120ms 实际够不到**：真实 meas_tpot 最大 ~61ms << 120ms，所以 window 永远不收缩、恒 256，GWF 退化成"固定 256 上限 + 淘汰 hopeless"。**这是 +5% 增益的天花板来源**：goodput 阈值 30ms 对应 running ≈ 64，而窗口上限 256 是它的 4 倍。
- **触发≠有 goodput 增益**：goodput 靠"淘汰 hopeless + 压低尾部"得来（本轮 tpot p99 −13%、ttft p99 −20%）。要更大增益，需 (a) 更多实例饱和（提高密度）、或 (b) 让窗口真正收缩（output≈3000 使 KV 上限≈289，撞满→thrash→meas_tpot 冲上 120ms）。
- **负载热点不平衡（2–3×）**：实测 decode0/1/2/3 running 峰值 61/28/96/18（bimodal8000）、105/115/256/256（dense_short）——predictor 路由把请求集中到部分 decode 实例，反而**降低**触发门槛（热实例先满）。
- **预测 miscalibration**（系统 8–13× 过预测）仍是独立短板；已用 `WT_SIMPLIFY_PREDICT=1` 隔离。
- **并发峰值的构造公式（实测拟合）**：`peak_running_inst ≈ λ_inst × min(B, S)`（λ_inst=每突发请求数/B/4，B=突发时长，S=output×tpot/1000）。dense_short 实测 λ=60/s、B=5s、S≈9.6s → 预测 300、实测 338。要跨 256 需每 5s 突发 ≥1024 条；「短而密」比「长而疏」更容易触发（B 拉长到 10s 时 1200 条只有 150 峰值）。
- **硬件信号能测什么、不能测什么**：能测——sm 判断 prefill/decode 瓶颈、mem_pct 判断是否 memory-bound、load1 判断 CPU（proxy/scheduler）瓶颈、`mem_avail_mib` 佐证 `--swap-space 0`（无 KV 换出）、`fb_mb` 判"有没有别的容器占卡"；不能测——framebuffer 当并发、`rxpci/txpci` 当 KV 流量（P2pNccl 走 NVLink，PCIe 计数恒 0；KV 按 57,344 B/token 记账）。
- **✅ 已找到有提升的负载（迭代 #6）**：**高并发短请求**（200 token、~40 req/s、bursty、300s）→ per-instance running 到 **256**（窗口上限）、meas_tpot 到 **46–48ms**（>30ms hopeless 阈值）→ GWF 的 demote(1525)/stop_window(179) 真正触发 → **goodput +2.8%、P99 TPOT −14.1%**。负载画像：**短输出（~200 token）+ 高到达率（~40 req/s）+ 突发**。这是 GWF 能赢的唯一 regime。

| 8 | 11-07 | 硬件采样 + 离线模拟校准（不跑新 A/B） | — | — | 见 `hw_guided_workload_and_sim.md` 与 `worked_example/sim_vs_real_dense_short.md` | ① 采样开销实测 0 tick/58s，可全程常驻；② **GPU fb 不能当并发代理**（vLLM 预分配 KV，running 5→338 时 fb 只差几十 MiB），并发只能看新增的 `kv_usage_pct`；③ 修掉模拟器跨策略状态污染（旧 GWF 模拟从空队列起跑，`max_running=11`），修后仍**符号相反**（实测 +3.6% vs 模拟 −15.7%）→ 模拟器暂只能用来筛"到不了 128"的负载 |

| 9 | 11-07 | 论文 mooncake trace（n=1790，6 req/s，output p90=12.7k） | 观察（别人在跑） | — | 首 7 分钟：greedy kv_usage **100 %**（80/1805 采样 ≥99.5 %）、running 257、waiting 18、**preempted_recompute_tokens=33,633**、日志 preempted 事件 1822 次、watchdog stalled 17 次 | **修正 #6 的过度概括**：长输出不是不能触发 GWF，准确门槛是「运行点平均 seq」——`slots=881616/avg_seq`，mooncake 运行时均值 ~4.6k → 仍有 ~190 slots > 128 门槛。这里 KV 打满 → 原生抢占+整段重算（--swap-space 0），正是论文 eV 该省的浪费。注意风险：100 % KV 时有 deadlock 看门狗警告，极端 20k 输出曾 exit 137，推进要小步（先 1.2× 到达率） |

## 踩过的坑（按时间，含修复）

1. **模型权重路径**：脚本默认 `/root/.cache/.../Qwen-2.5-7b-Instruct` 在慢盘 /dev/sdb1 → 卡死。修复：`QWEN_MODEL_PATH=/root/myshare/models/...`（NVMe）。
2. **proxy InvalidURL**：`no_proxy` 含 `::1`/`[::1]` → httpx 崩。修复：`no_proxy=localhost,127.0.0.1`。
3. **c10d 卡死**：未设 `VLLM_HOST_IP` 时 `get_ip()` 取 hostname 连不通。修复：`VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo`。判据：GPU 显存停在 ~498 MiB。
4. **zmq 端口冲突**：上一轮被杀进程的 predictor ZMQ socket（32323）没释放 → 下一轮首启动 `Address already in use`。评估脚本的重启机制会自动清理。
5. **`pkill -f` 自匹配**：自己 shell 命令行含匹配串被杀。改用 `ps -eo pid,cmd | grep -E '...' | grep -v grep | awk '{print $1}'` 取 PID 再 kill。
6. **vllm 是 git 子模块**：`git add vllm/...` 会报 "is in submodule"；需 `cd vllm && git add/commit/push` 单独提交。
7. **extreme 请求拖慢 benchmark**：`--ignore-eos` 强制生成满 30k token → 单次 1.7h；封顶 output（≤8192）可降到 ~11min。
8. **长请求撞 KV 容量**：单 decode 实例 GPU KV=881,616 tokens。output 8000/20000 → 并发分别卡在 ~107/~44，触发不了 GWF 门。判据：`grep "GPU KV cache size" decode*.log`。
9. **GPU deadlock → benchmark exit 137**：日志 `[WARN] GPU deadlock detected (decode GPUs ... util=0)`，随后 benchmark 被 SIGKILL。重负载 trace 下 decode 实例会短时假死，看门狗 300s 倒计时；若没恢复则整轮报废。
10. **离线模拟器 `worked_example/sim_admission.py`**：秒级预测 greedy/GWF 的 max_running/waiting/gate 计数器。注意 feed 用 `share=imbalance/N`（实测热实例 ~3× 平衡，默认 `--imbalance 3.0`）；goodput 预测受 TPOT 曲线校准限制（旧 4080 曲线不可靠，需 `wt_concurrency_probe.py` 重测干净负载曲线）。

## 环境（每次实验都应记录）

- 8× A800-80GB，4P4D，Qwen-2.5-7B-Instruct，TTFT SLO 1000ms / TPOT SLO 30ms，vLLM v0.11.0。
- 权重路径 `/root/myshare/models/`；启动前必设 `VLLM_HOST_IP=127.0.0.1`、`no_proxy=localhost,127.0.0.1`。
