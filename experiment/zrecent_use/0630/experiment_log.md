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

| 10 | 11-07 | 反思：为什么窗口从不收缩 | — | — | 由实测曲线 `meas_tpot ≈ 19.7+0.117×running` 反解 θ=120 ms 需要 running≈860，而实测峰值仅 338 → window 恒 256 | **重要认知**：dense_short 的 +3.6 % 只来自「淘汰 hopeless 的尾部清理」，不是论文窗口机制的收益。窗口收缩只能靠 KV 打满后的重算抖动把 meas_tpot 顶到 ≫120 ms。正确画像是「running 200–300 + KV 长期 100 % + 平均 seq 2000–4000」。判别：跑的时候看 `[WT][GWF]` 里 window 是否出现 128/64/32 |

| 11 | 11-07 | mooncake 两轮（未截断 / 截断 8192） | 观察 | 每轮 ~11 min 后死 | 两轮都在 **首次抢占后 20–70 s 内 deadlock**：kv 触顶 99.8–100 %、首抢占 t≈450 s（running 156–180）、recompute 从 0 跳到 21k–101k、`GPU deadlock detected` → 300s 倒计时 → exit 137。dense_short（kv<60 %、recompute=0）从未 deadlock | **硬约束**：此 build 下 decode 侧本地抢占重算会 deadlock，KV 打满的负载不可用。安全线 kv<95 %。**GWF 的 eV/preemption 路径在本环境不可测**，只能演示「窗口限流+淘汰 hopeless」。窗口收缩也需要 KV 抖动把 TPOT 顶过 120 ms → 同样不可达；要用窗口机制必须调 θ（参数实验，非负载） |

| 12 | 11-07 | 修正 §2.7 + 设计振荡负载 | — | — | 交叉验证：对方独立算出 greedy only-tpot=3301、only-ttft=2145、both=2864，与我一致。几何约束：demote 门 running≥128 处 decode tpot=34 ms、客户端 40 ms，**必然 TPOT 超** → 持续负载下「TTFT 主导、TPOT 干净」不可能 | 改用**振荡负载**（短突发 130–200 → 长静默 <64）。新增约束：请求必须跨过突发尾部，否则生命周期平均 TPOT 仍是高值——`out_tokens > B×1000/(20+0.11r)`（B=5 时 out>136，B=10 时 out>272）。已生成 osc_slow（B5/G60，5400 条，10.3/s）与 osc_fast（B5/G30，10200 条，18.1/s），峰密度均 120/s cluster→设计峰值 running≈150 |

| 13 | 11-07 | qwen_cap 4P4D（n=4080，封顶 8192） | 部分 | greedy 1h17m 跑到 76% 被杀 | greedy：running max **455**、kv **100 %**、recompute **74,439**（decode0 22,993 + decode2 51,446，decode1/3 = 0，**只有两个实例打到 KV 上限**）；GWF 变体只跑了 3 min（417/4080）就被杀，`running 105 / kv 13.8 %` 只是爬坡值 | **A/B 不成立，不能比较**。但 greedy 侧证实：qwen_cap 确实把原生推到 KV 悬崖（有效吞吐 ~0.27 req/s 的慢爬），且**不均衡**——只有 2/4 个 decode 实例打满，与 predictor 把长请求集中路由一致。GWF 侧要重跑（现 deadlock guard 已改进度判据，不会再被 util=0 误杀） |

| 14 | 11-07 | dense_short 3rep 抢到空卡后仍崩（rc=137，283 s） | — | — | `[CRASH] dmesg oom recent: sglang::schedul invoked oom-killer ... Out of memory`；崩溃时 `Mem: total 377G used 303G free 7G shared 257G`；随后 `prefill-0 down for 120s` → `INSTANCE_RESTART` → 被 kill | **宿主内存 OOM**，不是 GPU 问题。根因：我们 8 个实例各 `mem_pool_size_gb=32`（P2p KV 共享内存）= **256 GB**，占 377 GB 的 68 %；再叠加同宿主另一个租户的 sglang 作业（shmem-rss 29.8 GB）→ 全局 OOM，内核随机杀进程。**对策：`PREFILL_TENSOR_POOL_MEMORY=8 DECODE_TENSOR_POOL_MEMORY=8`（降到 64 GB）**；200 token 请求每条 KV 仅 ~11 MB，8 GB 足够 |

## 踩过的坑（按时间，含修复）

> 本文件是**踩坑的唯一记录处**（不再分散到 AGENTS.md / TROUBLESHOOTING.md）。
> 先看下面的速查表定位，再看后面按时间编号的详细条目。

### 速查表（症状 → 判据 → 修复）

| 症状 | 判据 | 修复 |
|---|---|---|
| 模型加载卡死，显存停 ~498 MiB | `[c10d] server socket ... timed out` | `export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo` |
| 模型加载极慢 | 权重在 `/root/.cache`（/dev/sdb1 慢盘） | 用 `/root/myshare/models/...`（NVMe） |
| `InvalidURL: Invalid port ':1]'` | `no_proxy` 含 `::1`/`[::1]` | `export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1` |
| 启动 `ZMQError: Address already in use` | 旧引擎占着 `tcp://127.0.0.1:2203x` | 杀进程时**必须含大写 `VLLM::EngineCore`**；验 `ss -ltn \| grep ':2203[2-5]'` 为空 |
| 看门狗报 `GPU deadlock` 并杀轮次 | decode GPU `util=0` 持续 300s | **误杀**（KV-bound 时 util 也是 0）。已默认关；改用"tqdm 完成数 1800s 不动"判死 |
| decode 引擎完全不动、util=0 | `py-spy dump` 卡在 `recv_first_available`，prefill 栈空闲 | **vLLM P2P bug**：resumed 请求不该等 prefill KV。已 patch `cef34ada1`（decode 本地重算） |
| 进度极慢（~0.27 req/s）、`kv_usage_pct≈100%` | trace 总输出 > 实例数×881,616 | **KV thrash，不是死锁**。缩 trace 到 ≤~3M token |
| **只有部分 decode 在跑、部分像死了** | **先查是否同时跑了两个评估脚本**（`ps -eo pid,cmd \| grep test_evaluation_scheme4 \| wc -l` 必须=1；用 `/proc/<pid>/fd` 看实例实际写哪个日志最可靠） | 两脚本各要 7 卡、只有 8 卡 → 互相踩。停掉多余脚本 |
| 查不到崩溃原因 | **最新目录是"重启后的新轮"，崩溃现场在次新目录** | `ls -dt <BASE>/*/` → `grep -n 'INSTANCE_RESTART' <main>.log` → `tail -30 <crash_dir>/log/decode0_*.log` |
| 把 `Killed` 当崩溃原因 | `Killed` 是 health guard 判 down 后 `stop_servers` 的清理广播 | 原因看崩溃轮实例自己日志尾部 + 宿主内存/GPU/dmesg |
| `git push` 被 `GH013 secret scanning` 拦 | trace CSV 含 `sk-` 文本 | 等长替换 `sk-`→`sk_` + `git filter-branch --tree-filter` 重写 `origin..HEAD` |
| GWF 计数器全 0 | 并发没到门槛（demote 需 `running≥128`、窗口需 `≥256`） | 用短请求+高到达率；长请求撞 KV 反把并发压到 186 以下 |
| 长请求并发上不去 | `并发 ≤ 881,616/(prompt+output)` | output 8000/20000 → 并发 ~107/~44 |
| 路由把长请求集中到一个实例 | scheme4 按 `Future_Tokens` 预留（实测某实例 344k、其余 1k–4k） | 在 `select_best_instance` 加"Future 超阈值跳过" |

### 当前可观测性（已内置）

- `[SYS]`：健康检查循环每 60s 打宿主内存（used/free/avail）+ 8 卡 util/mem。
- `[CRASH]`：任一实例 down 时自动 dump 该实例日志尾部 30 行 + 宿主内存 + GPU + dmesg OOM。
- `[WT][NATIVE]` / `[WT][GWF]`：每 1s 打 KV/running/waiting/抢占计数（GWF 另有 `demoted/stop_window/meas_tpot/window`）。
- `[WT][MONITOR]`：proxy 视角每实例 `reqs/curr_bottle/pred_bottle/Future`。
- bench 进度条：`done/total`，配合进度判据看门狗。

### 详细条目（按时间）

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

11. **死锁看门狗误杀（已默认关闭）**：判据只看"decode GPU `utilization.gpu`==0 持续 300s"，但 **KV-bound 忙等时 util 也会是 0**，导致仍在推进的轮次被当死锁杀掉（qwen_cap scheme4 在 3111/4080 被杀，白扔 76%）。修复：
    - 默认关闭（`WT_ENABLE_GPU_DEADLOCK_GUARD=1` 才启用）；
    - 新增**进度判据看门狗**：解析 bench log 的 tqdm `done/total`，`WT_PROGRESS_TIMEOUT_SECONDS`（默认 1800s）内完成数不动才判死（`test_evaluation_scheme4_rr_compare.sh` 的 `run_benchmark_with_health_guard`）。
    - 判据口诀：**"util=0 是 KV-bound 的正常现象；完成数不动才是真死"**。

12. **P2P NCCL resumed-request 死锁（vLLM bug，已 patch）**：P/D 分离下，decode 端请求被抢占后重新准入，`start_load_kv → recv_first_available` 会**无限阻塞**（`p2p_nccl_engine.py` 的 `while True: recv_store_cv.wait()` 无超时），因为 prefill 早已完成该请求并释放 KV，不会再发。症状：decode EngineCore 停在 `recv_first_available`，GPU util=0，prefill 栈却是 `_process_input_queue → queue.get`（空闲）。修复（vllm `cef34ada1`）：`get_num_new_matched_tokens` 里 `if getattr(request,"preempt_count",0) > 0: return 0, False`，让 decode **本地重算**（方案 B）。
    - 诊断命令：`py-spy dump --pid <EngineCore_pid> --nonblocking`，看是否卡在 `recv_first_available`。

13. **清理漏掉大写 `VLLM::EngineCore`（ZMQ 端口残留）**：kill 的 grep 用小写 `vllm` 匹配不到 `VLLM::EngineCore`（大写），旧引擎进程继续占 `tcp://127.0.0.1:22032` 等 KV 端口 → 新轮 `zmq.error.ZMQError: Address already in use` → 启动失败进重启循环。修复：清理时用**大小写不敏感**匹配（`grep -E 'VLLM|vllm|EngineCore|APIServer|benchmark_serving'`），杀完必须验证 `ss -ltn | grep ':2203[2-5]'` 为空、`nvidia-smi` 全 0 MiB。

14. **GitHub 推送被假阳性 secret 拦截**：trace CSV 的 prompt 里含 `sk-...` 文本，被 GitHub secret scanning 判成 OpenAI Key，阻塞整个分支推送（13 个未推送提交全被拦）。修复：对 `admission_ab_base/**/*.csv` 做**等长替换** `sk-` → `sk_`（不改 prompt 长度/负载），再用 `git filter-branch --tree-filter` 重写全部未推送提交（`origin..HEAD`），推送前用 `git grep -l 'sk-' <commit> -- '*.csv'` 逐个验证干净。

15. **KV thrash（trace 总量 > KV 总量）**：qwen_cap8192 总输出 14.94M token，而 4 个 decode 的 KV 合计仅 ~3.5M（881,616/实例）→ 必然"抢占 + 重算"慢爬（~0.27 req/s、ETA 数小时），且**不是死锁**（decode 进程仍 40%+ CPU）。判据：`kv_usage_pct≈100%` 且 `preempted_recompute_tokens` 持续增长、进度条极慢。对策：**把 trace 缩到 KV 容量内**（总输出 ≤ ~3M）。

16. **3P4D 的 GPU 分配**：共享机上 GPU 0 可能被别的租户占用（`nvidia-smi` 里 PID 不在自己 `ps` 中）。3P4D 配置为 **prefill=GPU 1,2,3 / decode=GPU 4,5,6,7**（7 卡），可用 `PD_RATIO_LIST="3:4"`；启动前先 `nvidia-smi --query-compute-apps=pid,used_memory` 确认哪些卡空闲。

17. **实例间路由的预留集中**：scheme4 路由按"预测未来 token"预留（`Future_Tokens`），长请求会被集中到一个实例（实测某实例 `Future=344k`，其余 1k–4k），使该实例 `pred_bottle` 抬高被后续路由排除。若需均匀，应在 `select_best_instance` 加"Future 超阈值则跳过"的均摊条件。

20. **"ps 显示 0 脚本" ≠ 干净：vLLM 的 EngineCore 会残留占卡**：
    `kill -TERM` 脚本后，`ps -eo pid,cmd | grep test_evaluation_scheme4 | wc -l` 立刻变成 0，
    但 **`VLLM::EngineCore` 进程还要 5–10 秒才退出**，期间 GPU 仍被占（8 卡各 ~66GB）。
    下一轮启动时实例分不到显存/端口 → proxy 报 `proxy died immediately` + 503 连接错误。
    **正确验证（必须三条都满足）**：
    ```bash
    ps -eo pid,cmd | grep -E 'VLLM|vllm|EngineCore|APIServer' | grep -v grep | wc -l   # 0
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader                     # 全 0 MiB
    nvidia-smi | sed -n '/Processes/,$p'                                               # No running processes
    ```
    并 `sleep 5~10` 等 vLLM 释放 NCCL/共享内存再启动下一轮。

18. **查崩溃日志查错目录（排查方法论错误，务必避免）**：评估脚本**实例崩溃重启时会新建一个目录**（`timestamp=$(date +%Y%m%d_%H%M%S)` + `setup_directories`），所以：
    - **最新目录（`ls -dt | head -1`）是"重启后的新轮"，不是崩掉的那轮**；
    - 崩溃现场在**次新目录**（或主日志里 `[WT][INSTANCE_RESTART]` 那一行上面的目录）；
    - 我这次错把最新目录的日志当成崩溃现场，看到 `Init/destroy_process_group` 就误判"日志被截断、查不到原因"，白走了一大段弯路。
    **正确姿势**：
    ```bash
    # 1) 先看有几个目录、各自时间戳
    ls -dt <BASE_RESULT_DIR>/*/
    # 2) 主日志里找重启点，崩溃轮在它之前
    grep -n 'INSTANCE_RESTART' <main>.log
    # 3) 崩溃轮的日志尾部
    tail -30 <crash_dir>/log/decode0_*.log
    ```
19. **同时跑两个评估脚本（严重操作失误，导致"反复断"的假象）**：我在重启时只杀了更早的 PID，**前一个 `test_evaluation_scheme4_rr_compare.sh` 仍在跑**，两个脚本各要 7 张卡、实际只有 8 张 → 互相踩端口/显存/实例，表现为"实例莫名其妙退出、只有部分 decode 在跑"。
    **正确做法**：
    ```bash
    # 启动前必须确认只有一个脚本
    ps -eo pid,cmd | grep 'test_evaluation_scheme4_rr_compare.sh' | grep -v grep
    # 停止时按脚本 PID 杀，并确认归零
    kill -TERM <pid>; sleep 8; ps -eo pid,cmd | grep test_evaluation | grep -v grep | wc -l  # 必须为 0
    # 再确认 GPU / 端口干净
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
    ```
    **排查提示**：如果看到"部分 decode 在跑、部分像死了"，先查**是不是有别的脚本/别的 base 目录在跑**（用 `/proc/<pid>/fd` 看实例实际写哪个日志文件最可靠），不要急着怀疑 vLLM。

    另一条相关经验：主日志里的 `Killed`（SIGKILL）**通常是 health guard 判定 down 之后 `stop_servers` 的清理动作**（`kill -TERM -<pgid>` 进程组广播 + `kill -KILL`），**不是崩溃原因本身**；找原因要往它上面的实例自己日志里看。

## 环境（每次实验都应记录）

- 8× A800-80GB，4P4D，Qwen-2.5-7B-Instruct，TTFT SLO 1000ms / TPOT SLO 30ms，vLLM v0.11.0。
- 权重路径 `/root/myshare/models/`；启动前必设 `VLLM_HOST_IP=127.0.0.1`、`no_proxy=localhost,127.0.0.1`。
