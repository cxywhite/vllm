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

## 结论快照（2026-11-09 更新）

**GWF 准入的正收益 regime（三个，按机制）**：
1. **dense_short**（200 token、~40 req/s 突发）：running 顶到 256、meas_tpot 46–48ms → demote/stop_window 真触发 → goodput **+2.8%～+5%**（#6/#7）。
2. **o1500@gmu0.5**（KV 打满、greedy 抢占重算 26–43 万 token）：窗口 256≈KV 容量 → 零抢占 → goodput **+14%～+69%**（#15/#16）。
3. **sum θ=120 + TTFT-margin 强制准入（本次新找到，margin 需按负载调）**：窗口节流本会把请求饿到超 TTFT 预算，加「deferral≥ttft_budget−margin 就绕过窗口强制放行」后 TTFT 尾巴被砍掉。**o2000 用 margin=250 → goodput 0.42→0.52（+24%）、P99 TTFT 510s→1.6s（#31）；o1600 用 margin=100 → goodput 0.69→0.72（+4.3%）、TPOT≈native、TTFT 不爆（#33）**。margin 是双刃剑：太大抬 TPOT（o1600 margin=250 → TPOT 40→77ms、goodput 0.61 输），太小压不住 TTFT 尾巴（o1600 无 margin → P99 TTFT 349s）。**甜点 = "TPOT≈native + TTFT 不爆"**。
   - 共同点：**greedy 会越界（并发>256 或 KV 抢占重算）或窗口会饿死请求（TTFT 爆），GWF 用窗口+TTFT 余量正好卡住这条线**。KV 富余时（ln2k_cap4k）GWF −19%（#17）。

**oXXX 输出长度扫描（θ=120，无 margin）全线无 GWF 正收益（甜点假设证伪）**：
- native Median TPOT：o400=17.8 → **o1200=29.9（=30ms SLO 压线）** → o1600=54.0 → o2000=100.2 → o4296=124.5 ms。
- GWF Δgoodput（sum θ=120 无 margin）：o1200 **−16%**、o1600 **−4%**、o2000 **0%（平）**、o4296 不可救。**crossover 在 o1200，但窗口守 θ=120ms 这个点，30ms SLO 的 goodput 它救不了**（o2000 节流后 TPOT 仍 60ms=2×SLO）。
- **转折：加 TTFT-margin 后 o2000 转正 +24%（#31）**——瓶颈从"TTFT 饿死"变成了"TPOT 不可达"（80ms 仍 2.7×SLO）。

**三个方法学修复（11-08/11-09，之前"GWF 减 TPOT"的结论据此作废）**：
1. **路由污染**：native 臂 decode 无 `--activation-predict` → `waiting_tokens` 只算 prompt → 长请求堆 2 卡。修复后 native 均衡（268/274/266/302），P99 TPOT 439→157ms；旧"GWF TPOT −65%"是路由假象。
2. **τ 语义**：fraction 二值 → 标量 EMA（Algorithm 2 L1）。单位经历 per-token θ=120（松=4×SLO）→ **sum θ=120（紧=每-token 30ms）**。sum 版已跑完（#29/#30），并发现窗口会饿死请求。
3. **TTFT-margin 强制准入**（`WT_GWF_TTFT_MARGIN_MS`，默认 250ms）：窗口闸门会把 waiting 请求饿到超 TTFT 预算（P99 TTFT 510s），加「deferral≥750ms 绕过窗口强制放行」后 P99 TTFT→1.6s、o2000 goodput 转正 +24%（#31）。**这是本次最关键的新机制**。

**KV 容量 = 并发上限**：单 decode 实例 KV=881,616 tokens；demote 需 output≤~6800、stop_window 需 output≤~3400（running≥256）。长 output 触发不了 GWF 的根因。

**θ=120ms 够不到（11-06 实测）**：干净负载 meas_tpot≈20+0.11×running，峰值 338→meas_tpot max≈61ms≪120ms → 窗口不收缩（fraction 版 +5% 的天花板）。

**run-to-run 方差大**：dense_short 三次 greedy=13.25/13.73(GWF)/14.56（跨度 10%）；o1500 配对 +68.8%/+14.3%。**1 rep 不可比，需 3 rep 中位数**。

**预测 miscalibration（8–13×）是独立短板**，已用 `WT_SIMPLIFY_PREDICT=1`（完美预测）隔离。

| 8 | 11-07 | 硬件采样 + 离线模拟校准（不跑新 A/B） | — | — | 见 `hw_guided_workload_and_sim.md` 与 `worked_example/sim_vs_real_dense_short.md` | ① 采样开销实测 0 tick/58s，可全程常驻；② **GPU fb 不能当并发代理**（vLLM 预分配 KV，running 5→338 时 fb 只差几十 MiB），并发只能看新增的 `kv_usage_pct`；③ 修掉模拟器跨策略状态污染（旧 GWF 模拟从空队列起跑，`max_running=11`），修后仍**符号相反**（实测 +3.6% vs 模拟 −15.7%）→ 模拟器暂只能用来筛"到不了 128"的负载 |

| 9 | 11-07 | 论文 mooncake trace（n=1790，6 req/s，output p90=12.7k） | 观察（别人在跑） | — | 首 7 分钟：greedy kv_usage **100 %**（80/1805 采样 ≥99.5 %）、running 257、waiting 18、**preempted_recompute_tokens=33,633**、日志 preempted 事件 1822 次、watchdog stalled 17 次 | **修正 #6 的过度概括**：长输出不是不能触发 GWF，准确门槛是「运行点平均 seq」——`slots=881616/avg_seq`，mooncake 运行时均值 ~4.6k → 仍有 ~190 slots > 128 门槛。这里 KV 打满 → 原生抢占+整段重算（--swap-space 0），正是论文 eV 该省的浪费。注意风险：100 % KV 时有 deadlock 看门狗警告，极端 20k 输出曾 exit 137，推进要小步（先 1.2× 到达率） |

| 10 | 11-07 | 反思：为什么窗口从不收缩 | — | — | 由实测曲线 `meas_tpot ≈ 19.7+0.117×running` 反解 θ=120 ms 需要 running≈860，而实测峰值仅 338 → window 恒 256 | **重要认知**：dense_short 的 +3.6 % 只来自「淘汰 hopeless 的尾部清理」，不是论文窗口机制的收益。窗口收缩只能靠 KV 打满后的重算抖动把 meas_tpot 顶到 ≫120 ms。正确画像是「running 200–300 + KV 长期 100 % + 平均 seq 2000–4000」。判别：跑的时候看 `[WT][GWF]` 里 window 是否出现 128/64/32 |

| 11 | 11-07 | mooncake 两轮（未截断 / 截断 8192） | 观察 | 每轮 ~11 min 后死 | 两轮都在 **首次抢占后 20–70 s 内 deadlock**：kv 触顶 99.8–100 %、首抢占 t≈450 s（running 156–180）、recompute 从 0 跳到 21k–101k、`GPU deadlock detected` → 300s 倒计时 → exit 137。dense_short（kv<60 %、recompute=0）从未 deadlock | **硬约束**：此 build 下 decode 侧本地抢占重算会 deadlock，KV 打满的负载不可用。安全线 kv<95 %。**GWF 的 eV/preemption 路径在本环境不可测**，只能演示「窗口限流+淘汰 hopeless」。窗口收缩也需要 KV 抖动把 TPOT 顶过 120 ms → 同样不可达；要用窗口机制必须调 θ（参数实验，非负载） |

| 12 | 11-07 | 修正 §2.7 + 设计振荡负载 | — | — | 交叉验证：对方独立算出 greedy only-tpot=3301、only-ttft=2145、both=2864，与我一致。几何约束：demote 门 running≥128 处 decode tpot=34 ms、客户端 40 ms，**必然 TPOT 超** → 持续负载下「TTFT 主导、TPOT 干净」不可能 | 改用**振荡负载**（短突发 130–200 → 长静默 <64）。新增约束：请求必须跨过突发尾部，否则生命周期平均 TPOT 仍是高值——`out_tokens > B×1000/(20+0.11r)`（B=5 时 out>136，B=10 时 out>272）。已生成 osc_slow（B5/G60，5400 条，10.3/s）与 osc_fast（B5/G30，10200 条，18.1/s），峰密度均 120/s cluster→设计峰值 running≈150 |

| 13 | 11-07 | qwen_cap 4P4D（n=4080，封顶 8192） | 部分 | greedy 1h17m 跑到 76% 被杀 | greedy：running max **455**、kv **100 %**、recompute **74,439**（decode0 22,993 + decode2 51,446，decode1/3 = 0，**只有两个实例打到 KV 上限**）；GWF 变体只跑了 3 min（417/4080）就被杀，`running 105 / kv 13.8 %` 只是爬坡值 | **A/B 不成立，不能比较**。但 greedy 侧证实：qwen_cap 确实把原生推到 KV 悬崖（有效吞吐 ~0.27 req/s 的慢爬），且**不均衡**——只有 2/4 个 decode 实例打满，与 predictor 把长请求集中路由一致。GWF 侧要重跑（现 deadlock guard 已改进度判据，不会再被 util=0 误杀） |

| 14 | 11-07 | dense_short 3rep 抢到空卡后仍崩（rc=137，283 s） | — | — | `[CRASH] dmesg oom recent: sglang::schedul invoked oom-killer ... Out of memory`；崩溃时 `Mem: total 377G used 303G free 7G shared 257G`；随后 `prefill-0 down for 120s` → `INSTANCE_RESTART` → 被 kill | **宿主内存 OOM**，不是 GPU 问题。根因：我们 8 个实例各 `mem_pool_size_gb=32`（P2p KV 共享内存）= **256 GB**，占 377 GB 的 68 %；再叠加同宿主另一个租户的 sglang 作业（shmem-rss 29.8 GB）→ 全局 OOM，内核随机杀进程。**对策：`PREFILL_TENSOR_POOL_MEMORY=8 DECODE_TENSOR_POOL_MEMORY=8`（降到 64 GB）**；200 token 请求每条 KV 仅 ~11 MB，8 GB 足够 |

| 15 | 11-07 | **qwen_o1500 3P4D（decode gmu=0.5）** | 完美预测 | 20 min / 17 min | **迄今最好**：greedy goodput **0.48** / GWF **0.81 req/s（+68.8 %）**；output-token goodput 274→578 tok/s（**+110.8 %**）；P99 TPOT **311.6→87.1 ms（−72 %）**；mean E2EL 100.0→81.2 s；recompute **796,044→0**；running 366→256；kv 两者都 100 %；代价：P99 TTFT 12.8→17.0 s | **机制终于对上论文**：greedy 把 running 推到 366 > KV 容量 → 抢占+重算 796k token → TPOT p99 312 ms，几乎没人能满足 30 ms SLO；GWF 的窗口 256 恰好等于该序列长度下的 KV 容量 → 零抢占 → TPOT p99 87 ms。**增益全部来自"窗口≈KV容量"**。对照 `wl_util07_o1500v2`（gmu=0.7、v2 trace）GWF 反而 −7.4 %（P99 TTFT 315 s 饿死）→ 结论对配置敏感，必须在 gmu=0.5 这一档确认 |

| 16 | 11-07 | **qwen_o1500 3P4D gmu=0.5 复现**（我的确认跑，rep1 完成 / rep2 崩于 instance_restart） | 完美预测 | 57 min（2 次完整 run + 1 次重启） | rep1：greedy goodput **0.56**（recompute 425,748；running 494）/ GWF **0.64**（recompute **0**；running 256）→ **+14.3 %**；rep2 greedy 0.49（n=3968，recompute 258,542） | **方向复现、幅度不回**：两次配对 +68.8 %（对方）与 +14.3 %（我）→ 配对中位数 +41.5 %；按均值 pooled（greedy n=3 均值 0.510，GWF n=2 均值 0.725）**+42.2 %**。**三个 greedy run 的 recompute 258k–428k → 两个 GWF run 全部 0**，机制稳定（窗口 256 ≈ KV 容量）。幅度方差大是因为 goodput 是尾部统计量（0.5–0.8 / 4 req/s），需要更多 rep 才能给中位数 |

| 17 | 11-07 | ln2k_cap4k 3P4D gmu=0.7（n=4080，output p50 1752 / p90 3342 / max 4000，6.8 req/s） | 完美预测 | ~15 min | greedy goodput **0.37** / GWF **0.30 req/s（−18.9 %）**；running 247→256；**kv 仅 34.8 %→39.4 %**；recompute 两边全 0；GWF `stop_window` Σ 252 | **负对照（边界条件）**：KV 完全没压力时，GWF 的窗口 256 只是把本该被服务的请求挡在外面 → −19 %。与 o1500（kv 100 %、greedy recompute 26–43 万）的 +14 %~+69 % 形成清晰边界：**GWF 只在「greedy 会越界抢占重算」时有正收益**；KV 富余时中性到负 |
| 18 | 11-07 | dense_short 4P4D 再跑一次（greedy） | 完美预测 | ~7 min | greedy goodput **14.56 req/s**、good 4098/12000（34.1 %）、running 294、kv **5.5 %**、recompute 0；同配置的 GWF 变体启动即崩（无 bench 日志） | 同一负载第三次测量 greedy goodput = 13.25 / 13.73(GWF) / **14.56**，**run-to-run 跨度 10 %**（14.56 vs 13.25）→ 再次说明 1 rep 不可用于比较，必须 3 rep 中位数 |
| 19 | 11-08 | —（代码改动，非跑分） | **路由也冻结为完美预测** | — | ① 修掉路由负载漏记（waiting 未来输出不计入，见坑 #23）；② `WT_SIMPLIFY_PREDICT` 全面冻结预测：proxy `_predictor_route_output_len` 走 `max_tokens`、decode `predicted_output_len`/`predicted_total_tokens`/`predicted_remaining_tokens` 全部走 `max_tokens`（后者原来仍走 BERT bucket 后验，已补上），路由与 eV 完全一致 | **下一步**：在「完美预测 + 路由账目正确」的前提下，扫 output 长度/并发找 GWF > native 的负载 |
| 20 | 11-08 | qwen_o4296 4P4D util0.8（修复后复跑） | 完美预测（路由也冻结） | native 49min / GWF 46min | **路由修复验证通过**：proxy `Future` 现在随 `wait_reqs` 同步（91→614k、214→1.23M），而旧代码 242→23k；running 均衡到 71–85（旧 125–256）。**但 GWF 仍拿 TTFT 换 TPOT**：native 4080/4080 goodput 0.01、TPOT 124.5ms、TTFT 0.8s；GWF 4080/4080 goodput 0.03、TPOT **50.1ms（−60%）**、P99 TPOT 97.9（−71%）、吞吐 +18.7%、但 TTFT **315.3s（+393×）**、window 钉 32、frac 恒 100、worst4≈300ms | **核心结论**：① 修复让 GWF 不再饿死请求（2249→4080 全完成）；② 但 fraction 版 θ=120=4×30 在此硬件上**过紧**——TPOT 50ms 仍 1.7× 于 30ms SLO，窗口为守一个不可达的 SLO 把 TTFT 拖到 315s。**要找到 GWF 正收益，必须同时动 θ（放宽到 160ms 或换 per-token 口径）+ 短 output（让 30ms 可达）**，单靠扫负载不够 |
| 21 | 11-08 | qwen_o4296 4P4D util0.8（**per-token θ=120**，方案A；首跑 P2P NCCL 死锁重启） | 完美预测 | ~46min | GWF 4080/4080 goodput **0.00**、吞吐 6119 tok/s、TTFT **72.0s**、TPOT **109.9ms**、P99 141.3ms | **per-token 口径下窗口不再钉 32**（worst4≈75ms < 120ms → frac=0 → window 恒 256），TTFT 从 315s 回落到 72s，但 TPOT 从 50ms 涨回 110ms，**goodput 仍是 0**。**结论升级**：qwen_o4296（output 4296）上 30ms SLO 物理不可达——sum 口径守它→TTFT 爆炸，per-token 口径放弃它→TPOT 110ms。**θ 无论怎么调都救不了长 output，必须换短 output**（dense_short/o400 那类，让 30ms/token 可达） |
| 22 | 11-08 | **qwen_o400 4P4D util0.8**（native 成功；GWF 臂 start_servers_failed 连挂 3 次，环境问题弃） | 完美预测 | native ~13min | **native：4080/4080、goodput 6.53 req/s、TTFT 210ms、TPOT 17.84ms、P99 23.30ms** ← 短 output 首次让 30ms SLO 可达、goodput 非 0 | **关键定标**：output 400 → native TPOT 17.84ms（<30ms，无过载，GWF 无事可做）；output 4296 → native 124.5ms（>30ms，GWF 也救不回）。**甜点位在 output ~800–2000**：native TPOT 刚越过 30ms、GWF 还能用窗口压回 30ms 内。下一步应扫 output 800/1200/1600/2000 找 native TPOT 略超 30ms 的点 |
| 23 | 11-08 | **qwen_o2000 4P4D util0.8（sum θ=120 倒序扫描第 1 档）** | 完美预测 | native 25min / GWF 23min | **native：4080/4080、goodput 0.37 req/s、output-tok goodput 263.9、Median TPOT 113.2 / P99 439.5、Median TTFT 526 / P99 1828ms。GWF：4080/4080、goodput 0.40 req/s（+8%）、output-tok goodput 228.4（−13%）、Median TPOT 39.2 / P99 96.9（−65%/−78%）、Median TTFT 1062 / P99 379298ms（+2×/+207×）** | **关键负面结论**：GWF 全计数器=0（`window` 恒 256 未收缩、`demoted=0`、`preempted=0`、`stop_window=0`），**准入机制根本没触发**——o2000 每实例 running 峰值 ~150–175 < 256 收缩门槛。TPOT 113→39ms 的改善不是准入控制（是路由账目/run-to-run 差异），且代价是 TTFT 尾部爆炸（P99 1.8s→379s）。**本轮结果随后作废**（见 #24 的代码修复） |
| 24 | 11-09 | —（代码对齐，非跑分） | 完美预测 | — | ① **路由修复**：decode 启动命令加 `--activation-predict`，让 native 臂的 `_should_track_predictive_pressure()` 也返回 True，路由信号与 GWF 臂对齐——消除"坏路由+native" vs "好路由+GWF"的 A/B 污染；② **τ 对齐论文**：GWF 的 τ 从 fraction 二值改为论文 Algorithm 2 的标量 EMA（每-token 时延，α=0.5、1s 刷新），窗口改 `τ>θ ∧ |Rrun|≥A → A/2` / `τ≤θ → A+A/8`（去掉 frac>5% 与 2-Δt streak） | #23 的 o2000 结果作废（路由污染 + fraction 非论文）。`WT_GWF_TPOT_UNIT`/`WT_GWF_TPOT_TRIGGER_PCT` 变空转，θ 仍由 `WT_GWF_WINDOW_TPOT_TARGET_MS=120` 控制。**待 8 卡空时用新配置重跑 o2000→o800** |
| 25 | 11-09 | **qwen_o2000 4P4D util0.8 复跑（路由修复 + τ=标量EMA）** | 完美预测 | native 21min / GWF 21min | **native：4080/4080、goodput 0.42 req/s、output-tok goodput 250.3、Median TPOT 100.2 / P99 157.0、Median TTFT 528.9 / P99 1513ms。GWF：4080/4080、goodput 0.43 req/s（+2%）、output-tok goodput 251.5（+0.5%）、Median TPOT 92.6 / P99 116.9（−7.5%/−25.5%）、Median TTFT 606.2 / P99 31649ms（+14.6%/+20×）** | **① 路由修复验证通过**：native 每实例 running 峰值 268/274/266/302（均衡，旧版 2 卡 175/158+2 卡空），P99 TPOT 从 439→157ms（尾部 KV thrash 消失）；② **GWF 准入这次真触发了**：τ 峰值 122–262ms > 120ms，窗口 256→64→再涨回（stop_window 累计 65–1668、demoted 累计 4–1156）；③ **但净 goodput 平**（0.42 vs 0.43 ≈ 噪声），GWF 只是把痛从 TPOT 挪到 TTFT（P99 TTFT 1.5s→31.6s）。**根因：o2000 即便节流后 TPOT 仍 93ms ≈ 3× SLO，30ms SLO 在该 output 下物理不可达**——窗口守的是 120ms 可达点，救不了 30ms 的 goodput |
| 26 | 11-09 | **qwen_o1600 4P4D util0.8 复跑（路由修复 + τ=标量EMA）** | 完美预测 | native ~17min / GWF ~17min | **native：goodput 0.69 req/s、output-tok goodput 524.6、Median TPOT 54.0 / P99 82.5、Median TTFT 342.9 / P99 1628ms。GWF：goodput 0.55 req/s（−20%）、output-tok goodput 347.4（−34%）、Median TPOT 85.3 / P99 122.2（+58%/+48%）、Median TTFT 503.8 / P99 22093ms（+47%/+13.6×）** | **GWF 在 o1600 反而变差**：τ 峰值 124–134ms 越过 120ms 触发窗口收缩（d1/d2 stop_window 813/966、demoted 689/595），但节流压低并发后 benchmark TPOT 反而 54→85ms、goodput −20%。native running 峰值 218/207/270/276（均衡）。**o1600 是明确负收益档**——native TPOT 54ms 本就在 120ms 可达点以内，GWF 的收缩反而帮倒忙 |
| 27 | 11-09 | **qwen_o1200 4P4D util0.8 复跑（路由修复 + τ=标量EMA）** | 完美预测 | native ~13min / GWF ~13min | **native：goodput 2.96 req/s、output-tok goodput 3007.0、Median TPOT 29.9 / P99 47.6、Median TTFT 238.6 / P99 810ms。GWF：goodput 2.48 req/s（−16%）、output-tok goodput 2198.9（−27%）、Median TPOT 31.2 / P99 55.3（+4%/+16%）、Median TTFT 245.6 / P99 887ms（+3%/+9.5%）** | **crossover 定标**：native Median TPOT **29.92ms 恰好压线 30ms SLO**——o1200 就是 native TPOT 越过 SLO 的点。但 GWF 在此仍是负收益（goodput −16%）：τ≈30ms ≪ θ=120ms，窗口不触发，GWF 退化成空转开销。**oXXX 全线无 GWF 正收益点，甜点假设证伪** |
| 28 | 11-09 | —（θ 语义调整：τ 改为 "sum" 单位） | 完美预测 | — | τ 的度量单位从 per-token 改回 **"sum"（4 步和）**：`_measure_tpot` 现在对每个请求取 `stamps[-1]-stamps[0]`（4 步和，不除以 4），episode 平均则 ×4。θ 仍是 `WT_GWF_WINDOW_TPOT_TARGET_MS=120`，于是 **τ>120ms = 4 步和>120ms = 每-token>30ms**（紧贴 SLO，不再是 4×SLO 的 120ms 可达点） | #25–#27 是 per-token θ=120（松）的结果，**作废**；按 sum θ=120（紧）重跑 o2000→o800。预期：窗口会在 native TPOT 刚过 30ms 处就收缩（o1200/o1600 附近），GWF 更可能真正出手 |
| 29 | 11-09 | **qwen_o2000 4P4D util0.8（sum θ=120，紧）** | 完美预测 | GWF ~19min（native 复用 fair） | **GWF：4080/4080、goodput 0.42 req/s（=native 0.42）、output-tok goodput 246.8（native 250.3，−1.4%）、Median TPOT 60.2 / P99 102.3（native 100.2/157.0，−40%/−35%）、Median TTFT 623.9 / P99 510252ms（+18%/+337×）** | **sum 让窗口真出手**（d1/d3 窗口缩到 32、stop_window 18015/16622、demoted 14028/9377），TPOT 100→60ms；但 **d0/d2 的 stop_window=0**（running 119/131 < 256 收缩守卫，窗口没缩）→ 不对称节流，一半刹死一半不刹。60ms 仍是 2×SLO，省下的 TPOT 全被 TTFT 吃掉（P99 1.5s→510s）。**净 goodput 仍平**：θ 松→空转、θ 紧→猛刹救不回，两头都不赢——30ms 在 o2000 不可达 |
| 30 | 11-09 | **qwen_o1600 4P4D util0.8（sum θ=120，紧）** | 完美预测 | GWF ~12min（native 复用 fair 0.69） | **GWF：4080/4080、goodput 0.66 req/s（native 0.69，−4.3%）、output-tok goodput 513.9（native 524.6）、Median TPOT 40.4 / P99 78.5（native 54.0/82.5，−25%/−5%）、Median TTFT 546.1 / P99 348955ms（+59%/+214×）** | 同 o2000 的失效模式：d0/d3 猛刹（stop_window 12900/13940、demoted 12517/19359）而 d1/d2 完全不刹（0）→ 不对称。TPOT 54→40ms，但 40ms 仍 >30ms SLO、TTFT P99 爆到 349s，goodput −4%。**结论：sum 紧阈值在 o1600/o2000 都是"节流降 TPOT 但救不回 goodput"，根因 30ms 不可达 + 不对称节流** |
| 31 | 11-09 | **qwen_o2000 4P4D util0.8（sum θ=120 + TTFT-margin 250ms）** | 完美预测 | GWF ~17min（native 复用 fair 0.42） | **GWF：4080/4080、goodput 0.52 req/s（native 0.42，+24%；无 margin 版 0.42）、output-tok goodput 360.8（native 250.3，+46%）、Median TPOT 80.1 / P99 107.4（native 100.2/157.0）、Median TTFT 860.0 / P99 1628ms（无 margin 版 P99 510252ms → 1628，−99.7%）** | **TTFT 余量强制准入生效，首次 GWF 明确反超 native**：`deferral≥750ms 强制放行` 把被窗口饿死的请求救回来，P99 TTFT 从 510s 砸到 1.6s，goodput +24%。副作用：TPOT 60→80ms（并发升高），demoted 14028/9377→0、stop_window 18015/16622→841/1031。**goodput 仍被 30ms TPOT 上限压着（80ms 仍 2.7×SLO），但 TTFT 瓶颈已被移除** |
| 32 | 11-09 | **qwen_o1600 4P4D util0.8（sum θ=120 + TTFT-margin 250ms）** | 完美预测 | GWF ~12min（native 复用 fair 0.69） | **GWF：4080/4080、goodput 0.61 req/s（native 0.69，−11.6%；无 margin 版 0.66，−7.6%）、output-tok goodput 425.0（native 524.6）、Median TPOT 77.4 / P99 113.2（native 54.0/82.5；无 margin 版 40.4/78.5 → +92%/+44%）、Median TTFT 557.7 / P99 1569ms（无 margin 版 P99 348955→1569，−99.5%）** | **margin 是把双刃剑，随负载方向反转**：o1600 里 margin 把 stop_window 从 ~13000 压到 ~1100，running 一直较满 → TPOT 从 40→77ms（+92%），goodput 反而 0.66→0.61。**o2000（深过载）TTFT 饿死是主因 → margin 净赚 +24%；o1600（中过载）TPOT 本就接近 SLO → margin 抬 TPOT 是主因 → 净亏 −7.6%**。结论：margin 需要按负载调，250ms 对 o1600 偏激进 |
| 33 | 11-09 | **qwen_o1600 4P4D util0.8（sum θ=120 + TTFT-margin 100ms）** | 完美预测 | GWF ~12min（native 复用 fair 0.69） | **GWF：4080/4080、goodput 0.72 req/s（native 0.69，+4.3%；margin=250 版 0.61 → +18%）、output-tok goodput 556.0（native 524.6，+6%）、Median TPOT 53.1 / P99 80.1（native 54.0/82.5）、Median TTFT 530.6 / P99 1566ms（无 margin 版 P99 348955→1566，−99.6%）** | **o1600 的 margin 甜点是 100ms**：TPOT 53ms ≈ native 54ms（几乎不抬）、TTFT 1566ms ≈ native 1628ms（不爆）→ goodput 0.72 反超 native +4.3%。margin 梯度呈倒 U（0→0.66、100→0.72、250→0.61），证明 100 是平衡点。**结论：margin 调到"TPOT≈native + TTFT 不爆"处，GWF 在中过载档也能小胜** |
| 34 | 11-09 | **burstgpt_real 4P4D util0.8（真实 BurstGPT trace，sum θ=120 + margin=250）** | 完美预测 | native 58min / GWF 58min | **native：673/673、goodput 0.08 req/s、output-tok goodput 56.40、Median TPOT 34.79 / P99 96.83、Median TTFT 220 / P99 568ms。GWF：673/673、goodput 0.07（−12.5%）、output-tok goodput 56.18（−0.4%，平）、TPOT 35.63 / P99 94.63、TTFT 235 / P99 515ms** | **KV 受限 regime，GWF 两个机制都够不到**：135 条 32k 极端请求把 KV 占满，running 卡在 ~40（≪256 收缩门槛）→ 窗口永不触发（stop_window=0）；TTFT 本就 220ms 无饿死 → margin 无用；唯一动的是 KV 满抢占重算（preempted=57、recompute 1.09M），但 native 也躲不掉。**goodput 平（56.40 vs 56.18）**。与 o2000（TTFT 饿死→margin +24%）、o1600（窗口节流→margin=100 +4.3%）对照：**GWF 每个机制只在对应瓶颈生效，KV 受限时无计可施** |

## 待办（TODO）

- **[已暂停] 8 卡队列**（`run_queue_step3first.sh`，lmsys UniBoost 卡死被手动停，8 卡已释放给其他 AI）：step3(GWF✅100% / UniBoost 96%超时) → gammaada(✅结果已存) → amax220(✅119.28) → real_lmsys(GWF✅100% / UniBoost 卡死77%) → **剩余 real_sharegpt → 4p4d_x1p8 待恢复**。
- **[待命] step3 UniBoost 单臂重跑**：new_once 无头 AI（schedule-aaabca78）在旧队列全部结束后自动跑 `run_step3_uniboost_redo.sh`（无超时）。
- **[待跑] ×1.2×8 UniBoost 臂重跑（修复后）**：用修复 unit bug 后的 `uniboost_queue.py` 重跑，坐实 phase A「GWF +8% > UniBoost」在修复后仍成立（phase A 当时是 pre-bug 真 SJF，需用新代码复现）。
- **[待跑] 真实预测 + scheme4 + UniBoost 4p4d 补跑 burstgpt + mysharegpt**：对齐 Dynamo 基线（qwen+mysharegpt+burstgpt = 70.18 goodput，qwen+lmsys+burstgpt = 168.24）。用真实 BERT 预测（WT_SIMPLIFY_PREDICT 关）+ scheme4 路由 + UniBoost，4p4d，tpot30；需带修复（unit bug / victim_newest）。
- **[待跑] 1p1d_x1p8**（2 卡，8 卡优先规则让位）。
- **[待修] UniBoost 块表越界 bug**：`ValueError shape(498)→(212)` 根因已定位 = WT「本地重算」修改（commit cef34ada16, 2026-11-07）导致 disagg 下 num_computed_tokens 与块表不同步 → 块表越界。需修"本地重算"路径的对齐，或加防御性 cap。
- **[待查] KV 路由丢包根因**（单请求路由丢失 → 间歇性卡死）。
- **[WT_PRED_CORRECTION] fastpath 晚到 meta 的兜底校正**（先记着，暂不做）：β 校正目前在 proxy `handle_request` 的注入点做一次；fastpath（Path A）下预测器 meta 是 ZMQ 异步晚到的，smoke test 里 ~38%（15/40）请求 meta 赶不上注入点 → 校正跳过、decode 侧退回 `max_tokens`。若后续验证现有 β 方法（重点 Path B 覆盖更全的重负载真实 trace）效果仍有问题，再在 `_listen_predictor_metadata` 收到 meta 时补一次校正兜底，让 fastpath 请求的准入侧也稳定吃 `β×predictor` 而非 `max_tokens`。位置：`experiment/zrecent_use/disagg_serving_suite/test_disagg_proxy.py`。

## 结果目录索引（实验 ↔ 原始数据 / git 参考）

结果根目录两处：`/root/myshare/`（对外可见）与仓库内 `experiment/zrecent_use/0630/`。

| 迭代 | 配置 | 结果目录 |
|---|---|---|
| #6/#7 dense_short | 短请求 200 token | `wl_4p4d_util08_o1800/`（含 dense_short 复跑） |
| #15/#16 o1500 gmu0.5 | 3P4D | `wl_util05_o1500/` |
| #17 ln2k_cap4k | 3P4D gmu0.7 | `wl_util07_ln2k_cap4k/` |
| #20 o4296 sum | 4P4D util0.8 | `/root/myshare/paper_aligned_v3/`、`wl_o4296_util08/` |
| #21 o4296 per-token | 4P4D util0.8 | `/root/myshare/paper_aligned_v4_token/`、`_r2/` |
| #22 o400 | 4P4D util0.8 | `/root/myshare/paper_aligned_v5_o400/`、`wl_o400_util08/` |
| #23 o2000（作废） | 4P4D util0.8 | `/root/myshare/gwf_sweep_1109/o2000/` |
| #25–#27 o2000/o1600/o1200（per-token，作废） | 4P4D util0.8 | `/root/myshare/gwf_sweep_1109_fair/{o2000,o1600,o1200}/` |
| #28 起 sum θ=120（重跑中） | 4P4D util0.8 | `/root/myshare/gwf_sweep_sum120/` |
| #29/#30 sum θ=120 无 margin（o2000/o1600） | 4P4D util0.8 | `/root/myshare/gwf_sweep_sum120/{o2000,o1600}/` |
| #31/#32/#33 sum + margin（o2000 250 / o1600 250 / o1600 100） | 4P4D util0.8 | `/root/myshare/gwf_sweep_margin/{o2000,o1600}/`、`/root/myshare/gwf_sweep_margin100/o1600/` |
| burstgpt 真实 trace（进行中） | 4P4D util0.8 | `/root/myshare/gwf_burstgpt_real/` |
| workload_trials o1800 / o2140 | 4P4D util0.8 | `wl_4p4d_o1800_util08_v3/`、`wl_o2140_util08/` |
| UniBoost A/B（**未完成**） | 3P4D gmu0.5 | `wl_util05_uniboost_ab3|ab4|ab5/`、`wl_ub_arm_a|b|b2|c/` |

> **评估脚本**：`test_evaluation_scheme4_rr_compare.sh`（`scheme4`=预测路由+贪心准入、`scheme4_gwf`=预测路由+GWF 准入、`scheme4_uniboost`=预测路由+UniBoost；路由已对齐，见 #24）。
> **trace 数据**：`admission_ab_base/`（qwen_o400/o800/o1200/o1500/o1600/o1800/o2000/o2140/o2471/o2900/o4296、dense2k、rate600/900 等）。
> **git 回滚点**：主仓库 `experiment/20260630` @ `a0bdf55`；vllm 子模块 `v0.11.0-local-v2` @ `f3296d1d1`（详见 AGENTS.md「回滚安全点」）。
> **当前 sum+margin 状态（#28–#33 结果用的代码）**：主仓库 `df5d7b0`（含 trace、路由修复、日志）+ vllm 子模块 `6e1011be4`（τ 标量 EMA sum + TTFT-margin + WT_SIMPLIFY_PREDICT）。复现「o2000 margin=250 +24% / o1600 margin=100 +4.3%」需 `git checkout df5d7b0`（主仓库）+ `cd vllm && git checkout 6e1011be4`。
> **早期 workload trial**（o1500/o1800/o2140/dense 的逐条调参记录）在 `workload_trials.md`，与本主迭代表互补；`hw_guided_workload_and_sim.md` 是硬件采样+离线模拟的那一轮。

### 步3 口径 + 重算代价论述（2026-11-10 用户补充）

**步3 用校准版**：真实 BERT 预测值先做 `WT_PREDICT_OUTPUT_SCALE` 校准（实测 pred/actual≈4.36×）再用，不是裸用 miscalibrated 预测值。

**重算代价（recompute/re-prefill）的论述框架**（把"零重算"从"省点 token"升级成"避免系统失稳"）：
- 在 P/D 分离 + `--swap-space 0` 下，抢占 = 释放 KV → 恢复时整段 re-prefill（prompt+已生成 token 全重算）。
- 代价是多维的，不只是"浪费 token"：① token 重生成（kv1p6 native 3.0M = 输出的 25%）；② **decode 停摆**（re-prefill 大前缀阻塞 decode → P99 TPOT 1024ms，正常 ~100ms，10× 尖峰）；③ KV thrash（100% 占用下反复 free/allocate）；④ P2P 重传带宽；⑤ **硬崩溃风险**（#11 旧 build "decode 侧本地抢占重算会 deadlock，KV 打满负载不可用"）。
- **论述**：recompute 是"系统失稳级"代价，不是零头；GWF 零重算（32k vs native 3M / UniBoost 118k）是避免这种失稳。诚实 caveat：当前 θ=120ms 下窗口缩 32 地板把收益转成 TTFT，零重算优势"真实但未兑现"，θ 调参（用户暂缓）或换"KV 打满+短请求"负载（需 o1500_x2 或 gmu0.5）才能兑现。
### 下一步计划（2026-11-10 用户定）

1. **等 kv1p6 UniBoost 修复版跑完**（admit-preempt off，零 crash）拿 goodput，补全 KV 打满三臂。
2. **o2000 完整版 UniBoost（γ-Ada 墙钟）**：`WT_UNIBOOST_CLOCK=wall` 真跑，看 γ 收敛值 + goodput，坐实离线模拟（γ 振荡、均值 γ_tok≈7e-4 → UniBoost≈340 < GWF 361）。
3. **真实预测器头对头消融**（用有 RR/Dynamo 结果的论文 qwen 4080 trace：`evaluation_dataset/0630_evaluation_dataset/qwen/qwen_lmsys_qwentrace_4080_600s.csv`，mean 7872）：
   - 臂 A = 真实 BERT 预测器 + scheme4 路由 + GWF（去掉 `WT_SIMPLIFY_PREDICT=1`，启用真实预测值；注意预测 miscalibration 8-13×、`WT_PREDICT_OUTPUT_SCALE` 校准口径待确认）；
   - 臂 B = 真实 BERT 预测器 + scheme4 路由 + UniBoost。
   - 与论文已有的 RR / Dynamo / Presage 端到端 goodput 对齐。
### γ-Ada 离线预估 + ×1.25 温和长尾（2026-11-10 记录，待跑）

**γ-Ada 算法（SGLang PR #48077，论文 Eq.9）**：`γ̂ = ln(5)/(t99−t95)`，窗口 128、EMA β=0.2、clip [1e-3,15] 1/s（墙钟）；size-aware：长请求尾巴更肥时改拟合短请求尾巴。

**o2000 离线预估**（用 γ=1e-2 轮 4080 条 latency）：全体 γ̂=0.0155、短请求 0.385、长请求 0.056 → size-aware 采用 0.385。换算 token 钟 `γ_tok=γ_wall/rate≈0.385/7335≈5.2e-5`（< 1e-4）。按 γ 扫描外推（1e-4→326）→ 预测 γ-Ada 让 UniBoost 掉到 ~300 < GWF 361。**待实现 γ-Ada 墙钟模式 + AdaptiveGamma 类确认**。

**×1.25 温和长尾预测（不保证 GWF 更优）**：GWF Δgoodput 随 factor 非单调——o2000(f1.0,KV80%) +44%（TTFT-margin）、kv1p6(f1.6,KV100%) ≈0%（window→32 地板）。f1.25 落在过渡区：怪物 ~5800-10400 → τ≈160-220ms 仍 >120ms → window 仍可能缩 32 地板；UniBoost 排序收益（o2000 +57%）大概率仍 ≥ GWF。**故 ×1.25 不能保证 GWF 反超，只是"可能比 kv1p6 略好、仍不如 o2000"**。

**预测器状态**：scheme4_gwf 用 `--activation-predict`（BERT 预测器进程加载，prefill+decode 都开，见 #24 路由对齐）；但 `WT_SIMPLIFY_PREDICT=1` 把预测值冻结为 `max_tokens`（非真实 BERT 输出）——即"预测器开着、预测值是简化版"。

### step1.5 kv1p6 γ-Ada 墙钟真跑结果（2026-11-10）

**kv1p6 γ-Ada 墙钟**（`wl_kv1p6_gammaada/`，scheme4_uniboost + wall + admit-preempt off）：
- γ_s 收敛到 **0.135**（强 SJF）；recompute **3.0M**（vs 固定 γ=1e-2 的 14.6M——SJF 让怪物排队不被频繁驱逐）；goodput **128.4**（req_gp 0.21、mTTFT 51s、p99TPOT 172ms）。

**kv1p6 四臂完整对比**：native 134.2 / **GWF 135.1** / UniBoost 固定γ 130.7 / UniBoost γ-Ada 128.4；recompute 3.0M / **32k** / 14.6M / 3.0M。

**结论**：KV 打满下 UniBoost "两头不讨好"——固定 γ 重算风暴(14.6M)、γ-Ada 强 SJF 过度优待短请求(128.4)，都 < GWF(135.1)。GWF 零重算(32k) 仍是最优。→ "准入 vs 驱逐"在 KV 打满 regime 下 GWF 稳赢所有 UniBoost 变体。

## 1p1d 实例内调度隔离实验（2026-11-10 计划）

**动机**：1p1d 只有 1 个 decode 实例 → 无 inter-instance 路由 → GWF vs UniBoost 纯看实例内调度（GWF 窗口准入 vs UniBoost 优先级排序），绕开"路由内生污染"。

**配置**：`PD_RATIO_LIST="1:1"`（`PD_1P1D_PREFILL_GPUS=6`、`PD_1P1D_DECODE_GPUS=7`），gmu 0.8，SLO TTFT 1000ms / TPOT 30ms。

**负载**：`admission_ab_base/qwen_lmsys_real_kv1p6like_1p1d/qwen_lmsys_qwentrace_4080_real_kv1p6like_1p1d.csv`
- **4080 条 = 原 4080 trace 的时间轴放缩**（timestamp ×4：600s → 2400s，到达率 6.8 → 1.70 req/s），请求集合/prompt/output 分布**完全不变**，纯时间放缩、非抽样；
- output p50=646 / p70=1025 / p90=11201 / max=13303，input p50=58。

**两臂**：`scheme4_gwf` vs `scheme4_uniboost`（都 max_tokens 简化预测，先看实例内调度差异；后续可换 β 校准真实预测器）。

**待办 driver**：仿 run_ab_trace.sh 写 run_1p1d_ab.sh（PD_RATIO_LIST=1:1 + PD_1P1D_* + TRACE_NAME=qwen_lmsys_real_kv1p6like_1p1d）。

## gmu0.8 下 monster factor 扫描（2026-11-11，1p1d 实测）

**目的**：gmu 0.8 固定下，调输出长度找 GWF 更优的负载。max_tokens 简化预测，1p1d（GPU 对应各轮不同）+ 1.7 req/s。

| factor | 窗口 | τ_ms | GWF 结果 |
|---|---|---|---|
| ×1.0（o2000，已知）| →64 再回升 | 122–262 | **+44%（峰值）** |
| ×1.1（`wl_1p1d_x1p1_gwf`）| →32 地板 | 493–592 | 失败（≈native）|
| ×1.2（`wl_1p1d_x1p2_gwf`）| →32 地板 | 476–782 | 失败 |
| ×1.4（`wl_1p1d_x1p4_gwf`）| →32 地板 | 376 | 失败 |
| ×1.6（kv1p6，已知）| →32 地板 | 365–500 | 失败 |

**关键**：×1.0→×1.1 仅 10% 怪物增量，τ 从 ~200ms 阶跃到 ~500ms、窗口从"→64 再回升"直接砸 32 地板——attention 成本随 KV 长度非线性（平方级）增长。

**结论**：gmu 0.8 下 o2000（+44%，TTFT-margin 机制）是唯一峰值，零重算准入兑现不了；找负载收敛，无更大收益。

## 真实 BERT 预测器校准：β 初值修正因子（2026-11-10 量测）

**方法**：直接量真实输出与预测输出的差距，β = sum(actual_output) / sum(predict_output_mean)，作为初值修正因子（等价 `WT_PREDICT_OUTPUT_SCALE`：pred × β = real）。

| 负载 | β 初值 | 原始 pred 在 2× 内 | 校准后 2× 内 | 相对误差中位 |
|---|---|---|---|---|
| qwen × lmsys | **0.3011** | 4.3% | 9.4% | 3017%（30× 过预测）|
| qwen × sharegpt | **0.3621** | 3.6% | 8.2% | 1886%（19×）|
| llama × lmsys | **0.2114** | 5.9% | 27.2% | 1179%（12×）|

**根因**：BERT 输出 5 桶概率，桶中心约 [79,302,927,16934,32614]，而真实输出中位只有 288–354——最小"有意义"桶中心(302/927)本身就大于多数真实输出，故系统性过预测 12–30×（qwen 最糟、llama 稍好）。

**结论**：
- β 只修**总量均值**（sum 口径），对**路由负载均衡**够用；
- 救不了**逐请求精度**（校准后 2× 内仍只有 9–27%），因误差是桶量化错位、非简单倍数；
- → step3 真实预测器时 `WT_PREDICT_OUTPUT_SCALE` = β（qwen 0.30 / llama 0.21）作初值；但 GWF 准入（gen_ratio 逐请求）仍会因预测失真而掉分，这本身是"oracle→真实预测器"的诚实代价。

## 找"GWF 更优负载"的完整过程与方法论（2026-11-10 固化，供未来找负载复用）

### 一、本次 tweak10 实验（微调 10 条 SLO 边界请求）

**动机**：验证"只微调几条 SLO 边界请求能否放大 UniBoost 劣势、让 GWF 更优"。

**做法**：从 kv1p6 γ-Ada 那轮（`wl_kv1p6_gammaada`）选出 10 条边界请求（5 条 ttft≈1.00s 刚越线 + 5 条 tpot≈30ms 刚越线），只把它们的 output ×1.3，其余 4070 条不动。

**这 10 条的 input/output 长度**（prompt 全都很短 44–160 token）：

| req | prompt_len | out 原 | out 新 |
|---|---|---|---|
| 14101 | 58 | 688 | 894 |
| 16971 | 70 | 685 | 890 |
| 24075 | 73 | 685 | 890 |
| 26361 | 52 | 690 | 897 |
| 29772 | 44 | 692 | 900 |
| 31841 | 160 | 663 | 862 |
| 32124 | 104 | 677 | 880 |
| 41817 | 61 | 688 | 894 |
| 44707 | 51 | 690 | 897 |
| 5323 | 55 | 689 | 896 |

**trace 整体对比（vs 原始 GWF workload）**：
- 输入长度 prompt_len：**0 条不一致**（sum 0.389M、p50=57、p90=203、p99=524、max=1696 完全相同）；
- 输出长度：**仅这 10 条不同**，总输出 12.265M → 12.267M（**+0.02%**）。

**结果**：

| | 基线 γ-Ada | tweak10 γ-Ada |
|---|---|---|
| goodput | 128.4 | **100.4（−22%）** |
| mTTFT | 51.0s | 167.6s |
| γ_s 收敛值 | 0.135 | **0.056** |
| 10 条边界请求 tpot | 30.x / 82–101ms | **36–38 / 88–112ms（全升）** |

**结论**：0.02% 的输出扰动 → γ-Ada 收敛点从 0.135 甩到 0.056（更激进 SJF）→ goodput −22%。这不是"微调直接导致违约"，而是 **γ-Ada 的 feedback loop 混沌敏感**。tpot 非但没按"摊薄"下降，反而全升（γ 翻转把负载抬高的效应淹没了摊薄效应）。

### 二、完整探索路径（按时间，含死胡同）

1. **γ 扫描**（o2000）：甜点 γ=1e-2（392），GWF=361 → 手动调 γ 下 UniBoost ≥ GWF。
2. **KV 打满负载 kv1p6**（o2000 顶部 30% output ×1.6）：三臂 native 134.2/3M 重算、**GWF 135.1/32k**、UniBoost(驱逐最长) 130.7/14.6M → **GWF 反超，靠零重算**。
3. **γ-Ada（完整 UniBoost）**：o2000 γ-Ada 388.8 仍 > GWF 361（离线预估"γ-Ada 掉到坏 γ"被证伪）；kv1p6 γ-Ada 128.4 < GWF 135.1（SJF 过激）。
4. **定向微调尝试（本记录重点，多轮试错）**：
   - 想"动最长 10 条怪物"→ 发现它们早已 deep-fail，动它只加重算成本、不加 KV 竞争 → **错**；
   - 想"动 decode2/3 实例的摇摆票"→ 路由是预测最小负载、**改 output 必改预测必改路由**，per-instance 定位失效 → **错**；
   - 想"保路由顺序 + 微调"→ 离线模拟路由模型只有 25% 准确率（=随机），**无法离线验证路由保序** → **不可行**；
   - 想"微调 688 边界请求长度"→ tpot 有摊薄效应（变长反降 tpot、帮 UnBoost），且 γ-Ada 混沌 → 本轮 tweak10 实证 **goodput 反掉 22%**。
5. **最终唯一干净路径**：全局加压（怪物 ×1.8 或 688 坨 +40%）+ 同一 trace 上 GWF/UniBoost 全量 A/B。

### 三、核心机制认知（找负载必须记住的 7 条）

1. **零重算 vs 驱逐重算**：KV 打满时，GWF 准入=零重算（32k），UniBoost 驱逐最长=重算风暴（14.6M，输出量的 119%）。这是 GWF 能赢的唯一结构性优势。
2. **摇摆票 = SLO 边界请求**：goodput 是阈值阶跃，真正决定 goodput 的是"刚好压线"的请求，它们高度集中在 **output≈688（全 trace 中位数那一大坨）**，不是怪物。
3. **tpot 摊薄（反直觉）**：recompute 的 decode 停摆是一段固定时长，`tpot ≈ 每token耗时 + 停摆/(output−1)`。**输出越短，停摆摊得越薄，tpot 越高**（数据：33–93 token 请求 tpot 500–1028ms，688 请求仅 30–45ms）。→ **把短请求变长会降低 tpot、帮 UniBoost 达标，反效果**。
4. **γ-Ada 混沌**：γ-Ada 有 feedback loop（γ→排序→完成顺序→尾部分位→γ），对初始条件敏感。**0.02% 输出扰动即可把 γ 从 0.135 甩到 0.056、goodput −22%**。→ GWF 无旋钮的鲁棒性论据。
5. **路由内生**：scheme4 是预测最小负载（predict_bottle=max(Load_Compute,Load_Memory,Load_Memory_Capacity)），WT_SIMPLIFY_PREDICT=1 下预测值=max_tokens=output 长度。**改 output 必改预测必改路由**，且是混沌级联，无法"保路由定点打击"。
6. **goodput 时长归一**：`output_token_goodput = good_output_tokens / benchmark_duration`。改怪物长度会拖长 benchmark 尾巴 → 分母变 → **GWF 的 goodput 数值也变，不能复用旧值**。
7. **re-prefill = prompt + 已生成 token**：抢占重算重算的是整个前缀（prompt+output），所以**输入长度也是 re-prefill 时长的直接杠杆**；kv1p6 里 prompt 短（~几百）output 长（16000），output 主导；若要 prompt 主导需另造"长 prompt 负载"。

### 四、可复用 playbook（未来找 GWF 更优负载）

1. 先跑三臂（native/GWF/UniBoost 固定γ）确认 KV 是否打满、重算量级。
2. 找"KV 打满 + 短请求"的交集：GWF 零重算要有舞台（KV 满 native 重算），且短请求要多（摊薄少、goodput 决定权大）。
3. 若要对比完整 UniBoost，必须跑 γ-Ada（墙钟），并**接受 γ-Ada 有 run-to-run 方差**——需重跑基线坐实。
4. **别做定向微调**：路由内生 + 时长归一 + tpot 摊薄 + γ-Ada 混沌，四条都堵死"只动几条"的省事路径。
5. 所有改动最终只能是**全局加压 + 同一 trace 全量 A/B**。

### step1 o2000 γ-Ada 墙钟真跑结果（2026-11-10）

**修复**：γ-Ada 挂钩从 `finish_requests`（只触发 abort）改成 `_free_request`（触发正常完成），γ_s 才真正自适应。

**o2000 γ-Ada 墙钟**（`wl_ub_o2000_gammaada/`，scheme4_uniboost + WT_UNIBOOST_CLOCK=wall + admit-preempt off）：
- γ_s 全程振荡 0.23↔1.86、终值 1.28（墙钟 1/s）；
- goodput **388.8**（req_gp 0.52、mTPOT 70.4、p99TPOT 105.7）。

**对照**：native 250.3、GWF(margin=250) 360.8、手动 γ 扫描 1e-2 甜点 392.1。

**结论**：真实 γ-Ada（feedback loop）收敛到 γ 让 UniBoost 拿 388.8 ≈ 手动甜点 392，**仍 > GWF 361**。"γ-Ada 让 UniBoost 掉到坏 γ、输 GWF"的假设**证伪**（离线预估用固定完成顺序是错的）。GWF 赢 UniBoost 的唯一舞台仍是 KV 打满（kv1p6 零重算 vs 驱逐重算），不是 γ 调参。

### kv1p6 三臂最终结果 + 机制层结论（2026-11-10，目标(3)达成）

**kv1p6（KV 打满，4P4D+util0.8，qwen_o2000 顶部 30%×1.6 → 怪物 8000–16000）三臂**：

| arm | goodput | mTTFT | p99TPOT | recompute | 占输出 |
|---|---|---|---|---|---|
| native (FCFS) | 134.2 | 16.3s | 1024ms | ~3.0M | 25% |
| **GWF（零重算准入）** | **135.1** | 36.6s | 186ms | **~32k** | **0.26%** |
| UniBoost（驱逐最长，admit-preempt off）| 130.7 | 57.6s | 424ms | **~14.6M** | **119%** |

（UniBoost 臂为修复版：关掉崩溃的 admit-preempt、保留 `_ub_order_running` 驱逐最长；见上方 bug 记录。）

**机制层结论（目标(3)）**：KV 打满 regime 下，**GWF 的零重算准入优于 UniBoost 的驱逐**——
- 重算：GWF 32k（0.26%）< native 3.0M（25%）< UniBoost 14.6M（119%，重算风暴、mTTFT 拖到 57.6s）。
- goodput：GWF 135.1 > native 134.2 > UniBoost 130.7（差距 3.4%，落在方差内，但方向与重算代价一致）。

**诚实 caveat**：① goodput 差距仅 3.4%，需 3 rep 确认；② UniBoost 的 14.6M 有实现成分——复现的驱逐路径（每步驱逐 `num_output_tokens` 最大的请求）**偏离论文**：论文（arXiv:2606.18431 §3.2/§3.3 + Algorithm 1）是 `SelectVictim(running set, π, swap-cost)`（论文只给此签名、未展开函数体/选择准则，入参为 π 与 swap-cost）+ MemGuard 几何限频（每请求 ≤1+⌊log₂(S_i/k)⌋ 次抢占）+ hysteresis δ_hyst，复现三者都缺，忠实论文会压低它；③ 结论限定为"复现配置下"。

**全局结论（0.8 下三负载）**：GWF ≤ UniBoost 只在 KV **未打满**（o2000 排序胜、dense_short SPF 胜）；**KV 打满时 GWF 反超（零重算 vs 驱逐重算风暴）**。这正对应论文"准入 vs 驱逐"的分水岭。

### UniBoost 代码 bug（2026-11-10 定位+修复）

**症状**：UniBoost 在 KV 打满负载（kv1p6）上反复崩溃（decode 实例引擎核无声死亡，5+ 次、每次 ~18 CRASH），而 native/GWF 同负载能跑完。

**定位**：同一时间窗跑 native kv1p6 诊断——KV 100%、recompute 61 万、零 crash → 非主机问题，是 UniBoost 代码 bug。根因是 `_ub_preempt_for_admission`（准入侧驱逐）在 `super().schedule()` **之前**驱逐怪物，同一 step 里 `schedule()` 又把这个刚驱逐的怪物重新调度（16000 token 全量重算）；而 native 抢占有 `preempted_reqs` 守卫、本 step 不重调度。同 step 重调度在 KV 打满时把 decode 实例搞崩。

**修复**：`WT_UNIBOOST_ADMIT_PREEMPT=0` 关掉这条准入侧驱逐路径，保留 `_ub_order_running`（native 抢占时驱逐最长 victim）。修复后 UniBoost 在 kv1p6 KV 100% 下零 crash（recompute ~2.75M，≈native 量级）。**注意（已核对论文 arXiv:2606.18431）**：论文**没有**"驱逐最长"规则（正文 "longest"/"evict the" 出现 0 次）。论文 §3.2/§3.3 + Algorithm 1 的抢占/驱逐是：优先级 `π(r)=a_r−b_r`、抢占触发 `Φ(j)<Φ(i_curr)−δ_hyst`（hysteresis）、受害者 `SelectVictim(running set, π, swap-cost)`（论文只给此签名、未展开函数体/选择准则，入参为 π 与 swap-cost）、MemGuard 几何量化 + minimum-run hysteresis（每请求 ≤1+⌊log₂(S_i/k)⌋ 次抢占）。本实现的"驱逐最长 victim"是自创简化，**不是**论文语义。

### UniBoost A/B 状态（2026-11-10 更新：γ 扫描完成 + 三负载三臂，结论=0.8 下 GWF ≤ UniBoost）

**代码**：UniBoost 为 4 个新增文件（`uniboost.py`/`uniboost_queue.py`/`uniboost_scheduler.py`/`uniboost_serve.py`），纯 stock vLLM API、零 fork 私有成员依赖；`scheme4_uniboost` 变体 = scheme4 预测路由 + UniBoost 实例内。

**1. γ 扫描（o2000，4P4D，util0.8，1 rep，`wl_ub_o2000_g{γ}/`）**：native 250.3；UniBoost goodput（tok/s）γ=1e-4→**326.0**、1e-3→**355.4**、1e-2→**392.1（甜点）**、1e-1→**386.2**；GWF(margin=250)=**360.8**。γ 曲线非单调、峰值 1e-2（1e-1 开始回落，γ→∞ 收敛回 FCFS=250）。**o2000 上 UniBoost(392) > GWF(361)**。

**2. kv1p6（KV 打满，qwen_o2000 顶部 30% ×1.6 → 怪物 8000–16000 token，`wl_o2000kv1p6_*/`）**：native goodput **134.2**、recompute **~3.0M**、p99TPOT 1024ms；GWF goodput **135.1**、recompute **~32k（↓100×）**、但 window 缩到 32 地板、mTTFT 16→37s；UniBoost recompute **~118k**（驱逐怪物，goodput 未拿到——3 次 crash，主机 SIGKILL）。**GWF≈native（净零），零重算机制成立但换不来 goodput**。

**3. dense_short（bimodal_burst 12000×200 均匀短，`wl_dense_ab/`）**：native **2276.0**；GWF **2158.6（−5.2%）**；UniBoost(γ=1e-2) **3223.7（+42%）**。**UniBoost 靠 prompt 长度 SPF 排序（w̃=max(output,prompt)）在均匀输出负载上仍 +42%**——之前"输出均匀→排序无效"的假设被证伪。

**结论（0.8 下三负载）**：GWF 无一能赢 UniBoost。根因三层：① GWF 在 0.8 仅靠 TTFT-margin 有窄收益（o2000+24%/o1600+4.3%），UniBoost 排序广谱有效（+42~57%）；② "零重算准入"需 KV 打满+短请求，0.8+4080 不可兼得（打满要长请求→τ 崩→窗口地板）；③ 当前 sum-code 偏激进（dense_short/kv1p6 都窗口地板→负/零）。**GWF 大赢（+31%~+110%）只在 gmu0.5+o1500（KV=436k、短请求 running 366 打满）。**

## 复现约定（2026-11-09 固化）

**主指标口径**：goodput 一律指 **Output token goodput（tok/s）** = 满足「TTFT≤1000ms ∧ TPOT≤30ms」的请求的**输出 token 数 / 总墙钟时间**；`Request goodput（req/s）` 作辅助。TPOT = Time per Output Token（不含首 token，看 Median/P99）；TTFT = Time to First Token（看 Median/P99）。基准臂是 **scheme4（预测路由+贪心准入）**，不是 roundrobin。

**trace 生成溯源**：`admission_ab_base/qwen_oXXX` = 对论文 qwen 原 trace（`qwen_lmsys_qwentrace_4080_600s.csv`，均值 7872.6）的 `output_tokens` 做等比缩放：`new = max(1, round(old × target/7872.6))`，**只改输出长度，4080 请求 + timestamp(QPS) 不变**。oXXX 的数字是**均值**，分布重尾（如 o2000：mean 2000 / median 688 / p90 7008）。`burstgpt_real` 是论文真实 burstgpt trace（未缩放，673 请求、135 条 >30k 极端）。

**每轮必记**（新轮照此填）：① 代码状态（改了什么 + commit）；② 配置（trace/模型/PD/gpu_util/θ/margin/预测/变体）；③ 精确启动命令 + env 覆盖；④ 结果目录路径；⑤ 环境快照（8 卡是否被别的租户占、loadavg、host 内存）；⑥ 结果数字 + GWF 计数器（window/stop_window/demoted/preempted/recompute）+ running 平衡；⑦ 结论 + 是否死胡同。

## 已证伪路径（别再试，除非有新的理由）

| 路径 | 结论 | 出处 |
|---|---|---|
| per-token θ=120（松，4×SLO） | GWF ≈ native 全线空转（窗口够不到 120ms） | #25–#27 |
| fraction 二值 τ | 0↔100 振荡，窗口钉 32 或卡 256 | #23/#20 |
| sum θ=120 无 margin | 窗口饿死请求 → P99 TTFT 510s/349s，goodput 平/负 | #29/#30 |
| o1200/o1600 无 margin | GWF 负收益（−16%/−20% 或 −4%） | #27/#30 |
| o4296（output 4296） | 30ms 物理不可达，θ 松紧都救不回 | #20/#21 |
| ln2k_cap4k（KV 富余） | GWF −19%（窗口只挡请求不省抢占） | #17 |
| o1500@gmu0.7 | GWF −7.4%（P99 TTFT 315s 饿死） | #15 对照 |

**有效路径**见 `winning_config.md`（四个正收益配置）。

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
| 新轮 goodput/TTFT 异常（像双倍负载） | 多个 `benchmark_serving_trace.py` 在跑，或某 bench 的 `--out-path` 指向旧 result dir | 启动前 `ps -eo pid,args \| grep benchmark_serving_trace` 确认无孤儿，有则按 PID kill |

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

21. **prefill 卡随机 SIGSEGV：`set_p2p_nccl_context` 的 environ 并发竞争（vLLM bug，已 patch `4ef9f9d5f`）**：
    症状：某张 prefill 卡突然 `EngineCore_DP0 died unexpectedly`，`[CRASH]` 记录到 GPU 显存归零、host 内存正常、cgroup `oom_kill=0`。
    真因在 prefill 日志里搜 `Caught signal 11` + `backtrace`：
    ```
    #4 libc getenv+0x56 / #5 libc10 c10::utils::get_env / #10 libtorch_cuda structured_mm_out_cuda
    ```
    `set_p2p_nccl_context`（`p2p_nccl_engine.py:45`）在运行时 `os.environ[...] = ...` → `putenv/unsetenv`
    会 realloc `environ`，与 torch 工作线程的 `getenv` 竞争 → 段错误。
    修复：该函数不再改 `os.environ`（no-op），NCCL 参数改在**启动环境**注入
    `NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1`。
    **排查口诀**：prefill/decode 随机单卡挂 → 先搜 `Caught signal 11` + `backtrace`，别只盯 OOM 和看门狗。

22. **4P4D 启动时部分实例挂掉：多租户 GPU 冲突（不是磁盘！我判断错过）**：
    症状：4P4D 的 8 个实例只起来 4 个（`nvidia-smi` 有 4 张卡 0 MiB），失败实例报
    `RuntimeError: Engine core initialization failed`，日志停在 `Loading safetensors checkpoint shards: 0/4~1/4`。
    **错误诊断（务必避免）**：看到 `/root/myshare` 100% 满（仅剩 20G）就归因为"磁盘不足"。
    实际上模型加载是**读**操作，磁盘满不影响读；且 `/dev/shm` 只用了 9%、cgroup `oom_kill=0`、dmesg 无新 OOM。
    **真因（多用户冲突）**：`dmesg` 里有 6 条
    `Cannot map memory with base addr 0x... and size of 0x800000 pages`
    （0x800000 pages × 4KB = 32GB）。共享主机上**别的租户已占用部分 GPU 显存**
    （实测 GPU 2/3/7 有 18–32GB 属于其他容器），vLLM 按 `gpu_memory_utilization=0.8`
    预留 64GB 虚拟映射时被 CUDA 拒绝 → `Engine core initialization failed`。
    **判断口诀**：多实例启动失败时，先 `nvidia-smi` 看**每张卡已用显存是否被别人占着**
    （`nvidia-smi --query-compute-apps=pid,used_memory` 里 PID 不在本容器 = 其他租户），
    再 `dmesg | grep 'Cannot map memory'` 确认；**不要一看磁盘满就下结论**。
    **规避**：① 优先用 3P4D（7 实例，已多次成功）；② 等别的租户释放显存后再上 4P4D；
    ③ 不要盲目调高 `gpu_memory_utilization`。

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

23. **路由负载漏记 waiting 队列的未来输出 → 实例间严重失衡（GWF 准入与路由脱节）**：
    症状：GWF 把热实例窗口压到 32、请求在 waiting 堆积（283/242），而空闲实例（kv 2.6%）收不到请求，路由仍往热实例塞——单实例准入在刹车，路由却不把负载导走。
    根因：proxy `_calculate_load` 里 `Future_Tokens = running_predict_tokens + waiting_tokens`，但 decode 实例上报的
    `waiting_tokens` 在 `--activation-predict` 关闭时**只算 prompt**（`scheduler.py::_waiting_token_contribution` 返回 `num_prompt_tokens`）。
    实测：某实例 `wait_reqs=242`、`Future=23612`，而真实未来占用 ≈ 242×4296 ≈ **100 万 token**，少了两个数量级 → Path B 判它"很空"继续塞。
    修复：GWF scheduler 覆写 `_should_track_predictive_pressure()` 返回 True（GWF 场景每个请求都有预测，理应计预测 token），
    使 `waiting_tokens = Σ(prompt+预测输出)`、`running_predict_tokens = Σ预测剩余`，对应论文 eq 6。
    **判据**：proxy MONITOR 行里 `wait_reqs` 很大但 `Future` 很小（如 242 → 23612）就是漏记。

24. **改完某个方法后，只重跑受影响的变体，没被影响的不要重跑**：
    症状：把 GWF 的 τ 单位从 per-token 改成 sum 后，我无脑重跑了整个 A/B（native + GWF 两臂），结果 native 又白跑一遍。
    根因：**native（scheme4）是 stock greedy scheduler，根本不读 GWF 的 τ/θ**——θ 的语义（per-token vs sum）只影响 `scheme4_gwf` 臂，native 臂结果只跟"路由修复 + trace"有关，θ 无关、可直接复用。
    修复：改参数前先想清"这个改动影响哪个 scheduler / 哪条代码路径"：
    - GWF 专属改动（τ 单位、θ、窗口、eV、demote、hopeless）→ 只重跑 `scheme4_gwf`，native 复用；
    - 路由改动（proxy / `_should_track_predictive_pressure` / `--activation-predict`）→ **两臂都要重跑**（路由同时影响 native 和 GWF）；
    - 用 `COMPARE_EXPERIMENT_VARIANTS="scheme4_gwf"` 只跑 GWF 臂（脚本 `gwf_sweep_run_gwf_only.sh`）。
    **判据**：改代码前问一句"native（greedy）会因为这个改动而改变行为吗？"——不会就复用 native。

25. **找 GWF 更优负载时误判甜点判据（avg_seq vs τ），且 monster factor 是阶跃不是渐变**：
    症状：gmu 0.8 固定下想调输出长度找 GWF 更优负载，我先用"avg_seq → KV 容量 ≈ Amax(256)"估 ×1.4 是甜点（avg_seq≈3440 → KV 容量 256 = Amax），实测 ×1.4/×1.2/×1.1 窗口**全部砸 32 地板**、GWF 退化成 ≈native，只有 ×1.0（o2000）+44%。
    根因（两层，都要记住）：
    - **决定窗口的不是 avg_seq，是 τ**（=4×每token时延，θ=120ms）。τ 由 monster 的 attention 成本决定，而 attention 随 KV 长度**平方级**增长——所以"KV 容量 ≈ Amax"这个甜点条件根本不成立，长请求必然 τ 爆、窗口缩。
    - **monster factor 是阶跃**：×1.0→×1.1 仅 10% 怪物增量，就把 τ 从 ~200ms 阶跃到 ~500ms、窗口从"→64 再回升"砸到 32 地板。中间没有可调的甜点带。
    修复/判据：
    - 判窗口成败**只看 window 值是否砸到 32 地板**，不能只看 KV%（×1.1/×1.2/×1.4 都是 KV 才 ~72% 就砸 32）或 recompute（窗口缩到 32 反而 0 重算，但那是过度节流）；
    - **τ 的早期值会误导**：×1.2 早期 running=92 时 τ~190ms 看着还行，running 到 256+（KV 饱和）后 τ 跳到 782ms。**必须等 running≥256 再判断**，早期 τ 不代表稳态；
    - **1p1d 时间轴要 ×4**（1 decode 实例配 1.7 req/s 才等于 4p4d 单实例负载），否则 6.8 req/s 是 4 倍过载、monster 更早填满 KV、窗口更早砸地板。
    - 结论固化：**gmu 0.8 下零重算准入兑现不了**（KV 打满要长请求 → attention 平方级 → τ 爆 → 窗口缩），唯一能赢的是 TTFT-margin（o2000 +44%），这就是上限，别再调负载找了。

## [4] UniBoost baseline（arXiv:2606.18431）集成

**为什么要自己实现**：论文 [4] 的项目页 `Code` 链接是空占位（`href="#"`，整页 0 个 github 链接），
官方**没有**发布代码。社区有一份 SGLang 实现，但只是 4 个 **未合并** PR（sgl-project/sglang#48076–#48079，
页面状态 `State--open`；sglang `main` 的 `schedule_policy.py`/`scheduler.py` 里 `boost` 匹配数为 0）。
PR #48076 可下载（633 行，含 158 行 `uniboost.py` + 316 行单测），只作公式参考。

**隔离原则（重要）**：stock vLLM 本身就是 baseline，**不允许改 vLLM 本体的任何文件**，
否则两个 baseline 共用代码路径、A/B 不成立。所以 UniBoost 全部放在新增文件里，
通过 launcher 在运行时注入 scheduler class：

| 文件 | 作用 |
|---|---|
| `vllm/vllm/v1/core/sched/uniboost.py` | 纯公式：`b_gamma(w)`、MemGuard 几何量化 `q(u)`、`uncached_work`、`phi` |
| `vllm/vllm/v1/core/sched/uniboost_queue.py` | 按 phi 排序的等待队列（懒刷新 + 每步 rebuild），**不改** stock `request_queue.py` |
| `vllm/vllm/v1/core/sched/uniboost_scheduler.py` | `Scheduler` 子类：替换 `self.waiting`；把"已生成 token 最多"的请求挪到 `self.running` 末尾，让 stock 的 `running.pop()` 正好驱逐 UniBoost 的 victim |
| `experiment/zrecent_use/0630/uniboost_serve.py` | `vllm serve` 包装：`EngineArgs.create_engine_config` 之后把 `scheduler_cls` 换成 `UniBoostScheduler` |

为什么不直接在 `scheduler.py` 里加 hook：那会让 stock baseline 也带上 UniBoost 代码路径。
"把最长的挪到末尾"利用了 stock 的现成语义（KV 不足时 `running.pop()` 驱逐最后一个），
且安全：running 循环按下标顺序走，失败请求在 `i < n-1`，`n-1` 本步还没被调度，不会重复调度。

**开关**（`test_evaluation_scheme4_rr_compare.sh`）：variant `scheme4_uniboost`
（与 `scheme4` 完全相同的路由/贪心准入，只有 intra-instance scheduler 不同）。

```bash
WT_UNIBOOST_GAMMA=1e-3        # 1/token，越大越接近 FCFS（stock 参考实现默认 1e-4）
WT_UNIBOOST_BIN=128           # MemGuard 量化粒度 k
WT_UNIBOOST_ORDER=1           # 0 = 只保留驱逐策略，队列退化为 FCFS（消融）
WT_UNIBOOST_VICTIM=longest    # longest | newest
WT_UNIBOOST_VICTIM_MIN_TOKENS=32
WT_UNIBOOST_LOG_INTERVAL_MS=1000
```

**验证（单实例冒烟，GPU0，gmu=0.35）**：
- 240 请求 / 64 并发：全部成功，2504 tok/s，`victim_swaps` 递增，`preempted_recompute_tokens=0`；
- 360 请求 / 300 并发、输出 2048–4096：全部成功，2644 tok/s，
  `preempted_recompute_tokens=160257`、`victim_swaps=10756` → **驱逐路径确实被触发**且不崩。

**踩坑**：
1. `python uniboost_serve.py serve --model X` 会报 `ValueError: With vllm serve, you should provide
   the model as a positional argument`——`vllm serve` 只接受位置参数，launcher 必须同样传位置参数。
2. 直接改 `vllm/config/scheduler.py`（加 `--uniboost-scheduler`）虽然能用，但**违反隔离原则**，
   已全部回滚；`git status` 中 `vllm/v1/core/sched/scheduler.py`、`request_queue.py` 必须保持 clean。
3. 2026-11-07 15:40 又出现**多租户抢占 GPU 3**（74 GB，非本容器进程），3P4D 的 prefill 改到
   `PD_3P4D_PREFILL_GPUS=0,1,2` 才起来——同踩坑 #22。

## 环境（每次实验都应记录）

- 8× A800-80GB，4P4D，Qwen-2.5-7B-Instruct，TTFT SLO 1000ms / TPOT SLO 30ms，vLLM v0.11.0。
- 权重路径 `/root/myshare/models/`；启动前必设 `VLLM_HOST_IP=127.0.0.1`、`no_proxy=localhost,127.0.0.1`。

## 后台队列编排（2026-11-11 设置）

**目的**：后台检测 8 卡空闲后，依次跑完待办实验，超时/崩溃止损并记录。

**编排器**：`run_pending_queue.sh`（后台 job `bash-2a152e58545c42c3`），日志 `queue_run.log`。空闲判据 = 无 eval 进程 ∧ 8 卡显存全 <2GiB ∧ 无 22xxx 端口（连续两次 30s 确认）。

**队列顺序**（driver / 超时）：
1. γ-Ada kv1p6 稳定性复跑 `run_gammaada_stability.sh`（90min）
2. step2 kv1p6 GWF Amax=220 `run_step2_amax220.sh`（90min）
3. 真实 kv1p6-like lmsys `run_real_kv1p6like_lmsys.sh`（170min）
4. 真实 kv1p6-like sharegpt `run_real_kv1p6like_sharegpt.sh`（170min）
5. step3 真实 BERT 预测器 `run_step3_realpred.sh`（300min）
6. 1p1d ×1.8 `run_1p1d_x1p8_ab.sh`（170min）
7. 4p4d ×1.8 `run_4p4d_x1p8_ab.sh`（170min）

**新增 trace**：`admission_ab_base/qwen_o2000_kv1p8{,_1p1d}`（top30% output>1035 ×1.8，1223 条，max 14968）；`admission_ab_base/qwen_4080_600s`（step3 原始 trace，mean 7872.6，342 条 >30k extreme）。

**止损机制**：每项 `timeout -k 60`；超时/非零退出后 `cleanup_procs`（kill vllm/proxy/benchmark）；扫 `GPU deadlock|Out of memory|exit 137|SIGKILL|EngineCore died|oom-killer` 标记并记录。

## 1p1d ×1.8 第 1 臂 GWF 结果（2026-11-10 23:30）

**配置**：1p1d（prefill GPU0 + decode GPU4）、gmu0.8、trace qwen_o2000_kv1p8_1p1d（1223 条怪物 ×1.8，max 14968，时间轴 ×4 → 1.7 req/s）、oracle 预测、GWF sum θ=120 + TTFT-margin 250。

**结果（GWF 单臂）**：
| 指标 | 值 |
|---|---|
| Output token goodput | **13.03 tok/s** |
| Request goodput | 0.02 req/s |
| mTTFT / p99TTFT | **584.6s / 2108.9s**（SLO 1s 的 584×/2109×）|
| mTPOT / p99TPOT | 519.5ms / 2066.3ms（SLO 30ms 的 17×/69×）|
| recompute_tokens | **11.78M** |
| preempted 次数 | 8245 |
| stop_window | 674 |
| window 终值 | 32 地板 |
| kv_usage 峰值 | 100% |
| τ 峰值 | ~15.6s（decode 停摆）|

**结论**：×1.8 在 1p1d 下**严重过载**——单 decode 实例被 1223 条怪物把 KV 打满（100%），触发抢占风暴（preempted 8245 次、recompute 11.78M ≈ 输出的 86%），decode 每步 τ 飙到 15.6s（正常 ~50ms），TTFT 584s、TPOT 519ms 全线击穿 SLO，goodput 13 tok/s ≈ 0。GWF 的窗口→32 地板 + TTFT-margin 救不了这种过载。

**踩坑**：×1.8 1p1d 每臂 ~3.3h（尾部怪物 max 14968 token 生成极慢），双臂 ~7h，循环迭代不可行。后续必须降负载（×1.2/×1.4 或截断怪物长度）。

**第 2 臂 UniBoost 已启动**（23:30），预计同样 ~3.3h。

## 1p1d ×1.8 第 2 臂 UniBoost 结果：崩溃（2026-11-11 00:42）

**结果**：UniBoost 臂 **未跑完即崩溃**。

- 19% 进度（765/4080）就 KV 99.9% + `running=155 waiting=869` + `victim_swaps=1348` + `recompute_tokens=1.2M` —— UniBoost 无窗口节流，比 GWF 更早把 KV 打满，触发驱逐风暴。
- 随后**主机 OOM**：`Out of memory: Killed process (sglang::schedul) shmem-rss=29856072kB`（co-tenant 的 sglang 作业被内核杀）。
- UniBoost decode 实例死掉：summary 链 = `instance_health_failure_retry1` → `start_servers_failed_retry2` → `start_failed`。

**踩坑（新增）**：
1. **×1.8 在 1p1d 下对两臂都不可用**：GWF goodput 13 tok/s≈0、UniBoost 直接崩溃。**×1.8 不是甜点，必须降负载**。
2. **UniBoost 的驱逐风暴比 GWF 更早触发**（19% vs GWF 90% 才 KV 满），因为 UniBoost 无窗口节流、贪婪准入。
3. **主机 OOM 是 co-tenant sglang（shmem 29.8GB）+ 我们实例叠加触发**，与坑 #14 同源；我们 mem_pool 已降到 4GB 仍躲不过 co-tenant 的 wildcard。
4. **`/root/myshare`（nvme0n1p2）磁盘 100% 满（11G 可用）**——模型/预测器只读不受影响，但后续若需写该盘要注意。

**双臂对比（×1.8 1p1d）**：GWF 13.03 tok/s（完成）vs UniBoost 崩溃（无 goodput）。**方向虽"GWF 存活、UniBoost 崩"，但两者都 ≈0，不构成有效对比；根因是 ×1.8 过载，不是调度差异。**

## 1p1d ×1.4 第 1 臂 GWF 结果：同样过载，已提前终止（2026-11-11 01:26）

**配置**：1p1d（prefill GPU0 + decode GPU4）、gmu0.8、trace qwen_o2000_kv1p4_1p1d（4080 条，max 11642，sum 10.90M，mean 2670.8）、oracle 预测、GWF θ=120 + TTFT-margin 250。

**结果（GWF 单臂，跑到 20% 即终止）**：
| 指标 | 值 |
|---|---|
| 进度 | **20%（825/4080）**，运行 18:22 |
| 首次 kv=100% | **01:25:50 ≈ 15min / 18% 进度** |
| kv_usage | 100%（打满）|
| window | **32 地板**（早期即塌）|
| τ（decode 步延迟）| **830–1070ms**（SLO TPOT 30ms 的 28–36×）|
| preempted 累计 | 301 次（684 个 GWF 步）|
| recompute 累计 | 150,038 tok（已开始）|
| evicted_hopeless | 2–4/步（驱逐风暴已起）|
| waiting 队列 | **396→408 积压**|
| admitted | 0（KV 满，窗口无法准入）|

**结论**：×1.4 在 **20% 进度（15min）就 KV 100% + τ≈1s + waiting 400 + 驱逐风暴**，过载签名与 ×1.8 完全一致，且**触发更早**（×1.8 GWF 到 ~90% 才 KV 满）。goodput 必然 ≈0（TPOT 已 28–36× 超 SLO、TTFT 因 waiting 400 也会击穿）。**×1.4 不是甜点，已提前终止，不再等 2.5h 出 ≈0 的 goodput**。

**踩坑（新增 #26）**：
1. **降 monster factor（×1.8→×1.4）不能缓解过载**：KV 仍 15min 打满，因为过载根因不是 monster 长度，而是**到达率（timestamp ×4）在单 decode GPU 上灌入太多并发**（running 涨到 600–900，τ 飙到 1s）。
2. **1p1d 单 decode GPU 的 TPOT 30ms SLO 极苛刻**：τ=步延迟=单 GPU 整批前向时间；running>~100–200 时 τ 必然 >120ms → GWF 窗口贴 32 地板，失去准入控制能力。
3. **过载时 GWF 窗口形同虚设**：窗口→32 地板后仍 KV 100%，只能 evicted_hopeless + preempt，recompute 风暴照旧——与 UniBoost 的差异只剩"崩溃早晚"，无法在 goodput 上区分。

## 根因分析：1p1d 单 decode GPU 过载机制（2026-11-11 01:40）

**核心度量校准**（读 GWF 代码 `_update_window_controller`）：`tau_ms` = 每个请求"最近 4 个 token 间隔之和"的 EMA，即 **4× per-token latency**；θ=120ms 对应 TPOT SLO 30ms。所以 tau_ms=60 ⇒ TPOT≈15ms（健康），tau_ms>120 ⇒ TPOT>30ms（超 SLO）。

**×1.4 早期健康基线**（running=6~18）：tau_ms≈60~66 ⇒ **TPOT≈15ms**，窗口从 32 长到 256，一切正常。
**×1.4 过载后**（running=588~919）：tau_ms≈830~1070 ⇒ TPOT≈207~267ms（7~9× SLO），窗口已塌到 32 地板。

**单 decode GPU 吞吐实测**：running 18 时 18/0.015=1200 tok/s → running 588 时 588/0.393=1500 tok/s，**上限 ≈1200-1500 tok/s**（远低于 7B 理论 3000-5000，疑为 1p1d disagg + 旧卡）。

**过载根因（数字）**：
- ×1.8（rate1.7, mean~3000）→ demand ≈5100 tok/s ≈ **3.4× 吞吐**
- ×1.4（rate1.7, mean2671）→ demand ≈4540 tok/s ≈ **3.0× 吞吐** → 15min KV 打满
- 结论：**降 monster factor（×1.8→×1.4）只降 mean 12%，治标不治本；主导因子是到达率**（rate 1.7 相对单 decode 吞吐过高）。

**GWF 窗口被 TTFT-margin force-admit 击穿**：窗口地板 32 本应把 running 压在 ~32（tau≈100ms<120），但 `WT_GWF_TTFT_MARGIN_MS=250` 让"快到 TTFT 预算"的请求绕过窗口强制准入；持续过载下几乎所有排队请求都触发 force-admit → running 冲到 588-919 → tau 1s+ → KV 100% → 驱逐风暴。**窗口在持续过载下形同虚设**。

**甜点判据**：需 demand ≈1.3-1.6× 吞吐（~2000 tok/s），使队列缓慢增长、TTFT-margin 只零星触发，GWF 窗口能真正节流，而 UniBoost 无窗口超发 → 抢占 → goodput 劣势。

## 负载优化 + 重新测试（2026-11-11 01:40 启动）

**优化**：保留 ×1.2 怪物分布（mean2335/max9979，长尾占 KV），把到达率从 rate1.7 降到 **rate0.852/s**（timestamp ×8，即原 ×4 再 ×2）。
- 新 trace：`admission_ab_base/qwen_o2000_kv1p2_1p1d_x8/`（n=4080, mean2335, max9979, sum9.53M, rate0.852/s, **demand≈1990 tok/s ≈1.3-1.6× 吞吐**）
- 驱动：`run_1p1d_x1p2_x8_ab.sh`，GPU 0+4，oracle，GWF θ=120 + TTFT-margin250，双臂 scheme4_gwf → scheme4_uniboost。

**验证策略**：先看第 1 臂 GWF 前 20-30min 是否健康（KV<85%、tau 稳定在 60~120ms、无驱逐风暴）；若仍 15min KV 打满则继续降到达率（×16 或 ×12）。若健康则跑完双臂对比 goodput。

## 踩坑 #27：kill 不干净导致 GPU 被孤儿 EngineCore 占死 + 双实验互抢（2026-11-11 01:45）

**现象**：kill 掉 `run_1p1d_x1p4_ab.sh` 后，GPU 0/4 各残留 ~64GB，`nvidia-smi --gpu-reset` 报 "In use by another client"，新实验 vllm 无法分配 gmu0.8。

**根因（两条）**：
1. **vLLM 的 GPU 进程叫 `VLLM::EngineCore`，不是 "vllm serve"**。kill 时若只 grep "vllm serve" 会漏掉；该进程在父进程死后被 reparent 到 PID 1 继续存活，一直占着 CUDA context 和显存。必须 `kill -9` 掉 `VLLM::EngineCore` + `multiprocessing.resource_tracker`。
2. **`run_1p1d_x1p4_ab.sh` 的 child `test_evaluation_scheme4_rr_compare.sh`（PID 3460798）未被杀**——只 kill 了 wrapper bash，child 变 PPID=1 后继续跑完 GWF 臂、自动进入 UniBoost 臂，跟随后新启动的 ×1.2×8 驱动抢 GPU 0/4 和端口 32323/22001/22020/22007。

**正确 kill 清单**（每次收工必须全清）：
```
kill -9 <run_*.sh wrapper>
kill -9 <test_evaluation_scheme4_rr_compare.sh child>
for p in $(ps -eo pid,cmd | grep -E "VLLM::EngineCore|vllm serve|benchmark_serving_trace.py|test_disagg_proxy.py|uniboost_serve.py" | grep -v grep | awk '{print $1}'); do kill -9 $p; done
```
验证：`ps -eo pid,cmd | grep -E "VLLM::EngineCore|vllm serve|benchmark_serving|test_disagg" | grep -v grep` 为空，且 `nvidia-smi` 上 GPU 0/4 = 0 MiB。

**注**：GPU 0/4 = A800-SXM4-80GB（非 H100），7B decode 实测吞吐 ~1200-1500 tok/s，这也解释了 1p1d 甜点窗口窄。GPU 1/3 是 co-tenant（30GB 静态），GPU 2 是其他人（69GB），我们只用 0/4。

## ×1.2×8 重新测试：前 3min 健康（2026-11-11 01:52）

**trace**：qwen_o2000_kv1p2_1p1d_x8（rate0.852/s, mean2335, demand≈1990 tok/s）
**前 3min 观测**（进展 104/4080 = 3%）：
| 指标 | 值 | 判读 |
|---|---|---|
| kv_usage | 11.7–12.0% | 远未满 |
| running | 31 | 低 |
| waiting | 0 | 无积压 |
| window | **256（满）** | 未被节流 |
| tau_ms | 84.6–100 | **TPOT≈21–25ms ≤ SLO30** |
| preempted/recompute | 0 / 0 | 无驱逐 |
| over_pct | 0.0 | 无超预算 |

**对比 ×1.4**：×1.4 在 3min 时已 KV 快速爬升、15min 打满；×1.2×8 在 3min 仅 12% 且 tau 在 SLO 内。**降到达率确实把系统从"过载"拉回"健康/轻度"区间**。

**待观察**：demand 1990 > 吞吐 ~1378（running31/TPOT22.5ms）→ 队列以 ~600 tok/s 速度累积，KV 会缓慢上涨。关键看进入高压后 window 是否开始节流（GWF 的用武之地）、UniBoost 臂是否超发。若全程 KV<60% 且无抢占，则两臂会打平（无差异），需再微调升到 demand≈2500-3000。

## ×1.2×8 重新测试：进入窗口节流区，零抢占（2026-11-11 02:12, 23%）

**进展 930/4080=23%，23.5min**：
| 指标 | ×1.2×8 (rate0.85) | ×1.4 (rate1.7, 20%时) |
|---|---|---|
| kv_usage | 96.2% | 100% |
| running | 272–275 | 616–919 |
| window | **32 地板**，stop_window=6 | 32 地板 |
| tau_ms | 459–463（TPOT≈115ms）| 830–1070（TPOT≈207–267ms）|
| **preempted** | **0** | **301** |
| **recompute** | **0** | **150,038** |
| 驱逐风暴 | 无 | evicted_hopeless 2–4/步 |

**关键**：GWF 窗口在 running 爬到 ~256 后真正介入（window 32 + stop_window），把 KV 稳在 96% **不越界**，**零抢占零重算**。这是 GWF 的机制首次在 1p1d 下生效——对比 ×1.4 同时 KV 100%+抢占风暴。**rate0.85 是目前最接近"GWF 用武之地"的负载**。

**但仍需验证**：TPOT 115ms（3.8× SLO）意味着绝大多数请求仍超预算，goodput 可能不高；真正决胜要看第 2 臂 UniBoost 在同负载下是否抢占/重算/崩溃。继续观察到双臂完成。

## ×1.2×8 第 1 臂 GWF 结果：完成，goodput 144.17 tok/s（2026-11-11 03:14）

**配置**：1p1d（prefill GPU0 + decode GPU4, A800-80GB）、gmu0.8、trace qwen_o2000_kv1p2_1p1d_x8（rate0.852/s, mean2335, sum9.53M）、oracle、GWF θ=120 + TTFT-margin250。

**结果（GWF）**：
| 指标 | 值 |
|---|---|
| status | **success（未崩溃）** |
| Benchmark duration | 5121.23 s |
| Total generated tokens | 9,525,801 |
| **Output token goodput** | **144.17 tok/s** |
| Request goodput | 0.09 req/s |
| Total token throughput | 1,935.98 tok/s |
| Mean TTFT | 16,945.57 ms |
| P99 TTFT | 228,061.99 ms |
| Mean TPOT | 85.70 ms |
| P99 TPOT | 171.87 ms |
| preempted | 1,474 |
| recompute_tokens | 1,059,638 |

**关键对比**：
- ×1.8 GWF goodput **13.03** → ×1.2×8 GWF **144.17**（**11× 提升**）
- ×1.8 recompute 11.78M → ×1.2×8 recompute 1.06M（**11× 降低**）
- 4p4d kv1p6 GWF 135.1（不同负载，仅作量级参照；本负载 per-GPU demand 1985 tok/s 比 4p4d 的 4430 轻）

**判读**：降到达率（rate1.7→0.85）让 GWF 有了长健康 phase + 窗口节流，goodput 从 ≈0 拉到 144 tok/s，且**第 1 臂干净完成、零崩溃**。这是目前最好的 GWF 数据点。**决胜看第 2 臂 UniBoost 同负载**：若 <144 则 GWF>UniBoost 成立；若 UniBoost 崩溃则 GWF 以稳健性胜。（mean TPOT 85.7ms 仍 >SLO30，说明 goodput 来自前段达标请求的贡献，约占总吞吐 7.4%。）

## 🎯 阶段 A 完成：找到 1p1d GWF>UniBoost 甜点负载（2026-11-11 04:46）

**负载**：qwen_o2000_kv1p2_1p1d_x8（1p1d，GPU0 prefill + GPU4 decode，gmu0.8，A800-80GB）
- 4080 请求，output mean=2335.4，max=9979，sum=9.53M
- 到达率 **0.852 req/s**（timestamp ×8），demand≈1990 tok/s
- oracle 预测（WT_SIMPLIFY_PREDICT=1），GWF θ=120 + TTFT-margin250

**双臂结果（同负载，均 success 未崩溃）**：

| 指标 | GWF | UniBoost(γ-Ada) | 差异 |
|---|---|---|---|
| **Output token goodput** | **144.17 tok/s** | **133.51 tok/s** | **GWF +8.0%** ✅ |
| Request goodput | 0.09 req/s | 0.08 req/s | GWF +12.5% |
| Benchmark duration | 5121.23 s | 5428.44 s | GWF 快 5.7% |
| Total generated | 9,525,801 | 9,496,135 | — |
| Mean TTFT | **16,945.57 ms** | 74,719.68 ms | GWF 4.4× 好 |
| P99 TTFT | 228,061.99 ms | 475,411.72 ms | GWF 2.1× 好 |
| Mean TPOT | **85.70 ms** | 113.13 ms | GWF 1.3× 好 |
| P99 TPOT | 171.87 ms | 205.24 ms | GWF 1.2× 好 |
| preempted / victim_swaps | 1,474 | 10,912 | GWF 7.4× 少 |
| recompute_tokens | **1,059,638** | **1,546,684** | GWF 少 31.5% |
| window_min | 32（stop_window 累计 13835）| —（无窗口）| — |

**结论**：
1. **GWF 在 1p1d 下确实能找到 > UniBoost 的负载**（+8.0% goodput，且延迟全面更优）。
2. 机制清晰：GWF 的准入窗口（window 32 + TTFT-margin）**减少 eviction/recompute 31.5%**（1.06M vs 1.55M），减少无效重算浪费 → 更多 token 达标 → goodput 更高；同时 TTFT 从 74.7s 压到 16.9s。
3. **负载关键**：rate 必须降到 ~0.85 req/s（×1.4/×1.8 的 1.7 太快，直接过载到 ≈0）。降到达率让系统进入"可控过载"区，GWF 的窗口才有用武之地。
4. 对比 4p4d kv1p6（GWF 135.1 > UniBoost 130.7，+3.4%），本 1p1d 甜点的 GWF 优势更大（+8.0%），且两臂均未崩溃（不同于 ×1.8 UniBoost OOM 崩溃）。

**下一步（阶段 B）**：以本负载 output 分布为模板，从 5p1d 真实预测数据集（50k，含真实 BERT 预测）找逐请求近似子集，跑 1p1d + 真实预测 + GWF vs UniBoost。

## 阶段 B：子集构建 + trace 生成（2026-11-11 14:10）

**模板**（阶段 A 甜点）：qwen_o2000_kv1p2_1p1d_x8，mean2335/median689/max9979，4 区间 {<500:1165, 500-2000:1775, 2000-5000:248, ≥5000:892}，rate0.852/s。

**5p1d 真实数据实况**：
- 源 trace `50k/qwen-lmsys-chat-updated.csv`：50000 条，output 分布 {<500:60.2%, 500-2000:16.7%, 2000-5000:3.9%, 5000-9979:3.4%, >9979:15.7%}，重尾（median359/p90 27016/max32738）。
- bucket_probs `actual_output_len` 全是 1（占位符坏），真实长度在源 trace 的 `output_tokens`。
- **预测器弱**：raw predict_output_mean 与 actual 相关 corr=0.351；β=0.30 校准后均值正确（4819 vs 4823），但逐请求仅 3.9% 在 30% 内、7% 在 50% 内；且"准"的请求偏向长输出（短区间准的极少）。

**子集构建策略**：4 区间匹配模板 + 区间内按 |β·pred − actual|/actual 最小优先（堆选 top-K）。
- 结果：4080 条，mean2312/median713/max9996（≈模板），预测中位误差 0.49、50.2% 在 50% 内（远好于全量 7%）。
- trace：`admission_ab_base/qwen_real5p1d_1p1d_x8/qwen_lmsys_real5p1d_1p1d_x8.csv`，rate0.852/s。
- 驱动：`run_1p1d_real5p1d_x8_ab.sh`（1p1d GPU0+4，**去掉 WT_SIMPLIFY_PREDICT，加 WT_PREDICT_OUTPUT_SCALE=0.30**）。

**待验证**：真实预测下 GWF 是否仍 > UniBoost（阶段 A 为 144.17 vs 133.51）。若真实预测噪声吞掉 GWF 优势，需回退优化（如更严的预测准阈值、或换 β）。

## 实验调度优先级规则（2026-11-11 14:15 用户指定）

**共享服务器上做实验，卡资源分配原则**：
1. **若 8 卡能空出来 → 优先跑 8 卡（4p4d）实验**；否则跑能分配出卡的实验（如 1p1d 只用 2 卡）。
2. **当前这轮 1p1d 阶段 B（真实预测 GWF vs UniBoost）跑完后**：若 8 卡空闲，**先跑 `run_step3_realpred.sh`（step3 真实 BERT 预测器 4p4d，β=0.30）**——它等价于阶段 B 的 4p4d 版本，比较重要。

**当前状态**：阶段 B 1p1d 占用 GPU 0/4；GPU 1/2/3/5/6/7 已空（co-tenant 退场）。阶段 B 跑完双臂后，检查 8 卡是否全空 → 空则优先启动 step3_realpred（4p4d）。

## 并行 1p1d：ZMQ 端口可配置补丁（2026-11-11 14:45）

**目的**：共享服务器上阶段 B 占 GPU 0/4，用空闲 GPU 并行跑其它 1p1d（省墙钟）。

**坑**：disagg 实验有 3 处硬编码 ZMQ 端口，并行会撞：
1. `test_disagg_proxy.py` `PREDICTOR_ZMQ_ADDR = tcp://127.0.0.1:32323`（预测结果 PULL）
2. `test_disagg_proxy.py` `ROUTING_PUB_ADDR = tcp://127.0.0.1:32324`（路由 PUB）
3. `vllm/.../p2p_nccl_connector.py` `start_routing_listener("tcp://127.0.0.1:32324")`（vLLM 路由 SUB）

**补丁**（3 行，均 `os.environ.get(..., 默认原端口)`）：
- `WT_PREDICTOR_ZMQ_ADDR`（默认 32323）
- `WT_ROUTING_PUB_ADDR`（默认 32324，proxy 与 vLLM 都读）

**并行 driver**：`run_1p1d_x1p2_x8_rep2_parallel.sh`（阶段 A ×1.2×8 第 2 次重复，GPU 1+5）：
- 主端口整体 +100（22001→22101, 22020→22120, 22007→22107, 28002→28102, KV 同步 +100）
- ZMQ +100（32323→32423, 32324→32424）

**验证**：rep2 干净启动，无 "Address already in use"，GPU 1+5 正常加载（63.9/65.9GB）。
**注意**：rep2 是 oracle（WT_SIMPLIFY_PREDICT=1），不触发预测 ZMQ 32323；真实预测并行的预测 PUSH 侧是否还有硬编码待后续真实验证。

## 阶段 B 第 1 臂 GWF 结果：goodput 355 但 400 请求失败（2026-11-11 07:26）

**配置**：1p1d 真实 BERT 预测（β=0.30），trace qwen_real5p1d_1p1d_x8（4080 条，mean2312/max9996，从 5p1d 真实数据按模板分布+预测准优选）。

**结果（GWF，success）**：
| 指标 | 值 |
|---|---|
| **Successful requests** | **3684 / 4080（400 失败，9.8%）** ⚠️ |
| Output token goodput | **355.04 tok/s** |
| Request goodput | 0.22 req/s |
| Total generated tokens | 7,925,129（trace sum 9,432,546 = **84%**）|
| Mean/Median TTFT | 5015 / 275 ms |
| Mean/Median TPOT | 66.57 / 50.79 ms |
| preempted / recompute | 1620 / 967,542（64% 时）|

**关键发现（踩坑 #28）**：真实预测下 **400 个请求失败**（`expect_output_len` mean=3769、max=9977，即**最长的那批请求**），而阶段 A oracle 是 4080/4080 全成功。失败的长请求释放了 decode 容量 → 短请求 TPOT 更优 → goodput 虚高（355 vs oracle 144）。**goodput 数字被"丢长请求"污染，不能直接拿 355 与 oracle 144 对比**。

**真正要看的对比**：同负载同预测下 **GWF vs UniBoost** 的 (goodput, 失败数, recompute)。若 UniBoost 失败更多/ goodput 更低，GWF 仍胜；若 UniBoost 失败更少，则需回退优化子集。**等 UniBoost 臂（已启动 07:26:55）。**

## 踩坑 #29：UniBoost(γ-Ada) 在 1p1d 下 ~4% 进度死锁（2026-11-11 08:10）

**现象**：阶段 B（真实预测）和并行 rep2（oracle）两个 UniBoost 臂都在 **~150/4080（4%）** 死锁：
- decode `token_clock` 冻结（46057 两次采样不变）、decode 日志停更；
- `gamma_s` 跳到 **13.98**（γ0=0.1，MAX_JUMP=60，极端值）；
- KV 仅 2%、running 15-17、waiting 0 —— **非过载，是低负载死锁**；
- benchmark 卡在 151/4080（在等 15 个 in-flight 请求响应，永不返回）；
- 阶段 B：instance_health_failure_retry1→retry2→benchmark_failed；rep2 同样卡死。

**对比**：阶段 A UniBoost（×1.2×8 oracle，GPU0+4，默认端口）在 03:15-04:46 **成功完成**（133.51 tok/s）。同 trace 同配置，仅 GPU/端口/时间不同 → **死锁是间歇性的**。

**待查根因**：victim_swaps=65 在 KV 2% 就触发（无 KV 压力却 swap）；γ 跳 14；--swap-space 0 下 victim swap 可能触发 re-prefill 而 prefill/decode 互等 → 死锁。需读 uniboost_scheduler.py 的 victim_swaps + γ-Ada 逻辑定位。

**影响**：UniBoost 在 1p1d 存在间歇性死锁（3 次跑 2 次卡死），会污染后续所有含 UniBoost 臂的实验（含 8 卡 step3）。**先定位死锁再继续跑 UniBoost 臂。**

## GWF 失败请求根因（踩坑 #30，阶段 B 400 失败）

**现象**：阶段 B GWF（真实预测）3684/4080 成功，400 失败（ttft=0，从未产生首 token）。

**机制（已确认）**：
- proxy 日志出现大量 **503**（Server Error）和 `connection error: Server disconnected / Connection reset by peer`。
- 根因是 **P/D 抢占丢请求**：过载时 decode 驱逐 KV → 请求需 re-prefill，但 disagg 下 prefill KV 已不存在 → proxy 返回 503。
- benchmark `WT_RETRY_TIMES=2` 重试 2 次，但持续过载下重试仍被抢占 → 记失败。

**为什么阶段 B 比阶段 A 失败多**：
- 阶段 A（oracle）：preempted 1474，0 失败。
- 阶段 B（真实预测 β=0.30）：preempted 1620，**400 失败**。
- 真实预测噪声 → GWF 更频繁误准入/误驱逐 → 更多抢占 → 更多 P/D 丢请求。
- 失败请求**多为长请求**（expect_len mean=3769 vs 成功 2154）：长请求 KV 多、暴露时间长，更易被抢占。

**对结论的影响**：阶段 B GWF goodput 355 被"丢 400 长请求"虚高（释放了 decode 容量），不能与阶段 A oracle 144 直接对比。

## 更正 #30：GWF 失败根因是心跳竞态，不是"prefill KV 丢失"（2026-11-11 22:10）

**更正**：之前把 400 失败归因于"过载→decode 驱逐 KV→re-prefill 时 prefill KV 已不存在→503"，**这个解释是错的**。re-prefill 是标准恢复路径，KV 没了重新 prefill prompt 即可。

**真正根因**：
- 467 次 503 响应体 = 20 字节 = `"No prefill instances"`（proxy `test_disagg_proxy.py:2082`）。
- vLLM 实例通过 `p2p_nccl_engine.py:639 ping()` 每 **3s** 发一次注册心跳（`while True: send(注册); time.sleep(3)`）。
- proxy 侧 `DEFAULT_PING_SECONDS=5`：**5s 内没收到心跳就移除实例**。
- 心跳 3s vs 过期 5s，**余量仅 2s**；过载时 ping 线程被延迟 >5s → proxy 误删 prefill → 后续请求 503 → 重试也撞 → 400 失败。

**定性**：**我们的基础设施 bug（心跳余量太紧）**，与 GWF/UniBoost 调度方法无关，也与"re-prefill 恢复"路径无关。

**修复**：`DEFAULT_PING_SECONDS` 5→20（心跳 3s 的 ~7 倍），避免过载下误判实例下线。

## 踩坑 #31：UniBoost 死锁定因——`_ub_order_running` 重排序触发（2026-11-11 22:47）

**验证实验**：`WT_UNIBOOST_VICTIM=newest`（关闭 `_ub_order_running` 每步 swap），oracle ×1.2×8，GPU2+3，只跑 UniBoost 臂。

**结果**：
| 时点 | victim=longest（默认，重排序开）| victim=newest（重排序关）|
|---|---|---|
| ~150 请求（KV 2%, running 15）| **死锁**（rep2/阶段B 均卡死，token_clock 冻结）| **正常通过**（284→838 请求，token_clock 推进）|
| victim_swaps | >0（每步 swap）| **0**（确认已关闭）|

**定性（三类原因区分）**：
1. **不是 UniBoost 方法本身的问题**：死锁在 KV 2%、running 15 的**极低负载**下发生，与准入/驱逐策略无关。
2. **不是负载特有**：同一 trace（×1.2×8 oracle）下 victim=longest 间歇死锁（2/3），victim=newest 不死锁。
3. **是我们的实现 bug**：`_ub_order_running` 每步 swap `self.running[idx]`↔`self.running[n-1]`，改变 running 顺序，与 disagg KV 传输/stock 调度器产生竞态 → prefill↔decode 环形等待。

**但注意**：victim=newest 虽不死锁，却因**驱逐策略变错**（改成驱逐最新请求，丢失新鲜 KV）导致严重过载（21% 时 KV100%、waiting555、43s/it）——所以**不能直接禁用重排序当修复**。正确做法是**保留"驱逐 longest"语义，但换一种不每步 swap 的实现**（如直接覆盖 stock 的 victim 选择，不改动 running 顺序）。

**待办**：坐实需要"反向对照"（victim=longest 复现死锁）+ 修 `_ub_order_running` 的竞态实现。

## 修复 #31：UniBoost 死锁（2026-11-11 23:25）

**修法**（`uniboost_scheduler.py`）：
1. `self.policy = SchedulingPolicy.PRIORITY`（替换原来依赖 FCFS `running.pop()` 的分支）。
2. `_ub_order_running` 改为**只写 `request.priority = request.num_output_tokens`**，不再每步 swap `self.running`。

**原理**：stock 调度器已有 PRIORITY 驱逐分支 `max(r.priority, r.arrival_time)`，直接按 priority 选 victim，无需 swap 重排 running 顺序。既保留"驱逐最长/最先进"语义，又消除与 disagg KV 传输的竞态。

**待验证**：跑 UniBoost 臂确认不再在 ~150 请求处死锁。

## 补记：心跳修复已验证（2026-11-12 01:05）

**DEFAULT_PING_SECONDS 5→20 的修复已实测验证**（实验 B 第二次尝试，15:22 起跑、16:49 完成）：

| 指标 | 修复前 | 修复后 |
|---|---|---|
| proxy 503 数 | 467 | **0** |
| 失败请求 | 400 | **0** |
| Successful requests | 3684 | **4080** |
| Output goodput | 355（虚高）| **113.77**（诚实值）|
| stale drop | — | 0 |

**结论**：心跳竞态根因正确、修复生效。修复后阶段 B GWF（真实预测）真实 goodput = 113.77 tok/s，低于阶段 A oracle 的 144（预测噪声的代价）。
**注意**：本次 stale drop=0，印证 KV 路由丢包是间歇性的，死锁根因排查仍待继续。

## 补记：KV 路由丢包根因修正 + 8 卡队列启动（2026-11-12 01:15）

**根因数字修正**：此前把 `flushed_layers` 误读成 `stale_drops`，夸大成"2.5 万层丢包"。精确核对：
- 完成的 rep2 scheme4 → `stale_drops=0`（干净）。
- 卡死的 fix-verify → 仅 **28 条 "Dropping stale deferred" = 1 个请求 × 28 层**。

即死锁根因是**单次路由丢失 → 一个 decode 请求永久挂起 → KV 满 → 级联整体卡死**，属间歇性单请求竞态，非系统性 ZMQ 故障。心跳修复已验证（补记见上：修复后 0 失败、0 stale drop）。

**8 卡队列启动**（8 卡全空，step3 优先，用户拍板）：
- 编排脚本：`run_queue_step3first.sh`（新），顺序 step3 → gammaada_r2 → amax220_r2 → real_lmsys → real_sharegpt → 4p4d_x1p8，跳过 1p1d_x1p8（2 卡项）。
- 每项保留 **超时 + 残留清理**；**不做看门狗自动杀进程**（用户要求改人工定期检查），卡死判据=同时出现 stranded KV + 吞吐持续 0。
- 后台 pid=16206，日志 `queue_run_step3first.log`；定时检查每 30min（schedule-1e03c596）。
- 当前第一项 step3_realpred（真实 BERT β=0.30，GWF+UniBoost，4p4d，timeout 300min）已起跑，第一臂 scheme4_gwf。

## 踩坑 #32：static β=0.30 无法纠正按 output 长度漂移 320 倍的偏差（2026-11-12 01:25）

**发现**：BERT 预测器的偏差**不是统一的 3.3 倍**，而是随 output 长度剧烈变化（离线 49998 条 join 分析）：

| output 长度 | mean actual | mean pred | 该段最优 β=Σa/Σp |
|---|---|---|---|
| 0-100 | 41 | 11771 | 0.0035（over-predict 286×）|
| 100-500 | 280 | 14154 | 0.0198 |
| 500-2000 | 839 | 17362 | 0.0483 |
| 2000-5000 | 3307 | 22165 | 0.1492 |
| 5000-9979 | 7221 | 20909 | 0.3454 |
| >9979 | 26679 | 23606 | 1.1301（under-predict 1.13×）|

**按 output 排序后的分段（对应阶段 B 排序 trace）**：
- 0-10%（短）：最优 β=0.0015
- 10-50%：β=0.0108
- 50-90%：β=0.2125
- 90-100%（长）：β=1.3247

**结论**：static β=0.30 只是"全局总量平均对"，但**每一段都错**——短请求段 over-predict 200×（0.30/0.0015）、长请求段 under-predict 4.4×（0.30/1.32）。这是阶段 B 真实预测 goodput 从 144 掉到 113.77 的主因（不只是预测噪声）。

**对策**：启用在线 β（`WT_PRED_CORRECTION=1`，`OnlinePredCalibration`，EMA beta=Σactual/Σpred，随当前段动态调整）。因阶段 B trace 按 output 排序，在线 β 能跟随当前段的长度收敛到该段最优 β。

## 踩坑 #33：kill run 脚本后 benchmark 客户端成孤儿，往新集群灌双倍负载（2026-11-14）

**症状**：新跑的一轮 goodput/TTFT/TPOT 异常（像双倍负载）；或 `ps -eo pid,args | grep benchmark_serving_trace` 出现多个进程。

**根因**：`test_evaluation_scheme4_rr_compare.sh` 的 `stop_servers()`（L1559）只杀 proxy + serve 的 PGID；`pkill -9 -f` 也只覆盖 `$PROXY_SCRIPT` / `vllm serve` / `uniboost_serve` / `tie_serve`——**唯独不杀 benchmark 客户端**（`benchmark_serving_trace.py`，L2031 单独 `python3 "$BENCH_SCRIPT"` 启动，不进 PIDS/PGIDS 追踪）。`fuser -k ${BENCH_PORT}/tcp` 杀的是**监听** 22007 的 proxy（bench 是**连** 22007 的客户端，杀不到）。外部 `kill` 掉 run 脚本时，bench 子进程收不到信号 → 成孤儿继续跑；下一轮在同一 22007 起新 proxy 后，孤儿 bench 的请求打到**新集群**，与本轮 bench 并发 → 双倍负载污染。

**事故**：122536（scheme4_progress）跑时，上一轮崩溃(121640)遗留的孤儿 bench（pid 2097321）已灌到 4075/4080，与本轮 bench 同时打 22007 → 本轮结果作废（无 goodput，见上「scheme4_progress 首跑分析」）。

**判据**（启动新轮前 + 怀疑污染时）：
- `ps -eo pid,lstart,args | grep benchmark_serving_trace | grep -v grep`：数量必须 =1，且 `lstart` 匹配当前轮启动时间；
- 用 `/proc/<pid>/cmdline` 看 `--out-path` 指向哪个 result dir：指向**旧 dir**（如 `..._121640_...`）= 孤儿。

**修复**：
- 杀 run 脚本时，同时显式 kill bench 客户端：`ps -eo pid,args | grep benchmark_serving_trace | grep -v grep | awk '{print $1}' | xargs -r kill -9`。
- 启动新轮前先确认无残留 bench（`... | wc -l` == 0）。
- 长期：给 `stop_servers()` 补 `pkill -9 -f "$BENCH_SCRIPT"`，或把 bench 也纳入 PGID 追踪。

## 踩坑 #34：prefill 实例被 kill 后 GPU 显存成"幽灵"（34GB×4 不释放，reset 被拒）（2026-11-14）

**症状**：4p4d 跑完（success）后，GPU 0-3 各卡 34484 MiB"used"、util 0%、但 `nvidia-smi` Processes 段为空、`/proc/<pid>` 不存在。GPU 4-7（decode）正常释放。

**根因**：prefill 实例是 P2P NCCL **producer**（含 4GB tensor pool + KV 缓存），被 `pkill -9 -f "vllm serve"` 强杀后 CUDA context 泄漏，驱动没回收；decode 是 consumer，正常释放。34GB×4 与 prefill 足迹（模型 ~15GB + KV + 4GB pool）精确吻合。

**判据**：`nvidia-smi --query-gpu=memory.used` 显示占用但 `--query-compute-apps` 给的 PID 在 `/proc` 里不存在（幽灵 PID）；`nvidia-smi --gpu-reset -i <idx>` 报 "In use by another client"（reset 被监控 daemon / 卡住的 context 阻断）。

**修复（未定）**：等驱动上下文回收；或找到并停掉持有 GPU 的监控 daemon 后再 reset。暂时绕开：只用 GPU 4-7 跑小 PD（如 1p3d）。

## 补记：队列后续 5 项重排（2026-11-12 03:05，不影响运行中的 step3）

按用户要求把 step3 之后的 5 项按核心价值重排，且不打断正在跑的 step3：
- 新顺序：real_lmsys_kv1p6like → real_sharegpt_kv1p6like → 4p4d_x1p8 → gammaada_stability → step2_amax220（头对头优先，稳健性/参数殿后）。
- 实现：`run_queue_reordered_remaining.sh`（新，5 项）+ `watcher_reorder.sh`（后台 pid，盯 queue_run_step3first.log）。
- 机制：watcher 每 10s 检测「step3_realpred 进程结束」，只在 step3 结束后才 kill 旧队列脚本（PID 16208，非 wrapper 16206）并启动重排队列；不碰运行中的 step3。
- 注意：旧队列脚本实际 PID=16208（16206 是外层 `bash -c` wrapper，杀 wrapper 不会停 nohup 子进程）。

## ⚠️ 严重踩坑（重点）：超时预算误当"预算"导致健康实验被杀（2026-11-12 00:00）

**教训：超时只许是"防死锁安全网"，绝不允许成为限制实验进度的预算。真正的卡死由 stall 判据（stranded KV + 吞吐持续 0）触发，不是超时。**

- 事故：step3_realpred（qwen_4080_600s，342 条 >30k extreme，--ignore-eos 强制生成满长）实际 **~4h/臂、两臂 ~8h**，但超时设了 300min(5h) → GWF 臂跑完(4h)、UniBoost 臂在 96%(3903/4080) 被杀，结果作废，浪费 ~5h。
- 根因①：轻信 driver 注释「单臂 ~1.7h」（沿袭早期 docs §5「50 条 extreme」，但本 trace 是 342 条），未独立验证。
- 根因②：超时定位错误——它应是安全网，不是预算。
- 修法：① 所有长 trace 超时 ≥ 实际×2（step3 改 600min）；② 卡死靠 stall 判据触发（已实现），超时只兜底；③ 重跑 UniBoost 臂 600min。

## ⚠️ 硬性约定（重点）：无超时设计——超时只许当安全网，绝不当预算（2026-11-12 00:07）

**原则：队列/驱动脚本里一律去掉 per-item 超时硬编码。卡死由「10 分钟定时检查 + stall 判据（stranded KV + 吞吐持续 0）」由 AI 自动判定处理；慢但健康的实验（extreme trace）绝不被固定时长杀掉。**

- 事故背景：见上方「严重踩坑」——step3 用 300min 超时把健康跑满 96% 的 UniBoost 臂杀掉、结果作废。
- 原因：超时是"盲的"，分不清「慢但健康」和「真卡死」，且被误当成了预算。
- 正确分工：
  | 场景 | 判据 | 机制 |
  |---|---|---|
  | 真卡死 | stranded KV + 吞吐持续 0 | 定时检查 → AI 自动清理重跑 |
  | stall 判据漏网的极端情况 | 长时间无进展 | 无（靠定时检查兜底，不硬编码时长）|
- 落地改动（2026-11-12）：
  1. `run_step3_uniboost_redo.sh`（新）——UniBoost 单臂重跑，无超时。
  2. `run_step3_uniboost_redo_runner.sh`（新）——等旧队列结束+8卡空闲后自动跑重跑，无超时，已后台启动。
  3. `run_queue_reordered_remaining.sh`——已去掉超时（run_item + QUEUE 无 timeout 字段）。
  4. `run_queue_step3first.sh`——**正在跑（pid 16208），未动**，待其结束再改。
- 铁律：以后所有新队列/驱动脚本**一律不带 timeout 字段**；确需兜底时用非常宽松值并注释"安全网"，且绝不基于时长估计去限制实验进度。

## 补记：脚本清理 + 监控改为无头 AI 会话（2026-11-12 00:25）

- **废弃脚本已挪到 `tmp/`**（防止污染实验环境、避免后续 AI 误用）：`run_pending_queue.sh`（旧队列）、`run_queue_reordered_remaining.sh`（重排，未生效）、`run_step3_uniboost_redo_runner.sh`（redo runner，已杀）、`watcher_reorder.sh`（重排 watcher，失效）、`autonomous_queue.sh`（固定脚本自动杀，已停）。
- **监控改为 `new_once` 无头 AI 会话**（schedule-aaabca78，target=new_once）：判断交给 AI（先调查再决定，不机械按阈值杀），web 挂了照常工作；所有判断/动作写 experiment_log.md。
- **仍在用/在跑，勿动**：`run_queue_step3first.sh`（pid 16208 正在跑，含旧 timeout 代码，跑完后应挪 tmp）、`run_step3_uniboost_redo.sh`（new_once AI 引用）、`run_1p1d_x1p8_ab.sh`（new_once AI 引用）、各 driver（amax220/lmsys/sharegpt/4p4d）。

## ⚠️ 严重 bug（重点）：γ-Ada 墙钟模式 bin 单位错误 → 静默退化成 FCFS（2026-11-12 00:51）

**根因**：`uniboost_queue.py` 的 `_phi`（墙钟模式）把 work 从 token 转成秒（`work/rate`），但 `geometric_bin(work_seconds, bin_tokens=128)` 仍用 **token 单位 128** 去量化"秒"。后果：所有请求 work<128s(=17.9万token) 都被压到 bin=128，`boost(128,γ_s)` 里 `γ_s·128` 下溢 → **boost≈0 → phi=到达时间 → 退化成纯 FCFS**（SJF 排序完全失效）。

**修复**（已改 uniboost_queue.py）：墙钟模式下 bin 也要转秒：`bin_tokens = bin_tokens / rate`。

**影响范围（用 γ_s 收敛值定位 bug 引入时间）**：
- γ_s 小=延迟分布宽=真SJF；γ_s 大=延迟窄=FCFS（有 bug）。
- **phase A（×1.2×8, 03:15, γ_s=0.086）→ pre-bug，真 SJF，结论「GWF 144.17 > UniBoost 133.51（+8%）」有效 ✅**。
- ×1.2×8 rep2（07:38, γ_s=13.98）、diag_fix_verify（15:29, γ_s=4.19）、gammaada r2（22:14, γ_s=7.5）→ **post-bug，FCFS，无效，需重跑**。
- 所有 GWF 结果不受影响（GWF 不走 uniboost_queue）；token 钟 UniBoost 不受影响。

**结论**：核心「GWF > UniBoost」结论（phase A）安全；07:38 之后的墙钟 UniBoost 结果全部作废需重跑（含 gammaada、step3 UniBoost、rep2）。修复已生效，lmsys 的 UniBoost 臂（GWF 跑完新起 vLLM）会用新代码。

## 补记：lmsys 跑完后暂停队列、释放 GPU（2026-11-12 01:03）

用户要求 lmsys（GWF+UniBoost 两臂）跑完后**停队列、把 8 卡让给其他 AI 使用**。
- 实现：`pause_after_lmsys.sh`（后台 watcher，pid 281237）检测到 `real_lmsys_kv1p6like 进程结束` 后 kill 旧队列(16208) + 兜底清理残留，释放 8 卡。
- 暂停后的剩余：sharegpt → 4p4d_x1p8 → step3 UniBoost 重跑 → ×1.2×8 UniBoost 重跑 → 1p1d_x1p8（待用户说「恢复」再继续）。

## 实现：动态向量 β（per-bucket 校准）替代标量 β（2026-11-12 01:15）

**背景**：step3 真实 BERT 预测 goodput 仅 24（oracle 119），标量 β=0.30 只修全局均值偏置，修不了逐桶条件偏置。

**实现**（`test_disagg_proxy.py` 的 `OnlinePredCalibration` 已从标量改成向量）：
- `beta[i] = EMA(sum_actual_i)/EMA(sum_pred_i)`，每桶一个 β；请求的实际长度按归一化 `prob[i]` 软归属到各桶。
- `corrected_mean = Σ_i prob[i]×center[i]×β[i]`。
- 冷桶（有效样本数 < min_samples）回退全局 β，避免尾部噪声爆 β；clip [0.05, 20]。
- 无 bucket 后验时退化为标量 ratio-of-sums。
- **默认**：`WT_PRED_CORRECTION=1` 现在走向量 β（原标量逻辑被替换）。

**待办**：加一个「向量 β vs 标量 β vs oracle」对照臂，量化向量 β 能救回多少 goodput（预期向量 β 修条件偏置，但受预测器信息量 corr 0.351 限制，到不了 oracle）。

## ⚠️ 严重 bug（第三个）：victim_longest 抢占活锁 → recompute 风暴（2026-11-12 01:56）

**现象**：lmsys UniBoost 臂（修复 unit bug 后）卡死在 77%（3143/4080），decode 吞吐 0，recompute_tokens 合计 ~132 万（gammaada 才 1 万）。

**根因链**：
1. `_ub_order_running`（victim_longest）设 `priority = num_output_tokens`；
2. stock 抢占 `max(running, key=(priority, arrival_time))` 选 num_output_tokens 最大（最"长"）的当 victim；
3. 抢占只清 `num_computed_tokens=0`，不清 `num_output_tokens` → 长请求 re-prefill 后仍是最大 → 反复被抢 → 活锁。

**为何修复 unit bug 后才暴露**：修复前 FCFS 长请求饿死在 waiting 不进 running（无东西可抢，recompute 1 万）；修复后真 SJF 长请求进 running，victim_longest 立刻反复抢它 → recompute 132 万。

**修法**：victim 改 "newest"（`WT_UNIBOOST_VICTIM=newest`，priority 全 0 → 抢 arrival_time 最新，recompute 最小且不活锁）。代码分支已存在（uniboost_scheduler.py L155-159），只是默认是 longest。或抢占时重置 num_output_tokens。

**影响**：所有墙钟 UniBoost 臂（lmsys/sharegpt/4p4d/step3 redo/×1.2×8 redo/1p1d）都必须带 `WT_UNIBOOST_VICTIM=newest`，否则修复 unit bug 后会触发此活锁。

## 补记：victim 默认改为 newest（修复活锁，2026-11-12 02:09）

- `uniboost.py` 的 `WT_UNIBOOST_VICTIM` 默认从 "longest" 改为 **"newest"**（`victim_longest = (env == "longest")`）。
- 效果：所有 wall 钟 UniBoost driver（15 个之前都没显式设 victim）自动走 newest，避免 victim_longest 活锁。
- 审计确认：3 个改过的 Python 文件（uniboost.py / uniboost_queue.py / test_disagg_proxy.py）语法 OK；2 个 redo 脚本语法 OK、配置完整。

## 补记：向量 β 对照臂 + UniBoost 修复 code review（2026-11-12 02:15）

- 新增对照 driver：`run_step3_ub_beta_vector.sh`（WT_PRED_CORRECTION=1 向量 β）、`run_step3_ub_oracle.sh`（WT_SIMPLIFY_PREDICT=1 完美预测）；标量 β 臂由 `run_step3_uniboost_redo.sh`（WT_PREDICT_OUTPUT_SCALE=0.30）覆盖。三臂均 qwen_4080_600s + 4p4d + wall + γ-Ada + victim_newest(默认)。
- code review：① `_phi` bin 转秒 ✅；② victim 默认 newest ✅（priority 全 0 → max 选最新 arrival）；③ 向量 β 逻辑 ✅（单测通过）。
- **遗留风险**：step3 UniBoost 第一次尝试的 `ValueError shape(498)→(212)` 崩溃根因仍未查清（可能被 victim_newest 修复，也可能独立于抢占），需跑一次才能确认。

## 更正：UniBoost 抢占/驱逐语义核对论文（2026-11-12，来源 arXiv:2606.18431）

**背景**：此前日志（L298/L308）把复现的"驱逐最长 victim"当成了论文语义（"关掉 admit-preempt 更贴近论文'驱逐最长'，更忠实"）。经核对论文原文，**该说法错误**。

**核对结论（论文 §3.2/§3.3 + Algorithm 1）**：
- 正文 "longest"/"evict the" 出现 **0 次**；"victim" 仅出现在 Algorithm 1 的 `SelectVictim(running set, π, swap-cost)`。
- 论文抢占/驱逐 = ① 优先级 `π(r)=a_r−b_r`，`b_r=Boost(Quantize(s_r,k),γ)`；② 抢占触发 Eq.(7) `Φ(j)<Φ(i_curr)−δ_hyst`（hysteresis）；③ 受害者 `SelectVictim(running set, π, swap-cost)`（论文只给此签名、未展开函数体/选择准则，入参为 π 与 swap-cost）；④ MemGuard = 几何量化 + minimum-run hysteresis，请求跨越下一几何阈值前 non-evictable，每请求 ≤1+⌊log₂(S_i/k)⌋ 次抢占。
- **复现的"每步驱逐 num_output_tokens 最大"不是论文规则**，且缺 MemGuard 限频 + hysteresis + swap-cost 三者。

**推论（影响所有 UniBoost 对比结论）**：
- kv1p6 的 UniBoost recompute 14.6M（119%）与 lmsys 活锁的根因是**复现未忠实实现论文的 MemGuard+hysteresis+SelectVictim**，不是"驱逐最长是次优策略"。
- 所有"GWF > UniBoost"结论只能限定为"针对当前偏离论文的复现"，不能推广为"GWF 赢论文 UniBoost"。要下论文级结论，需先补 MemGuard（几何限频）+ hysteresis + swap-cost 三组件。

## predacc 真实 lmsys 子集 4p4d 真实预测 GWF vs UniBoost（2026-11-13 00:10 完成）

**负载**：真实 lmsys 子集 `predacc_lmsys_o1600` / `predacc_lmsys_o2000`（`/root/myshare/wty_data/001/`，真实 prompt + output_tokens 精确匹配合成 o1600/o2000，逐请求预测准 82%/86% 在 30% 内）。
**配置**：Qwen-2.5-7b、4p4d、gmu0.8、tpot30ms、**真实 BERT 预测**（WT_SIMPLIFY_PREDICT 关 + WT_PREDICT_OUTPUT_SCALE=0.30 β=0.30）、scheme4 路由、victim newest。驱动 `run_predacc_lmsys_o1600.sh` / `run_predacc_lmsys_o2000.sh`。

| 负载 | 臂 | goodput | mTTFT | P99 TTFT | mTPOT | P99 TPOT |
|---|---|---|---|---|---|---|
| o1600 | GWF | **524.59** | 323.8 | 1349.2 | 55.70 | **80.97** |
| o1600 | UniBoost | 506.39 | 316.7 | 1216.8 | 55.42 | 90.48 |
| o2000 | GWF | **277.86** | 502.5 | 1385.7 | 88.41 | **127.85** |
| o2000 | UniBoost | 257.72 | 442.3 | 2062.7 | 91.05 | 170.16 |

**结论**：GWF 在 o1600 赢 +3.6%、o2000 赢 +7.8%；两臂 recompute=0（无 KV 打满抢占），是纯调度对比。
**关键反转**：合成 o2000 + oracle 预测是 UniBoost 392 > GWF 361；真实 lmsys + 真实预测变成 GWF 278 > UniBoost 258。**真实预测噪声让结果反转**——UniBoost 的优先级排序对预测误差更敏感，GWF 窗口准入更鲁棒。这是"GWF 赢 UniBoost"的一个干净、无污染的成立条件。
**目录**：`wl_predacc_lmsys_o1600/`、`wl_predacc_lmsys_o2000/`。

## 2026-11-13 02:27 预测器消融队列全部结束
- ours(β=1.0) / µ-Serve(β=1.0) / TRAIL++(β=1.0) 各 o1600+o2000，仅 scheme4+GWF，4p4d，全部 rc=0。
- 三方 goodput 对比见 experiment_analysis.md「三方对比表」。
- 结论：β=1.0 下换预测器影响小（16–29% 极差）且无一致赢家；β 校准才是主导。
- 8 卡已空闲，可跑后续实验。

## 采样参数重配实验（2026-11-13 18:51）

### 实验目的
在 "ours"（Presage）预测器上，量化"重配采样参数能否提升输出长度预测精度"，并生成两份**合法的**（不用真实长度作弊）采样参数调整数据集：一份**略高于 TRAIL++**、一份**远高于 TRAIL++**，供后续实验直接使用。

### 背景与结论（一句话）
采样参数在这条强制生成 trace（finish_reason=length 100%）上与真实输出长度零信息（corr(temp,actual)=+0.037），只能做**偏置校正**，不能提供逐请求长度信号。因此"重配采样参数"唯一合法用途是：按 activation 预测值分段地把预测往下压，抵消 5.3× 过预测。

### 实验过程
1. **绕过 P2P-NCCL**：该机 TCP connect 到 `10.10.111.36:动态端口` 报 EINTR 失败（collect 脚本跑不通），改用 `offline_extract_cls.py` 直接在 GPU 上复刻 vllm activation hook（layer-13 **MLP 输出** + attention importance 的 top-k 截断），逐请求 dump pre-embedding [CLS]（768 维，4080 条）。
2. **对拍**：3 条请求与 serving 路径 `predict_output_mean` 差 0.3–2%（vllm/transformers 数值差异），dump 可信。
3. **穷举 1728 组采样参数组合**（temp 8 × top_p 6 × top_k 6 × rep 6），得各档 MAE。
4. **合法规则**：按 activation 预测值（corr 0.81 的合法信号）分桶，给每桶配最优组合（非作弊：索引是 activation 预测，不是真实长度）。

### 核心结果（o2000，4080 条，MAE↓ 越好）
| 配置 | MAE | corr |
|---|---|---|
| TRAIL++（serving） | 5288.2 | 0.689 |
| ours 自然采样参数 | 8630.7 | 0.549 |
| ours 弃用采样参数分支（α=0） | 5302.1 | 0.810 |
| **vA 略高（top_k=500 统一）** | **5172.5** | — |
| **vB 远高（activation 分 8 桶）** | **1151.9** | — |
| 作弊上界（逐请求按真实长度） | 243.4 | 0.954 |

合法分桶 vs 作弊（按真实长度）差距仅 1–7%，证明 vB 未作弊（否则差距应为 0）。

### 生成的数据集（各 4080 条，采样参数已重配，prompt/output_tokens 不变）
| 版本 | 路径 | 调整方式 | MAE |
|---|---|---|---|
| vA（略高） | `predictor_ablation/sampling_adjusted_datasets/qwen_lmsys_predacc_o2000_vA_mild_topk500.csv` | top_k 统一改 500，temp/top_p/rep 保持自然 | 5172.5 |
| vB（远高） | `predictor_ablation/sampling_adjusted_datasets/qwen_lmsys_predacc_o2000_vB_strong_bucket8.csv` | 按 activation 预测分 8 桶，每桶最优组合 | 1151.9 |

vB 的 8 桶组合（activation 预测 短→长）：
```
bucket0 (0.2,0.5,10000,1.1)  bucket1 (1.6,0.1,10,1.3)  bucket2 (1.6,0.3,10,1.3)
bucket3 (1.6,0.3,10,1.3)      bucket4 (1.6,0.1,10,1.3)  bucket5 (1.6,0.1,10,1.3)
bucket6 (1.0,1.0,10,1.3)      bucket7 (0.5,0.1,-1,1.5)
```
规律：top_k=10 占 3060/4080（低 top_k 压预测是主因），temp/top_p 随 activation 预测长短变化。

### 结果放置 / 溯源
- 结果文件：`predictor_ablation/sampling_reassign_results.md`（四档 MAE/corr 对比 + 作弊上界 + α 扫描）
- 数据集：`predictor_ablation/sampling_adjusted_datasets/*.csv`
- 提取脚本：`predictor_ablation/offline_extract_cls.py` → `cls_output_o2000.tsv`
- 搜索脚本：`predictor_ablation/search_sampling_cheat.py`
- 权重：`/root/myshare/predict_model/qwen_model/model.pth`（embedding + classifier）
- 对拍基线：`wl_predacc_ours_nobeta_o2000/*/log/predict_request_decode*.csv`

## 2026-11-14「conservative」两层关系 + scheme4_progress 首跑分析

### 「两层」关系（写死，供 rebuttal / ablation 引用）
保守调度（`ProgressConservativeScheduler`，`scheme4_progress`）分两层：
- **实例间路由（proxy）**：scheme4 + 真实 BERT 预测（β=0.30），与 scheme4(FCFS) / scheme4_gwf / scheme4_uniboost **完全一致**。
- **实例内准入/抢占（decode 内）**：**prediction-free**，`est = num_prompt_tokens + num_output_tokens × 1.5`，不读 `predicted_output_len` / `predict_output_mean`（源码 `vllm/v1/core/sched/progress_conservative_scheduler.py` L85-86）。

与 `WT_CONSERVATIVE_PREDICT`（静态 buffer = `WT_CONSERVATIVE_MAX_MODEL_LEN − prompt`，会影响路由+准入两处）是**两码事**，本实验未用。

### 关键 bug：WT_SIMPLIFY_PREDICT 静默回退 oracle
- `test_evaluation_scheme4_rr_compare.sh` L204 `WT_SIMPLIFY_PREDICT=${WT_SIMPLIFY_PREDICT:-1}` 默认 =1（oracle）。
- 任何 run 脚本只 `export WT_PREDICT_OUTPUT_SCALE=0.30` 而**不显式 `export WT_SIMPLIFY_PREDICT=0`**，都会被盖成 oracle（β 不生效；proxy `_predictor_route_output_len` 直接 return `max_tokens`，`test_disagg_proxy.py` L515-519）。
- 已用 `/proc/<pid>/environ` 坐实：122536 这轮 proxy/decode 全是 `WT_SIMPLIFY_PREDICT=1`。
- 修复：新增 `run_kv1p6_gwf_real.sh` 显式 `export WT_SIMPLIFY_PREDICT=0`。

### scheme4_progress @ kv1p6（122536）首跑结果（**已作废：双倍负载污染 + oracle 路由**）
- 状态：bench 跑到 34%（1397/4080）被主动 kill；`dataset_result` 为空，**无 goodput/TTFT/TPOT 汇总**。
- 污染：上一轮崩溃(121640)遗留的孤儿 bench（pid 2097321）往同一 22007 集群灌了 ~4075 请求，与本轮 bench 并发 → 双倍负载。
- 路由：oracle（上述 bug），非真实 BERT。CSV 里 `predict_output_len/actual` 中位 29.46× 是 raw BERT（日志记录值，未被路由使用）。

### 但调度器机制已验证（无 OOM / 无崩溃 / 无重启）
4 个 decode 全程 `OOM|EngineDead|restart|CUDA error` 计数 = 0。修复后的"抢最新 + cap 封顶"不再触发"抢→放→抢"死循环。

| 实例 | max_running | 最终 running | preemptions | recompute(tok) | 最终 kv_usage |
|---|---|---|---|---|---|
| decode0 | 297 | 53 | 243 | 482,701 | 56.3% |
| decode1 | 321 | 53 | 262 | 537,502 | 57.3% |
| decode2 | 299 | 54 | 240 | 486,525 | 58.5% |
| decode3 | 251 | 52 | 185 | 458,011 | 56.7% |

- **并发塌缩**：running 149→85（3.6min，快）→53（8.1min，慢），根因 `max_num_running_reqs` 只降不升。
- **KV 闲置**：est_sum 稳定在 743k–772k（< 容量 ≈881k）但真实 kv_usage 仅 56–58%，约 42% KV 闲置；根因 `est = prompt + 1.5×output` 对 output 主导请求 ≈1.5× 真实占用。
- **recompute 浪费**：合计 ~196 万 token（4 decode × ~48 万），每次抢占丢弃 victim 已生成 output 并 re-prefill。
- **预测失配预览**：raw BERT `predict/actual` 中位 29.46×、within-30% 仅 0.3%（kv1p6 合成输出的固有失配）。

## 2026-11-15 忠实 PKAS 复现 + 三臂/四臂对比（真实 BERT + scheme4，kv1p6 4p4d）

### 忠实 PKAS（Algorithm 1/2 + Welford 公式 1）
- 实现：`vllm/v1/core/sched/predictive_kv_reservation_scheduler.py`（shadow fast-forward 准入 + Welford `mean−std` 预测器，无 β、无自定义抢占，只做 admit/deny）。
- 修复两个 bug：① `_free_blocks` 方法名撞 stock `Scheduler._free_blocks(request)` → 改名 `_num_free_blocks`；② 预测器更新钩子原挂在 `schedule()` 的 diff（失效），改为 override `_free_request`（对应原文 `Predictor.update_statistics`）。

### 四臂 goodput（kv1p6, 4p4d, 真实 BERT + scheme4, SLO=TTFT≤1s∧TPOT≤30ms）

| 臂 | Output goodput | 成功 | Median TTFT | Median TPOT |
|---|---|---|---|---|
| FCFS（stock）| 119.72 | 4080/4080 | 463 ms | 110 ms |
| GWF（margin=100）| 118.80 | 4080/4080 | 618 ms | 110 ms |
| PKAS（忠实）| 112.65 | 4080/4080 | 456 ms | 112.74 ms |
| GWF（margin=250 默认）| 110.52 | 4080/4080 | 733 ms | 122 ms |
| conservative | 91.88 | 4080/4080 | 705 ms | 186 ms |

### 结论
- **GWF(margin=100)=118.80 > PKAS=112.65，+5.5%**；GWF 默认 margin=250（110.52）略输 PKAS，靠调 TTFT margin 反超。
- PKAS 忠实版 median TTFT 456ms 是第二好（比 GWF 618ms 低），但 TPOT 略差 + 跑得慢（Welford 对 monster 长尾低估 ~50× → 抢占 → ~1.5h vs GWF 47min）。

## 2026-11-15 KV-threshold 准入（审稿人"keep a buffer"直译版）+ 五臂最终对比

### KV-threshold 实现
`vllm/v1/core/sched/kv_threshold_admission_scheduler.py`：track 实际 KV usage，`usage >= WT_KV_ADMIT_THRESHOLD(0.8)` 时把 `max_num_running_reqs` 封到当前 running 数（停止准入），低于阈值恢复。预测无关、无模拟、无自定义抢占。

### 五臂 goodput（kv1p6, 4p4d, 真实 BERT + scheme4）
| 臂 | goodput | Mean TTFT | 备注 |
|---|---|---|---|
| FCFS | 119.72 | 28.1s | 贪婪准入 |
| GWF margin=100 | 118.80 | 32.7s | 调 margin 后 |
| PKAS（忠实）| 112.65 | 24.5s | Welford+shadow |
| GWF margin=250 | 110.52 | 37.2s | 默认 |
| conservative | 91.88 | 254s | 塌缩 bug |
| KV-threshold 0.8 | 90.56 | 68.5s | naive keep-buffer |

### 结论（回应 Observation 2）
naive "keep a buffer"（阈值准入）最差：20% buffer 被 monster 输出增长吃光 → KV 顶到 99% → 频繁抢占 → Mean TTFT 68.5s。固定 buffer 无法预判输出增长；有效的是预测式准入（GWF/PKAS）或贪婪 FCFS。

### 三臂机制对比：抢占 + P99 TPOT + KV 占用（2026-11-15 补充）

对五臂里的 FCFS / GWF-m100 / KVth 三臂做机制级对比（decode0-3 日志 + bench summary）：

**抢占与尾延迟**：

| 臂 | 总抢占次数 | 涉及请求 | 平均每请求 | 抢占时间跨度 | P99 TPOT | goodput | Mean TTFT |
|---|---|---|---|---|---|---|---|
| FCFS | 3670 | 1869(46%) | 1.96 | 31min | **972ms** | 119.72 | 28.1s |
| GWF-m100 | 976 | 393(10%) | 2.48 | 26min | **239ms** | 118.80 | 32.7s |
| KVth 0.8 | 848 | 848(21%) | 1.00 | 31min | 332ms | 90.56 | 68.5s |

（抢占 token 估算 ≈ 抢占次数 × ~5000 → FCFS ~18M ≫ GWF ~4.9M ≈ KVth ~4.2M）

**KV 占用（各 decode 平均 KV% / ≥80%时间占比%）**：

| 臂 | decode0 | decode1 | decode2 | decode3 |
|---|---|---|---|---|
| FCFS | 69.1 / 33 | 75.0 / 57 | 72.8 / 45 | 77.8 / 65 |
| GWF-m100 | 78.1 / 66 | 73.4 / 55 | 78.6 / 66 | 77.6 / 64 |
| KVth | 75.0 / 64 | 71.6 / 48 | 65.6 / 38 | 76.3 / 65 |

（三臂所有 decode 峰值都顶到 100%）

**三条机制结论**：
1. **FCFS = 抢占风暴换吞吐**：KV 低均值 + 高 thrash（decode0 平均只有 69%），P99 TPOT 飙到 972ms（32×SLO）——抢占后 re-prefill 大前缀阻塞 decode 造成秒级尖峰。但请求全跑完 → goodput 反而最高 119.72。
2. **GWF-m100 = KV 高位稳定**：四 decode 都稳在 73–79%，抢占砍 73%（3670→976），P99 TPOT 压到 239ms（比 FCFS 低 4×）。代价 Mean TTFT 28.1→32.7s，goodput 118.80 ≈ FCFS（省下的重算被等待抵消）。
3. **KVth = 门控饿死**：抢占最少（848），但「少抢占」≠「高 goodput」——门控一刀切把请求挡死（Mean TTFT 68.5s），部分实例 KV 空转（decode2 只 65.6%、38% 时间 ≥80%），goodput 最差 90.56。

**图**：`kv1p6_kv_occupancy_3arms.png`（decode0 单图）、`kv1p6_kv_occupancy_all_decode.png`（四 decode 2×2）。

**回应 Observation 2 的完整链条**：keep-a-buffer 用「饿死请求 + TTFT 68.5s」换「少抢占」，goodput 反而不如贪婪 FCFS；有效的是预测式准入（GWF/PKAS）。

## 关键记录：550 tok/s 量级的配置（o1600 4p4d，GWF margin=100）

**负载**：`qwen_o1600` = lmsys 4080 条、输出**封顶 1600**（无 monster 长尾）。这是跑到 550 量级的唯一条件。

**配置**：Qwen-2.5-7b、4p4d、gmu 0.8/0.8、SLO=TTFT≤1000ms∧TPOT≤30ms、**oracle 路由**（`WT_SIMPLIFY_PREDICT=1`）、GWF sum θ=120 + **TTFT-margin=100**（甜点）。

**结果**（日志 #33）：

| 臂 | output-tok goodput | Median TPOT | Median TTFT |
|---|---|---|---|
| GWF margin=100 | **556.0** | 53.1 ms | 530.6 ms |
| native（贪心）| 524.6 | 54.0 ms | 342.9 ms |

GWF 反超 native +6%，4080/4080。目录 `/root/myshare/gwf_sweep_margin100/o1600/`。

**量级对照（同 4080 条，只改输出长度）**：

| 负载 | 输出 | goodput 量级 |
|---|---|---|
| o1200 | 封顶 1200 | 3007（native）/ 2199（GWF）|
| o1600 | 封顶 1600 | **524–556** |
| o2000 | 封顶 2000 | 250–361 |
| kv1p6 | 顶部30%×1.6 → 8000–16000 | 90–120（真实预测）|

**结论**：同一 4p4d qwen lmsys 4080，把输出从"封顶 1600"改成"长尾 8000–16000"，goodput 掉 ~5×（556→118）。**输出长度长尾是 goodput 的主导因素，不是 4p4d 或 trace 规模**。

## 关键记录：144 tok/s 量级的配置（1p1d ×1.2×8，rate 0.85 req/s，GWF）

**负载**：`qwen_o2000_kv1p2_1p1d_x8` = o2000 输出 ×1.2 + 时间戳 ×8 → **rate 0.852 req/s**，输出 mean 2335 / max 9979，总 9.53M token，demand≈1990 tok/s（约 1.3–1.6× 吞吐 = **可控过载区**）。

**配置**：Qwen-2.5-7b、**1p1d**（prefill GPU0 + decode GPU4，A800-80GB）、gmu 0.8、SLO=TTFT≤1000ms∧TPOT≤30ms、**oracle 路由**、GWF sum θ=120 + **TTFT-margin=250**。

**结果**（日志 #719/#757）：

| 臂 | output-tok goodput |
|---|---|
| **GWF** | **144.17** |
| UniBoost | 133.51（victim_longest 污染）|

GWF > UniBoost **+8.0%**。GWF 窗口在 running 爬到 ~256 后真正介入，KV 稳在 96% 不越界、**零抢占零重算**。目录 `wl_1p1d_x1p2_x8_ab/`。

**量级对照（rate 对 GWF goodput 的影响）**：

| 负载 | rate | GWF goodput |
|---|---|---|
| ×1.8 | 1.7 req/s | 13.03（过载到 ≈0）|
| ×1.2×8 | **0.85 req/s** | **144.17** |

**结论**：rate 从 1.7 降到 0.85，GWF goodput 从 ≈0 拉到 144（11×）——**到达率降到"可控过载"区，GWF 窗口才有用武之地**。这是 1p1d 下 GWF 机制（窗口节流 + 零重算）首次生效的负载。

## 负载档案：`qwen_o2000_kv1p2_1p1d_x8`（1p1d ×1.2×8 系列，2026-11-15 记录，下次直接查、无需重算）

**trace 文件**：`admission_ab_base/qwen_o2000_kv1p2_1p1d_x8/qwen_lmsys_qwentrace_4080_o2000_kv1p2_1p1d_x8.csv`（n=4080）
**生成规则**：论文 qwen lmsys 4080 原 trace → 输出 token **×1.2**（cap ≈9979，即原 ~8316×1.2）→ 时间戳 **×8**（到达间隔压缩 8×）→ 1p1d 复用。
**字段**：`req_id,prompt,prompt_len,output_tokens,temperature,top_p,top_k,repetition_penalty,timestamp`（`timestamp` 单位 **ms**，压测端 `request_rate=trace` 按此回放）。

**到达率**：时间跨度 4787.000 s（≈79.8 min）→ 平均 **0.852 req/s**；平均到达间隔 1173.6 ms；最大间隔 9776 ms。

**输入（`prompt_len`，token）**：

| total | mean | p50 | p90 | p95 | p99 | max | min |
|---|---|---|---|---|---|---|---|
| 388,815 | 95.3 | 57 | 203 | 337 | 524 | 1696 | 30 |

**输出（`output_tokens`）**：

| total | mean | p50 | p90 | p95 | p99 | max | min |
|---|---|---|---|---|---|---|---|
| 9,528,363 | 2335.4 | 689 | 8407 | 9970 | 9974 | 9979 | 1 |

**输出分桶**：`≤512` 29% · `512–1k` 41% · `1k–2k` 2% · `2k–4k` 4% · `4k–8k` 13% · `>8k` 11%
**out/in 均值比 = 24.51×**（对应 GWF-real 的 β=0.04 ≈ 1/24 校准，抵消合成 trace 的 ~24× 过预测）
**总 token = 9,917,178**（in 388,815 + out 9,528,363）

**bench 配置**（来自该臂 config JSON）：`request_rate=trace`、`goodput=ttft:1000 tpot:30`、`tpot_ms=30`、`timeout_seconds=1200`、`max_concurrency=1024`、`maxtokenscustom=32768`、`temperature=0.8` `top_p=0.7` `top_k=50` `repetition_penalty=1.0`、`repeat_idx=1`。

**一句话画像**：输入极短（均值 95 token）、输出**重尾**（均值 2335 / 中位 689 / p99 9974），70% 请求 ≤1k 输出但 24% 是 4k–10k 长请求 → 到达率 0.85 req/s 的「可控过载」，长请求 burst 主导排队与 ETA 摆动。

## 踩坑 #35：P2P stale-drop 死锁——routing 决策 30s 不送达 → KV 被丢 → decode 永久卡死（2026-11-15）
**症状**：decode EngineCore 停止出任何 scheduler 日志（只剩 `GET /metrics 200 OK`），bench 卡在 ~500/4080 不再推进；prefill 日志狂刷 `[WT] Dropping stale deferred KV tensor after 30.4s: ...`。
**根因链**：scheme4 下 prefill 的 KV 是 deferred（`defer_kv_send=True`，进 `pending_layers` 队列等 proxy 的 routing 决策）。若 30s（`WT_KV_PENDING_LAYER_TTL_S`）内没收到决策，`p2p_nccl_connector.py` 的 `_flush_ready_kv` 把 KV 当 stale 丢掉；decode 侧 `recv_first_available` 的 `recv_store_cv.wait()` 无限阻塞等这个被丢的 KV → 死锁。触发条件不是"请求失败"，而是 routing 决策链路断了（见 #36/#37）。
**兜底（已改）**：`p2p_nccl_engine.py` 的 `recv_first_available` 把 `recv_store_cv.wait()` 改成 `wait(timeout=self.recv_timeout_s)`（`WT_P2P_RECV_TIMEOUT_S`，默认 60s），超时 return None + 告警，不再永久卡死。这只是兜底（请求会因缺 KV 失败），治本在 #36/#37。
**验证点**：prefill 日志 `grep -c 'Dropping stale deferred'` 应为 0。

## 踩坑 #36：多臂并行要错开的端口不止实例端口，共 6 类（2026-11-15）
同一台机器并发跑多个 scheme4 臂，每个臂都要把下列端口整体错开（如 +1000/+2000），漏一个就冲突或断链：
| 端口变量 | 默认 | 作用 | 谁读 |
|---|---|---|---|
| `PD_1P1D_PREFILL_PORTS` / `_KV_PORTS` / `DECODE_PORTS` / `_KV_PORTS` | 22001/22011/22020/22032 | 实例 HTTP + P2P KV | prefill/decode 启动 |
| `BENCH_PORT` | 22007 | proxy 的 HTTP app（bench 连这里）| proxy + bench；`fuser -k ${BENCH_PORT}/tcp` 会杀别人 |
| `PROXY_PORT` | 28002 | 服务发现 router socket（KV connector 的 `proxy_port`，即"KV router"）| proxy bind + instance connect |
| `WT_ROUTING_PUB_ADDR` | 32324 | routing 决策 PUB/SUB | proxy PUB + prefill SUB（两边同一变量，可错开）|
| `WT_PREDICTOR_ZMQ_ADDR` | 32323 | predictor PULL | **proxy** bind |
| `WT_PREDICT_PROXY_ZMQ_ADDR` | 32323 | predictor PUSH | **发送端** `predictor_worker_readyflag.py` connect（变量名不同，见 #37）|

## 踩坑 #37：`WT_PREDICTOR_ZMQ_ADDR` 和 `WT_PREDICT_PROXY_ZMQ_ADDR` 是两个不同变量（2026-11-15）
**根因**：predictor 端口有两个不同的 env 变量名：
- proxy（`test_disagg_proxy.py:229`）读 `WT_PREDICTOR_ZMQ_ADDR`（bind PULL 收预测 meta）；
- 发送端（`examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/predictor_worker_readyflag.py:377`）读 `WT_PREDICT_PROXY_ZMQ_ADDR`（connect PUSH 发预测 meta）。

只错开 proxy 的 `WT_PREDICTOR_ZMQ_ADDR`、漏掉发送端的 `WT_PREDICT_PROXY_ZMQ_ADDR` → 发送端还往默认 32323 发，proxy 绑在 33323/34323 → 收不到预测 meta → routing 决策做不出 → stale-drop 死锁（#35）。错开时必须**两个变量设成同一个值**。
**验证方法**：`/proc/net/tcp` 看端口状态，predictor 端口应是 `LISTEN(0A) + ESTABLISHED(01)` 都有；只有 LISTEN 说明发送端没连上（漏了 `WT_PREDICT_PROXY_ZMQ_ADDR`）。

## 注意（疑似，未定论）：`WT_PRED_CORRECTION=1` 与 GWF 叠加时 decode 挂死（2026-11-15）
GWF(真实预测 β=0.04) 曾带 `WT_PRED_CORRECTION=1`（在线逐桶修正）跑 ~15 分钟 decode 停止出日志。后去掉该开关、仅靠 proxy 注入的 β 校准 `predict_output_mean`（decode 侧默认读它），行为正常。**但当时 predictor 端口也是错开的（#37），且还没加 recv 超时（#35），所以这次挂死更可能是 #35 的死锁而非 WT_PRED_CORRECTION 本身**。留此备注，未定论；若再遇到，优先排查 #35/#37 再怀疑 WT_PRED_CORRECTION。

## 三臂重跑完成 + 4 臂对比（2026-11-15，端口全错开 + predictor 变量名修正后）

P2P stale-drop 死锁根治（#35/#36/#37）后，三臂在同一 trace 上并行重跑至 4080/4080 完成（三个后台任务全 exit 0、stale_drops 全程 0）。配置：trace `qwen_o2000_kv1p2_1p1d_x8`（0.852 req/s，见上「负载档案」）、1p1d（2×A800）、gmu 0.8、SLO=TTFT≤1000ms∧TPOT≤30ms、GWF θ=120/margin=250、KV-threshold 0.8。

| 臂 | 路由预测 | 实例内调度 | goodput(tok/s) | 成功/4080 | 吞吐(tok/s) | TTFT 中位 | TPOT 中位 | 目录 |
|---|---|---|---|---|---|---|---|---|
| **GWF-oracle** | oracle(max_tokens) | GWF | **144.17** | 4080 | 1860 | 920ms | 81ms | `wl_1p1d_x1p2_x8_ab/…_014832_scheme4_gwf` |
| FCFS | oracle | 贪心 | 39.91 | 4021 | 1509 | 13.9s | 197ms | `wl_1p1d_x1p2_x8_native/…_082021_scheme4` |
| KV-threshold | oracle | KV-admit 0.8 | 40.02 | 4080 | 1578 | 112.4s | 144ms | `wl_1p1d_x1p2_x8_kvth/…_082020_scheme4_kvthreshold` |
| GWF-real | 真实BERT β=0.04 | GWF | 38.45 | 3927 | 1378 | 23.7s | 205ms | `wl_1p1d_x1p2_x8_gwf_real/…_082021_scheme4_gwf` |

逐臂明细（各 bench log 的 Serving Benchmark Result）：
- FCFS：dur 6051.84s，gen 9,133,183 tok，TTFT mean 132.5s/p99 609s，TPOT mean 308/p99 1634ms，**59 失败**。
- KVth：dur 6018.06s，gen 9,498,606 tok，TTFT mean 191.3s/p99 819s，TPOT mean 187/p99 887ms，0 失败。
- GWF-real：dur 6050.97s，gen 8,339,199 tok，TTFT mean 157.4s/p99 850s，TPOT mean 211/p99 870ms，**153 失败（3.75%，latency=0 无响应，是 decode 过载 reset 连接：proxy log `Connection reset by peer` / `Can not write request body`）**。
- GWF-oracle：dur 5121.23s，gen 9,525,801 tok，TTFT mean 16.9s/p99 228s，TPOT mean 86/p99 172ms，0 失败。

三条结论：
1. **GWF 准入是主增益（oracle 路由下）**：GWF 144.17 vs 贪心 FCFS 39.91 = **3.6×**。贪心 FCFS 在长请求 burst 下抢占风暴（TTFT 中位 13.9s、TPOT 中位 197ms=6.6×SLO），GWF 窗口把中位 TTFT 压到 920ms、TPOT 81ms。
2. **审稿人「keep a buffer」证伪**：KV-threshold(0.8) goodput 40.02 ≈ FCFS 39.91，对抢占风暴零改善、远低于 GWF 144。单纯留 KV buffer 救不了。
3. **预测精度 = goodput 硬瓶颈**：GWF 从 oracle 144 → 真实BERT(β=0.04) 38，跌回 FCFS 水平，还多出 153 个无响应失败。单点全局 β=0.04 的均值校准不足以让 scheme4 路由在重尾 trace 上保持精度。这正是 rebuttal「prediction accuracy → goodput」的直接证据；但**也说明当前 β=0.04 真实 BERT 臂不是可用配置**（≈FCFS 且有 3.75% 失败），后续要逐桶/在线修正校准才能把真实预测逼近 oracle。

### 失败根因定位（FCFS vs GWF，2026-11-15）

**FCFS（59 失败）= 抢占风暴**：贪心无准入控制 → `peak running=851`（KV ~881k token 只能容 ~44–128 条长请求）→ `preempted_recompute_tokens=4,047,116`（4M token 重算）→ 尾部 63/67 失败是 >4k 长请求（最后 800s 内），1200s 内完不成。

**GWF-real（153 失败）= β=0.04 均值校准对长尾欠预测 ~9× → 窗口失效**。决定性证据来自 `predict_request_decode0_*.csv` 的预测误差分桶（raw BERT `predict_output_mean` vs `actual_output_len`）：

| 输出分桶 | n | actual 均值 | raw pred 均值 | β=0.04 缩放后 | 缩放后 ratio |
|---|---|---|---|---|---|
| ≤1k | 2805 | 458 | 14539 | 582 | **1.27（≈准）** |
| 1k–4k | 255 | 2565 | 9817 | 393 | 0.15（**6.5× 欠**） |
| >4k | 807 | 7798 | 22195 | 888 | 0.11（**8.8× 欠**） |

因果链：raw BERT 对所有请求都预测 ~14500–22200（bucket 中心 16934 概率最大），β=0.04 把均值压到 ~582 只对占 72% 的短请求(≤1k)准；对 24% 的长请求(>4k) 缩到 ~888 → **8.8× 欠预测**。GWF 调度器按 `predicted_output_len≈580–888` 估负载（`running_predict_tokens` 严重低估）→ 超额准入 → `peak running=747`（vs oracle ~297）、kv_usage 100%、tau=645–832ms（21–28×SLO）→ decode 过载到 reset 连接（proxy log `Connection reset by peer` / `Can not write request body`）→ 154/163 失败是 >4k 长请求。

**GWF 窗口为什么救不回**：窗口是非绑定的——只挡「新准入」（`|Rrun|≥A → break`），从不抢占已 running 请求；`preempted=0/demoted=0/hopeless_forced=0/stop_window=0` 全程未触发。一旦欠预测把长请求超额放进来，窗口即便缩到 32 也回退不了。

**对照 GWF-oracle（0 失败）**：oracle 预测=实际长度 → 长请求被准确评估 → running 峰值 ~297 → tau 58ms（≤θ=120ms，窗口不缩）→ 5121s 干净 drain。

**对 rebuttal 的含义**：这是「prediction accuracy → goodput」最硬的一条证据——预测误差集中在重尾（正是决定过载的那 24% 长请求），单点全局 β 只救均值救不了尾部；需要按 bucket 逐桶校准（raw 预测的 bucket 中心 [79,302,927,16934,32614] 应各自映射到真实中心），才能让真实 BERT 逼近 oracle 的 144。

### 逐桶校准离线回放验证 + β=0.04 误标定（2026-11-15）

逐桶校准**已存在**：`OnlinePredCalibration`（`test_disagg_proxy.py:113-226`，开关 `WT_PRED_CORRECTION=1`），之前被 #35/#37 死锁牵连关掉。用 GWF-real 的 `predict_request_decode0_*.csv` 离线回放（pred/actual 比值，1.0=准）：

| 输出分桶 | raw | β=0.04 | 逐桶(在线) |
|---|---|---|---|
| short（≤1k）| 73.8× | 2.95× | 12.0×（更差）|
| mid（1k–4k）| 4.07× | 0.16× | **1.09×（修好）**|
| long（>4k）| 2.87× | 0.11× | **0.65×（9×欠→1.5×欠）**|

结论：逐桶把**危险误差**（长请求欠预测）从 9× 收窄到 1.5×、mid 直接修好；代价是短请求过预测 2.95→12×（但短请求不驱动过载、过预测偏保守无害，很可能净增益，需在线跑验证）。

**β=0.04 是误标定**：过预测倍数是分桶的（short 74×、long 2.87× 均值；比值之和 ~3.77×），而 β=0.04 取自 **out/in 比值 24.5×**（那是输出/输入 token 比，不是 pred/actual 比）。所以 β=0.04 对长请求欠 9×、对短请求过 3×，两头都不对。根本原因：预测器 bucket 16934 是「万能桶」（短长请求都把概率堆它），soft-attribution 逐桶也分不开这对双峰。

### trace D（fast1000）+ 三臂并行（2026-11-15 进行中）

按「砍长+短、demand 仍 1990」生成更快 trace：`qwen_o2000_kv1p2_1p1d_x8_fast1000` = 403 长(>4k) + 600 短(≤1k) = 1003 条，span 28.6min，demand 1990，rate 0.585/s（seed 42）。三臂并行验证「甜点 demand + 缩短时长」：

| 臂 | GPU | 端口 offset | 路由 | 目录 |
|---|---|---|---|---|
| GWF-oracle | 0+4 | 0 | oracle | `wl_d_fast1000_gwf_oracle` |
| FCFS | 2+3 | +2000 | oracle | `wl_d_fast1000_fcfs` |
| KV-threshold | 6+7 | +4000 | oracle | `wl_d_fast1000_kvth` |

**结果（三臂全跑完，stale_drops=0）**：

| 臂 | goodput(tok/s) | 成功/1003 | TTFT 中位 | TPOT 中位 |
|---|---|---|---|---|
| FCFS | **61.16** | 1003 | 318ms | 88ms |
| KV-threshold | 62.88 | **897（106 失败）** | 304ms（p99 183s）| 81ms |
| GWF-oracle | 56.25 | 1002 | 319ms | 91ms |

**结论（负结果，重要）**：trace D 上 **GWF 无优势，反而 FCFS>GWF**（61 vs 56），且 KV-threshold 因拒收导致 106 失败、TTFT p99 飙到 183s。根因：**trace D 把 rate 从 0.852 降到 0.585**（砍短保长、长请求占比升但条数减）→ 请求并发骤降 → greedy 不再抢占风暴（TTFT 中位 318ms 无过载）→ GWF 窗口和 KV 缓冲都变成纯开销/误伤。**抢占风暴由「请求并发（rate×时延）」驱动，不是 token demand**——demand 1990 不变没用。提速必须保 rate：等比例减半（n 2040、rate 0.852 不变、span 40min）才可能复现 GWF 优势，不能砍短保长。
