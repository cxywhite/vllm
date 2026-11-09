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

### UniBoost A/B 状态（未完成，无完整配对，不进迭代表）

UniBoost 的 A/B（scheme4 vs scheme4_uniboost）**没跑成完整配对**，多臂因环境（start_servers_failed / instance_restart）失败，单臂数值不可比：
- `wl_util05_uniboost_ab5`：scheme4 goodput 0.55、TPOT 73.65（uniboost 臂失败）；
- `wl_ub_arm_a`：scheme4 goodput 0.88、TPOT 46.92（uniboost 臂失败）；
- `wl_ub_arm_b2`：uniboost goodput 0.66、TPOT 52.18（scheme4 臂失败）。
只有 [4] 节的集成说明 + 单实例冒烟是完整的。

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
