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
| 6 | 11-06 | dense_short（12000 短×200，300s 内 ~40 req/s） | 完美预测 | 跑中 | 待测 | 换成"高并发短请求"撞悬崖。离线模拟预测：greedy running→2974（悬崖）、GWF gate 触发（stop_window=3376、demoted=9920、window 256→32），但**两者 goodput 都≈0**（256 并发 decode tpot 97ms→客户端 ~107ms > 30ms SLO） |

## 结论快照（随迭代更新）

- **KV 容量是并发上限（关键新认知）**：单 decode 实例 GPU KV=**881,616 tokens**（decode0 日志 kv_cache_utils）。并发 running ≤ 881616/(prompt+output)。所以：
  - 要触发 demote（running≥A/2=128）→ output ≤ ~6800；
  - 要触发悬崖/stop_window（running≥A=256）→ output ≤ ~3400。
  - 之前的 8000/20000 长请求分别把并发卡在 ~107/~44，**根本到不了触发阈值**——这就是 GWF 计数器全 0 的根因。
- **GWF 只在两个过载区出手**：并发过载（running≥256→stop_window + demote）和 KV 过载（KV 满→preempt+eV）。当前 8000/20000 trace 是"KV 受限"但并发不到 128，两者都不触发。
- **触发≠有 goodput 增益**：256 并发下 decode tpot ~97ms（旧 4080 数据）→ 客户端 ~107ms > 30ms SLO，goodput 仍归零。真正要赢需要**振荡负载**：短暂高压触发 GWF 甩载 + 随后的低并发期服务短请求（低并发时短请求客户端 tpot ~28ms、49% good）。
- **负载热点不平衡（3×）**：实测 decode0/1/2/3 running 峰值 61/28/96/18，最热实例 ~3× 平衡值——predictor 路由把请求集中到单个 decode 实例，反而**降低**触发门槛（热实例先到 128）。
- **预测 miscalibration**（系统 8–13× 过预测）仍是独立短板；已用 `WT_SIMPLIFY_PREDICT=1` 隔离。

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
