# 分离式 vLLM（disagg serving）实验：跑之前必查

> 跑分离式 vLLM 实验、A/B 准入隔离前必读。

评估脚本：`experiment/zrecent_use/0630/test_evaluation_scheme4_rr_compare.sh`。
`scheme4` = 预测路由 + 贪心准入；`scheme4_gwf` = 预测路由 + GWF 准入——两者唯一差别是准入控制器，
所以 `COMPARE_EXPERIMENT_VARIANTS="scheme4 scheme4_gwf"` 就是"准入隔离 A/B"。
（trace 数据在 `/root/.cache/huggingface/hub/wt_predictor/0630/dataset/evaluation_dataset/0630_evaluation_dataset/`
下的 `qwen`(n=4080) / `mooncake`(n=1790) / `burstgpt`(n=673)。）

## 1. 模型权重路径（脚本默认值是坑）

脚本默认 `QWEN_MODEL_PATH=/root/.cache/huggingface/hub/Qwen-2.5-7b-Instruct`，该目录挂在慢盘
`/dev/sdb1`（15T 机械盘），读 14.8 GB 权重会极慢、看起来像"卡死"。正确路径在 NVMe 快盘：

- `QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct`
- `LLAMA_MODEL_PATH=/root/myshare/models/Meta-Llama-3-8B-Instruct`

（`/root/myshare` 与 `/openrlhf/myshare` 是同一份；`/root/.cache/huggingface/hub` 挂 /dev/sdb1。）

## 2. 代理环境变量会让 proxy 崩（InvalidURL）

`no_proxy`/`NO_PROXY` 里含 `::1` / `[::1]`（IPv6 回环）时，httpx 0.28.1 解析 proxy map 会崩：
`httpx.InvalidURL: Invalid port: ':1]'`，导致 `test_disagg_proxy.py` 的 `fetch_metrics` 死掉、
scheme4 路由拿不到 decode 指标。修复：启动前
`export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"`。

## 3. c10d 卡死（GPU 显存停在 ~498 MiB）

vLLM `get_ip()` 未设 `VLLM_HOST_IP` 时会 UDP 探测 8.8.8.8 取机器 IP（如 `node36.h3c.com`=10.10.111.36）；
若该地址连不通，c10d TCPStore 超时，模型加载卡死在 "Initializing a V1 LLM engine"。
诊断特征：`[c10d] The server socket on [<hostname>]:<随机端口> has timed out, will retry`。
修复：启动前
`export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo`。

## 4. 卡死 vs 加载慢的判据

- GPU 显存停在 **~498 MiB**（CUDA context 之后、权重加载之前）= 卡在分布式初始化（问题 3），不是加载慢；
- 正常加载时显存会爬到 ~15–18 GB（prefill/decode 实例各 ~15 GB）。

## 5. qwen trace 的 extreme 请求会把单次 benchmark 拖到 ~1.7h~4h（⚠️ 见 5b）

`qwen` trace（n=4080）有 50 条 `output_tokens≈32719` 的 extreme 请求，叠加 benchmark 的
`--ignore-eos` 会强制生成满 30k token，单次 benchmark ≈ 1.7h（不是 trace 跨度 600s）。完整 6 次 A/B ≈ 9h。
需要快速扫参数时，换封顶 output（≤8192）或 300s 短 trace。

### 5b. ⚠️ 严重踩坑：`qwen_4080_600s` 有 342 条 >30k extreme，单臂 ~4h（不是 1.7h）

`admission_ab_base/qwen_4080_600s/` 这个 trace **有 342 条 >30k 的 extreme 请求**（比上面 §5 的
`qwen` 50 条多得多），叠加 `--ignore-eos` 强制生成满长，**单臂 ≈ 4h、两臂 ≈ 8h**。
- **绝不许用"1.7h"来给这个 trace 定超时**，超时至少设 **600min**（实际×2）。
- **超时只许是"防死锁安全网"，不是预算**：真正卡死由 stall 判据（stranded KV + 吞吐持续 0）触发。
- 事故记录：step3 用 300min 超时 → UniBoost 臂 96% 被杀、结果作废，浪费 ~5h（见 experiment_log.md 严重踩坑）。

## 6. 停止实验

- 先 `kill -TERM <脚本PID>` 触发脚本的 cleanup trap，再 `kill -9` 残留的 vllm/proxy 进程；
- **不要用 `pkill -f`**：你自己的 shell 命令行里含匹配串（如 "vllm serve"）会被自匹配杀掉。
  改用 `ps -eo pid,cmd | grep -E 'vllm|test_disagg|EngineCore|APIServer' | grep -v grep | awk '{print $1}'` 取 PID 再 kill；
- **benchmark 客户端也要显式杀**：`stop_servers()` 不杀 `benchmark_serving_trace.py`（它连 22007，是客户端不是监听方，`fuser -k` 也杀不到）。孤儿 bench 会在下一轮起新 proxy 后往新集群灌双倍负载（详见 experiment_log.md 踩坑 #33）。
  `ps -eo pid,args | grep benchmark_serving_trace | grep -v grep | awk '{print $1}' | xargs -r kill -9`；
- 验证：`nvidia-smi` 8 张 GPU 显存全部 0 MiB，且 `ps -eo pid,args | grep -E 'benchmark_serving_trace|VLLM|vllm|EngineCore|APIServer' | grep -v grep | wc -l` == 0。

## 7. A/B 迭代：1 rep 快速试错（不要 3 次重复）

准入隔离 A/B 的**迭代阶段**用 `RUN_REPEAT_PER_CONFIG=1`（每方法各 1 次，共 2 次）快速看信号：

- 看 GWF 计数器是否触发（`demoted`/`stop_window`/`preempted` > 0）；
- 看 goodput 是否转正（`scheme4_gwf` vs `scheme4` 的 Δ%）。

找到有提升的负载后，才对该配置补 `RUN_REPEAT_PER_CONFIG=3` 取中位数做最终确认。
（注意："关键指标至少 3 次取中位数"的约定适用于 I/O 压测；disagg A/B 迭代先用 1 次快速试错。）

## 8. GPU 资源分配优先级（共享服务器）

在共享服务器上做实验，卡资源分配遵循：

1. **8 卡能空出来 → 优先跑 8 卡（4p4d）实验**；否则跑能分到卡的实验（如 1p1d 只用 2 卡：prefill 1 卡 + decode 1 卡）。
2. **8 卡队列已启动**（`run_queue_step3first.sh`，2026-11-11 起）：step3 → gammaada → amax220 → real_lmsys → real_sharegpt → 4p4d_x1p8。
   - step3 的 **GWF 臂已 100% 完成**；**UniBoost 臂因 300min 超时在 96% 被杀**，由 **new_once 无头 AI 会话**（schedule-aaabca78）在队列结束后重跑 `run_step3_uniboost_redo.sh`（**无超时**，见 §5b）。
3. 剩余待办：`1p1d_x1p8`（2 卡，按 8 卡优先规则让位）。
4. **铁律：队列/驱动脚本一律不带 timeout 硬编码**（超时误杀过健康实验，见 §5b 严重踩坑）；卡死靠 10 分钟定时检查 + stall 判据人工处理。

> 8 卡是否空闲的判据：`nvidia-smi` 8 张 GPU 显存全 <2 GiB，且无 22xxx 端口占用，且无 `VLLM::EngineCore` / `benchmark_serving_trace` 进程（连续两次 30s 确认）。

> **踩坑详细记录统一放在 `experiment/zrecent_use/0630/experiment_log.md` 的「踩过的坑」**
> （含速查表 + 按时间编号的完整条目）。本文件只保留上面 1–9 条"跑之前必查"。

## 9. 实验循环与监控流程（agent 操作约定）

### 9.1 实验闭环
`执行 → 每10分钟监控 → 单项完成总结（结果+踩坑）→ 进入下一项 → … → 队列全部结束 → 汇总结论报告`。

### 9.2 定时监控（schedule，active）
- 每 10 分钟一次，检查：当前项/arm、进度、卡死信号。
- **卡死判据**：stranded KV（`Dropping stale deferred KV tensor`）+ 活跃实例 `Avg generation throughput` 持续 0，**两者同时满足才判卡死**；extreme 长请求吞吐恒 >0 不误判。
- **异常处理**：resume goal → 连续定位根因（prefill/proxy 日志、stale drop 触发请求、GPU 显存/进程状态、心跳 503）→ 修复 → 清理进程树重跑 → pause goal，根因+修复记录到 `experiment_log.md`。
- **单项完成**：总结 goodput / SLO / 失败请求 / recompute / stale_drop + 踩坑写入 `experiment_log.md`，确认进入下一项。
- **队列全部结束 + 8 卡空闲**：提醒用户可跑 colocated 观测实验（见 `colocated_obs_design.md`），等确认再跑。

### 9.3 两个机制互补（不要重叠）
| 机制 | 状态 | 职责 |
|---|---|---|
| 定时任务 schedule | active | 例行监控 + 检测异常 + 完成总结 |
| 持久目标 goal | paused | **仅异常时 resume** 驱动连续排查修复，完成后 pause |

> goal 绝不用来"周期监控"——它会每轮无间隔触发（曾因此打扰用户）。监控只靠 schedule。

### 9.4 无超时原则（见 §5b）
队列/驱动脚本一律不带 timeout 硬编码；慢但健康不杀，卡死靠 §9.2 判据由 AI 自动判定处理。

### 9.5 文档同步
每个里程碑更新 `experiment_log.md`（TODO 状态 + 结果/踩坑）+ 本文件，保持与实验进度同步。
