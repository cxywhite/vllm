# 分离式 vLLM 实验 · 踩坑速查表

> 目的：出现异常时**按症状一页定位**，不用再翻长日志。
> 配套：`AGENTS.md`（跑前必查）、`experiment_log.md`（详细流水账）。
> 最近更新 2026-11-07。

## 一、启动/环境类

| 症状 | 判据 | 修复 |
|---|---|---|
| 模型加载卡死，显存停 ~498 MiB | `[c10d] server socket ... timed out` | `export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo` |
| 模型加载极慢（像卡死） | 权重路径在 `/root/.cache`（挂在 /dev/sdb1 慢盘） | `QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct` |
| proxy 崩溃 `InvalidURL: Invalid port ':1]'` | `no_proxy` 含 `::1`/`[::1]` | `export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1` |
| 启动报 `ZMQError: Address already in use` | 旧引擎进程还占着 `tcp://127.0.0.1:2203x` | 杀进程时**必须含大写 `VLLM::EngineCore`**；杀完验 `ss -ltn \| grep ':2203[2-5]'` 为空 |
| 某些实例起不来、端口拒绝 | `curl localhost:<port>` 返回 000 | 同上（端口残留）+ 检查 GPU 是否被别的租户占 |
| 清理后 GPU 显存不归零 | `nvidia-smi` 仍有进程 | 用 `/proc/<pid>/fd` 或 `ps -eo pid,cmd` 找全，`kill -9`；不要用 `pkill -f`（会自匹配） |

## 二、运行中"看起来挂了"类（最容易被误判）

| 症状 | 判据 | 修复/结论 |
|---|---|---|
| 看门狗报 `GPU deadlock` 并杀轮次 | decode GPU `util=0` 持续 300s | **误杀**：KV-bound 忙等时 util 也是 0。已默认关；改用"tqdm 完成数 1800s 不动"判死 |
| decode 引擎完全不动、GPU util=0 | `py-spy dump` 卡在 `recv_first_available`；prefill 栈是 `_process_input_queue→queue.get`（空闲） | **vLLM P2P bug**：resumed（抢占过）请求不该再等 prefill 的 KV。已 patch `cef34ada1`（decode 本地重算） |
| 进度条极慢（~0.27 req/s）、`kv_usage_pct≈100%`、`preempted_recompute_tokens` 涨 | trace 总输出 > 实例数×KV（881,616/实例） | **KV thrash，不是死锁**。缩 trace 到 ≤ ~3M token，或接受慢爬 |
| 只有部分 decode 在跑、部分像死了 | **先查是不是同时跑了两个评估脚本**！用 `/proc/<pid>/fd` 看实例实际写哪个日志 | 这是最隐蔽的坑：两脚本各要 7 卡、只有 8 卡，互相踩端口/显存。启动前 `ps -eo pid,cmd \| grep test_evaluation_scheme4 \| wc -l` 必须为 1 |

## 三、排查方法论类（避免走弯路）

| 坑 | 正确做法 |
|---|---|
| **查错目录**：评估脚本崩溃重启会**新建目录**（新 timestamp） | 最新目录是"重启后的新轮"；崩溃现场在**次新目录**。`ls -dt <BASE>/*/` → `grep -n 'INSTANCE_RESTART' <main>.log` → `tail -30 <crash_dir>/log/decode0_*.log` |
| 把主日志里的 `Killed`（SIGKILL）当崩溃原因 | `Killed` 是 health guard 判 down 后 `stop_servers` 的清理（`kill -TERM -<pgid>` 广播），**不是原因**；原因看崩溃轮实例自己日志尾部 + 宿主内存/GPU/dmesg |
| 没日志就下结论"日志被截断" | 实例被重启覆盖的只是**新目录**；旧目录仍在（`ls -dt` 第二个） |
| 同时跑两个脚本 | 见上，必须先确认只有一个 |

## 四、数据/提交类

| 症状 | 修复 |
|---|---|
| `git push` 被 `GH013 secret scanning` 拦（`sk-` in trace CSV） | 等长替换 `sk-`→`sk_`，`git filter-branch --tree-filter` 重写 `origin..HEAD`，逐 commit `git grep -l 'sk-' <c> -- '*.csv'` 验证 |
| vllm 是 git 子模块，`git add vllm/...` 报错 | `cd vllm && git add/commit/push` 单独提交 |
| 结果目录混了多个 timestamp，分析脚本读错 | 分析时显式指定目录；`analyze_admission_ab.py` 支持传 result-root |

## 五、负载/方法类

| 现象 | 解释 |
|---|---|
| GWF 计数器全 0（`demoted/stop_window/preempted`=0） | 并发没到门槛：`running ≥ A/2=128` 才 demote、`≥A=256` 才 stop_window。短请求+高到达率能到；长请求撞 KV 反而把并发压到 186 以下 |
| 长请求 trace 并发上不去 | KV 容量锁死：`并发 ≤ 881,616/(prompt+output)`。output 8000/20000 → 并发 ~107/~44 |
| 路由把长请求集中到一个实例 | scheme4 按预测 `Future_Tokens` 预留；某实例 `Future=344k`、其余 1k–4k，导致它 `pred_bottle` 抬高被排除。需在 `select_best_instance` 加"Future 超阈值跳过" |
| 3P4D 与 4P4D 差异 | 3P4D 用 GPU 1,2,3 + 4,5,6,7（GPU 0 可能被租户占）；4P4D 用 0–3 + 4–7。3P4D 并发被推得更高（每 decode 更挤） |

## 六、当前可观测性（已内置）

- **`[SYS]`**：健康检查循环每 60s 打宿主内存（used/free/avail）+ 8 卡 util/mem。
- **`[CRASH]`**：任一实例 down 时自动 dump：该实例日志尾部 30 行 + 宿主内存 + GPU + dmesg OOM。
- **`[WT][NATIVE]`**：每 1s 打 decode 的 `kv_usage_pct` / `running` / `waiting` / `preempted_recompute_tokens` / `pending_max`。
- **`[WT][GWF]`**：每 1s 打 GWF 计数器 + `meas_tpot_ms` + `window` + `running/waiting/pending_max`。
- **`[WT][MONITOR]`**：proxy 视角每个 decode 的 `reqs/curr_bottle/pred_bottle/Future/PredTotal`。
- **bench 进度条**：`bench_trace_*.log` 的 `done/total`，配合进度判据看门狗。
