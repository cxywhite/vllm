# AGENTS.md

本仓库是 vLLM 调度 / 预测相关的实验代码（`predict-schedule`）。
以下是给编码代理的约定，**新增脚本前请先读这一节**。

## 测量与压测脚本的约定

### 1. 变量名必须带单位（最重要）

单位错误在性能测试里极其常见，而且**不会报错，只会让结论系统性偏差**。
把单位写进变量名，比写在注释里可靠得多。

```python
# 反例：看不出是字节还是 MiB，也看不出 MB/s 是十进制还是 MiB/s
bw = 140
size = 4096
speed = total / elapsed / 1e6

# 正例
bw_mb_decimal = 140.0        # 140×10^6 B/s
size_bytes    = 4_194_304
size_mib      = 4.0
speed_mb_s    = bytes_done / elapsed_s / 1e6     # 十进制 MB/s
```

规则：

- 字节数用 `_bytes` 后缀；MiB 用 `_mib`；十进制 MB 用 `_mb_decimal` 或 `_mb_s`；
- 时间统一 `_s` / `_ms`，换算处显式写出 `* 1000`；
- **同一份结果里只允许一种 MB 口径**，跨脚本对比前先确认口径一致；
- `1 MiB = 1.048576 MB`，混用会造成 **4.86%** 的系统性偏差——足以让"拟合 R²"和"模型误差"失真。

> 2026-10-21 的实际事故：拟合用 `size/1048576`（MiB），吞吐列用 `bytes/1e6`（十进制 MB），
> 结果整张"传输耗时"表偏低 4.9%，16M 行的模型误差从 0.1% 被放大到 4%，还导致
> "拟合带宽 218 还是 237"的争论。

### 2. 每次 I/O 后校验返回值

`preadv` / `pwritev` 的返回值必须等于请求长度，否则说明越界或短读，**不能当作成功计入吞吐**。
`block4.py` 曾因顺序测试用 `O_TRUNC` 截断文件、随机偏移仍按 8 GiB 计算，
越界读被立即返回 0，测出"机械盘 4K 随机读 13,443 IOPS、延迟 0.001 ms"的假数据。

### 3. 结果里必须记录运行环境

至少记录 `loadavg`（**load1 和 load5 都要**）、主机名、时间、测试参数、随机种子。
共享主机上 `load5` 包含 D 状态（不可中断 I/O）线程，比 load1 更能解释延迟波动：
同一天两次运行 load1 都是 ~40，但 load5 为 180 vs 40，4K 随机读延迟正好差 2 倍（12.28 vs 6.13 ms）。

### 4. 关键指标重复测量

每项至少 3 次取中位数，并给出 min/max。单次测量在共享机器上没有区分能力。
同时注意慢盘随机写会被盘内缓存吞掉：判据是 **"O_DSYNC 随机写如果远快于随机读，说明没真正落盘"**
（实测 4K 随机写 4,962 IOPS vs 随机读 163 IOPS，差 30 倍，且两个脚本对同一操作差 7.7 倍）。

### 5. 压测文件与跨度要足够大

测试文件过小会掩盖寻道代价；顺序测试跨度小于盘内缓存（数百 MB）会被缓存放大。
`block4.py` 的顺序段只覆盖 128/512 MB，属于已知的不确定度来源。

## 其他约定

- 跑重型 I/O 压测用 `ionice -c2 -n7 nice -n19`，并在结果里注明；
- 压测脚本的产物写到 `/root/myshare/` 下（该目录对宿主机可见），不要写进仓库；
- 文档中的每个数字都应能追溯到某个 JSON 字段或某条命令的输出。

## 分离式 vLLM（disagg serving）实验：跑之前必查

评估脚本：`experiment/zrecent_use/0630/test_evaluation_scheme4_rr_compare.sh`。
`scheme4` = 预测路由 + 贪心准入；`scheme4_gwf` = 预测路由 + GWF 准入——两者唯一差别是准入控制器，
所以 `COMPARE_EXPERIMENT_VARIANTS="scheme4 scheme4_gwf"` 就是"准入隔离 A/B"。
（trace 数据在 `/root/.cache/huggingface/hub/wt_predictor/0630/dataset/evaluation_dataset/0630_evaluation_dataset/`
下的 `qwen`(n=4080) / `mooncake`(n=1790) / `burstgpt`(n=673)。）

### 1. 模型权重路径（脚本默认值是坑）

脚本默认 `QWEN_MODEL_PATH=/root/.cache/huggingface/hub/Qwen-2.5-7b-Instruct`，该目录挂在慢盘
`/dev/sdb1`（15T 机械盘），读 14.8 GB 权重会极慢、看起来像"卡死"。正确路径在 NVMe 快盘：

- `QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct`
- `LLAMA_MODEL_PATH=/root/myshare/models/Meta-Llama-3-8B-Instruct`

（`/root/myshare` 与 `/openrlhf/myshare` 是同一份；`/root/.cache/huggingface/hub` 挂 /dev/sdb1。）

### 2. 代理环境变量会让 proxy 崩（InvalidURL）

`no_proxy`/`NO_PROXY` 里含 `::1` / `[::1]`（IPv6 回环）时，httpx 0.28.1 解析 proxy map 会崩：
`httpx.InvalidURL: Invalid port: ':1]'`，导致 `test_disagg_proxy.py` 的 `fetch_metrics` 死掉、
scheme4 路由拿不到 decode 指标。修复：启动前
`export no_proxy="localhost,127.0.0.1" NO_PROXY="localhost,127.0.0.1"`。

### 3. c10d 卡死（GPU 显存停在 ~498 MiB）

vLLM `get_ip()` 未设 `VLLM_HOST_IP` 时会 UDP 探测 8.8.8.8 取机器 IP（如 `node36.h3c.com`=10.10.111.36）；
若该地址连不通，c10d TCPStore 超时，模型加载卡死在 "Initializing a V1 LLM engine"。
诊断特征：`[c10d] The server socket on [<hostname>]:<随机端口> has timed out, will retry`。
修复：启动前
`export VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo`。

### 4. 卡死 vs 加载慢的判据

- GPU 显存停在 **~498 MiB**（CUDA context 之后、权重加载之前）= 卡在分布式初始化（问题 3），不是加载慢；
- 正常加载时显存会爬到 ~15–18 GB（prefill/decode 实例各 ~15 GB）。

### 5. qwen trace 的 extreme 请求会把单次 benchmark 拖到 ~1.7h

`qwen` trace（n=4080）有 50 条 `output_tokens≈32719` 的 extreme 请求，叠加 benchmark 的
`--ignore-eos` 会强制生成满 30k token，单次 benchmark ≈ 1.7h（不是 trace 跨度 600s）。完整 6 次 A/B ≈ 9h。
需要快速扫参数时，换封顶 output（≤8192）或 300s 短 trace。

### 6. 停止实验

- 先 `kill -TERM <脚本PID>` 触发脚本的 cleanup trap，再 `kill -9` 残留的 vllm/proxy 进程；
- **不要用 `pkill -f`**：你自己的 shell 命令行里含匹配串（如 "vllm serve"）会被自匹配杀掉。
  改用 `ps -eo pid,cmd | grep -E 'vllm|test_disagg|EngineCore|APIServer' | grep -v grep | awk '{print $1}'` 取 PID 再 kill；
- 验证：`nvidia-smi` 8 张 GPU 显存全部 0 MiB。

### 7. A/B 迭代：1 rep 快速试错（不要 3 次重复）

准入隔离 A/B 的**迭代阶段**用 `RUN_REPEAT_PER_CONFIG=1`（每方法各 1 次，共 2 次）快速看信号：

- 看 GWF 计数器是否触发（`demoted`/`stop_window`/`preempted` > 0）；
- 看 goodput 是否转正（`scheme4_gwf` vs `scheme4` 的 Δ%）。

找到有提升的负载后，才对该配置补 `RUN_REPEAT_PER_CONFIG=3` 取中位数做最终确认。
（注意："关键指标至少 3 次取中位数"的约定适用于 I/O 压测；disagg A/B 迭代先用 1 次快速试错。）

### 8. 死锁看门狗会误杀 KV-bound 的慢跑（默认已关）

旧判据只看 decode GPU `utilization.gpu`==0 持续 300s，但 **KV-bound 忙等时 util 也是 0**，
曾把已跑到 3111/4080 的 qwen_cap scheme4 杀掉。现在：
- 默认关闭，`WT_ENABLE_GPU_DEADLOCK_GUARD=1` 才启用；
- 改用**进度判据**：bench log 的 tqdm `done/total` 在 `WT_PROGRESS_TIMEOUT_SECONDS`（默认 1800s）
  内不动才判死。
- 一句话：**"util=0 是正常现象，完成数不动才是真死"**。

### 9. P2P NCCL resumed-request 死锁（vLLM bug，已 patch `cef34ada1`）

P/D 分离下 decode 请求被抢占后重新准入，`start_load_kv → recv_first_available` 会**无限阻塞**
（prefill 早已完成该请求并释放 KV，不会再发）。症状：decode EngineCore 卡在 `recv_first_available`，
GPU util=0，而 prefill 栈是 `_process_input_queue → queue.get`（空闲）。
诊断：`py-spy dump --pid <EngineCore_pid> --nonblocking`。
修复：`get_num_new_matched_tokens` 里 `preempt_count>0 → return 0, False`（decode 本地重算，方案 B）。
注意：方案 B 会把重算算力压在 decode（瓶颈）上；论文口径更理想的是**被抢占请求回 prefill 重算**（方案 A）。

### 10. 清理必须包含大写 `VLLM::EngineCore`

kill 时 grep 用小写 `vllm` 匹配不到 `VLLM::EngineCore`（大写），旧引擎会继续占
`tcp://127.0.0.1:22032` 等 KV 端口 → 新轮 `zmq.error.ZMQError: Address already in use` → 启动失败循环。
清理后必须验证：`ss -ltn | grep ':2203[2-5]'` 为空 + `nvidia-smi` 全 0 MiB。

### 11. GitHub 推送被假阳性 secret 拦

trace CSV 的 prompt 里含 `sk-...` 文本会被判成 OpenAI Key，阻塞整条分支推送。
修复：对 `admission_ab_base/**/*.csv` 等长替换 `sk-`→`sk_`，`git filter-branch --tree-filter`
重写 `origin..HEAD` 全部未推送提交，推送前逐 commit `git grep -l 'sk-' <c> -- '*.csv'` 验证。

### 12. KV thrash：trace 总量 > KV 总量

单 decode 实例 KV ≈ 881,616 token（util=0.8）。若 trace 总输出 > 实例数×KV，必然"抢占+重算"慢爬
（`kv_usage_pct≈100%`、`preempted_recompute_tokens` 涨、进度 ~0.27 req/s），**不是死锁**。
对策：把 trace 缩到 KV 容量内（总输出 ≤ ~3M），或接受慢爬只测"谁先卡死"。

### 13. 3P4D 的 GPU 分配（共享机）

GPU 0 可能被别的租户占用（`nvidia-smi` 的 PID 不在自己 `ps` 里）。3P4D 用
`PD_RATIO_LIST="3:4"`，配置为 **prefill=GPU 1,2,3 / decode=GPU 4,5,6,7**（7 卡）。
启动前先 `nvidia-smi --query-compute-apps=pid,used_memory` 确认空闲卡。

## 回滚安全点（git）

已建立并推送到远程的回滚点（记录于 2026-11-07）：

| 仓库 | remote | 分支 | commit | 内容 |
|---|---|---|---|---|
| 主仓库 `/root/predict-schedule` | `cxywhite/vllm` | `experiment/20260630` | `a0bdf55` | worked_example 脚本 + workload 假设 + 双峰 trace 构造器 |
| vllm 子模块 `/root/predict-schedule/vllm` | `cxywhite/Custom-vllm` | `v0.11.0-local-v2` | `f3296d1d1` | 简化预测（`WT_SIMPLIFY_PREDICT`）+ GWF/原生 scheduler 都记 `preempted_recompute_tokens`（重算代价）|

回滚命令（**丢弃未提交改动**，先 `git status` 确认不要了再执行）：

```bash
# 主仓库
cd /root/predict-schedule && git checkout experiment/20260630 && git reset --hard a0bdf55
# vllm 子模块（必须单独 checkout，见下）
cd /root/predict-schedule/vllm && git checkout v0.11.0-local-v2 && git reset --hard f3296d1d1
```

注意：

- 主仓库 @ `a0bdf55` 里记录的 vllm gitlink 是 `8669f4d02`，**不是** `f3296d1d1`，
  所以 `git submodule update` 不会给到 f3296d1d1，必须按上面单独 checkout 子模块；
- 两仓库当前都有未提交改动（主仓库删掉的 `bandwith_test/raw_trace_stats_20260629/` 图与 log；
  vllm 的 AIMD→GWF 迁移未提交改动），`reset --hard` 会清掉它们；
- 两 commit 均已推送：`origin/experiment/20260630`、`origin/v0.11.0-local-v2`，本地丢了也能从远程拉。
