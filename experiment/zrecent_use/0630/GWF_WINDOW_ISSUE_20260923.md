# Ours (scheme4_gwf) 结果崩塌：GWF 准入窗口死锁分析（2026-09-23）

数据集：`tmp_dataset3` / Qwen-2.5-7B-Instruct / 4P4D / TTFT 1000ms / TPOT 30ms / 4080 请求

## 一、现象

| run | pool | goodput | Median TTFT | Mean TPOT | Median TPOT |
|---|---|---|---|---|---|
| Ours 9/23 | 8 | **0.12** | **419,138 ms** | 30.88 ms | 31.51 ms |
| Ours 8/27 | 32 | 0.34 | 157 ms | 51.25 ms | 42.81 ms |

请求级分布（`dataset_result/test_results_*.csv`，时间单位是**秒**）：

- 9/23：`ttft p10=0.15s p50=419.62s p90=1379.74s p99=2432.84s`，`>60s` 有 **2818** 条，`>300s` 有 2362 条
- 8/27：`p50=0.16s p90=0.25s`，`>60s` 只有 **4** 条

即：TTFT 崩塌、TPOT 反而很好（并发低 → 单请求快），是典型的"准入被卡住导致排队"形态。

## 二、机制（有日志证据）

日志格式：`[WT][GWF] steps=.. admitted=.. window=.. running=.. waiting=.. meas_tpot_ms=.. budget_ms=30.00`

统计 decode0（约 3400 条 GWF 采样）：

| 窗口状态 | 样本数 | 平均实测 TPOT |
|---|---|---|
| `window <= 2` | 2357 | **32.18 ms**（其中 2325 条 > 30ms） |
| `window > 500` | 587 | **17.69 ms** |

排队情况：有排队的样本 11316 个，平均 `window=3.8`、`running=19.8`、`waiting=220.4`；其中 **87%（9815）的排队样本 `window<=2`**。

结论：窗口被压小后，解码 batch 变小且被 prefill 频繁穿插，墙钟/令牌被抬到刚过 30ms 预算；而扩张分支要求 `tp <= budget`，永远不成立，于是收缩分支不断把它钉到下限 1 —— **正反馈死锁**，请求在队列里等几百秒。

## 三、代码位置

- `vllm/vllm/v1/core/sched/geometric_window_scheduler.py:586-593` — 每秒一次的窗口更新（收缩 `max(1, A // 2)` / 扩张 `A + max(1, A // 8)`）
- `:228-256` `_measure_tpot_ema()` / `:207` `_observed_tpot_s()` — 用 **episode 起点至今的墙钟时间 ÷ 生成长度**，包含了等待 KV、被抢占、被 prefill 穿插的时间
- `:617-624` — 准入判断：`len(self.running) >= self.admission_window` 就 `break`，计入 `stop_window`
- `:97` — 初始窗口 = `max_num_running_reqs`

## 四、修改选项（建议先小改验证）

1. **滞回**：扩张条件放宽为 `tp <= budget * 1.15`（避免在预算边缘反复收缩）。
2. **保底窗口**：窗口设下限（如 `max_num_seqs // 8`）；并在 `waiting > 0 且 window < len(running)` 时强制扩张。
3. **修正测量口径**：`_observed_tpot_s` 只累计连续 decode step 之间的时间，剔除等待 KV / 被抢占区间，否则窗口越小测得越差。
4. 建议顺序：先做 1+2（改动小、可快速验证），仍不稳定再加 3。

## 五、验证判据

- 稳定性：`rg -o 'window=[0-9]+' <run>/log/decode0_*.log | sort -n | uniq -c`，窗口不应长期停留在 ≤2
- 效果：请求级 TTFT 的 p50 回到亚秒级；goodput 应回到 0.3 以上（参照 8/27 的 0.34）
- 复跑入口：`experiment/zrecent_use/0630/run_rr_ours_pool32.sh`（池固定 32 的 RR+Ours）

## 六、RR 侧的另一条线索（KV 池 32→8）

请求级对比（两次都是 4080 条、`output_tokens` 逐条完全相同）：

| run | TTFT>1s | TPOT>30ms | 同时达标（goodput） | TPOT≤30ms 的请求数 | 中位 TPOT |
|---|---|---|---|---|---|
| RR 9/23 (pool=8) | 157 | 3488 | **586** | 592 | 54.6 ms |
| RR 8/27 (pool=32) | 212 | 3272 | **804** | 808 | 37.6 ms |

TTFT 违规数反而更少 → **退化在 TPOT 侧**。两次 RR 的 trace、并发、客户端参数、调度器（日志均为 `intra_scheduler=stock`）都一致，配置逐项 diff 后**唯一差异是 `prefill/decode_tensor_pool_memory` 32→8**。

该池是 **pinned 主机内存**（`.../kv_connector/v1/p2p/tensor_memory_pool.py`：`torch.empty(max_block_size//4, dtype=float32, pin_memory=True)`，大小 = `mem_pool_size_gb * 1024**3` 字节/实例），8 实例 × 32GB = 256GB 主机内存，需要主机余量足够（8/27 成功、8/30–8/31 因主机内存不足而 `start_servers_failed`）。

待验证：pool=32 复跑 RR，看中位 TPOT 是否回到 ~37ms。

## 七、运维提示

该脚本有 `trap cleanup INT TERM EXIT`（第 1839 行），但 `cleanup` 不会 `exit` —— **发 TERM 只会触发一次清理，脚本随后会进入下一个配置并重新拉起服务**。要真正停止批次，必须 `kill -KILL` 父进程，再终止其子进程组（proxy + 8 个 vllm）。
