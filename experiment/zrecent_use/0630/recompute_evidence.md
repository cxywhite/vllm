# 抢占时 KV 去哪了：现有数据能证明到什么程度

问题：**被从 decode 实例驱逐的请求，其 KV 是「卸载」到别处（CPU/其他实例），还是直接释放 + 重算？**

结论先行：**现有数据可以证明「KV 被释放、请求被标记为从 0 重算、且不会回去向 prefill 再要 KV」；
但还不能证明「重算最终跑完」以及「宿主内存没有被搬运」**（后者缺硬件记录）。逐条如下。

---

## 证据 1：vLLM v1 根本没有 swap 路径（代码级）

```bash
grep -rn 'swap_space' vllm/v1/core/            # 无结果
grep -n  'CPU' vllm/v1/core/kv_cache_manager.py  # 无结果（没有 CPU block pool）
```

v1 的 KV 管理器只有 GPU block pool，抢占只有一条实现（`scheduler.py:511-520`）：

```python
self.kv_cache_manager.free(preempted_req)        # KV 块直接还给 GPU 池
preempted_req.status = RequestStatus.PREEMPTED
preempted_req.num_computed_tokens = 0            # ← 关键：清零
self.waiting.prepend_request(preempted_req)      # 放回等待队列首
self._preempted_recompute_tokens += preempted_req.num_output_tokens   # [WT] 计数
```

评估脚本的 `--swap-space 0`（`test_evaluation_scheme4_rr_compare.sh:1616`）在 v1 下只是冗余保险。

## 证据 2：逐请求日志显示 resume 时 `num_computed_tokens = 0`

```
[test] Resuming preempted request:chatcmpl-___prefill_addr_127.0.0.1:22013___decode_addr_...
    num_prompt_tokens:59 num_computed_tokens:0  preempted count:1
```

这行在 `scheduler.py:752`，打印的是抢占时被置零的同一个字段（打印发生在
`request.num_computed_tokens = num_computed_tokens` 赋值之前），所以它等于「引擎认为这个请求一个
token 都没算过」——原先算过的 KV 已经不存在了。

出现次数（说明抢占确实发生过、且请求确实被重新准入）：

| run | resume 次数 |
|---|---|
| `admission_ab_qwen_cap`（qwen 封顶 8192，4P4D） | 16 |
| `admission_ab_mooncake_cap` | 10 |
| `admission_ab_mooncake` | 8 |

## 证据 3：为「不让它回去取 KV」而打的补丁，本身就是反证

- 补丁前：被抢占请求 resume 时走 `start_load_kv → recv_first_available`，等 prefill 再发一次 KV；
  但 prefill 早已完成该请求并释放了 KV，永远不会再发 → **无限阻塞**（AGENTS.md §9）。
- 补丁 `cef34ada1`：`get_num_new_matched_tokens` 在 `preempt_count > 0` 时返回 `(0, False)`
  → **不去向 prefill 取 KV**，直接在 decode 本地重算。

也就是说：既没有 CPU 卸载路径，也没有「回头取 KV」路径——只剩本地重算。

## 证据 4（间接）：重算量计数

`preempted_recompute_tokens` 在每次抢占时累加 `num_output_tokens`：

| run | max 累计 |
|---|---|
| `admission_ab_mooncake` | 101,446 |
| `admission_ab_mooncake_cap` | 89,982 |
| `admission_ab_qwen_cap` | 51,446 |

---

## 还缺什么（必须诚实标注）

1. **没有观测到「被抢占请求最终完成」**：`Request ... finished ... preempted count:N` 这条日志在
   所有 run 里 **0 条**。这些 run 都在重算过程中被 watchdog/kill 打断，所以「重算 → 跑完 → 产出 token」
   这条链目前只有代码推断，没有实测样本。
2. **宿主内存证据缺失**：我挂的两个采样器在 22:21 同时挂掉（`nvidia-smi dmon` 随引擎被杀一起退出），
   所以抢占窗口内没有 `mem_avail_mib` 记录。采样器已改成 **dmon 死亡自动重启**（`hw_metrics_sampler_v2.py`）。
3. **计数口径偏小**：只累加 output tokens，重算实际包含 prompt（`num_prompt_tokens` 50–120），
   所以 `preempted_recompute_tokens` 是下界。

## 补成铁证的四步（都很便宜）

| # | 动作 | 能证明什么 |
|---|---|---|
| 1 | 常驻修复版采样器，跑一轮 KV 压测 | `mem_avail_mib` 平直 + `fb_mb` 不变 → 没有任何 CPU 搬运 |
| 2 | resume 处加日志：`num_scheduled_tokens`（应 = prompt + 已生成）、`num_cached_tokens`（应 0） | 重算的规模与「从头开始」 |
| 3 | 计数器改成 prompt+output；请求完成时打 `finished ... preempted count:N` | 「抢占后完成」的请求数 + 额外耗时 |
| 4 | （论文口径）对比「回 prefill 重算」的实现 | 当前是 **decode 本地重算**，把重算算力压在瓶颈侧；论文理想是回 prefill |
