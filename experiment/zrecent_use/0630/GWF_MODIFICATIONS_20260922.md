# GWF / scheme4 改动总结（2026-09-22）

目标：让 `scheme4_gwf`（proxy 预测式跨实例路由 + GWF 实例内调度）在 goodput 上超过
**原生 vLLM 的实例间轮询 + 实例内 stock scheduler**。

---

## 0. 总览

| 文件 | 基线 → 现在（行） | 增/删 | 备份 |
|---|---|---|---|
| `vllm/vllm/v1/core/sched/geometric_window_scheduler.py` | 599 → 1020 | +471 / −85 | `.bak_20260922`（基线）, `b`, `c`, `d` |
| `vllm/vllm/v1/core/sched/geometric_window_queue.py` | 89 → 132 | +38 / −0 | `.bak_20260922`（基线）, `c`, `d` |
| `experiment/.../disagg_serving_suite/test_disagg_proxy.py` | 2164 → 2200 | +57 / −22 | `.bak_20260922`（基线）, `b` |
| `experiment/zrecent_use/0630/test_evaluation_scheme4_rr_compare.sh` | 1961 → 2029 | +70 / −2 | `.bak_20260922`（基线）, `c`, `d` |

备份规则：**改任何现有文件前先 `cp -p` 一份带日期后缀的备份**。
后缀 `b/c/d` 对应下面的批次 2/3/4；无后缀 = 本次会话开始前的原始状态。

### 批次与触发指令

| 批次 | 触发 | 内容 |
|---|---|---|
| 0 | "先备份再改 proxy" | proxy 三处独立小修 |
| 1 | "regime 用 `P_f ≥ N·R`，N=9" | 两 regime 重写 + `promote_request` + shell 旋钮 |
| 2 | "要打败原生轮询的 goodput" | `client_arrival_ts` + block-pool 精确空闲量 + 可观测日志 |
| 3 | "超时的请求准入低优先、驱逐优先" | hopeless 判定 + 驱逐硬覆盖 + 贪心降级 + `demote_request` |
| 4 | "怎么保证不饿死" | demote gap 修正 + 硬性反饥饿截止 |

---

## 1. `test_disagg_proxy.py`（4 处）

### 1.1 删除 `_scheme4_add_reservation()`（批次 0）
- **问题**：死代码，全仓库零调用点。
- **为什么是删而不是用**：它内部要 `with self.scheme4_lock`，而唯一逻辑等价的调用点
  `select_scheme4_fastpath()` 是**在已持有该锁时**内联做同一件事的 —— 改成调用它会**自死锁**
  （`threading.Lock` 不可重入）。所以只能删。

### 1.2 新增 `_request_activation_vram_bytes()`，两处共用（批次 0）
- **问题**：正向负载模型 `_calculate_baseline_load()` 用 `L·(3H+6I)`，而 scheme4 剩余量反解
  `_scheme4_estimate_true_remaining_tokens()` 用 `L·(2H+2I)` —— 反解不是正向模型的逆。
- **做法**：抽成一个 helper，取值保留正向模型的 `(3H+6I)`。
- **影响量化**（Llama-3-8B, TPOT=50ms, budget=1.0）：激活项 2.36→6.29 MB/请求，
  memory token_limit 减少 30（R=1）～7680（R=256）tokens，即 **0.005%～1.2%，方向更保守**。
  不会推翻已有结论。

### 1.3 新增 `clear_scheme4_instance_state()` 并接线（批次 0）
- **问题**：shell 有实例崩溃重启循环（`INSTANCE_RESTART_MAX_RETRIES`）。decode 重启后
  `http_address` 不变，于是 `scheme4_reserved_tokens` / `scheme4_last_running_requests`
  等 5 个字典的**陈旧条目残留**——永久压小该实例的可用额度，并污染 reconciler 的增量基线。
- **做法**：新增方法清 5 个字典中该地址的条目；接两处线：
  - `_remove_oldest_instances()`（地址过期被摘除时）
  - `_listen_for_register()`（decode **重新注册**时）
- 锁序检查：两条取双锁的路径都是 `decode_cv → scheme4_lock`，第三条只取 `scheme4_lock`，无死锁。

### 1.4 新增 `client_arrival_ts`（批次 2）
- **问题**：decode 侧 `Request.arrival_time` 是 **proxy 转发那一刻**（`processor.py:352`），
  **不含** "proxy 等预测器 + prefill" 的时间。而 `T_ttft=1000ms`、`U=0.8` 时贪心门会在
  decode 侧等了 800ms 才放行 → 客户端观测 TTFT ≈ 800 + prefill + route，**大概率已破 1000ms**。
  这些请求无论后面多快都拿不到 goodput。
- **做法**：`handle_request()` 入口打一个 `time.time()` 戳，在 `_merge_predictor_meta(...)`
  之后对**所有分支**（含 fastpath）`predictor_meta.setdefault("client_arrival_ts", ...)`。
  只加一个 key，`predicted_output_len` / bucket tracking 行为完全不变。
- 调度器侧本来就优先读它，取不到才退回 `arrival_time`，所以不依赖本改动也能跑。

---

## 2. `geometric_window_queue.py`（3 处）

### 2.1 新增 `promote_request()`（批次 1）
- **问题**：等待队列是**按 `arrival_time` 的堆**。TTFT 紧急度对 `arrival_time` 单调
  → 队首天然是最 TTFT 紧急的（所以 TTFT 侧 O(1) 精确）。但**被抢占过**的请求
  `arrival_time` 很老、`u_ttft` 可能仍低，而其 `u_tpot` 已高 → 会排在后面被漏掉。
- **做法**：以 `当前最小 key − 1.0` 重新入堆，其余请求相对 FCFS 顺序不变。

### 2.2 新增 `demote_request()`（批次 3）
- **问题**：hopeless（已超时）的等待请求就是**等得最久的** → 天然在队首 → 会白吃紧张的准入额度。
- **做法**：以"紧贴当前队尾"重新入堆，只有当前队列里的请求都在它前面。

### 2.3 demote 的 key gap：`1.0` → `1e-6`（批次 4）
- **问题（本会话挖出的第二个 bug）**：key 就是 `arrival_time`（秒），所以 `+1.0` 不是
  "队尾一格"，而是 **"1 秒的到达时间之后"** → 之后 1 秒内到达的**所有**新请求都会插到它前面。
  到达率越高埋得越深，饥饿风险随到达率放大。
- **做法**：`max + 1e-6` —— 紧贴当前最大值，降级后到达的新请求排在它**后面**（后备队列的正常语义）。
  `1e-6` 有依据：epoch 量级（~1.7e9）下 float64 分辨率约 `2.4e-7`，高出 4 倍可安全区分，
  又远小于任何真实到达间隔。

---

## 3. `geometric_window_scheduler.py`（核心重写）

### 3.1 两 regime 重写（批次 1）
```
P_f = block_pool.get_num_free_blocks() × block_size
R   = WT_SCHEME4_RESERVATION_TOKENS      # 与 proxy fastpath 共享同一个 env
idle ⇔ P_f ≥ N·R                         # N = WT_GWF_IDLE_MIN_RESERVATIONS，默认 9
```
- **IDLE**：完全原生式放行——无任何额外门控、不受 `A` 限制、running 不重排，
  只靠 `allocate_slots` 决定（与 stock scheduler 逐行一致）。
- **GREEDY**：只放 `urgency ≥ U`（默认 0.8）的请求，每步上限 `A`。
- **为什么 R 复用同一个 env**：让"Iproxy 说还有 1 份预留空间 → 走 fastpath"与
  "scheduler 说还有 ≥9 份 → 认为空闲"共享同一个 `R`，调 `R` 时两边自动同步。
- **`N=9` 的依据**：`0.1 × 374k tokens ≈ 9.1 × 4096` —— 精确复现原来的 0.1 空闲比例。

### 3.2 删除 horizon/gate 机制（批次 1）
删掉 `_should_admit_waiting_request`、`_request_horizon_tokens/units`、`_current_bottleneck_ratio`、
`_immediate_step_feasible`、`_running_priority`、`admission_token_limit`、`short_horizon_steps`、
`waiting_budget_s`（及 `WT_GWF_HORIZON_STEPS` / `WT_GWF_WAIT_BUDGET_S`）。
- **原因**：IDLE 靠 `allocate_slots`（原生 ground truth）已足够；GREEDY 靠 urgency。模型化的
  预算检查冗余，且带着两个精度问题（见 3.5 的前身：块共享重复计数、块取整增量双计）。

### 3.3 urgency 体系（批次 1）
- `u_ttft = (now − arrived)/T_ttft`；`u_tpot = 1 − tpot_slack`；`urgency = max(两者)`。
- **episode 时钟**（`_start_episode` 每次准入重置 `_episode_start_ts/gen`）：
  抢占后 `num_computed_tokens=0` 要重算 prefill，这段时间**不能**算进 TPOT，
  否则被抢占过的请求会被误判为"极不紧急"。
- 引擎原本**没有** SLO 概念（`grep -iE "tpot|ttft|deadline" request.py` 为空），
  预算靠新增 env 从 shell 注入：`WT_GWF_TTFT_BUDGET_MS ← BENCH_GOODPUT_TTFT_MS`、
  `WT_GWF_TPOT_BUDGET_MS ← CURRENT_TPOT_MS`。

### 3.4 驱逐价值三因子（批次 1）
```
gen_ratio = num_output_tokens / predicted_output_len          # 已生成长度（相对）
slack     = 1 − observed_tpot / TPOT预算                       # TPOT 剩余时间
rem_ratio = (predicted − generated) / max_model_len            # 预测−已生成（绝对）
V = w_gen·gen_ratio − w_slack·slack − w_rem·rem_ratio          # 越大越该保留
```
实现技巧：`sort(key=V, reverse=True)` + 原生 `self.running.pop()` → 尾部即价值最低，
**完全不动 running 循环的索引管理**。
- **融合方法选型**：固定分母归一化 + 加权线性标量化，取 argmin。
  不用乘积/几何平均（量纲不同，单项为 0 会让单维度独断）；不用批内 min-max
  （分数依赖同批邻居 → 不可复现；全同请求时除零）。
- **为什么 `gen` 用相对量、`rem` 用绝对量**：两者若都按 `predicted` 归一化会**线性退化**
  （`gen/pred + rem/pred ≡ 1`），三因子塌成两因子、`w_rem` 失效。
- 权重默认 `(1.0, 1.0, 0.5)`。

### 3.5 GWF 窗口 `A`（批次 1）
只在 GREEDY 期更新：有抢占 `A = max(1, ⌊βA⌋)`，否则 `A = min(A_max, A+α)`，夹取 `[1, A_max]`。
- 保留"用 GWF 壳子"的要求；只在 GREEDY 更新，避免 IDLE 期 `A` 涨爆、进入过载首步形同失效。

### 3.6 `P_f` 改用 block pool 精确值（批次 2）
`_recompute_physical_mem_free()` 改为 `block_pool.get_num_free_blocks() × block_size`。
- **问题**：原 `Σ ceil(num_computed/block)·block` 把 **prefix cache 共享的物理块重复计入已用**
  → `P_f` 系统性偏小 → 提前进入 GREEDY → 多扣 TTFT。
- **附带收益**：新口径与 `allocate_slots` 内部判据**完全同源**，`P_f ≥ N·R` 才真正等价于
  "还剩 N 份预留的物理块"，`N` 才可标定、可跨模型迁移；顺带 O(M) → O(1)。
- 无 block pool 时回退到旧的求和。

### 3.7 可观测性（批次 2）
`[WT][GWF]` 周期日志（`WT_GWF_LOG_INTERVAL_MS`，0 关闭）：
```
steps idle_steps admitted_idle admitted_greedy preempted
evicted_hopeless demoted hopeless_forced
stop_window stop_urgency P_f idle_min window running waiting
```
- **为什么必须加**：goodput 调参的前提是能区分"**是窗口 A 卡住准入**"还是"**紧迫度门限卡住**"，
  以及"regime 分布"。没有它，调参只能靠猜，实验也无法解释。

### 3.8 hopeless（已超时）判定（批次 3）
`_is_hopeless = _ttft_blown or _tpot_blown`
- `_ttft_blown`：等待中用 `u_ttft ≥ 1`；**已开始生成的用准入时刻冻结的结论**
  （`_episode_ttft_blown`）。**为什么必须冻结**：运行中请求的 `u_ttft` 是"到达至今总耗时"，
  会一直涨——一个 TTFT 本来合格的请求，跑够 1s 后会被误判。
  实测：3s 前到达但准入时只等 0.2s → `u_ttft=1.00`（误判）但 `_ttft_blown=False`（正确）。
- `_tpot_blown`：`观测每 token 时间 > TPOT 预算`，且 `gen_delta ≥ WT_GWF_TPOT_MIN_SAMPLES`（默认 4）。
  样本下限防止"准入→首 token"空档造成的假阳性（实测 `gen_delta=2, obs=100ms → False`）。

### 3.9 驱逐优先赶 hopeless（批次 3）
`_eviction_priority = -inf if hopeless else _eviction_value`。
- **理由**：已超时的请求**永远无法再贡献 goodput**，把它占的 KV 让给还能达标的请求是
  **严格的 goodput 增益**。它不会被丢弃，只是最后服务。
- 实测：一个 `value=+0.966`（进度 290/300，按旧规则会被保护）但 TPOT 已破的请求
  被第一个驱逐。

### 3.10 贪心阶段降级 hopeless（批次 3）
门里对 hopeless 的队首执行 `demote_request`，每步上限 `WT_GWF_MAX_DEMOTE_PER_STEP`（默认 16）。
- **为什么必须主动做**：hopeless 的等待请求就是等得最久的 → 天然在队首 → 原门只会优先放它们。

### 3.11 硬性反饥饿截止（批次 4）
- **问题**：降级把它排到**新到达之后**，所以只要新请求持续到达，队首永远是 live 请求 →
  "降级超限就放行"的分支**根本不会执行**。实测（模拟）：cap 关闭时一个 hopeless 请求
  40s 内被降级 **49 次、从未被服务**。
- **做法**：`_hopeless_pending: {req_id: (首次降级时刻, request)}` 在降级时起表；
  每步 `_force_overdue_hopeless()` 找出超期最久者，用 `promote_request()` 强提到队首，
  返回其 id；门里对该 id **同时跳过降级与紧迫度判断**。
- **实测**（每步持续 1 个新到达）：

  | 配置 | 首次被降级 | 被服务 |
  |---|---|---|
  | `cap=0`（关闭） | 0.000s | 从未服务（饿死） |
  | `cap=2000ms` | 0.000s | **2.000s** |
  | `cap=5000ms` | 0.000s | **5.000s** |

- **保证陈述**：一个 hopeless 请求被本规则首次降级后，最多再等
  `WT_GWF_HOPELESS_MAX_DEFER_MS`（默认 2000ms），到期即被强提到队首、下次准入机会优先放行。
  例外三种：① 该步 `preempted_reqs` 非空（原生结构整步跳过等待循环）→ 推迟一个调度步；
  ② `allocate_slots` 真的失败（物理上放不下）→ 仍留在队首，下轮优先重试；
  ③ 该 env 设为 0 → 保证被显式关闭。
  多个同时到期按降级时间 FIFO，每步放行 1 个。
- **本质**：这是"**有界的容量税**换**有界的尾延迟**"的旋钮。hopeless 请求越晚服务对 goodput 越好，
  但必须服务完才能结束这一轮。

---

## 4. `test_evaluation_scheme4_rr_compare.sh`（3 组）

1. **USER CONFIG 段新增 12 个 `WT_GWF_*` 默认值**。
2. **decode 的 `setsid env` 改多行并注入**：`WT_GWF_TTFT_BUDGET_MS`、
   `WT_GWF_TPOT_BUDGET_MS`、共享的 `WT_SCHEME4_RESERVATION_TOKENS`，以及全部 GWF 旋钮。
   - **为什么必须**：改之前**没有任何脚本导出 `WT_GWF_*`**，所以所有实验跑的都是代码内默认值，
     无法调参；且 TTFT/TPOT 预算只能从 `BENCH_GOODPUT_TTFT_MS` / `CURRENT_TPOT_MS` 来
     （引擎不知道 SLO）。
3. **`config_*.json` 新增 `gwf_policy` 块**记录全部生效值。
   - **为什么**：实验可复现 + 论文可直接引用配置。已实测展开后 `json.loads` 通过。

---

## 5. 明确**没有**改的（及原因）

| 项 | 原因 |
|---|---|
| `vllm/v1/request.py` | episode 时钟放在 scheduler 侧 dict，不加字段 |
| `config/scheduler.py`、`arg_utils.py` | `--geometric-window-scheduler` 与 `scheduler_cls` 接线**已存在** |
| proxy 的 fastpath `predictor_meta` 竞态 | `L2033` 已对**所有**分支做 merge，预测器若在 prefill 期间算完就能白拿真预测。需先量 `predict_hidden_by_prefill` 比例再决定 |
| `w_slack` 符号/默认、`GREEDY_URGENCY=0.8`、`A_INIT=1` | 属于需要实验标定的 policy 参数，交给你扫 |
| 深度过载下 hopeless 沉底导致 wall time 变长 | 只用 cap 控制，未改策略 |
| `test_disagg_proxy.py.bak_20260922` 之外的其它 proxy 副本 | 另有 3 份近似拷贝（`vllm/examples/...`、`paper/...`）未打同样补丁 |

---

## 6. 待办与风险

1. **`A_INIT=1` + "有抢占的步整步不放行"**（原生结构）→ 饱和时准入可能被 `A` 卡住，
   表现为 `stop_window` 远大于 `stop_urgency`。**第一次跑完先看这行日志**再决定调参方向。
2. **`predict_hidden_by_prefill` 未量化**（5 分钟诊断，只读 CSV，不改代码）：
   - 比例高 → 驱逐用真预测；
   - 比例低 → `predicted_output_len` 退回 `max_tokens`，三因子退化（`w_rem` 变成常数偏移）。
3. **`request.max_tokens` = 数据集 `output_tokens`**（`benchmark_serving_trace.py:577`，
   且 shell 传 `--ignore-eos` → 恰好生成这么多）。所以 fallback 时 `predicted_output_len`
   拿到的是**真值（oracle）**，而 roundrobin 基线 `_should_track_predictive_pressure()==False`
   → 这些量恒为 0。**"免费真值"只出现在 ours 一侧，是对比有效性的风险点**（已同意先搁置）。
   注：`--maxtokenscustom` 是**死参数**（argparse 声明后从未使用），不要以为它能改 `max_tokens`。
4. **`GeometricWindowQueue` 用 `arrival_time` 做堆 key**：若两个请求 `arrival_time` 完全相同，
   `heapq` 会去比较 `Request` → `TypeError`。µs 级重复概率极低，但可加 tie-breaker 硬化。
5. **阈值不是严格同一个量**：同 R、同量纲，但 scheduler 只用 KV 维、且不知道 proxy 的
   `reserved_tokens`；scheduler 的是每步本地精确值、proxy 的是 10ms 前抓的远端近似值。
   **论文表述应写成"同一量纲、共享同一个 R，调度器用本地精确状态逐步复核"**，不要写"同一个阈值"。

---

## 7. 回滚

```bash
# 全部回滚到本次会话开始前
cp vllm/vllm/v1/core/sched/geometric_window_scheduler.py.bak_20260922  <...>/geometric_window_scheduler.py
cp vllm/vllm/v1/core/sched/geometric_window_queue.py.bak_20260922      <...>/geometric_window_queue.py
cp experiment/zrecent_use/disagg_serving_suite/test_disagg_proxy.py.bak_20260922 <...>
cp experiment/zrecent_use/0630/test_evaluation_scheme4_rr_compare.sh.bak_20260922 <...>

# 只回滚某一批（b=批次2前, c=批次3前, d=批次4前）
cp <file>.bak_20260922d <file>
```

## 8. 新增旋钮一览

| env | 默认 | 作用 |
|---|---|---|
| `WT_SCHEME4_RESERVATION_TOKENS` | `max_position_embeddings//2` = 4096 | **与 proxy 共享**的预留单位 R |
| `WT_GWF_IDLE_MIN_RESERVATIONS` | 9 | `N`；`P_f ≥ N·R` → IDLE。设极大=永远 GREEDY，设 1=几乎永远 IDLE |
| `WT_GWF_GREEDY_URGENCY` | 0.8 | GREEDY 放行门限 `U`（调低=更早放行，利 TTFT 损 TPOT） |
| `WT_GWF_TTFT_BUDGET_MS` | `BENCH_GOODPUT_TTFT_MS` | TTFT 预算 |
| `WT_GWF_TPOT_BUDGET_MS` | `CURRENT_TPOT_MS` | TPOT 预算 |
| `WT_GWF_ALPHA` / `WT_GWF_BETA` / `WT_GWF_A_INIT` / `WT_GWF_A_MAX` | 1 / 0.9 / 1 / `max_num_seqs` | GWF 窗口 |
| `WT_GWF_W_GEN` / `WT_GWF_W_SLACK` / `WT_GWF_W_REM` | 1.0 / 1.0 / 0.5 | 驱逐三因子权重 |
| `WT_GWF_TPOT_MIN_SAMPLES` | 4 | 判 TPOT 已破所需样本下限 |
| `WT_GWF_MAX_DEMOTE_PER_STEP` | 16 | 每步最多降级几个 hopeless |
| `WT_GWF_HOPELESS_MAX_DEFER_MS` | 2000 | **反饥饿硬截止**；0 = 关闭保证 |
| `WT_GWF_LOG_INTERVAL_MS` | 1000 | 观测日志周期；0 = 关闭 |
