# WT 源码改动核查报告（基于 `WT Prefill Activation Predictor.md`）

- 核查时间：2026-04-25
- 核查范围：`/root/predict-schedule/vllm`
- 核查方式：仅使用只读命令（`rg`、`read`、`git status` 等）
- 核查目标：
  - 按文档定位你在源码中的改动位置
  - 说明改动作用
  - 排查是否存在未登记改动

---

## 1. 总体结论

1. 在 `vllm` 仓库中共发现 **65 个文件**包含 `[WT]` 标记。
2. 文档列出的 22 个目标文件中：
   - **20 个**文件存在且包含 `[WT]` 标记。
   - **1 个**文件存在但**没有**`[WT]` 标记：
     - `vllm/v1/core/sched/aimd_queue.py`
   - **1 个**文件在文档给定路径下**不存在**：
     - `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_test.sh`
3. 发现 **45 个含 `[WT]` 标记但未登记在该 md 文件中的文件**（见附录 B）。
4. 一个关键路径差异：
   - 文档中将 proxy 关键逻辑登记在 `disagg_proxy_p2p_nccl_xpyd.py`。
   - 实际“predictor_meta 汇聚、路由决策、广播到 KV connector”的核心链路主要在：
     - `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy.py`
5. 统计口径说明：
    - 本报告“65/45”等统计均以 `[WT]` 标记为口径。
    - 该口径不会覆盖“无 `[WT]` 标记但被主链路直接依赖”的文件，例如：
       - `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_metadata.py`
       - `vllm/v1/core/sched/aimd_queue.py`

---

## 2. 已登记改动：定位与作用清单

## 2.1 调度开关与配置注入

### 文件
- `vllm/engine/arg_utils.py`
- `vllm/config/scheduler.py`

### 关键位置
- `arg_utils.py`: 488, 953, 1406
- `scheduler.py`: 144, 238

### 作用
1. 新增自定义调度/实验开关字段：
   - `pastfuture_scheduler`
   - `activation_predict`
   - `aimd_scheduler`
   - `test_p2d`
   - `test_optimal`
   - `test_model`
2. 将上述字段暴露为 CLI 参数。
3. 在 `SchedulerConfig` 阶段动态切换 `scheduler_cls`：
   - PastFuture -> `FuturePastScheduler`
   - AIMD -> `AIMDScheduler`

---

## 2.2 Prefill 激活预测主链路（Runner + Llama）

### 文件
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/model_executor/models/llama.py`

### 关键位置
- `gpu_model_runner.py`: 2319, 2373
- `llama.py`: 249, 259, 466, 493, 512, 522, 570+

### 作用
1. 在 runner 中为每个 scheduled request 组装 `Custom_Metadata`，包含 token 区间和采样参数。
2. 在模型 forward 时根据开关传入 `my_metadata` 和 `activation_predict`。
3. 在 Llama 中间层计算 `token_importance`，并在目标层导出 `hidden_states` + `importance`。
4. 将中间激活写入 `GPURingBuffer`，交给 predictor 异步消费。
5. 记录 `predict_start_ts_ns` / `prefill_end_ts_ns`，用于预测与 prefill 重叠时延分析。

---

## 2.3 延迟 KV 传输（P2P NCCL Connector）

### 文件
- `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`

### 关键位置
- 97, 112, 303, 339, 360, 363

### 作用
1. 引入 `defer_kv_send`：在 prefill 侧先缓存 KV，不立即发送。
2. 新增路由监听线程（订阅 proxy 广播），维护 `routing_table`。
3. 新增 flush 线程：收到 decode 路由后再发送缓存的 KV tensor。
4. 修改 `save_kv_layer`：defer 模式下仅入队缓存。
5. 修改 `wait_for_save`：defer 模式下不在 prefill 阶段阻塞等待发送完成。

---

## 2.4 Proxy 路由与 predictor_meta 汇聚链路

### 文件
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy.py`

### 关键位置
- 53, 71, 281, 350, 408, 452, 457, 464, 470

### 作用
1. 维护 predictor req_id 别名与 Future 映射，解决异步到达与命名差异。
2. 从 decode metrics 读取 `running_predict_tokens`，构造 `Future_Tokens` 负载指标。
3. 在路由选择时将 `predict_output_len` 叠加到 `Future_Tokens`，做负载敏感选点。
4. 将路由结果通过 PUB 广播给 KV connector（驱动延迟 KV flush）。
5. 将 `predictor_meta` 注入 decode 请求。

---

## 2.5 predictor_meta 贯通 OpenAI -> Engine -> Request

### 文件
- `vllm/engine/protocol.py`
- `vllm/v1/engine/__init__.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/processor.py`
- `vllm/v1/request.py`
- `vllm/entrypoints/openai/serving_chat.py`
- `vllm/entrypoints/openai/serving_completion.py`

### 关键位置（示例）
- `protocol.py`: 58
- `__init__.py`: 72
- `async_llm.py`: 271, 286, 337, 390
- `processor.py`: 337, 461
- `request.py`: 46, 129, 156
- `serving_chat.py`: 312
- `serving_completion.py`: 226

### 作用
1. 扩展接口签名，允许 `predictor_meta` 从入口请求一路透传。
2. 注入到 `EngineCoreRequest` / `Request`，供 scheduler 和监控逻辑读取。

---

## 2.6 Prometheus 指标扩展（decode 侧）

### 文件
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/metrics/stats.py`
- `vllm/v1/metrics/loggers.py`

### 关键位置
- `scheduler.py`: 178, 1030, 1235
- `stats.py`: 49
- `loggers.py`: 206, 562

### 作用
1. 增加 scheduler 内部计数：
   - `running_tokens`
   - `running_predict_tokens`
   - `waiting_tokens`
2. 将上述计数写入 `SchedulerStats`。
3. 在 Prometheus logger 中注册并上报 gauge 指标。

---

## 2.7 Decode 端 AIMD 调度

### 文件
- `vllm/v1/core/sched/aimd_scheduler.py`
- `vllm/v1/core/sched/aimd_queue.py`（无 `[WT]` 标记但实现存在）

### 关键位置
- `aimd_scheduler.py`: 46, 65, 193, 234, 297, 510, 524, 693, 788
- `aimd_queue.py`: 37 (`class AIMDRequestQueue`)

### 作用
1. 在 `Scheduler` 基础上加入 virtual/physical memory 预算控制。
2. 根据请求完成度和预测输出长度驱动抢占/恢复策略。
3. 增加 waiting backoff（`waiting_times`）与 no-preempt 调整机制。
4. 延续 Prometheus token 指标逻辑。

---

## 2.8 新增文件功能核查

### `examples/.../wt_gpu_ring_buffer.py`
- 关键位置：17, 60, 83, 97
- 作用：
  - GPU 数据面缓存 (`hidden`, `importance`)
  - CPU 控制面状态机 (`EMPTY/WRITING/READY/READING`)
  - 提供 prefill 写入 + predictor 按 batch_id 读取

### `examples/.../predictor_worker_readyflag.py`
- 关键位置：149, 243, 339, 347, 367, 517
- 作用：
  - 加载 BERT 预测器并消费 ring buffer batch
  - 基于分桶统计估计 `predict_output_len`（`miu + sigma`）
  - 填充预测时间戳并异步写 CSV
  - 通过 ZMQ PUSH 发送 metadata 给 proxy

### `examples/.../test_disagg_proxy.py`
- 作用见 2.4；此文件在文档“新增文件”中已登记，且为当前关键链路实现载体。

### `examples/.../wt_metadata.py`（无 `[WT]` 标记，但为活跃依赖）
- 当前仓库内存在：`examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_metadata.py`
- 导入关系（主链路）：
   - `vllm/v1/worker/gpu_model_runner.py`
   - `vllm/model_executor/models/llama.py`
   - `vllm/model_executor/models/qwen2.py`
   - `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/predictor_worker_readyflag.py`
   - `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_gpu_ring_buffer.py`
- 作用：提供 `Custom_Metadata` 数据结构，承载请求 token 区间、采样参数、预测长度与时间戳字段，是激活预测闭环的数据契约。

---

## 3. 文档与源码不一致项

1. `wt_test.sh` 路径不一致：
   - 文档登记路径：`vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_test.sh`
   - 实际未找到该文件。
   - 另在仓库外层目录发现：`/root/predict-schedule/design_predict_activation_experiment/wt_test.sh`
2. proxy 关键逻辑文件偏差：
   - 文档登记 `disagg_proxy_p2p_nccl_xpyd.py`。
   - 当前 active 的 predictor metadata 汇聚、负载感知 decode 选点、路由广播闭环不在 `disagg_proxy_p2p_nccl_xpyd.py` 中，而在 `test_disagg_proxy.py` 中。
   - `disagg_proxy_p2p_nccl_xpyd.py` 当前主要为 self-check、监控输出和基础转发（prefill `max_tokens=1` + prefill/decode 轮询后转发）。
3. `aimd_queue.py` 已实现 AIMD 队列，但未使用 `[WT]` 标记。

---

## 4. 未登记但与方案强相关的核心改动

以下文件不在当前 md 清单中，但明显属于同一方案链路：

1. `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_metadata.py`（无 `[WT]` 标记）
   - 被 `gpu_model_runner.py` / `llama.py` / `qwen2.py` / `predictor_worker_readyflag.py` / `wt_gpu_ring_buffer.py` 直接导入。
2. `vllm/model_executor/models/qwen2.py`
   - 与 `llama.py` 类似，已接入激活预测链路（ring buffer、token_importance、metadata）。
3. `vllm/v1/core/sched/pastfuture_queue.py`
   - PastFuture 估算与历史输出长度窗口逻辑。
4. `vllm/v1/core/sched/pastfuture_scheduler.py`
   - PastFuture 请求接纳与历史长度回写逻辑。
5. `vllm/v1/metrics/wt_gpu_stats.py`
   - NVML 解码侧 GPU 统计辅助模块。

---

## 5. 附录 A：文档中“存在且含 WT”文件（20）

- `vllm/engine/arg_utils.py`
- `vllm/config/scheduler.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/model_executor/models/llama.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py`
- `vllm/v1/metrics/loggers.py`
- `vllm/v1/metrics/stats.py`
- `vllm/v1/core/sched/scheduler.py`
- `vllm/v1/request.py`
- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/processor.py`
- `vllm/v1/engine/__init__.py`
- `vllm/engine/protocol.py`
- `vllm/entrypoints/openai/serving_chat.py`
- `vllm/entrypoints/openai/serving_completion.py`
- `vllm/v1/core/sched/aimd_scheduler.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_gpu_ring_buffer.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/predictor_worker_readyflag.py`

---

## 6. 附录 B：含 WT 但未登记于当前 md 的文件（45）

- `benchmarks/backend_request_func_timestamp.py`
- `benchmarks/benchmark_serving_ablation_decode.py`
- `benchmarks/benchmark_serving_baseline.py`
- `benchmarks/benchmark_serving_baseline_v2.py`
- `benchmarks/benchmark_serving_baseline_v3.py`
- `benchmarks/benchmark_serving_batchsize.py`
- `benchmarks/benchmark_serving_dataset.py`
- `benchmarks/benchmark_serving_load_balance.py`
- `benchmarks/benchmark_serving_mean_and_sigma.py`
- `benchmarks/benchmark_serving_predict_dataset.py`
- `benchmarks/benchmark_serving_timestamp.py`
- `benchmarks/benchmark_serving_trace.py`
- `benchmarks/test2.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/README.md`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2d_mean_and_sigma.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd copy 2.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd copy.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd_copy 3.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd_modeling_notes.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd_reviewfix.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/plot_decode_load2.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/predictor_worker_readyflag copy.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy_p2d copy.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy_p2d.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy_p2d_loadbalance.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy_p2d_loadbalance_threahold.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy_p2d_mean_and_sigma.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/1-3DLoad/3Dload/plot_3dload_triptych.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/modeling_analysis/A800_P2P_NCCL_Llama3_8B_Decode3DLoad_Modeling_Review.txt`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_gpu_ring_buffer copy.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_test_tmp_2.9_not_remove/disagg_proxy_p2p_nccl_xpyd.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_test_tmp_2.9_not_remove/test_disagg_proxy.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_test_tmp_2.9_not_remove/test_disagg_proxy_decode.py`
- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_test_tmp_2.9_not_remove/test_disagg_proxy_p2d.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector copy.py`
- `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector_reviewfix.py`
- `vllm/entrypoints/openai/protocol.py`
- `vllm/entrypoints/openai/serving_chat copy.py`
- `vllm/model_executor/models/llama copy.py`
- `vllm/model_executor/models/qwen2.py`
- `vllm/v1/core/sched/pastfuture_queue.py`
- `vllm/v1/core/sched/pastfuture_scheduler.py`
- `vllm/v1/core/sched/scheduler copy.py`
- `vllm/v1/engine/core.py`
- `vllm/v1/metrics/wt_gpu_stats.py`

---

## 7. 附录 C：文档中登记但缺失的文件

- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_test.sh`

---

## 8. 备注

- 本报告未修改 `vllm` 源码，仅新增文档文件。
- 报告中的行号来自只读检索（`rg -n` 和上下文读取）。后续如文件变更，行号可能漂移。

---

## 9. 针对 b 复核反馈的处理结果（再次源码复核后）

### 9.1 反馈点：漏记 in-repo `wt_metadata.py`

- 处理结论：**接受**。
- 理由：
   1. `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_metadata.py` 在仓库内存在。
   2. 虽无 `[WT]` 标记，但被 `gpu_model_runner.py`、`llama.py`、`qwen2.py`、`predictor_worker_readyflag.py`、`wt_gpu_ring_buffer.py` 直接导入。
   3. 其 `Custom_Metadata` 是主链路数据结构，属于“原始 md 与当前实现映射”应覆盖的关键文件。
- 已落实：本报告 2.8、4、1（口径说明）已补充。

### 9.2 反馈点：`disagg_proxy_p2p_nccl_xpyd.py` 现态描述应更收紧

- 处理结论：**接受**。
- 理由：
   1. 对该文件检索 `predictor_meta` / `routing_pub` / `Future_Tokens` / `running_predict_tokens` / `pending_predictor_futures` / `select_best_instance` / `_listen_predictor_metadata`，当前无命中。
   2. 该文件 `handle_request` 当前关键流程为：`prefill_request["max_tokens"]=1`，prefill/decode 轮询，再转发 decode。
   3. 上述 predictor 闭环关键词在 `test_disagg_proxy.py` 中有完整命中与实现。
- 已落实：本报告第 3 节第 2 条已改为“active 闭环已从 `disagg_proxy_p2p_nccl_xpyd.py` 漂移至 `test_disagg_proxy.py`”。

### 9.3 反馈点：补充统计口径说明（marker audit vs 完整映射）

- 处理结论：**接受**。
- 理由：
   1. 65/45 统计严格来自 `[WT]` marker，天然会漏掉无 marker 的活跃依赖。
   2. 若用于“完整实现映射”，必须显式补充无 marker 的关键文件。
- 已落实：本报告第 1 节新增“统计口径说明”。
