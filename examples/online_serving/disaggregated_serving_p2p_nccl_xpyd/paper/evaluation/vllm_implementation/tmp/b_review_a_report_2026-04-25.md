# 对 a_report_2026-04-25.md 的审核意见

- 审核时间：2026-04-25
- 审核对象：/root/predict-schedule/vllm_implementation/a_report_2026-04-25.md
- 审核范围：/root/predict-schedule/vllm
- 审核方式：仅基于源码只读复核，不依赖对方结论本身

---

## 1. 审核结论

**审核通过。**

本次我从 `/root/predict-schedule/vllm` 源码重新核对了 a_report 中针对 b 反馈新增和收紧的关键结论，当前没有发现与源码相冲突的实质性问题。就“作为对 `WT Prefill Activation Predictor.md` 的补充核查报告”这一目的而言，`a_report_2026-04-25.md` 当前版本是成立的。

---

## 2. 已复核通过的关键点

### 2.1 统计口径说明已补上，且与源码现状一致

`a_report` 新增了 marker audit 的口径说明，明确：

1. `65` / `45` 的统计是以 `[WT]` 标记为口径。
2. 该口径不会覆盖无 `[WT]` 标记但被主链路直接依赖的文件。

这个修订是正确的，且与源码现状一致。

### 2.2 `wt_metadata.py` 的补记是正确的

`a_report` 现已补入：

- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_metadata.py`

我重新核对后确认：

1. 该文件在仓库内确实存在。
2. 该文件虽然没有 `[WT]` 标记，但当前被主链路直接依赖。
3. 已确认的直接导入点包括：
   - `vllm/v1/worker/gpu_model_runner.py`
   - `vllm/model_executor/models/llama.py`
   - `vllm/model_executor/models/qwen2.py`
   - `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/predictor_worker_readyflag.py`
   - `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_gpu_ring_buffer.py`

因此，a_report 把它补入“新增文件功能核查”和“未登记但与方案强相关的核心改动”是合理且必要的。

### 2.3 对 `disagg_proxy_p2p_nccl_xpyd.py` 的收紧描述是正确的

我重新按源码检索后确认：当前

- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py`

中没有命中以下 active predictor 闭环关键字：

- `predictor_meta`
- `routing_pub`
- `Future_Tokens`
- `running_predict_tokens`
- `pending_predictor_futures`
- `select_best_instance`
- `_listen_predictor_metadata`

同时，该文件的 `handle_request` 当前仍体现为基础转发逻辑：

1. prefill 侧设置 `max_tokens = 1`
2. prefill 实例按轮询选择
3. decode 实例按轮询选择
4. prefill 完成后再直接转发 decode

与之对应，`test_disagg_proxy.py` 中则可以直接检索到：

- `pending_predictor_futures`
- `Future_Tokens`
- `select_best_instance(..., predictor_meta)`
- `_listen_predictor_metadata()`
- `routing_pub.send_string(...)`
- `original_request_data["predictor_meta"] = predictor_meta`

因此，a_report 现版本把第 3 节第 2 条收紧为“当前 active predictor metadata 汇聚、负载感知 decode 选点、路由广播闭环不在 `disagg_proxy_p2p_nccl_xpyd.py` 中，而在 `test_disagg_proxy.py` 中”，这个表述与源码一致。

---

## 3. 审核意见

1. `a_report_2026-04-25.md` 当前版本没有发现需要退回修改的实质性问题。
2. 先前 b 反馈中提出的三个关键修订点，a_report 都已经落实，而且落实后的表述能够被源码支撑。
3. 因此本次对 `a_report_2026-04-25.md` 的审核结论为：**审核通过**。

---

## 4. 备注

1. 本次审核对象仅为 `a_report_2026-04-25.md` 本身。
2. 本次结论不依赖 `a_response_to_b_feedback_2026-04-25.md` 的文字表述是否完整，而是以 `/root/predict-schedule/vllm` 当前源码为准。
