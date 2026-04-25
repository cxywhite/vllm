# 对 a 同学二轮审查与回复的复核意见

- 时间：2026-04-25
- 复核人：GitHub Copilot
- 复核对象：
  - /root/predict-schedule/paper/evaluation/vllm_implementation/a_status_review_2026-04-25.md
  - /root/predict-schedule/paper/evaluation/vllm_implementation/a_response_to_yib_feedback_2026-04-25.md
- 代码范围：/root/predict-schedule/vllm
- 复核方式：仅基于源码只读检查，不直接采信已有报告

---

## 1. 结论

结论：**a 同学这轮更新后的主结论大体成立，但我不建议直接标记为“审核通过”；还需要补充 2 个源码层面的新问题。**

我这次重点复核了会直接影响结论的断言，包括：

1. 短请求是否真的会保留全部 token。
2. KV 发送是否确实是异步 defer + 后台 flush。
3. qwen 的中间层导出条件是否仍与设计说明不一致。
4. 默认评测脚本是否仍指向 baseline proxy。
5. benchmark 请求级元数据链路是否真的打通。

复核结果是：

1. a 报告对以上 5 类问题的主判断，基本都能从源码中找到支持。
2. 但源码里还能继续抽出 2 个 a 报告没有明确写出的新问题。
3. 因此，我的判断是：**不能算“有问题很多、主结论失真”，但也还不到“无需补充即可审核通过”的程度。**

---

## 2. 我确认 a 报告成立的部分

以下判断，我复核后认为成立：

1. `predictor_worker_readyflag.py` 里的 token 选择逻辑确实是 `k=min(max_req_len, seq_len)` 后再 `topk`，所以短请求不会发生真实截断，a 报告这次对逻辑 2 的收紧是对的。
2. `p2p_nccl_connector.py` 的 defer 语义确实是“prefill 侧仅缓存，后台 flush 线程等待路由后发送”，`wait_for_save()` 在 defer 模式下不会阻塞 prefill 前向，a 报告这次对逻辑 4 的修正是对的。
3. 默认评测脚本 [predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/evaluation/test_evaluation.sh#L122](predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/evaluation/test_evaluation.sh#L122) 仍然指向 baseline proxy `disagg_proxy_p2p_nccl_xpyd.py`，而不是 `test_disagg_proxy.py`，所以 a 报告关于“默认评测入口与 predictor-aware proxy 错位”的判断成立。
4. benchmark trace 链路里，`req_id`、`prompt_len`、`output_tokens` 和采样参数确实会从 [predict-schedule/vllm/benchmarks/benchmark_serving_trace.py](predict-schedule/vllm/benchmarks/benchmark_serving_trace.py) 进入 `RequestFuncInput`，再由 [predict-schedule/vllm/benchmarks/backend_request_func.py](predict-schedule/vllm/benchmarks/backend_request_func.py) 写入 HTTP 请求，a 报告关于“请求级元数据链路已通”的判断成立。
5. qwen 当前导出条件仍是 [predict-schedule/vllm/vllm/model_executor/models/qwen2.py#L475](predict-schedule/vllm/vllm/model_executor/models/qwen2.py#L475) 的 `idx==7`，因此 a 报告关于“qwen 层号与设计说明不一致”的判断成立。

---

## 3. 我认为还需要补充的 2 个新问题

## 3.1 qwen 不只是“层号写成 7”，而是实现语义也和 llama 不一致

这是我认为本轮最需要补充的新问题。

源码对比后可以看到：

1. llama 的实现先定义了 [predict-schedule/vllm/vllm/model_executor/models/llama.py#L514](predict-schedule/vllm/vllm/model_executor/models/llama.py#L514) 的 `target_layer_idx = 15`。
2. 然后又在 [predict-schedule/vllm/vllm/model_executor/models/llama.py#L519](predict-schedule/vllm/vllm/model_executor/models/llama.py#L519) 把循环局部索引转换成全局层号：`layer_idx = self.start_layer + idx`。
3. 也就是说，llama 的导出判断是按“全局层号”做的。
4. 但 qwen 当前在 [predict-schedule/vllm/vllm/model_executor/models/qwen2.py#L475](predict-schedule/vllm/vllm/model_executor/models/qwen2.py#L475) 直接使用 `idx==7`，没有对应的全局层号换算。

这比“7 不等于 13”更进一步，意味着：

1. qwen 当前选择的是局部分片内的第 8 层，而不是稳定的全局模型层。
2. 一旦 pipeline partition 方式变化，qwen 的导出层语义就可能跟着漂移。
3. 所以这里的问题不只是“和设计说明不一致”，还包括“和 llama 的实现语义不一致，实验可复现性与跨模型可比性都受影响”。

我建议 a 同学把这一点明确补到问题清单里，而不是只写“当前 idx==7，不是 13”。

## 3.2 benchmark openai-chat 路径当前会同时发送 `max_completion_tokens` 和 `max_tokens`

这是我认为第二个需要补上的问题。

源码链路是：

1. [predict-schedule/vllm/benchmarks/benchmark_serving_trace.py](predict-schedule/vllm/benchmarks/benchmark_serving_trace.py) 会把 `request.expected_output_len` 放进 `sampling_params["max_tokens"]`。
2. [predict-schedule/vllm/benchmarks/backend_request_func.py#L417](predict-schedule/vllm/benchmarks/backend_request_func.py#L417) 又把这个值写成 `max_completion_tokens`。
3. 但后面在 [predict-schedule/vllm/benchmarks/backend_request_func.py#L426](predict-schedule/vllm/benchmarks/backend_request_func.py#L426) 执行了 `payload.update(request_func_input.extra_body)`。
4. 这会把 `extra_body` 里的 `max_tokens` 再次塞回请求体。

也就是说，openai-chat 请求体里现在会同时出现：

1. `max_completion_tokens`
2. `max_tokens`

这不一定立刻报错，但它至少带来两个风险：

1. 不同服务端对这两个字段的优先级可能不同，评测结果解释会出现歧义。
2. 如果某条服务链只接受其中一种写法，当前请求体会表现出兼容性不稳定。

a 报告这轮正确写出了“元数据链路已通”，但没有把这个请求体字段重复问题单独列出来。我认为它值得补进“评测链路实现问题”里。

---

## 4. 对本轮 a 报告的最终判断

我的最终判断是：**主结论基本可信，但本轮还不能直接写“审核通过”。**

原因不是因为 a 的主要判断错了，而是因为：

1. qwen 导出层问题目前还只写到了“层号偏差”，没有继续上提到“局部索引/全局索引语义不一致”。
2. benchmark 请求体里 `max_completion_tokens` 与 `max_tokens` 并存的问题还没有被纳入报告。

如果把上面这 2 点补进去，我认为这轮 a 报告就基本可以通过。

---

## 5. 建议补充到 a 报告中的最小修改

1. 在 qwen 问题处补一句：
   - 当前不仅与设计要求的层号 13 不一致，而且实现方式也没有像 llama 一样使用全局层号，存在 pipeline partition 变化时导出层语义漂移的风险。
2. 在 benchmark/评测链路问题处补一句：
   - openai-chat 请求当前会同时携带 `max_completion_tokens` 与 `max_tokens`，存在接口兼容性和评测解释歧义风险。

如果只问这轮报告是否可以直接给出最终结论，我的答案是：

- **暂不直接标记审核通过，建议先补完上述两点。**