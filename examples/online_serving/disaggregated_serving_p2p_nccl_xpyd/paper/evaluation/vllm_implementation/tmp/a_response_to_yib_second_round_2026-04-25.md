# a 对 b 同学二轮反馈的回复（源码复核）

- 时间：2026-04-25
- 作者：a同学
- 复核方式：仅基于源码只读检查，不盲信反馈结论
- 反馈文件：/root/predict-schedule/paper/evaluation/vllm_implementation/yib_review_a_second_round_2026-04-25.md
- 关联状态报告：/root/predict-schedule/paper/evaluation/vllm_implementation/a_status_review_2026-04-25.md

---

## 1. 总体结论

二轮反馈的关键新增意见经源码复核后成立。

本轮处理结论：

1. 接受“暂不直接标记审核通过”的建议。
2. 接受“qwen 不仅层号偏差，还存在局部/全局层号语义不一致”的问题。
3. 接受“openai-chat 请求体 `max_completion_tokens` 与 `max_tokens` 并存”的问题。
4. 本轮无明确拒绝项。

---

## 2. 逐条回复

## 2.1 关于“当前还不建议直接审核通过”

处理：接受。

理由：

1. 二轮指出的两条新增问题均可从源码直接验证。
2. 这两条问题会影响实验可复现性和结果解释，不应忽略。

已落实动作：

1. 已在 a_status 新增/升级对应问题条目。
2. 已在 a_status 新增“针对 b 同学二轮反馈的复核处理结果”章节。

---

## 2.2 关于“qwen 不只是层号偏差，还与 llama 的层号语义不一致”

处理：接受。

源码证据：

1. qwen 当前导出条件是局部循环索引 `idx==7`：
   - /root/predict-schedule/vllm/vllm/model_executor/models/qwen2.py
2. llama 的导出条件使用全局层号：
   - 先计算 `layer_idx = self.start_layer + idx`
   - 再比较 `layer_idx == target_layer_idx`
   - /root/predict-schedule/vllm/vllm/model_executor/models/llama.py

结论：

1. 问题不止是“7 不是 13”。
2. 还包括“qwen 当前导出语义依赖局部分片索引”，在 pipeline partition 变化时会有语义漂移风险。

已落实动作：

1. a_status 的 3.3 第 4 条已升级为“层号偏差 + 实现语义不一致 + partition 漂移风险”。

---

## 2.3 关于“openai-chat 请求体会同时包含 max_completion_tokens 和 max_tokens”

处理：接受。

源码证据：

1. benchmark 侧会把 `max_tokens` 放进 `extra_body`：
   - /root/predict-schedule/vllm/benchmarks/benchmark_serving_trace.py
2. openai-chat 请求体先写 `max_completion_tokens`，然后执行 `payload.update(extra_body)`：
   - /root/predict-schedule/vllm/benchmarks/backend_request_func.py

结论：

1. 当前请求体中可能并存这两个字段。
2. 这会带来接口优先级歧义与评测结果解释风险。

已落实动作：

1. a_status 的 3.3 新增第 6 条，已将该问题纳入清单。

---

## 3. 对二轮反馈的最终回应

1. b 同学二轮新增问题成立，且都已纳入状态报告。
2. 在完成这两项补充前，不建议对实现状态做“无保留审核通过”。
3. 当前更准确状态是：主链路结论基本可信，但仍有需修复的实现细节与评测口径风险。

---

## 4. 后续最小动作建议

1. 修正 qwen 导出层选择逻辑，统一到“全局层号语义”。
2. 在 openai-chat 请求构造中统一最大输出字段（仅保留一种规范字段）。
3. 将两条修复纳入下一轮评测前的阻断项（gate）。
