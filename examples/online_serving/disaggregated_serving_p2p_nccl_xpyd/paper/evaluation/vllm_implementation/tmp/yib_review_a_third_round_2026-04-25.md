# 对 a 同学三轮更新报告的复核结论

- 时间：2026-04-25
- 复核人：GitHub Copilot
- 复核对象：
  - /root/predict-schedule/paper/evaluation/vllm_implementation/a_status_review_2026-04-25.md
  - /root/predict-schedule/paper/evaluation/vllm_implementation/a_response_to_yib_second_round_2026-04-25.md
- 代码范围：/root/predict-schedule/vllm
- 复核方式：仅基于源码只读检查，不直接采信已有报告

---

## 1. 复核结论

结论：**本轮审核通过。**

原因是：a 同学在上一轮遗漏的两条关键问题，本轮都已经补入状态报告，且表述与源码一致；我这次没有再发现新的源码级问题。

---

## 2. 本轮重点复核项与结果

我本轮重点复核了会直接影响“是否通过”的几条断言。

1. qwen 中间层导出问题：
   - 当前源码里，qwen 仍然使用局部索引 `idx==7`。
   - llama 则使用 `layer_idx = self.start_layer + idx` 后再与 `target_layer_idx` 比较。
   - a 同学本轮已经把这个问题从“单纯层号偏差”升级为“局部/全局层号语义不一致 + partition 漂移风险”，这一点与源码一致。
2. openai-chat 请求体字段并存问题：
   - benchmark 链路会把 `max_tokens` 写入 `extra_body`。
   - openai-chat 请求体又会先写 `max_completion_tokens`，随后执行 `payload.update(extra_body)`。
   - a 同学已把“`max_completion_tokens` 与 `max_tokens` 并存”的兼容性与解释风险补入问题清单，这一点与源码一致。
3. 默认评测入口是否仍错位：
   - `test_evaluation.sh` 默认 `PROXY_SCRIPT` 仍指向 `disagg_proxy_p2p_nccl_xpyd.py`。
   - baseline proxy 代码仍是 prefill/decode 轮询路径，不是 predictor-aware 闭环。
   - a 同学此前关于“默认评测入口未命中 predictor-aware proxy”的判断没有被改坏，仍然成立。
4. 短请求 token 筛选边界是否仍表述准确：
   - predictor 侧仍是 `k=min(max_req_len, seq_len)` 后做 `topk`。
   - 所以短请求不发生真实截断，这一表述仍与源码一致。

---

## 3. 对 a 同学当前报告的判断

我本轮没有发现新的源码级遗漏项。

当前 [a_status_review_2026-04-25.md](/root/predict-schedule/paper/evaluation/vllm_implementation/a_status_review_2026-04-25.md) 的状态已经满足以下条件：

1. 之前指出的关键遗漏问题已经补齐。
2. 关键判断都能从源码直接找到支撑。
3. 新增回复文件 [a_response_to_yib_second_round_2026-04-25.md](/root/predict-schedule/paper/evaluation/vllm_implementation/a_response_to_yib_second_round_2026-04-25.md) 与状态报告之间没有出现新的实质性冲突。

因此，我对 a 同学这一轮更新后的报告给出：

- **审核通过。**

---

## 4. 说明

这里的“审核通过”是指：

1. a 同学当前这份实现状态报告，对源码现状的描述已经基本准确。
2. 报告中列出的主要风险和问题清单，已覆盖我前几轮指出的关键点。

这不等于代码本身已经没有问题；相反，代码中仍然存在报告第 3 节列出的那些实现风险。只是这些风险目前已经被报告正确反映出来，因此本轮不再追加新的审查问题。