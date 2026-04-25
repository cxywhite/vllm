# a 对 yib 反馈的逐条回复（源码复核版）

- 时间：2026-04-25
- 作者：a同学
- 复核方式：仅基于源码只读检查，不盲信已有结论
- 代码范围：/root/predict-schedule/vllm
- 对应反馈文件：/root/predict-schedule/paper/evaluation/vllm_implementation/yib_feedback_2026-04-25.md

---

## 1. 总体结论

本轮对 yib 反馈逐条复核后，关键意见整体成立。

我对主要问题的处理结果如下：

1. 接受：逻辑 2 的表述需要收紧，短请求不一定发生真实 token 截断。
2. 接受：逻辑 4 应描述为异步 defer + 后台 flush，而非 prefill 线程阻塞等待。
3. 接受：qwen 导出层号与设计说明存在偏差（当前 `idx==7`，不是期望的 13）。
4. 接受：predictor-aware active proxy 与默认评测入口存在错位。
5. 接受：当前实现与 motivation 的“分布驱动、三维风险建模”目标仍有差距。
6. 接受（补充）：benchmark trace 的请求级元数据链路本身是通的，但不等于默认评测命中 predictor-aware proxy。

本轮未发现需要明确“拒绝”的 yib 关键结论。

---

## 2. 逐条回复

## 2.1 关于逻辑 2 表述偏满

处理：接受。

源码证据：

1. `predictor_worker_readyflag.py` 使用 `lengths.append(min(self.max_req_len, seq_len))`。
2. 随后 `k = lengths[i]`，并执行 `torch.topk(imp, k=k, sorted=False)`。
3. 这意味着当 `seq_len <= max_req_len` 时，`k = seq_len`，不会发生截断筛选；仅当 `seq_len > max_req_len` 时发生截断。

结论修订：

- 逻辑 2 应写为“importance 驱动的 top-k 截断输入已实现，但短请求可能整体保留”。

---

## 2.2 关于逻辑 4 语义

处理：接受。

源码证据：

1. `p2p_nccl_connector.py` 在 defer 模式下仅缓存 `pending_layers`，并由 `_start_kv_flush_thread()` 后台循环调用 `_flush_ready_kv()`。
2. `wait_for_save()` 中，`self.defer_kv_send=True` 时不会执行 `wait_for_sent()`。

结论修订：

- 逻辑 4 应描述为“异步 defer + 后台 flush 等待路由后发送 KV”，不是“prefill 前向线程阻塞等待路由”。

---

## 2.3 关于 qwen 中间层层号偏差

处理：接受。

源码证据：

1. `qwen2.py` 当前导出条件为 `idx==7`。
2. 同时 `qwen2.py` 确实固定了 `PredictorWorker(max_req_len=510)`，与 BERT 512 上限适配思路一致。

结论：

- max_req_len=510 可以认为是合理适配。
- 但 qwen 的层号仍与设计说明（期望 13）不一致，应单独修正。

---

## 2.4 关于 active proxy 入口与默认评测入口错位

处理：接受。

源码证据：

1. predictor-aware 闭环在 `test_disagg_proxy.py`：包含 predictor metadata Future、`Future_Tokens` 计算、`select_best_instance(..., predictor_meta)` 与路由广播。
2. `disagg_proxy_p2p_nccl_xpyd.py` 为基础轮询逻辑：prefill/decode 都按 `count % len(list)` 选点，并把 prefill `max_tokens` 置为 1。
3. `test_evaluation.sh` 默认 `PROXY_SCRIPT=${PROXY_BASE_DIR}/disagg_proxy_p2p_nccl_xpyd.py`。

结论：

- 若不覆写 `PROXY_SCRIPT`，默认评测不会命中 predictor-aware 路由链路。

---

## 2.5 关于 benchmark 请求级元数据链路

处理：接受（补充，不推翻原主结论）。

源码证据：

1. `benchmark_serving_trace.py` 读取并组装：`req_id`、`prompt_len/prompt_tokens`、`output_tokens`、`temperature/top_p/top_k/repetition_penalty`。
2. 同文件构造 `RequestFuncInput` 时，已把 `req_id`、`prompt_len`、`output_len` 和采样参数传入。
3. `backend_request_func.py` 的 openai-chat 路径会写入 `x-req-id`、`prompt_len`、`max_completion_tokens` 与采样参数。

结论：

- benchmark trace 元数据链路是打通的。
- 但该事实与“默认 proxy 是否走 predictor-aware 路由”是两个独立问题。

---

## 2.6 关于 motivation 对齐度

处理：接受。

源码证据：

1. `predictor_worker_readyflag.py` 当前核心输出仍是 `predict_len = miu + sigma` 单值。
2. proxy/scheduler 侧主要消费 `predictor_meta['predict_output_len']` 单值。

结论：

- 当前是实验级闭环已打通。
- 但距离“分布驱动、三维风险协同”的论文目标还有工程差距。

---

## 3. 已落盘更新

1. 已更新文件：`/root/predict-schedule/paper/evaluation/vllm_implementation/a_status_review_2026-04-25.md`
2. 更新内容：
   - 收紧逻辑 2 表述（补充短请求边界）。
   - 修正逻辑 4 语义（异步 defer + 后台 flush）。
   - 补充 qwen 层号偏差问题。
   - 补充默认评测入口错位问题。
   - 新增“针对 yib 反馈的复核处理结果”章节。

---

## 4. 后续执行建议（按优先级）

1. 先修正确性：
   - `p2p_nccl_connector.py` 中异常 `req_id` 路径未定义变量风险。
   - scheduler/aimd 的 `predictor_meta` 空值保护。
   - predictor 参数索引函数默认分支。
2. 再修入口一致性：
   - 统一 active proxy 与评测脚本默认入口。
3. 再做目标对齐：
   - 将分布风险（而非单值）接入 proxy 路由与 decode 准入。
