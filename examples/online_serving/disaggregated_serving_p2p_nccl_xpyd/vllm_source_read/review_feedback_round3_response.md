# 第三轮审核反馈回复（源码复核后）

回复时间: 2026-04-10  
回复对象: review_feedback_round3.md  
复核范围:
- disagg proxy: /root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py
- v1 engine/scheduler/model runner/mixin/p2p engine
- 当前调研文档: llama_prefill_decode_p2p_nccl_call_flow.txt

---

## 1. 结论总览（不盲信，逐条复核）

对第三轮提出的新问题，我的判定如下：

1. 2.1 最小调用图函数名不符: **成立**。  
2. 2.2 Step 10 `:916` 语义偏差: **成立**。  
3. 2.3 Step 11 “并行链路”措辞: **部分成立**。  
   - “不是并行”这点成立；
   - “在不同 step 触发”这点不成立（源码显示同一 step 内串行触发）。
4. 2.4 Step 8 超阈值逻辑漏写 else: **成立（补充性修正）**。

---

## 2. 逐条复核与证据

## 2.1 关于最小调用图中 prefill_request/decode_request

反馈判断：文档把不存在的函数名写入调用图。  
复核结果：**成立**。

证据：
- 真正存在的函数是 `forward_request`: `disagg_proxy_p2p_nccl_xpyd.py:413`
- prefill 转发调用：`:475`
- decode 转发调用：`:481`
- `prefill_request` 是局部变量（dict）构造：`:435`

已修正：调用图已改为
- `forward_request(prefill_url, prefill_request_dict, request_id)`
- `forward_request(decode_url, original_request_data, request_id)`

---

## 2.2 关于 Step 10 的 `:916` 与 `:1076`

反馈判断：`:916` 不是 token append。  
复核结果：**成立**。

证据：
- `update_from_output` 入口：`scheduler.py:903`
- `outputs` 变量声明：`scheduler.py:916`
- 真正 append token：`request.append_output_token_ids(...)` 在 `scheduler.py:1076`

已修正：Step 10 改为拆分两句：
- `update_from_output` 入口与输出容器构建（903, 916）
- `_update_request_with_output` 追加 token（1076）

---

## 2.3 关于 Step 11 “并行链路”措辞

反馈判断：不应写并行。并且反馈进一步称两条链路在不同 step 触发。  
复核结果：**部分成立**。

成立部分：
- “并行”措辞不准确，已修正为“串行、不同时机”。

不成立部分（需要纠正）：
- “不同 step 触发”不准确。

源码时序证据：
1. `EngineCore.step` 内顺序是：
   - 先 `execute_model_with_error_logging(...)`（`core.py:284`）
   - 后 `scheduler.update_from_output(...)`（`core.py:287`）
2. `get_finished` 在 `execute_model` 的 connector 上下文 `finally` 触发：
   - `kv_connector.get_finished(...)`（`kv_connector_model_runner_mixin.py:122`）
3. `request_finished` 在同一次 `update_from_output` 的停止分支触发：
   - `if stopped: kv_transfer_params = self._free_request(request)`（`scheduler.py:974-975`）
   - `_free_request -> _connector_finished -> connector.request_finished`（`scheduler.py:1189, 1265, 1277`）

结论：
- 两者不是并行；
- 但通常在**同一 step**内先后发生（先 get_finished，再 request_finished），不是“跨 step 必然分离”。

已修正：Step 11 已改为“同一 step 的不同时机串行触发”。

---

## 2.4 关于 Step 8 超阈值逻辑

反馈判断：应同时标注 if/else 两分支。  
复核结果：**成立（补充更完整）**。

证据：
- 计算 `tensor_size`: `p2p_nccl_engine.py:345`
- if 超阈值分支：`346-350`
- else 分支 `buffer_size += tensor_size`: `356`

已修正：Step 8 已补全 else 分支引用。

---

## 3. 已执行修改

已修改文件：
- /root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/vllm_source_read/llama_prefill_decode_p2p_nccl_call_flow.txt

本轮改动点：
1. 最小调用图改为真实函数名 `forward_request(...)`。  
2. Step 10 行号语义拆分（903/916 vs 1076）。  
3. Step 11 改为“同一 step 不同时机串行触发”，并去除“并行”表述。  
4. Step 8 增补 else 分支（:356）。

---

## 4. 最终说明

第三轮反馈总体质量高，新增问题中 2.1/2.2/2.4 成立，2.3 需做“同 step 串行 vs 不同 step”的精确化校正。当前文档已按源码事实修正完成。