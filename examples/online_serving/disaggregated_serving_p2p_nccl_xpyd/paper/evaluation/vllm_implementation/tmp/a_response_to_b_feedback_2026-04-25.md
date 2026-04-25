# 对 b_feedback_2026-04-25.md 的逐条回应（基于源码复核）

- 回应时间：2026-04-25
- 复核范围：`/root/predict-schedule/vllm`
- 复核方式：仅只读命令（`rg`/`read`），不修改 `vllm` 源码
- 回应对象：`/root/predict-schedule/vllm_implementation/b_feedback_2026-04-25.md`

---

## 1. 回应结论总览

针对 b 报告提出的“需要补充或修正”的问题，本次复核结论如下：

1. 问题 3.1（漏记 in-repo `wt_metadata.py`）：**接受**
2. 问题 3.2（`disagg_proxy_p2p_nccl_xpyd.py` 现态描述应收紧）：**接受**
3. 问题 4（补充统计口径说明）：**接受**

本次没有需要“拒绝”的反馈点。

---

## 2. 逐条处理

## 2.1 问题 3.1：漏记 in-repo `wt_metadata.py`

- b 反馈主张：`wt_metadata.py` 虽无 `[WT]` 标记，但属于主链路依赖，应纳入报告。
- 复核结论：**接受**。
- 源码证据：
  1. 文件存在：
     - `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_metadata.py`
  2. 主链路导入命中：
     - `vllm/v1/worker/gpu_model_runner.py`: 141, 149, 150, 153
     - `vllm/model_executor/models/llama.py`: 80, 87, 91
     - `vllm/model_executor/models/qwen2.py`: 80, 87, 91
     - `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/predictor_worker_readyflag.py`: 15
     - `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_gpu_ring_buffer.py`: 10
  3. `[WT]` 标记检索：该文件当前无 `[WT]` 标记（marker 统计口径确实不会覆盖它）。
- 已执行修订：
  1. 在 `a_report_2026-04-25.md` 的“2.8 新增文件功能核查”中新增 `wt_metadata.py` 条目。
  2. 在“4. 未登记但与方案强相关的核心改动”中加入 `wt_metadata.py`。
  3. 在“1. 总体结论”补充统计口径说明。

---

## 2.2 问题 3.2：`disagg_proxy_p2p_nccl_xpyd.py` 现态描述应收紧

- b 反馈主张：不是“主要在 test_disagg_proxy.py”，而是 active predictor 闭环已不在 `disagg_proxy_p2p_nccl_xpyd.py`。
- 复核结论：**接受**。
- 源码证据：
  1. 在 `disagg_proxy_p2p_nccl_xpyd.py` 中检索以下关键词无命中：
     - `predictor_meta`
     - `routing_pub`
     - `Future_Tokens`
     - `running_predict_tokens`
     - `pending_predictor_futures`
     - `select_best_instance`
     - `_listen_predictor_metadata`
  2. 同文件 `handle_request` 关键行为命中：
     - `prefill_request["max_tokens"] = 1`（437）
     - prefill 轮询：`count % len(prefill_list)`（448）
     - decode 轮询：`count % len(decode_list)`（457）
     - prefill 完成后直接转发 decode（475, 481）
  3. 在 `test_disagg_proxy.py` 中上述闭环关键词有完整命中，例如：
     - `pending_predictor_futures`（42）
     - `running_predict_tokens`（215）
     - `Future_Tokens`（271, 278）
     - `select_best_instance`（281）
     - `_listen_predictor_metadata`（350）
     - `routing_pub.send_string`（464）
     - `original_request_data["predictor_meta"] = predictor_meta`（470）
- 已执行修订：
  1. `a_report_2026-04-25.md` 第 3 节第 2 条已改为更严格表述：
     - active predictor metadata 汇聚/负载选点/路由广播闭环不在 `disagg_proxy_p2p_nccl_xpyd.py`，而在 `test_disagg_proxy.py`。

---

## 2.3 问题 4：统计口径说明需补充

- b 反馈主张：应明确“65 文件”是 marker 口径，避免被误读为完整依赖覆盖。
- 复核结论：**接受**。
- 理由：
  1. marker 统计天然会漏掉无 `[WT]` 的活跃依赖（如 `wt_metadata.py`、`aimd_queue.py`）。
  2. 在“实现全景映射”场景下必须显式补这类文件。
- 已执行修订：
  1. `a_report_2026-04-25.md` 总体结论新增“统计口径说明”条目。

---

## 3. 对 b 报告其余内容的处理

1. b 报告中“主体结论基本成立”的判断，与本次复核一致。
2. b 报告未提出其他需要新增拒绝项的问题；本次未发现其关键结论与源码冲突。

---

## 4. 产出文件

1. 已更新：`/root/predict-schedule/vllm_implementation/a_report_2026-04-25.md`
2. 本回应：`/root/predict-schedule/vllm_implementation/c_response_to_b_feedback_2026-04-25.md`
