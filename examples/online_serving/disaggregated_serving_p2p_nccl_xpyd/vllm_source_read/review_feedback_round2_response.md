# 第二轮审核反馈回复（基于源码复核）

回复时间: 2026-04-10  
回复对象: review_feedback_round2.md  
涉及文档: llama_prefill_decode_p2p_nccl_call_flow.txt

---

## 1. 复核结论

本轮我没有盲目信任反馈内容，而是再次按源码逐点核对。结论如下：

1. Step 3、Step 5、Step 11 的修订正确性判断：**成立**。  
2. 新提出的“最小调用图里 `connector.build_connector_meta` 归属问题”：**核心建议成立，表述有一处措辞可更精确**。

---

## 2. 对“新问题 3.1”的逐条回应

## 2.1 源码事实

`connector.build_connector_meta` 的真实调用点在 Scheduler 内：
- `scheduler.py:648`
- 发生在 `Scheduler.schedule()` 方法中
- 先于 `GPUModelRunner.execute_model` 执行

这点完全正确。

## 2.2 对反馈措辞的校正

反馈写道“当前错误写法把它写在 GPUModelRunner.execute_model 之下”。

我核对了旧版调用图，旧版实际上是：
- `Scheduler.schedule`
- `connector.build_connector_meta`
- `GPUModelRunner.execute_model`

三者是同级顺序表达，不是缩进上的“GPUModelRunner 之下”。

所以更准确的说法应是：
- 旧版没有把 `build_connector_meta` 明确嵌入 `Scheduler.schedule` 内部语义层级；
- 容易导致读者误解为“独立于 schedule 的并列步骤”。

也就是说：
- 反馈的**修正建议方向正确**；
- 对旧文本“写在 GPUModelRunner 之下”的表述不够精确。

---

## 3. 已执行修正

我已直接修改最小调用图，将其明确为：

- `Scheduler.schedule`
  - `connector.build_connector_meta`
- `GPUModelRunner.execute_model`

修改文件：
- /root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/vllm_source_read/llama_prefill_decode_p2p_nccl_call_flow.txt

修正后层级与 `scheduler.py:648` 的实际调用归属一致。

---

## 4. 最终状态

- Step 3/5/11: 已正确。
- 最小调用图 build_connector_meta 层级: 已修正完成。
- 第二轮反馈中“核心技术结论”: 认可。
- 第二轮反馈中“个别措辞（GPUModelRunner 之下）”: 已做精确化说明。
