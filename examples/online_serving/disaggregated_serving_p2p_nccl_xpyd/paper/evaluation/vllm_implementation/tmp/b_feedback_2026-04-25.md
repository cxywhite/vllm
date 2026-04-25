# WT 源码改动复核反馈（针对 wt_source_audit_report_2026-04-25.md）

- 复核时间：2026-04-25
- 复核范围：/root/predict-schedule/vllm
- 复核依据：
  - /root/predict-schedule/WT Prefill Activation Predictor.md
  - /root/predict-schedule/vllm_implementation/wt_source_audit_report_2026-04-25.md
  - /root/predict-schedule/vllm 中源码只读复查
- 复核方式：仅使用只读检索与源码阅读；本文件为新增反馈，不修改 vllm 源码

---

## 1. 复核结论

总体上，kana 同学这份核查报告主体判断是成立的，没有发现与当前源码明显矛盾的重大结论。以下结论我复核后确认没有问题：

1. 以 `[WT]` 标记为口径统计，`/root/predict-schedule/vllm` 中包含 `[WT]` 的文件数为 **65**，这个总数与报告一致。
2. 文档登记的 `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_test.sh` 确实不在当前 `vllm` 仓库中。
3. `vllm/v1/core/sched/aimd_queue.py` 确实存在实现，但当前文件内没有 `[WT]` 标记。
4. `test_disagg_proxy.py` 才是当前 predictor metadata 汇聚、`Future_Tokens` 计算、decode 选点、路由广播的主链路载体；这一判断与报告一致。
5. `p2p_nccl_connector.py` 中的延迟 KV 发送、路由表监听、缓存 flush、`wait_for_save` 非阻塞改动，和报告描述基本一致。

但是，这份报告仍有两处需要补充或修正，否则会让“按原始 md 全量对照源码”的结果不够完整。

---

## 2. 复核后确认无问题的部分

### 2.1 调度类切换判断无误

`vllm/config/scheduler.py` 中当前仍然是：

- `pastfuture_scheduler=True` 时切到 `vllm.v1.core.sched.pastfuture_scheduler.FuturePastScheduler`
- `aimd_scheduler=True` 时切到 `vllm.v1.core.sched.aimd_scheduler.AIMDScheduler`

这说明 kana 同学报告里关于调度类名和配置切换的结论是正确的。

### 2.2 Proxy 主链路定位判断无误

`examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy.py` 中可以直接检索到以下关键实现：

- `pending_predictor_futures` 与别名管理
- `running_predict_tokens` 和 `waiting_tokens`
- `Future_Tokens`
- `select_best_instance(..., predictor_meta)`
- `_listen_predictor_metadata()`
- `routing_pub.send_string(...)`
- `original_request_data["predictor_meta"] = predictor_meta`

这条链路完整对应了“预测结果到达 -> decode 选点 -> 路由广播 -> predictor_meta 注入 decode 请求”的闭环，因此 kana 同学把核心逻辑归到 `test_disagg_proxy.py` 是对的。

### 2.3 延迟 KV 传输链路判断无误

`vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py` 中当前确实有以下行为：

1. `self.defer_kv_send = True` 的条件启用。
2. 启动 routing listener 维护 `routing_table`。
3. 启动独立 flush 线程处理缓存的 KV。
4. `save_kv_layer()` 在 defer 模式下仅缓存 `pending_layers`。
5. `wait_for_save()` 在 defer 模式下不阻塞 prefill 阶段。

因此报告第 2.3 节的功能归纳与源码相符。

---

## 3. 需要补充或修正的地方

### 3.1 漏记了 in-repo 的 wt_metadata.py 及其主链路依赖关系

这是本次复核里最重要的补充点。

原始文档 `WT Prefill Activation Predictor.md` 在“2.2 新增文件”里登记了 `wt_metadata.py`，但登记路径写的是：

- `predict-schedule/design_predict_activation_experiment/wt_metadata.py`

kana 同学的报告没有把它纳入 `vllm` 仓库复核结果，这在“按 `[WT]` 标记统计文件”这个口径下可以理解，因为该文件没有 `[WT]` 标记；但如果目标是“根据原始 md 对照当前源码完整梳理实现链路”，这一点应当补上，而且它不是旁支文件，而是主链路依赖文件。

原因如下：当前 `vllm` 仓库内部实际存在一个同名文件：

- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_metadata.py`

且这个文件已被主链路直接导入：

1. `vllm/v1/worker/gpu_model_runner.py`
   - 先将 `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd` 插入 `sys.path`
   - 再执行 `from wt_metadata import Custom_Metadata`
2. `vllm/model_executor/models/llama.py`
   - 同样将上述 examples 目录插入 `sys.path`
   - 再导入 `Custom_Metadata`
3. `vllm/model_executor/models/qwen2.py`
   - 也使用同样的导入路径
4. `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/predictor_worker_readyflag.py`
   - 直接 `from wt_metadata import Custom_Metadata`
5. `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_gpu_ring_buffer.py`
   - 直接 `from wt_metadata import Custom_Metadata`

这说明：

1. `wt_metadata.py` 并不是“文档提到但与当前 vllm 源码无关”的文件。
2. 当前活跃实现实际依赖的是 `vllm/examples/.../wt_metadata.py` 这份 in-repo 文件。
3. kana 同学报告若继续保留“仅按 `[WT]` 标记核查”的表述，没有问题；但若要作为“对原始 md 的完整源码映射”，应新增一条说明：
   - `wt_metadata.py` 在 `vllm` 仓库内存在 active copy，虽无 `[WT]` 标记，但被主链路直接依赖。

建议把这一点补到“不一致项”或“未登记但与方案强相关的核心文件”中。

### 3.2 对 disagg_proxy_p2p_nccl_xpyd.py 的现态描述应再收紧

kana 同学报告里已经指出：

- 文档把 proxy 关键逻辑登记在 `disagg_proxy_p2p_nccl_xpyd.py`
- 实际 predictor 路由闭环主要在 `test_disagg_proxy.py`

这个方向判断是对的，但从当前源码来看，表述还能更明确一些。

我复查 `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py` 后看到：

1. 该文件当前的 `[WT]` 内容主要是 self-check、decode load monitor 日志、配置打印等内容。
2. 文件中没有检索到以下主链路关键字：
   - `predictor_meta`
   - `routing_pub`
   - `Future_Tokens`
   - `running_predict_tokens`
   - `pending_predictor_futures`
3. 当前 `handle_request()` 逻辑本质上仍是：
   - prefill 只跑 1 token
   - prefill/decode 按轮询取实例
   - prefill 完成后直接将原始请求转发给 decode

因此，更准确的说法不是“核心链路主要在 `test_disagg_proxy.py`”，而是：

- **当前 active 的 predictor metadata 汇聚、负载感知 decode 选点、路由广播闭环不在 `disagg_proxy_p2p_nccl_xpyd.py` 中，而是在 `test_disagg_proxy.py` 中。**

也就是说，这里不是“主次关系”那么简单，而是“当前源码实现位置已经发生实质性漂移”。

---

## 4. 对 kana 同学报告统计口径的建议说明

我建议给 kana 同学这份报告补一句统计口径说明，否则读者容易把“`[WT]` 标记核查”误解成“所有实现依赖文件已完整覆盖”。

更准确的表述应为：

1. **65 个文件**是“包含 `[WT]` 标记的文件数”。
2. 这个口径会漏掉没有 `[WT]` 标记、但被方案主链路直接依赖的文件，例如：
   - `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_metadata.py`
   - `vllm/v1/core/sched/aimd_queue.py`
3. 因此这份报告作为“`[WT]` marker audit”是成立的；作为“原始 md 与当前实现的完整映射”，还需要补上无 marker 但活跃依赖的文件。

---

## 5. 最终反馈

最终判断如下：

1. kana 同学报告的主体结论没有大问题，可以继续作为梳理当前 WT 改动的基础文档使用。
2. 需要补充的关键点主要有 2 个：
   - 补记 `vllm/examples/.../wt_metadata.py` 这份无 `[WT]` 标记但被主链路直接依赖的 in-repo 文件。
   - 将 `disagg_proxy_p2p_nccl_xpyd.py` 的描述收紧为“当前 active predictor 闭环已不在该文件中”。
3. 除上述两点外，我没有发现 kana 同学报告中存在明显错误的代码定位、错误的功能归因或错误的文件统计。
