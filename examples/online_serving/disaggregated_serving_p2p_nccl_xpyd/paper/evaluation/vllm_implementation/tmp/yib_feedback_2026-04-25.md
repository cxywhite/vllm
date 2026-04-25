# yib同学反馈：PD分离下预测器与两阶段调度源码复核

- 时间：2026-04-25
- 作者：yib同学
- 代码范围：/root/predict-schedule/vllm
- 参考文档：
  - /root/predict-schedule/paper/motivation/motivation_meeting_note.md
  - /root/predict-schedule/paper/evaluation/vllm_implementation/vllm_implementation.md
  - /root/predict-schedule/paper/evaluation/vllm_implementation/a_status_review_2026-04-25.md
- 复核方式：仅基于源码只读检查，不直接采信 a 同学结论

---

## 1. 我对你当前工作的理解

从 motivation meeting note 看，你要解决的不是“在 vLLM 上加一个 predictor”这么简单，而是一个由未来输出长度信息驱动的两阶段协同调度问题：

1. 在 Prefill 阶段尽早拿到请求未来输出长度信息。
2. Proxy 在 P2D 路由时，不只看当前瞬时负载，而要看请求落地后的未来瓶颈风险。
3. Decode 实例内调度不只做一步能不能跑，而要控制 waiting 请求以什么节奏进入 running。
4. 更进一步，你的 motivation 明确希望长度信息最好是“带不确定性/分布信息”的，而不是单个点预测值。

因此，判断“代码是否完成”，不能只看 predictor 能不能跑通，还要看：

1. 这 5 条链路是否真的打通。
2. 打通的是不是 active 实现路径。
3. 实现是否与 motivation 中的目标一致。

---

## 2. 对 5 条逻辑的源码复核结论

## 2.1 在 prefill 实例上部署 predictor

结论：**基本已实现，但存在角色边界风险。**

源码依据：

1. `llama.py` 已导入 `PredictorWorker` 和 ring buffer 组件。
2. `LlamaModel.__init__` 中，当 `scheduler_config.activation_predict` 为真时，会初始化 `GPURingBuffer` 并启动 `PredictorWorker`。
3. `gpu_model_runner.py` 在 `activation_predict` / `test_p2d` / `test_optimal` 打开时，已经为请求构造 `Custom_Metadata` 并传给模型前向。

问题：

1. predictor 初始化条件当前只绑定 `activation_predict`，没有在初始化处显式绑定 `kv_role == 'kv_producer'`。
2. `llama.py` 真正写 ring buffer 时确实又加了 `kv_role == 'kv_producer'` 判断，但 predictor worker 和 ring buffer 本身已经被创建并启动。

这意味着：

1. 如果 decode 侧也错误打开 `activation_predict`，可能也会额外加载 predictor、占用显存并启动线程。
2. 所以这条逻辑不能简单写成“已实现”，更准确是“实现存在，但部署角色边界靠配置自律保证”。

---

## 2.2 在前向过程中根据 attention 分数筛选关键 token 作为 predictor 输入

结论：**已基本实现，但 a 同学这里表述偏满，应该收紧。**

源码依据：

1. `llama.py` 的 attention 前向中，在 `activate_predict` 且 `compute_importance=True` 时，会按请求片段计算 `token_importance`。
2. 在 `LlamaModel.forward` 中，只在 `kv_role == 'kv_producer'` 且命中 `target_layer_idx = 15` 时，才会计算 importance 并将 `hidden_states + importance + metadata` 写入 ring buffer。
3. `predictor_worker_readyflag.py` 中，predictor 消费 ring buffer 后，会按 importance 做 `topk` 选 token。

需要收紧的地方：

1. 当前不是“所有请求都基于 attention 分数筛出关键 token”。
2. `predictor_worker_readyflag.py` 使用的是：
   - `k = min(self.max_req_len, seq_len)`
   - 再做 `torch.topk(imp, k=k)`
3. 这意味着：
   - 当 `seq_len <= max_req_len` 时，`k = seq_len`，实际上会保留该请求的全部 token，而不是发生真正的“筛选”。
   - 只有当 `seq_len > max_req_len` 时，attention importance 才真正起到截断筛选作用。

因此，我更倾向于把这条逻辑写成：

- **已实现“importance 驱动的 top-k 截断输入”，但并非所有请求都会发生真正的 token 筛选；短请求会整体保留。**

此外，还有两个实现边界：

1. 你补充说明的设计意图是：由于 predictor 内含 BERT，最大输入长度是 512，因此有效 token 上限写成 510，预留 `[CLS]/[SEP]`；并且只使用模型的中间层激活与中间层 attention 分数。
2. 这一点和当前源码的总体方向是一致的：
   - `llama.py` 中 `PredictorWorker(max_req_len=510)` 已固定为 510。
   - `qwen2.py` 中 `PredictorWorker(max_req_len=510)` 也已固定为 510。
   - 因此，`max_req_len=510` 本身不应判为拍脑袋硬编码，而应理解为对 BERT 512 长度上限的显式适配。
3. 但“固定中间层”的实现目前并不完全一致：
   - `llama.py` 明确使用 `target_layer_idx = 15`。
   - 你补充说明里要求 `qwen` 使用索引 13。
   - 但当前 `qwen2.py` 实际写入 ring buffer 的条件是 `idx == 7`，和“索引 13”并不一致。
4. 所以这部分不能写成“llama/qwen 都已按设计完成”，更准确应写成：
   - **llama 中间层导出逻辑与设计一致；qwen 路径当前实现与设计说明存在层号偏差，需要单独修正。**
2. 重要度计算本身仍要构造单请求片段上的 `score [s, s]` 矩阵，代价仍是 $O(s^2)$。

---

## 2.3 predictor 结果传到 proxy，proxy 结合 decode 负载选实例

结论：**实验链路已打通，但 active 入口和 motivation 对齐度都有明显限制。**

源码依据：

1. `test_disagg_proxy.py` 中存在 predictor metadata 的异步接收与 Future 管理。
2. 同文件从 `/metrics` 中读取：
   - `running_tokens`
   - `running_predict_tokens`
   - `waiting_tokens`
   - `kv_cache_usage_perc`
3. 同文件内部会构造 `Future_Tokens = running_predict_tokens + waiting_tokens`。
4. `select_best_instance(..., predictor_meta)` 已将 `predict_output_len` 叠加到 `Future_Tokens` 参与选点。
5. 路由结果会通过 `routing_pub.send_string(...)` 广播，并把 `predictor_meta` 注入 decode 请求。

这里我同意 a 同学“逻辑打通了”的判断，但有两个重要补充：

### 2.3.1 active 路由入口漂移

完整 predictor 闭环当前在：

- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy.py`

而不是：

- `examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py`

后者当前仍是：

1. prefill 设置 `max_tokens = 1`
2. prefill 按轮询选择
3. decode 按轮询选择
4. prefill 完成后直接转发 decode

所以这条逻辑不是“源码里已经有了就算完成”，而是：

- **只有运行 `test_disagg_proxy.py` 这条实现链时，逻辑 3 才真正生效。**

这里还需要结合你的补充说明再收紧一次表述：

1. 你明确说明 `test_disagg_proxy.py` 是你改过但尚未补完的 predictor-aware proxy。
2. `disagg_proxy_p2p_nccl_xpyd.py` 只是 baseline proxy 之一，当前主要承担轮询方法基线，而不是 predictor-aware active proxy。
3. 因此，如果后续实验脚本没有显式切到 `test_disagg_proxy.py`，那它跑到的仍然是 baseline 路由，不应把结果解释为“预测器驱动路由已经被实际评测”。
4. 另外，按你的补充，`disagg_proxy_p2p_nccl_xpyd.py` 中已有的三维负载建模思路，后续可以继续向 `test_disagg_proxy.py` 迁移；从当前代码现状看，这一迁移还没有完成。

### 2.3.2 实现与 motivation 的目标仍有明显落差

motivation 里写的是“分布信息驱动 + 三维瓶颈感知路由”，但当前 proxy 侧实际实现是：

1. predictor 最终输出被压缩成单值 `predict_len = miu + sigma`。
2. 路由侧消费的是 `predict_output_len` 单值，而不是完整分布或风险区间。
3. `test_disagg_proxy.py` 中的负载模型只近似使用了：
   - 计算压力：基于 `n_req` 和 `m_tok`
   - 带宽压力：`weights_io + kv_io`
   - 容量压力：直接用 `kv_usage`
4. 它没有把 motivation 里显式写到的 `M_act` 纳入带宽项，也没有把 `C_weights + C_act + C_kv` 作为容量项显式建模。

所以这部分更准确的结论应该是：

- **P2D 路由实验链路已打通，但当前实现仍主要是“点预测 + 近似瓶颈负载”的实验版本，还没有完整落到 motivation 中要求的“分布驱动、三维风险感知路由”。**

---

## 2.4 prefill 侧延迟 KV，等 proxy 回传 decode 路由后再发到指定实例

结论：**已实现“延迟发送 KV”语义，但 a 同学把它写成“阻塞直到收到路由”不够准确。**

源码依据：

1. `p2p_nccl_connector.py` 中，当 `activation_predict=True` 且 `kv_role == 'kv_producer'` 时，会打开 `defer_kv_send`。
2. `save_kv_layer()` 在 defer 模式下只缓存 `pending_layers`，不立即发送。
3. connector 内部启动了 routing listener 维护 `routing_table`。
4. 另有独立 flush 线程轮询 `pending_layers`，拿到路由后再发送到指定 decode。
5. `wait_for_save()` 在 defer 模式下不会阻塞 prefill 侧等待发送完成。

所以从“功能结果”看，你要的逻辑 4 已经做到了：

- KV 不会在 prefill 完成时立刻发出，而是等 proxy 广播 decode 路由后再定向发出。

但从实现语义上，当前更准确的描述是：

- **异步延迟发送 KV**

而不是：

- **prefill 线程阻塞直到拿到路由**

因为当前实现里：

1. prefill 前向不会在 `wait_for_save()` 被卡住。
2. 真实等待动作发生在后台 flush 线程上。

此外，这条链路确实存在 a 同学指出的两个实锤问题：

1. 当 `req_id` 不符合预期格式时，只记日志但后续仍使用 `part1/part2`，存在未定义变量风险。
2. `pending_layers` 与 `routing_table` 当前都没有容量上界或过期回收，路由异常时有内存增长风险。

---

## 2.5 decode 实例内按节奏控制 waiting -> running

结论：**已实现 AIMD 风格的节奏控制，但仅在显式切换到 AIMDScheduler 时生效，且仍是点预测驱动。**

源码依据：

1. `scheduler.py` 中确实支持 `aimd_scheduler=True` 时切换到 `AIMDScheduler`。
2. `aimd_scheduler.py` 中维护了：
   - `virtual_mem`
   - `physical_mem_free`
   - `waiting_times`
   - `no_preempt_time`
3. `schedule()` 中会根据 running 请求的剩余长度估计、waiting 请求插入后的最大 token 需求和 `virtual_mem` 进行准入控制。
4. 发生 preempt 时会提高 `waiting_times`，随后通过 `waiting_times -= 1` 的方式在若干轮内抑制 waiting 请求继续进入 running。

因此，“按节奏控制 waiting -> running” 这一条从调度器代码上看是成立的。

但需要补充两点：

1. 它只在 `AIMDScheduler` 被显式启用时生效，不是默认 decode 调度路径。
2. 它当前消费的仍然是 `predictor_meta['predict_output_len']` 单值，不是分布风险信息。

所以从 motivation 对齐角度看，这部分属于：

- **AIMD 版实例内节奏控制已经实现，但尚未升级为“分布/风险驱动的准入控制”。**

---

## 3. 对 a 同学报告的总体评价

结论：**a 报告整体方向基本可信，但有 3 处需要收紧或补充。**

### 3.1 我同意的部分

1. 逻辑 1、3、4、5 的主链路基本都能在源码中找到。
2. active proxy 入口漂移、P2P flush 健壮性、predictor_meta 空值保护、ring busy wait、高频监控等问题，a 报告列得基本对。
3. “当前实现与 motivation 的分布驱动目标仍有落差”这个大方向判断也是对的。

### 3.2 我认为需要修正的部分

1. 逻辑 2 不能直接写成“已实现”，应补一句边界：
   - 只有长请求才发生真正的 top-k 截断筛选；短请求会整体保留。
2. 逻辑 4 不应表述为“prefill 侧阻塞直到 proxy 回传路由”，更准确是：
   - connector 延迟发送 KV，后台 flush 线程等待路由后再定向发出。
3. 3.3 节“与 motivation 一致性”的问题列得还不够完整：
   - 当前不只是“没有分布接入”，连 proxy 负载模型本身也还没有完整实现 motivation 里写的三维瓶颈公式。

---

## 4. 我确认存在的问题

## 4.1 实现错误 / 健壮性问题

1. `p2p_nccl_connector.py` 的 flush 路径在异常 `req_id` 格式下存在 `part1/part2` 未定义风险。
2. `predictor_worker_readyflag.py` 中 `get_index_temperature/get_index_topp/get_index_topk/get_index_repetition_penalty` 都没有默认返回分支，未枚举采样参数可能导致后续索引异常。
3. `scheduler.py` 与 `aimd_scheduler.py` 中多处直接读取 `req.predictor_meta['predict_output_len']`，缺少空值和字段缺失保护。
4. predictor 初始化与 `kv_role` 解耦，错误配置下可能在 decode 侧也加载 predictor。

## 4.2 真实测试中时间 / 空间开销偏大的问题

1. `llama.py` 中 importance 计算仍要构造单请求片段上的 `score [s, s]`，代价仍是 $O(s^2)$。
2. `wt_gpu_ring_buffer.py` 写侧获取 slot 使用忙等，没有超时或降级策略，满队列时会把反压直接传回 prefill 前向。
3. `predictor_worker_readyflag.py` 使用多线程共享单 predictor、伴随高频 `print` 和持续 CSV 写入，容易在高 QPS 下放大扰动。
4. `test_disagg_proxy.py` 以 `10ms` 周期逐实例抓 `/metrics`，实例数增加时监控本身会变重。
5. `p2p_nccl_connector.py` 中 `pending_layers` 和 `routing_table` 缺少上界与回收机制，路由延迟或异常时会累积内存压力。

## 4.3 其他实现问题（工程与目标一致性）

1. active predictor 路由闭环在 `test_disagg_proxy.py`，不在常见命名入口 `disagg_proxy_p2p_nccl_xpyd.py`，工程入口存在漂移风险。
2. predictor 输入筛选逻辑是“importance 驱动的 top-k 截断”，不是所有请求都真的筛掉非关键 token。
3. predictor 输出当前仍被压缩成单值 `miu + sigma`，路由和准入侧消费的也是单值 `predict_output_len`。
4. proxy 当前负载模型只实现了部分三维瓶颈近似：
   - capacity 直接拿 `kv_usage`
   - memory 只统计 `weights_io + kv_io`
   - 没有把 motivation 中写的 `M_act`、`C_weights + C_act + C_kv` 完整建模进去
5. 多处存在硬编码路径、设备和 ZMQ 地址，可移植性较差。

## 4.4 baseline 与评测脚本链路需要一并复核

这一部分建议 a 同学一起分析，因为它直接决定“你测出来的到底是哪条实现链”。

1. `test_evaluation.sh` 当前默认配置是：
   - `PROXY_SCRIPT=${PROXY_BASE_DIR}/disagg_proxy_p2p_nccl_xpyd.py`
   - `BENCH_SCRIPT=.../benchmarks/benchmark_serving_trace.py`
   - `BENCH_BACKEND=openai-chat`
   - `BENCH_ENDPOINT=/v1/chat/completions`
2. 这意味着：
   - 如果不手工覆写 `PROXY_SCRIPT`，默认评测跑到的是 baseline 轮询 proxy，而不是 `test_disagg_proxy.py`。
   - 所以“代码里已经有 predictor-aware proxy”与“默认实验脚本实际测到了 predictor-aware proxy”是两件事，当前默认并不等价。
3. `benchmark_serving_trace.py` 已经把 trace/CSV 中的以下字段打通到了请求对象：
   - `req_id`
   - `prompt_len` 或 `prompt_tokens`
   - `output_tokens`
   - `temperature`
   - `top_p`
   - `top_k`
   - `repetition_penalty`
4. 同文件在构造 `RequestFuncInput` 时，已经把：
   - `req_id`
   - `prompt_len`
   - `max_tokens`
   - 采样参数
   一起放进请求链路，因此 benchmark 侧并不是只在“发一个 prompt 字符串”，而是已经具备传递请求级元数据的能力。
5. `backend_request_func.py` 的 openai-chat 请求路径会进一步把：
   - `x-req-id` 请求头
   - `prompt_len`
   - `max_completion_tokens`
   - `temperature/top_p/top_k/repetition_penalty`
   写进实际 HTTP 请求；返回对象里也会回填这些字段。
6. `benchmark_dataset.py` 里的 `SampleRequest` 数据结构也已经补上了 `req_id` 和采样参数字段，但要注意：
   - 在当前 trace 回放主链路里，真正负责把 CSV 字段填进 `SampleRequest` 的主逻辑是在 `benchmark_serving_trace.py`。
   - `benchmark_dataset.py` 更多是“数据结构层面的支持”，不能单独拿它证明“trace 字段已经被完整用起来了”。
7. 因此，这组 baseline/评测脚本的更准确结论应写成：
   - **请求级 trace 元数据和采样参数链路基本已经打通。**
   - **但默认评测入口仍指向 baseline proxy，导致 predictor-aware 路由逻辑与默认实验脚本之间存在入口错位。**

---

## 5. 最终判断

如果按你提出的 5 条逻辑逐条判断：

1. 逻辑 1：**基本完成**，但存在部署角色边界风险。
2. 逻辑 2：**基本完成**，但应明确这是“有阈值的 top-k 截断筛选”，不是对所有请求都发生真实筛选。
3. 逻辑 3：**实验链路完成**，但只在 `test_disagg_proxy.py` 这条 active 路径下成立；默认评测脚本并不会自动测到这条链，而且它本身也还不是 motivation 中目标的完整实现。
4. 逻辑 4：**完成延迟 KV 发送**，但实现语义更准确是异步 defer + 后台 flush，不是严格的 prefill 阻塞式等待路由。
5. 逻辑 5：**AIMD 节奏控制已完成**，但需显式启用，而且仍是点预测驱动。

综合结论：

- 这套功能链路已经基本打通。
- 但如果按 motivation 中“分布信息驱动的两阶段协同调度”来衡量，目前更准确的状态不是“已经完整做完”，而是：
  - **实验级闭环已经成立**
   - **qwen 中间层导出与设计说明仍有偏差**
   - **默认评测入口与 predictor-aware proxy 仍有错位**
  - **工程入口和健壮性仍需修正**
  - **距离论文目标中的分布驱动、多维风险协同调度还有一段差距**

---

## 6. 我建议优先修复的顺序

1. 先修正确性：
   - `p2p_nccl_connector.py` 的异常 `req_id` 路径
   - `predictor_meta` 空值保护
   - predictor 参数映射默认分支
   - predictor 初始化的角色边界
2. 再修工程可用性：
   - active proxy 入口统一
   - pending KV / routing_table 上界与回收
   - ring buffer 降级策略
3. 最后做目标对齐：
   - 把 predictor 的分布/风险信息而不只是 `miu + sigma` 单值接入 proxy 路由与 AIMD 准入
   - 把 proxy 负载模型补到更接近 motivation 中的三维瓶颈定义
