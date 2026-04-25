# a同学总结：PD分离下预测器与两阶段调度实现状态复盘

- 时间：2026-04-25
- 作者：a同学
- 代码范围：/root/predict-schedule/vllm
- 参考文档：
  - /root/predict-schedule/paper/motivation/motivation_meeting_note.md
  - /root/predict-schedule/paper/evaluation/vllm_implementation/vllm_implementation.md

---

## 1. 我对 motivation 的理解

你当前工作不是单点优化，而是一个统一目标：

1. 在 PD 分离服务中，把 预测输出长度（且最好是带不确定性的分布信息） 作为跨实例路由和实例内调度的共同信号。
2. 跨实例侧：根据请求落地后的未来瓶颈风险做 P2D 决策，而不是只看当前瞬时负载。
3. 实例内侧：控制 waiting 到 running 的准入节奏，避免激进与保守两端导致 Goodput 下降。

从 meeting note 看，核心主线是：
- 预测器输出不应只是一条长度点值，而应尽量表达风险（分布/方差信息）。
- 路由与准入都应消费这类风险信息，形成两阶段协同。

---

## 2. 你要求的 5 条逻辑：完成度判定

## 2.1 在 prefill 实例部署 predictor

判定：已实现（但有配置边界风险）

证据：
1. predictor 相关组件已接入模型文件：
   - vllm/vllm/model_executor/models/llama.py:89, 90
2. activation_predict 打开时会初始化并启动 PredictorWorker：
   - vllm/vllm/model_executor/models/llama.py:468, 469, 475

说明：
- 逻辑实现存在，但初始化条件仅绑定 activation_predict，没有显式绑定 kv_role==kv_producer。
- 这意味着如果 decode 侧也打开 activation_predict，可能也会加载 predictor（带来额外显存和线程开销）。

---

## 2.2 前向过程中用 attention 分数筛选关键 token 作为 predictor 输入

判定：已实现（但需说明筛选边界）

证据：
1. 在 Llama attention 中计算 token_importance：
   - vllm/vllm/model_executor/models/llama.py:283, 284, 287, 292
2. 在目标层把 hidden_states + importance 写入 ring buffer：
   - vllm/vllm/model_executor/models/llama.py:544, 546
3. predictor 侧按 importance 做 top-k token 选择：
   - examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/predictor_worker_readyflag.py:297, 299

边界说明：
- predictor 实际选择逻辑是 `k=min(max_req_len, seq_len)` 后再做 `topk`。
- 因此当 `seq_len<=max_req_len` 时会保留该请求全部 token，不发生真正截断；只有长请求（`seq_len>max_req_len`）才发生 importance 驱动筛选。

---

## 2.3 predictor 结果传到 proxy，proxy 结合 decode 负载选实例

判定：已实现（但实现文件与常见入口存在漂移风险）

证据（闭环在 test_disagg_proxy.py 中）：
1. predictor metadata 异步接收与 Future 管理：
   - examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy.py:42, 350
2. 读取 running_predict_tokens 与 waiting_tokens，计算 Future_Tokens：
   - examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy.py:215, 271, 278
3. 按 predictor_meta + 负载做实例选择：
   - examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy.py:281, 283, 452
4. 路由广播 + predictor_meta 注入 decode 请求：
   - examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy.py:464, 470

关键提醒：
- 同目录的 disagg_proxy_p2p_nccl_xpyd.py 当前仍是基础轮询转发形态（非完整 predictor 闭环）：
  - examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py:437, 448, 457
- 如果线上脚本启动了 disagg_proxy_p2p_nccl_xpyd.py 而不是 test_disagg_proxy.py，逻辑 3 实际不会生效。
- 默认评测脚本 `test_evaluation.sh` 的 `PROXY_SCRIPT` 仍指向 `disagg_proxy_p2p_nccl_xpyd.py`，不是 `test_disagg_proxy.py`；如不覆写变量，默认评测不会命中 predictor-aware 路由链路。

---

## 2.4 prefill 侧延迟 KV 发送，待 proxy 回传 decode 路由后再定向发送

判定：已实现（但有健壮性和内存风险）

证据：
1. defer_kv_send 与缓存队列初始化：
   - vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:98, 101, 103
2. 路由监听与 routing_table：
   - vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:113, 127
3. save_kv_layer 在 defer 模式仅缓存：
   - vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:305, 340, 344
4. flush 线程按路由发送：
   - vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:138, 170
5. wait_for_save 在 defer 模式不阻塞发送等待：
   - vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:362, 365

语义澄清：
- 当前实现是“异步 defer + 后台 flush 线程等待路由后发送”，不是“prefill 前向线程阻塞到路由到达”。

---

## 2.5 decode 实例内按节奏控制 waiting -> running

判定：已实现（AIMD 版本，需显式启用）

证据：
1. AIMDScheduler 已实现节奏变量与虚拟内存门控：
   - vllm/vllm/v1/core/sched/aimd_scheduler.py:56, 79, 277, 519, 526
2. waiting 阶段依据虚拟内存/预测长度进行准入控制：
   - vllm/vllm/v1/core/sched/aimd_scheduler.py:299, 312

注意：
- 该逻辑依赖调度器切换到 AIMDScheduler；未切换时不会生效。

---

## 3. 问题清单（按严重性）

## 3.1 实现错误/健壮性问题

1. p2p flush 中 req_id 异常格式会触发潜在未定义变量错误
- 位置：vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:163-169
- 问题：当 req_id 不含预期分隔符时，只记录 error，但后续仍使用 part1/part2 拼接 tensor_id，存在运行时异常风险。

2. predictor 参数离散映射函数无兜底分支
- 位置：examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/predictor_worker_readyflag.py:63, 80, 93, 106
- 问题：get_index_* 系列函数对未枚举采样参数没有默认返回，后续构造 tensor 索引时可能出现异常。

3. scheduler/aimd 对 predictor_meta 字段读取缺少空值保护
- 位置：
  - vllm/vllm/v1/core/sched/scheduler.py:1039, 1042
  - vllm/vllm/v1/core/sched/aimd_scheduler.py:240, 299, 795, 797
- 问题：直接访问 req.predictor_meta['predict_output_len']，若某请求无 predictor_meta 或字段缺失，会触发 KeyError。

4. predictor 部署条件与角色条件解耦，可能误在 decode 侧加载
- 位置：vllm/vllm/model_executor/models/llama.py:468（初始化），523-524（仅 producer 才真正写入）
- 问题：初始化时只看 activation_predict，不看 kv_role；可能造成 decode 侧不必要模型加载与线程启动。

---

## 3.2 真实压测中时间/空间开销偏大的实现点

1. attention importance 计算仍有 O(s^2) 代价
- 位置：vllm/vllm/model_executor/models/llama.py:287
- 说明：按每个请求片段计算 score 矩阵，序列长时计算和显存带宽压力显著。

2. ring buffer 写入为忙等，满队列时会阻塞前向
- 位置：examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_gpu_ring_buffer.py:51, 52, 58
- 说明：没有超时/降级策略，slot 紧张时会把延迟反压到 prefill 前向。

3. predictor 端并发与 I/O 开销偏高
- 位置：
  - examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/predictor_worker_readyflag.py:153, 224（8 线程）
  - 同文件:361（高频 print）
  - 同文件:193-507（持续 CSV 写入）
- 说明：多线程共享单模型 GPU 推理 + 高频日志与落盘，在高 QPS 下会放大调度抖动。

4. proxy 监控轮询频率高（10ms）且逐实例抓 /metrics
- 位置：examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/test_disagg_proxy.py:195, 199, 201, 205, 317
- 说明：decode 实例数增加时，监控本身会消耗可观 CPU/网络开销。

5. defer KV 队列缺少容量上界与过期回收
- 位置：
  - vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py:103, 140, 143, 175
  - routing_table 写入:127
- 说明：路由延迟或异常时，pending KV tensor 和路由表可能持续增长，带来内存风险。

---

## 3.3 其他实现问题（工程与目标一致性）

1. active 路由实现文件与常规入口文件漂移
- 现状：完整 predictor 闭环在 test_disagg_proxy.py，不在 disagg_proxy_p2p_nccl_xpyd.py。
- 影响：启动脚本选错入口时，功能看起来“代码有了但运行无效”。

2. 多处硬编码路径和地址影响可移植性
- 例如 predictor 权重路径、ZMQ 地址 127.0.0.1:32323/32324、模型配置路径等。
- 影响：跨机部署、容器重构或目录变更时容易失效。

3. 与 motivation 中“分布信息驱动调度”仍有落差
- 位置：predictor 输出核心仍是单值 predict_len=miu+sigma：
  - examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/predictor_worker_readyflag.py:339-341
- 影响：当前调度侧主要消费单值预测，尚未把完整分布/风险区间系统性接入两阶段决策。

4. qwen 中间层导出不仅层号偏差，且实现语义与 llama 不一致
- 位置：
   - qwen 当前写入 ring buffer 条件是局部索引 `idx==7`：vllm/vllm/model_executor/models/qwen2.py:475
   - llama 采用全局层号比较：先计算 `layer_idx=self.start_layer+idx`，再比较 `layer_idx==target_layer_idx`：vllm/vllm/model_executor/models/llama.py:514, 519, 524
   - 设计说明期望 qwen 使用中间层索引 13。
- 影响：这不仅是“7 不等于 13”的层号偏差，还会在 pipeline partition 变化时引入导出层语义漂移，影响复现性与跨模型可比性。

5. 默认评测入口与 predictor-aware proxy 实现链错位
- 位置：examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/evaluation/test_evaluation.sh:122
- 现状：默认 `PROXY_SCRIPT` 指向 `disagg_proxy_p2p_nccl_xpyd.py`（基础轮询版本）。
- 影响：若不覆写 `PROXY_SCRIPT`，实验结果不能直接解释为“predictor-aware 路由已被评测”。

6. openai-chat 评测请求同时携带 `max_completion_tokens` 与 `max_tokens`
- 位置：
  - sampling 参数里写入 `max_tokens`：benchmarks/benchmark_serving_trace.py:568
  - openai-chat payload 先设置 `max_completion_tokens`，随后 `payload.update(extra_body)` 再写入 `max_tokens`：benchmarks/backend_request_func.py:417, 426
- 影响：不同服务端对两字段优先级处理可能不同，导致接口兼容性与评测解释存在歧义风险。

---

## 4. 结论（是否完成）

按你提出的 5 条修改目标：

1. 逻辑 1：已完成（有角色边界风险）
2. 逻辑 2：已完成（importance 驱动 top-k 截断；短请求可能不截断）
3. 逻辑 3：已完成（但只在 test_disagg_proxy.py 这条实现链）
4. 逻辑 4：已完成（异步 defer + 后台 flush，不是 prefill 线程阻塞等待）
5. 逻辑 5：已完成（AIMD 版本，需显式启用）

综合判断：
- 功能链路已经基本打通。
- 但要进入稳定可复现实验阶段，还需要先处理第 3 节中列出的高优先级问题，尤其是：
  - p2p flush 的异常路径健壮性
  - predictor_meta 空值保护
   - active 入口统一（proxy 与评测脚本同时对齐）
  - 关键开销点（O(s^2) 计算、忙等阻塞、监控/日志/落盘负担）

---

## 5. 针对 yib 反馈的复核处理结果（2026-04-25）

1. 关于“逻辑 2 表述偏满”
- 处理结论：接受。
- 复核依据：`predictor_worker_readyflag.py` 使用 `k=min(max_req_len, seq_len)`，短请求不会发生 token 截断。
- 已更新：2.2 节与 4 节结论已改为“importance 驱动 top-k 截断，短请求可能不截断”。

2. 关于“逻辑 4 语义应为异步 defer，不是 prefill 阻塞”
- 处理结论：接受。
- 复核依据：`wait_for_save` 在 defer 模式不等待发送完成，后台 flush 线程轮询并按路由发送。
- 已更新：2.4 节标题与语义说明已修改，4 节结论同步修订。

3. 关于“qwen 层号与设计说明不一致”
- 处理结论：接受。
- 复核依据：qwen 当前导出条件为 `idx==7`，而设计说明目标层号为 13。
- 已更新：3.3 节新增该问题，标记为需要单独修复项。

4. 关于“active proxy 与默认评测入口错位”
- 处理结论：接受。
- 复核依据：
   - predictor-aware 闭环在 `test_disagg_proxy.py`；
   - baseline 轮询 proxy 在 `disagg_proxy_p2p_nccl_xpyd.py`；
   - `test_evaluation.sh` 默认 `PROXY_SCRIPT` 指向 baseline。
- 已更新：2.3 与 3.3 已补充“代码存在不等于默认评测命中”的明确说明。

5. 关于“benchmark 元数据链路是否真实打通”
- 处理结论：接受（补充说明，不构成对原结论的推翻）。
- 复核依据：`benchmark_serving_trace.py` 已把 req_id/prompt_len/output_tokens/采样参数装入 `RequestFuncInput`，`backend_request_func.py` 会透传 `x-req-id`、`prompt_len`、`max_completion_tokens` 与采样参数。
- 已更新：本报告在“入口错位”结论中明确区分“请求元数据链路已通”与“默认 proxy 入口未对齐”是两件事。

6. 关于“逻辑 3/5 与 motivation 的分布驱动目标仍有差距”
- 处理结论：接受。
- 复核依据：当前 predictor 输出核心仍压缩为 `predict_len=miu+sigma` 单值，proxy/调度消费的也是单值 `predict_output_len`，尚未形成分布风险闭环。
- 已更新：3.3 与 4 节结论保留并强调该差距。

7. 关于“是否需要反驳 yib”
- 处理结论：本轮无明确反驳项。
- 理由：逐条复核后，yib 的关键修订意见均可由当前源码证据支持。

---

## 6. 针对 b 同学二轮反馈的复核处理结果（2026-04-25）

1. 关于“本轮暂不直接标记审核通过”
- 处理结论：接受。
- 复核依据：二轮指出的两个新增问题（qwen 层语义、openai-chat 双字段）均能被源码直接支持。
- 处理动作：已将两问题补入 3.3 节，并将其纳入后续修复优先级。

2. 关于“qwen 不只是层号偏差，还存在局部/全局层号语义不一致”
- 处理结论：接受。
- 复核依据：
   - qwen 用局部索引 `idx==7`；
   - llama 用全局层号 `layer_idx=self.start_layer+idx` 与 `target_layer_idx` 比较。
- 处理动作：3.3 的第 4 条已升级为“层号偏差 + 语义不一致 + partition 漂移风险”。

3. 关于“benchmark openai-chat 请求体字段并存风险”
- 处理结论：接受。
- 复核依据：`benchmark_serving_trace.py` 在 `extra_body` 写 `max_tokens`，`backend_request_func.py` 同时写 `max_completion_tokens` 且随后 `payload.update(extra_body)`。
- 处理动作：3.3 新增第 6 条，明确兼容性与解释歧义风险。

4. 关于“是否存在需要拒绝的二轮观点”
- 处理结论：本轮无拒绝项。
- 理由：二轮反馈的关键主张与源码行为一致，仅需在报告中补齐表述与问题清单。

---

## 7. 建议的最小修复顺序

1. 先修正确性：3.1 的 1/2/3/4。
2. 再降开销：3.2 的 1/2/3/4/5。
3. 最后做目标对齐：把长度分布风险信息明确接入路由与准入（而不是只用单值）。
