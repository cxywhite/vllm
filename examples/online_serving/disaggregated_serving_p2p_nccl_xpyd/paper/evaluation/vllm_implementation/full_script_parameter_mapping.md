# 当前层脚本/参数/文件关系总表

- 统计时间: 2026-04-25
- 统计范围:
  - `/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd` 当前层目录下的 `.sh` / `.py`
  - `/root/predict-schedule/vllm/benchmarks` 当前目录下的 `.py`
- 统计目标:
  - 建立“参数与脚本映射”
  - 建立“脚本之间映射”
  - 建立“当前层 sh -> 当前层 py / benchmarks py”的完整入口关系
  - 建立“benchmark 入口 py -> backend/dataset/support py”的关系

说明:

1. 本文件是“结构总表”，重点是当前层脚本如何互相调用、如何把参数传进 Python 文件。
2. 对“死参数/伪可配置参数”的细粒度审计，已经在 `parameter_mapping_and_deprecation_audit_2026-04-25.md` 中单独展开；本文件不重复展开每一个风险结论，而是把结构关系一次性收齐。
3. 对未确认关系，本文件统一标成“当前未见当前层 sh 直接引用”或“当前层 py 未直接 grep 到 import”。
4. 本表中的依赖列按主链展示，不等价于完整 import 图；若某文件还存在辅助依赖，会在备注中单独注明。

---

## 1) 当前层 `.sh` 文件总表

| 脚本 | 角色 | 默认 benchmark py | proxy py 映射 | 直接使用的其他 py | 是否启动 `vllm serve` | 是否写 config json | 备注 |
|---|---|---|---|---|---|---|---|
| `disagg_example_p2p_nccl_xpyd.sh` | 最小示例/手工启动脚本 | 无 | 固定 `disagg_proxy_p2p_nccl_xpyd.py` | 无 | 是 | 否 | 直接拉起 proxy + prefill/decode，不跑 benchmark |
| `1-llama_baseline_vllm_3Dload.sh` | llama 基线实验脚本 | `benchmark_serving_timestamp.py` | `USE_CUSTOM_PROXY=true -> test_disagg_proxy.py`; `TEST_ABLATION_P2D=true -> test_disagg_proxy_p2d.py`; 否则 `disagg_proxy_p2p_nccl_xpyd.py` | `plot_decode_load2.py` | 是 | 是 | 1p1d 结构，benchmark 命令固定调用 openai-chat 路径 |
| `2-Loadbalance_llama_baseline_vllm_loadbalance1p3d_dualmode.sh` | llama 负载均衡实验脚本 | `benchmark_serving_load_balance.py` | `PROXY_SCRIPT_OVERRIDE` 最高优先级；其后 `test_disagg_proxy.py` / `test_disagg_proxy_p2d_loadbalance.py` / `test_disagg_proxy_p2d.py` / `disagg_proxy_p2p_nccl_xpyd.py` | `plot_decode_load2.py` | 是 | 是 | dual-mode 实验入口 |
| `2-Loandbalance_qwen_baseline_vllm_loadbalance1p3d_dualmode.sh` | qwen 负载均衡实验脚本 | `benchmark_serving_load_balance.py` | 代理选择逻辑与 llama 版本同族，但数据入口和部分控制流存在差异 | `plot_decode_load2.py` | 是 | 是 | qwen 版本；支持 `BENCH_DATASET_PATH_LIST` 数据集列表循环 |
| `3-Mean_and_sigma_llama_baseline_vllm_loadbalance1p3d_rr_optimal_dataset.sh` | mean/sigma 实验脚本 | `benchmark_serving_mean_and_sigma.py` | `RUN_METHOD=optimal -> test_disagg_proxy_p2d_mean_and_sigma.py`; `RUN_METHOD=rr -> disagg_proxy_p2p_nccl_xpyd.py`; 默认 `disagg_proxy_p2d_mean_and_sigma.py` | `plot_decode_load2.py` | 是 | 是 | 会自动从 mean_pred_sched 结果生成 optimal 数据集 |
| `4-Predict_latency_llama.sh` | predictor/时延实验脚本 | `benchmark_serving_load_balance.py` | `PROXY_SCRIPT_OVERRIDE` / `test_disagg_proxy.py` / `test_disagg_proxy_p2d_loadbalance.py` / `test_disagg_proxy_p2d.py` / `disagg_proxy_p2p_nccl_xpyd.py` | `plot_decode_load2.py` | 是 | 是 | `USE_CUSTOM_PROXY` 在该脚本中是活参数 |
| `5-DynamicQPS_llama_baseline_vllm_loadbalance1p3d_trace.sh` | trace / 动态 QPS 实验脚本 | `benchmark_serving_trace.py` | `PROXY_SCRIPT_OVERRIDE` / `PROXY_SCRIPT_CUSTOM=test_disagg_proxy.py` / `PROXY_SCRIPT_OPTIMAL=test_disagg_proxy_p2d_loadbalance.py` / `PROXY_SCRIPT_ABLATION_P2D=test_disagg_proxy_p2d.py` / `PROXY_SCRIPT_DEFAULT=disagg_proxy_p2p_nccl_xpyd.py` | `plot_decode_load2.py` | 是 | 是 | trace 模式还支持目录级数据集遍历 |
| `5-DynamicQPS_test_1p1d_prefill_out_memory.sh` | 1p1d prefill OOM 边界测试脚本 | `benchmark_serving_load_balance.py` | 固定 `disagg_proxy_p2p_nccl_xpyd.py` | 无 | 是 | 是 | 简化版随机数据集测试 |
| `6-Random_inputlen_outputlen_qps_shard1.sh` | 随机输入/输出/QPS 组合实验脚本 | `benchmark_serving_baseline.py` | 固定 `disagg_proxy_p2p_nccl_xpyd.py` | `plot_decode_load2.py` | 是 | 是 | 自带 watchdog、skip-known-result、分片运行 |
| `6-Random_inputlen_outputlen_qps_shard2.sh` | 随机输入/输出/QPS 组合实验脚本 | `benchmark_serving_baseline.py` | 固定 `disagg_proxy_p2p_nccl_xpyd.py` | `plot_decode_load2.py` | 是 | 是 | 与 shard1 同构，只是分片索引不同 |
| `llama_vllm_roundroubin_1p3d.sh` | llama round-robin 基线脚本 | `benchmark_serving_baseline.py` | 固定 `disagg_proxy_p2p_nccl_xpyd.py` | `plot_decode_load2.py` | 是 | 是 | 额外向 proxy 注入 `TTFT` 与 `TPOT` env |
| `qwen_vllm_roundroubin_1p3d.sh` | qwen round-robin 基线脚本 | `benchmark_serving_baseline.py` | 固定 `disagg_proxy_p2p_nccl_xpyd.py` | `plot_decode_load2.py` | 是 | 是 | qwen 版本，同样注入 `TTFT` 与 `TPOT` env |
| `start_vllm_instance.sh` | 手工拉起多实例工具脚本 | 无 | 无 | 无 | 是 | 否 | 只启动多个独立 `vllm serve`，不走 proxy/benchmark |
| `start_vllm_instance_mean_and_sigma.sh` | 手工拉起多实例工具脚本 | 无 | 无 | 无 | 是 | 否 | mean/sigma 场景的手工实例启动脚本 |

---

## 2) 当前层 `.sh` -> 当前层 `.py` 映射

### 2.1 固定映射

| sh 脚本 | 固定调用/依赖的 py |
|---|---|
| `disagg_example_p2p_nccl_xpyd.sh` | `disagg_proxy_p2p_nccl_xpyd.py` |
| `5-DynamicQPS_test_1p1d_prefill_out_memory.sh` | `disagg_proxy_p2p_nccl_xpyd.py` |
| `6-Random_inputlen_outputlen_qps_shard1.sh` | `disagg_proxy_p2p_nccl_xpyd.py`, `plot_decode_load2.py` |
| `6-Random_inputlen_outputlen_qps_shard2.sh` | `disagg_proxy_p2p_nccl_xpyd.py`, `plot_decode_load2.py` |
| `llama_vllm_roundroubin_1p3d.sh` | `disagg_proxy_p2p_nccl_xpyd.py`, `plot_decode_load2.py` |
| `qwen_vllm_roundroubin_1p3d.sh` | `disagg_proxy_p2p_nccl_xpyd.py`, `plot_decode_load2.py` |

### 2.2 条件映射

| sh 脚本 | 条件变量 | 可选 proxy py |
|---|---|---|
| `1-llama_baseline_vllm_3Dload.sh` | `USE_CUSTOM_PROXY`, `TEST_ABLATION_P2D` | `test_disagg_proxy.py`, `test_disagg_proxy_p2d.py`, `disagg_proxy_p2p_nccl_xpyd.py` |
| `2-Loadbalance_llama_baseline_vllm_loadbalance1p3d_dualmode.sh` | `PROXY_SCRIPT_OVERRIDE`, `USE_CUSTOM_PROXY`, `TEST_OPTIMAL`, `TEST_ABLATION_P2D` | `test_disagg_proxy.py`, `test_disagg_proxy_p2d_loadbalance.py`, `test_disagg_proxy_p2d.py`, `disagg_proxy_p2p_nccl_xpyd.py` |
| `2-Loandbalance_qwen_baseline_vllm_loadbalance1p3d_dualmode.sh` | 同上 | 同上 |
| `4-Predict_latency_llama.sh` | `PROXY_SCRIPT_OVERRIDE`, `USE_CUSTOM_PROXY`, `TEST_OPTIMAL`, `TEST_ABLATION_P2D` | `test_disagg_proxy.py`, `test_disagg_proxy_p2d_loadbalance.py`, `test_disagg_proxy_p2d.py`, `disagg_proxy_p2p_nccl_xpyd.py` |
| `5-DynamicQPS_llama_baseline_vllm_loadbalance1p3d_trace.sh` | `PROXY_SCRIPT_OVERRIDE`, `USE_CUSTOM_PROXY`, `TEST_OPTIMAL`, `TEST_ABLATION_P2D` | `test_disagg_proxy.py`, `test_disagg_proxy_p2d_loadbalance.py`, `test_disagg_proxy_p2d.py`, `disagg_proxy_p2p_nccl_xpyd.py` |
| `3-Mean_and_sigma_llama_baseline_vllm_loadbalance1p3d_rr_optimal_dataset.sh` | `RUN_METHOD` | `test_disagg_proxy_p2d_mean_and_sigma.py`, `disagg_proxy_p2d_mean_and_sigma.py`, `disagg_proxy_p2p_nccl_xpyd.py` |

补充:

1. `3-Mean_and_sigma...sh` 中虽然保留了 `PROXY_SCRIPT_OVERRIDE` / `USE_CUSTOM_PROXY` 等旧变量与注释分支，但当前真实控制流是 `RUN_METHOD`。
2. 多个脚本都会在 benchmark 结束后调用 `plot_decode_load2.py` 处理 proxy 日志，但这属于后处理工具，不参与在线请求路径。

---

## 3) 当前层 `.sh` -> `benchmarks/*.py` 映射

| sh 脚本 | 默认 benchmark py | 典型用途 |
|---|---|---|
| `1-llama_baseline_vllm_3Dload.sh` | `benchmark_serving_timestamp.py` | 3Dload / timestamp 型 benchmark |
| `2-Loadbalance_llama_baseline_vllm_loadbalance1p3d_dualmode.sh` | `benchmark_serving_load_balance.py` | 负载均衡 benchmark |
| `2-Loandbalance_qwen_baseline_vllm_loadbalance1p3d_dualmode.sh` | `benchmark_serving_load_balance.py` | qwen 负载均衡 benchmark |
| `3-Mean_and_sigma_llama_baseline_vllm_loadbalance1p3d_rr_optimal_dataset.sh` | `benchmark_serving_mean_and_sigma.py` | mean/sigma 数据集与路由实验 |
| `4-Predict_latency_llama.sh` | `benchmark_serving_load_balance.py` | predictor-aware 时延实验 |
| `5-DynamicQPS_llama_baseline_vllm_loadbalance1p3d_trace.sh` | `benchmark_serving_trace.py` | trace 回放 / 动态 QPS |
| `5-DynamicQPS_test_1p1d_prefill_out_memory.sh` | `benchmark_serving_load_balance.py` | 简化 1p1d OOM 边界测试 |
| `6-Random_inputlen_outputlen_qps_shard1.sh` | `benchmark_serving_baseline.py` | 随机输入/输出长度与 QPS 组合实验 |
| `6-Random_inputlen_outputlen_qps_shard2.sh` | `benchmark_serving_baseline.py` | 随机输入/输出长度与 QPS 组合实验 |
| `llama_vllm_roundroubin_1p3d.sh` | `benchmark_serving_baseline.py` | llama round-robin 基线 |
| `qwen_vllm_roundroubin_1p3d.sh` | `benchmark_serving_baseline.py` | qwen round-robin 基线 |
| `disagg_example_p2p_nccl_xpyd.sh` | 无 | 仅起服务，不跑 benchmark |
| `start_vllm_instance.sh` | 无 | 工具脚本 |
| `start_vllm_instance_mean_and_sigma.sh` | 无 | 工具脚本 |

---

## 4) 当前层 `.py` 文件总表

### 4.1 直接被当前层 `.sh` 启动的 proxy / 入口 py

| py 文件 | 角色 | 被哪些 sh 直接启动 |
|---|---|---|
| `disagg_proxy_p2p_nccl_xpyd.py` | baseline proxy / P2P NCCL 转发代理 | `disagg_example_p2p_nccl_xpyd.sh`, `1-llama...sh`, `2-Loadbalance...sh`, `2-Loandbalance_qwen...sh`, `3-Mean_and_sigma...sh(RR)`, `4-Predict_latency_llama.sh`, `5-DynamicQPS...trace.sh(默认)`, `5-DynamicQPS_test_1p1d_prefill_out_memory.sh`, `6-Random...shard1.sh`, `6-Random...shard2.sh`, `llama_vllm_roundroubin_1p3d.sh`, `qwen_vllm_roundroubin_1p3d.sh` |
| `test_disagg_proxy.py` | 自定义 proxy / predictor-aware 路由实验 | `1-llama...sh`, `2-Loadbalance...sh`, `2-Loandbalance_qwen...sh`, `4-Predict_latency_llama.sh`, `5-DynamicQPS...trace.sh` |
| `test_disagg_proxy_p2d.py` | P2D ablation proxy | `1-llama...sh`, `2-Loadbalance...sh`, `2-Loandbalance_qwen...sh`, `4-Predict_latency_llama.sh`, `5-DynamicQPS...trace.sh` |
| `test_disagg_proxy_p2d_loadbalance.py` | P2D load-balance / optimal proxy | `2-Loadbalance...sh`, `2-Loandbalance_qwen...sh`, `4-Predict_latency_llama.sh`, `5-DynamicQPS...trace.sh` |
| `disagg_proxy_p2d_mean_and_sigma.py` | mean/sigma 默认 proxy | `3-Mean_and_sigma...sh` |
| `test_disagg_proxy_p2d_mean_and_sigma.py` | mean/sigma optimal proxy | `3-Mean_and_sigma...sh` |
| `plot_decode_load2.py` | proxy 日志绘图工具 | 被多数实验 sh 间接调用，但不是主流程服务进程 |

### 4.2 当前层 py 之间已确认的 import 关系

| 被 import 的文件 | import 它的当前层 py | 关系说明 |
|---|---|---|
| `wt_metadata.py` | `predictor_worker_readyflag.py`, `wt_gpu_ring_buffer.py`, `disagg_proxy_p2d_mean_and_sigma.py`, `test_disagg_proxy_p2d.py`, `test_disagg_proxy_p2d_loadbalance.py`, `test_disagg_proxy_p2d_loadbalance_threahold.py`, `test_disagg_proxy_p2d_mean_and_sigma.py` | 当前层 py 中最明确的共享数据结构模块，提供 `Custom_Metadata` |

当前未在本次当前层 grep 中直接看到的关系:

1. 当前层 py 对 `predictor_worker_readyflag.py` 的直接 import 未在当前层 grep 到。
2. 当前层 py 对 `wt_gpu_ring_buffer.py` 的直接 import 未在当前层 grep 到。
3. `disagg_proxy_p2p_nccl_xpyd.py` 本身未直接 import `wt_metadata.py`，它和 `test_disagg_proxy_p2d*.py` 不是同一条元数据链实现。

### 4.3 当前层支持模块 / 工具 / 说明文件

| py 文件 | 分类 | 当前状态 |
|---|---|---|
| `wt_metadata.py` | 支持模块 | 提供 `Custom_Metadata`，被多份 p2d / predictor 相关 py import |
| `wt_gpu_ring_buffer.py` | 支持模块 | 提供 `GPURingBuffer`，当前未见当前层 sh 直接调用 |
| `predictor_worker_readyflag.py` | 支持模块 | predictor worker 实现，当前未见当前层 sh 直接调用 |
| `static_dataset_range.py` | 数据工具 | 当前未见当前层 sh 直接调用 |
| `test_avoid_oom_logic.py` | 测试/验证脚本 | 当前未见当前层 sh 直接调用 |
| `test_disagg_proxy_p2d_loadbalance_threahold.py` | 实验 proxy 变体 | 当前未见当前层 sh 直接调用 |
| `disagg_proxy_p2p_nccl_xpyd_modeling_notes.py` | 说明/笔记型文件 | 当前未见当前层 sh 直接调用 |
| `disagg_proxy_p2p_nccl_xpyd_reviewfix.py` | reviewfix 变体文件 | 当前未见当前层 sh 直接调用 |

---

## 5) `benchmarks/*.py` 文件总表

### 5.1 被当前层 `.sh` 直接调用的 benchmark 入口 py

| benchmark py | 当前层 sh 直接引用 | 直接 import 的支撑模块 |
|---|---|---|
| `benchmark_serving_timestamp.py` | `1-llama_baseline_vllm_3Dload.sh` | `backend_request_func_timestamp.py`, `benchmark_dataset.py`, `customfunction.py`, `benchmark_utils.py`；另显式从 `backend_request_func.py` 导入 `get_tokenizer` |
| `benchmark_serving_load_balance.py` | `2-Loadbalance...sh`, `2-Loandbalance_qwen...sh`, `4-Predict_latency_llama.sh`, `5-DynamicQPS_test_1p1d_prefill_out_memory.sh` | `backend_request_func.py`, `benchmark_dataset.py`, `customfunction.py`, `benchmark_utils.py` |
| `benchmark_serving_mean_and_sigma.py` | `3-Mean_and_sigma...sh` | `backend_request_func_mean_and_sigma.py`, `benchmark_dataset_mean_and_sigma.py`, `customfunction.py`, `benchmark_utils.py` |
| `benchmark_serving_trace.py` | `5-DynamicQPS_llama_baseline_vllm_loadbalance1p3d_trace.sh` | `backend_request_func.py`, `benchmark_dataset.py`, `customfunction.py`, `benchmark_utils.py` |
| `benchmark_serving_baseline.py` | `6-Random...shard1.sh`, `6-Random...shard2.sh`, `llama_vllm_roundroubin_1p3d.sh`, `qwen_vllm_roundroubin_1p3d.sh` | `backend_request_func.py`, `benchmark_dataset.py`, `customfunction.py`, `benchmark_utils.py` |

### 5.2 benchmark 支撑模块

| 文件 | 分类 | 说明 |
|---|---|---|
| `backend_request_func.py` | backend 请求模块 | openai/openai-chat 等通用请求实现 |
| `backend_request_func_timestamp.py` | backend 请求模块 | timestamp 版本请求实现 |
| `backend_request_func_mean_and_sigma.py` | backend 请求模块 | mean/sigma 版本请求实现 |
| `backend_request_func_dataset.py` | backend 请求模块 | dataset 变体支撑模块 |
| `benchmark_dataset.py` | dataset 模块 | 通用数据集入口 |
| `benchmark_dataset_mean_and_sigma.py` | dataset 模块 | mean/sigma 数据集入口 |
| `benchmark_dataset_dataset.py` | dataset 模块 | dataset 专用变体 |
| `benchmark_utils.py` | 工具模块 | benchmark 结果转换与通用工具 |
| `customfunction.py` | 自定义工具模块 | 数据集加载、输出落盘、样本保存等 |

### 5.3 当前未见当前层 `.sh` 直接引用的 benchmark 入口/实验 py

| 文件 | 分类 | 当前状态 |
|---|---|---|
| `benchmark_serving.py` | benchmark 入口 | 当前未见当前层 sh 直接引用 |
| `benchmark_serving_ablation_decode.py` | benchmark 入口 | 当前未见当前层 sh 直接引用 |
| `benchmark_serving_baseline_v2.py` | benchmark 入口 | 当前未见当前层 sh 直接引用 |
| `benchmark_serving_baseline_v3.py` | benchmark 入口 | 当前未见当前层 sh 直接引用 |
| `benchmark_serving_batchsize.py` | benchmark 入口 | 当前未见当前层 sh 直接引用 |
| `benchmark_serving_dataset.py` | benchmark 入口 | 当前未见当前层 sh 直接引用 |
| `benchmark_serving_predict_dataset.py` | benchmark 入口 | 当前未见当前层 sh 直接引用 |
| `benchmark_serving_structured_output.py` | benchmark 入口 | 当前未见当前层 sh 直接引用 |
| `benchmark_block_pool.py` | benchmark 入口/实验 | 当前未见当前层 sh 直接引用 |
| `benchmark_latency.py` | benchmark 入口/实验 | 当前未见当前层 sh 直接引用 |
| `benchmark_long_document_qa_throughput.py` | benchmark 入口/实验 | 当前未见当前层 sh 直接引用 |
| `benchmark_ngram_proposer.py` | benchmark 入口/实验 | 当前未见当前层 sh 直接引用 |
| `benchmark_prefix_caching.py` | benchmark 入口/实验 | 当前未见当前层 sh 直接引用 |
| `benchmark_prioritization.py` | benchmark 入口/实验 | 当前未见当前层 sh 直接引用 |
| `benchmark_throughput.py` | benchmark 入口/实验 | 当前未见当前层 sh 直接引用 |

### 5.4 其他 benchmark 目录文件

| 文件 | 分类 | 当前状态 |
|---|---|---|
| `run_select_qwen_dataset.py` | 数据集工具 | 当前未见当前层 sh 直接引用 |
| `test1.py` | 测试/临时文件 | 当前未见当前层 sh 直接引用 |
| `test2.py` | 测试/临时文件 | 当前未见当前层 sh 直接引用 |

---

## 6) 参数簇 -> 脚本 / Python 落点映射

### 6.1 shell 变量 -> proxy env

这是当前层 sh 和当前层 proxy py 之间最稳定的一条链。

| 参数簇 | 典型 shell 变量 | 典型落点 py |
|---|---|---|
| proxy 监听与暴露端口 | `PROXY_PORT`, `BENCH_PORT` | `disagg_proxy_p2p_nccl_xpyd.py`, `test_disagg_proxy.py`, `test_disagg_proxy_p2d.py`, `test_disagg_proxy_p2d_loadbalance.py`, `disagg_proxy_p2d_mean_and_sigma.py`, `test_disagg_proxy_p2d_mean_and_sigma.py` |
| proxy 负载模型参数 | `MODEL_CONFIG_PATH`, `VLLM_DTYPE`, `TPOT` | baseline proxy 与若干 test proxy |
| 特殊基线阈值 | `TTFT`, `TPOT` | `llama_vllm_roundroubin_1p3d.sh`, `qwen_vllm_roundroubin_1p3d.sh` 启动 `disagg_proxy_p2p_nccl_xpyd.py` 时会额外注入 |

### 6.2 shell 变量 -> `vllm serve` CLI

| 参数簇 | 典型 shell 变量 | 典型消费位置 |
|---|---|---|
| 模型与 dtype | `MODEL`, `VLLM_DTYPE`, `TEST_MODEL` | prefill/decode `vllm serve` 命令 |
| 部署拓扑 | `PREFILL_GPUS`, `DECODE_GPUS`, `PREFILL_PORTS`, `DECODE_PORTS`, `PREFILL_KV_PORTS`, `DECODE_KV_PORTS` | 当前层大多数实验 sh |
| 调度容量 | `VLLM_MAX_MODEL_LEN`, `PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS`, `DECODE_VLLM_MAX_NUM_BATCHED_TOKENS`, `VLLM_MAX_NUM_SEQS` | 当前层大多数实验 sh |
| GPU 使用率 | `PREFILL_GPU_MEMORY_UTILIZATION`, `DECODE_GPU_MEMORY_UTILIZATION`, `GPU_MEMORY_UTILIZATION` | 当前层大多数实验 sh |
| KV 传输 | `KV_CONNECTOR`, `KV_PRODUCER_BUFFER`, `KV_CONSUMER_BUFFER`, `KV_NCCL_CHANNELS`, `KV_SEND_TYPE`, `*_TENSOR_POOL_MEMORY`, `*_MEM_POOL_SIZE_GB` | 当前层大多数实验 sh，通过 `--kv-transfer-config` 进入 vLLM |
| 调度/实验开关 | `USE_PASTFUTURE_SCHEDULER`, `USE_ACTIVATION_PREDICTOR`, `USE_AIMD_SCHEDULER`, `TEST_ABLATION_P2D`, `TEST_OPTIMAL` | 条件拼接为 `--pastfuture-scheduler`, `--activation-predict`, `--aimd-scheduler`, `--test-p2d`, `--test-optimal` |

### 6.3 shell 变量 -> benchmark CLI

| 参数簇 | 典型 shell 变量 | 典型 benchmark py |
|---|---|---|
| benchmark 入口选择 | `BENCH_SCRIPT` | 所有实验 sh |
| 模型与端口 | `BENCH_MODEL`, `BENCH_PORT`, `BENCH_ENDPOINT` | `benchmark_serving_load_balance.py`, `benchmark_serving_trace.py`, `benchmark_serving_baseline.py`, `benchmark_serving_timestamp.py`, `benchmark_serving_mean_and_sigma.py` |
| 数据集（直接进入 benchmark CLI） | `BENCH_DATASET_NAME`, `BENCH_DATASET_PATH` | 多数 benchmark 入口 py |
| 请求规模（shell sweep + 直传 CLI） | `NUM_PROMPTS_LIST`, `BENCH_NUM_PROMPTS`, `BENCH_REQUEST_RATE_LIST`, `BENCH_REQUEST_RATES`, `BENCH_MAX_TOKENS_LIST`, `OUTPUT_TOKENS` | benchmark 最终接收的是 shell 展开后的标量参数，如 `--num-prompts`、`--request-rate`、`--maxtokenscustom`、`--random-output-len` |
| 采样参数 | `BENCH_TEMPERATURE`, `BENCH_TOP_P`, `BENCH_TOP_K`, `BENCH_REPETITION_PENALTY` | 多数 benchmark 入口 py，经 `RequestFuncInput.extra_body` 继续进入请求 payload |
| goodput / 阈值（直传 CLI + 部分 proxy env） | `BENCH_GOODPUT`, `BENCH_GOODPUT_TTFT_MS`, `BENCH_GOODPUT_TPOT_MS`, `BENCH_GOODPUT_TPOT_MS_LIST`, `CACULATE_GOODPUT` | benchmark 通过 `--goodput` 做统计；部分脚本还会把 `CACULATE_GOODPUT` 或展开后的 TPOT/TTFT 标量注入 proxy env |
| 其他开关 | `SAVE_OUTPUT`, `IGNORE`, `SAVE_SAMPLE`, `USE_TRACE_TIMESTAMPS`, `BENCH_BURSTINESS`, `BENCH_READY_CHECK_TIMEOUT_SEC`, `ENABLE_BENCH_SKIP_KNOWN_RESULT` | 分布在不同 benchmark 入口与外层 sh 控制流中 |

补充:

1. `BENCH_DATASET_PATH_LIST` 是 qwen dualmode 脚本的 shell 侧数据集列表循环变量；真实传给 benchmark 的仍然是每轮回写后的 `BENCH_DATASET_PATH`。
2. `BENCH_DATASET_RECURSIVE` 是 round-robin / trace 等脚本在 shell 内部控制目录遍历深度的变量；benchmark 入口最终接收的仍然是 `--dataset-path`。
3. `NUM_PROMPTS_LIST`、`BENCH_REQUEST_RATE_LIST`、`BENCH_MAX_TOKENS_LIST`、`BENCH_REQUEST_RATES` 这类变量大多先在 shell 内部展开循环，benchmark 实际接收的是每轮展开后的标量值，而不是这些列表变量本身。
4. `OUTPUT_TOKENS` 仅在 `5-DynamicQPS_test_1p1d_prefill_out_memory.sh` 这类随机数据集脚本中落到 `--random-output-len`，不属于多数 benchmark 的通用数据集参数。
5. `BENCH_GOODPUT_TPOT_MS_LIST` 主要用于 round-robin 类脚本的 shell 侧 sweep；benchmark 侧接收的是每轮展开后的 `ttft/tpot` goodput 阈值，而 proxy 侧接收的是同步展开后的 TTFT/TPOT env。

### 6.4 benchmark 入口 py -> 支撑 py

| benchmark 入口 py | backend 请求模块 | dataset 模块 | 其他常见支撑模块 |
|---|---|---|---|
| `benchmark_serving_timestamp.py` | `backend_request_func_timestamp.py` | `benchmark_dataset.py` | `benchmark_utils.py`, `customfunction.py`；另显式从 `backend_request_func.py` 导入 `get_tokenizer` |
| `benchmark_serving_load_balance.py` | `backend_request_func.py` | `benchmark_dataset.py` | `benchmark_utils.py`, `customfunction.py` |
| `benchmark_serving_mean_and_sigma.py` | `backend_request_func_mean_and_sigma.py` | `benchmark_dataset_mean_and_sigma.py` | `benchmark_utils.py`, `customfunction.py` |
| `benchmark_serving_trace.py` | `backend_request_func.py` | `benchmark_dataset.py` | `benchmark_utils.py`, `customfunction.py` |
| `benchmark_serving_baseline.py` | `backend_request_func.py` | `benchmark_dataset.py` | `benchmark_utils.py`, `customfunction.py` |

---

## 7) 参数审计摘要

本节补充汇总 [parameter_mapping_source_review_2026-04-25.md](parameter_mapping_source_review_2026-04-25.md) 中已经由源码复核确认的高置信结论。定位是对前述结构映射的补充，而不是替代；这里只记录“看起来能配、但当前脚本未真实驱动行为”或“只在局部脚本中失效”的参数结论。

| 参数 / 参数组 | 作用范围 | 当前判断 | 核心原因 |
|---|---|---|---|
| `BENCH_MAX_CONCURRENCY` | 脚本 1 / 3 / 4 / 5 | 伪可配置参数 | benchmark 下游支持 `--max-concurrency`，但这些脚本实际执行命令时固定写死 `--max-concurrency 1024`，没有把 shell 变量真正透传下去 |
| `VLLM_ENFORCE_EAGER` | 脚本 1 / 3 / 4 / 5 / 6 | 伪可配置参数 | 脚本虽然定义变量并写入 config json，但启动 `vllm serve` 时固定带 `--enforce-eager`，变量值不参与真实条件分支 |
| `USE_CUSTOM_PROXY` / `PROXY_SCRIPT_OVERRIDE` | 仅脚本 3 | 已失效，但只限脚本 3 | 脚本 3 的 `update_proxy_script()` 现在只看 `RUN_METHOD`；不过脚本 4 / 5 中这两个参数仍然活跃，不能外推成全局废弃 |
| `BENCH_RANDOM_INPUT_LEN` / `BENCH_RANDOM_OUTPUT_LEN` / `BENCH_REQUEST_RATE` | 仅脚本 6 | 单值参数已失效 | 脚本 6 真正驱动组合的是 `BENCH_RANDOM_INPUT_LENS`、`BENCH_RANDOM_OUTPUT_LENS`、`BENCH_REQUEST_RATES`；benchmark 最终接收的是 shell 循环展开后的标量，不是这 3 个单值变量 |
| `BENCH_READY_CHECK_TIMEOUT_SEC` | 仅脚本 6 | 当前不驱动真实 benchmark 行为 | 变量会被记录到日志和 config json，但没有透传到 benchmark CLI；`benchmark_serving_baseline.py` 也没有对应参数入口 |

补充:

1. 本节只并入“高置信伪可配置/失效参数”结论，没有把原审计文档中的全部证据链和治理建议原样展开。
2. `BENCH_MAX_CONCURRENCY` 和 `VLLM_ENFORCE_EAGER` 的问题，本质上都不是下游入口不支持，而是当前 shell 命令拼装没有把变量值变成真实控制流。
3. 脚本 6 中随机压测参数需要区分“shell 内部 sweep 列表变量”和“最终落到 benchmark CLI 的每轮标量值”；失效的是那 3 个单值变量，不是对应的列表 sweep 机制。

---

## 8) 最核心的两张总映射

### 8.1 当前层 sh -> 当前层 py -> benchmark py

| sh | proxy py | benchmark py |
|---|---|---|
| `disagg_example_p2p_nccl_xpyd.sh` | `disagg_proxy_p2p_nccl_xpyd.py` | 无 |
| `1-llama_baseline_vllm_3Dload.sh` | `test_disagg_proxy.py` / `test_disagg_proxy_p2d.py` / `disagg_proxy_p2p_nccl_xpyd.py` | `benchmark_serving_timestamp.py` |
| `2-Loadbalance_llama_baseline_vllm_loadbalance1p3d_dualmode.sh` | `test_disagg_proxy.py` / `test_disagg_proxy_p2d_loadbalance.py` / `test_disagg_proxy_p2d.py` / `disagg_proxy_p2p_nccl_xpyd.py` | `benchmark_serving_load_balance.py` |
| `2-Loandbalance_qwen_baseline_vllm_loadbalance1p3d_dualmode.sh` | `test_disagg_proxy.py` / `test_disagg_proxy_p2d_loadbalance.py` / `test_disagg_proxy_p2d.py` / `disagg_proxy_p2p_nccl_xpyd.py` | `benchmark_serving_load_balance.py` |
| `3-Mean_and_sigma_llama_baseline_vllm_loadbalance1p3d_rr_optimal_dataset.sh` | `test_disagg_proxy_p2d_mean_and_sigma.py` / `disagg_proxy_p2d_mean_and_sigma.py` / `disagg_proxy_p2p_nccl_xpyd.py` | `benchmark_serving_mean_and_sigma.py` |
| `4-Predict_latency_llama.sh` | `test_disagg_proxy.py` / `test_disagg_proxy_p2d_loadbalance.py` / `test_disagg_proxy_p2d.py` / `disagg_proxy_p2p_nccl_xpyd.py` | `benchmark_serving_load_balance.py` |
| `5-DynamicQPS_llama_baseline_vllm_loadbalance1p3d_trace.sh` | `test_disagg_proxy.py` / `test_disagg_proxy_p2d_loadbalance.py` / `test_disagg_proxy_p2d.py` / `disagg_proxy_p2p_nccl_xpyd.py` | `benchmark_serving_trace.py` |
| `5-DynamicQPS_test_1p1d_prefill_out_memory.sh` | `disagg_proxy_p2p_nccl_xpyd.py` | `benchmark_serving_load_balance.py` |
| `6-Random_inputlen_outputlen_qps_shard1.sh` | `disagg_proxy_p2p_nccl_xpyd.py` | `benchmark_serving_baseline.py` |
| `6-Random_inputlen_outputlen_qps_shard2.sh` | `disagg_proxy_p2p_nccl_xpyd.py` | `benchmark_serving_baseline.py` |
| `llama_vllm_roundroubin_1p3d.sh` | `disagg_proxy_p2p_nccl_xpyd.py` | `benchmark_serving_baseline.py` |
| `qwen_vllm_roundroubin_1p3d.sh` | `disagg_proxy_p2p_nccl_xpyd.py` | `benchmark_serving_baseline.py` |
| `start_vllm_instance.sh` | 无 | 无 |
| `start_vllm_instance_mean_and_sigma.sh` | 无 | 无 |

### 8.2 benchmarks 入口 py -> 支撑 py

| benchmark 入口 py | backend 请求 py | dataset py | 其他支撑 py |
|---|---|---|---|
| `benchmark_serving_timestamp.py` | `backend_request_func_timestamp.py` | `benchmark_dataset.py` | `customfunction.py`, `benchmark_utils.py`；另显式从 `backend_request_func.py` 导入 `get_tokenizer` |
| `benchmark_serving_load_balance.py` | `backend_request_func.py` | `benchmark_dataset.py` | `customfunction.py`, `benchmark_utils.py` |
| `benchmark_serving_mean_and_sigma.py` | `backend_request_func_mean_and_sigma.py` | `benchmark_dataset_mean_and_sigma.py` | `customfunction.py`, `benchmark_utils.py` |
| `benchmark_serving_trace.py` | `backend_request_func.py` | `benchmark_dataset.py` | `customfunction.py`, `benchmark_utils.py` |
| `benchmark_serving_baseline.py` | `backend_request_func.py` | `benchmark_dataset.py` | `customfunction.py`, `benchmark_utils.py` |

---

## 9) 结论

1. 当前层目录的主执行面其实非常集中: 大多数 `.sh` 最终都落到少数几个 proxy py 与少数几个 benchmark 入口 py。
2. 当前层 `.py` 的直接共享模块关系里，最稳定的是 `wt_metadata.py -> p2d/predictor 相关 py` 这条链；baseline proxy `disagg_proxy_p2p_nccl_xpyd.py` 是一条相对独立的实现链。
3. `benchmarks` 目录中，真正被当前层 `.sh` 直接消费的核心入口只有 5 个: `benchmark_serving_timestamp.py`、`benchmark_serving_load_balance.py`、`benchmark_serving_mean_and_sigma.py`、`benchmark_serving_trace.py`、`benchmark_serving_baseline.py`。
4. 其余 `benchmarks/*.py` 主干上可以分成两类: 一类是这些入口的 support module，另一类是当前未见当前层 `.sh` 直接引用的其他 benchmark/实验脚本；此外还有少量数据集工具与测试/临时文件。
