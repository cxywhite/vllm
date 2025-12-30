# WT Prefill Activation Predictor

## 1. 目录功能说明

本文件夹实现如下功能：

- 在 **prefill 实例** 中部署预测器  
- 接收来自模型的 **hidden_states、KV cache 以及 metadata**
- 基于上述信息执行激活与调度相关的预测逻辑

该设计用于配合 PD（Prefill/Decode）分离式推理架构，在不影响 decode 实例的前提下，于 prefill 阶段完成预测计算。

---

## 2. 文件修改内容与放置位置

### 2.1 修改文件

基于中间激活和kvcache预测修改内容通过以下标记进行标识：

```python
# [WT] predict activation 2025-12-19 13:55:34
# 修改内容
# [WT] end
```

具体修改文件与路径如下：

`arg_utils.py`和`scheduler.py`文件在predict-schedule/baseline_experiment/pastfuture目录下

- `arg_utils.py`  
  放置路径：  
  ```
  vllm/vllm/engine
  ```
  
- `scheduler.py`
  放置路径：  
  ```
  vllm/vllm/config
  ```

- `gpu_model_runner.py`  
  放置路径：  
  ```
  vllm/vllm/v1/worker/
  ```

- `llama.py`  
  放置路径：  
  ```
  vllm/vllm/model_executor/models/
  ```

- `wt_test.sh`  
  放置路径：  
  ```
  vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
  ```

proxy的p2d分发逻辑中kvcache延迟传输修改内容通过以下标记进行标识：

```python
# [WT]delay kvcache transfer 2025-12-27 22:09:30
# 修改内容
# [WT] end
```

具体修改文件与路径如下：

- `p2p_nccl_connector.py`  
  放置路径：  
  ```
  vllm/vllm/distributed/kv_transfer/kv_connector/v1/p2p
  ```

- `disagg_proxy_p2p_nccl_xpyd.py`  
  放置路径：  
  ```
  vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd
  ```

---

### 2.2 新增文件

本方案新增以下文件：

- `wt_gpu_ring_buffer.py`  
  放置路径：  
  ```
  vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
  ```

- `predictor_worker_readyflag.py`  
  放置路径：  
  ```
  vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
  ```

- `wt_metadata.py`  
  放置路径：  
  ```
  predict-schedule/design_predict_activation_experiment
  ```

---

## 3. 文件功能说明

### 3.1 wt_test.sh

该脚本基于如下脚本扩展实现：

```
predict-schedule/batchsize_experiment/disaggeration_pd_experiment/wt_disagg_p2p_nccl_xpyd.sh
```

在原有 PD 分离式部署脚本基础上，新增了：

- **PastFuture 调度逻辑**
- **Activation 预测功能**

#### 关键环境变量说明

- `USE_PASTFUTURE_SCHEDULER="true"`  
  - 在 **decode 实例** 中启用 PastFuture 调度策略

- `USE_ACTIVATION_PREDICTOR="true"`  
  - 在 **prefill 实例** 中启用激活预测功能
  - 在 prefill 实例中：
    - 通过 `llama.py` 初始化 **GPU Ring Buffer**
    - 在同一张 GPU 上加载预测器权重
    - 接收请求对应的 `hidden_states`、`KV cache` 和 `metadata`
    - 执行预测逻辑并输出结果

#### 当前限制与待优化点

- GPU Ring Buffer 的每个 slot 目前 **静态设置为 8192**
- 同时要求：
  - `prefill max_num_batched_tokens = 8192`
- 该设计在当前阶段用于功能验证，后续需要：
  - 动态 slot 管理
  - 与实际 batch size / token 分布解耦

### 3.2 TODO

---

## 4. 当前实验数据集

当前版本使用 **测试数据集** 进行验证，数据集路径如下：

```
predict-schedule/baseline_experiment/pastfuture/dataset/lmsys-50k-filtered-with-sample
```

该数据集主要用于验证：

- 激活捕获与传输链路正确性
- Predictor 与 prefill 实例的协同工作能力
- PastFuture 调度逻辑的可用性

