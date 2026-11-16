# 实验环境迁移清单（→ 另一台服务器）

> 生成时间：2026-09-25
> 迁移源：`wt` 容器（8× A800-SXM4-80GB，Ubuntu 24.04.1，内核 5.4.0-100）
> 实验主线：PD 分离部署 + 预测式路由（scheme4 / scheme4_gwf）+ GWF 准入调度

---

## 0. 迁移总原则（先读这一节）

### 0.1 ⚠️ 路径必须**原样保持一致**

代码里有**几十处硬编码的绝对路径**（`/root/predict-schedule/...`、`/root/myshare/...`），
包括 `sys.path`、模型权重、BERT 主干目录、结果输出目录等。**不要改成新路径**，
否则要逐个改源码，极易漏改。

建议在新机器上直接复刻：

```
/root/predict-schedule/     ← 代码 + 实验脚本
/root/myshare/              ← 数据集 + 模型 + 预测器权重
/root/miniconda3/envs/openrlhf/   ← Python 环境
```

### 0.2 `/root/myshare` 在本机是**共享卷**

本机 `/root/myshare` 与另一台 Dynamo 容器的 `/home/dynamo/myshare` 是同一个共享卷。
迁移时**只搬 `/root/myshare` 下这份清单列出的子目录**即可，不需要整个共享卷。

### 0.3 🚫 安全约束（沿用原环境）

- **绝对不要删除或修改 `/root/.cache/huggingface/hub` 下的任何内容。**
  该目录在本机曾是 81 GB `/dev/shm` 泄漏与 OOM 的来源，且含其他实验的软链接目标。
  迁移时**只需读取**（部分脚本默认路径指向它，现已失效或为空）。

---

## 0.4 ⚠️ 前置条件：基础镜像 + 容器启动参数（**最容易漏，漏了直接跑不起来**）

> 本节是 2026-09-25 复核时补的。前面的第 1~7 节只覆盖了「应用层」（代码/脚本/数据/模型/conda 环境）。
> **一个真正的空容器（如 `ubuntu:24.04`）照那份清单迁完，仍然跑不起来。**

### 0.4.1 基础镜像是 **NVIDIA NGC PyTorch 容器**（Ubuntu 24.04 + CUDA 12.8）

本机容器的判定依据：

```
/opt/{pytorch,hpcx,tensorrt,amazon,nvidia}/      ← NGC 镜像标志目录
/etc/apt/sources.list.d/cuda-ubuntu2404-x86_64.list
/etc/environment 里有 NVIDIA_REQUIRE_JETPACK_HOST_MOUNTS / _CUDA_COMPAT_PATH
/etc/shinit_v2                                  ← NGC 入口脚本
```

**空容器需要自己补齐的系统层依赖（都不在 conda env 里）：**

| 组件 | 本机版本 | 作用 |
|---|---|---|
| CUDA Toolkit 12.8 全套 apt 包 | `cuda-nvcc/cudart/nvrtc/cupti/nvvm/nvml-dev/nvprof/... 12.8.5x` | vLLM 编译 `.so` |
| `cuda-compat-12-8` | 570.86.10 | 驱动前向兼容层（`/usr/local/cuda/compat`） |
| `libnccl2` / `libnccl-dev` | **2.25.1-1+cuda12.8** | PD 分离的 KV 传输 |
| `libibverbs1` / `libibverbs-dev` / `ibverbs-providers` / `librdmacm1t64` | 50.0-2build2 | NCCL 底层 |
| HPC-X / OpenMPI | `/opt/hpcx`、`/usr/local/mpi` | — |
| TensorRT | `/opt/tensorrt` | 镜像自带 |
| EFA | `/opt/amazon/efa` | 镜像自带 |
| cmake / gcc / g++ | `/usr/local/bin/cmake`、gcc 13.3.0 | 编译 |

> **最稳做法**：新容器直接起同一个 NGC 镜像 tag，而不是从 `ubuntu:24.04` 手工装。

### 0.4.2 容器启动参数

| 参数 | 本机值 | 为什么必需 |
|---|---|---|
| `--shm-size` | **189 GB** | tensor pool 32 GiB × 8 实例 = 256 GiB；不够会 OOM（见 §6.1） |
| `--gpus all`（或 `--runtime=nvidia`） | 8 卡 | — |
| `--ulimit memlock=-1:-1` | ⚠️ 本机 `ulimit -l` 只有 **64 KB** | pinned host memory。本机因为用的是 shmem 而非 `mlock` 所以没炸，换环境可能触发 |
| `--ulimit stack=67108864` | — | NGC 惯例 |
| `--cap-add=IPC_LOCK` | — | 锁内存 |
| RDMA 设备 | `/dev/infiniband` 已挂（`issm0` / `rdma_cm` / `umad0` / `uverbs0`） | NCCL |
| NVLink 设备 | `/dev/nvidia-nvlink`、`/dev/nvidia-nvswitch{0,1,2}` | 跨卡 KV 传输必须 |
| `--network=host` 或端口全开 | proxy :28002、bench :22007、实例 :22001-22004 / :22020-22023 | 全部走 127.0.0.1 |

### 0.4.3 镜像级环境变量（在 `/etc/environment`，**不在 conda env 里**）

这些会**直接影响数值结果**，新容器若没有会跑出不可比的数：

```
OMP_NUM_THREADS=4                          # 注意：GNU nproc 会读它并返回 4（本机真实 52 核）
CUDA_CACHE_DISABLE=1
TORCH_CUDA_ARCH_LIST=7.5 8.0 8.6 9.0 10.0 12.0+PTX   # 决定 vLLM 编译产物
TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1         # 影响 FP32 矩阵乘精度
TORCH_NCCL_USE_COMM_NONBLOCKING=0          # 影响 NCCL 行为
CUDA_MODULE_LOADING=LAZY
NCCL_VERSION=2.25.1
CUDA_VERSION=12.8.0.038
LD_LIBRARY_PATH=...:/usr/local/cuda/compat/lib          # 驱动兼容层
PATH=...:/usr/local/mpi/bin:/opt/tensorrt/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin
```

### 0.4.4 驱动兼容性

| 项 | 值 |
|---|---|
| 宿主驱动（`nvidia-smi`） | **550.54.14** |
| 镜像 CUDA 12.8 的目标驱动（`cuda-compat-12-8`） | **570.86.10** |
| 桥接方式 | `/usr/local/cuda/compat` 前向兼容层 |

> 新机器驱动版本若与本机不同，**必须重跑 §9 的硬件自检**确认兼容。

### 0.4.5 conda 本体（33 GB）也需求迁移或安装

`/root/miniconda3`（**33 GB**）本身不在第 5 节的 `conda env export` 覆盖范围内。
空容器没有 conda，必须：
- 方案 A（推荐）：新容器同样把 miniconda 装在 `/root/miniconda3`，再 `conda env create`
- 方案 B：`rsync` 整个 `/root/miniconda3`（保证路径一致，省去重装）
  ⚠️ 整个 rsync 会带上所有 env（含 `predict-orign`、`vllm_baseline`，共 33 GB），
  只想带 `openrlhf` 时需要过滤


---

## 1. 代码：vLLM fork（**最高优先级**）

| 项 | 值 |
|---|---|
| 源路径 | `/root/predict-schedule/vllm` |
| 体积 | 7.0 GB（含 `.git`；其中编译产物约 1 GB） |
| git remote | `https://github.com/cxywhite/vllm.git` |
| 当前分支 | `v0.11.0-local-v2` |
| HEAD | `eacc1b9cd` — "refactor: baseline predict multi-kind switch + cleanup"（2026-07-04） |

### 1.1 ⚠️ 有**未提交的本地改动**，必须手动带走

`git status` 结果：

```
 M vllm/v1/core/sched/geometric_window_queue.py
 M vllm/v1/core/sched/geometric_window_scheduler.py      ← 核心算法，改动 962 行
```

| 文件 | 改动 | 作用 |
|---|---|---|
| `vllm/vllm/v1/core/sched/geometric_window_scheduler.py` | **962 行**（+676/−413 含 queue） | **GWF 准入调度器**：`--geometric-window-scheduler` 时加载。含 `WT_GWF_WINDOW_CAP=256` 窗口上限、TPOT 目标 120 ms、动态窗口收缩/增长。这是 Ours 相对 RR 好 21.5% 的来源 |
| `vllm/vllm/v1/core/sched/geometric_window_queue.py` | 127 行 | `GeometricWindowQueue`，仅被 `geometric_window_scheduler.py` import |

### 1.2 未跟踪但**实验必需**的脚本

```
examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
  ├── decode_predict_3Dload.py                    (symlink → disagg_serving_suite/)
  ├── disagg_proxy_p2p_nccl_xpyd_roundrobin_pp.py (roundrobin_pp 变体 proxy)
  ├── p2d_current_3Dload.py                       (symlink)
  ├── p2d_current_capacity.py                     (symlink)
  ├── p2d_oracle_3Dload.py                        (oracle 变体 proxy)
  ├── plot_decode_load2.py                        (symlink)
  └── test_disagg_proxy_p2d_loadbalance.py        (symlink)
```

### 1.3 含 `[WT]` 标记的改动文件（29 个，说明都改过）

```
vllm/model_executor/models/qwen2.py                    ← 预测器挂载点（layer 13）
vllm/model_executor/models/llama.py                    ← 预测器挂载点（layer 15）
vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py
vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_engine.py
vllm/config/scheduler.py
vllm/v1/metrics/{wt_gpu_stats.py,loggers.py,stats.py}
vllm/v1/core/sched/{geometric_window_scheduler.py,pastfuture_scheduler.py,scheduler.py,
                    pastfuture_queue.py,load_gate_scheduler.py}
vllm/v1/worker/gpu_model_runner.py
vllm/v1/request.py
vllm/v1/engine/{processor.py,async_llm.py,core.py,__init__.py}
vllm/entrypoints/openai/{serving_engine.py,protocol.py,serving_chat.py,
                         serving_completion.py}
vllm/engine/{arg_utils.py,protocol.py}
benchmarks/{benchmark_serving_trace.py,backend_request_func.py}
examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
  {wt_gpu_ring_buffer.py,wt_metadata.py,p2d_oracle_3Dload.py,
   disagg_proxy_p2p_nccl_xpyd.py,disagg_proxy_p2p_nccl_xpyd_roundrobin_pp.py,
   test_disagg_proxy.py,predictor_worker_readyflag.py}
```

> **最稳的做法**：直接 `rsync` 整个 `/root/predict-schedule/vllm`，**排除 `.git` 和所有 `*.bak*`**，
> 换机后原地 `pip install -e .` 重编译，而不是靠 git clone 再打补丁。

### 1.4 ⚠️ 编译产物必须重建

```
vllm/_C.abi3.so                 772 MB
vllm/_moe_C.abi3.so             145 MB
vllm/_flashmla_C.abi3.so         32 MB
vllm/_flashmla_extension_C.abi3.so 12 MB
vllm/cumem_allocator.abi3.so
```

这些是 2025-12-30 在本机编译的，**换机后 ABI 不一定兼容，必须重新编译**。
迁移时建议先删除它们再打包（省 1 GB），新机上 `pip install -e .` 会重建。

---

## 2. 实验脚本与工具

### 2.1 主入口脚本目录 `/root/predict-schedule/experiment/zrecent_use/0630/`

体积 5.8 GB（**其中绝大部分是历史结果目录，可裁剪**）。

**必迁 —— 主入口脚本：**

| 路径 | 体积 | 作用 |
|---|---|---|
| `test_evaluation_strategyA_oracle_gwf.sh` | 85 KB | **主力入口**。跑出 `goodput 172.23` 那次用它。数据集 glob `qwen_*_600s_strategyA.csv`。支持 `oracle_gwf` 变体 |
| `test_evaluation_scheme4_rr_compare.sh` | 85 KB | **burstgpt 专用入口**（glob `qwen_*.csv`）。也是历史 burstgpt 结果的产生者。⚠️ 其 GPU 空闲检查被人为改成 `return 0`，**必须靠外层 launcher 自带守卫** |

**必迁 —— launcher / 编排脚本：**

| 路径 | 作用 |
|---|---|
| `run_dataset3_lmsys_and_2p6d.sh` | 3 轮串行编排（含代码指纹守卫 + GPU 空闲守卫 + 环境指纹）。**推荐作为新脚本模板** |
| `run_samewindow_sharegpt.sh` | 同窗口 RR/Ours 背靠背 |
| `run_ours_sharegpt.sh` | 单配置启动器 |
| `run_rr_ours_pool32.sh` | pool=32 的 RR/Ours 启动器 |
| `stop_batch.sh` | 停止批次 |
| `run_missing_*.sh` | 补跑脚本 |

**必迁 —— 诊断工具（[WT] 新增）：**

| 路径 | 作用 |
|---|---|
| `wt_env_fingerprint.sh` | 记录 date/loadavg/nvidia-smi/计算进程/`/dev/shm` 清单，用于判断窗口可比性 |
| `wt_resource_watch.py` | 把主机 CPU 分解为 `host_busy_pct` / `our_cpu_pct` / `other_cpu_pct` |
| `wt_gpu_hw_audit.py` | GPU 硬件审计（算力/带宽/NVLink + 理论峰值对比） |
| `wt_gpu_sweep_all.py` | 8 卡逐一对比，查单卡降级 |
| `wt_vllm_decode_probe.py` | 真实推理 roofline 验证（CUDA Graph 对照） |
| `wt_analyze_run.py` | 从 `test_results_*.csv` 重算达标率与分位数（复现 bench goodput） |
| `wt_scan_burstgpt.py` | 全盘扫描 burstgpt 结果覆盖矩阵 |
| `wt_gpu_microbench.py` / `wt_gpu_hbm_bw.py` / `wt_cpu_microbench.py` / `wt_mem_bw.py` / `wt_mem_latency.py` | 早期微基准（被 `wt_gpu_hw_audit.py` 取代，可一并带走） |

**必迁 —— 数据集目录：**

| 路径 | 体积 | 作用 |
|---|---|---|
| `derived_datasets/` | 41 MB | **全部派生 trace**。`tmp_dataset3`（4080 请求主力集）、`tmp_dataset3_ours_lmsys` / `_ours_sharegpt`（**单符号链接目录**，用于让脚本只匹配一个数据集）、`tmp_dataset4_*`、`tmp_dataset5_*`、`strategyA_aggressive_*`、`tmp_dataset`、`tmp_dataset2` |

> ⚠️ `derived_datasets/tmp_dataset3_ours_*` 里是**相对符号链接**（`../tmp_dataset3/...`），
> 打包时必须保留符号链接本身（`rsync -a` / `tar` 默认保留；`cp -r` 需要 `-a`）。

**必迁 —— 说明文档：**

| 路径 | 作用 |
|---|---|
| `GWF_MODIFICATIONS_20260922.md` | GWF 调度器改动记录 |
| `GWF_WINDOW_ISSUE_20260923.md` | 窗口失效问题分析与修复 |
| `README.md` | 入口脚本使用说明 |
| `existing_experiment_matrix.csv` | 历史实验覆盖矩阵（判断哪些配置缺数据） |
| `existing_experiment_summary.md` | 同上，可读版 |

**可选（历史结果，不迁也能跑）**：`1tmp/`、`experiment_result_2/`、`retry_test/`、`retry_fail/`、
`experiment_1p1d_compare/`、`experiment_result_qwen_ds5_new/`、`run_datasets/`、`*.log`、`__pycache__`

### 2.2 `disagg_serving_suite/` `/root/predict-schedule/experiment/zrecent_use/disagg_serving_suite/`

体积 1.8 MB（**体积小，建议整个目录都带走**）。

| 路径 | 体积 | 作用 |
|---|---|---|
| `test_disagg_proxy.py` | 95 KB | ⚠️ **被修改的预测式路由 proxy**。`scheme4`/`scheme4_gwf` 用它。含 fastpath 门 `WT_ROUTE_IDLE_TPOT_MS`（默认 120 ms）修复 |
| `benchmark_serving_trace.py` | 70 KB | **压测客户端**（主脚本 `BENCH_SCRIPT` 指向它）。⚠️ 与 `vllm/benchmarks/` 下同名文件**差 62 行**，两者都要带走，别弄混 |
| `customfunction.py` | 28 KB | trace 数据加载与结果落盘 |
| `backend_request_func.py` → | symlink | 指向 `vllm/benchmarks/backend_request_func.py` |
| `benchmark_dataset.py` → | symlink | 指向 `vllm/benchmarks/benchmark_dataset.py` |
| `benchmark_utils.py` → | symlink | 指向 `vllm/benchmarks/benchmark_utils.py` |
| `decode_predict_3Dload.py` / `p2d_current_3Dload.py` / `p2d_current_capacity.py` / `plot_decode_load2.py` / `test_disagg_proxy_p2d_loadbalance.py` | — | 被 `vllm/examples/.../` 下的 symlink 指过来，**这两个位置都要保持** |
| `analyze_method_results.py` / `check_experiment_completion.py` / `calibrate_scheme4_offline_with_actual.py` / `offline_route_policy_sim.py` | — | 离线分析工具 |

> ⚠️ `test_evaluation_*` 系列脚本里还有另一份同名副本
> （`disagg_serving_suite/test_evaluation_scheme4_rr_compare.sh`，76 KB，Jul 5），
> **是旧版本**，不要误用 —— 活跃版本在 `0630/` 下。

---

## 3. 数据集

| 路径 | 体积 | 作用 | 必需 |
|---|---|---|---|
| `/root/myshare/dataset/` | 4.4 MB | burstgpt 等原始 trace（`qwen_{lmsys,mysharegpt}_burstgpt_673.csv`，各 673 请求） | ✅ |
| `/root/predict-schedule/experiment/zrecent_use/0630/derived_datasets/` | 41 MB | 全部派生 trace（见 2.1） | ✅ |
| `/root/predict-schedule/experiment/wt_experiment/dataset/select/` | 3.4 MB | 早期 select 数据集 | ⚠️ 视需要 |
| `/root/predict-schedule/experiment/wt_experiment/dataset/`（其余 9 个子目录） | 32 GB | 早期原始/评估数据集全集 | ❌ 当前实验线不用 |

> **最小集只需前两行（约 46 MB）。**

**注意**：`test_evaluation_scheme4_rr_compare.sh` 的 `DATASET_BASE_DIR` 默认值
`/root/.cache/huggingface/hub/wt_predictor/0630/dataset/evaluation_dataset/0630_evaluation_dataset`
**在本机已是空壳（csv 全为 0 字节）**，跑 burstgpt 时必须显式覆盖为：
```bash
DATASET_BASE_DIR=/root/myshare/dataset   TRACE_NAME=burstgpt
```

---

## 4. 模型与预测器权重（**激活预测链路，体积大但必需**）

| 路径 | 体积 | 作用 |
|---|---|---|
| `/root/myshare/models/Qwen-2.5-7b-Instruct/` | **15 GB** | 主力被测模型（Qwen2.5-7B，hidden=3584，28 层） |
| `/root/myshare/models/Meta-Llama-3-8B-Instruct/` | **30 GB** | 第二被测模型（hidden=4096） |
| `/root/myshare/predict_model/qwen_model/model.pth` | 452 MB | ⚠️ **Qwen 激活长度分布预测器权重**（`hidden_dim=512`, `num_classes=5`） |
| `/root/myshare/predict_model/llama_model/model.pth` | 452 MB | ⚠️ **Llama 预测器权重** |
| `/root/myshare/predict_project/input_ids_predictor_use/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594/` | 421 MB | ⚠️ **BERT 主干**（预测器的 backbone + tokenizer + config） |
| `/root/myshare/predict_project/act_predictor_train/results/train_{llama3-8b,qwen2.5-7b-instruct}_.../model.pth` | 433 / 431 MB | 预测器训练产物（当前链路用 `predict_model/` 那份，这两份是**训练来源**）。整个 `results/` 目录共 867 MB / 6 个子目录，建议一并带走 |

> ⚠️ **两个模型默认路径在本机都已失效**，跑实验时**必须显式传**：
> ```
> /root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct   ← 不存在
> /root/.cache/huggingface/hub/Qwen-2.5-7b-Instruct       ← 不存在
> ```
> Llama 用 `LLAMA_MODEL_PATH=/root/myshare/models/Meta-Llama-3-8B-Instruct`，
> Qwen 用 `QWEN_MODEL_PATH=/root/myshare/models/Qwen-2.5-7b-Instruct`
> （`test_evaluation_strategyA_oracle_gwf.sh` 的 Qwen 默认值已是对的，Llama 默认值是失效路径）。

**对应代码**：`vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/predictor_worker_readyflag.py`

```python
DEFAULT_BERT_MODEL_DIR = "/root/myshare/predict_project/input_ids_predictor_use/..."   # 可用 WT_PREDICT_BERT_MODEL_DIR 覆盖
DEFAULT_QWEN_PREDICTOR_WEIGHT_PATH   = "/root/myshare/predict_model/qwen_model/model.pth"    # WT_PREDICT_QWEN_WEIGHT_PATH
DEFAULT_LLAMA_PREDICTOR_WEIGHT_PATH  = "/root/myshare/predict_model/llama_model/model.pth"   # WT_PREDICT_LLAMA_WEIGHT_PATH
```

> 预测器按 `input_dim` 自动选家族：`4096→llama`，`3584→qwen`。
> `WT_PREDICTOR_KIND` 默认 `ours`（本实验线只用这个值）。

**预测器运行时还依赖两个目录**（会被自动创建，但建议预建避免权限问题）：

```
/root/predict-schedule/experiment/wt_experiment/experiment_paper/4-Predict_latency/tmp/
/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/
```

---

## 5. 运行环境

### 5.1 Conda 环境

| 项 | 值 |
|---|---|
| 环境名 | **`openrlhf`** |
| 路径 | `/root/miniconda3/envs/openrlhf` |
| Python | 3.10.19 |
| 迁移方式 | `conda env export -n openrlhf --no-builds > env_openrlhf.yml`（401 行），新机 `conda env create -f` |

### 5.2 关键包版本（环境名一致但包版本可能漂移，需核对）

| 包 | 版本 |
|---|---|
| **vllm** | **0.11.0，editable 安装 → `/root/predict-schedule/vllm`** |
| torch | 2.8.0（+cu128） |
| triton | 3.4.0 |
| transformers | 4.56.1 |
| xformers | 0.0.32.post1 |
| numpy | 2.2.5 |
| pandas | 2.3.3 |
| ray | 2.48.0 |
| pyzmq | 27.1.0 |
| msgspec | 0.19.0 |
| einops | 0.8.1 |
| uvloop | 0.22.1 |

⚠️ **vLLM 必须是 editable 指向 `/root/predict-schedule/vllm`**，否则改的调度器代码不生效：
```bash
cd /root/predict-schedule/vllm && pip install -e .
```

⚠️ **`OMP_NUM_THREADS=4`** 来自**容器镜像的 `/etc/environment`**（不是 conda）。注意：
GNU `nproc` 会读取它并**返回 4 而不是真实核数**（本机 52 核），排查 CPU 问题时别被误导。
真实核数用 `nproc --all`。详见 §0.4.3。

### 5.3 编译工具链

| 项 | 版本 |
|---|---|
| CUDA Toolkit | 12.8（`/usr/local/cuda-12.8`，`nvcc` 12.8.61） |
| gcc / g++ | 13.3.0 |
| ninja | 由 vllm 依赖提供 |

### 5.4 CUDA / NCCL

| 项 | 值 |
|---|---|
| NVIDIA 驱动 | **550.54.14**（CUDA 12.8 运行时） |
| torch NCCL | 2.27.3 |
| 系统 libnccl | `/usr/lib/x86_64-linux-gnu/libnccl.so.2` (2.25.1) |

> PD 分离的 KV 传输走 **NCCL P2P over NVLink**（`P2pNcclConnector`），
> 新机器必须有可用的 NVLink/NCCL，否则跨卡 KV 传输会退化到 PCIe。

---

## 6. 硬件与系统要求

| 项 | 本机 | 要求 |
|---|---|---|
| GPU | 8× A800-SXM4-80GB | **至少 8 张 80 GB 卡**（2p6d / 4p4d 都要用满 0–7） |
| GPU 互联 | NVLink（P2P 单向实测 187 GB/s of 200 峰值） | 必须有 NVLink，否则 KV 传输成为瓶颈 |
| CPU 核 | 52（2× Xeon Gold 5320，NUMA 2 节点） | ≥ 20 核（8 实例 × 引擎 + proxy + client） |
| 内存 | 377 GiB | **≥ 300 GiB 可用** |
| **`/dev/shm`** | **189 GB** | ⚠️ **必须 ≥ 160 GB**。tensor pool = 32 GiB × 8 实例 = **256 GiB pinned host memory**，不足会 OOM |
| 磁盘 | `/root/myshare` 3.3 TB（需 ≥ 100 GB 余量放结果） | — |
| OS | Ubuntu 24.04.1，内核 5.4.0-100-generic | 建议同系 |
| HugePages | 154 × 2 MB | 非必需，记录备查 |

### ⚠️ 6.1 `/dev/shm` 泄漏历史（迁移后要防）

vLLM 的 KV Offload 会在 `/dev/shm` 创建 `vllm_offload_<uuid>.mmap`，
**只有优雅退出才 `unlink`；被 SIGKILL 就泄漏**。本机曾泄漏 81 GB 导致 OOM。

迁移后建议：
- 把 `/dev/shm` 设到 ≥ 189 GB（`--shm-size`）
- 跑批次前检查残留：`ls /dev/shm/vllm_offload_*.mmap | wc -l`
- **清理前必须确认无 fd / mmap 持有者**（UUID 命名保证不会被重新 attach，相对安全）

---

## 7. 结果目录（可选迁移）

| 路径 | 体积 | 作用 |
|---|---|---|
| `/root/myshare/wty_tmp_result/` | — | 本轮实验的结果根（`retry_qwen_4p4d/`、`burstgpt_qwen/`、`retry_burstgpt_2p6d/`、`env_fingerprint/`、`resource_watch/`） |
| `/root/predict-schedule/experiment/zrecent_use/0630/{1tmp,experiment_result_2,retry_test,retry_fail}/` | 5.8 GB 的大部分 | 历史结果 |
| `/root/myshare/paper_figures/` | — | 论文图表 |
| `/root/myshare/models/wt_predictor/0630/{result_0630,tmp_result_0630,dynamo_0630}/` | — | 早期结果（**含 burstgpt 的历史 bench 日志**，是唯一留存处） |

> 这些**不是跑实验的必要条件**，但丢了就再也拿不回来（本机 `/root/.cache/huggingface/hub/wt_predictor/` 下对应的已空）。
> 建议至少把 `wty_tmp_result/`、`paper_figures/`、`wt_predictor/0630/{result,tmp_result,dynamo}_0630/` 打包留档。

---

## 8. 硬编码路径清单（迁移后**必须逐条确认存在**）

| 硬编码路径 | 出现在 | 作用 |
|---|---|---|
| `/root/predict-schedule/vllm` | conda editable 安装、多处 `sys.path` | vLLM 源码 |
| `/root/predict-schedule/experiment/zrecent_use/disagg_serving_suite/benchmark_serving_trace.py` | 主脚本 `BENCH_SCRIPT` | 压测客户端 |
| `/root/predict-schedule/experiment/zrecent_use/disagg_serving_suite/test_disagg_proxy.py` | 主脚本 `DISAGG_SERVING_SUITE_PROXY` | 预测式路由 proxy |
| `/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd` | 主脚本 `PROXY_BASE_DIR` | proxy 与配套模块目录 |
| `/root/predict-schedule/experiment/zrecent_use/0630/derived_datasets` | `test_evaluation_strategyA_oracle_gwf.sh` 的 `DATASET_BASE_DIR` 默认值 | 派生 trace 根 |
| `/root/myshare/models/Qwen-2.5-7b-Instruct` | 主脚本 `QWEN_MODEL_PATH` | Qwen 权重 |
| `/root/myshare/models/Meta-Llama-3-8B-Instruct` | 主脚本 `LLAMA_MODEL_PATH` | ⚠️ **两脚本默认值都指向已失效的 hf 缓存路径**，必须显式传 `LLAMA_MODEL_PATH` |
| `/root/myshare/predict_model/{qwen_model,llama_model}/model.pth` | `predictor_worker_readyflag.py` | 预测器权重 |
| `/root/myshare/predict_project/input_ids_predictor_use/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594` | `predictor_worker_readyflag.py` | BERT 主干 |
| `/root/predict-schedule/experiment/wt_experiment/experiment_paper/4-Predict_latency/tmp/` | `predictor_worker_readyflag.py`（predict 计时 csv） | 预测耗时日志 |
| `/root/predict-schedule/vllm/examples/.../wt_experiment/experiment_paper/4-Predict_latency/tmp/` | 同上（symlink 路径） | 同上 |
| `/root/.cache/huggingface/hub/wt_predictor/0630/predict_experiment` | `predictor_worker_readyflag.py`（flush 输出） | ⚠️ 本机已不存在，会被自动创建 |
| `/root/.cache/huggingface/hub/wt_predictor/0630/dataset/evaluation_dataset/0630_evaluation_dataset` | `test_evaluation_scheme4_rr_compare.sh` 默认 `DATASET_BASE_DIR` | ⚠️ 本机已空壳，**跑 burstgpt 必须覆盖** |
| `/root/predict/predict_experiment/dataset_result` | `benchmark_serving_trace.py` | 备用输出路径 |
| `/tmp/wt_scheme4_rr_compare_skip_success_once_20260505.done` | `test_evaluation_scheme4_rr_compare.sh` | 一次性 skip 标记（**新机器上要确认没有这个文件**，否则会误跳过配置） |

---

## 9. 迁移后验证清单（按顺序做）

```bash
# 0) 容器与镜像层（最先做，见 §0.4）
cat /etc/environment | grep -E 'OMP_NUM_THREADS|TORCH_|CUDA_|LD_LIBRARY_PATH'  # 应与源机一致
ls /opt/{pytorch,hpcx,tensorrt,amazon} 2>/dev/null        # NGC 镜像标志
nvcc --version | tail -2                                   # 期望 CUDA 12.8
ldconfig -p | grep -E 'libnccl|libibverbs' | head          # NCCL/RDMA
ls /dev/infiniband /dev/nvidia-nvlink 2>/dev/null          # NCCL / NVLink 设备
df -h /dev/shm | tail -1                                   # 期望 >= 189G
ulimit -l                                                  # 期望 -1 或足够大

# 1) 环境
conda activate openrlhf
python -c "import vllm, os; print(vllm.__version__, os.path.dirname(vllm.__file__))"
#   期望 0.11.0 /root/predict-schedule/vllm     ← 必须是这个路径（editable）

# 2) 源码改动是否带过来了（对比 sha256 前 32 位）
sha256sum vllm/vllm/v1/core/sched/geometric_window_scheduler.py | cut -c1-32   # 期望 ecc3a294f6ad9d88f4a8c75a56864660
sha256sum experiment/zrecent_use/disagg_serving_suite/test_disagg_proxy.py | cut -c1-32
#   期望 b01c644ae0c8bb47d6fd468fd28d063f

# 3) 编译产物（若已删除则重建）
cd /root/predict-schedule/vllm && pip install -e . --no-build-isolation

# 4) 关键文件存在性
for p in /root/myshare/models/Qwen-2.5-7b-Instruct \
         /root/myshare/models/Meta-Llama-3-8B-Instruct \
         /root/myshare/predict_model/qwen_model/model.pth \
         /root/myshare/predict_project/input_ids_predictor_use/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594 \
         /root/myshare/dataset/burstgpt/qwen_lmsys_burstgpt_673.csv ; do
  [ -e "$p" ] && echo "✅ $p" || echo "❌ 缺失 $p"
done

# 5) 数据集记录数（⚠️ 不要用 wc -l，csv 有内嵌换行）
python3 -c "
import csv,sys; csv.field_size_limit(sys.maxsize)
for p in ['/root/myshare/dataset/burstgpt/qwen_lmsys_burstgpt_673.csv',
          '/root/predict-schedule/experiment/zrecent_use/0630/derived_datasets/tmp_dataset3/qwen_lmsys_qwentrace_4080_600s_strategyA.csv']:
    with open(p,newline='') as f:
        r=csv.reader(f); next(r); print(sum(1 for _ in r), p)"
#   期望 673 和 4080

# 6) 硬件自检（换机后必做，确认新卡没有降级）
cd /root/predict-schedule/experiment/zrecent_use/0630
python3 wt_gpu_sweep_all.py        # 8 卡带宽/算力离散度应 < 5%
python3 wt_gpu_hw_audit.py 1       # 逐项对比理论峰值

# 7) 端到端冒烟（用最小配置，~5 分钟）
python3 wt_vllm_decode_probe.py --eager 1 --device 1 --batch 1
#   batch=1 每步应在 11~14 ms 区间

# 8) 首个真实实验：跑 1 组 RR，确认 goodput 量级正常
```

### 9.1 ⚠️ 迁移后最容易踩的 4 个坑

1. **基础镜像不对** → 空容器没有 CUDA Toolkit / libnccl / libibverbs / HPC-X，
   `pip install -e .` 直接编译失败。**先解决 §0.4，再谈应用层。**
2. **`/dev/shm` 太小** → tensor pool 分配失败 / OOM（表现：实例日志 `Engine core initialization failed`，**无 Python traceback**，实为 SIGKILL）
3. **GPU 空闲检查被 stub 掉** → `test_evaluation_scheme4_rr_compare.sh` 里
   `check_selected_gpus_idle_once() { return 0; }`，若新机器有其他租户占卡，脚本会直接启动然后失败。
   **务必用 `run_dataset3_lmsys_and_2p6d.sh` 里那段 `wait_for_gpu_idle()` 做外层守卫。**
4. **`nproc` 返回 4** → 是镜像 `/etc/environment` 里 `OMP_NUM_THREADS=4` 造成的假象，不是 CPU 被限制。排查时用 `nproc --all`。

---

## 10. 迁移量汇总

| 类别 | 路径 | 体积 | 必需 |
|---|---|---|---|
| **基础镜像** | **NVIDIA NGC PyTorch（Ubuntu 24.04 + CUDA 12.8）** | **~20 GB** | ✅ **最优先，见 §0.4** |
| **miniconda 本体** | `/root/miniconda3` | **33 GB** | ✅ 见 §0.4.5 |
| vLLM 源码（去 `.git` 去编译产物） | `/root/predict-schedule/vllm` | ~3 GB | ✅ |
| 实验脚本 + 派生数据集 + 工具 | `/root/predict-schedule/experiment/zrecent_use/` | 5.8 GB（可裁到 ~50 MB） | ✅ |
| Python 环境 | `conda env openrlhf` | ~10 GB | ✅（用 yml 重建） |
| Qwen 权重 | `/root/myshare/models/Qwen-2.5-7b-Instruct` | 15 GB | ✅ |
| Llama 权重 | `/root/myshare/models/Meta-Llama-3-8B-Instruct` | 30 GB | ✅（只跑 qwen 时可选） |
| 预测器权重 | `/root/myshare/predict_model/` | 864 MB | ✅ |
| BERT 主干 | `/root/myshare/predict_project/input_ids_predictor_use/` | 1.7 GB | ✅ |
| 数据集 | `/root/myshare/dataset/` + `0630/derived_datasets/` | 46 MB | ✅ |
| 预测器训练产物（留档） | `/root/myshare/predict_project/act_predictor_train/results/` | 867 MB | ⚠️ 建议 |
| 历史结果（留档） | `/root/myshare/wty_tmp_result/`、`paper_figures/`、`models/wt_predictor/0630/` | 数十 GB | ❌ 仅留档 |
| 早期数据集全集 | `experiment/wt_experiment/dataset/`（除 `select/`） | 32 GB | ❌ |

**最小迁移集 ≈ 60 GB**（含 Llama）/ **≈ 30 GB**（只跑 Qwen）。

> 若从零建容器，还要加上**基础镜像 ~20 GB + miniconda 33 GB**，但这两项是镜像层，
> 通常由镜像 tag 解决而不是 `rsync`。

---

## 11. 建议的迁移命令（rsync 保符号链接）

```bash
# 在源机执行（注意 -a 保留符号链接与权限，--exclude 去掉可重建项）
rsync -aH --info=progress2 \
  --exclude='.git' --exclude='__pycache__' --exclude='*.bak*' \
  --exclude='vllm/_C.abi3.so' --exclude='vllm/_moe_C.abi3.so' \
  --exclude='vllm/_flashmla_C.abi3.so' --exclude='vllm/_flashmla_extension_C.abi3.so' \
  --exclude='vllm/cumem_allocator.abi3.so' \
  --exclude='experiment/zrecent_use/0630/{1tmp,retry_test,retry_fail,experiment_result_2,experiment_result_qwen_ds5_new,experiment_1p1d_compare}' \
  /root/predict-schedule/  <新机>:/root/predict-schedule/

rsync -aH --info=progress2 /root/myshare/dataset/    <新机>:/root/myshare/dataset/
rsync -aH --info=progress2 /root/myshare/models/Qwen-2.5-7b-Instruct/ \
                            <新机>:/root/myshare/models/Qwen-2.5-7b-Instruct/
rsync -aH --info=progress2 /root/myshare/predict_model/ <新机>:/root/myshare/predict_model/
rsync -aH --info=progress2 /root/myshare/predict_project/input_ids_predictor_use/ \
                            <新机>:/root/myshare/predict_project/input_ids_predictor_use/

# 环境
conda env export -n openrlhf --no-builds > /tmp/env_openrlhf.yml
scp /tmp/env_openrlhf.yml <新机>:/tmp/
# 在新机：conda env create -f /tmp/env_openrlhf.yml && conda activate openrlhf
#         cd /root/predict-schedule/vllm && pip install -e . --no-build-isolation
```

> ⚠️ **不要**用 `cp -r` 或 `tar --dereference`，会破坏必需符号链接；
> ⚠️ **不要**把 `/root/.cache/huggingface/hub` 整目录搬过去，也不要删改它。
