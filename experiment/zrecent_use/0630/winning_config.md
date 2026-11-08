# GWF 准入 A/B 实验：正收益配置总表

> 记录时间 2026-11-09。汇总目前所有「GWF goodput > scheme4（贪心）」的正收益配置。
> 完整逐次实验见 `experiment_log.md`；本文件只做**可复现的配置快照**。

## 通用运行配置（所有档共用，除非单独标注）

| 项 | 值 |
|---|---|
| 模型 | **Qwen-2.5-7B-Instruct**，路径 `/root/myshare/models/Qwen-2.5-7b-Instruct`（NVMe 快盘） |
| 硬件 | 8× A800-80GB，vLLM v0.11.0（V1） |
| SLO | TTFT 1000ms / TPOT 30ms（`--goodput ttft:1000 tpot:30`） |
| 预测 | `WT_SIMPLIFY_PREDICT=1`（完美预测，用 max_tokens 绕过 BERT） |
| 对比 | `COMPARE_EXPERIMENT_VARIANTS="scheme4 scheme4_gwf"`（唯一差别 = 准入控制器） |
| 路由 | 预测路由（decode 必须加 `--activation-predict`，否则 native 臂路由缺预测输出、A/B 污染） |
| tensor pool | `PREFILL_TENSOR_POOL_MEMORY=4 DECODE_TENSOR_POOL_MEMORY=4`（GB） |
| 启动环境 | `VLLM_HOST_IP=127.0.0.1 NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo`、`NCCL_MAX_NCHANNELS=8 NCCL_MIN_NCHANNELS=8 NCCL_CUMEM_ENABLE=1`、`no_proxy=localhost,127.0.0.1` |
| swap | `--swap-space 0`（decode 无 CPU 换出） |

## 关键前置：两代 GWF 代码

- **第一代（fraction 二值 τ）**：11-06/11-07 的 dense_short、o1500 结果。现已改为标量 EMA，**此代码已不在当前工作树**（见 experiment_log.md #24/#28）。
- **第二代（标量 EMA τ = sum 单位 + TTFT-margin）**：11-09 的 o2000/o1600 结果，**当前代码**。
  - τ = EMA(4 步和)，θ=120ms（`WT_GWF_WINDOW_TPOT_TARGET_MS=120`，= 4×30ms SLO）。
  - TTFT-margin 强制准入：`WT_GWF_TTFT_MARGIN_MS`，deferral ≥ ttft_budget−margin 就绕过窗口强制放行。
  - **代码状态（复现用）**：主仓库 `df5d7b0`（含 trace、路由修复、日志）+ vllm 子模块 `6e1011be4`（τ sum + TTFT-margin + WT_SIMPLIFY_PREDICT）。`git checkout df5d7b0`（主仓库）+ `cd vllm && git checkout 6e1011be4`。

## 正收益配置

### 1. o2000 + sum θ=120 + margin=250（**+24%，最大增益**）

| 项 | 值 |
|---|---|
| 测试数据集 | `admission_ab_base/qwen_o2000/qwen_lmsys_qwentrace_4080_o2000.csv`（4080 请求，输出**均值 2000**、中位 688、p90 7008，重尾） |
| PD 配比 | **4:4**（prefill GPU 0,1,2,3 / decode GPU 4,5,6,7） |
| gpu_memory_utilization | **0.8**（prefill + decode 均为 0.8） |
| θ / margin | θ=120（sum）/ `WT_GWF_TTFT_MARGIN_MS=250` |
| 结果 | goodput 0.42→**0.52（+24%）**、output-tok goodput 250→361（+46%）、P99 TTFT 510s→1.6s |
| 机制 | 窗口节流把请求饿到超 TTFT 预算（510s 尾巴），margin 强制放行砍掉尾巴 |

### 2. o1600 + sum θ=120 + margin=100（**+4.3%**）

| 项 | 值 |
|---|---|
| 测试数据集 | `admission_ab_base/qwen_o1600/qwen_lmsys_qwentrace_4080_o1600.csv`（4080 请求，输出**均值 1600**） |
| PD / gpu util | 4:4 / 0.8（同 #1） |
| θ / margin | θ=120（sum）/ `WT_GWF_TTFT_MARGIN_MS=100` |
| 结果 | goodput 0.69→**0.72（+4.3%）**、TPOT 53ms≈native 54ms、TTFT 不爆 |
| 机制 | 中过载档 margin 要小：250ms 会把 TPOT 从 40 抬到 77ms 反而输；100ms 平衡"TPOT≈native + TTFT 不爆" |

### 3. o1500 + 3P4D + gmu=0.5（**+14%~+69%，第一代 fraction 代码**）

| 项 | 值 |
|---|---|
| 测试数据集 | `admission_ab_base/qwen_o1500/qwen_lmsys_qwentrace_4080_o1500.csv`（输出均值 1500） |
| PD 配比 | **3:4**（prefill GPU 1,2,3 / decode GPU 4,5,6,7，7 卡，GPU 0 留给别的租户） |
| gpu_memory_utilization | **decode = 0.5**（prefill 0.8） |
| 对比 | scheme4 scheme4_gwf（fraction 版 θ=120） |
| 结果 | goodput 0.48→0.81（+68.8%）、复现 0.56→0.64（+14.3%）、recompute 796k→0 |
| 机制 | greedy 把 running 推到 366 > KV 容量 → 抢占重算 796k token；GWF 窗口 256≈KV 容量 → 零抢占 |

### 4. dense_short（**+2.8%~+5%，第一代 fraction 代码**）

| 项 | 值 |
|---|---|
| 测试数据集 | `admission_ab_base/bimodal_burst/qwen_lmsys_bimodal_burst_600s.csv`（12000×200 token，10 突发 ~40 req/s，见 build_bimodal_burst_trace.py） |
| PD / gpu util | 4:4 / 0.8 |
| 对比 | scheme4 scheme4_gwf（fraction 版） |
| 结果 | goodput 2566→2637（+2.8%）、复现 2327→2443（+5.0%）、demoted 1525、stop_window 179 |
| 机制 | running 顶到 256、meas_tpot 46–48ms → demote/stop_window 真触发 |

## 启动命令模板（第二代代码，当前）

```bash
bash /root/myshare/gwf_sweep_run_gwf_only.sh <TRACE_NAME> <BASE_RESULT_DIR> <TTFT_MARGIN_MS>
# 例：o2000 margin=250
#   bash /root/myshare/gwf_sweep_run_gwf_only.sh qwen_o2000 /root/myshare/gwf_sweep_margin/o2000 250
# 例：o1600 margin=100
#   bash /root/myshare/gwf_sweep_run_gwf_only.sh qwen_o1600 /root/myshare/gwf_sweep_margin100/o1600 100
```

（native/scheme4 臂与 θ/margin 无关，可复用；改路由才需两臂都重跑，见坑 #24。）

## 待办

- o2000 margin=250 的 +24% 只有 1 rep，建议补 3 rep 取中位数确认（goodput 尾部统计量，run-to-run 方差 7–10%）。
- o1600 margin=100 的 +4.3% 幅度小，同样需要 3 rep 才能跟噪声区分。
