# 0630 test_evaluation_scheme4_rr_compare.sh 使用说明

## 环境要求

- 使用 **conda 环境 `openrlhf`**：
  ```bash
  conda activate openrlhf
  ```
- 需要 8 张 GPU（2p6d / 4p4d 配置均假设 GPU 0-7 可用）。
- 依赖：`pandas`、`datasets`、`vllm`、`quart`。

## 脚本位置

```bash
/root/predict-schedule/experiment/zrecent_use/0630/test_evaluation_scheme4_rr_compare.sh
```

## 功能

用于在 0630 数据集上只做 **roundrobin** 与 **scheme4** 的对比评测，按以下顺序遍历：

```text
for trace in [burstgpt, mooncake, qwen]:
    for tpot in [30ms, 50ms]:
        for model in [llama, qwen]:
            for pd_ratio in [2:6, 4:4]:
                for dataset in [lmsys, mysharegpt]:
                    for variant in [roundrobin, scheme4, scheme4_gwf]:
                        run benchmark
```

## 数据集

- 根目录：`/root/.cache/huggingface/hub/wt_predictor/0630/dataset/evaluation_dataset/0630_evaluation_dataset`
- 子目录（trace）：`burstgpt/`、`mooncake/`、`qwen/`
- 每个 trace 下 4 个 CSV，命名规则：
  ```text
  {llama,qwen}_{lmsys,mysharegpt}_{trace}_N.csv
  ```
- 文件名前缀 `llama` / `qwen` 表示该数据集使用哪个模型跑。

## 运行方式

### 默认运行（推荐）

直接执行即可跑完全部三个 trace：

```bash
cd /root/predict-schedule/experiment/zrecent_use/0630
conda activate openrlhf
bash test_evaluation_scheme4_rr_compare.sh
```

### 只跑单个 trace

```bash
TRACE_NAME=mooncake bash test_evaluation_scheme4_rr_compare.sh
```

### 只跑某个 TPOT

```bash
TPOT_MS_LIST=30 bash test_evaluation_scheme4_rr_compare.sh
```

### 只跑 llama

```bash
MODEL_TYPE=llama bash test_evaluation_scheme4_rr_compare.sh
```

### 自定义输出目录

```bash
BASE_RESULT_DIR=./my_results bash test_evaluation_scheme4_rr_compare.sh
```

## 关键环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_TYPE` | `all` | `llama` / `qwen` / `all`（all 时先 llama 后 qwen） |
| `TRACE_NAME` | （空） | 空则按 `TRACE_LIST` 遍历；设为非空则只跑该 trace |
| `TRACE_LIST` | `burstgpt,mooncake,qwen` | trace 遍历顺序 |
| `PD_RATIO_LIST` | `2:6,4:4` | 只跑 2p6d 和 4p4d |
| `TPOT_MS_LIST` | `30,50` | TPOT SLO，单位 ms |
| `COMPARE_EXPERIMENT_VARIANTS` | `roundrobin scheme4 scheme4_gwf` | 可跑三个 variant；`scheme4_gwf` 在 scheme4 代理路由基础上额外在 decode 实例启用 `--geometric-window-scheduler` |
| `BENCH_SCRIPT` | `/root/predict-schedule/vllm/benchmarks/benchmark_serving_trace.py` | benchmark 脚本路径 |
| `BASE_RESULT_DIR` | `./experiment_result_2` | 结果输出根目录 |
| `RUN_REPEAT_PER_CONFIG` | `1` | 每个配置重复次数 |

## Benchmark 脚本说明

当前脚本默认使用：

```bash
/root/predict-schedule/experiment/zrecent_use/disagg_serving_suite/benchmark_serving_trace.py
```

经检查，该文件与以下路径的 `benchmark_serving_trace.py` **完全一致**：

```bash
/root/predict-schedule/vllm/benchmarks/benchmark_serving_trace.py
```

因此两者等价。如需显式指定 vllm 下的路径，可：

```bash
BENCH_SCRIPT=/root/predict-schedule/vllm/benchmarks/benchmark_serving_trace.py \
  bash test_evaluation_scheme4_rr_compare.sh
```

## 结果目录命名示例

```text
experiment_result_2/
  llama_llama-lmsys-chat-burstgpt_n673_pd2p6d_tpot30ms_traceburstgpt_rep1_20260708_120000_roundrobin/
  llama_llama-lmsys-chat-burstgpt_n673_pd2p6d_tpot30ms_traceburstgpt_rep1_20260708_120000_scheme4/
  llama_llama-lmsys-chat-burstgpt_n673_pd2p6d_tpot30ms_traceburstgpt_rep1_20260708_120000_scheme4_gwf/
  ...
```

目录名包含：模型、数据集标签、样本数、PD 比例、TPOT、trace 名、重复次数、时间戳、variant。

## 注意事项

- 脚本内部会自动启动/停止 prefill、decode vLLM 实例以及 proxy，不需要手动起服务。
- 若中途 GPU 被占用，`check_selected_gpus_idle` 会跳过当前配置。
- 运行前确保 conda 环境 `openrlhf` 已激活，否则可能缺少 `vllm` 等依赖。
