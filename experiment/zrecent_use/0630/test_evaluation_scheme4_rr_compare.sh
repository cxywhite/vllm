#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------------------
# Disaggregated vLLM evaluation script
# Iterates over CSV datasets under evaluation_dataset and benchmarks each one
# using benchmark_serving_trace.py.
# ----------------------------------------------------------------------------

####################
# USER CONFIG
####################

# Which subtree to run:
#   llama -> llama model + llama dataset + all configured PD/TPOT combos
#   qwen  -> qwen model + qwen dataset + all configured PD/TPOT combos
#   all   -> llama and qwen both
# ---------------------------------------------------------------------------
# 参数模板（放在顶部便于切换）
#
# 方案1: baseline proxy (roundrobin)
#   USE_CUSTOM_PROXY=false
#   USE_ACTIVATION_PREDICTOR=false
#   TEST_ABLATION_P2D=false
#   TEST_ABLATION_P2D_CURRENT_3DLOAD=false
#   TEST_ABLATION_P2D_CURRENT_CAPACITY=false
#   TEST_OPTIMAL=false
#
# 方案1b: baseline proxy (roundrobin++ with FIFO queue + kv_usage serialisation)
#   USE_CUSTOM_PROXY=false
#   USE_ACTIVATION_PREDICTOR=false
#   TEST_ABLATION_P2D=false
#   TEST_ABLATION_P2D_CURRENT_3DLOAD=false
#   TEST_ABLATION_P2D_CURRENT_CAPACITY=false
#   TEST_OPTIMAL=false
#   WT_PREDICT_ROUTE_SCHEME=roundrobin_pp
#   RRPP_KV_THRESHOLD=0.8      # kv_usage above this → serial mode
#   RRPP_POLL_INTERVAL_MS=5    # rate-limiter poll interval
#
# 方案2: predictor-aware proxy
#   USE_CUSTOM_PROXY=false
#   USE_ACTIVATION_PREDICTOR=true
#   TEST_ABLATION_P2D=true
#   TEST_ABLATION_P2D_CURRENT_3DLOAD=false
#   TEST_ABLATION_P2D_CURRENT_CAPACITY=false
#   TEST_OPTIMAL=false
#
# 方案3: p2d current-capacity ablation proxy
#   USE_CUSTOM_PROXY=false
#   TEST_ABLATION_P2D=false
#   TEST_ABLATION_P2D_CURRENT_3DLOAD=false
#   TEST_ABLATION_P2D_CURRENT_CAPACITY=true
#   TEST_OPTIMAL=false
#
# 方案4: p2d current-3Dload ablation proxy
#   USE_CUSTOM_PROXY=false
#   TEST_ABLATION_P2D=false
#   TEST_ABLATION_P2D_CURRENT_3DLOAD=true
#   TEST_ABLATION_P2D_CURRENT_CAPACITY=false
#   TEST_OPTIMAL=false
# ---------------------------------------------------------------------------
USE_CUSTOM_PROXY=false
USE_ACTIVATION_PREDICTOR=true
TEST_ABLATION_P2D=true
TEST_ABLATION_P2D_CURRENT_3DLOAD=false
TEST_ABLATION_P2D_CURRENT_CAPACITY=false
TEST_OPTIMAL=false
MODEL_TYPE=${MODEL_TYPE:-all}

LLAMA_MODEL_PATH=${LLAMA_MODEL_PATH:-/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct}
QWEN_MODEL_PATH=${QWEN_MODEL_PATH:-/root/.cache/huggingface/hub/Qwen-2.5-7b-Instruct}

# Active model fields are populated at runtime from MODEL_TYPE.
TEST_MODEL=""
MODEL=""
# Relative to this script directory, or absolute path.
DATASET_BASE_DIR=${DATASET_BASE_DIR:-/root/.cache/huggingface/hub/wt_predictor/0630/dataset/evaluation_dataset/0630_evaluation_dataset}
SCHEME4_SELECTION_DATASET_DIR=${SCHEME4_SELECTION_DATASET_DIR:-./scheme4_selection_dataset_4}
COMPARE_SCHEME4_SELECTION_ONLY=${COMPARE_SCHEME4_SELECTION_ONLY:-false}
ROUNDROBIN_RESULT_SUFFIX=${ROUNDROBIN_RESULT_SUFFIX:-roundrobin}
ROUNDROBIN_PP_RESULT_SUFFIX=${ROUNDROBIN_PP_RESULT_SUFFIX:-roundrobin_pp}
SCHEME4_RESULT_SUFFIX=${SCHEME4_RESULT_SUFFIX:-scheme4}
SCHEME4_GWF_RESULT_SUFFIX=${SCHEME4_GWF_RESULT_SUFFIX:-scheme4_gwf}
SCHEME5_RESULT_SUFFIX=${SCHEME5_RESULT_SUFFIX:-scheme5}
CURRENT_CAPACITY_RESULT_SUFFIX=${CURRENT_CAPACITY_RESULT_SUFFIX:-p2d_current_capacity}
CURRENT_3DLOAD_RESULT_SUFFIX=${CURRENT_3DLOAD_RESULT_SUFFIX:-p2d_current_3Dload}
DECODE_PREDICT_3DLOAD_RESULT_SUFFIX=${DECODE_PREDICT_3DLOAD_RESULT_SUFFIX:-decode_predict_3Dload}
ORACLE_RESULT_SUFFIX=${ORACLE_RESULT_SUFFIX:-oracle}
COMPARE_EXPERIMENT_VARIANTS=${COMPARE_EXPERIMENT_VARIANTS:-"roundrobin scheme4_gwf"}

# Trace selection. Empty TRACE_NAME means use TRACE_LIST.
TRACE_NAME=${TRACE_NAME:-}
TRACE_LIST=${TRACE_LIST:-"qwen,mooncake"}

TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}

# Base output directory. Per-run subdirectories are created automatically.
BASE_RESULT_DIR=${BASE_RESULT_DIR:-./experiment_result_2}

# User-specified conflict resolutions
PROXY_PORT=${PROXY_PORT:-28002}
BENCH_PORT=${BENCH_PORT:-22007}
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}

# Shared runtime knobs across all PD ratios.
PREFILL_GPU_MEMORY_UTILIZATION=${PREFILL_GPU_MEMORY_UTILIZATION:-0.8}
PREFILL_TENSOR_PARALLEL_SIZE=${PREFILL_TENSOR_PARALLEL_SIZE:-1}
PREFILL_TENSOR_POOL_MEMORY=${PREFILL_TENSOR_POOL_MEMORY:-32}

DECODE_GPU_MEMORY_UTILIZATION=${DECODE_GPU_MEMORY_UTILIZATION:-0.8}
DECODE_TENSOR_PARALLEL_SIZE=${DECODE_TENSOR_PARALLEL_SIZE:-1}
DECODE_TENSOR_POOL_MEMORY=${DECODE_TENSOR_POOL_MEMORY:-32}

KV_CONNECTOR=${KV_CONNECTOR:-P2pNcclConnector}
KV_PRODUCER_BUFFER=${KV_PRODUCER_BUFFER:-1e9}
KV_CONSUMER_BUFFER=${KV_CONSUMER_BUFFER:-8e9}
KV_NCCL_CHANNELS=${KV_NCCL_CHANNELS:-8}
KV_SEND_TYPE=${KV_SEND_TYPE:-PUT_ASYNC}

VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-1}
VLLM_SEED=${VLLM_SEED:-42}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-}
PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS:-}
DECODE_VLLM_MAX_NUM_BATCHED_TOKENS=${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-1024}

BENCH_SCRIPT=${BENCH_SCRIPT:-/root/predict-schedule/experiment/zrecent_use/disagg_serving_suite/benchmark_serving_trace.py}
BENCH_MODEL=${BENCH_MODEL:-}
BENCH_BACKEND=${BENCH_BACKEND:-openai-chat}
BENCH_ENDPOINT=${BENCH_ENDPOINT:-/v1/chat/completions}
BENCH_DATASET_NAME=${BENCH_DATASET_NAME:-trace}
BENCH_MAX_CONCURRENCY=${BENCH_MAX_CONCURRENCY:-1024}
BENCH_GOODPUT_TTFT_MS=${BENCH_GOODPUT_TTFT_MS:-1000}

BENCH_TEMPERATURE=${BENCH_TEMPERATURE:-0.8}
BENCH_TOP_P=${BENCH_TOP_P:-0.7}
BENCH_TOP_K=${BENCH_TOP_K:-50}
BENCH_REPETITION_PENALTY=${BENCH_REPETITION_PENALTY:-1.0}

SAVE_OUTPUT=${SAVE_OUTPUT:-True}
SAVE_SAMPLE=${SAVE_SAMPLE:-true}
BENCH_IGNORE_EOS=${BENCH_IGNORE_EOS:-ignore-eos}

USE_PASTFUTURE_SCHEDULER=${USE_PASTFUTURE_SCHEDULER:-false}
USE_ACTIVATION_PREDICTOR=${USE_ACTIVATION_PREDICTOR:-false}
USE_CUSTOM_PROXY=${USE_CUSTOM_PROXY:-false}
USE_GWF_SCHEDULER=${USE_GWF_SCHEDULER:-false}
TEST_ABLATION_P2D=${TEST_ABLATION_P2D:-false}
TEST_ABLATION_P2D_CURRENT_3DLOAD=${TEST_ABLATION_P2D_CURRENT_3DLOAD:-false}
TEST_ABLATION_P2D_CURRENT_CAPACITY=${TEST_ABLATION_P2D_CURRENT_CAPACITY:-false}
TEST_OPTIMAL=${TEST_OPTIMAL:-false}
TEST_DECODE_PREDICT_3DLOAD=${TEST_DECODE_PREDICT_3DLOAD:-false}

# Predictor-aware route scheme.
# Current scheme3 behavior in test_disagg_proxy.py:
# 1. compute current and predictive 3D load for each decode instance;
# 2. shortlist the lowest half by Pred_Load_Bottle;
# 3. select the lowest Load_Bottle within that shortlist;
# 4. immediately write the chosen request's token contribution back to the
#    decode instance predictive state to avoid stale routing decisions.
WT_PREDICT_ROUTE_SCHEME=${WT_PREDICT_ROUTE_SCHEME:-scheme4}
WT_SCHEME4_LOAD_THRESHOLD=${WT_SCHEME4_LOAD_THRESHOLD:-1.0}
WT_SCHEME4_RESERVATION_TOKENS=${WT_SCHEME4_RESERVATION_TOKENS:-}
# [WT] predictor output-length calibration (measured pred/actual ~4.36x on the
# eval traces). 1.0 = off.  This is a one-time calibration, not a swept knob.
WT_PREDICT_OUTPUT_SCALE=${WT_PREDICT_OUTPUT_SCALE:-1.0}
# [WT] cleanup start: scheme5 reroute defaults
WT_SCHEME5_MAX_REROUTES=${WT_SCHEME5_MAX_REROUTES:-1}
WT_SCHEME5_TIMEOUT_RATIO=${WT_SCHEME5_TIMEOUT_RATIO:-0.9}
WT_SCHEME5_FIRST_CHUNK_TIMEOUT_MS=${WT_SCHEME5_FIRST_CHUNK_TIMEOUT_MS:-${BENCH_GOODPUT_TTFT_MS}}
WT_SCHEME5_EXCLUDE_PREVIOUS_DECODE=${WT_SCHEME5_EXCLUDE_PREVIOUS_DECODE:-1}
# [WT] cleanup end
# [WT] GWF intra-instance scheduler (vllm/v1/core/sched/geometric_window_scheduler.py).
# Admission is self-calibrating on the measured TPOT; the only exposed knobs are
# the SLO budgets (WT_GWF_TTFT_BUDGET_MS / WT_GWF_TPOT_BUDGET_MS, derived from
# the benchmark below) and the anti-starvation safety bound.
# Hard anti-starvation bound: a demoted (hopeless) request is force-served
# after at most this long, even while new requests keep arriving. 0 disables.
WT_GWF_HOPELESS_MAX_DEFER_MS=${WT_GWF_HOPELESS_MAX_DEFER_MS:-2000}
WT_GWF_LOG_INTERVAL_MS=${WT_GWF_LOG_INTERVAL_MS:-1000}
# 1 = reproduce the baseline scheduler exactly (stock FCFS waiting queue, no
# admission gate, no running reorder). "Worst case == baseline" reference and
# a clean ablation of the whole mechanism.
WT_GWF_FORCE_IDLE=${WT_GWF_FORCE_IDLE:-0}
# RoundRobin++ knobs
RRPP_KV_THRESHOLD=${RRPP_KV_THRESHOLD:-0.8}
RRPP_POLL_INTERVAL_MS=${RRPP_POLL_INTERVAL_MS:-5}
# Performance-sensitive logging knobs.
# For throughput/latency evaluation, keep both disabled unless debugging.
WT_PREDICT_CHAIN_VERBOSE=${WT_PREDICT_CHAIN_VERBOSE:-0}
WT_ROUTE_DECISION_LOG_ENABLED=${WT_ROUTE_DECISION_LOG_ENABLED:-0}
WT_MONITOR_LOG_INTERVAL_MS=${WT_MONITOR_LOG_INTERVAL_MS:-1000}
WT_PREDICT_ANALYSIS_ENABLE=${WT_PREDICT_ANALYSIS_ENABLE:-1}
WT_TIMING_VERBOSE=${WT_TIMING_VERBOSE:-0}
WT_TIMING_PENDING_CHECK_MS=${WT_TIMING_PENDING_CHECK_MS:-1}
WT_KV_FLUSH_LOG_INTERVAL_MS=${WT_KV_FLUSH_LOG_INTERVAL_MS:-1000}
WT_KV_FLUSH_SLEEP_MS=${WT_KV_FLUSH_SLEEP_MS:-1}
WT_KV_FLUSH_VERBOSE=${WT_KV_FLUSH_VERBOSE:-0}
WT_KV_LAYER_PACK_ENABLE=${WT_KV_LAYER_PACK_ENABLE:-1}
WT_KV_ROUTE_AWARE_IMMEDIATE_SEND=${WT_KV_ROUTE_AWARE_IMMEDIATE_SEND:-1}

USE_TRACE_TIMESTAMPS=${USE_TRACE_TIMESTAMPS:-true}
# PD_RATIO_LIST=${PD_RATIO_LIST:-"1:3"}
PD_RATIO_LIST=${PD_RATIO_LIST:-"2:6,4:4"}
TPOT_MS_LIST=${TPOT_MS_LIST:-"30,50"}

# PD profile mappings (override these to match your machine layout).
PD_1P1D_PREFILL_GPUS=${PD_1P1D_PREFILL_GPUS:-6}
PD_1P1D_PREFILL_PORTS=${PD_1P1D_PREFILL_PORTS:-22001}
PD_1P1D_PREFILL_KV_PORTS=${PD_1P1D_PREFILL_KV_PORTS:-22011}
PD_1P1D_DECODE_GPUS=${PD_1P1D_DECODE_GPUS:-7}
PD_1P1D_DECODE_PORTS=${PD_1P1D_DECODE_PORTS:-22020}
PD_1P1D_DECODE_KV_PORTS=${PD_1P1D_DECODE_KV_PORTS:-22032}

PD_1P3D_PREFILL_GPUS=${PD_1P3D_PREFILL_GPUS:-0}
PD_1P3D_PREFILL_PORTS=${PD_1P3D_PREFILL_PORTS:-22001}
PD_1P3D_PREFILL_KV_PORTS=${PD_1P3D_PREFILL_KV_PORTS:-22011}
PD_1P3D_DECODE_GPUS=${PD_1P3D_DECODE_GPUS:-1,3,2}
PD_1P3D_DECODE_PORTS=${PD_1P3D_DECODE_PORTS:-22020,22021,22022}
PD_1P3D_DECODE_KV_PORTS=${PD_1P3D_DECODE_KV_PORTS:-22032,22033,22034}

PD_2P6D_PREFILL_GPUS=${PD_2P6D_PREFILL_GPUS:-0,1}
PD_2P6D_PREFILL_PORTS=${PD_2P6D_PREFILL_PORTS:-22001,22002}
PD_2P6D_PREFILL_KV_PORTS=${PD_2P6D_PREFILL_KV_PORTS:-22011,22012}
PD_2P6D_DECODE_GPUS=${PD_2P6D_DECODE_GPUS:-2,3,4,5,6,7}
PD_2P6D_DECODE_PORTS=${PD_2P6D_DECODE_PORTS:-22020,22021,22022,22023,22024,22025}
PD_2P6D_DECODE_KV_PORTS=${PD_2P6D_DECODE_KV_PORTS:-22032,22033,22034,22035,22036,22037}

PD_4P4D_PREFILL_GPUS=${PD_4P4D_PREFILL_GPUS:-0,1,2,3}
PD_4P4D_PREFILL_PORTS=${PD_4P4D_PREFILL_PORTS:-22001,22002,22003,22004}
PD_4P4D_PREFILL_KV_PORTS=${PD_4P4D_PREFILL_KV_PORTS:-22011,22012,22013,22014}
PD_4P4D_DECODE_GPUS=${PD_4P4D_DECODE_GPUS:-4,5,6,7}
PD_4P4D_DECODE_PORTS=${PD_4P4D_DECODE_PORTS:-22020,22021,22022,22023}
PD_4P4D_DECODE_KV_PORTS=${PD_4P4D_DECODE_KV_PORTS:-22032,22033,22034,22035}

# 3P4D: 3 prefill + 4 decode = 7 GPUs (frees GPU 3 for other work).
PD_3P4D_PREFILL_GPUS=${PD_3P4D_PREFILL_GPUS:-0,1,2}
PD_3P4D_PREFILL_PORTS=${PD_3P4D_PREFILL_PORTS:-22001,22002,22003}
PD_3P4D_PREFILL_KV_PORTS=${PD_3P4D_PREFILL_KV_PORTS:-22011,22012,22013}
PD_3P4D_DECODE_GPUS=${PD_3P4D_DECODE_GPUS:-4,5,6,7}
PD_3P4D_DECODE_PORTS=${PD_3P4D_DECODE_PORTS:-22020,22021,22022,22023}
PD_3P4D_DECODE_KV_PORTS=${PD_3P4D_DECODE_KV_PORTS:-22032,22033,22034,22035}

# Active runtime assignment is selected by configure_pd_ratio().
# Initialize with 1:3 profile so cleanup paths always have valid values.
PREFILL_GPUS="$PD_1P3D_PREFILL_GPUS"
PREFILL_PORTS="$PD_1P3D_PREFILL_PORTS"
PREFILL_KV_PORTS="$PD_1P3D_PREFILL_KV_PORTS"
DECODE_GPUS="$PD_1P3D_DECODE_GPUS"
DECODE_PORTS="$PD_1P3D_DECODE_PORTS"
DECODE_KV_PORTS="$PD_1P3D_DECODE_KV_PORTS"

RUN_REPEAT_PER_CONFIG=${RUN_REPEAT_PER_CONFIG:-1}
SLEEP_BETWEEN_RUNS=${SLEEP_BETWEEN_RUNS:-5}
GPU_IDLE_WAIT_SECONDS=${GPU_IDLE_WAIT_SECONDS:-120}
GPU_IDLE_POLL_INTERVAL_SECONDS=${GPU_IDLE_POLL_INTERVAL_SECONDS:-2}
GPU_IDLE_STABLE_SAMPLES=${GPU_IDLE_STABLE_SAMPLES:-3}
SKIP_EXISTING_RESULT_DIRS=${SKIP_EXISTING_RESULT_DIRS:-}
WT_TEMP_SKIP_EXISTING_RESULT_DIRS_ONCE_MARKER=${WT_TEMP_SKIP_EXISTING_RESULT_DIRS_ONCE_MARKER:-/tmp/wt_scheme4_rr_compare_skip_success_once_20260505.done}
WT_TEMP_SKIP_EXISTING_RESULT_DIRS_ONCE=${WT_TEMP_SKIP_EXISTING_RESULT_DIRS_ONCE:-"./experiment_result_2/llama_llama-lmsys-chat_n1683_pd2p6d_tpot30ms_rep1_20260505_023513_roundrobin
./experiment_result_2/llama_llama-lmsys-chat_n1683_pd2p6d_tpot30ms_rep1_20260505_025332_scheme4
./experiment_result_2/llama_llama-lmsys-chat_n1683_pd2p6d_tpot30ms_rep1_20260505_031154_p2d_current_capacity
./experiment_result_2/llama_llama-lmsys-chat_n1683_pd2p6d_tpot30ms_rep1_20260505_033017_p2d_current_3Dload
./experiment_result_2/llama_llama-lmsys-chat_n1683_pd2p6d_tpot30ms_rep1_20260505_034839_oracle
./experiment_result_2/llama_llama-lmsys-chat_n1683_pd4p4d_tpot30ms_rep1_20260504_101941_roundrobin
./experiment_result_2/llama_llama-lmsys-chat_n1683_pd4p4d_tpot30ms_rep1_20260504_103838_scheme4
./experiment_result_2/llama_llama-lmsys-chat_n1683_pd4p4d_tpot30ms_rep1_20260504_105737_p2d_current_capacity
./experiment_result_2/llama_llama-lmsys-chat_n1683_pd4p4d_tpot30ms_rep1_20260504_210607_roundrobin
./experiment_result_2/llama_llama-lmsys-chat_n1683_pd4p4d_tpot30ms_rep1_20260504_212439_scheme4
./experiment_result_2/llama_llama-lmsys-chat_n1683_pd4p4d_tpot30ms_rep1_20260504_214305_p2d_current_capacity
./experiment_result_2/llama_llama-lmsys-chat_n1683_pd4p4d_tpot30ms_rep1_20260504_220133_p2d_current_3Dload
./experiment_result_2/llama_llama-lmsys-chat_n1683_pd4p4d_tpot30ms_rep1_20260504_221954_oracle
./experiment_result_2/llama_llama-lmsys-chat_n2262_pd2p6d_tpot30ms_rep1_20260505_053903_roundrobin
./experiment_result_2/llama_llama-lmsys-chat_n2262_pd2p6d_tpot30ms_rep1_20260505_054605_scheme4
./experiment_result_2/llama_llama-lmsys-chat_n2262_pd4p4d_tpot30ms_rep1_20260504_122633_roundrobin
./experiment_result_2/llama_llama-lmsys-chat_n2262_pd4p4d_tpot30ms_rep1_20260504_123400_scheme4
./experiment_result_2/llama_llama-lmsys-chat_n2262_pd4p4d_tpot30ms_rep1_20260504_124042_p2d_current_capacity
./experiment_result_2/llama_llama-lmsys-chat_n2262_pd4p4d_tpot30ms_rep1_20260505_002156_scheme4
./experiment_result_2/llama_llama-lmsys-chat_n2262_pd4p4d_tpot30ms_rep1_20260505_002843_p2d_current_capacity
./experiment_result_2/llama_llama-lmsys-chat_n2262_pd4p4d_tpot30ms_rep1_20260505_003528_p2d_current_3Dload
./experiment_result_2/llama_llama-lmsys-chat_n2262_pd4p4d_tpot30ms_rep1_20260505_004213_oracle
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd2p6d_tpot30ms_rep1_20260504_131329_p2d_current_capacity
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd2p6d_tpot30ms_rep1_20260504_132042_p2d_current_3Dload
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd2p6d_tpot30ms_rep1_20260505_012746_roundrobin
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd2p6d_tpot30ms_rep1_20260505_013427_scheme4
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd2p6d_tpot30ms_rep1_20260505_014109_p2d_current_capacity
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd2p6d_tpot30ms_rep1_20260505_014749_p2d_current_3Dload
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd2p6d_tpot30ms_rep1_20260505_015436_oracle
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd4p4d_tpot30ms_rep1_20260504_083309_roundrobin
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd4p4d_tpot30ms_rep1_20260504_084220_scheme4
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd4p4d_tpot30ms_rep1_20260504_084936_p2d_current_capacity
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd4p4d_tpot30ms_rep1_20260504_085637_p2d_current_3Dload
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd4p4d_tpot30ms_rep1_20260504_194939_scheme4
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd4p4d_tpot30ms_rep1_20260504_195635_p2d_current_capacity
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd4p4d_tpot30ms_rep1_20260504_200322_p2d_current_3Dload
./experiment_result_2/llama_llama-lmsys-chat_n2521_pd4p4d_tpot30ms_rep1_20260504_201009_oracle
./experiment_result_2/llama_llama-mysharegpt_n1683_pd2p6d_tpot30ms_rep1_20260505_040702_roundrobin
./experiment_result_2/llama_llama-mysharegpt_n1683_pd2p6d_tpot30ms_rep1_20260505_042525_scheme4
./experiment_result_2/llama_llama-mysharegpt_n1683_pd2p6d_tpot30ms_rep1_20260505_044353_p2d_current_capacity
./experiment_result_2/llama_llama-mysharegpt_n1683_pd2p6d_tpot30ms_rep1_20260505_050214_p2d_current_3Dload
./experiment_result_2/llama_llama-mysharegpt_n1683_pd2p6d_tpot30ms_rep1_20260505_052036_oracle
./experiment_result_2/llama_llama-mysharegpt_n1683_pd4p4d_tpot30ms_rep1_20260504_113047_roundrobin
./experiment_result_2/llama_llama-mysharegpt_n1683_pd4p4d_tpot30ms_rep1_20260504_223822_roundrobin
./experiment_result_2/llama_llama-mysharegpt_n1683_pd4p4d_tpot30ms_rep1_20260504_225649_scheme4
./experiment_result_2/llama_llama-mysharegpt_n1683_pd4p4d_tpot30ms_rep1_20260504_231517_p2d_current_capacity
./experiment_result_2/llama_llama-mysharegpt_n1683_pd4p4d_tpot30ms_rep1_20260504_233343_p2d_current_3Dload
./experiment_result_2/llama_llama-mysharegpt_n1683_pd4p4d_tpot30ms_rep1_20260504_235207_oracle
./experiment_result_2/llama_llama-mysharegpt_n2262_pd4p4d_tpot30ms_rep1_20260505_010043_scheme4
./experiment_result_2/llama_llama-mysharegpt_n2262_pd4p4d_tpot30ms_rep1_20260505_010728_p2d_current_capacity
./experiment_result_2/llama_llama-mysharegpt_n2262_pd4p4d_tpot30ms_rep1_20260505_011413_p2d_current_3Dload
./experiment_result_2/llama_llama-mysharegpt_n2262_pd4p4d_tpot30ms_rep1_20260505_012059_oracle
./experiment_result_2/llama_llama-mysharegpt_n2521_pd2p6d_tpot30ms_rep1_20260505_020117_roundrobin
./experiment_result_2/llama_llama-mysharegpt_n2521_pd2p6d_tpot30ms_rep1_20260505_020802_scheme4
./experiment_result_2/llama_llama-mysharegpt_n2521_pd2p6d_tpot30ms_rep1_20260505_021454_p2d_current_capacity
./experiment_result_2/llama_llama-mysharegpt_n2521_pd2p6d_tpot30ms_rep1_20260505_022142_p2d_current_3Dload
./experiment_result_2/llama_llama-mysharegpt_n2521_pd2p6d_tpot30ms_rep1_20260505_022828_oracle
./experiment_result_2/llama_llama-mysharegpt_n2521_pd4p4d_tpot30ms_rep1_20260504_090705_roundrobin
./experiment_result_2/llama_llama-mysharegpt_n2521_pd4p4d_tpot30ms_rep1_20260504_091544_scheme4
./experiment_result_2/llama_llama-mysharegpt_n2521_pd4p4d_tpot30ms_rep1_20260504_092251_p2d_current_capacity
./experiment_result_2/llama_llama-mysharegpt_n2521_pd4p4d_tpot30ms_rep1_20260504_093037_p2d_current_3Dload
./experiment_result_2/llama_llama-mysharegpt_n2521_pd4p4d_tpot30ms_rep1_20260504_205200_p2d_current_3Dload
./experiment_result_2/llama_llama-mysharegpt_n2521_pd4p4d_tpot30ms_rep1_20260504_205901_oracle"}

BENCH_GOODPUT=""
CACULATE_GOODPUT=""
INSTANCE_DOWN_TIMEOUT_SECONDS=${INSTANCE_DOWN_TIMEOUT_SECONDS:-120}
INSTANCE_HEALTH_CHECK_INTERVAL_SECONDS=${INSTANCE_HEALTH_CHECK_INTERVAL_SECONDS:-5}
# [WT] Max number of times to restart a config when a prefill/decode instance crashes during a benchmark run (0 = no retry)
INSTANCE_RESTART_MAX_RETRIES=${INSTANCE_RESTART_MAX_RETRIES:-2}
# [WT] GPU deadlock detection: all decode GPUs memory>0 but util==0 persists for this many seconds => skip config
GPU_DEADLOCK_TIMEOUT_SECONDS=${GPU_DEADLOCK_TIMEOUT_SECONDS:-300}

PROXY_BASE_DIR=${PROXY_BASE_DIR:-/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd}
# [WT] scheme4/scheme4_gwf use the disagg_serving_suite proxy (more complete:
# scheme5 reroute, oracle routing, corrected load model formulas).
DISAGG_SERVING_SUITE_PROXY=${DISAGG_SERVING_SUITE_PROXY:-/root/predict-schedule/experiment/zrecent_use/disagg_serving_suite/test_disagg_proxy.py}
# [WT] cleanup kept for audit:
# PROXY_SCRIPT=${PROXY_SCRIPT:-${PROXY_BASE_DIR}/test_disagg_proxy.py}
PROXY_SCRIPT=${PROXY_SCRIPT:-${PROXY_BASE_DIR}/disagg_proxy_p2p_nccl_xpyd.py}
PROXY_SCRIPT_OVERRIDE=${PROXY_SCRIPT_OVERRIDE:-}
PLOT_SCRIPT=${PLOT_SCRIPT:-${PROXY_BASE_DIR}/plot_decode_load2.py}

####################
# END USER CONFIG
####################

cd "$(dirname "${BASH_SOURCE[0]}")"
DATASET_BASE_DIR="$(realpath "$DATASET_BASE_DIR")"
SCHEME4_SELECTION_DATASET_DIR="$(realpath -m "$SCHEME4_SELECTION_DATASET_DIR")"

PIDS=()
PGIDS=()
PREFILL_PIDS=()
DECODE_PIDS=()

CURRENT_BENCH_DATASET_PATH=""
CURRENT_BENCH_DATASET_LABEL=""
CURRENT_BENCH_DATASET_TAG=""

BENCHMARK_DIR=""
CONFIG_DIR=""
LOG_DIR=""
RESULT_DIR=""
MODEL_MT_TAG_FOR_NAME=""
CURRENT_MODEL_NAME=""
CURRENT_DATASET_ROOT=""
CURRENT_TRACE_NAME=""
CURRENT_TRACE_PATH=""
CURRENT_PD_RATIO=""
CURRENT_PD_RATIO_TAG=""
CURRENT_TPOT_MS=""
CURRENT_DATASET_SIZE=""
CURRENT_BENCH_MAX_TOKENS=""
CURRENT_BENCH_MODEL=""
CURRENT_EXPERIMENT_MODE=""
CURRENT_RUN_METHOD=""
CURRENT_RUN_SUFFIX=""
RUN_BATCH_TIMESTAMP=${RUN_BATCH_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
FAILED_CONFIG_LOG=""
SUMMARY_CSV=""

USER_VLLM_MAX_MODEL_LEN_OVERRIDE=${VLLM_MAX_MODEL_LEN:-}
USER_PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS_OVERRIDE=${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS:-}
USER_WT_PREDICT_ANALYSIS_ENABLE_OVERRIDE=${WT_PREDICT_ANALYSIS_ENABLE:-1}

_CLEANED_UP=0
LAST_GPU_IDLE_FAILURE_REASON=""
declare -A SKIP_RESULT_DIR_KEYS=()

is_true() {
    local v="${1:-}"
    [[ "${v,,}" == "true" ]]
}

set_proxy_script_path() {
    local proxy_name="$1"
    PROXY_SCRIPT="${PROXY_BASE_DIR}/${proxy_name}"
}

configure_experiment_variant() {
    local variant="$1"

    case "$variant" in
        roundrobin)
            USE_CUSTOM_PROXY=false
            USE_ACTIVATION_PREDICTOR=false
            USE_GWF_SCHEDULER=false
            TEST_ABLATION_P2D=false
            TEST_ABLATION_P2D_CURRENT_3DLOAD=false
            TEST_ABLATION_P2D_CURRENT_CAPACITY=false
            TEST_OPTIMAL=false
            WT_PREDICT_ROUTE_SCHEME=roundrobin
            WT_PREDICT_ANALYSIS_ENABLE=0
            CURRENT_RUN_METHOD="roundrobin"
            CURRENT_RUN_SUFFIX="$ROUNDROBIN_RESULT_SUFFIX"
            ;;
        roundrobin_pp)
            USE_CUSTOM_PROXY=false
            USE_ACTIVATION_PREDICTOR=false
            USE_GWF_SCHEDULER=false
            TEST_ABLATION_P2D=false
            TEST_ABLATION_P2D_CURRENT_3DLOAD=false
            TEST_ABLATION_P2D_CURRENT_CAPACITY=false
            TEST_OPTIMAL=false
            WT_PREDICT_ROUTE_SCHEME=roundrobin_pp
            WT_PREDICT_ANALYSIS_ENABLE=0
            CURRENT_RUN_METHOD="roundrobin_pp"
            CURRENT_RUN_SUFFIX="$ROUNDROBIN_PP_RESULT_SUFFIX"
            ;;
        scheme4)
            USE_CUSTOM_PROXY=false
            USE_ACTIVATION_PREDICTOR=true
            TEST_ABLATION_P2D=true
            TEST_ABLATION_P2D_CURRENT_3DLOAD=false
            TEST_ABLATION_P2D_CURRENT_CAPACITY=false
            TEST_OPTIMAL=false
            USE_GWF_SCHEDULER=false
            WT_PREDICT_ROUTE_SCHEME=scheme4
            WT_PREDICT_ANALYSIS_ENABLE="$USER_WT_PREDICT_ANALYSIS_ENABLE_OVERRIDE"
            CURRENT_RUN_METHOD="scheme4"
            CURRENT_RUN_SUFFIX="$SCHEME4_RESULT_SUFFIX"
            ;;
        scheme4_gwf)
            USE_CUSTOM_PROXY=false
            USE_ACTIVATION_PREDICTOR=true
            TEST_ABLATION_P2D=true
            TEST_ABLATION_P2D_CURRENT_3DLOAD=false
            TEST_ABLATION_P2D_CURRENT_CAPACITY=false
            TEST_OPTIMAL=false
            USE_GWF_SCHEDULER=true
            WT_PREDICT_ROUTE_SCHEME=scheme4
            WT_PREDICT_ANALYSIS_ENABLE="$USER_WT_PREDICT_ANALYSIS_ENABLE_OVERRIDE"
            CURRENT_RUN_METHOD="scheme4_gwf"
            CURRENT_RUN_SUFFIX="$SCHEME4_GWF_RESULT_SUFFIX"
            ;;
        scheme5)
            USE_CUSTOM_PROXY=false
            USE_ACTIVATION_PREDICTOR=true
            USE_GWF_SCHEDULER=false
            TEST_ABLATION_P2D=true
            TEST_ABLATION_P2D_CURRENT_3DLOAD=false
            TEST_ABLATION_P2D_CURRENT_CAPACITY=false
            TEST_OPTIMAL=false
            WT_PREDICT_ROUTE_SCHEME=scheme5
            WT_PREDICT_ANALYSIS_ENABLE="$USER_WT_PREDICT_ANALYSIS_ENABLE_OVERRIDE"
            CURRENT_RUN_METHOD="scheme5"
            CURRENT_RUN_SUFFIX="$SCHEME5_RESULT_SUFFIX"
            ;;
        current_capacity)
            USE_CUSTOM_PROXY=false
            USE_ACTIVATION_PREDICTOR=false
            USE_GWF_SCHEDULER=false
            TEST_ABLATION_P2D=false
            TEST_ABLATION_P2D_CURRENT_3DLOAD=false
            TEST_ABLATION_P2D_CURRENT_CAPACITY=true
            TEST_OPTIMAL=false
            WT_PREDICT_ROUTE_SCHEME=current_capacity
            WT_PREDICT_ANALYSIS_ENABLE=0
            CURRENT_RUN_METHOD="p2d_current_capacity"
            CURRENT_RUN_SUFFIX="$CURRENT_CAPACITY_RESULT_SUFFIX"
            ;;
        current_3dload)
            USE_CUSTOM_PROXY=false
            USE_ACTIVATION_PREDICTOR=false
            USE_GWF_SCHEDULER=false
            TEST_ABLATION_P2D=false
            TEST_ABLATION_P2D_CURRENT_3DLOAD=true
            TEST_ABLATION_P2D_CURRENT_CAPACITY=false
            TEST_OPTIMAL=false
            WT_PREDICT_ROUTE_SCHEME=current_3dload
            WT_PREDICT_ANALYSIS_ENABLE=0
            CURRENT_RUN_METHOD="p2d_current_3Dload"
            CURRENT_RUN_SUFFIX="$CURRENT_3DLOAD_RESULT_SUFFIX"
            ;;
        decode_predict_3Dload)
            USE_CUSTOM_PROXY=false
            USE_ACTIVATION_PREDICTOR=false
            USE_GWF_SCHEDULER=false
            TEST_ABLATION_P2D=false
            TEST_ABLATION_P2D_CURRENT_3DLOAD=false
            TEST_ABLATION_P2D_CURRENT_CAPACITY=false
            TEST_OPTIMAL=false
            TEST_DECODE_PREDICT_3DLOAD=true
            WT_PREDICT_ROUTE_SCHEME=decode_predict_3Dload
            WT_PREDICT_ANALYSIS_ENABLE=0
            CURRENT_RUN_METHOD="decode_predict_3Dload"
            CURRENT_RUN_SUFFIX="$DECODE_PREDICT_3DLOAD_RESULT_SUFFIX"
            ;;
        oracle)
            USE_CUSTOM_PROXY=false
            USE_ACTIVATION_PREDICTOR=false
            USE_GWF_SCHEDULER=false
            TEST_ABLATION_P2D=true
            TEST_ABLATION_P2D_CURRENT_3DLOAD=false
            TEST_ABLATION_P2D_CURRENT_CAPACITY=false
            TEST_OPTIMAL=false
            WT_PREDICT_ROUTE_SCHEME=oracle
            WT_PREDICT_ANALYSIS_ENABLE=0
            CURRENT_RUN_METHOD="oracle"
            CURRENT_RUN_SUFFIX="$ORACLE_RESULT_SUFFIX"
            ;;
        *)
            echo "ERROR: unsupported experiment variant=${variant}, expected roundrobin/roundrobin_pp/scheme4/scheme4_gwf/scheme5/current_capacity/current_3dload/oracle"
            return 1
            ;;
    esac

    update_proxy_script
}

resolve_experiment_mode() {
    if [[ -n "$PROXY_SCRIPT_OVERRIDE" ]]; then
        CURRENT_EXPERIMENT_MODE="proxy_override"
    elif is_true "$TEST_DECODE_PREDICT_3DLOAD"; then
        CURRENT_EXPERIMENT_MODE="decode_predict_3Dload"
    elif is_true "$TEST_ABLATION_P2D_CURRENT_3DLOAD"; then
        CURRENT_EXPERIMENT_MODE="current_3dload"
    elif is_true "$TEST_ABLATION_P2D_CURRENT_CAPACITY"; then
        CURRENT_EXPERIMENT_MODE="current_capacity"
    elif is_true "$TEST_ABLATION_P2D"; then
        CURRENT_EXPERIMENT_MODE="full_method"
    elif is_true "$USE_CUSTOM_PROXY"; then
        CURRENT_EXPERIMENT_MODE="custom_proxy"
    elif is_true "$TEST_OPTIMAL"; then
        CURRENT_EXPERIMENT_MODE="optimal"
    elif [[ "$WT_PREDICT_ROUTE_SCHEME" == "roundrobin_pp" ]]; then
        CURRENT_EXPERIMENT_MODE="roundrobin_pp"
    else
        CURRENT_EXPERIMENT_MODE="baseline"
    fi
}

update_proxy_script() {
    if [[ -n "$PROXY_SCRIPT_OVERRIDE" ]]; then
        PROXY_SCRIPT="$PROXY_SCRIPT_OVERRIDE"
    elif is_true "$TEST_DECODE_PREDICT_3DLOAD"; then
        set_proxy_script_path "decode_predict_3Dload.py"
    elif is_true "$TEST_ABLATION_P2D_CURRENT_3DLOAD"; then
        set_proxy_script_path "p2d_current_3Dload.py"
    elif is_true "$TEST_ABLATION_P2D_CURRENT_CAPACITY"; then
        set_proxy_script_path "p2d_current_capacity.py"
    elif is_true "$TEST_ABLATION_P2D"; then
        PROXY_SCRIPT="$DISAGG_SERVING_SUITE_PROXY"
    elif is_true "$USE_CUSTOM_PROXY"; then
        set_proxy_script_path "test_disagg_proxy.py"
    elif is_true "$TEST_OPTIMAL"; then
        set_proxy_script_path "test_disagg_proxy_p2d_loadbalance.py"
    elif [[ "$WT_PREDICT_ROUTE_SCHEME" == "roundrobin_pp" ]]; then
        set_proxy_script_path "disagg_proxy_p2p_nccl_xpyd_roundrobin_pp.py"
    else
        set_proxy_script_path "disagg_proxy_p2p_nccl_xpyd.py"
    fi
    resolve_experiment_mode
}

sanitize_name() {
    local raw_name="$1"
    raw_name="${raw_name// /_}"
    raw_name="$(printf '%s' "$raw_name" | sed 's/[^[:alnum:]_.-]/_/g')"
    printf '%s' "$raw_name"
}

apply_model_dependent_defaults() {
    local model_lc
    model_lc="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"

    if [[ -z "$VLLM_MAX_MODEL_LEN" ]]; then
        if [[ "$model_lc" == *"llama-3-8b-instruct"* ]] || [[ "$model_lc" == *"meta-llama-3-8b-instruct"* ]]; then
            VLLM_MAX_MODEL_LEN=8192
        elif [[ "$model_lc" == *"qwen2.5-7b-instruct"* ]]; then
            VLLM_MAX_MODEL_LEN=32768
        else
            VLLM_MAX_MODEL_LEN=32768
        fi
    fi

    if [[ -z "$PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS" ]]; then
        if [[ "$model_lc" == *"llama-3-8b-instruct"* ]] || [[ "$model_lc" == *"meta-llama-3-8b-instruct"* ]]; then
            PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=32768
        elif [[ "$model_lc" == *"qwen2.5-7b-instruct"* ]]; then
            PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=40960
        else
            PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=32768
        fi
    fi
}

resolve_model_mt_tag_for_name() {
    local type_lc model_lc test_model_lc
    type_lc="$(echo "$CURRENT_MODEL_NAME" | tr '[:upper:]' '[:lower:]')"
    model_lc="$(echo "$MODEL" | tr '[:upper:]' '[:lower:]')"
    test_model_lc="$(echo "$TEST_MODEL" | tr '[:upper:]' '[:lower:]')"

    if [[ "$type_lc" == "llama" ]] || [[ "$model_lc" == *"llama"* ]] || [[ "$test_model_lc" == *"llama"* ]]; then
        MODEL_MT_TAG_FOR_NAME="8192"
    elif [[ "$type_lc" == "qwen" ]] || [[ "$model_lc" == *"qwen"* ]] || [[ "$test_model_lc" == *"qwen"* ]]; then
        MODEL_MT_TAG_FOR_NAME="32768"
    else
        # Fallback to current runtime max model len when model family is unclear.
        MODEL_MT_TAG_FOR_NAME="${VLLM_MAX_MODEL_LEN}"
    fi
}

join_by_comma() {
    local result=""
    local item
    for item in "$@"; do
        if [[ -z "$result" ]]; then
            result="$item"
        else
            result+=",$item"
        fi
    done
    printf '%s' "$result"
}

generate_comma_range() {
    local start="$1"
    local count="$2"
    local values=()
    local index
    for ((index = 0; index < count; index++)); do
        values+=("$((start + index))")
    done
    join_by_comma "${values[@]}"
}

generate_port_list() {
    local base_port="$1"
    local count="$2"
    generate_comma_range "$base_port" "$count"
}

get_selected_model_names() {
    case "${MODEL_TYPE,,}" in
        llama)
            printf '%s\n' "llama"
            ;;
        qwen)
            printf '%s\n' "qwen"
            ;;
        all)
            printf '%s\n%s\n' "llama" "qwen"
            ;;
        *)
            echo "ERROR: unsupported MODEL_TYPE=${MODEL_TYPE}, expected llama/qwen/all"
            return 1
            ;;
    esac
}

resolve_trace_paths() {
    if [[ -n "$TRACE_NAME" ]]; then
        local trace_path="${DATASET_BASE_DIR}/${TRACE_NAME}"
        if [[ ! -d "$trace_path" ]]; then
            echo "ERROR: trace directory not found: $trace_path"
            exit 1
        fi
        TRACE_PATH_ARRAY=("$trace_path")
    elif [[ -n "${TRACE_LIST:-}" ]]; then
        TRACE_PATH_ARRAY=()
        IFS=',' read -ra _trace_names <<< "$TRACE_LIST"
        for _tname in "${_trace_names[@]}"; do
            _tname="$(echo "$_tname" | xargs)"
            [[ -z "$_tname" ]] && continue
            local _tpath="${DATASET_BASE_DIR}/${_tname}"
            if [[ ! -d "$_tpath" ]]; then
                echo "ERROR: trace directory not found: $_tpath"
                exit 1
            fi
            TRACE_PATH_ARRAY+=("$_tpath")
        done
        if [[ ${#TRACE_PATH_ARRAY[@]} -eq 0 ]]; then
            echo "ERROR: no valid trace directories from TRACE_LIST=${TRACE_LIST}"
            exit 1
        fi
    else
        mapfile -d '' -t TRACE_PATH_ARRAY < <(
            find "$DATASET_BASE_DIR" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
        )
        if [[ ${#TRACE_PATH_ARRAY[@]} -eq 0 ]]; then
            echo "ERROR: no trace directories found under: $DATASET_BASE_DIR"
            exit 1
        fi
    fi
}

configure_trace() {
    CURRENT_TRACE_PATH="$1"
    CURRENT_TRACE_NAME="$(basename "$CURRENT_TRACE_PATH")"
}

configure_active_model() {
    local model_name="$1"

    CURRENT_MODEL_NAME="$model_name"
    case "$model_name" in
        llama)
            MODEL="$LLAMA_MODEL_PATH"
            TEST_MODEL="llama"
            ;;
        qwen)
            MODEL="$QWEN_MODEL_PATH"
            TEST_MODEL="qwen"
            ;;
        *)
            echo "ERROR: unsupported model_name=${model_name}"
            return 1
            ;;
    esac

    VLLM_MAX_MODEL_LEN="$USER_VLLM_MAX_MODEL_LEN_OVERRIDE"
    PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS="$USER_PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS_OVERRIDE"
    apply_model_dependent_defaults
    resolve_model_mt_tag_for_name

    CURRENT_BENCH_MAX_TOKENS="$MODEL_MT_TAG_FOR_NAME"
    if [[ -n "$BENCH_MODEL" ]]; then
        CURRENT_BENCH_MODEL="$BENCH_MODEL"
    else
        CURRENT_BENCH_MODEL="$MODEL"
    fi
}

configure_pd_ratio() {
    local pd_ratio="$1"
    CURRENT_PD_RATIO="$pd_ratio"
    CURRENT_PD_RATIO_TAG="$(printf '%s' "$pd_ratio" | sed -E 's/^([0-9]+):([0-9]+)$/\1p\2d/')"

    case "$pd_ratio" in
        1:1)
            PREFILL_GPUS="$PD_1P1D_PREFILL_GPUS"
            PREFILL_PORTS="$PD_1P1D_PREFILL_PORTS"
            PREFILL_KV_PORTS="$PD_1P1D_PREFILL_KV_PORTS"
            DECODE_GPUS="$PD_1P1D_DECODE_GPUS"
            DECODE_PORTS="$PD_1P1D_DECODE_PORTS"
            DECODE_KV_PORTS="$PD_1P1D_DECODE_KV_PORTS"
            ;;
        1:3)
            PREFILL_GPUS="$PD_1P3D_PREFILL_GPUS"
            PREFILL_PORTS="$PD_1P3D_PREFILL_PORTS"
            PREFILL_KV_PORTS="$PD_1P3D_PREFILL_KV_PORTS"
            DECODE_GPUS="$PD_1P3D_DECODE_GPUS"
            DECODE_PORTS="$PD_1P3D_DECODE_PORTS"
            DECODE_KV_PORTS="$PD_1P3D_DECODE_KV_PORTS"
            ;;
        2:6)
            PREFILL_GPUS="$PD_2P6D_PREFILL_GPUS"
            PREFILL_PORTS="$PD_2P6D_PREFILL_PORTS"
            PREFILL_KV_PORTS="$PD_2P6D_PREFILL_KV_PORTS"
            DECODE_GPUS="$PD_2P6D_DECODE_GPUS"
            DECODE_PORTS="$PD_2P6D_DECODE_PORTS"
            DECODE_KV_PORTS="$PD_2P6D_DECODE_KV_PORTS"
            ;;
        4:4)
            PREFILL_GPUS="$PD_4P4D_PREFILL_GPUS"
            PREFILL_PORTS="$PD_4P4D_PREFILL_PORTS"
            PREFILL_KV_PORTS="$PD_4P4D_PREFILL_KV_PORTS"
            DECODE_GPUS="$PD_4P4D_DECODE_GPUS"
            DECODE_PORTS="$PD_4P4D_DECODE_PORTS"
            DECODE_KV_PORTS="$PD_4P4D_DECODE_KV_PORTS"
            ;;
        3:4)
            PREFILL_GPUS="$PD_3P4D_PREFILL_GPUS"
            PREFILL_PORTS="$PD_3P4D_PREFILL_PORTS"
            PREFILL_KV_PORTS="$PD_3P4D_PREFILL_KV_PORTS"
            DECODE_GPUS="$PD_3P4D_DECODE_GPUS"
            DECODE_PORTS="$PD_3P4D_DECODE_PORTS"
            DECODE_KV_PORTS="$PD_3P4D_DECODE_KV_PORTS"
            ;;
        *)
            echo "ERROR: unsupported PD ratio=${pd_ratio}, expected 1:3 / 2:6 / 4:4 / 3:4"
            return 1
            ;;
    esac
}

configure_tpot() {
    local tpot_ms="$1"
    CURRENT_TPOT_MS="$tpot_ms"
    BENCH_GOODPUT="ttft:${BENCH_GOODPUT_TTFT_MS} tpot:${tpot_ms}"
    sync_tpot_between_proxy_and_benchmark
}

resolve_dataset_paths() {
    local query_dir="$CURRENT_TRACE_PATH"
    local legacy_dataset_path=""
    local rr_suffix=""

    if is_true "$COMPARE_SCHEME4_SELECTION_ONLY"; then
        case "$CURRENT_MODEL_NAME" in
            llama)
                legacy_dataset_path="${SCHEME4_SELECTION_DATASET_DIR}/llama-lmsys-chat-scheme4-rrgain-azure-trace-ts.csv"
                ;;
            qwen)
                legacy_dataset_path="${SCHEME4_SELECTION_DATASET_DIR}/qwen-lmsys-chat-scheme4-rrgain-azure-trace-ts.csv"
                ;;
            *)
                echo "ERROR: unsupported model for selected scheme4 dataset: ${CURRENT_MODEL_NAME}"
                exit 1
                ;;
        esac

        if [[ -f "$legacy_dataset_path" ]]; then
            DATASET_PATH_ARRAY=("$legacy_dataset_path")
            return 0
        fi

        mapfile -d '' -t DATASET_PATH_ARRAY < <(
            find "$SCHEME4_SELECTION_DATASET_DIR" -maxdepth 1 \( -type f -o -type l \) \
                -name "${CURRENT_MODEL_NAME}-*-trace-ts.csv" -print0 | sort -z
        )

        if [[ ${#DATASET_PATH_ARRAY[@]} -eq 0 ]]; then
            echo "ERROR: selected comparison dataset not found under: ${SCHEME4_SELECTION_DATASET_DIR}"
            echo "ERROR: expected legacy path: ${legacy_dataset_path}"
            exit 1
        fi

        if [[ ${#DATASET_PATH_ARRAY[@]} -gt 1 ]]; then
            echo "ERROR: multiple selected comparison datasets matched for model=${CURRENT_MODEL_NAME} under: ${SCHEME4_SELECTION_DATASET_DIR}"
            printf '  %s\n' "${DATASET_PATH_ARRAY[@]}"
            exit 1
        fi

        return 0
    fi

    if [[ ! -d "$query_dir" ]]; then
        echo "ERROR: trace dataset directory not found: $query_dir"
        exit 1
    fi

    local model_prefix="${CURRENT_MODEL_NAME}_"
    local csv_pattern="${model_prefix}*.csv"
    # [WT] mooncake trace: only match *_new.csv datasets
    if [[ "$CURRENT_TRACE_NAME" == "mooncake" ]]; then
        csv_pattern="${model_prefix}*_new.csv"
    fi
    mapfile -d '' -t DATASET_PATH_ARRAY < <(
        find "$query_dir" -maxdepth 1 \( -type f -o -type l \) \
            -name "$csv_pattern" -print0 | sort -z
    )
    if [[ ${#DATASET_PATH_ARRAY[@]} -eq 0 ]]; then
        echo "ERROR: no matching CSV datasets found under: $query_dir for model=${CURRENT_MODEL_NAME}"
        exit 1
    fi
}

log_failed_config() {
    local reason="$1"
    mkdir -p "$BASE_RESULT_DIR"
    if [[ -z "$FAILED_CONFIG_LOG" ]]; then
        FAILED_CONFIG_LOG="${BASE_RESULT_DIR}/failed_configs_${RUN_BATCH_TIMESTAMP}.log"
    fi
    printf '%s | model=%s | dataset=%s | pd=%s | tpot=%sms | reason=%s\n' \
        "$(date --iso-8601=seconds)" \
        "$CURRENT_MODEL_NAME" \
        "$CURRENT_BENCH_DATASET_PATH" \
        "$CURRENT_PD_RATIO" \
        "$CURRENT_TPOT_MS" \
        "$reason" >> "$FAILED_CONFIG_LOG"
}

csv_escape() {
    local value="${1:-}"
    value="${value//\"/\"\"}"
    printf '"%s"' "$value"
}

init_summary_csv() {
    if [[ -z "$SUMMARY_CSV" ]]; then
        SUMMARY_CSV="${BASE_RESULT_DIR}/summary_${RUN_BATCH_TIMESTAMP}.csv"
    fi
    if [[ ! -f "$SUMMARY_CSV" ]]; then
        printf '%s\n' 'timestamp,status,model,dataset_path,pd_ratio,tpot_ms,repeat_idx,result_dir,failure_reason,compare_run_method,result_suffix,experiment_mode,proxy_script' > "$SUMMARY_CSV"
    fi
}

append_summary_csv() {
    local status="$1"
    local repeat_idx="$2"
    local result_dir="$3"
    local reason="$4"
    init_summary_csv
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "$(csv_escape "$(date --iso-8601=seconds)")" \
        "$(csv_escape "$status")" \
        "$(csv_escape "$CURRENT_MODEL_NAME")" \
        "$(csv_escape "$CURRENT_BENCH_DATASET_PATH")" \
        "$(csv_escape "$CURRENT_PD_RATIO")" \
        "$(csv_escape "${CURRENT_TPOT_MS}")" \
        "$(csv_escape "$repeat_idx")" \
        "$(csv_escape "$result_dir")" \
        "$(csv_escape "$reason")" \
        "$(csv_escape "$CURRENT_RUN_METHOD")" \
        "$(csv_escape "$CURRENT_RUN_SUFFIX")" \
        "$(csv_escape "$CURRENT_EXPERIMENT_MODE")" \
        "$(csv_escape "$PROXY_SCRIPT")" >> "$SUMMARY_CSV"
}

benchmark_dir_key_from_path() {
    local path="$1"
    local base_name

    path="${path%/}"
    base_name="$(basename "$path")"
    # [WT] Strip from _YYYYMMDD_HHMMSS onward (including any trailing method suffix like _roundrobin)
    printf '%s' "$(printf '%s' "$base_name" | sed -E 's/_[0-9]{8}_[0-9]{6}.*$//')"
}

current_benchmark_dir_key() {
    local repeat_idx="$1"
    printf '%s' "${CURRENT_MODEL_NAME}_${CURRENT_BENCH_DATASET_TAG}_n${CURRENT_DATASET_SIZE}_pd${CURRENT_PD_RATIO_TAG}_tpot${CURRENT_TPOT_MS}ms_trace${CURRENT_TRACE_NAME}_rep${repeat_idx}"
}

init_skip_result_dir_keys() {
    local entry skip_key

    if [[ -z "${SKIP_EXISTING_RESULT_DIRS//[$' \t\r\n,']/}" ]]; then
        return
    fi

    while IFS= read -r entry; do
        entry="$(echo "$entry" | xargs)"
        [[ -z "$entry" ]] && continue
        skip_key="$(benchmark_dir_key_from_path "$entry")"
        [[ -z "$skip_key" ]] && continue
        SKIP_RESULT_DIR_KEYS["$skip_key"]=1
    done < <(printf '%s' "$SKIP_EXISTING_RESULT_DIRS" | tr ',' '\n')
}

apply_temp_skip_existing_results_once() {
    if [[ -n "${SKIP_EXISTING_RESULT_DIRS//[$' \t\r\n,']/}" ]]; then
        return
    fi

    if [[ -z "${WT_TEMP_SKIP_EXISTING_RESULT_DIRS_ONCE//[$' \t\r\n,']/}" ]]; then
        return
    fi

    if [[ -f "$WT_TEMP_SKIP_EXISTING_RESULT_DIRS_ONCE_MARKER" ]]; then
        return
    fi

    SKIP_EXISTING_RESULT_DIRS="$WT_TEMP_SKIP_EXISTING_RESULT_DIRS_ONCE"
    mkdir -p "$(dirname "$WT_TEMP_SKIP_EXISTING_RESULT_DIRS_ONCE_MARKER")"
    : > "$WT_TEMP_SKIP_EXISTING_RESULT_DIRS_ONCE_MARKER"
    echo "INFO: applying one-time temporary skip list from experiment_result_2 successes"
}

should_skip_existing_result() {
    local repeat_idx="$1"
    local skip_key

    skip_key="$(current_benchmark_dir_key "$repeat_idx")"
    [[ -n "${SKIP_RESULT_DIR_KEYS[$skip_key]:-}" ]]
}

check_selected_gpus_idle_once() { return 0; }  # MODIFIED: skip GPU idle check
_original_check_selected_gpus_idle_once() {
    local all_gpu_ids=()
    local gpu_index memory_used gpu_util
    local status_output
    local selected_gpu_lookup="," 

    LAST_GPU_IDLE_FAILURE_REASON=""

    IFS=',' read -ra _prefill_gpu_array <<< "$PREFILL_GPUS"
    IFS=',' read -ra _decode_gpu_array <<< "$DECODE_GPUS"
    all_gpu_ids=("${_prefill_gpu_array[@]}" "${_decode_gpu_array[@]}")

    for gpu_index in "${all_gpu_ids[@]}"; do
        selected_gpu_lookup+="${gpu_index},"
    done

    status_output=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)
    while IFS=',' read -r gpu_index memory_used gpu_util; do
        gpu_index="$(echo "$gpu_index" | xargs)"
        memory_used="$(echo "$memory_used" | xargs)"
        gpu_util="$(echo "$gpu_util" | xargs)"

        if [[ "$selected_gpu_lookup" != *",${gpu_index},"* ]]; then
            continue
        fi

        if [[ "$memory_used" != "0" || "$gpu_util" != "0" ]]; then
            echo "GPU busy: gpu=${gpu_index}, memory_used_mib=${memory_used}, util_percent=${gpu_util}"
            LAST_GPU_IDLE_FAILURE_REASON="gpu_not_idle(gpu=${gpu_index},memory=${memory_used},util=${gpu_util})"
            return 1
        fi
    done <<< "$status_output"

    return 0
}

check_selected_gpus_idle() { return 0; }  # MODIFIED: skip GPU idle wait
_original_check_selected_gpus_idle() {
    local wait_seconds="${1:-$GPU_IDLE_WAIT_SECONDS}"
    local poll_interval="${2:-$GPU_IDLE_POLL_INTERVAL_SECONDS}"
    local stable_samples_required="${3:-$GPU_IDLE_STABLE_SAMPLES}"
    local start_time now elapsed stable_samples

    start_time=$(date +%s)
    stable_samples=0

    while true; do
        if check_selected_gpus_idle_once; then
            stable_samples=$((stable_samples + 1))
            if (( stable_samples >= stable_samples_required )); then
                return 0
            fi
        else
            stable_samples=0
        fi

        now=$(date +%s)
        elapsed=$((now - start_time))
        if (( elapsed >= wait_seconds )); then
            if [[ -n "$LAST_GPU_IDLE_FAILURE_REASON" ]]; then
                log_failed_config "$LAST_GPU_IDLE_FAILURE_REASON"
            fi
            return 1
        fi

        echo "Waiting for selected GPUs to become idle (${elapsed}s/${wait_seconds}s, stable=${stable_samples}/${stable_samples_required})"
        sleep "$poll_interval"
    done
}

sync_tpot_between_proxy_and_benchmark() {
    local tpot_from_bench=""
    local bench_tokens="${BENCH_GOODPUT//,/ }"
    local item

    for item in $bench_tokens; do
        if [[ "$item" == tpot:* ]]; then
            tpot_from_bench="${item#tpot:}"
            break
        fi
    done

    if [[ -z "$tpot_from_bench" ]]; then
        echo "ERROR: BENCH_GOODPUT must include 'tpot:<value>', current: ${BENCH_GOODPUT}"
        exit 1
    fi

    local synced_tpot="tpot:${tpot_from_bench}"
    if [[ "$CACULATE_GOODPUT" != "$synced_tpot" ]]; then
        echo "Sync CACULATE_GOODPUT to BENCH_GOODPUT tpot: ${CACULATE_GOODPUT} -> ${synced_tpot}"
    fi
    CACULATE_GOODPUT="$synced_tpot"
}

array_to_json() {
    local s="$1"
    if [[ -z "$s" ]]; then
        printf '[]'
        return
    fi
    IFS=',' read -ra _arr <<< "$s"
    printf '['
    local first=1
    for v in "${_arr[@]}"; do
        v="$(echo "$v" | sed -e 's/^\s*//' -e 's/\s*$//')"
        if [[ $first -eq 1 ]]; then
            printf '"%s"' "$v"
            first=0
        else
            printf ',"%s"' "$v"
        fi
    done
    printf ']'
}

count_csv_records() {
    local dataset_path="$1"
    python3 - "$dataset_path" <<'PY'
import csv
import sys

path = sys.argv[1]
try:
    csv.field_size_limit(sys.maxsize)
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = sum(1 for _ in csv.reader(f))
    print(max(rows - 1, 0))
except Exception:
    print("na")
PY
}

set_current_dataset() {
    local dataset_path="$1"
    local dataset_file_name
    local dataset_name_no_ext
    local dataset_name_norm
    local parsed_dataset_size
    local filename_dataset_size
    dataset_file_name="$(basename "$dataset_path")"
    dataset_name_no_ext="${dataset_file_name%.csv}"
    dataset_name_norm="$(echo "$dataset_name_no_ext" | tr '[:upper:]' '[:lower:]' | sed 's/_/-/g')"

    CURRENT_BENCH_DATASET_PATH="$dataset_path"
    CURRENT_BENCH_DATASET_LABEL="$(sanitize_name "$dataset_name_no_ext")"

    case "$dataset_name_norm" in
        *lmsys-chat*|*lmsys*)
            CURRENT_BENCH_DATASET_TAG="${CURRENT_MODEL_NAME}-lmsys-chat"
            ;;
        *mysharegpt*|*sharegpt*)
            CURRENT_BENCH_DATASET_TAG="${CURRENT_MODEL_NAME}-mysharegpt"
            ;;
        *)
            CURRENT_BENCH_DATASET_TAG="${CURRENT_MODEL_NAME}-$(echo "$dataset_name_norm" | sed 's/[^a-z0-9-]/-/g')"
            ;;
    esac

    # Include trace name to keep result directories unique across traces.
    if [[ -n "$CURRENT_TRACE_NAME" ]]; then
        CURRENT_BENCH_DATASET_TAG="${CURRENT_BENCH_DATASET_TAG}-${CURRENT_TRACE_NAME}"
    fi

    filename_dataset_size=""
    if [[ "$dataset_file_name" =~ (^|[_-])n([0-9]+)([_-]|\.) ]]; then
        filename_dataset_size="${BASH_REMATCH[2]}"
    fi

    # Prefer real CSV record count (header excluded), not raw line count.
    parsed_dataset_size="$(count_csv_records "$dataset_path")"
    if [[ "$parsed_dataset_size" =~ ^[0-9]+$ ]]; then
        CURRENT_DATASET_SIZE="$parsed_dataset_size"
    elif [[ -n "$filename_dataset_size" ]]; then
        CURRENT_DATASET_SIZE="$filename_dataset_size"
    else
        CURRENT_DATASET_SIZE="na"
    fi
}

setup_directories() {
    local timestamp="$1"
    local repeat_idx="$2"
    local run_suffix_part=""

    if [[ -n "$CURRENT_RUN_SUFFIX" ]]; then
        run_suffix_part="_${CURRENT_RUN_SUFFIX}"
    fi

    BENCHMARK_DIR="${BASE_RESULT_DIR}/${CURRENT_MODEL_NAME}_${CURRENT_BENCH_DATASET_TAG}_n${CURRENT_DATASET_SIZE}_pd${CURRENT_PD_RATIO_TAG}_tpot${CURRENT_TPOT_MS}ms_trace${CURRENT_TRACE_NAME}_rep${repeat_idx}_${timestamp}${run_suffix_part}"
    CONFIG_DIR="${BENCHMARK_DIR}/config"
    LOG_DIR="${BENCHMARK_DIR}/log"
    RESULT_DIR="${BENCHMARK_DIR}/dataset_result"

    mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$RESULT_DIR"
}

check_required_files() {
    update_proxy_script
    if [[ ! -f "$PROXY_SCRIPT" ]]; then
        echo "ERROR: proxy script not found: $PROXY_SCRIPT"
        exit 1
    fi
    if [[ ! -f "$BENCH_SCRIPT" ]]; then
        echo "ERROR: benchmark script not found: $BENCH_SCRIPT"
        exit 1
    fi
    if [[ ! -d "$DATASET_BASE_DIR" ]]; then
        echo "ERROR: dataset base dir not found: $DATASET_BASE_DIR"
        exit 1
    fi
}

check_num_gpus() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi not found"
        exit 1
    fi
    local num_gpus
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [[ "$num_gpus" -lt 4 ]]; then
        echo "ERROR: need at least 4 GPUs for configured PD ratios, found $num_gpus"
        exit 1
    fi
}

ensure_python_library_installed() {
    if ! python3 -c "import $1" >/dev/null 2>&1; then
        echo "ERROR: python library '$1' is not installed"
        exit 1
    fi
}

# Effective scheme4 reservation size.  When WT_SCHEME4_RESERVATION_TOKENS is
# unset, both the proxy (from MODEL/config.json) and GeometricWindowScheduler (from
# hf_config) default to max_position_embeddings // 2, so resolve the real number
# here.  Recording a hard-coded 4096 was only correct for an 8K-context model
# and silently wrong for e.g. Qwen (32K -> 16384).
resolve_effective_reservation_tokens() {
    if [[ -n "${WT_SCHEME4_RESERVATION_TOKENS:-}" ]]; then
        printf '%s' "$WT_SCHEME4_RESERVATION_TOKENS"
        return
    fi
    local cfg="${MODEL}/config.json"
    local resolved=""
    if [[ -f "$cfg" ]]; then
        resolved="$(python3 - "$cfg" <<'PY' 2>/dev/null || true
import json
import sys
try:
    with open(sys.argv[1], "r") as fh:
        raw = json.load(fh)
    print(max(1, int(raw.get("max_position_embeddings", 8192)) // 2))
except Exception:
    pass
PY
)"
    fi
    if [[ -z "$resolved" ]]; then
        resolved="model_max_context_half"
    fi
    printf '%s' "$resolved"
}

dump_config_json() {
        local repeat_idx="$1"
        local ts="$2"
    local cfgfile="${CONFIG_DIR}/config_${ts}.json"
    local effective_reservation_tokens
    effective_reservation_tokens="$(resolve_effective_reservation_tokens)"

    cat > "$cfgfile" <<EOF
{
  "generated_at": "$(date --iso-8601=seconds)",
    "selected_model_type": "${MODEL_TYPE}",
    "current_model_name": "${CURRENT_MODEL_NAME}",
                "compare_run_method": "${CURRENT_RUN_METHOD}",
                "result_suffix": "${CURRENT_RUN_SUFFIX}",
        "experiment_mode": "${CURRENT_EXPERIMENT_MODE}",
        "proxy_script": "${PROXY_SCRIPT}",
  "model": "${MODEL}",
  "test_model": "${TEST_MODEL}",
  "dataset_path": "${CURRENT_BENCH_DATASET_PATH}",
  "dataset_label": "${CURRENT_BENCH_DATASET_LABEL}",
    "pd_ratio": "${CURRENT_PD_RATIO}",
    "tpot_ms": ${CURRENT_TPOT_MS},
  "timeout_seconds": ${TIMEOUT_SECONDS},
  "proxy_port": ${PROXY_PORT},
  "bench_port": ${BENCH_PORT},
  "prefill_gpus": $(array_to_json "$PREFILL_GPUS"),
  "decode_gpus": $(array_to_json "$DECODE_GPUS"),
  "prefill_ports": $(array_to_json "$PREFILL_PORTS"),
  "decode_ports": $(array_to_json "$DECODE_PORTS"),
  "prefill_kv_ports": $(array_to_json "$PREFILL_KV_PORTS"),
  "decode_kv_ports": $(array_to_json "$DECODE_KV_PORTS"),
  "prefill_gpu_memory_utilization": ${PREFILL_GPU_MEMORY_UTILIZATION},
  "decode_gpu_memory_utilization": ${DECODE_GPU_MEMORY_UTILIZATION},
  "kv_transfer_config": {
    "kv_connector": "${KV_CONNECTOR}",
    "producer_buffer": "${KV_PRODUCER_BUFFER}",
    "consumer_buffer": "${KV_CONSUMER_BUFFER}",
    "nccl_num_channels": ${KV_NCCL_CHANNELS},
    "send_type": "${KV_SEND_TYPE}",
    "prefill_tensor_pool_memory": ${PREFILL_TENSOR_POOL_MEMORY},
    "decode_tensor_pool_memory": ${DECODE_TENSOR_POOL_MEMORY}
  },
    "predictor_route": {
        "scheme": "${WT_PREDICT_ROUTE_SCHEME}",
        "request_tokens": "prompt_len + predict_output_median_or_mean",
        "score": "pred_load_bottle",
        "scheme3_shortlist": "top_half_by_pred_load_bottle",
        "scheme3_select": "min_load_bottle_in_shortlist",
        "scheme4_fastpath": "reservation_admission_then_normal_kv_send",
        "fastpath_overload_gate": "measured_TPOT >= tpot_budget (no hand-tuned threshold)",
        "predict_output_scale": ${WT_PREDICT_OUTPUT_SCALE},
        "scheme4_reservation_tokens": "${effective_reservation_tokens}",
        "scheme4_load_threshold": ${WT_SCHEME4_LOAD_THRESHOLD},
        "scheme5_reroute": {
            "enabled": $( [[ "${WT_PREDICT_ROUTE_SCHEME}" == "scheme5" ]] && echo true || echo false ),
            "max_reroutes": ${WT_SCHEME5_MAX_REROUTES},
            "first_chunk_timeout_ms": "${WT_SCHEME5_FIRST_CHUNK_TIMEOUT_MS}",
            "timeout_ratio": ${WT_SCHEME5_TIMEOUT_RATIO},
            "exclude_previous_decode": "${WT_SCHEME5_EXCLUDE_PREVIOUS_DECODE}",
            "trigger": "decode_first_chunk_timeout",
            "retry_policy": "one_extra_prefill_then_decode_on_new_instance"
        },
        "decode_metrics": "compute_memory_capacity_bottle",
        "immediate_update": "future_tokens_and_pred_load"
    },
    "predictor_timing": {
        "timing_csv_pattern": "${LOG_DIR}/predict_timing_prefill{idx}_${ts}.csv",
        "verbose": "${WT_TIMING_VERBOSE}",
        "pending_check_ms": ${WT_TIMING_PENDING_CHECK_MS}
    },
    "predictor_analysis": {
        "enabled": "${WT_PREDICT_ANALYSIS_ENABLE}",
        "csv_pattern": "${LOG_DIR}/predict_request_decode{idx}_${ts}.csv"
    },
    "logging": {
        "predict_chain_verbose": "${WT_PREDICT_CHAIN_VERBOSE}",
        "route_decision_log_enabled": "${WT_ROUTE_DECISION_LOG_ENABLED}",
        "monitor_log_interval_ms": ${WT_MONITOR_LOG_INTERVAL_MS},
        "timing_verbose": "${WT_TIMING_VERBOSE}",
        "kv_flush_verbose": "${WT_KV_FLUSH_VERBOSE}"
    },
    "kv_deferred_flush": {
        "log_interval_ms": ${WT_KV_FLUSH_LOG_INTERVAL_MS},
        "sleep_ms": ${WT_KV_FLUSH_SLEEP_MS},
        "verbose": "${WT_KV_FLUSH_VERBOSE}",
        "layer_pack_enable": "${WT_KV_LAYER_PACK_ENABLE}",
        "route_aware_immediate_send": "${WT_KV_ROUTE_AWARE_IMMEDIATE_SEND}",
        "layer_pack_budget": "kv_buffer_size"
    },
    "decode_scheduler": {
        "use_gwf_scheduler": "${USE_GWF_SCHEDULER}",
        "use_pastfuture_scheduler": "${USE_PASTFUTURE_SCHEDULER}",
        "activation_predict": "${USE_ACTIVATION_PREDICTOR}",
        "predicted_remaining_default_path": "stats/prometheus/proxy_load_only",
        "gwf_path": "two_regime_greedy",
        "gwf_policy": {
            "admission": "GWF on measured TPOT: halve window when measured_tpot > budget, else +1 (self-calibrating)",
            "ttft_budget_ms": ${BENCH_GOODPUT_TTFT_MS},
            "tpot_budget_ms": ${CURRENT_TPOT_MS},
            "eviction": "hopeless class first, then gen_ratio - slack - 0.5*rem_ratio",
            "hopeless_policy": "past TTFT/TPOT -> lower eviction class, demoted in admission, served last",
            "hopeless_max_defer_ms": ${WT_GWF_HOPELESS_MAX_DEFER_MS},
            "force_idle": ${WT_GWF_FORCE_IDLE},
            "log_interval_ms": ${WT_GWF_LOG_INTERVAL_MS}
        }
    },
  "vllm_options": {
    "enforce_eager": $( [[ "$VLLM_ENFORCE_EAGER" -eq 1 ]] && echo true || echo false ),
    "seed": ${VLLM_SEED},
    "dtype": "${VLLM_DTYPE}",
    "max_model_len": ${VLLM_MAX_MODEL_LEN},
    "prefill_max_num_batched_tokens": ${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS},
    "decode_max_num_batched_tokens": ${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS},
    "max_num_seqs": ${VLLM_MAX_NUM_SEQS}
  },
  "benchmark": {
    "bench_script": "${BENCH_SCRIPT}",
        "bench_model": "${CURRENT_BENCH_MODEL}",
    "backend": "${BENCH_BACKEND}",
    "endpoint": "${BENCH_ENDPOINT}",
    "dataset_name": "${BENCH_DATASET_NAME}",
        "request_rate": "trace",
        "maxtokenscustom": ${CURRENT_BENCH_MAX_TOKENS},
    "repeat_idx": ${repeat_idx},
    "save_output": "${SAVE_OUTPUT}",
    "save_sample": "${SAVE_SAMPLE}",
    "max_concurrency": ${BENCH_MAX_CONCURRENCY},
    "temperature": ${BENCH_TEMPERATURE},
    "top_p": ${BENCH_TOP_P},
    "top_k": ${BENCH_TOP_K},
    "repetition_penalty": ${BENCH_REPETITION_PENALTY},
    "goodput": "${BENCH_GOODPUT}"
  }
}
EOF
}

wait_for_server() {
    local port="$1"
    local start_time
    start_time=$(date +%s)

    echo "Waiting for server on port $port ..."
    while true; do
        if curl -s "localhost:${port}" >/dev/null 2>&1; then
            return 0
        fi
        local now
        now=$(date +%s)
        if (( now - start_time >= TIMEOUT_SECONDS )); then
            echo "ERROR: timeout waiting for server on port $port"
            return 1
        fi
        sleep 1
    done
}

wait_for_port_free() {
    local port="$1"
    local timeout="${2:-10}"
    local start
    start=$(date +%s)

    while true; do
        if ! ss -ltn "( sport = :$port )" 2>/dev/null | tail -n +2 | grep -q .; then
            return 0
        fi
        local now
        now=$(date +%s)
        if (( now - start >= timeout )); then
            return 1
        fi
        sleep 0.5
    done
}

stop_servers() {
    if [[ ${#PGIDS[@]} -eq 0 && ${#PIDS[@]} -eq 0 ]]; then
        return
    fi

    echo "Stopping servers..."
    for pg in "${PGIDS[@]}"; do
        [[ -n "$pg" ]] && kill -TERM -"$pg" 2>/dev/null || true
    done

    sleep 3

    for pg in "${PGIDS[@]}"; do
        pgrep -g "$pg" >/dev/null 2>&1 && kill -KILL -"$pg" 2>/dev/null || true
    done

    pkill -9 -f "$PROXY_SCRIPT" >/dev/null 2>&1 || true
    pkill -9 -f "vllm serve" >/dev/null 2>&1 || true
    fuser -k ${PROXY_PORT}/tcp 2>/dev/null || true  # kill stale proxy by port
    fuser -k ${BENCH_PORT}/tcp 2>/dev/null || true

    PIDS=()
    PGIDS=()
    PREFILL_PIDS=()
    DECODE_PIDS=()
}

cleanup() {
    if [[ $_CLEANED_UP -eq 1 ]]; then
        return
    fi
    _CLEANED_UP=1

    stop_servers

    local ports_to_check=()
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
    ports_to_check+=("${PROXY_PORT}")
    for p in "${PREFILL_PORT_ARRAY[@]}"; do ports_to_check+=("$p"); done
    for p in "${DECODE_PORT_ARRAY[@]}"; do ports_to_check+=("$p"); done
    for port in "${ports_to_check[@]}"; do
        wait_for_port_free "$port" 10 || true
    done
}

plot_decode_load() {
    local ts="$1"
    local proxy_log="${LOG_DIR}/proxy_${ts}.log"
    [[ -f "$PLOT_SCRIPT" ]] || return 0
    [[ -f "$proxy_log" ]] || return 0

    python3 "$PLOT_SCRIPT" --log_path "$proxy_log" > "${LOG_DIR}/plot_${ts}.log" 2>&1 || true
}

start_servers() {
    local timestamp="$1"
    update_proxy_script
    PIDS=()
    PGIDS=()
    PREFILL_PIDS=()
    DECODE_PIDS=()

    echo "Launching servers..."

    setsid env \
        PYTHONPATH="${PROXY_BASE_DIR}:${PYTHONPATH:-}" \
        PROXY_PORT="${PROXY_PORT}" \
        BENCH_PORT="${BENCH_PORT}" \
        VLLM_DTYPE="${VLLM_DTYPE}" \
        MODEL_CONFIG_PATH="${MODEL}/config.json" \
        TPOT="${CACULATE_GOODPUT}" \
        WT_PREDICT_ROUTE_SCHEME="${WT_PREDICT_ROUTE_SCHEME}" \
        WT_SCHEME4_LOAD_THRESHOLD="${WT_SCHEME4_LOAD_THRESHOLD}" \
        WT_SCHEME4_RESERVATION_TOKENS="${WT_SCHEME4_RESERVATION_TOKENS}" \
        WT_PREDICT_OUTPUT_SCALE="${WT_PREDICT_OUTPUT_SCALE}" \
        WT_SCHEME5_MAX_REROUTES="${WT_SCHEME5_MAX_REROUTES}" \
        WT_SCHEME5_TIMEOUT_RATIO="${WT_SCHEME5_TIMEOUT_RATIO}" \
        WT_SCHEME5_FIRST_CHUNK_TIMEOUT_MS="${WT_SCHEME5_FIRST_CHUNK_TIMEOUT_MS}" \
        WT_SCHEME5_EXCLUDE_PREVIOUS_DECODE="${WT_SCHEME5_EXCLUDE_PREVIOUS_DECODE}" \
        RRPP_KV_THRESHOLD="${RRPP_KV_THRESHOLD}" \
        RRPP_POLL_INTERVAL_MS="${RRPP_POLL_INTERVAL_MS}" \
        WT_PREDICT_CHAIN_VERBOSE="${WT_PREDICT_CHAIN_VERBOSE}" \
        WT_ROUTE_DECISION_LOG_ENABLED="${WT_ROUTE_DECISION_LOG_ENABLED}" \
        WT_MONITOR_LOG_INTERVAL_MS="${WT_MONITOR_LOG_INTERVAL_MS}" \
        WT_ORACLE_DATASET_PATH="${CURRENT_BENCH_DATASET_PATH}" \
        bash -c "exec python3 \"$PROXY_SCRIPT\"" \
        &> "${LOG_DIR}/proxy_${timestamp}.log" &

    local proxy_pid=$!
    local proxy_pgid
    proxy_pgid=$(ps -o pgid= -p "$proxy_pid" | tr -d ' ')
    PIDS+=("$proxy_pid")
    PGIDS+=("$proxy_pgid")

    # Verify proxy is alive after 5s
    sleep 5
    if ! ps -p "$proxy_pid" >/dev/null 2>&1; then
        echo "ERROR: proxy died immediately, check ${LOG_DIR}/proxy_${timestamp}.log"
        stop_servers
        return 1
    fi

    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra PREFILL_KV_PORT_ARRAY <<< "$PREFILL_KV_PORTS"

    for i in "${!PREFILL_GPU_ARRAY[@]}"; do
        local gpu_id=${PREFILL_GPU_ARRAY[$i]}
        local port=${PREFILL_PORT_ARRAY[$i]:-$((20002 + i))}
        local kv_port=${PREFILL_KV_PORT_ARRAY[$i]:-$((21001 + i))}
        local dist_port=$((30000 + port))

        local cmd
        cmd="vllm serve \"$MODEL\" \
            --enforce-eager \
            --host 0.0.0.0 \
            --port \"$port\" \
            --tensor-parallel-size ${PREFILL_TENSOR_PARALLEL_SIZE} \
            --seed ${VLLM_SEED} \
            --dtype ${VLLM_DTYPE} \
            --max-model-len ${VLLM_MAX_MODEL_LEN} \
            --max-num-batched-tokens ${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS} \
            --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
            --test-model ${TEST_MODEL} \
            --gpu-memory-utilization ${PREFILL_GPU_MEMORY_UTILIZATION}"

        if is_true "$USE_ACTIVATION_PREDICTOR" \
            || { [[ "$WT_PREDICT_ROUTE_SCHEME" != "oracle" ]] && is_true "$TEST_ABLATION_P2D"; } \
            || { [[ "$WT_PREDICT_ROUTE_SCHEME" != "oracle" ]] && is_true "$USE_CUSTOM_PROXY"; }; then
            cmd="$cmd --activation-predict"
        fi

        local kv_json_cfg='{'
        kv_json_cfg+='"kv_connector":"'"${KV_CONNECTOR}"'",'
        kv_json_cfg+='"kv_role":"kv_producer",'
        kv_json_cfg+='"kv_buffer_size":"'"${KV_PRODUCER_BUFFER}"'",'
        kv_json_cfg+='"kv_port":"'"$kv_port"'",'
        kv_json_cfg+='"kv_connector_extra_config":{'
        kv_json_cfg+='"proxy_ip":"0.0.0.0",'
        kv_json_cfg+='"proxy_port":"'"${PROXY_PORT}"'",'
        kv_json_cfg+='"http_port":"'"$port"'",'
        kv_json_cfg+='"send_type":"'"${KV_SEND_TYPE}"'",'
        kv_json_cfg+='"mem_pool_size_gb":"'"${PREFILL_TENSOR_POOL_MEMORY}"'",'
        kv_json_cfg+='"nccl_num_channels":"'"${KV_NCCL_CHANNELS}"'"'
        kv_json_cfg+='}}'

        cmd="$cmd --kv-transfer-config '$kv_json_cfg'"

        setsid env \
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            MASTER_ADDR="127.0.0.1" \
            MASTER_PORT="$dist_port" \
            VLLM_USE_V1=1 \
            WT_PREDICT_CHAIN_VERBOSE="${WT_PREDICT_CHAIN_VERBOSE}" \
            WT_PREDICT_TIMING_CSV="${LOG_DIR}/predict_timing_prefill${i}_${timestamp}.csv" \
            WT_TIMING_VERBOSE="${WT_TIMING_VERBOSE}" \
            WT_TIMING_PENDING_CHECK_MS="${WT_TIMING_PENDING_CHECK_MS}" \
            WT_KV_FLUSH_LOG_INTERVAL_MS="${WT_KV_FLUSH_LOG_INTERVAL_MS}" \
            WT_KV_FLUSH_SLEEP_MS="${WT_KV_FLUSH_SLEEP_MS}" \
            WT_KV_FLUSH_VERBOSE="${WT_KV_FLUSH_VERBOSE}" \
            WT_KV_LAYER_PACK_ENABLE="${WT_KV_LAYER_PACK_ENABLE}" \
            WT_KV_ROUTE_AWARE_IMMEDIATE_SEND="${WT_KV_ROUTE_AWARE_IMMEDIATE_SEND}" \
            bash -c "exec $cmd" > "${LOG_DIR}/prefill${i}_${timestamp}.log" 2>&1 &
        local pid=$!
        local pgid
        pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
        PIDS+=("$pid")
        PGIDS+=("$pgid")
        PREFILL_PIDS+=("$pid")
    done

    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
    IFS=',' read -ra DECODE_KV_PORT_ARRAY <<< "$DECODE_KV_PORTS"

    for i in "${!DECODE_GPU_ARRAY[@]}"; do
        local gpu_id=${DECODE_GPU_ARRAY[$i]}
        local port=${DECODE_PORT_ARRAY[$i]:-$((20003 + i))}
        local kv_port=${DECODE_KV_PORT_ARRAY[$i]:-$((22001 + i))}
        local dist_port=$((40000 + port))

        local cmd
        cmd="vllm serve \"$MODEL\" \
            --enforce-eager \
            --host 0.0.0.0 \
            --port \"$port\" \
            --tensor-parallel-size ${DECODE_TENSOR_PARALLEL_SIZE} \
            --seed ${VLLM_SEED} \
            --dtype ${VLLM_DTYPE} \
            --max-model-len ${VLLM_MAX_MODEL_LEN} \
            --max-num-batched-tokens ${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS} \
            --max-num-seqs ${VLLM_MAX_NUM_SEQS} \
            --gpu-memory-utilization ${DECODE_GPU_MEMORY_UTILIZATION} \
            --test-model ${TEST_MODEL} \
            --swap-space 0"

        if is_true "$USE_PASTFUTURE_SCHEDULER"; then
            cmd="$cmd --pastfuture-scheduler"
        fi
        if is_true "$USE_GWF_SCHEDULER"; then
            cmd="$cmd --geometric-window-scheduler"
        fi
        if is_true "$TEST_ABLATION_P2D"; then
            cmd="$cmd --test-p2d"
        fi
        if is_true "$TEST_OPTIMAL"; then
            cmd="$cmd --test-optimal"
        fi

        local kv_json_cfg='{'
        kv_json_cfg+='"kv_connector":"'"${KV_CONNECTOR}"'",'
        kv_json_cfg+='"kv_role":"kv_consumer",'
        kv_json_cfg+='"kv_buffer_size":"'"${KV_CONSUMER_BUFFER}"'",'
        kv_json_cfg+='"kv_port":"'"$kv_port"'",'
        kv_json_cfg+='"kv_connector_extra_config":{'
        kv_json_cfg+='"proxy_ip":"0.0.0.0",'
        kv_json_cfg+='"proxy_port":"'"${PROXY_PORT}"'",'
        kv_json_cfg+='"http_port":"'"$port"'",'
        kv_json_cfg+='"send_type":"'"${KV_SEND_TYPE}"'",'
        kv_json_cfg+='"mem_pool_size_gb":"'"${DECODE_TENSOR_POOL_MEMORY}"'",'
        kv_json_cfg+='"nccl_num_channels":"'"${KV_NCCL_CHANNELS}"'"'
        kv_json_cfg+='}}'

        cmd="$cmd --kv-transfer-config '$kv_json_cfg'"

        local predict_analysis_csv=""
        if [[ "${WT_PREDICT_ANALYSIS_ENABLE}" == "1" ]]; then
            predict_analysis_csv="${LOG_DIR}/predict_request_decode${i}_${timestamp}.csv"
        fi
        setsid env \
            DECODE_INSTANCE_ID="decode-${i}" \
            WT_PREDICT_CHAIN_VERBOSE="${WT_PREDICT_CHAIN_VERBOSE}" \
            WT_PREDICT_ANALYSIS_ENABLE="${WT_PREDICT_ANALYSIS_ENABLE}" \
            WT_PREDICT_ANALYSIS_CSV="${predict_analysis_csv}" \
            WT_GWF_TTFT_BUDGET_MS="${BENCH_GOODPUT_TTFT_MS}" \
            WT_GWF_TPOT_BUDGET_MS="${CURRENT_TPOT_MS}" \
            WT_SCHEME4_RESERVATION_TOKENS="${WT_SCHEME4_RESERVATION_TOKENS}" \
            WT_GWF_HOPELESS_MAX_DEFER_MS="${WT_GWF_HOPELESS_MAX_DEFER_MS}" \
            WT_GWF_FORCE_IDLE="${WT_GWF_FORCE_IDLE}" \
            WT_GWF_LOG_INTERVAL_MS="${WT_GWF_LOG_INTERVAL_MS}" \
            VLLM_USE_V1=1 \
            CUDA_VISIBLE_DEVICES="$gpu_id" \
            MASTER_ADDR="127.0.0.1" \
            MASTER_PORT="$dist_port" \
            bash -c "exec $cmd" > "${LOG_DIR}/decode${i}_${timestamp}.log" 2>&1 &
        local pid=$!
        local pgid
        pgid=$(ps -o pgid= -p "$pid" | tr -d ' ')
        PIDS+=("$pid")
        PGIDS+=("$pgid")
        DECODE_PIDS+=("$pid")
    done

    echo "Waiting for servers..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server "$port"; then
            echo "ERROR: failed to start server on port $port"
            stop_servers
            return 1
        fi
    done
    return 0
}
run_benchmark_with_health_guard() {
    local client_pid="$1"
    local timeout_seconds="${INSTANCE_DOWN_TIMEOUT_SECONDS}"
    local check_interval="${INSTANCE_HEALTH_CHECK_INTERVAL_SECONDS}"
    declare -A down_since=()
    local deadlock_since=""

    while ps -p "$client_pid" >/dev/null 2>&1; do
        local now
        now=$(date +%s)

        for i in "${!PREFILL_PIDS[@]}"; do
            local pid="${PREFILL_PIDS[$i]}"
            local key="prefill-${i}"
            if ! ps -p "$pid" >/dev/null 2>&1; then
                if [[ -z "${down_since[$key]:-}" ]]; then
                    down_since[$key]="$now"
                    echo "[WARN] ${key} down, start countdown"
                elif (( now - down_since[$key] >= timeout_seconds )); then
                    echo "[ERROR] ${key} down for ${timeout_seconds}s"
                    return 2
                fi
            else
                unset 'down_since[$key]'
            fi
        done

        for i in "${!DECODE_PIDS[@]}"; do
            local pid="${DECODE_PIDS[$i]}"
            local key="decode-${i}"
            if ! ps -p "$pid" >/dev/null 2>&1; then
                if [[ -z "${down_since[$key]:-}" ]]; then
                    down_since[$key]="$now"
                    echo "[WARN] ${key} down, start countdown"
                elif (( now - down_since[$key] >= timeout_seconds )); then
                    echo "[ERROR] ${key} down for ${timeout_seconds}s"
                    return 2
                fi
            else
                unset 'down_since[$key]'
            fi
        done

        # [WT] GPU deadlock watchdog DISABLED (2026-11-07).
        # It false-positived on KV-bound decode instances: they are busy (many
        # running requests) yet GPU compute util can sit at 0% while they wait
        # on KV memory, so the util-only heuristic killed runs that were still
        # making progress (e.g. qwen_cap scheme4 killed at 3111/4080).
        # The operator monitors bench_trace progress + decode logs instead and
        # kills a genuinely stuck run manually.  Set
        # WT_ENABLE_GPU_DEADLOCK_GUARD=1 to re-enable the old heuristic.
        if [[ "${WT_ENABLE_GPU_DEADLOCK_GUARD:-0}" == "1" ]]; then
            local _deadlock_detected=1
            local _util _mem
            while IFS=',' read -r _util _mem; do
                _util="${_util// /}"
                _mem="${_mem// /}"
                if [[ -z "$_mem" ]] || (( _mem <= 0 )) || (( _util > 0 )); then
                    _deadlock_detected=0
                    break
                fi
            done < <(nvidia-smi --query-gpu=utilization.gpu,memory.used \
                         --format=csv,noheader,nounits -i "${DECODE_GPUS}" 2>/dev/null)
            if (( _deadlock_detected )); then
                if [[ -z "$deadlock_since" ]]; then
                    deadlock_since="$now"
                    echo "[WARN] GPU deadlock detected (decode GPUs: ${DECODE_GPUS} all mem>0 util=0), start ${GPU_DEADLOCK_TIMEOUT_SECONDS}s countdown"
                elif (( now - deadlock_since >= GPU_DEADLOCK_TIMEOUT_SECONDS )); then
                    echo "[ERROR] GPU deadlock persisted ${GPU_DEADLOCK_TIMEOUT_SECONDS}s, skipping config"
                    return 3
                else
                    echo "[WARN] GPU deadlock ongoing: $((now - deadlock_since))s / ${GPU_DEADLOCK_TIMEOUT_SECONDS}s"
                fi
            else
                if [[ -n "$deadlock_since" ]]; then
                    echo "[INFO] GPU deadlock cleared after $((now - deadlock_since))s"
                    deadlock_since=""
                fi
            fi
        fi
        # [WT] end GPU deadlock detection

        sleep "$check_interval"
    done

    wait "$client_pid"
}

run_one_benchmark() {
    local repeat_idx="$1"
    local timestamp="$2"
    local log_file="${LOG_DIR}/bench_trace_pd${CURRENT_PD_RATIO_TAG}_tpot${CURRENT_TPOT_MS}ms_mt${MODEL_MT_TAG_FOR_NAME}_rep${repeat_idx}.log"

    local bench_cmd=(
        python3 "$BENCH_SCRIPT"
        --backend "$BENCH_BACKEND"
        --port "$BENCH_PORT"
        --endpoint "$BENCH_ENDPOINT"
        --model "$CURRENT_BENCH_MODEL"
        --dataset-name "$BENCH_DATASET_NAME"
        --dataset-path "$CURRENT_BENCH_DATASET_PATH"
        --save-output "$SAVE_OUTPUT"
        --out-path "$RESULT_DIR"
        --maxtokenscustom "$CURRENT_BENCH_MAX_TOKENS"
        --seed "$VLLM_SEED"
        --max-concurrency "$BENCH_MAX_CONCURRENCY"
        --temperature "$BENCH_TEMPERATURE"
        --top-p "$BENCH_TOP_P"
        --top-k "$BENCH_TOP_K"
        --repetition-penalty "$BENCH_REPETITION_PENALTY"
    )

    local goodput_arr=()
    if [[ -n "$BENCH_GOODPUT" ]]; then
        read -r -a goodput_arr <<< "$BENCH_GOODPUT"
        bench_cmd+=(--goodput "${goodput_arr[@]}")
    fi

    if [[ -n "$BENCH_IGNORE_EOS" ]]; then
        bench_cmd+=(--ignore-eos)
    fi
    if is_true "$SAVE_SAMPLE"; then
        bench_cmd+=(--save-sample)
    fi

    echo "Command: ${bench_cmd[*]}"

    setsid env BENCHMARK_INSTANCE_ID="benchmark-${CURRENT_BENCH_DATASET_LABEL}" bash -c "exec ${bench_cmd[*]}" > "$log_file" 2>&1 &
    local client_pid=$!

    local bench_exit_code=0
    if run_benchmark_with_health_guard "$client_pid"; then
        bench_exit_code=0
    else
        bench_exit_code=$?
    fi

    if [[ "$bench_exit_code" -eq 2 ]]; then
        echo "Dataset ${CURRENT_BENCH_DATASET_PATH}: instance down >= ${INSTANCE_DOWN_TIMEOUT_SECONDS}s"
        kill -TERM "$client_pid" >/dev/null 2>&1 || true
        sleep 2
        kill -KILL "$client_pid" >/dev/null 2>&1 || true
        return 2
    fi

    # [WT] GPU deadlock timeout: kill benchmark client and propagate
    if [[ "$bench_exit_code" -eq 3 ]]; then
        echo "Dataset ${CURRENT_BENCH_DATASET_PATH}: GPU deadlock >= ${GPU_DEADLOCK_TIMEOUT_SECONDS}s, skipping"
        kill -TERM "$client_pid" >/dev/null 2>&1 || true
        sleep 2
        kill -KILL "$client_pid" >/dev/null 2>&1 || true
        return 3
    fi

    if [[ "$bench_exit_code" -ne 0 ]]; then
        echo "ERROR: benchmark failed with exit code ${bench_exit_code}"
        log_failed_config "benchmark_exit_code=${bench_exit_code}"
        return "$bench_exit_code"
    fi

    plot_decode_load "$timestamp"
    return 0
}

trap cleanup INT TERM EXIT

main() {
    # Disable -e for the entire batch run so a single dataset failure never
    # aborts the remaining datasets. Individual error paths use explicit checks.
    set +e

    check_required_files
    check_num_gpus

    ensure_python_library_installed pandas
    ensure_python_library_installed datasets
    ensure_python_library_installed vllm
    ensure_python_library_installed quart

    mkdir -p "$BASE_RESULT_DIR"
    FAILED_CONFIG_LOG="${BASE_RESULT_DIR}/failed_configs_${RUN_BATCH_TIMESTAMP}.log"
    SUMMARY_CSV="${BASE_RESULT_DIR}/summary_${RUN_BATCH_TIMESTAMP}.csv"
    init_summary_csv
    apply_temp_skip_existing_results_once
    init_skip_result_dir_keys

    resolve_trace_paths
    IFS=',' read -ra TPOT_ARRAY <<< "$TPOT_MS_LIST"
    IFS=',' read -ra PD_RATIO_ARRAY <<< "$PD_RATIO_LIST"

    local failed=0
    local total_datasets=0
    local trace_path tpot_ms model_name pd_ratio dataset_idx dataset_path repeat_idx timestamp code experiment_variant

    for trace_path in "${TRACE_PATH_ARRAY[@]}"; do
        configure_trace "$trace_path"

        echo "============================================================"
        echo "Trace=${CURRENT_TRACE_NAME}"
        echo "Dataset root=${CURRENT_TRACE_PATH}"
        echo "============================================================"

        mapfile -t SELECTED_MODEL_ARRAY < <(get_selected_model_names)

        for tpot_ms in "${TPOT_ARRAY[@]}"; do
            tpot_ms="$(echo "$tpot_ms" | xargs)"
            [[ -z "$tpot_ms" ]] && continue
            configure_tpot "$tpot_ms"

            for model_name in "${SELECTED_MODEL_ARRAY[@]}"; do
                if ! configure_active_model "$model_name"; then
                    failed=$((failed + 1))
                    continue
                fi

                echo "============================================================"
                echo "Trace=${CURRENT_TRACE_NAME} Model=${CURRENT_MODEL_NAME}"
                echo "MODEL=${MODEL}"
                echo "VLLM_DTYPE=${VLLM_DTYPE}"
                echo "VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN}"
                echo "MODEL_MT_TAG_FOR_NAME=${MODEL_MT_TAG_FOR_NAME}"
                echo "PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS=${PREFILL_VLLM_MAX_NUM_BATCHED_TOKENS}"
                echo "DECODE_VLLM_MAX_NUM_BATCHED_TOKENS=${DECODE_VLLM_MAX_NUM_BATCHED_TOKENS}"
                echo "PD_RATIO_LIST=${PD_RATIO_LIST}"
                echo "TPOT_MS_LIST=${TPOT_MS_LIST}"
                echo "============================================================"

                for pd_ratio in "${PD_RATIO_ARRAY[@]}"; do
                    pd_ratio="$(echo "$pd_ratio" | xargs)"
                    [[ -z "$pd_ratio" ]] && continue
                    if ! configure_pd_ratio "$pd_ratio"; then
                        failed=$((failed + 1))
                        continue
                    fi

                    resolve_dataset_paths
                    total_datasets="${#DATASET_PATH_ARRAY[@]}"

                    echo "------------------------------------------------------------"
                    echo "PD=${CURRENT_PD_RATIO} matched ${total_datasets} dataset(s)."
                    for dataset_idx in "${!DATASET_PATH_ARRAY[@]}"; do
                        echo "  [$((dataset_idx + 1))/${total_datasets}] ${DATASET_PATH_ARRAY[$dataset_idx]}"
                    done

                    for dataset_idx in "${!DATASET_PATH_ARRAY[@]}"; do
                        dataset_path="${DATASET_PATH_ARRAY[$dataset_idx]}"
                        set_current_dataset "$dataset_path"

                        echo "============================================================"
                        echo "Trace=${CURRENT_TRACE_NAME} Model=${CURRENT_MODEL_NAME} Dataset [$((dataset_idx + 1))/${total_datasets}]: ${CURRENT_BENCH_DATASET_PATH}"
                        echo "PD=${CURRENT_PD_RATIO} TPOT=${CURRENT_TPOT_MS}ms"
                        echo "============================================================"

                        for repeat_idx in $(seq 1 "$RUN_REPEAT_PER_CONFIG"); do
                            for experiment_variant in ${COMPARE_EXPERIMENT_VARIANTS}; do
                                if ! configure_experiment_variant "$experiment_variant"; then
                                    failed=$((failed + 1))
                                    continue
                                fi

                                timestamp=$(date +%Y%m%d_%H%M%S)

                                if should_skip_existing_result "$repeat_idx"; then
                                    echo "SKIP: matched existing result config $(current_benchmark_dir_key "$repeat_idx") method=${CURRENT_RUN_METHOD}"
                                    append_summary_csv "skipped_existing_result" "$repeat_idx" "" "existing_result_dir_match"
                                    continue
                                fi

                                if ! check_selected_gpus_idle; then
                                    echo "SKIP: GPUs are not idle for method=${CURRENT_RUN_METHOD}, model=${CURRENT_MODEL_NAME}, pd=${CURRENT_PD_RATIO}, tpot=${CURRENT_TPOT_MS}ms, dataset=${CURRENT_BENCH_DATASET_PATH}"
                                    failed=$((failed + 1))
                                    append_summary_csv "skipped_gpu_busy" "$repeat_idx" "" "gpu_not_idle"
                                    sleep "$SLEEP_BETWEEN_RUNS"
                                    continue
                                fi

                                setup_directories "$timestamp" "$repeat_idx"
                                dump_config_json "$repeat_idx" "$timestamp"

                                echo "Run: method=${CURRENT_RUN_METHOD}, trace=${CURRENT_TRACE_NAME}, model=${CURRENT_MODEL_NAME}, dataset=${CURRENT_BENCH_DATASET_LABEL}, pd=${CURRENT_PD_RATIO}, tpot=${CURRENT_TPOT_MS}ms, repeat=${repeat_idx}/${RUN_REPEAT_PER_CONFIG}"

                                # [WT] Instance crash restart loop: retry up to INSTANCE_RESTART_MAX_RETRIES times when
                                # start_servers fails or run_one_benchmark returns exit code 2 (instance_health_failure).
                                local restart_attempt=0
                                local config_done=0
                                while [[ "$config_done" -eq 0 ]]; do
                                    if ! start_servers "$timestamp"; then
                                        echo "ERROR: failed to start servers (attempt $((restart_attempt + 1)))"
                                        log_failed_config "start_servers_failed(attempt=$((restart_attempt + 1)))"
                                        stop_servers
                                        if (( restart_attempt < INSTANCE_RESTART_MAX_RETRIES )); then
                                            restart_attempt=$((restart_attempt + 1))
                                            echo "[WT][INSTANCE_RESTART] start_servers failed, restarting config (attempt ${restart_attempt}/${INSTANCE_RESTART_MAX_RETRIES})..."
                                            append_summary_csv "instance_restart" "$repeat_idx" "$BENCHMARK_DIR" "start_servers_failed_retry${restart_attempt}"
                                            sleep "$SLEEP_BETWEEN_RUNS"
                                            timestamp=$(date +%Y%m%d_%H%M%S)
                                            setup_directories "$timestamp" "$repeat_idx"
                                            dump_config_json "$repeat_idx" "$timestamp"
                                            continue
                                        fi
                                        failed=$((failed + 1))
                                        append_summary_csv "start_failed" "$repeat_idx" "$BENCHMARK_DIR" "start_servers_failed"
                                        config_done=1
                                        break
                                    fi

                                    if run_one_benchmark "$repeat_idx" "$timestamp"; then
                                        echo "Benchmark finished successfully"
                                        append_summary_csv "success" "$repeat_idx" "$BENCHMARK_DIR" ""
                                        config_done=1
                                    else
                                        code=$?
                                        if [[ "$code" -eq 2 ]] && (( restart_attempt < INSTANCE_RESTART_MAX_RETRIES )); then
                                            restart_attempt=$((restart_attempt + 1))
                                            echo "[WT][INSTANCE_RESTART] Instance crashed during benchmark, restarting config (attempt ${restart_attempt}/${INSTANCE_RESTART_MAX_RETRIES})..."
                                            append_summary_csv "instance_restart" "$repeat_idx" "$BENCHMARK_DIR" "instance_health_failure_retry${restart_attempt}"
                                            stop_servers
                                            local _all_ports_restart=()
                                            IFS=',' read -ra _prefill_ports_restart <<< "$PREFILL_PORTS"
                                            IFS=',' read -ra _decode_ports_restart <<< "$DECODE_PORTS"
                                            _all_ports_restart+=("${PROXY_PORT}")
                                            _all_ports_restart+=("${_prefill_ports_restart[@]}")
                                            _all_ports_restart+=("${_decode_ports_restart[@]}")
                                            for _p in "${_all_ports_restart[@]}"; do
                                                wait_for_port_free "$_p" 15 || true
                                            done
                                            sleep "$SLEEP_BETWEEN_RUNS"
                                            timestamp=$(date +%Y%m%d_%H%M%S)
                                            setup_directories "$timestamp" "$repeat_idx"
                                            dump_config_json "$repeat_idx" "$timestamp"
                                            continue
                                        fi
                                        failed=$((failed + 1))
                                        if [[ "$code" -eq 2 ]]; then
                                            echo "Benchmark stopped due to instance health failure (all ${INSTANCE_RESTART_MAX_RETRIES} retries exhausted)"
                                            append_summary_csv "benchmark_failed" "$repeat_idx" "$BENCHMARK_DIR" "instance_health_failure"
                                        elif [[ "$code" -eq 3 ]]; then
                                            # [WT] GPU deadlock skip
                                            echo "Benchmark skipped due to GPU deadlock timeout"
                                            append_summary_csv "benchmark_skipped" "$repeat_idx" "$BENCHMARK_DIR" "gpu_deadlock_timeout"
                                        else
                                            append_summary_csv "benchmark_failed" "$repeat_idx" "$BENCHMARK_DIR" "benchmark_exit_code=${code}"
                                        fi
                                        config_done=1
                                    fi
                                done
                                # [WT] end instance crash restart loop

                                stop_servers

                                local all_ports=()
                                IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
                                IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"
                                all_ports+=("${PROXY_PORT}")
                                all_ports+=("${PREFILL_PORT_ARRAY[@]}")
                                all_ports+=("${DECODE_PORT_ARRAY[@]}")
                                for p in "${all_ports[@]}"; do
                                    wait_for_port_free "$p" 15 || true
                                done

                                sleep "$SLEEP_BETWEEN_RUNS"
                            done
                        done
                    done
                done
            done
        done
    done

    echo "============================================================"
    echo "Evaluation finished"
    echo "Failed runs: $failed"
    echo "Results root: $(realpath "$BASE_RESULT_DIR")"
    echo "Failed config log: $FAILED_CONFIG_LOG"
    echo "Summary CSV: $SUMMARY_CSV"
    echo "============================================================"
}

main "$@"
