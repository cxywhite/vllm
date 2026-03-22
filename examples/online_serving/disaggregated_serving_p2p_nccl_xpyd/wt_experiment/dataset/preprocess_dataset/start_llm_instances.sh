#!/bin/bash
ulimit -n 65535
# 设置模型路径
MODEL_PATH="/root/.cache/huggingface/hub/Qwen2.5-7B-Instruct"
# unset SETUPTOOLS_USE_DISTUTILS
# 启动实例 1 在 GPU 0 上
CUDA_VISIBLE_DEVICES=4 \
vllm serve $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --port 10080 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-batched-tokens 40960 \
    --host 0.0.0.0 &

# 启动实例 2 在 GPU 1 上
CUDA_VISIBLE_DEVICES=5 \
vllm serve $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --port 10081 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-batched-tokens 40960 \
    --host 0.0.0.0 &

CUDA_VISIBLE_DEVICES=6 \
vllm serve $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --port 10082 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-batched-tokens 40960 \
    --host 0.0.0.0 &

# 启动实例 2 在 GPU 1 上
CUDA_VISIBLE_DEVICES=7 \
vllm serve $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --port 10083 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-batched-tokens 40960 \
    --host 0.0.0.0 &

CUDA_VISIBLE_DEVICES=2 \
vllm serve $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --port 10084 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-batched-tokens 40960 \
    --host 0.0.0.0 &

# 启动实例 2 在 GPU 1 上
CUDA_VISIBLE_DEVICES=3 \
vllm serve $MODEL_PATH \
    --tokenizer $MODEL_PATH \
    --port 10085 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --max-num-batched-tokens 40960 \
    --host 0.0.0.0 &

# CUDA_VISIBLE_DEVICES=6 \
# vllm serve $MODEL_PATH \
#     --tokenizer $MODEL_PATH \
#     --port 10194 \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 32768 \
#     --max-num-batched-tokens 40960 \
#     --host 0.0.0.0 &

# # 启动实例 2 在 GPU 1 上
# CUDA_VISIBLE_DEVICES=7 \
# vllm serve $MODEL_PATH \
#     --tokenizer $MODEL_PATH \
#     --port 10077 \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.9 \
#     --max-model-len 32768 \
#     --max-num-batched-tokens 40960 \
#     --host 0.0.0.0 &

# 运行基准测试
# python benchmark_serving.py   --backend vllm   --model /root/share/models/Qwen2.5-7B-Instruct   --endpoint /v1/completions   --host 127.0.0.1   --port 9002   --dataset-name random --random-input-len 100  --num-prompts 32   --max-concurrency 16