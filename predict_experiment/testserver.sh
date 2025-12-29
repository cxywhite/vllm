export CUDA_VISIBLE_DEVICES=1
nohup vllm serve /root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct \
  --host 0.0.0.0 --port 8007 \
  --disable-log-requests \
  --max-num-seqs 1024 \
  --max-num-batched-tokens 40960 \
  --gpu_memory_utilization 0.95 \
  --enable_prefix_caching --enable_chunked_prefill > dataset_result/log/servernew.log 2>&1 &
