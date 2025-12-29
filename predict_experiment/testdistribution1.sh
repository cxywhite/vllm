export CUDA_VISIBLE_DEVICES=7
nohup python testdistribution1.py \
  --base-url http://localhost:8004 \
  --model-name /root/.cache/huggingface/hub/Qwen2.5-7B-Instruct \
  --dataset /root/myshare/sharegpt_merged.jsonl \
  --n 500 --m 10 --seed 42 --split train \
  --max-tokens-custom 32768 \
  --temperature 1.0 --top-p 1.0 --topk -1 --repetition_penalty 1.0 \
  --max-concurrency 128 \
  --rps 5.0 \
  --request-timeout 7200 \
  --burstiness 1.0 --verbose \
  --ensure-success --ensure-retries 5 --ensure-backoff 2.0 \
  --output dataset_result/result/results.csv > dataset_result/log/client5.log 2>&1 &