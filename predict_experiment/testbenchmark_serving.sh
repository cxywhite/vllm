# nohup python /root/vllm/benchmarks/benchmark_serving.py \
#   --backend vllm \
#   --base-url http://127.0.0.1:8007 \
#   --endpoint '/v1/completions' \
#   --model /root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct \
#   --dataset-name custom \
#   --seed 42 \
#   --num-prompts 500 \
#   --max-concurrency 64 \
#   --request-rate 1 \
#   --dataset-path /root/myshare/sharegpt_merged.jsonl \
#   --temperature 0.7 \
#   --top-p 0.8 \
#   --top-k 20 \
#   --repetition-penalty 1.05 \
#   --save-output True \
#   --out-path /root/predict/predict_experiment/dataset_result/benchmarkserving_result \
#   --maxtokenscustom 8192 \
#   --use-custom-csv True \
#   --m 10 > dataset_result/log/client15.log 2>&1 &
 
 nohup python /root/vllm/benchmarks/benchmark_serving_baseline.py \
  --backend vllm \
  --base-url http://127.0.0.1:8007 \
  --endpoint '/v1/completions' \
  --model /root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct \
  --dataset-name custom \
  --dataset-path /root/predict-schedule/baseline_experiment/pastfuture/dataset/lmsys-50k-filtered \
  --maxtokenscustom 8192 \
  --seed 42 \
  --num-prompts 500 \
  --max-concurrency 64 \
  --request-rate 1 \
  --temperature 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --repetition-penalty 1.05 \
  --save-output True \
  --out-path /root/predict/predict_experiment/dataset_result/benchmarkserving_result \
  > dataset_result/log/clientnew.log 2>&1 &
