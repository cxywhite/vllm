"""
Benchmark vLLM chat prefill latency on GPU across input_length and batch_size.

Behavior:
1) Use offline local model path (default: /root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct)
2) For each (input_length, batch_size) config, generate random chat requests
3) Measure generation latency with max_tokens=1 (prefill-focused)
4) Run each config 3 times (configurable) and report average latency
5) Save 2D table and details to a timestamped txt file

Example:
python benchmark_prefill_latency.py \
  --input_lengths 8192 \
  --batch_sizes 1024 \
  --gpu 6
python benchmark_prefill_latency.py \
  --input_lengths 32 64 128 256 512 1024 2048 4096 8192 \
  --batch_sizes 1 16 32 64 128 256 512 1024 \
  --gpu 7
python benchmark_prefill_latency.py \
  --input_lengths 8192 \
  --batch_sizes 1 16 32 64 128 256 512 1024 \
  --gpu 7
"""

import argparse
import os
import random
import string
import time
from datetime import datetime

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


EMPTY_MARK = "空"
DEFAULT_MODEL_PATH = "/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct"
DEFAULT_OUTPUT_DIR = f'/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/tmp/baseline_316/prefill_oom_test_1p1d_random'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM offline chat prefill latency on an input_length x batch_size grid."
    )
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input_lengths", type=int, nargs="+", required=True)
    parser.add_argument("--batch_sizes", type=int, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup_iters", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def random_text(num_words: int, rng: random.Random) -> str:
    words = []
    for _ in range(num_words):
        wlen = rng.randint(3, 10)
        words.append("".join(rng.choice(string.ascii_lowercase) for _ in range(wlen)))
    return " ".join(words)


def _prompt_token_len(tokenizer, prompt: str) -> int:
    return len(tokenizer(prompt, add_special_tokens=False)["input_ids"])


def _truncate_prompt_to_target(tokenizer, prompt: str, target_tokens: int) -> str:
    token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(token_ids) <= target_tokens:
        return prompt

    # Decode truncated ids and re-check because decode->encode can slightly drift.
    token_ids = token_ids[:target_tokens]
    truncated = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    while _prompt_token_len(tokenizer, truncated) > target_tokens and token_ids:
        token_ids = token_ids[:-1]
        truncated = tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    return truncated


def build_chat_prompt_with_target_length(
    tokenizer,
    target_len: int,
    rng: random.Random,
    max_adjust_rounds: int = 20,
) -> str:
    """
    Generate a chat prompt whose total token length is at most target_len - 1.
    (Reserves 1 token for the generated output so the request fits within max_model_len.)

    Strategy:
    1) Search for a long-enough random user message.
    2) Clamp token length to avoid exceeding model max length.
    """
    exact_target = max(1, target_len - 1)

    # Binary search the user text length in words, then clamp at token level.
    lo, hi = 1, max(2, exact_target)
    best_prompt = None
    best_len = -1

    for _ in range(max_adjust_rounds):
        mid = (lo + hi) // 2
        user_content = random_text(mid, rng)
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        n_tokens = _prompt_token_len(tokenizer, prompt)

        if n_tokens <= exact_target and n_tokens > best_len:
            best_prompt = prompt
            best_len = n_tokens
        if n_tokens == exact_target:
            return prompt
        if n_tokens < exact_target:
            lo = mid + 1
        else:
            hi = mid - 1
        if lo > hi:
            break

    if best_prompt is None:
        # Fallback for very small targets or tokenizer edge cases.
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "a"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return _truncate_prompt_to_target(tokenizer, prompt, exact_target)

    return _truncate_prompt_to_target(tokenizer, best_prompt, exact_target)


def build_requests(tokenizer, input_length: int, batch_size: int, rng: random.Random) -> list[str]:
    return [
        build_chat_prompt_with_target_length(tokenizer, input_length, rng)
        for _ in range(batch_size)
    ]


def run_one_config(
    llm: LLM,
    tokenizer,
    input_length: int,
    batch_size: int,
    repeats: int,
    warmup_iters: int,
    rng: random.Random,
) -> dict:
    prompts = build_requests(tokenizer, input_length, batch_size, rng)
    sampling_params = SamplingParams(max_tokens=1, temperature=0.80, top_p=0.90)

    for _ in range(warmup_iters):
        _ = llm.generate(prompts, sampling_params, use_tqdm=False)

    runs_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        _ = llm.generate(prompts, sampling_params, use_tqdm=False)
        runs_ms.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = sum(runs_ms) / len(runs_ms)
    return {
        "avg_ms": avg_ms,
        "runs_ms": runs_ms,
        "status": "ok",
        "reason": "",
    }


def format_2d_table(results: dict, input_lengths: list[int], batch_sizes: list[int]) -> list[str]:
    lines = []
    header = ["input_len\\bs"] + [str(bs) for bs in batch_sizes]
    col_width = 14

    lines.append("".join(f"{h:>{col_width}}" for h in header))
    lines.append("-" * (col_width * len(header)))

    for il in input_lengths:
        row = [str(il)]
        for bs in batch_sizes:
            item = results.get((il, bs))
            if item is None or item.get("avg_ms") is None:
                row.append(EMPTY_MARK)
            else:
                row.append(f"{item['avg_ms']:.3f}")
        lines.append("".join(f"{cell:>{col_width}}" for cell in row))
    return lines


def save_txt(output_dir: str, lines: list[str]) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"vllm_chat_prefill_grid_{ts}.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return out_path


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    # Pin the visible GPU for this process before initializing vLLM engine.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    llm = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    results = {}
    detail_lines = []

    for input_length in args.input_lengths:
        for batch_size in args.batch_sizes:
            # if input_length * batch_size >= 4096 * 1024:
            #     print(f'[Warn] skip input_length * batch_size >= 4096 *1024')
            #     continue
            try:
                stats = run_one_config(
                    llm=llm,
                    tokenizer=tokenizer,
                    input_length=input_length,
                    batch_size=batch_size,
                    repeats=args.repeats,
                    warmup_iters=args.warmup_iters,
                    rng=rng,
                )
                results[(input_length, batch_size)] = stats
                detail_lines.append(
                    "input_length={} batch_size={} runs_ms=[{}] avg_ms={:.3f}".format(
                        input_length,
                        batch_size,
                        ", ".join(f"{x:.3f}" for x in stats["runs_ms"]),
                        stats["avg_ms"],
                    )
                )
                print(
                    "[DONE] input_length={} batch_size={} runs_ms=[{}] avg_ms={:.3f}".format(
                        input_length,
                        batch_size,
                        ", ".join(f"{x:.3f}" for x in stats["runs_ms"]),
                        stats["avg_ms"],
                    )
                )
            except (RuntimeError, ValueError) as exc:
                reason = str(exc).replace("\n", " ").strip()
                results[(input_length, batch_size)] = {
                    "avg_ms": None,
                    "runs_ms": [],
                    "status": "empty",
                    "reason": reason,
                }
                detail_lines.append(
                    "input_length={} batch_size={} result={} reason={}".format(
                        input_length,
                        batch_size,
                        EMPTY_MARK,
                        reason,
                    )
                )
                print(
                    "[SKIP] input_length={} batch_size={} result={} reason={}".format(
                        input_length,
                        batch_size,
                        EMPTY_MARK,
                        reason,
                    )
                )

    table_lines = format_2d_table(results, args.input_lengths, args.batch_sizes)
    lines = [
        "=" * 72,
        "vLLM Offline Chat Prefill Benchmark (max_tokens=1)",
        f"time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"model_path: {args.model_path}",
        f"gpu: {args.gpu}",
        f"repeats: {args.repeats}",
        f"warmup_iters: {args.warmup_iters}",
        "Metric in 2D table: avg latency per config in ms",
        f"{EMPTY_MARK} means this config was skipped or failed",
        "=" * 72,
        "",
        "2D Summary Table",
    ]
    lines.extend(table_lines)
    lines.append("")
    lines.append("Per-Configuration Details")
    lines.extend(detail_lines)

    out_path = save_txt(args.output_dir, lines)
    print("\n".join(lines))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
