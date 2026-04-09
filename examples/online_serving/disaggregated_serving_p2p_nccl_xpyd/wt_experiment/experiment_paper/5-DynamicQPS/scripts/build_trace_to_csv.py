import json
import csv
import math
import argparse
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

from trace_utils import add_trace_type_argument, load_trace_records


DEFAULT_BLOCK_SIZE = 512


def _trace_block_size(trace_type):
    if trace_type == "qwen":
        return 16
    return DEFAULT_BLOCK_SIZE


class BlockGenerator:
    """
    为每个 hash_id 生成固定 token block
    保证 prefix reuse
    """

    def __init__(self, tokenizer, block_size):

        self.tokenizer = tokenizer
        self.block_size = block_size
        self.cache = {}
        self.fragment_cache = {}

    def _encode_fragment(self, text):

        if text in self.fragment_cache:
            return self.fragment_cache[text]

        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        self.fragment_cache[text] = token_ids

        return token_ids

    def _build_deterministic_block(self, block_id):

        # Build the block incrementally so we do not repeatedly tokenize an ever-growing string.
        # Same block_id always maps to the same token sequence.
        token_ids = []
        idx = 0

        while len(token_ids) < self.block_size:
            fragment = f" block_{block_id}_{idx} \n"
            fragment_ids = self._encode_fragment(fragment)

            if not fragment_ids:
                raise ValueError(f"empty token fragment generated for block_id={block_id}, idx={idx}")

            token_ids.extend(fragment_ids)

            idx += 1

        return token_ids[:self.block_size]

    def get_block(self, block_id):

        if block_id in self.cache:
            return self.cache[block_id]

        tokens = self._build_deterministic_block(block_id)

        self.cache[block_id] = tokens

        return tokens


def build_prompt(hash_ids, input_length, block_gen):

    tokens = []

    for hid in hash_ids:

        block = block_gen.get_block(hid)

        tokens.extend(block)

    tokens = tokens[:input_length]

    return tokens


def build_synthetic_hash_ids(req_id, target_no_special, block_size):

    if target_no_special <= 0:
        return []

    block_count = max(1, math.ceil(target_no_special / block_size))
    base_id = req_id * 1000003

    return [base_id + offset for offset in range(block_count)]


def decode_tokens(tokenizer, token_ids):

    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False
    )


def canonicalize_token_ids(tokenizer, token_ids):

    text = decode_tokens(tokenizer, token_ids)

    return tokenizer.encode(text, add_special_tokens=False)


def find_stable_atoms(tokenizer):

    candidates = [
        " a", " b", " c", " d", " e", " f", " g", " h", " i", " j",
        " k", " l", " m", " n", " o", " p", " q", " r", " s", " t",
        " u", " v", " w", " x", " y", " z", " .", " ,", " !", " ?",
        " 0", " 1", " 2", " 3", " 4", " 5", " 6", " 7", " 8", " 9"
    ]

    atoms = []

    for atom in candidates:
        ids = tokenizer.encode(atom, add_special_tokens=False)

        if len(ids) != 1:
            continue

        ok = True

        # Ensure additive behavior for repetition.
        for n in (2, 4, 8, 16, 32, 64):
            if len(tokenizer.encode(atom * n, add_special_tokens=False)) != n:
                ok = False
                break

        if ok:
            atoms.append(atom)

    # Keep only pairwise-additive atoms so adjacent block boundaries are stable.
    pairwise_atoms = []

    for atom_i in atoms:
        ok = True

        for atom_j in atoms:
            pair_len = len(tokenizer.encode(atom_i + atom_j, add_special_tokens=False))

            if pair_len != 2:
                ok = False
                break

        if ok:
            pairwise_atoms.append(atom_i)

    if pairwise_atoms:
        return pairwise_atoms

    return atoms


def build_fallback_prompt(hash_ids, target_no_special, tokenizer, atoms, block_size):

    if not atoms:
        raise ValueError("fallback requires at least one stable atom")

    # Build deterministic token ids keyed by hash_id using single-token atoms.
    atom_ids = [tokenizer.encode(atom, add_special_tokens=False)[0] for atom in atoms]

    token_ids = []

    for hid in hash_ids:
        atom_id = atom_ids[hid % len(atom_ids)]
        token_ids.extend([atom_id] * block_size)

    token_ids = token_ids[:target_no_special]

    prompt = decode_tokens(tokenizer, token_ids)

    # Canonicalize once to avoid decode/encode drift at the tail.
    canonical_ids = tokenizer.encode(prompt, add_special_tokens=False)

    if len(canonical_ids) < target_no_special:
        pad_atom_id = atom_ids[0]
        canonical_ids.extend([pad_atom_id] * (target_no_special - len(canonical_ids)))
    elif len(canonical_ids) > target_no_special:
        canonical_ids = canonical_ids[:target_no_special]

    prompt = decode_tokens(tokenizer, canonical_ids)

    return prompt


def build_token_ids_for_len(hash_ids, token_len, atom_ids, block_cache, block_size):

    token_len = max(1, token_len)

    out = []

    for hid in hash_ids:
        if hid not in block_cache:
            atom_id = atom_ids[hid % len(atom_ids)]
            block_cache[hid] = [atom_id] * block_size

        out.extend(block_cache[hid])

        if len(out) >= token_len:
            break

    return out[:token_len]


def fast_match_prompt_len(tokenizer, hash_ids, target_no_special, target_len, atom_ids, block_cache, block_size):

    max_pool = len(hash_ids) * block_size
    token_len = max(1, min(target_no_special, max_pool))
    eval_cache = {}

    def evaluate(candidate_len):
        candidate_len = max(1, min(candidate_len, max_pool))

        if candidate_len in eval_cache:
            return eval_cache[candidate_len]

        token_ids = build_token_ids_for_len(hash_ids, candidate_len, atom_ids, block_cache, block_size)
        prompt = decode_tokens(tokenizer, token_ids)
        actual_len = encode_len(tokenizer, prompt)
        eval_cache[candidate_len] = (prompt, candidate_len, actual_len)

        return eval_cache[candidate_len]

    # Fast correction loop; usually converges in a few steps.
    for _ in range(6):
        prompt, used_len, actual_len = evaluate(token_len)

        if actual_len == target_len:
            return prompt, used_len, actual_len, True

        delta = target_len - actual_len
        next_len = max(1, min(max_pool, used_len + delta))

        if next_len == token_len:
            break

        token_len = next_len

    # Tiny local search around the current guess.
    left = max(1, token_len - 32)
    right = min(max_pool, token_len + 32)

    for cand_len in range(left, right + 1):
        prompt, used_len, actual_len = evaluate(cand_len)

        if actual_len == target_len:
            return prompt, used_len, actual_len, True

    return "", token_len, -1, False


def build_messages(prompt):

    return [{"role": "user", "content": str(prompt)}]


def ensure_chat_template(tokenizer, model_name):

    if getattr(tokenizer, "chat_template", None):
        return

    raise ValueError(
        f"model '{model_name}' does not provide a chat_template; this script requires tokenizer.apply_chat_template(...) support"
    )


def build_templated_prompt(tokenizer, prompt):

    return tokenizer.apply_chat_template(
        build_messages(prompt),
        tokenize=False,
        add_generation_prompt=True
    )


def chat_token_ids(tokenizer, prompt):
    result = tokenizer.apply_chat_template(
        build_messages(prompt),
        tokenize=True,
        add_generation_prompt=True
    )

    # Newer transformers may return BatchEncoding instead of a plain token-id list.
    # Normalize to a flat list[int] so downstream length logic is stable.
    if hasattr(result, "get"):
        input_ids = result.get("input_ids")
        if isinstance(input_ids, list):
            return input_ids

    return result


def encode_len(tokenizer, text):

    return len(chat_token_ids(tokenizer, text))


def align_prompt_to_target_len(tokenizer, source_tokens, target_len, init_token_len):

    if init_token_len > len(source_tokens):
        raise ValueError(
            f"init token length {init_token_len} is larger than source token pool {len(source_tokens)}"
        )

    token_len = max(1, init_token_len)
    eval_cache = {}

    def evaluate(candidate_len):

        if candidate_len in eval_cache:
            return eval_cache[candidate_len]

        prompt = decode_tokens(tokenizer, source_tokens[:candidate_len])
        actual_len = encode_len(tokenizer, prompt)
        eval_cache[candidate_len] = (prompt, actual_len)

        return eval_cache[candidate_len]

    # First, do a few gradient-like correction steps by using current length error.
    for _ in range(8):
        prompt, actual_len = evaluate(token_len)

        if actual_len == target_len:
            return prompt, token_len, actual_len

        diff = target_len - actual_len
        next_len = max(1, min(len(source_tokens), token_len + diff))

        if next_len == token_len:
            break

        token_len = next_len

    # If still not exact, do a local window search around current token_len.
    left = max(1, token_len - 256)
    right = min(len(source_tokens), token_len + 256)

    for cand_len in range(left, right + 1):
        prompt, actual_len = evaluate(cand_len)

        if actual_len == target_len:
            return prompt, cand_len, actual_len

    # Rare fallback: the chat-template overhead can drift more than the local window.
    # Search the remaining candidate lengths across the whole source range.
    for cand_len in range(1, len(source_tokens) + 1):
        if left <= cand_len <= right:
            continue

        prompt, actual_len = evaluate(cand_len)

        if actual_len == target_len:
            return prompt, cand_len, actual_len

    raise ValueError(
        f"failed to align prompt length: target={target_len}, init_token_len={init_token_len}, search_range=[{left}, {right}], source_token_pool={len(source_tokens)}"
    )


def get_special_token_overhead(tokenizer):

    return encode_len(tokenizer, "")


def build_dataset(trace_path, output_csv, model_name, trace_type, enable_slow_path=False):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ensure_chat_template(tokenizer, model_name)

    trace_block_size = _trace_block_size(trace_type)
    block_gen = BlockGenerator(tokenizer, trace_block_size) if enable_slow_path else None
    special_overhead = get_special_token_overhead(tokenizer)
    stable_atoms = find_stable_atoms(tokenizer)
    atom_ids = [tokenizer.encode(atom, add_special_tokens=False)[0] for atom in stable_atoms]
    block_cache = {}

    if not stable_atoms:
        raise ValueError("no stable one-token atoms found for fallback construction")

    loaded = load_trace_records(
        Path(trace_path),
        trace_type=trace_type,
        require_timestamp=True,
        require_lengths=True,
        skip_invalid=False,
    )
    trace = loaded.records

    print("total trace requests:", len(trace))
    print("chat template overhead:", special_overhead)
    print("stable fallback atoms:", len(stable_atoms))
    print("slow path enabled:", enable_slow_path)
    print("trace type:", trace_type)
    print("trace block size:", trace_block_size)

    corrected = 0
    fallback_used = 0
    slow_path_used = 0

    with open(output_csv, "w", newline="") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow([
            "req_id",
            "timestamp",
            "prompt",
            "templated_prompt",
            "prompt_tokens",
            "output_tokens",
            "hash_ids"
        ])

        for req_id, item in enumerate(tqdm(trace)):

            # Always write dataset timestamps in milliseconds for all trace types.
            if item.timestamp_ms is None:
                raise ValueError(f"req_id={req_id}: missing timestamp_ms")
            timestamp = int(round(item.timestamp_ms))

            input_len = item.input_length
            output_len = item.output_length

            if input_len is None or output_len is None:
                raise ValueError(f"req_id={req_id}: missing input/output length")

            if trace_type in ("mooncake", "qwen"):
                hash_ids = list(item.hash_ids)
                expected_blocks = math.ceil(input_len / trace_block_size)

                if len(hash_ids) != expected_blocks:
                    original = len(hash_ids)
                    if len(hash_ids) > expected_blocks:
                        hash_ids = hash_ids[:expected_blocks]
                    else:
                        start_id = req_id * 1000003 + len(hash_ids)
                        hash_ids.extend(start_id + offset for offset in range(expected_blocks - len(hash_ids)))

                    print(
                        f"[WARN] req_id {req_id}: expected {expected_blocks} blocks based on input_length, "
                        f"but got {original} hash_ids. Adjusted to {len(hash_ids)}."
                    )
            else:
                hash_ids = []

            target_no_special = max(0, input_len - special_overhead)

            source_block_ids = hash_ids if trace_type in ("mooncake", "qwen") else build_synthetic_hash_ids(req_id, target_no_special, trace_block_size)

            if target_no_special == 0:
                prompt = ""
                used_token_len = 0
                actual_len = encode_len(tokenizer, prompt)
                fast_ok = actual_len == input_len
            else:
                prompt, used_token_len, actual_len, fast_ok = fast_match_prompt_len(
                    tokenizer=tokenizer,
                    hash_ids=source_block_ids,
                    target_no_special=target_no_special,
                    target_len=input_len,
                    atom_ids=atom_ids,
                    block_cache=block_cache,
                    block_size=trace_block_size,
                )

            if not fast_ok:
                if enable_slow_path:
                    slow_path_used += 1

                    source_tokens = build_prompt(
                        source_block_ids,
                        len(source_block_ids) * trace_block_size,
                        block_gen
                    )

                    source_tokens = canonicalize_token_ids(tokenizer, source_tokens)

                    if target_no_special > len(source_tokens):
                        raise ValueError(
                            f"req_id={req_id}: insufficient source tokens, need={target_no_special}, got={len(source_tokens)}"
                        )

                    try:
                        prompt, used_token_len, actual_len = align_prompt_to_target_len(
                            tokenizer=tokenizer,
                            source_tokens=source_tokens,
                            target_len=input_len,
                            init_token_len=target_no_special
                        )
                    except ValueError:
                        prompt = build_fallback_prompt(
                            hash_ids=source_block_ids,
                            target_no_special=target_no_special,
                            tokenizer=tokenizer,
                            atoms=stable_atoms,
                            block_size=trace_block_size,
                        )
                        used_token_len = target_no_special
                        actual_len = encode_len(tokenizer, prompt)
                        fallback_used += 1
                else:
                    prompt = build_fallback_prompt(
                        hash_ids=source_block_ids,
                        target_no_special=target_no_special,
                        tokenizer=tokenizer,
                        atoms=stable_atoms,
                        block_size=trace_block_size,
                    )
                    used_token_len = target_no_special
                    actual_len = encode_len(tokenizer, prompt)
                    fallback_used += 1

            if used_token_len != target_no_special:
                corrected += 1

            # Safety assertion: align function guarantees exact length.
            if actual_len != input_len:
                raise AssertionError(
                    f"req_id={req_id}: align returned wrong length, expected={input_len}, actual={actual_len}"
                )

            templated_prompt = build_templated_prompt(tokenizer, prompt)

            # prompt_tokens should be the source trace's input_length
            # and should satisfy: len(apply_chat_template(prompt)) == prompt_tokens
            writer.writerow([
                req_id,
                timestamp,
                prompt,
                templated_prompt,
                input_len,  # 使用源trace的原始input_length作为prompt_tokens
                output_len,
                json.dumps(hash_ids, ensure_ascii=True)
            ])

    print("dataset saved to:", output_csv)
    print("length-corrected requests:", corrected)
    print("slow-path requests:", slow_path_used)
    print("fallback-used requests:", fallback_used)


def main():

    parser = argparse.ArgumentParser()
    add_trace_type_argument(parser)
    parser.set_defaults(trace_type="qwen")# qwen,burstgpt,mooncake, azure
    parser.add_argument(
        "--trace",
        default="/root/.cache/huggingface/hub/datasets/wt_predictor/filter_trace_8192/qwen_trace/qwen_thinking_blksz_16_th8192.jsonl",
        help="输入 trace 文件路径"
    )

    parser.add_argument(
        "--output",
        default="/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/5-DynamicQPS/tmp_dataset/qwen_thinking_blksz_16_th8192.csv",
    )

    parser.add_argument(
        "--model",
        default="/root/.cache/huggingface/hub/Meta-Llama-3-8B-Instruct"
    )
    # /root/share/models/Meta-Llama-3-8B-Instruct
    # /root/share/models/Qwen2.5-7B-Instruct
    parser.add_argument(
        "--enable_slow_path",
        action="store_true",
        help="Enable legacy slow alignment path before fallback"
    )
    
    args = parser.parse_args()

    build_dataset(
        args.trace,
        args.output,
        args.model,
        args.trace_type,
        enable_slow_path=args.enable_slow_path
    )


if __name__ == "__main__":
    main()