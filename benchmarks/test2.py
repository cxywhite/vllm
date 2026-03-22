async def async_request_openai_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:

    api_url = request_func_input.api_url
    assert api_url.endswith(("completions", "profile"))

    output = RequestFuncOutput()
    output.prompt_len = request_func_input.prompt_len
    output.req_id = request_func_input.req_id
    output.prompt = request_func_input.prompt
    output.expect_output_len = request_func_input.output_len

    payload = {
        "model": request_func_input.model_name
        if request_func_input.model_name
        else request_func_input.model,
        "prompt": request_func_input.prompt,
        "temperature": 0.7,
        "repetition_penalty": 1.05,
        "top_k": 20,
        "top_p": 0.8,
        "max_tokens": request_func_input.output_len,
        "logprobs": request_func_input.logprobs,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    if request_func_input.ignore_eos:
        payload["ignore_eos"] = request_func_input.ignore_eos
    if request_func_input.extra_body:
        payload.update(request_func_input.extra_body)

    headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
    if request_func_input.request_id:
        headers["x-request-id"] = request_func_input.request_id

    st = time.perf_counter()
    most_recent_timestamp = st
    generated_text = ""

    try:
        async with aiohttp.ClientSession(
            trust_env=True, timeout=AIOHTTP_TIMEOUT
        ) as session:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                # [WT] 2026-01-23 03:55:45
                # ⭐ 从 proxy 读取 prefill_done
                prefill_done_ts = response.headers.get("x-prefill-done-ts")

                if response.status != 200:
                    output.success = False
                    output.error = response.reason or ""
                    return output

                first_chunk_received = False

                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
                    if chunk == "[DONE]":
                        break

                    data = json.loads(chunk)

                    if choices := data.get("choices"):
                        text = choices[0].get("text")
                        timestamp = time.perf_counter()

                        if not first_chunk_received:
                            first_chunk_received = True
                            
                            # ⭐ TTFT = prefill_done − client_start
                            if prefill_done_ts is not None:
                                print(f'prefill_done_ts: {prefill_done_ts}, st: {st}')
                                output.ttft = float(prefill_done_ts) - st
                            else:
                                output.ttft = timestamp - st
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp
                        generated_text += text or ""

                    if usage := data.get("usage"):
                        output.output_tokens = usage.get("completion_tokens", 0)

                output.generated_text = generated_text
                output.latency = most_recent_timestamp - st
                output.success = first_chunk_received

                if output.itl:
                    output.tpot = sum(output.itl) / len(output.itl)

    except Exception:
        import traceback, sys
        output.success = False
        output.error = "".join(traceback.format_exception(*sys.exc_info()))

    if pbar:
        pbar.update(1)

    return output