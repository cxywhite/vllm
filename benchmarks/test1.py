async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    import pdb
    api_url = request_func_input.api_url
    assert api_url.endswith(("chat/completions", "profile")), (
        "OpenAI Chat Completions API URL must end with 'chat/completions'."
    )

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        if request_func_input.multi_modal_content:
            mm_content = request_func_input.multi_modal_content
            if isinstance(mm_content, list):
                content.extend(mm_content)
            elif isinstance(mm_content, dict):
                content.append(mm_content)
            else:
                raise TypeError(
                    "multi_modal_content must be a dict or list[dict] for openai-chat"
                )
        payload = {
            "model": request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model,
            "messages": [
                {"role": "user", "content": content},
            ],
            "temperature": request_func_input.extra_body.get("temperature", 0.7),
            "top_p": request_func_input.extra_body.get("top_p", 1.0),
            "top_k": request_func_input.extra_body.get("top_k", 20),
            "repetition_penalty": request_func_input.extra_body.get("repetition_penalty", 1.0),
            "max_completion_tokens": request_func_input.extra_body.get("max_tokens", request_func_input.output_len),
            "stream": True,
            "stream_options": {
                "include_usage": True,
            },
        }
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }
        if request_func_input.request_id:
            headers["x-request-id"] = request_func_input.request_id
        if request_func_input.req_id:
            headers["x-req-id"] = str(request_func_input.req_id)
        output = RequestFuncOutput()
        output.req_id = request_func_input.req_id
        output.prompt = request_func_input.prompt
        output.prompt_len = request_func_input.prompt_len
        output.expect_output_len = request_func_input.output_len
        output.temperature = request_func_input.extra_body.get("temperature", 0.7)
        output.top_p = request_func_input.extra_body.get("top_p", 1.0)
        output.top_k = request_func_input.extra_body.get("top_k", 20)
        output.repetition_penalty = request_func_input.extra_body.get("repetition_penalty", 1.0)
        generated_text = ""
        ttft = 0.0
        st = time.perf_counter()
        most_recent_timestamp = st
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    async for chunk_bytes in response.content:
                        chunk_bytes = chunk_bytes.strip()
                        if not chunk_bytes:
                            continue
                        chunk_bytes = chunk_bytes.decode("utf-8")
                        # NOTE: SSE comments (often used as pings) start with a colon.
                        # These are not JSON data payload and should be skipped.
                        if chunk_bytes.startswith(":"):
                            continue

                        chunk = chunk_bytes.removeprefix("data: ")

                        if chunk != "[DONE]":
                            timestamp = time.perf_counter()
                            data = json.loads(chunk)

                            if choices := data.get("choices"):
                                content = choices[0]["delta"].get("content")
                                # First token
                                if ttft == 0.0:
                                    ttft = timestamp - st
                                    output.ttft = ttft

                                # Decoding phase
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)

                                generated_text += content or ""
                            elif usage := data.get("usage"):
                                output.output_tokens = usage.get("completion_tokens")

                            most_recent_timestamp = timestamp

                    output.generated_text = generated_text
                    output.success = True
                    output.latency = most_recent_timestamp - st
                else:
                    output.error = response.reason or ""
                    output.success = False
                if output.itl:
                    output.tpot = sum(output.itl) / len(output.itl)
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

    if pbar:
        pbar.update(1)
    return output