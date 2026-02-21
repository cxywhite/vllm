# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import io
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional, Union

import aiohttp
import huggingface_hub.constants
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

# NOTE(simon): do not import vLLM here so the benchmark script
# can run without vLLM installed.

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    prompt_len: int
    output_len: int
    model: str
    model_name: Optional[str] = None
    logprobs: Optional[int] = None
    extra_body: Optional[dict] = None
    multi_modal_content: Optional[dict | list[dict]] = None
    ignore_eos: bool = False
    language: Optional[str] = None
    # wt
    # temperature: Optional[float] = None
    # top_p: Optional[float] = None
    # top_k: Optional[int] = None
    # repetition_penalty: Optional[float] = None
    req_id: Optional[str] = None


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    output_tokens: int = 0
    ttft: float = 0.0  # Time to first token
    itl: list[float] = field(default_factory=list)  # list of inter-token latencies
    tpot: float = 0.0  # avg next-token latencies
    prompt_len: int = 0
    error: str = ""
    req_id:Optional[str] =None
    prompt:Optional[str] =None
    expect_output_len:Optional[int] =None
    temperature:Optional[float] =None
    top_p:Optional[float] =None
    top_k:Optional[int] =None
    repetition_penalty:Optional[float] =None

async def async_request_tgi(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        params = {
            "max_new_tokens": request_func_input.output_len,
            "do_sample": True,
            "temperature": 0.01,  # TGI does not accept 0.0 temperature.
            "top_p": 0.99,  # TGI does not accept 1.0 top_p.
            "truncate": request_func_input.prompt_len,
            "ignore_eos_token": request_func_input.ignore_eos,
        }
        payload = {
            "inputs": request_func_input.prompt,
            "parameters": params,
        }
        headers = None
        if request_func_input.request_id:
            headers = {"x-request-id": request_func_input.request_id}
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        if request_func_input.ignore_eos:
            output.output_tokens = request_func_input.output_len
        else:
            output.output_tokens = None

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

                        # NOTE: Sometimes TGI returns a ping response without
                        # any data, we should skip it.
                        if chunk_bytes.startswith(":"):
                            continue
                        chunk = chunk_bytes.removeprefix("data:")

                        data = json.loads(chunk)
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = time.perf_counter() - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True
                    output.generated_text = data["generated_text"]
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_trt_llm(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith("generate_stream")

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input.prompt,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": request_func_input.output_len,
            "stream": True,
        }
        if request_func_input.ignore_eos:
            payload["min_length"] = request_func_input.output_len
        headers = None
        if request_func_input.request_id:
            headers = {"x-request-id": request_func_input.request_id}
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

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

                        chunk = chunk_bytes.decode("utf-8").removeprefix("data:")

                        data = json.loads(chunk)
                        output.generated_text += data["text_output"]
                        timestamp = time.perf_counter()
                        # First token
                        if ttft == 0.0:
                            ttft = timestamp - st
                            output.ttft = ttft

                        # Decoding phase
                        else:
                            output.itl.append(timestamp - most_recent_timestamp)

                        most_recent_timestamp = timestamp

                    output.latency = most_recent_timestamp - st
                    output.success = True

                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


async def async_request_deepspeed_mii(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url
    assert api_url.endswith(("completions", "profile")), (
        "OpenAI Completions API URL must end with 'completions' or 'profile'."
    )

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        payload = {
            "model": request_func_input.model,
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.output_len,
            "temperature": 0.01,  # deepspeed-mii does not accept 0.0 temp.
            "top_p": 1.0,
        }
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
        if request_func_input.request_id:
            headers["x-request-id"] = request_func_input.request_id

        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len

        # NOTE: DeepSpeed-MII doesn't support streaming as of Jan 28 2024,
        # will use 0 as placeholder.
        # See https://github.com/microsoft/DeepSpeed-MII/pull/311
        output.ttft = 0

        st = time.perf_counter()
        try:
            async with session.post(
                url=api_url, json=payload, headers=headers
            ) as response:
                if response.status == 200:
                    parsed_resp = await response.json()
                    output.latency = time.perf_counter() - st
                    if "choices" in parsed_resp:
                        output.generated_text = parsed_resp["choices"][0]["text"]
                    elif "text" in parsed_resp:
                        output.generated_text = parsed_resp["text"][0]
                    else:
                        output.error = (
                            "Unexpected response format: "
                            "neither 'choices' nor 'text' found"
                        )
                        output.success = False
                    output.success = True
                else:
                    output.error = response.reason or ""
                    output.success = False
        except Exception:
            output.success = False
            exc_info = sys.exc_info()
            output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output

import json
import time
import aiohttp
import traceback
import sys
import os
from typing import Optional

# 对应 vLLM 的 RequestFuncInput 和 RequestFuncOutput 类定义（假设已导入）

async def async_request_openai_completions(
    request_func_input,
    pbar: Optional[object] = None,
) -> object:
    api_url = request_func_input.api_url
    
    # 客户端也需要设置长超时，connect 超时设长是为了应对 vLLM 队列满载的情况
    timeout = aiohttp.ClientTimeout(total=None, connect=60, sock_read=None)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        payload = {
            "model": request_func_input.model_name or request_func_input.model,
            "prompt": request_func_input.prompt,
            "max_tokens": request_func_input.output_len,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        
        # 合并采样参数
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        
        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos

        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}"}
        if request_func_input.req_id:
            headers["x-request-id"] = str(request_func_input.req_id)
        output = RequestFuncOutput()
        output.req_id = request_func_input.req_id
        output.prompt= request_func_input.prompt
        generated_text = ""
        st = time.perf_counter()
        most_recent_timestamp = st
        first_chunk_received = False

        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    output.error = f"HTTP {response.status}: {await response.text()}"
                    output.success = False
                    return output

                # aiohttp 的 response.content 迭代默认是按行读取，非常适合 SSE
                async for line_bytes in response.content:
                    line = line_bytes.decode("utf-8").strip()
                    if not line or line == "data: [DONE]":
                        continue
                    
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                        except json.JSONDecodeError:
                            continue # 忽略不完整的 JSON 块

                        # 统计 Token 生成
                        if choices := data.get("choices"):
                            if choices[0].get("text"):
                                chunk_text = choices[0]["text"]
                                timestamp = time.perf_counter()
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    output.ttft = timestamp - st
                                else:
                                    output.itl.append(timestamp - most_recent_timestamp)
                                
                                most_recent_timestamp = timestamp
                                generated_text += chunk_text

                        # 关键：获取最终的输出长度
                        if usage := data.get("usage"):
                            output.output_tokens = usage.get("completion_tokens")

                output.generated_text = generated_text
                output.success = first_chunk_received
                output.latency = time.perf_counter() - st
                
        except Exception:
            output.success = False
            output.error = traceback.format_exc()

    if pbar:
        pbar.update(1)
    return output



async def async_request_openai_chat_completions(
    request_func_input,
    pbar: Optional[object] = None,
) -> object:
    api_url = request_func_input.api_url
    assert api_url.endswith(("chat/completions", "profile")), (
        "OpenAI Chat Completions API URL must end with 'chat/completions'."
    )

    # 1. 消除超时限制：connect 设为 180s 应对排队，total 和 sock_read 设为 None 允许无限时生成
    timeout = aiohttp.ClientTimeout(total=None, connect=180, sock_read=None)

    async with aiohttp.ClientSession(
        trust_env=True, timeout=timeout
    ) as session:
        # 构建多模态内容
        # content = [{"type": "text", "text": request_func_input.prompt}]
        # if request_func_input.multi_modal_content:
        #     mm_content = request_func_input.multi_modal_content
        #     if isinstance(mm_content, list):
        #         content.extend(mm_content)
        #     elif isinstance(mm_content, dict):
        #         content.append(mm_content)
        #     else:
        #         raise TypeError(
        #             "multi_modal_content must be a dict or list[dict] for openai-chat"
        #         )
        content = request_func_input.prompt
        payload = {
            "model": request_func_input.model_name if request_func_input.model_name else request_func_input.model,
            "messages": [{"role": "user", "content": content}],
            "max_completion_tokens": request_func_input.output_len,
            "stream": True,
            "stream_options": {"include_usage": True}, # 必须包含以获取准确 token 数
        }

        if request_func_input.ignore_eos:
            payload["ignore_eos"] = request_func_input.ignore_eos
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', 'EMPTY')}",
        }
        if request_func_input.req_id:
            headers["x-request-id"] = str(request_func_input.req_id)
        output = RequestFuncOutput()
        output.prompt_len = request_func_input.prompt_len
        output.req_id = request_func_input.req_id
        output.prompt= request_func_input.prompt
        output.temperature= request_func_input.extra_body.get('temperature') if request_func_input.extra_body else None
        output.top_p= request_func_input.extra_body.get('top_p') if request_func_input.extra_body else None
        output.top_k= request_func_input.extra_body.get('top_k') if request_func_input.extra_body else None
        output.repetition_penalty= request_func_input.extra_body.get('repetition_penalty') if request_func_input.extra_body else None
        generated_text = ""
        first_chunk_received = False
        st = time.perf_counter()
        most_recent_timestamp = st

        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    output.error = f"HTTP {response.status}: {await response.text()}"
                    output.success = False
                    return output

                # 2. 健壮的 SSE 流解析
                # aiohttp 的 response.content 会按行产生数据，非常适合处理 data: 格式
                async for line_bytes in response.content:
                    line = line_bytes.decode("utf-8").strip()
                    if not line or line.startswith(":"): # 跳过空行和 SSE 评论(ping)
                        continue

                    if line == "data: [DONE]":
                        break

                    if line.startswith("data: "):
                        data_str = line.removeprefix("data: ")
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue # 忽略不完整的 JSON

                        timestamp = time.perf_counter()

                        # 处理文本内容
                        if choices := data.get("choices"):
                            delta = choices[0].get("delta", {})
                            if content_piece := delta.get("content"):
                                # 计算 TTFT (First Token)
                                if not first_chunk_received:
                                    first_chunk_received = True
                                    output.ttft = timestamp - st
                                else:
                                    # 计算 ITL (Inter-token Latency)
                                    output.itl.append(timestamp - most_recent_timestamp)
                                
                                generated_text += content_piece
                                most_recent_timestamp = timestamp

                        # 3. 核心：获取 Usage 统计 (生成结束时的 token 总数)
                        if usage := data.get("usage"):
                            output.output_tokens = usage.get("completion_tokens")

                output.generated_text = generated_text
                output.success = first_chunk_received # 如果收到了有效内容则视为成功
                output.latency = time.perf_counter() - st

        except Exception:
            output.success = False
            output.error = traceback.format_exc()

    if pbar:
        pbar.update(1)
    return output

# async def async_request_openai_completions(
#     request_func_input: RequestFuncInput,
#     pbar: Optional[tqdm] = None,
# ) -> RequestFuncOutput:
#     api_url = request_func_input.api_url
#     assert api_url.endswith(("completions", "profile")), (
#         "OpenAI Completions API URL must end with 'completions' or 'profile'."
#     )

#     async with aiohttp.ClientSession(
#         trust_env=True, timeout=AIOHTTP_TIMEOUT
#     ) as session:
#         # 如果想用benchmark_serving.py测试，改采样参数，直接修改这里没有用，必须在执行benchmark_serving.py中直接设置采样参数
#         payload = {
#             "req_id":request_func_input.req_id,
#             "model": request_func_input.model_name
#             if request_func_input.model_name
#             else request_func_input.model,
#             "prompt": request_func_input.prompt,
#             "temperature": request_func_input.extra_body['temperature'],
#             "repetition_penalty": request_func_input.extra_body['repetition_penalty'],
#             "top_k": request_func_input.extra_body['top_k'],
#             "top_p": request_func_input.extra_body['top_p'],
#             "max_tokens": request_func_input.output_len,
#             "logprobs": request_func_input.logprobs,
#             "stream": True,
#             "stream_options": {
#                 "include_usage": True,
#             },
#         }
#         if request_func_input.ignore_eos:
#             payload["ignore_eos"] = request_func_input.ignore_eos
#         if request_func_input.extra_body:
#             payload.update(request_func_input.extra_body)
#         headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"}
#         if request_func_input.req_id:
#             headers["x-request-id"] = request_func_input.req_id

#         output = RequestFuncOutput()
        
#         output.prompt_len = request_func_input.prompt_len
#         output.req_id=request_func_input.req_id
#         output.prompt=request_func_input.prompt
#         output.expect_output_len=request_func_input.output_len
#         generated_text = ""
#         st = time.perf_counter()
#         most_recent_timestamp = st
#         try:
#             async with session.post(
#                 url=api_url, json=payload, headers=headers
#             ) as response:
#                 if response.status == 200:
#                     first_chunk_received = False
#                     async for chunk_bytes in response.content:
#                         chunk_bytes = chunk_bytes.strip()
#                         if not chunk_bytes:
#                             continue

#                         chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
#                         # print(f'chunk: {chunk}')
#                         if chunk != "[DONE]":
#                             data = json.loads(chunk)

#                             # NOTE: Some completion API might have a last
#                             # usage summary response without a token so we
#                             # want to check a token was generated
#                             if choices := data.get("choices"):
#                                 # Note that text could be empty here
#                                 # e.g. for special tokens
#                                 text = choices[0].get("text")
#                                 timestamp = time.perf_counter()
#                                 # First token
#                                 if not first_chunk_received:
#                                     first_chunk_received = True
#                                     ttft = time.perf_counter() - st
#                                     output.ttft = ttft

#                                 # Decoding phase
#                                 else:
#                                     output.itl.append(timestamp - most_recent_timestamp)

#                                 most_recent_timestamp = timestamp
#                                 generated_text += text or ""
#                             if usage := data.get("usage"):
#                                 output.output_tokens = usage.get("completion_tokens")
#                     if first_chunk_received:
#                         output.success = True
#                     else:
#                         output.success = False
#                         output.error = (
#                             "Never received a valid chunk to calculate TTFT."
#                             "This response will be marked as failed!"
#                         )
#                     output.generated_text = generated_text
#                     output.latency = most_recent_timestamp - st
#                 else:
#                     output.error = response.reason or ""
#                     output.success = False
#         except Exception:
#             output.success = False
#             exc_info = sys.exc_info()
#             output.error = "".join(traceback.format_exception(*exc_info))

#     if pbar:
#         pbar.update(1)
#     return output

# async def async_request_openai_chat_completions(
#     request_func_input: RequestFuncInput,
#     pbar: Optional[tqdm] = None,
# ) -> RequestFuncOutput:
#     api_url = request_func_input.api_url
#     assert api_url.endswith(("chat/completions", "profile")), (
#         "OpenAI Chat Completions API URL must end with 'chat/completions'."
#     )

#     async with aiohttp.ClientSession(
#         trust_env=True, timeout=AIOHTTP_TIMEOUT
#     ) as session:
#         content = [{"type": "text", "text": request_func_input.prompt}]
#         if request_func_input.multi_modal_content:
#             mm_content = request_func_input.multi_modal_content
#             if isinstance(mm_content, list):
#                 content.extend(mm_content)
#             elif isinstance(mm_content, dict):
#                 content.append(mm_content)
#             else:
#                 raise TypeError(
#                     "multi_modal_content must be a dict or list[dict] for openai-chat"
#                 )
#         payload = {
#             "model": request_func_input.model_name
#             if request_func_input.model_name
#             else request_func_input.model,
#             "messages": [
#                 {"role": "user", "content": content},
#             ],
#             "temperature": 0.0,
#             "max_completion_tokens": request_func_input.output_len,
#             "stream": True,
#             "stream_options": {
#                 "include_usage": True,
#             },
#         }
#         if request_func_input.ignore_eos:
#             payload["ignore_eos"] = request_func_input.ignore_eos
#         if request_func_input.extra_body:
#             payload.update(request_func_input.extra_body)
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
#         }
#         if request_func_input.request_id:
#             headers["x-request-id"] = request_func_input.request_id

#         output = RequestFuncOutput()
#         output.prompt_len = request_func_input.prompt_len

#         generated_text = ""
#         ttft = 0.0
#         st = time.perf_counter()
#         most_recent_timestamp = st
#         try:
#             async with session.post(
#                 url=api_url, json=payload, headers=headers
#             ) as response:
#                 if response.status == 200:
#                     async for chunk_bytes in response.content:
#                         chunk_bytes = chunk_bytes.strip()
#                         if not chunk_bytes:
#                             continue
#                         chunk_bytes = chunk_bytes.decode("utf-8")
#                         # NOTE: SSE comments (often used as pings) start with a colon.
#                         # These are not JSON data payload and should be skipped.
#                         if chunk_bytes.startswith(":"):
#                             continue

#                         chunk = chunk_bytes.removeprefix("data: ")

#                         if chunk != "[DONE]":
#                             timestamp = time.perf_counter()
#                             data = json.loads(chunk)

#                             if choices := data.get("choices"):
#                                 content = choices[0]["delta"].get("content")
#                                 # First token
#                                 if ttft == 0.0:
#                                     ttft = timestamp - st
#                                     output.ttft = ttft

#                                 # Decoding phase
#                                 else:
#                                     output.itl.append(timestamp - most_recent_timestamp)

#                                 generated_text += content or ""
#                             elif usage := data.get("usage"):
#                                 output.output_tokens = usage.get("completion_tokens")

#                             most_recent_timestamp = timestamp

#                     output.generated_text = generated_text
#                     output.success = True
#                     output.latency = most_recent_timestamp - st
#                 else:
#                     output.error = response.reason or ""
#                     output.success = False
#         except Exception:
#             output.success = False
#             exc_info = sys.exc_info()
#             output.error = "".join(traceback.format_exception(*exc_info))

#     if pbar:
#         pbar.update(1)
#     return output


async def async_request_openai_audio(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    # Lazy import without PlaceholderModule to avoid vllm dep.
    import soundfile

    api_url = request_func_input.api_url
    assert api_url.endswith(("transcriptions", "translations")), (
        "OpenAI Chat Completions API URL must end with 'transcriptions' "
    )
    "or `translations`."

    async with aiohttp.ClientSession(
        trust_env=True, timeout=AIOHTTP_TIMEOUT
    ) as session:
        content = [{"type": "text", "text": request_func_input.prompt}]
        payload = {
            "model": request_func_input.model_name
            if request_func_input.model_name
            else request_func_input.model,
            "temperature": 0.0,
            "max_completion_tokens": request_func_input.output_len,
            "stream": True,
            "language": "en",
            # Flattened due to multipart/form-data
            "stream_include_usage": True,
            "stream_continuous_usage_stats": True,
        }
        if request_func_input.extra_body:
            payload.update(request_func_input.extra_body)
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        }
        if request_func_input.request_id:
            headers["x-request-id"] = request_func_input.request_id

        # Send audio file
        def to_bytes(y, sr):
            buffer = io.BytesIO()
            soundfile.write(buffer, y, sr, format="WAV")
            buffer.seek(0)
            return buffer

        mm_audio = request_func_input.multi_modal_content
        if not isinstance(mm_audio, dict) or "audio" not in mm_audio:
            raise TypeError("multi_modal_content must be a dict containing 'audio'")
        with to_bytes(*mm_audio["audio"]) as f:
            form = aiohttp.FormData()
            form.add_field("file", f, content_type="audio/wav")
            for key, value in payload.items():
                form.add_field(key, str(value))

            output = RequestFuncOutput()
            output.prompt_len = request_func_input.prompt_len

            generated_text = ""
            ttft = 0.0
            st = time.perf_counter()
            most_recent_timestamp = st
            try:
                async with session.post(
                    url=api_url, data=form, headers=headers
                ) as response:
                    if response.status == 200:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue

                            chunk = chunk_bytes.decode("utf-8").removeprefix("data: ")
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
                                        output.itl.append(
                                            timestamp - most_recent_timestamp
                                        )

                                    generated_text += content or ""
                                elif usage := data.get("usage"):
                                    output.output_tokens = usage.get(
                                        "completion_tokens"
                                    )

                                most_recent_timestamp = timestamp

                        output.generated_text = generated_text
                        output.success = True
                        output.latency = most_recent_timestamp - st
                    else:
                        output.error = response.reason or ""
                        output.success = False
            except Exception:
                output.success = False
                exc_info = sys.exc_info()
                output.error = "".join(traceback.format_exception(*exc_info))

        if pbar:
            pbar.update(1)
        return output


def get_model(pretrained_model_name_or_path: str) -> str:
    if os.getenv("VLLM_USE_MODELSCOPE", "False").lower() == "true":
        from modelscope import snapshot_download

        from vllm.model_executor.model_loader.weight_utils import get_lock

        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(pretrained_model_name_or_path):
            model_path = snapshot_download(
                model_id=pretrained_model_name_or_path,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                ignore_file_pattern=[".*.pt", ".*.safetensors", ".*.bin"],
            )

            return model_path
    return pretrained_model_name_or_path


def get_tokenizer(
    pretrained_model_name_or_path: str,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    if pretrained_model_name_or_path is not None and not os.path.exists(
        pretrained_model_name_or_path
    ):
        pretrained_model_name_or_path = get_model(pretrained_model_name_or_path)
    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    if tokenizer_mode == "mistral":
        try:
            from vllm.transformers_utils.tokenizer import MistralTokenizer
        except ImportError as e:
            raise ImportError(
                "MistralTokenizer requires vllm package.\n"
                "Please install it with `pip install vllm` "
                "to use mistral tokenizer mode."
            ) from e
        return MistralTokenizer.from_pretrained(str(pretrained_model_name_or_path))
    else:
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )


ASYNC_REQUEST_FUNCS = {
    "tgi": async_request_tgi,
    "vllm": async_request_openai_completions,
    "lmdeploy": async_request_openai_completions,
    "deepspeed-mii": async_request_deepspeed_mii,
    "openai": async_request_openai_completions,
    "openai-chat": async_request_openai_chat_completions,
    "openai-audio": async_request_openai_audio,
    "tensorrt-llm": async_request_trt_llm,
    "scalellm": async_request_openai_completions,
    "sglang": async_request_openai_completions,
    "llama.cpp": async_request_openai_completions,
}

OPENAI_COMPATIBLE_BACKENDS = [
    k
    for k, v in ASYNC_REQUEST_FUNCS.items()
    if v in (async_request_openai_completions, async_request_openai_chat_completions)
]
