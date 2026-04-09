# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket
import threading
import time
import uuid
from typing import Any,Dict, List
from contextlib import asynccontextmanager
import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request
import json
import asyncio
from dataclasses import dataclass
import httpx
import re
import os
import sys
from typing import Any, Dict, List, Optional
count = 0
prefill_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)
decode_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)

prefill_cv = threading.Condition()
decode_cv = threading.Condition()

DEFAULT_PING_SECONDS = 5
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
PREFILL_TIMEOUT_SECONDS = 600.0
PREFILL_INFLIGHT_LIMIT = int(os.environ.get("PREFILL_INFLIGHT_LIMIT", "64"))
PREFILL_QUEUE_TIMEOUT_SECONDS = float(os.environ.get("PREFILL_QUEUE_TIMEOUT_SECONDS", "0"))

_request_counter_lock = threading.Lock()
_prefill_semaphore: asyncio.Semaphore | None = None
_shared_http_session: aiohttp.ClientSession | None = None

app = Quart(__name__)


class PrefillQueueTimeoutError(RuntimeError):
    pass


@dataclass
class ModelConfig:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    bytes_per_param: int = 2


@dataclass
class DecodeLoadModel:
    config: ModelConfig
    precision_bytes: int
    peak_tflops: float
    bandwidth_gbs: float
    mem_capacity_gb: float
    tpot: float

    def __post_init__(self):
        c = self.config
        H = c.hidden_size
        I = c.intermediate_size
        V = c.vocab_size
        L = c.num_hidden_layers
        self.kv_dim = (H // c.num_attention_heads) * c.num_key_value_heads
        self.layer_params = (H * (H + 2 * self.kv_dim) + H * H + 3 * H * I)
        self.total_params = L * self.layer_params + V * H
        self.weights_vram = self.total_params * self.precision_bytes
        self.linear_flops_coeff = 2 * (L * self.layer_params + V * H)
        self.attn_flops_coeff = 4 * L * H

# =========================
# Decode Monitor
# =========================

class DecodeMonitor:
    def __init__(
        self,
        config_path: str,
        precision: str = "bfloat16",
        tpot: float = 50.0,
        check_interval: float = 0.01,
        enable_monitor_log: bool = True,
    ):
        self.check_interval = check_interval
        self.enable_monitor_log = enable_monitor_log
        self.stats: Dict[str, dict] = {}
        self.latest_load: Dict[str, dict] = {}
        self.is_running = True
        self.config = self._load_config(config_path)
        self.precision = precision.lower()

        precision_map = {
            "int8": 1,
            "float8": 1,
            "float16": 2,
            "bfloat16": 2,
            "float32": 4,
        }
        self.precision_bytes = precision_map.get(self.precision, 2)
        self.config.bytes_per_param = self.precision_bytes

        self.bandwidth_gbs = 2039
        self.mem_capacity = 80
        self.tpot = float(tpot)

        if self.precision == "float32":
            peak = 19.5
        elif self.precision in ("float16", "bfloat16"):
            peak = 312
        else:
            peak = 624

        self.load_model = DecodeLoadModel(
            config=self.config,
            precision_bytes=self.precision_bytes,
            peak_tflops=peak,
            bandwidth_gbs=self.bandwidth_gbs,
            mem_capacity_gb=self.mem_capacity,
            tpot=self.tpot,
        )
        self._last_log_time = 0.0
        self.log_interval = 1.0

    def _load_config(self, path: str) -> ModelConfig:
        try:
            with open(path, "r") as f:
                raw = json.load(f)
        except Exception:
            raw = {}
        return ModelConfig(
            hidden_size=raw.get("hidden_size", 4096),
            num_hidden_layers=raw.get("num_hidden_layers", 32),
            vocab_size=raw.get("vocab_size", 32000),
            intermediate_size=raw.get("intermediate_size", 11008),
            num_attention_heads=raw.get("num_attention_heads", 32),
            num_key_value_heads=raw.get("num_key_value_heads", raw.get("num_attention_heads", 32)),
        )

    async def fetch_metrics(self):
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=50)
        async with httpx.AsyncClient(timeout=httpx.Timeout(0.5), limits=limits) as client:
            while self.is_running:
                urls = list(decode_instances.keys())
                if urls:
                    await asyncio.gather(*[self._update_instance(client, u) for u in urls], return_exceptions=True)
                    self._compute_all_loads()
                await asyncio.sleep(self.check_interval)

    async def _update_instance(self, client: httpx.AsyncClient, addr: str):
        try:
            r = await client.get(f"http://{addr}/metrics")
            if r.status_code != 200: return
            def p(name):
                m = re.search(rf"{name}\{{.*?\}}\s+([\d\.e\+]+)", r.text)
                return float(m.group(1)) if m else 0.0
            
            self.stats[addr] = {
                "kv_usage": p("vllm:kv_cache_usage_perc"),
                "running_tokens": int(p("vllm:running_tokens")),
                "running_requests": int(p("vllm:num_requests_running")),
                "status": "healthy",
                "dirty": True,
            }
        except Exception:
            self.stats.setdefault(addr, {})["status"] = "unhealthy"
    def _compute_all_loads(self):
        now = time.time()
        do_log = self.enable_monitor_log and (now - self._last_log_time >= self.log_interval)
        for addr, stat in self.stats.items():
            if stat.get("status") != "healthy" or not stat.get("dirty"): continue
            stat["dirty"] = False
            load = self._calculate_load(
                stat["running_requests"],
                stat["running_tokens"],
                stat["kv_usage"],
            )
            self.latest_load[addr] = load

            if do_log:
                print(
                    f"[WT][MONITOR] decode={addr} "
                    f"requests={stat['running_requests']} "
                    f"tokens={stat['running_tokens']} "
                    f"bottle={load['Load_Bottle']:.3f} "
                    f"compute={load['Load_Compute']:.3f} "
                    f"memory={load['Load_Memory']:.3f} "
                    f"capacity={load['Load_Memory_Capacity']:.3f}",
                    flush=True,
                )
        if do_log:
            self._last_log_time = now
    def _calculate_load(self, n_req, m_tok, kv_usage):
        lm = self.load_model
        
        # 统一单位：每毫秒能读取的字节数 (GB/s -> Bytes/ms)
        bw_bytes_ms = (lm.bandwidth_gbs * 1e9) / 1000

        # 1. KV Cache 访存耗时 (动态开销: 读取当前所有Token的历史KV)
        kv_vram_bytes = (
            2 * lm.config.num_hidden_layers 
            * m_tok 
            * lm.kv_dim 
            * lm.precision_bytes
        )
        t_kv_io = kv_vram_bytes / bw_bytes_ms

        # 2. 计算耗时 (TFLOPS 维度)
        flops = (n_req * lm.linear_flops_coeff + lm.attn_flops_coeff * m_tok)
        t_compute = (flops / 1e12) / lm.peak_tflops * 1000

        # 3. 归一化负载
        lc = t_compute / lm.tpot

        # 按要求将权重项从分子移到分母:
        # Load_Memory = kv_vram_bytes / (bw_bytes_ms * tpot - weights_vram)
        memory_denominator = bw_bytes_ms * lm.tpot - lm.weights_vram
        lm_ = (
            kv_vram_bytes / memory_denominator
            if memory_denominator > 0
            else float("inf")
        )
        lcap = kv_usage
        return {
            "Load_Compute": lc,
            "Load_Memory": lm_,
            "Load_Memory_Capacity": lcap,
            "Load_Bottle": max(lc, lm_, lcap),
        }

    # def select_best_instance(self, instances):
    #     best_addr,best_zmq_addr, best_load = None,None, float("inf")
    #     for addr, zmq_addr in instances:
    #         load = self.latest_load.get(addr)
    #         if not load:
    #             continue
    #         if load["Load_Bottle"] < best_load:
    #             best_addr = addr
    #             best_load = load["Load_Bottle"]
    #             best_zmq_addr = zmq_addr[0]
    #     return best_addr,best_zmq_addr, best_load

monitor = DecodeMonitor(
    config_path=os.environ.get("MODEL_CONFIG_PATH", "model_config.json"),
    precision=os.environ.get("VLLM_DTYPE", "bf16"),
    tpot=os.environ.get("TPOT", 'tpot:50').split(':')[1],
    check_interval=0.01,
    enable_monitor_log=True,
)


@app.before_serving
async def start_monitor():
    global _prefill_semaphore
    global _shared_http_session

    _prefill_semaphore = asyncio.Semaphore(max(1, PREFILL_INFLIGHT_LIMIT))
    connector = aiohttp.TCPConnector(limit=2048, limit_per_host=1024, enable_cleanup_closed=True)
    _shared_http_session = aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT, connector=connector)

    print(
        f"[WT][PROXY] prefill_inflight_limit={max(1, PREFILL_INFLIGHT_LIMIT)} "
        f"prefill_queue_timeout_s={PREFILL_QUEUE_TIMEOUT_SECONDS:.1f} "
        f"prefill_timeout_s={PREFILL_TIMEOUT_SECONDS:.1f}",
        flush=True,
    )
    asyncio.create_task(monitor.fetch_metrics())


@app.after_serving
async def stop_monitor():
    global _shared_http_session
    if _shared_http_session is not None and not _shared_http_session.closed:
        await _shared_http_session.close()
    _shared_http_session = None


@asynccontextmanager
async def _acquire_prefill_slot(request_id: str):
    if _prefill_semaphore is None:
        raise RuntimeError("Prefill semaphore is not initialized")

    queue_start = time.perf_counter()
    try:
        if PREFILL_QUEUE_TIMEOUT_SECONDS > 0:
            await asyncio.wait_for(
                _prefill_semaphore.acquire(),
                timeout=PREFILL_QUEUE_TIMEOUT_SECONDS,
            )
        else:
            await _prefill_semaphore.acquire()
    except asyncio.TimeoutError as e:
        waited = time.perf_counter() - queue_start
        raise PrefillQueueTimeoutError(
            f"Prefill queue wait timeout after {waited:.2f}s, request_id={request_id}"
        ) from e

    waited = time.perf_counter() - queue_start
    if waited >= 1.0:
        print(
            f"[WT][PREFILL_QUEUE] request_id={request_id} waited_s={waited:.2f}",
            flush=True,
        )

    try:
        yield
    finally:
        _prefill_semaphore.release()

def _remove_expired_instances(instances: dict[str, Any]) -> None:
    now = time.time()
    expired_keys = [k for k, v in instances.items() if v[1] <= now]
    for key in expired_keys:
        value = instances.get(key)
        if value is not None:
            print(f"🔴Remove [HTTP:{key}, ZMQ:{value[0]}, stamp:{value[1]}]")
            instances.pop(key, None)


def _drop_instance(
    cv: threading.Condition,
    instances: dict[str, Any],
    http_addr: str,
    reason: str,
) -> None:
    with cv:
        value = instances.pop(http_addr, None)
        if value is not None:
            print(f"[WARN] Drop instance HTTP:{http_addr}, ZMQ:{value[0]}, reason={reason}")
            cv.notify_all()


def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            # data: {"type": "P", "http_address": "ip:port",
            #        "zmq_address": "ip:port"}
            data = msgpack.loads(message)
            if data["type"] == "P":
                global prefill_instances
                global prefill_cv
                with prefill_cv:
                    node = prefill_instances.get(data["http_address"], None)
                    prefill_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_expired_instances(prefill_instances)
                    prefill_cv.notify_all()

            elif data["type"] == "D":
                global decode_instances
                global decode_cv
                with decode_cv:
                    node = decode_instances.get(data["http_address"], None)
                    decode_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_expired_instances(decode_instances)
                    decode_cv.notify_all()
            else:
                print(
                    "Unexpected, Received message from %s, data: %s",
                    remote_address,
                    data,
                )
                return

            if node is None:
                print(f"🔵Add [HTTP:{data['http_address']}, ZMQ:{data['zmq_address']}]")


def start_service_discovery(hostname, port):
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    _listener_thread = threading.Thread(
        target=_listen_for_register, args=[poller, router_socket], daemon=True
    )
    _listener_thread.start()
    return _listener_thread

def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def _pick_instance_with_wait(
    cv: threading.Condition,
    instances: dict[str, Any],
    role: str,
    timeout_s: float = 60.0,
) -> tuple[str, str]:
    deadline = time.time() + timeout_s
    while True:
        with cv:
            _remove_expired_instances(instances)
            items = list(instances.items())
            if items:
                # Use global round-robin counter to keep pair selection behavior.
                idx = count % len(items)
                http_addr, zmq_info = items[idx]
                return http_addr, zmq_info[0]

            remaining = deadline - time.time()
            if remaining <= 0:
                raise RuntimeError(f"No available {role} instances within {timeout_s:.1f}s")

            cv.wait(timeout=min(0.2, remaining))


async def forward_request(url, data, request_id):
    if _shared_http_session is None or _shared_http_session.closed:
        raise RuntimeError("Shared HTTP session is unavailable")

    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }
    async with _shared_http_session.post(url=url, json=data, headers=headers) as response:
        if response.status == 200:
            async for chunk_bytes in response.content.iter_chunked(1024):
                yield chunk_bytes
        else:
            err_msg = await response.text()
            raise RuntimeError(
                f"Upstream returned {response.status} for {url}: {err_msg[:400]}"
            )


async def _drain_stream(stream):
    async for _ in stream:
        continue

# [WT] 2026-01-24 07:55:08
# 修改pd分离ttft计算逻辑,在prefill完从proxy返回给benchmark时间戳
# @app.route("/v1/completions", methods=["POST"])
# @app.route("/v1/chat/completions", methods=["POST"])
# async def handle_request():
#     try:
#         import time

#         original_request_data = await request.get_json()
#         original_request_data["predictor_meta"] = None

#         # ---------- Prefill request ----------
#         prefill_request = original_request_data.copy()
#         prefill_request["max_tokens"] = 1
#         prefill_request["stream"] = True   # ⚠️ 必须是 True

#         if "max_completion_tokens" in prefill_request:
#             prefill_request["max_completion_tokens"] = 1

#         global count, prefill_instances, prefill_cv
#         global decode_instances, decode_cv

#         with prefill_cv:
#             prefill_list = list(prefill_instances.items())
#             prefill_addr, prefill_zmq_addr = prefill_list[count % len(prefill_list)]
#             prefill_zmq_addr = prefill_zmq_addr[0]

#         with decode_cv:
#             decode_list = list(decode_instances.items())
#             decode_addr, decode_zmq_addr = decode_list[count % len(decode_list)]
#             decode_zmq_addr = decode_zmq_addr[0]

#         request_id = (
#             f"___prefill_addr_{prefill_zmq_addr}"
#             f"___decode_addr_{decode_zmq_addr}_{random_uuid()}"
#         )

#         print(
#             f"handle_request count={count} "
#             f"[prefill:{prefill_addr}] -> [decode:{decode_addr}]"
#         )
#         count += 1

#         # ---------- Prefill phase ----------
#         prefill_start_ts = time.perf_counter()

#         async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
#             async with session.post(
#                 f"http://{prefill_addr}{request.path}",
#                 json=prefill_request,
#                 headers={
#                     "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
#                     "X-Request-Id": request_id,
#                 },
#             ) as resp:
#                 if resp.status != 200:
#                     text = await resp.text()
#                     raise RuntimeError(f"Prefill failed: {resp.status}, {text}")

#                 # ⚠️ 必须把 stream 消耗完
#                 async for _ in resp.content:
#                     pass

#         prefill_done_ts = time.perf_counter()

#         # ---------- Decode phase (streaming) ----------
#         generator = forward_request(
#             f"http://{decode_addr}{request.path}",
#             original_request_data,
#             request_id,
#         )

#         response = await make_response(generator)
#         response.timeout = None

#         # ⭐ 传给 benchmark
#         response.headers["x-prefill-done-ts"] = str(prefill_done_ts)

#         return response

#     except Exception as e:
#         import sys, traceback
#         print("Error occurred in disagg prefill proxy server")
#         print(e)
#         print("".join(traceback.format_exception(*sys.exc_info())))
#         return await make_response(str(e), 500)


# 原始过程
@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()
        original_request_data["predictor_meta"] = None
        prefill_request = original_request_data.copy()
        prefill_request["max_tokens"] = 1
        # Prefill is only used to materialize KV; non-streaming avoids
        # long-lived SSE responses blocking the proxy request lifecycle.
        prefill_request["stream"] = False
        # OpenAI-compatible servers reject stream_options when stream is False.
        prefill_request.pop("stream_options", None)
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1
        global count
        global prefill_instances
        global prefill_cv
        global decode_instances
        global decode_cv
        global _request_counter_lock
        last_prefill_error: Exception | None = None
        selected_prefill_addr = ""
        selected_prefill_zmq_addr = ""
        selected_decode_addr = ""
        selected_decode_zmq_addr = ""
        request_id = ""

        with _request_counter_lock:
            request_no = count
            count += 1

        for _attempt in range(3):
            selected_prefill_addr, selected_prefill_zmq_addr = _pick_instance_with_wait(
                prefill_cv,
                prefill_instances,
                "prefill",
            )
            selected_decode_addr, selected_decode_zmq_addr = _pick_instance_with_wait(
                decode_cv,
                decode_instances,
                "decode",
            )
            request_id = (
                f"___prefill_addr_{selected_prefill_zmq_addr}___decode_addr_"
                f"{selected_decode_zmq_addr}_{random_uuid()}"
            )

            print(
                f"handle_request req_no:{request_no} attempt:{_attempt + 1}/3 "
                f"[HTTP:{selected_prefill_addr}, "
                f"ZMQ:{selected_prefill_zmq_addr}] -> [HTTP:{selected_decode_addr}, "
                f"ZMQ:{selected_decode_zmq_addr}] request_id={request_id}",
                flush=True,
            )

            try:
                prefill_start = time.perf_counter()
                async with _acquire_prefill_slot(request_id):
                    await asyncio.wait_for(
                        _drain_stream(
                            forward_request(
                                f"http://{selected_prefill_addr}{request.path}",
                                prefill_request,
                                request_id,
                            )),
                        timeout=PREFILL_TIMEOUT_SECONDS,
                    )
                prefill_ms = (time.perf_counter() - prefill_start) * 1000.0
                print(
                    f"[WT][PREFILL] req_no={request_no} request_id={request_id} "
                    f"done_ms={prefill_ms:.1f}",
                    flush=True,
                )
                last_prefill_error = None
                break
            except RuntimeError as conn_err:
                # Request-level 4xx usually indicates invalid input rather
                # than an unhealthy prefill instance.
                if "Upstream returned 4" in str(conn_err):
                    raise
                last_prefill_error = conn_err
                print(
                    f"[WARN] prefill runtime error req_no={request_no} "
                    f"request_id={request_id} type={type(conn_err).__name__} "
                    f"err={conn_err!r}",
                    flush=True,
                )
                _drop_instance(
                    prefill_cv,
                    prefill_instances,
                    selected_prefill_addr,
                    f"prefill-connect-failed: {conn_err}",
                )
            except PrefillQueueTimeoutError as queue_err:
                last_prefill_error = queue_err
                print(
                    f"[WARN] prefill queue timeout req_no={request_no} "
                    f"request_id={request_id} type={type(queue_err).__name__} "
                    f"err={queue_err!r}",
                    flush=True,
                )
            except (aiohttp.ClientError, asyncio.TimeoutError) as conn_err:
                last_prefill_error = conn_err
                print(
                    f"[WARN] prefill connection/timeout req_no={request_no} "
                    f"request_id={request_id} type={type(conn_err).__name__} "
                    f"err={conn_err!r}",
                    flush=True,
                )
                _drop_instance(
                    prefill_cv,
                    prefill_instances,
                    selected_prefill_addr,
                    f"prefill-connect-failed: {conn_err}",
                )

        if last_prefill_error is not None:
            raise RuntimeError(
                f"Prefill connection failed after retries: {last_prefill_error}"
            )

        # return decode
        generator = forward_request(
            f"http://{selected_decode_addr}{request.path}",
            original_request_data,
            request_id,
        )
        response = await make_response(generator)
        response.timeout = None
        return response
    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        return await make_response(str(e), 500)



if __name__ == "__main__":
    proxy_port=os.environ.get("PROXY_PORT", "28001")
    bench_port=os.environ.get("BENCH_PORT", "22006")
    t = start_service_discovery("0.0.0.0", proxy_port)
    app.run(host="0.0.0.0", port=bench_port)
    t.join()