# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket
import threading
import time
import uuid
from typing import Any, Dict, List, Optional
import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request
import json
import asyncio
from dataclasses import dataclass
import httpx
import re

count = 0
prefill_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)
decode_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)

prefill_cv = threading.Condition()
decode_cv = threading.Condition()
latest_load_cv = threading.Condition()
DEFAULT_PING_SECONDS = 5
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
MAX_DECODE_LOAD_BOTTLE = float(os.environ.get("MAX_DECODE_LOAD_BOTTLE", "0.9"))
MAX_DECODE_KV_USAGE = float(os.environ.get("MAX_DECODE_KV_USAGE", "0.9"))
MAX_DECODE_FUTURE_TOKENS = int(os.environ.get("MAX_DECODE_FUTURE_TOKENS", "12000"))
MAX_PREDICT_OUTPUT_LEN = int(os.environ.get("MAX_PREDICT_OUTPUT_LEN", "2048"))
FALLBACK_PREDICT_OUTPUT_LEN = int(os.environ.get("FALLBACK_PREDICT_OUTPUT_LEN", "256"))
ROUTING_LOCK = asyncio.Lock()

from wt_metadata import Custom_Metadata
app = Quart(__name__)
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
        H, I, V, L = (
            c.hidden_size,
            c.intermediate_size,
            c.vocab_size,
            c.num_hidden_layers,
        )
        self.kv_dim = (H // c.num_attention_heads) * c.num_key_value_heads

        self.layer_params = (
            H * (H + 2 * self.kv_dim)
            + H * H
            + 3 * H * I
        )
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
        precision: str,
        tpot: float,
        check_interval: float,
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

        self.log_interval = 1.0
        self._last_log_time = 0.0

    def _load_config(self, path: str) -> ModelConfig:
        with open(path, "r") as f:
            raw = json.load(f)
        return ModelConfig(
            hidden_size=raw.get("hidden_size", 4096),
            num_hidden_layers=raw.get("num_hidden_layers", 32),
            vocab_size=raw.get("vocab_size", 32000),
            intermediate_size=raw.get("intermediate_size", 11008),
            num_attention_heads=raw.get("num_attention_heads", 32),
            num_key_value_heads=raw.get(
                "num_key_value_heads", raw.get("num_attention_heads", 32)
            ),
        )

    async def fetch_metrics(self):
        limits = httpx.Limits(
            max_connections=100,
            max_keepalive_connections=50,
        )
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(0.5),
            limits=limits,
        ) as client:
            while self.is_running:
                urls = list(decode_instances.keys())
                if not urls:
                    await asyncio.sleep(0.1)
                    continue

                await asyncio.gather(
                    *[self._update_instance(client, u) for u in urls]
                )
                self._compute_all_loads()
                await asyncio.sleep(self.check_interval)

    async def _update_instance(self, client: httpx.AsyncClient, addr: str):
        try:
            r = await client.get(f"http://{addr}/metrics")
            if r.status_code != 200:
                return

            def p(name):
                m = re.search(rf"{name}\{{.*?\}}\s+([\d\.e\+]+)", r.text)
                return float(m.group(1)) if m else 0.0

            self.stats[addr] = {
                "kv_usage": p("vllm:kv_cache_usage_perc"),
                "running_tokens": int(p("vllm:running_tokens")),
                "running_requests": int(p("vllm:num_requests_running")),
                "running_predict_tokens": int(p("vllm:running_predict_tokens")),
                "waiting_tokens": int(p("vllm:waiting_tokens")),
                "status": "healthy",
                "dirty": True,
            }
        except Exception:
            self.stats.setdefault(addr, {})["status"] = "unhealthy"

    def _compute_all_loads(self):
        now = time.time()
        do_log = self.enable_monitor_log and (
            now - self._last_log_time >= self.log_interval
        )
        global latest_load_cv
        for addr, stat in self.stats.items():
            if stat.get("status") != "healthy" or not stat.get("dirty"):
                continue

            stat["dirty"] = False
            load = self._calculate_load(
                stat["running_requests"],
                stat["running_tokens"],
                stat["kv_usage"],
                stat["running_predict_tokens"],
                stat["waiting_tokens"],
            )
            # with latest_load_cv:
            #     self.latest_load[addr] = load
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
                    f"Future_Tokens={load['Future_Tokens']}",
                    flush=True,
                )

        if do_log:
            self._last_log_time = now

    def _calculate_load(self, n_req, m_tok, kv_usage, running_predict_tokens,
                        waiting_tokens):
        lm = self.load_model
        kv_vram = (
            2
            * lm.config.num_hidden_layers
            * m_tok
            * lm.kv_dim
            * lm.precision_bytes
        )
        if n_req == 0 and m_tok == 0:
            mem_bytes = 0
        else:
            mem_bytes = lm.weights_vram + kv_vram
        flops = (
            n_req * lm.linear_flops_coeff
            + lm.attn_flops_coeff * m_tok
        )

        t_compute = (flops / 1e12) / lm.peak_tflops * 1000
        t_memory = (mem_bytes / 1e9) / lm.bandwidth_gbs * 1000

        lc = t_compute / lm.tpot
        lm_ = t_memory / lm.tpot
        lcap = kv_usage
        Future_Tokens = running_predict_tokens + waiting_tokens
        return {
            "Load_Compute": lc,
            "Load_Memory": lm_,
            "Load_Memory_Capacity": lcap,
            "Load_Bottle": max(lc, lm_, lcap),
            "Future_Tokens": Future_Tokens,
        }

    def select_best_instance(self, instances, predictor_meta: Optional[dict] = None):
        if not instances:
            return None, None, float("inf")

        predicted_output_len = FALLBACK_PREDICT_OUTPUT_LEN
        if predictor_meta is not None:
            predicted_output_len = int(
                predictor_meta.get("predict_output_len", FALLBACK_PREDICT_OUTPUT_LEN))
        predicted_output_len = max(1, min(predicted_output_len, MAX_PREDICT_OUTPUT_LEN))

        valid_instances = []
        safe_instances = []
        global latest_load_cv
        for addr, zmq_addr in instances:
            load = self.latest_load.get(addr)
            if not load or "Load_Bottle" not in load or "Future_Tokens" not in load:
                continue

            projected_future = load["Future_Tokens"] + predicted_output_len
            score = (
                projected_future,
                load["Load_Bottle"],
                addr,
                zmq_addr[0],
            )
            valid_instances.append(score)

            if (load["Load_Bottle"] <= MAX_DECODE_LOAD_BOTTLE
                    and load.get("Load_Memory_Capacity", 0.0) <= MAX_DECODE_KV_USAGE
                    and projected_future <= MAX_DECODE_FUTURE_TOKENS):
                safe_instances.append(score)

        if not valid_instances:
            return None, None, float("inf")

        # Prefer instances below safety thresholds; if none exists, degrade to
        # least-loaded to avoid selecting a random overloaded decode.
        candidates = safe_instances if safe_instances else valid_instances
        candidates.sort(key=lambda x: (x[0], x[1]))
        top_instance = candidates[0]
        best_future_tokens, best_bottle, best_addr, best_zmq_addr = top_instance

        self.latest_load[best_addr]["Future_Tokens"] = (
            self.latest_load[best_addr]["Future_Tokens"] + predicted_output_len)
        return best_addr, best_zmq_addr, best_bottle


monitor = DecodeMonitor(
    config_path=os.environ.get("MODEL_CONFIG_PATH", "model_config.json"),
    precision=os.environ.get("VLLM_DTYPE", "bf16"),
    tpot=os.environ.get("TPOT", 'tpot:50').split(':')[1],
    check_interval=0.01,
    enable_monitor_log=True,
)


@app.before_serving
async def start_monitor():
    asyncio.create_task(monitor.fetch_metrics())
    
def _remove_oldest_instances(instances: dict[str, Any]) -> None:
    oldest_key = next(iter(instances), None)
    while oldest_key is not None:
        value = instances[oldest_key]
        if value[1] > time.time():
            break
        print(f"🔴Remove [HTTP:{oldest_key}, ZMQ:{value[0]}, stamp:{value[1]}]")
        instances.pop(oldest_key, None)
        oldest_key = next(iter(instances), None)


def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
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
                    _remove_oldest_instances(prefill_instances)

            elif data["type"] == "D":
                global decode_instances
                global decode_cv
                with decode_cv:
                    node = decode_instances.get(data["http_address"], None)
                    decode_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(decode_instances)
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


def _extract_predict_output_len(req_data: Dict[str, Any]) -> int:
    raw = req_data.get("max_completion_tokens", req_data.get("max_tokens", FALLBACK_PREDICT_OUTPUT_LEN))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = FALLBACK_PREDICT_OUTPUT_LEN
    return max(1, min(value, MAX_PREDICT_OUTPUT_LEN))


async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"Downstream {url} failed: status={response.status}, body={body[:300]}")

            async for chunk_bytes in response.content.iter_chunked(1024):
                yield chunk_bytes

@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()
        if not isinstance(original_request_data, dict):
            return {"error": "invalid json body"}, 400

        predict_output_len = _extract_predict_output_len(original_request_data)
        predictor_meta = Custom_Metadata(
            req_id=None,
            token_range=None,
            num_tokens=None,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            predict_output_len=predict_output_len,
        ).to_dict()
        original_request_data["predictor_meta"] = predictor_meta

        prefill_request = original_request_data.copy()
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1

        global count
        global prefill_instances
        global prefill_cv

        global decode_instances
        global decode_cv

        async with ROUTING_LOCK:
            with prefill_cv:
                prefill_list = list(prefill_instances.items())
                if not prefill_list:
                    return {"error": "no prefill instance available"}, 503
                prefill_addr, prefill_zmq_addr = prefill_list[count % len(prefill_list)]
                prefill_zmq_addr = prefill_zmq_addr[0]

            with decode_cv:
                decode_list = list(decode_instances.items())

            decode_addr, decode_zmq_addr, min_score = monitor.select_best_instance(
                decode_list, predictor_meta)
            if decode_addr is None or decode_zmq_addr is None:
                return {"error": "no decode instance available"}, 503

            request_id = (
                f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
                f"{decode_zmq_addr}_{random_uuid()}"
            )
            this_count = count
            count += 1

        print(
            f"handle_request count: {this_count}, [HTTP:{prefill_addr}, "
            f"ZMQ:{prefill_zmq_addr}] 👉 [HTTP:{decode_addr}, "
            f"ZMQ:{decode_zmq_addr}]"
        )

        async for _ in forward_request(
            f"http://{prefill_addr}{request.path}", prefill_request, request_id
        ):
            continue

        generator = forward_request(
            f"http://{decode_addr}{request.path}", original_request_data, request_id
        )
        response = await make_response(generator)
        response.timeout = None
        return response
    except Exception as e:
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        return {"error": str(e)}, 500



if __name__ == "__main__":
    proxy_port=os.environ.get("PROXY_PORT", "28001")
    bench_port=os.environ.get("BENCH_PORT", "22006")
    t = start_service_discovery("0.0.0.0", proxy_port)
    app.run(host="0.0.0.0", port=bench_port)
    t.join()
