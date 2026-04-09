# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import socket
import threading
import time
import uuid
import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import aiohttp
import httpx
import msgpack
import zmq
from quart import Quart, jsonify, make_response, request

count = 0
prefill_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)
decode_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)

prefill_cv = threading.Condition()
decode_cv = threading.Condition()

DEFAULT_PING_SECONDS = 5


def _remove_oldest_instances(instances: dict[str, Any]) -> None:
    oldest_key = next(iter(instances), None)
    while oldest_key is not None:
        value = instances[oldest_key]
        if value[1] > time.time():
            break
        print(f"🔴Remove [HTTP:{oldest_key}, ZMQ:{value[0]}, stamp:{value[1]}]")
        instances.pop(oldest_key, None)
        oldest_key = next(iter(instances), None)

# def _remove_oldest_instances(instances: dict) -> None:
#     now = time.time()
#     keys_to_del = [k for k, v in instances.items() if v[1] < now]
#     for k in keys_to_del:
#         val = instances.pop(k, None)
#         if val: print(f"🔴Remove [HTTP:{k}, ZMQ:{val[0]}]")
def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll(1000))
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


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

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
        h_dim = c.hidden_size
        i_dim = c.intermediate_size
        v_dim = c.vocab_size
        l_dim = c.num_hidden_layers
        self.kv_dim = (h_dim // c.num_attention_heads) * c.num_key_value_heads
        self.layer_params = (h_dim * (h_dim + 2 * self.kv_dim) + h_dim * h_dim + 3 * h_dim * i_dim)
        self.total_params = l_dim * self.layer_params + v_dim * h_dim
        self.weights_vram = self.total_params * self.precision_bytes
        self.linear_flops_coeff = 2 * (l_dim * self.layer_params + v_dim * h_dim)
        self.attn_flops_coeff = 4 * l_dim * h_dim


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
        # [wt] startup self-check tracks config source and parse errors.
        self.config_path = config_path
        self.config, self.config_source, self.config_error = self._load_config(config_path)
        self.precision = precision.lower()
        if self.precision == "bf16":
            self.precision = "bfloat16"

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
        # [wt] print effective runtime constants once at startup.
        self._log_startup_self_check()
        self._last_log_time = 0.0
        self.log_interval = 1.0

    def _load_config(self, path: str) -> tuple[ModelConfig, str, Optional[str]]:
        source = "defaults"
        error_detail = None
        try:
            with open(path, "r") as f:
                raw = json.load(f)
            source = "file"
        except Exception as exc:
            raw = {}
            error_detail = f"{type(exc).__name__}: {exc}"
        return (
            ModelConfig(
            hidden_size=raw.get("hidden_size", 4096),
            num_hidden_layers=raw.get("num_hidden_layers", 32),
            vocab_size=raw.get("vocab_size", 32000),
            intermediate_size=raw.get("intermediate_size", 11008),
            num_attention_heads=raw.get("num_attention_heads", 32),
            num_key_value_heads=raw.get("num_key_value_heads", raw.get("num_attention_heads", 8)),
            ),
            source,
            error_detail,
        )

    def _log_startup_self_check(self):
        # [wt] self-check explains what config/precision/tpot the proxy really uses.
        c = self.config
        lm = self.load_model
        bw_bytes_ms = (lm.bandwidth_gbs * 1e9) / 1000
        memory_denominator = bw_bytes_ms * lm.tpot - lm.weights_vram
        print(
            "[WT][SELF-CHECK] "
            f"cwd={os.getcwd()} "
            f"config_path={self.config_path} "
            f"config_source={self.config_source} "
            f"dtype={self.precision} "
            f"tpot_ms={self.tpot}",
            flush=True,
        )
        print(
            "[WT][SELF-CHECK] "
            f"model hidden={c.hidden_size} layers={c.num_hidden_layers} "
            f"inter={c.intermediate_size} attn_heads={c.num_attention_heads} "
            f"kv_heads={c.num_key_value_heads} vocab={c.vocab_size} "
            f"precision_bytes={self.precision_bytes}",
            flush=True,
        )
        print(
            "[WT][SELF-CHECK] "
            f"kv_dim={lm.kv_dim} weights_vram_gb={lm.weights_vram / 1e9:.3f} "
            f"bw_gbs={lm.bandwidth_gbs} memory_denominator_gb={memory_denominator / 1e9:.3f} "
            f"peak_tflops={lm.peak_tflops}",
            flush=True,
        )
        if self.config_source != "file":
            print(
                "[WT][SELF-CHECK][WARN] "
                "model config file load failed, using built-in defaults. "
                f"error={self.config_error}",
                flush=True,
            )
        if c.num_key_value_heads == c.num_attention_heads:
            print(
                "[WT][SELF-CHECK][HINT] "
                "kv_heads equals attn_heads. For GQA models this can indicate fallback defaults.",
                flush=True,
            )
        if memory_denominator <= 0:
            print(
                "[WT][SELF-CHECK][WARN] "
                f"invalid memory denominator={memory_denominator}; Load_Memory may be inf.",
                flush=True,
            )

    async def fetch_metrics(self):
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=50)
        # Decode metrics are from internal service addresses; bypass env proxies.
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(0.5),
            limits=limits,
            trust_env=False,
        ) as client:
            while self.is_running:
                urls = list(decode_instances.keys())
                if urls:
                    await asyncio.gather(
                        *[self._update_instance(client, decode_addr) for decode_addr in urls],
                        return_exceptions=True,
                    )
                    self._compute_all_loads()
                await asyncio.sleep(self.check_interval)

    async def _update_instance(self, client: httpx.AsyncClient, addr: str):
        try:
            response = await client.get(f"http://{addr}/metrics")
            if response.status_code != 200:
                return

            def parse_metric(name):
                match = re.search(rf"{name}\{{.*?\}}\s+([\d\.e\+]+)", response.text)
                return float(match.group(1)) if match else 0.0

            self.stats[addr] = {
                "kv_usage": parse_metric("vllm:kv_cache_usage_perc"),
                "running_tokens": int(parse_metric("vllm:running_tokens")),
                "running_requests": int(parse_metric("vllm:num_requests_running")),
                "status": "healthy",
                "dirty": True,
            }
        except Exception:
            self.stats.setdefault(addr, {})["status"] = "unhealthy"

    def _compute_all_loads(self):
        now = time.time()
        do_log = self.enable_monitor_log and (now - self._last_log_time >= self.log_interval)
        for addr, stat in self.stats.items():
            if stat.get("status") != "healthy" or not stat.get("dirty"):
                continue
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
        load_model = self.load_model

        # Decode-side 3D load model (single concentrated reference block).
        #
        # Assumptions:
        # 1) Pure decode window: n_req = running decode requests, m_tok = running decode tokens.
        # 2) Peak hardware constants are used (peak_tflops, bandwidth_gbs), so outputs are
        #    pressure scores for online scheduling instead of strict hardware-utilization values.
        #
        # Symbols:
        # H = hidden_size, I = intermediate_size, L = num_hidden_layers,
        # V = vocab_size,
        # K = kv_dim, b = precision_bytes,
        # BW = bandwidth_gbs * 1e9 / 1000 (bytes/ms), TPOT in ms/token.
        #
        # 1) Compute pressure
        #    Original (no aggregated coeff symbols):
        #    Attention fixed part (QKV + O projections):
        #    attn_proj_params = H*(H + 2*K) + H*H
        #    说明: 单层注意力线性层参数，含 QKV 投影和 O 投影。
        #    per_req_attn_proj_flops = 2 * L * attn_proj_params
        #    说明: 每个请求每步都要做的注意力线性投影计算，2 表示乘加近似 2 FLOPs。
        #
        #    MLP part:
        #    mlp_params = 3 * H * I
        #    说明: 单层 MLP 近似参数规模（门控结构按 3HI 记）。
        #    per_req_mlp_flops = 2 * L * mlp_params
        #    说明: 每个请求每步都要做的 MLP 线性计算。
        #
        #    Output head part:
        #    per_req_vocab_flops = 2 * V * H
        #    说明: 输出 logits 线性层近似开销。
        #
        #    Attention context-length part:
        #    attn_ctx_flops = 4 * L * H * m_tok
        #    说明: 对历史 token 的 QK^T + AV 代价，随 m_tok 线性增长。
        #
        #    Combined original form:
        #    per_req_linear_flops = per_req_attn_proj_flops + per_req_mlp_flops + per_req_vocab_flops
        #    FLOPs = n_req * per_req_linear_flops + attn_ctx_flops
        #    说明: 总计算量拆成“请求数驱动固定成本 + 上下文长度驱动成本”。
        #
        #    Define aggregated coefficients used in code:
        #    layer_params = attn_proj_params + mlp_params = H*(H + 2*K) + H*H + 3*H*I
        #    说明: 与代码中的 layer_params 定义保持一致。
        #    linear_flops_coeff = 2 * (L*layer_params + V*H)
        #    说明: 把 Attention 固定部分 + MLP + 输出头聚合成 n_req 的统一系数。
        #    attn_flops_coeff = 4 * L * H
        #    说明: 把 Attention 长度相关部分聚合成 m_tok 的统一系数。
        #
        #    Coefficient form (implementation):
        #    FLOPs = n_req * linear_flops_coeff + m_tok * attn_flops_coeff
        #    说明: 与原始式等价，但在线计算只需两次乘法和一次加法。
        #    t_compute_ms = (FLOPs / 1e12) / peak_tflops * 1000
        #    说明: 先把 FLOPs 转 TFLOPs，再按峰值算理论耗时并换算为毫秒。
        #    Load_Compute = t_compute_ms / TPOT
        #    说明: 用计算耗时除以每 token 时间预算，得到无量纲计算压力。
        #
        # 2) Memory/working-set pressure
        #    Original form:
        #    kv_vram_bytes = 2 * L * m_tok * K * b
        #    说明: K/V 两路缓存、每层都保留、每元素按 b 字节计，且随 m_tok 线性增长。
        #    activation_vram_bytes = n_req * L * (2 * H + 2 * I) * b
        #    说明: 近似把 decode 中与请求并发相关的激活/中间态按层累加。
        #    total_dynamic_io_bytes = kv_vram_bytes + activation_vram_bytes
        #    说明: 将 token 相关和请求相关的动态内存工作集合并成总压力项。
        #
        #    Denominator and final pressure:
        #    memory_denominator = BW * TPOT - weights_vram
        #    说明: BW*TPOT 是单 token 时间窗内可供搬运预算，减去权重常驻占用后的有效余量。
        #    Load_Memory = total_dynamic_io_bytes / memory_denominator
        #    说明: 用动态工作集与可用预算做比值，得到无量纲内存压力。
        #
        # 3) Capacity pressure
        #    Load_Memory_Capacity = kv_usage
        #    说明: 直接采用运行时 KV cache 使用率，刻画容量接近上限的风险。
        #
        # Bottleneck score
        #    Load_Bottle = max(Load_Compute, Load_Memory, Load_Memory_Capacity)
        #    说明: 取三者最大值作为瓶颈负载，符合短板决定吞吐的调度直觉。
        h_dim = load_model.config.hidden_size
        i_dim = load_model.config.intermediate_size
        l_dim = load_model.config.num_hidden_layers

        bw_bytes_ms = (load_model.bandwidth_gbs * 1e9) / 1000

        kv_vram_bytes = (
            2 * l_dim * m_tok * load_model.kv_dim * load_model.precision_bytes
        )

        activation_vram_bytes = (
            n_req * l_dim * (2 * h_dim + 2 * i_dim) * load_model.precision_bytes
        )

        flops = (n_req * load_model.linear_flops_coeff + load_model.attn_flops_coeff * m_tok)
        t_compute = (flops / 1e12) / load_model.peak_tflops * 1000

        load_compute = t_compute / load_model.tpot

        memory_denominator = bw_bytes_ms * load_model.tpot - load_model.weights_vram
        total_dynamic_io_bytes = kv_vram_bytes + activation_vram_bytes

        load_memory = (
            total_dynamic_io_bytes / memory_denominator
            if memory_denominator > 0
            else float("inf")
        )

        load_memory_capacity = kv_usage

        return {
            "Load_Compute": load_compute,
            "Load_Memory": load_memory,
            "Load_Memory_Capacity": load_memory_capacity,
            "Load_Bottle": max(load_compute, load_memory, load_memory_capacity),
        }


_tpot_raw = os.environ.get("TPOT", "tpot:50")
_tpot_value = _tpot_raw.split(":", 1)[1] if ":" in _tpot_raw else _tpot_raw

monitor = DecodeMonitor(
    config_path=os.environ.get("MODEL_CONFIG_PATH", "model_config.json"),
    precision=os.environ.get("VLLM_DTYPE", "bf16"),
    tpot=_tpot_value,
    check_interval=0.01,
    enable_monitor_log=True,
)


@app.before_serving
async def start_monitor():
    asyncio.create_task(monitor.fetch_metrics())


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                if True:
                    async for chunk_bytes in response.content.iter_chunked(1024):
                        yield chunk_bytes
                else:
                    content = await response.read()
                    yield content


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()

        prefill_request = original_request_data.copy()
        # change max_tokens = 1 to let it only do prefill
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1

        global count
        global prefill_instances
        global prefill_cv
        with prefill_cv:
            prefill_list = list(prefill_instances.items())
            if not prefill_list:
                return jsonify({"error": "No prefill instance available"}), 503
            prefill_addr, prefill_zmq_addr = prefill_list[count % len(prefill_list)]
            prefill_zmq_addr = prefill_zmq_addr[0]

        global decode_instances
        global decode_cv
        with decode_cv:
            decode_list = list(decode_instances.items())
            if not decode_list:
                return jsonify({"error": "No decode instance available"}), 503
            decode_addr, decode_zmq_addr = decode_list[count % len(decode_list)]
            decode_zmq_addr = decode_zmq_addr[0]

        print(
            f"handle_request count: {count}, [HTTP:{prefill_addr}, "
            f"ZMQ:{prefill_zmq_addr}] 👉 [HTTP:{decode_addr}, "
            f"ZMQ:{decode_zmq_addr}]"
        )
        count += 1

        request_id = (
            f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
            f"{decode_zmq_addr}_{random_uuid()}"
        )

        # finish prefill
        async for _ in forward_request(
            f"http://{prefill_addr}{request.path}", prefill_request, request_id
        ):
            continue

        # return decode
        generator = forward_request(
            f"http://{decode_addr}{request.path}", original_request_data, request_id
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
        return jsonify({"error": "disagg proxy internal error", "detail": str(e)}), 500


if __name__ == "__main__":
    proxy_port = int(os.environ.get("PROXY_PORT", "30001"))
    bench_port = int(os.environ.get("BENCH_PORT", "10001"))
    t = start_service_discovery("0.0.0.0", proxy_port)
    app.run(host="0.0.0.0", port=bench_port)
    t.join()
