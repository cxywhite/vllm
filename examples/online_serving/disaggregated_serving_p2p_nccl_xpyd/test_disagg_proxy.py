import os
import socket
import threading
import time
import uuid
from typing import Any,Dict, List
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

# =========================
# 配置与全局变量
# =========================
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
app = Quart(__name__)

count = 0
prefill_instances: dict[str, Any] = {}
decode_instances: dict[str, Any] = {}

prefill_cv = threading.Condition()
decode_cv = threading.Condition()
latest_load_cv = threading.Condition()
DEFAULT_PING_SECONDS = 5

# Predictor ZMQ
PREDICTOR_ZMQ_ADDR = "tcp://127.0.0.1:32323"
predictor_ctx = zmq.Context.instance()
predictor_sock = predictor_ctx.socket(zmq.PULL)
predictor_sock.bind(PREDICTOR_ZMQ_ADDR)

# request_id -> asyncio.Future
pending_predictor_futures: dict[str, asyncio.Future] = {}
prefilled_finished_requests: dict[str, Any] = {}

# Routing PUB
ROUTING_PUB_ADDR = "tcp://127.0.0.1:32324"
routing_ctx = zmq.Context.instance()
routing_pub = routing_ctx.socket(zmq.PUB)
routing_pub.bind(ROUTING_PUB_ADDR)

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

        self.bandwidth_gbs = 1935
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
                "running_predict_tokens": int(p("vllm:running_predict_tokens")),
                "waiting_tokens": int(p("vllm:waiting_tokens")),
                "status": "healthy", "dirty": True,
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
                stat["running_requests"], stat["running_tokens"], stat["kv_usage"],
                stat["running_predict_tokens"], stat["waiting_tokens"]
            )
            self.latest_load[addr] = load
            if do_log:
                print(f"[WT][MONITOR] decode={addr} reqs={stat['running_requests']} "
                      f"bottle={load['Load_Bottle']:.3f} compute={load['Load_Compute']:.3f} "
                      f"mem={load['Load_Memory']:.3f} cap={load['Load_Memory_Capacity']:.3f} "
                      f"Future={load['Future_Tokens']}", flush=True)
        if do_log: self._last_log_time = now

    def _calculate_load(self, n_req, m_tok, kv_usage, running_predict_tokens, waiting_tokens):
        lm = self.load_model
        
        # 统一单位：每毫秒能读取的字节数 (GB/s -> Bytes/ms)
        bw_bytes_ms = (lm.bandwidth_gbs * 1e9) / 1000

        # 1. 权重访存耗时 (固定开销: 每一采样步都要完整读取权重)
        # 如果没有请求(n_req=0)，理论上不进行Step，开销为0
        t_weights_io = lm.weights_vram / bw_bytes_ms if n_req > 0 else 0

        # 2. KV Cache 访存耗时 (动态开销: 读取当前所有Token的历史KV)
        kv_vram_bytes = (
            2 * lm.config.num_hidden_layers 
            * m_tok 
            * lm.kv_dim 
            * lm.precision_bytes
        )
        t_kv_io = kv_vram_bytes / bw_bytes_ms

        # 总访存耗时
        t_memory = t_weights_io + t_kv_io

        # 3. 计算耗时 (TFLOPS 维度)
        flops = (n_req * lm.linear_flops_coeff + lm.attn_flops_coeff * m_tok)
        t_compute = (flops / 1e12) / lm.peak_tflops * 1000

        # 4. 归一化负载
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

    def select_best_instance(self, instances, predictor_meta: Optional[dict]=None):
        if not instances: return None, None, float("inf")
        predict_len = predictor_meta.get('predict_output_len', 0) if predictor_meta else 0
        
        valid_candidates = []
        for addr, zmq_info in instances:
            load = self.latest_load.get(addr)
            if load and "Load_Bottle" in load:
                # 记录 (未来Token + 本次预测长度, 瓶颈负载, 地址, ZMQ地址)
                valid_candidates.append((
                    load["Future_Tokens"] + predict_len, 
                    load["Load_Bottle"], addr, zmq_info[0]
                ))
        
        if not valid_candidates: return None, None, float("inf")
        
        # 逻辑：按Future_tokens排序取Top3，再从Top3选负载最小
        valid_candidates.sort(key=lambda x: x[0])
        top_candidates = valid_candidates[:3]
        top_candidates.sort(key=lambda x: x[1])
        
        best_future, best_bottle, best_addr, best_zmq_addr = top_candidates[0]
        # 原子更新：防止瞬时请求堆积
        if best_addr in self.latest_load:
            self.latest_load[best_addr]["Future_Tokens"] += predict_len
            
        return best_addr, best_zmq_addr, best_bottle

# =========================
# 辅助函数
# =========================

monitor = DecodeMonitor(
    config_path=os.environ.get("MODEL_CONFIG_PATH", "model_config.json"),
    precision=os.environ.get("VLLM_DTYPE", "bf16"),
    tpot=float(os.environ.get("TPOT", "tpot:50").split(':')[-1]),
)

@app.before_serving
async def start_monitor():
    asyncio.create_task(monitor.fetch_metrics())

def _remove_oldest_instances(instances: dict) -> None:
    now = time.time()
    keys_to_del = [k for k, v in instances.items() if v[1] < now]
    for k in keys_to_del:
        val = instances.pop(k, None)
        if val: print(f"🔴Remove [HTTP:{k}, ZMQ:{val[0]}]")

def _listen_for_register(poller, router_socket):
    while True:
        socks = dict(poller.poll(1000))
        if router_socket in socks:
            try:
                remote_address, message = router_socket.recv_multipart()
                data = msgpack.loads(message)
                target_dict = prefill_instances if data["type"] == "P" else decode_instances
                cv = prefill_cv if data["type"] == "P" else decode_cv
                
                with cv:
                    is_new = data["http_address"] not in target_dict
                    target_dict[data["http_address"]] = (
                        data["zmq_address"], time.time() + DEFAULT_PING_SECONDS
                    )
                    _remove_oldest_instances(target_dict)
                    if is_new: print(f"🔵Add [HTTP:{data['http_address']}, ZMQ:{data['zmq_address']}]")
            except Exception as e:
                print(f"Register error: {e}")

def _listen_predictor_metadata():
    """监听 ZMQ 并通过 loop.call_soon_threadsafe 唤醒异步 Future"""
    global prefilled_finished_requests
    loop = None
    while True:
        try:
            msg = predictor_sock.recv_string()
            meta_list = json.loads(msg)
            for meta in meta_list:
                req_id = meta.get("req_id")
                if not req_id: continue
                
                # 如果 handle_request 正在等待这个 req_id
                if req_id in pending_predictor_futures:
                    fut = pending_predictor_futures.pop(req_id)
                    if not fut.done():
                        # 获取主事件循环并设置结果
                        fut.get_loop().call_soon_threadsafe(fut.set_result, meta)
                else:
                    # 提前到达的消息存入暂存区
                    prefilled_finished_requests[req_id] = (meta, time.time())
            
            # 定期清理暂存区 (TTL 60s)
            now = time.time()
            expired = [k for k, v in prefilled_finished_requests.items() if now - v[1] > 60]
            for k in expired: prefilled_finished_requests.pop(k)

        except Exception:
            import traceback
            traceback.print_exc()

async def forward_request(url, data, request_id):
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}", "X-Request-Id": request_id}
        try:
            async with session.post(url=url, json=data, headers=headers) as response:
                if response.status == 200:
                    async for chunk in response.content.iter_chunked(1024):
                        yield chunk
                else:
                    err_msg = await response.text()
                    print(f"Forward error {response.status}: {err_msg}")
        except Exception as e:
            print(f"Request connection error: {e}")

# =========================
# 核心路由逻辑
# =========================

@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()
        prefill_request = original_request_data.copy()
        prefill_request["max_tokens"] = 1
        prefill_request["predictor_meta"] = None
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1

        # 1. 选择 Prefill 节点
        global count
        with prefill_cv:
            if not prefill_instances: return "No prefill instances", 503
            prefill_list = list(prefill_instances.items())
            prefill_addr, (prefill_zmq_addr, _) = prefill_list[count % len(prefill_list)]
            count += 1

        request_id = f"___prefill_addr_{prefill_zmq_addr}__{uuid.uuid4().hex}"
        predictor_req_id = f"cmpl-{request_id}-0"

        # 2. 发起 Prefill
        # 这里必须完整耗尽 generator
        async for _ in forward_request(f"http://{prefill_addr}{request.path}", prefill_request, request_id):
            continue

        # 3. 获取 Predictor Metadata (使用 Future 模式)
        predictor_meta = None
        # 检查是否已经提前到达
        if predictor_req_id in prefilled_finished_requests:
            predictor_meta, _ = prefilled_finished_requests.pop(predictor_req_id)
        else:
            # 创建 Future 并等待 ZMQ 线程填充
            fut = asyncio.get_event_loop().create_future()
            pending_predictor_futures[predictor_req_id] = fut
            try:
                predictor_meta = await asyncio.wait_for(fut, timeout=180.0)
            except asyncio.TimeoutError:
                pending_predictor_futures.pop(predictor_req_id, None)
                return "Predictor metadata timeout", 504

        # 4. 选择最佳 Decode 节点
        with decode_cv:
            decode_list = list(decode_instances.items())
            decode_addr, decode_zmq_addr, _ = monitor.select_best_instance(decode_list, predictor_meta)

        if not decode_addr: return "No decode instances available", 503

        # 5. 广播路由决策并更新 ID
        routing_msg = f"{predictor_req_id}|{decode_zmq_addr}"
        routing_pub.send_string(routing_msg)

        new_id_part = f"___prefill_addr_{prefill_zmq_addr}___decode_addr_{decode_zmq_addr}_"
        request_id = request_id.replace(f"___prefill_addr_{prefill_zmq_addr}__", new_id_part)
        
        # 6. 发送最终 Decode 请求
        original_request_data["predictor_meta"] = predictor_meta
        generator = forward_request(f"http://{decode_addr}{request.path}", original_request_data, request_id)
        
        resp = await make_response(generator)
        resp.timeout = None
        return resp

    except Exception as e:
        import traceback
        traceback.print_exc()
        return str(e), 500

# =========================
# 启动入口
# =========================

def start_service_discovery(hostname, port):
    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")
    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)
    t = threading.Thread(target=_listen_for_register, args=[poller, router_socket], daemon=True)
    t.start()
    return t

if __name__ == "__main__":
    proxy_port = os.environ.get("PROXY_PORT", "28001")
    bench_port = os.environ.get("BENCH_PORT", "22006")
    
    t1 = start_service_discovery("0.0.0.0", proxy_port)
    t2 = threading.Thread(target=_listen_predictor_metadata, daemon=True)
    t2.start()
    
    app.run(host="0.0.0.0", port=int(bench_port))