# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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
from wt_metadata import Custom_Metadata
# 导入ModelConfig
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
app = Quart(__name__)


count = 0
prefill_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)
decode_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)
prefill_cv = threading.Condition()
decode_cv = threading.Condition()
DEFAULT_PING_SECONDS = 5

# [WT]delay kvcache transfer 2025-12-27 22:09:30
# Predictor metadata 接收（ZMQ）
PREDICTOR_ZMQ_ADDR = "tcp://127.0.0.1:32323"
predictor_ctx = zmq.Context.instance()
predictor_sock = predictor_ctx.socket(zmq.PULL)
predictor_sock.bind(PREDICTOR_ZMQ_ADDR)
# prefilled_finished_cv=threading.Condition()
prefilled_finished_requests: dict[str, Any] = {}  # request_id: metadata
# 新增：路由广播端口 (Proxy -> Prefill Connector)
ROUTING_PUB_ADDR = "tcp://127.0.0.1:32324"  # 新增端口
routing_ctx = zmq.Context.instance()
routing_pub = routing_ctx.socket(zmq.PUB)
routing_pub.bind(ROUTING_PUB_ADDR)
# [WT] end
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

        # KV 真实维度（GQA）
        self.kv_dim = (H // c.num_attention_heads) * c.num_key_value_heads

        # -------- 参数量 --------
        self.layer_params = (
            H * (H + 2 * self.kv_dim)  # QKV
            + H * H                    # Output proj
            + 3 * H * I                # FFN
        )

        self.total_params = L * self.layer_params + V * H
        self.weights_vram = self.total_params * self.precision_bytes

        # 激活 buffer（经验上界）
        # self.act_vram = 1024 * L * H * self.precision_bytes * 2

        # -------- FLOPs 系数 --------
        self.linear_flops_coeff = 2 * (L * self.layer_params + V * H)
        self.attn_flops_coeff = 4 * L * H

        
class DecodeMonitor:
    def __init__(
        self,
        config_path: str,
        precision: str = "bfloat16",
        tpot: float = 50.0,
        check_interval: float = 0.01,
    ):
        self.check_interval = check_interval
        self.stats: Dict[str, dict] = {}
        self.is_running = True

        self.config = self._load_config(config_path)
        self.precision = precision.lower()
        print(f'[WT] DecodeMonitor initialized with precision={self.precision}', flush=True)
        precision_map = {
            "int8": 1, "float8": 1,
            "float16": 2, "bfloat16": 2,
            "float32": 4
        }
        self.precision_bytes = precision_map.get(self.precision, 2)
        self.config.bytes_per_param = self.precision_bytes

        # 硬件参数（A800）
        self.bandwidth_gbs = 1935
        self.mem_capacity = 80
        self.tpot = float(tpot)

        if self.precision == "float32":
            self.peak_tflops = 19.5
        elif self.precision in ["float16", "bfloat16"]:
            self.peak_tflops = 312
        elif self.precision in ["int8", "float8"]:
            self.peak_tflops = 624
        else:
            self.peak_tflops = 312
        # print(f'[WARN]!!! llama-3-8b,gpu_utilization=0.8,max_kv_cache set 382832',flush=True)
        # print(f'[WARN]!!! if modify model or gpu_utilization need to be changed')
        # self.kv_cache=382832
        # ✅ 静态建模一次完成
        self.load_model = DecodeLoadModel(
            config=self.config,
            precision_bytes=self.precision_bytes,
            peak_tflops=self.peak_tflops,
            bandwidth_gbs=self.bandwidth_gbs,
            mem_capacity_gb=self.mem_capacity,
            tpot=self.tpot
        )

        # ===== 新增：监控日志控制 =====
        self.log_interval = 1.0          # 每 1 秒打印一次
        self._last_log_time = 0.0

        # ===== 新增：最近一次计算出的负载（仅用于监控，不用于调度）=====
        self.latest_load: Dict[str, dict] = {}
    def _load_config(self, path: str) -> ModelConfig:
        with open(path, 'r') as f:
            raw = json.load(f)
        return ModelConfig(
            hidden_size=raw.get("hidden_size", 4096),
            num_hidden_layers=raw.get("num_hidden_layers", 32),
            vocab_size=raw.get("vocab_size", 32000),
            intermediate_size=raw.get("intermediate_size", 11008),
            num_attention_heads=raw.get("num_attention_heads", 32),
            num_key_value_heads=raw.get("num_key_value_heads", raw.get("num_attention_heads", 32))
        )
    async def fetch_metrics(self):
        """动态感知 decode_instances 并抓取指标"""
        async with httpx.AsyncClient(timeout=0.2) as client:
            while self.is_running:
                # 从全局动态注册表获取当前的 http 地址
                global decode_instances
                global decode_cv
                current_urls = list(decode_instances.keys())
                # 将url中的ip转换成127.0.0.1
                # current_urls = ["127.0.0.1:" + url.split(":")[1] for url in current_urls]
                
                # print(f'[WT] fetch_metrics current_urls: {current_urls}',flush=True)
                if not current_urls:
                    await asyncio.sleep(0.1)
                    continue

                tasks = [self._update_instance(client, url) for url in current_urls]
                await asyncio.gather(*tasks)
                # ===== 新增：周期性计算并打印 decode 实例负载 =====
                self._compute_and_maybe_log_all_loads()
                await asyncio.sleep(self.check_interval)

    async def _update_instance(self, client: httpx.AsyncClient, addr: str):
        try:
            # addr 格式通常是 "127.0.0.1:22020"
            metrics_url = f"http://{addr}/metrics"
            response = await client.get(metrics_url)
            # print(f'[WT]@@@ fetch_metrics response: {response}',flush=True)
            if response.status_code == 200:
                kv_usage = self._parse_metric(response.text, "vllm:kv_cache_usage_perc")
                num_generation_tokens=self._parse_metric(response.text, "vllm:generation_tokens")
                running_tokens=self._parse_metric(response.text, "vllm:running_tokens")
                running_requests=self._parse_metric(response.text, "vllm:num_requests_running")
                running_predict_tokens=self._parse_metric(response.text, "vllm:running_predict_tokens")
                waiting_tokens=self._parse_metric(response.text, "vllm:waiting_tokens")
                # print(f'[WT]@@@ fetch_metrics kv_usage: {kv_usage} ,vllm:running_tokens: {running_tokens} ,vllm:num_requests_running: {running_requests}',flush=True)
                print(f'[WT]@@@ fetch_metrics kv_usage: {kv_usage} vllm:running_tokens: {running_tokens} ,vllm:num_requests_running: {running_requests} ,vllm:running_predict_tokens: {running_predict_tokens} ,vllm:waiting_tokens: {waiting_tokens}',flush=True)
                if kv_usage is None:
                    print(f'[WT]warning: fetch_metrics kv_usage is None',flush=True)
                if running_tokens is None:
                    print(f'[WT]warning: fetch_metrics running_tokens is None',flush=True)
                # num_tokens = self._parse_metric(text, "vllm:num_running_tokens")
                if running_requests is None:
                    print(f'[WT]warning: fetch_metrics running_tokens is None',flush=True)
                self.stats[addr] = {
                    "kv_usage": kv_usage if kv_usage is not None else 0.0,
                    "running_tokens": int(running_tokens) if running_tokens is not None else 0,
                    "running_requests": int(running_requests) if running_requests is not None else 0,
                    "running_predict_tokens": int(running_predict_tokens) if running_predict_tokens is not None else 0,
                    "waiting_tokens": int(waiting_tokens) if waiting_tokens is not None else 0,
                    "status": "healthy",
                    "last_update": time.time()
                }
        except Exception as e:
            if addr in self.stats:
                self.stats[addr]["status"] = "unhealthy"
            print(f'[WT]error: fetch_metrics error: {e}',flush=True)

    def _parse_metric(self, text: str, metric_name: str):
        pattern = rf"{metric_name}\{{.*?\}}\s+([\d\.e\+]+)"
        match = re.search(pattern, text)
        if match: return float(match.group(1))
        return None
    
    def calculate_load(self, n_requests: int, m_running_tokens: int,kv_cache_usage: float):
        lm = self.load_model

        # -------- KV Cache --------
        kv_cache_vram = (
            2 * lm.config.num_hidden_layers
            * m_running_tokens
            * lm.kv_dim
            * lm.precision_bytes
        )
        # total_vram = lm.weights_vram + kv_cache_vram + lm.act_vram

        # -------- Memory Access --------
        mem_access_bytes = lm.weights_vram + kv_cache_vram

        # -------- FLOPs --------
        linear_flops = n_requests * lm.linear_flops_coeff
        attn_flops = lm.attn_flops_coeff * m_running_tokens
        total_flops = linear_flops + attn_flops

        # -------- Latency --------
        t_compute = (total_flops / 1e12) / lm.peak_tflops * 1000
        t_memory = (mem_access_bytes / 1e9) / lm.bandwidth_gbs * 1000

        load_compute = t_compute / lm.tpot
        load_memory = t_memory / lm.tpot
        load_mem_capacity = kv_cache_usage
        # load_mem_capacity = (total_vram / 1e9) / lm.mem_capacity_gb

        load_bottle = max(load_compute, load_memory, load_mem_capacity)

        return {
            # "VRAM_Usage_GB": total_vram / 1e9,
            "Mem_Access_Step_GB": mem_access_bytes / 1e9,
            "Compute_Step_TFLOPs": total_flops / 1e12,
            "Arithmetic_Intensity": total_flops / mem_access_bytes,
            "Load_Compute": load_compute,
            "Load_Memory": load_memory,
            "Load_Memory_Capacity": load_mem_capacity,
            "Load_Bottle": load_bottle,
            "kv_cache_vram": kv_cache_vram / 1e9,
            "weights_vram": lm.weights_vram/ 1e9,
            # "act_vram": lm.act_vram/ 1e9,
        }
    def select_best_instance(
        self,
        available_instances: List[str],
        predictor_meta: Optional[Custom_Metadata]=None,  # 新增：当前请求
    ) -> str:
        """
        Request-aware two-stage decode instance selection.

        Stage 1: Top-K by instantaneous 3D load (Load_Bottle)
        Stage 2: Among Top-K, select minimal future token backlog
                INCLUDING the incoming request.
        """

        candidates = []
        if predictor_meta is not None:
            print(f'[WT][SCHED] predictor_meta found: {predictor_meta}', flush=True)
            r_predict_tokens = int(predictor_meta.get("predict_output_len", 0))
            print(f'[WT][SCHED] r_predict_tokens: {r_predict_tokens}', flush=True)
        else:
            print(f'[WT][SCHED][REQ-AWARE] Warning: predictor_meta is None, using r_predict_tokens=0', flush=True)
            r_predict_tokens=0
        for addr in available_instances:
            stat = self.stats.get(
                addr,
                {
                    "status": "unknown",
                    "running_tokens": 131072,
                    "running_requests": 1024,
                    "running_predict_tokens": 0,
                    "waiting_tokens": 0,
                    "kv_usage": 0.0,
                },
            )

            if stat.get("status") != "healthy":
                continue

            running_tokens = int(stat.get("running_tokens", 0))
            running_requests = int(stat.get("running_requests", 0))
            kv_cache_usage = float(stat.get("kv_usage", 0.0))
            # -------- Stage 1: instantaneous load --------
            load_dict = self.calculate_load(
                n_requests=running_requests,
                m_running_tokens=running_tokens,
                kv_cache_usage=kv_cache_usage,
            )
            load_now = load_dict["Load_Bottle"]

            # -------- Stage 2: request-aware future workload --------
            base_future_tokens = (
                int(stat.get("running_predict_tokens", 0))
                + int(stat.get("waiting_tokens", 0))
            )

            # >>> 核心修改点 <<<
            future_tokens_with_r = base_future_tokens + r_predict_tokens

            candidates.append(
                {
                    "addr": addr,
                    "load_now": load_now,
                    "future_tokens": future_tokens_with_r,
                    "base_future_tokens": base_future_tokens,
                    "load_dict": load_dict,
                }
            )

        if not candidates:
            addr = available_instances[0]
            return addr, float("inf")

        # ---------- Stage 1: Top-K ----------
        candidates.sort(key=lambda x: x["load_now"])
        # K = min(len(candidates), 3)
        K=1
        topk = candidates[:K]

        # ---------- Stage 2: future token aware ----------
        best = min(topk, key=lambda x: x["future_tokens"])

        print(
            "[WT][SCHED][REQ-AWARE]\n"
            f"  request_predict_tokens={r_predict_tokens}\n"
            f"  selected={best['addr']}\n"
            f"  load_now={best['load_now']:.4f}\n"
            f"  base_future_tokens={best['base_future_tokens']}\n"
            f"  future_tokens_with_r={best['future_tokens']}",
            flush=True,
        )

        return best["addr"], best["load_now"]

    def _compute_and_maybe_log_all_loads(self):
        now = time.time()
        do_log = (now - self._last_log_time) >= self.log_interval

        for addr, stat in self.stats.items():
            if stat.get("status") != "healthy":
                continue

            running_tokens = int(stat.get("running_tokens", 0))
            running_requests = int(stat.get("running_requests", 0))
            kv_cache_usage = float(stat.get("kv_usage", 0.0))
            load_dict = self.calculate_load(
                n_requests=running_requests,
                m_running_tokens=running_tokens,
                kv_cache_usage=kv_cache_usage,
            )

            # 缓存最新负载（仅用于观测）
            self.latest_load[addr] = load_dict

            if do_log:
                print(
                    f"[WT][MONITOR] decode={addr} | "
                    f"req={running_requests} | tok={running_tokens} | "
                    f"bottle={load_dict['Load_Bottle']:.3f} | "
                    f"compute={load_dict['Load_Compute']:.3f} | "
                    f"memory={load_dict['Load_Memory']:.3f} | "
                    f"capacity={load_dict['Load_Memory_Capacity']:.3f}",
                    flush=True,
                )

        if do_log:
            self._last_log_time = now

    
# 全局初始化（在 handle_request 之前）
# 通过环境读取模型配置
# global decode_profiler
# decode_profiler = DecodeProfiler(config_path=os.environ.get("MODEL_CONFIG_PATH", "model_config.json"), precision="fp32",tpot=os.environ.get("TPOT", 50.0))
monitor = DecodeMonitor(config_path=os.environ.get("MODEL_CONFIG_PATH", "model_config.json"), precision=os.environ.get("VLLM_DTYPE", 'bf16'),tpot=os.environ.get("TPOT", 'tpot:50').split(':')[1],check_interval=0.5)

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

# [WT]delay kvcache transfer 2025-12-27 22:10:39
# 监听predictor发送预测后metadata
def _listen_predictor_metadata():
    global prefilled_finished_requests
    # global prefilled_finished_cv
    while True:
        try:
            msg = predictor_sock.recv_string()
            meta_list = json.loads(msg)
            for meta in meta_list:
                # req_id只包含prefill_add信息 例如：cmpl-___prefill_addr_10.10.111.36:22010__5a13bb3bc84f45dbb48cffd13f321689-0
                req_id = meta.get("req_id")
                # with prefilled_finished_cv:
                prefilled_finished_requests[req_id] = meta
                # print(f"[WT][Proxy] 222 Received predictor meta for {req_id}: {meta}",flush=True)
                    # prefilled_finished_cv.notify_all()  # [WT][FIX] 唤醒等待者

        except Exception as e:
            import traceback
            traceback.print_exc()
# [WT] end
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
# [WT]delay kvcache transfer 2025-12-27 22:14:21
# 重新设计 handle_request
# 1. 选择 prefill 实例，发起 prefill 请求
# 2. 等待 predictor metadata 回传
# 3. 根据 metadata 选择 decode 实例，发起 decode 请求
# 4. 返回 decode 响应
@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    try:
        original_request_data = await request.get_json()
        # print(f'[WT]>>> Received original request data: {original_request_data}',flush=True)
        prefill_request = original_request_data.copy()
        prefill_request["max_tokens"] = 1
        # [WT]prometheus 2026-01-04 12:14:48
        prefill_request["predictor_meta"] = None
        # [WT] end
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1
        global count
        global prefill_instances
        global prefill_cv
        with prefill_cv:
            prefill_list = list(prefill_instances.items())
            prefill_addr, prefill_zmq_addr = prefill_list[count % len(prefill_list)]
            prefill_zmq_addr = prefill_zmq_addr[0]
        
        request_id = (
            f"___prefill_addr_{prefill_zmq_addr}__{random_uuid()}"
        )
        # print(f'[WT]333 handle_request request_id: {request_id}',flush=True)
        # finish prefill
        async for _ in forward_request(
            f"http://{prefill_addr}{request.path}", prefill_request, request_id
        ):
            continue
        global prefilled_finished_requests
        original_req_id = request_id  # [WT][FIX] 保存 KV 使用的 request_id
        predictor_req_id = 'cmpl-' + request_id + '-0'  # predictor 使用
        predictor_meta = None
        timeout_limit = 180.0
        start_wait = asyncio.get_event_loop().time()
        # with prefilled_finished_cv:
        while True:
            # predictor_req_id 例如：cmpl-___prefill_addr_10.10.111.36:22010__5a13bb3bc84f45dbb48cffd13f321689-0
            if predictor_req_id in prefilled_finished_requests:
                predictor_meta = prefilled_finished_requests.pop(predictor_req_id)
                # print(
                #     f"[WT][Proxy] 444 Found meta for {predictor_req_id}",
                #     flush=True
                # )
                break
            if (asyncio.get_event_loop().time() - start_wait) > timeout_limit:
                # print(
                #     f"[WT][Proxy] 555 Timeout waiting for {predictor_req_id}",
                #     flush=True
                # )
                break

            await asyncio.sleep(0.001)

        # TODO:差一个根据多个decode实例负载选择decode实例逻辑
        
        
        global decode_instances
        global decode_cv

        with decode_cv:
            
            decode_list_keys = list(decode_instances.keys())
            # 使用 Monitor 进行智能决策，而不是 random
            decode_addr, min_score = monitor.select_best_instance(decode_list_keys,predictor_meta)
            
            decode_zmq_addr, _ = decode_instances[decode_addr]
        
        print(f'[WT]!!! Selected optimal decode_addr: {decode_addr} with metrics min_score: {min_score}', flush=True)
        # import random 
        # random_idx = random.randint(0, 10)
        # random_decode_idx = random_idx % len(list(decode_instances.items()))
        # print(f'[WT] Selected random_decode_idx: {random_decode_idx}',flush=True)
        
        # with decode_cv:
        #     decode_list = list(decode_instances.items())
        #     # decode_addr, decode_zmq_addr = decode_list[count % len(decode_list)]
        #     decode_addr, decode_zmq_addr = decode_list[random_decode_idx]
        #     decode_zmq_addr = decode_zmq_addr[0]
        # print(
        #     f"[WT] 666 handle_request count: {count}, [HTTP:{prefill_addr}, ZMQ:{prefill_zmq_addr}] 👉 [HTTP:{decode_addr}, ZMQ:{decode_zmq_addr}"
        #     ,flush=True
        # )
        count += 1
        # [WT]广播路由决策
        # 消息格式: "predictor_req_id|decode_zmq_addr"
        routing_msg = f"{predictor_req_id}|{decode_zmq_addr}"
        routing_pub.send_string(routing_msg)
        # print(
        #     f"[WT][Proxy] 777 Broadcast Route: {routing_msg}",
        #     flush=True
        # )
        # 更新predictor_req_id和request_id，加入decode地址信息
        predictor_req_id = predictor_req_id.replace(
            f"___prefill_addr_{prefill_zmq_addr}__",
            f"___prefill_addr_{prefill_zmq_addr}___decode_addr_{decode_zmq_addr}_"
        )
        request_id=request_id.replace(
            f"___prefill_addr_{prefill_zmq_addr}__",
            f"___prefill_addr_{prefill_zmq_addr}___decode_addr_{decode_zmq_addr}_"
        )
        # print(f'[WT] 888 request_id after update: {predictor_req_id}',flush=True)
        # print(f'[WT] 999 request_id after update: {request_id}',flush=True)
        # 发起 decode 请求
        original_request_data["predictor_meta"]= predictor_meta
        # print(f'[WT]<<<original_request_data: {original_request_data}',flush=True)
        generator = forward_request(
            f"http://{decode_addr}{request.path}", original_request_data, request_id
        )
        # 返回 decode 响应
        response = await make_response(generator)
        response.timeout = None
        return response
    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in disagg prefill proxy server",flush=True)
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
# [WT] end

# [WT]delay kvcache transfer 2025-12-27 22:20:14
if __name__ == "__main__":
    proxy_port=os.environ.get("PROXY_PORT", "28001")
    bench_port=os.environ.get("BENCH_PORT", "22006")
    t1 = start_service_discovery("0.0.0.0", proxy_port)
    t2 = threading.Thread(
        target=_listen_predictor_metadata,
        daemon=True,
    )
    t2.start()
    app.run(host="0.0.0.0", port=bench_port)
    t1.join()
    t2.join()
# [WT] end
