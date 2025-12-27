# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import socket
import threading
import time
import uuid
from typing import Any
import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request
import json
import asyncio
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
                print(f"[WT][Proxy] 222 Received predictor meta for {req_id}: {meta}",flush=True)
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
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)
app = Quart(__name__)
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
        prefill_request = original_request_data.copy()
        prefill_request["max_tokens"] = 1
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
        print(f'[WT]333 handle_request request_id: {request_id}',flush=True)
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
                print(
                    f"[WT][Proxy] 444 Found meta for {predictor_req_id}",
                    flush=True
                )
                break
            if (asyncio.get_event_loop().time() - start_wait) > timeout_limit:
                print(
                    f"[WT][Proxy] 555 Timeout waiting for {predictor_req_id}",
                    flush=True
                )
                break

            await asyncio.sleep(0.001)

        # TODO:差一个根据多个decode实例负载选择decode实例逻辑
        
        import random 
        random_idx = random.randint(0, 10)
        global decode_instances
        global decode_cv
        random_decode_idx = random_idx % len(list(decode_instances.items()))
        print(f'[WT] Selected random_decode_idx: {random_decode_idx}',flush=True)
        
        with decode_cv:
            decode_list = list(decode_instances.items())
            # decode_addr, decode_zmq_addr = decode_list[count % len(decode_list)]
            decode_addr, decode_zmq_addr = decode_list[random_decode_idx]
            decode_zmq_addr = decode_zmq_addr[0]
        print(
            f"[WT] 666 handle_request count: {count}, [HTTP:{prefill_addr}, ZMQ:{prefill_zmq_addr}] 👉 [HTTP:{decode_addr}, ZMQ:{decode_zmq_addr}"
            ,flush=True
        )
        count += 1
        # [WT]广播路由决策
        # 消息格式: "predictor_req_id|decode_zmq_addr"
        routing_msg = f"{predictor_req_id}|{decode_zmq_addr}"
        routing_pub.send_string(routing_msg)
        print(
            f"[WT][Proxy] 777 Broadcast Route: {routing_msg}",
            flush=True
        )
        # 更新predictor_req_id和request_id，加入decode地址信息
        predictor_req_id = predictor_req_id.replace(
            f"___prefill_addr_{prefill_zmq_addr}__",
            f"___prefill_addr_{prefill_zmq_addr}___decode_addr_{decode_zmq_addr}_"
        )
        request_id=request_id.replace(
            f"___prefill_addr_{prefill_zmq_addr}__",
            f"___prefill_addr_{prefill_zmq_addr}___decode_addr_{decode_zmq_addr}_"
        )
        print(f'[WT] 888 request_id after update: {predictor_req_id}',flush=True)
        print(f'[WT] 999 request_id after update: {request_id}',flush=True)
        # 发起 decode 请求
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
    t1 = start_service_discovery("0.0.0.0", 28001)
    t2 = threading.Thread(
        target=_listen_predictor_metadata,
        daemon=True,
    )
    t2.start()
    app.run(host="0.0.0.0", port=22006)
    t1.join()
    t2.join()
# [WT] end
