import itertools
import aiohttp
import asyncio
import os
import uuid
import json
from quart import Quart, request, Response, stream_with_context

app = Quart(__name__)

# --- 彻底消除超时限制 ---
app.config['RESPONSE_TIMEOUT'] = None  # 永久等待后端响应
app.config['BODY_TIMEOUT'] = None      # 永久等待接收请求体
app.config['KEEP_ALIVE_TIMEOUT'] = 3600 # 保持连接活跃

# 全局 Session，配置长连接池
_session = None

async def get_session():
    global _session
    if _session is None:
        # 增加连接池大小，配置 TCP Keep-alive 防止链路断开
        connector = aiohttp.TCPConnector(
            limit=1000, 
            keepalive_timeout=600,
            force_close=False
        )
        _session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None, connect=60, sock_read=None)
        )
    return _session

# 后端 vLLM 实例列表
# INSTANCES = ["10.10.111.43:10070","10.10.111.43:10071","10.10.111.43:10072","10.10.111.43:10073","10.10.111.43:10074","10.10.111.43:10075","127.0.0.1:10080", "127.0.0.1:10081", "127.0.0.1:10082", "127.0.0.1:10083", "127.0.0.1:10084", "127.0.0.1:10085", "127.0.0.1:10086", "127.0.0.1:10087"]
# INSTANCES = ["10.10.111.43:10070","10.10.111.43:10071","10.10.111.43:10072","10.10.111.43:10073","10.10.111.43:10074","10.10.111.43:10075"]
INSTANCES=["10.10.111.43:10072","10.10.111.43:10073"]
rr = itertools.cycle(INSTANCES)

@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle():
    data = await request.get_json()
    inst = next(rr)
    target_url = f"http://{inst}{request.path}"
    # 如果是发往ip为10.10.111.43的请求，更新data中的model字段为"/root/share/models/Meta-Llama-3-8B-Instruct"
    if inst.startswith("10.10.111.43"):
        data["model"] = "/root/share/models/Meta-Llama-3-8B-Instruct"
        print(f"DEBUG: Forwarding to {inst} with data: {data}")  # 只打印前500字符避免日志过大
    req_id = data.get("req_id") or str(uuid.uuid4())
    session = await get_session()
    
    headers = {
        "Authorization": request.headers.get("Authorization", ""),
        "Content-Type": "application/json",
        "X-Request-Id": str(req_id)
    }

    print(f"DEBUG: Forwarding {req_id} to {target_url}")

    async def generate():
        try:
            async with session.post(url=target_url, json=data, headers=headers) as response:
                if response.status != 200:
                    err_text = await response.text()
                    print(f"ERROR: Backend {inst} returned {response.status}: {err_text}")
                    yield json.dumps({"error": err_text, "status": response.status}).encode()
                    return

                # 直接按原始数据块转发，保证 SSE 协议不被破坏
                async for chunk in response.content.iter_any():
                    yield chunk
        except Exception as e:
            print(f"CRITICAL: Proxy error for {req_id}: {str(e)}")

    return Response(
        stream_with_context(generate)(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )

if __name__ == "__main__":
    # 使用单进程异步模式处理高并发
    app.run(host="0.0.0.0", port=9003, use_reloader=False)