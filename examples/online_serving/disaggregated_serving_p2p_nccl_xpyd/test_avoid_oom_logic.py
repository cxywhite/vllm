#!/usr/bin/env python3
"""[wt] avoid oom tests for proxy admission and memory pool accounting."""

import asyncio
import importlib.util
import logging
import sys
import threading
import time
import types
import unittest
from pathlib import Path


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ensure_proxy_stubs():
    """[wt] avoid oom stub optional runtime deps for pure-logic tests."""
    if "aiohttp" not in sys.modules:
        aiohttp = types.ModuleType("aiohttp")

        class _ClientTimeout:
            def __init__(self, total=None):
                self.total = total

        class _DummySession:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        aiohttp.ClientTimeout = _ClientTimeout
        aiohttp.ClientSession = _DummySession
        sys.modules["aiohttp"] = aiohttp

    if "httpx" not in sys.modules:
        httpx = types.ModuleType("httpx")

        class _Timeout:
            def __init__(self, *args, **kwargs):
                pass

        class _Limits:
            def __init__(self, *args, **kwargs):
                pass

        class _AsyncClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, tb):
                return False

        httpx.Timeout = _Timeout
        httpx.Limits = _Limits
        httpx.AsyncClient = _AsyncClient
        sys.modules["httpx"] = httpx

    if "msgpack" not in sys.modules:
        msgpack = types.ModuleType("msgpack")
        msgpack.loads = lambda data: data
        msgpack.dumps = lambda data: data
        sys.modules["msgpack"] = msgpack

    if "zmq" not in sys.modules:
        zmq = types.ModuleType("zmq")
        zmq.POLLIN = 1

        class _DummyContext:
            def socket(self, *_args, **_kwargs):
                return object()

        class _DummyPoller:
            def register(self, *_args, **_kwargs):
                return None

        zmq.Context = _DummyContext
        zmq.Poller = _DummyPoller
        sys.modules["zmq"] = zmq

    if "quart" not in sys.modules:
        quart = types.ModuleType("quart")

        class _DummyQuart:
            def __init__(self, *args, **kwargs):
                pass

            def before_serving(self, fn):
                return fn

            def route(self, *_args, **_kwargs):
                def deco(fn):
                    return fn
                return deco

        async def _dummy_make_response(generator):
            return types.SimpleNamespace(timeout=None, generator=generator)

        class _DummyRequest:
            path = "/v1/chat/completions"

            async def get_json(self):
                return {}

        quart.Quart = _DummyQuart
        quart.jsonify = lambda x, *args, **kwargs: x
        quart.make_response = _dummy_make_response
        quart.request = _DummyRequest()
        sys.modules["quart"] = quart


def _ensure_pool_stubs():
    if "vllm" not in sys.modules:
        vllm = types.ModuleType("vllm")
        vllm.__path__ = []
        sys.modules["vllm"] = vllm

    if "vllm.logger" not in sys.modules:
        logger_mod = types.ModuleType("vllm.logger")
        logger_mod.init_logger = lambda name: logging.getLogger(name)
        sys.modules["vllm.logger"] = logger_mod


class AvoidOomTests(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[3]

        _ensure_proxy_stubs()
        proxy_path = (
            repo_root
            / "examples"
            / "online_serving"
            / "disaggregated_serving_p2p_nccl_xpyd"
            / "disagg_proxy_p2p_nccl_xpyd_reviewfix.py"
        )
        if not proxy_path.exists():
            proxy_path = (
                repo_root
                / "examples"
                / "online_serving"
                / "disaggregated_serving_p2p_nccl_xpyd"
                / "disagg_proxy_p2p_nccl_xpyd.py"
            )
        cls.proxy = _load_module("proxy_avoid_oom_test", proxy_path)

        _ensure_pool_stubs()
        pool_path = (
            repo_root
            / "vllm"
            / "distributed"
            / "kv_transfer"
            / "kv_connector"
            / "v1"
            / "p2p"
            / "tensor_memory_pool.py"
        )
        cls.pool_mod = None
        cls.pool_import_error = None
        try:
            cls.pool_mod = _load_module("pool_avoid_oom_test", pool_path)
        except ModuleNotFoundError as exc:
            cls.pool_import_error = exc

    def setUp(self):
        self.proxy.decode_instances.clear()
        self.proxy.decode_capacity.clear()
        with self.proxy.waiting_cv:
            self.proxy.waiting_queue.clear()
        self.proxy.require_capacity_report = True
        self.proxy.waiting_timeout_s = 0.2
        self.proxy.waiting_poll_s = 0.01

    def test_estimate_request_kv_bytes_is_positive(self):
        req = {"prompt_token_ids": [1] * 128}
        kv_bytes = self.proxy._estimate_request_kv_bytes(req)
        self.assertGreater(kv_bytes, 0)

    def test_estimate_request_kv_bytes_matches_formula(self):
        req = {"prompt_token_ids": [1] * 64}
        cfg = self.proxy.monitor.config
        prompt_tokens = 64
        kv_dim = (cfg.hidden_size // cfg.num_attention_heads) * cfg.num_key_value_heads
        base_kv_bytes = (
            2 * cfg.num_hidden_layers * prompt_tokens * kv_dim * cfg.bytes_per_param
        )
        # [wt] avoid oom expected admission bytes include capacity_safety_ratio.
        expected_kv_bytes = int(base_kv_bytes * self.proxy.capacity_safety_ratio)
        got = self.proxy._estimate_request_kv_bytes(req)
        self.assertEqual(got, max(1, expected_kv_bytes))

    def test_rr_capacity_fallback(self):
        now = time.time() + 10
        self.proxy.decode_instances["d1"] = ("z1", now)
        self.proxy.decode_instances["d2"] = ("z2", now)
        self.proxy.decode_capacity["d1"] = {
            "kv_buffer_free_bytes": 100.0,
            "pool_free_bytes": 100.0,
        }
        self.proxy.decode_capacity["d2"] = {
            "kv_buffer_free_bytes": 2000.0,
            "pool_free_bytes": 2000.0,
        }

        decode_addr, decode_zmq, reason = self.proxy._select_decode_rr_by_capacity(
            rr_index=0,
            required_bytes=500,
        )
        self.assertEqual(decode_addr, "d2")
        self.assertEqual(decode_zmq, "z2")
        self.assertIn("rr_step", reason)

    def test_single_target_capacity_blocks_sum_only_case(self):
        # [wt] avoid oom decode stores tensor in one target (buffer or pool),
        # so free-space admission must reject cases where only the sum fits.
        now = time.time() + 10
        self.proxy.decode_instances["d1"] = ("z1", now)
        self.proxy.decode_capacity["d1"] = {
            "kv_buffer_free_bytes": 700.0,
            "pool_free_bytes": 700.0,
        }
        decode_addr, decode_zmq, reason = self.proxy._select_decode_rr_by_capacity(
            rr_index=0,
            required_bytes=1000,
        )
        self.assertIsNone(decode_addr)
        self.assertIsNone(decode_zmq)
        self.assertEqual(reason, "capacity_insufficient")

    def test_single_target_capacity_accepts_pool_only_fit(self):
        now = time.time() + 10
        self.proxy.decode_instances["d1"] = ("z1", now)
        self.proxy.decode_capacity["d1"] = {
            "kv_buffer_free_bytes": 200.0,
            "pool_free_bytes": 1500.0,
        }
        decode_addr, decode_zmq, reason = self.proxy._select_decode_rr_by_capacity(
            rr_index=0,
            required_bytes=1000,
        )
        self.assertEqual(decode_addr, "d1")
        self.assertEqual(decode_zmq, "z1")
        self.assertIn("rr_step", reason)

    def test_impossible_request_uses_single_target_capacity(self):
        now = time.time() + 10
        self.proxy.decode_instances["d1"] = ("z1", now)
        self.proxy.decode_capacity["d1"] = {
            "kv_buffer_capacity_bytes": 1024.0,
            "pool_total_bytes": 1024.0,
        }
        # With single-target semantics, request larger than each target is impossible.
        self.assertTrue(self.proxy._is_impossible_request(1500))
        self.assertFalse(self.proxy._is_impossible_request(1024))

    async def test_wait_queue_wakeup_after_capacity_update(self):
        now = time.time() + 10
        self.proxy.decode_instances["d1"] = ("z1", now)
        self.proxy.decode_capacity["d1"] = {
            "kv_buffer_free_bytes": 0.0,
            "pool_free_bytes": 0.0,
            "kv_buffer_capacity_bytes": 1024.0,
            "pool_total_bytes": 1024.0,
        }
        self.proxy.waiting_timeout_s = 1.0

        async def _make_capacity_available():
            await asyncio.sleep(0.05)
            with self.proxy.decode_cv:
                self.proxy.decode_capacity["d1"]["kv_buffer_free_bytes"] = 4096.0
                self.proxy.decode_capacity["d1"]["pool_free_bytes"] = 4096.0
            with self.proxy.waiting_cv:
                self.proxy.waiting_cv.notify_all()

        task = asyncio.create_task(
            self.proxy._acquire_decode_with_queue(rr_index=0, required_bytes=1024)
        )
        await _make_capacity_available()
        decode_addr, decode_zmq, _reason = await task
        self.assertEqual(decode_addr, "d1")
        self.assertEqual(decode_zmq, "z1")

    def test_tensor_memory_pool_capacity_accounting(self):
        if self.pool_mod is None:
            self.skipTest(f"pool test skipped: {self.pool_import_error}")
        pool = self.pool_mod.TensorMemoryPool(max_block_size=4096, min_block_size=512)
        addr = pool.allocate(600)
        try:
            used = pool.get_used_bytes()
            free = pool.get_free_bytes()
            total = pool.get_total_bytes()
            self.assertEqual(total, 4096)
            self.assertEqual(used, 1024)
            self.assertEqual(free, 3072)
        finally:
            pool.free(addr)
        self.assertEqual(pool.get_used_bytes(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
