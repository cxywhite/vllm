# vllm_futurepast.py
"""
Past/Future scheduling adapter for vLLM v0.11.0.

Usage:
    vllm serve --scheduler-cls vllm_futurepast.FuturePastScheduler ...
Or import FuturePastRequestQueue and plug into create_request_queue(...) if you prefer source-file patch.
"""

from typing import List, Tuple, Iterable, Optional, Any
from collections import deque
import bisect
import random
import numpy as np

# import vLLM classes (paths used in v0.11.0)
try:
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue, RequestQueue
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request
except Exception:
    # Fallback import locations (if package layout differs); adjust if necessary.
    from vllm.v1.core.sched.request_queue import FCFSRequestQueue, RequestQueue  # type: ignore
    from vllm.v1.core.sched.scheduler import Scheduler  # type: ignore
    Request = Any  # type: ignore
from vllm.logger import init_logger
logger = init_logger(__name__)
# ------------------------- PastFutureRequestQueue -------------------------
class PastFutureRequestQueue(FCFSRequestQueue):
    """
    基于 LightLLM 的 FuturePastReqQueue 思路的 RequestQueue 实现。
    继承 FCFSRequestQueue（保持 FCFS 的接口），并添加:
      - history_output_len (滑动窗口)
      - _sample_cache_list, _calc_max_token_num_needed, _init_cache_list, _can_add_new_req
      - record_output_lengths
    注意：使用时把 Scheduler.waiting 替换为该类的实例（或在 create_request_queue 中返回它）。
    """

    WINDOW_SIZE = 40
    MINIMUM_SAMPLES = 200
    MAXIMUM_LISTS = 5
    REVERSED = 0.05

    def __init__(self, engine_args: Optional[Any] = None,
                 prompt_cache_used_tokens: int = 0,
                 prompt_cache_req_num: int = 0):
        super().__init__()  # deque 初始化
        # 初始历史长度：尽量从 engine_args 读取保守值，否则用 2048
        initial_len = 2048
        try:
            if engine_args is not None:
                max_req_total_len = getattr(engine_args, "max_req_total_len", None)
                max_req_input_len = getattr(engine_args, "max_req_input_len", None)
                if max_req_total_len is not None and max_req_input_len is not None:
                    initial_len = int(max_req_total_len - max_req_input_len)
                else:
                    # initial_len = int(getattr(engine_args, "max_model_len", initial_len))
                    initial_len = 32768
        except Exception:
            pass

        self.history_output_len = deque([initial_len] * (self.WINDOW_SIZE // 2), maxlen=self.WINDOW_SIZE)
        # logger.info(f"[WT]&&& PastFutureRequestQueue initial history_output_len {self.history_output_len}")
        # prompt cache counters（如果外部需要）
        self.prompt_cache_used_tokens = int(prompt_cache_used_tokens or 0)
        self.prompt_cache_req_num = int(prompt_cache_req_num or 0)

        # 预算参数（可由外部覆盖）
        self.max_total_tokens = int(getattr(engine_args, "max_total_tokens", 733968))# 部署qwen2.5 gpu 0.7 max-num-sequences 1024 max-num-batched-tokens 32768 
        self.running_max_req_size = int(getattr(engine_args, "running_max_req_size", 1024))

        # # 暂停请求占用（如需维护，外部需更新）
        # self.pause_req_used_tokens = 0
        # self.pause_req_dict = {}

        # 内部 cache lists (用于多次采样估计)
        self._cache_len_lists: List[List[Tuple[int, int]]] = [[]]
        self.cache_len_list: List[Tuple[int, int]] = self._cache_len_lists[0]
        # self.cache_pause_reqs_used_tokens = self.pause_req_used_tokens
        # self.cache_pause_reqs_num = len(self.pause_req_dict)
        # self.waiting_req_list: List[Request] = []
    # ---------- 辅助：从 Request 提取常用字段（兼容多个字段名） ----------
    def _get_request_produced_len(self, req: Request) -> int:
        """返回 request 已生成（computed）的 token 数量（兼容多个字段名）。"""
        v = getattr(req, "num_computed_tokens", None)
        if v is not None:
            # logger.info(f"[WT]&&& using num_computed_tokens: {v}")
            return int(v)
        out_ids = getattr(req, "output_ids", None)
        if out_ids is not None:
            try:
                return int(len(out_ids))
            except Exception:
                pass
        v = getattr(req, "num_generated_tokens", None) or getattr(req, "generated_tokens", None)
        if v is not None:
            return int(v)
        return 0

    def _get_request_max_output_len(self, req: Request) -> int:
        """返回 request 的 max 输出长度（兼容多个字段名）"""
        for fname in ("max_tokens","max_output_len", "max_num_output_tokens", "max_new_tokens", "num_total_tokens", "num_tokens_total", "num_tokens"):
            v = getattr(req, fname, None)
            if v is not None:
                try:
                    # logger.info(f"[WT]&&& using {fname}: {v}")
                    return int(v)
                except Exception:
                    pass
        return 1024

    # ---------- LightLLM 的采样与估算逻辑 ----------
    def _sample_cache_list(self, reqs: List[Request], samples: int = 1) -> List[List[Tuple[int, int]]]:
        cache_len_lists = [[] for _ in range(samples)]
        his_Lo = sorted(list(self.history_output_len))
        # logger.info(f"[WT]&&& _sample_cache_list his_Lo:{his_Lo} ")
        for req in reqs:
            dl = self._get_request_produced_len(req)
            pos = bisect.bisect(his_Lo, dl)
            sample_range = [dl] + his_Lo[pos:]
            max_out = self._get_request_max_output_len(req)
            if sample_range[-1] < max_out:
                sample_range.append(max_out)

            for i in range(samples):
                if len(sample_range) < 2:
                    sampled = sample_range[0]
                else:
                    random_p = np.random.random() * (len(sample_range) - 1)
                    l_pos = int(random_p)
                    l_val, r_val = sample_range[l_pos:l_pos+2]
                    sampled = round(l_val + (r_val - l_val) * (random_p - l_pos))
                # logger.info(f'has_run:{dl}, sampled:{sampled}')
                has_run = self._get_request_produced_len(req)
                left_out = max(0, sampled - has_run)
                cache_len_lists[i].append((has_run, left_out))
        return cache_len_lists

    def _calc_max_token_num_needed(self, cache_len_list: List[Tuple[int, int]]) -> int:
        if not cache_len_list:
            return 0
        cache_len_list = sorted(cache_len_list, key=lambda x: -x[1])
        left_out_len_array = np.array([e[1] for e in cache_len_list], dtype=np.int64)
        has_run_len_array = np.array([e[0] for e in cache_len_list], dtype=np.int64)
        cum_run_len_array = np.cumsum(has_run_len_array)
        size_array = np.arange(1, len(cache_len_list) + 1, 1, dtype=np.int64)
        need_max_token_num = int((left_out_len_array * size_array + cum_run_len_array).max())
        return need_max_token_num

    def _init_cache_list(self, current_batch: Optional[List[Request]], is_busy: bool):
        """初始化 _cache_len_lists（在 schedule 的适当时机调用）"""
        # self.cache_pause_reqs_used_tokens = self.pause_req_used_tokens
        # self.cache_pause_reqs_num = len(self.pause_req_dict)
        if current_batch:
            n_lists = min(self.MAXIMUM_LISTS, int(self.MINIMUM_SAMPLES / max(1, len(current_batch))) + 1)
            self._cache_len_lists = self._sample_cache_list(current_batch, samples=n_lists)
            # logger.info(f"[WT]&&& _init_cache_list:{self._cache_len_lists} with : {n_lists} lists")
        else:
            self._cache_len_lists = [[]]
        self.cache_len_list = self._cache_len_lists[0]

    def _can_add_new_req(self, req: Request, is_busy: bool) -> bool:
        """基于采样列表估计将 req 加入后的最大 token 需求，判断是否超预算。"""
        need_max_token_nums = []
        for li in [list(l) for l in self._cache_len_lists]:
            newreq_output_len_sample = random.choice(list(self.history_output_len))
            has_run = self._get_request_produced_len(req)
            left_out = max(0, int(newreq_output_len_sample) - has_run)
            li.append((has_run, left_out))
            need_max_token_nums.append(self._calc_max_token_num_needed(li))
        need_max_token_num = int(max(need_max_token_nums) if need_max_token_nums else 0)

        # 如果 req 在 paused 状态，调整暂停计数（兼容字段名）
        # req_status = getattr(req, "req_status", None) or getattr(req, "status", None)
        # if req_status in ["PAUSED_AND_KVKEEP", "PAUSED_AND_OFFLOAD"]:
        #     used_tokens = getattr(req, "used_tokens", getattr(req, "get_used_tokens", 0))
        #     try:
        #         if callable(used_tokens):
        #             used_tokens = used_tokens()
        #     except Exception:
        #         used_tokens = 0
        #     self.cache_pause_reqs_used_tokens -= int(used_tokens or 0)
        #     self.cache_pause_reqs_num -= 1

        # ok_token_num = need_max_token_num < self.max_total_tokens * (1 - self.REVERSED) - self.cache_pause_reqs_used_tokens - self.prompt_cache_used_tokens
        # ok_req_num = len(self.cache_len_list) + self.cache_pause_reqs_num + self.prompt_cache_req_num <= self.running_max_req_size
        ok_token_num = need_max_token_num < self.max_total_tokens * (1 - self.REVERSED) - self.prompt_cache_used_tokens
        ok_req_num = self.prompt_cache_req_num <= self.running_max_req_size
        return bool(ok_token_num and ok_req_num)

    def record_output_lengths(self, lengths: List[int]):
        """把真实完成的长度记进历史窗口（供后续采样）"""
        if not lengths:
            return
        self.history_output_len.extend(lengths)
