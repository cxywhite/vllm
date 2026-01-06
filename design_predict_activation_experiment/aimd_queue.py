# vllm_futurepast.py
"""
Past/Future scheduling adapter for vLLM v0.11.0.

Usage:
    vllm serve --scheduler-cls vllm_futurepast.FuturePastScheduler ...
Or import FuturePastRequestQueue and plug into create_request_queue(...) if you prefer source-file patch.
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Any
from collections import deque
from collections.abc import Iterable, Iterator
import bisect
import random
import numpy as np

import heapq
from abc import ABC, abstractmethod

from enum import Enum

from vllm.v1.request import Request
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
class AIMDRequestQueue(RequestQueue):
    def __init__(self) -> None:
        self._heap: list[tuple[float, Request]] = []
    def add_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap,
                       (request.arrival_time, request))
    def pop_request(self) -> Request:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, request = heapq.heappop(self._heap)
        return request

    def peek_request(self) -> Request:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, request = self._heap[0]
        return request

    def prepend_request(self, request: Request) -> None:
        """Add a request to the queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.
        
        Note: In a priority queue, there is no concept of prepending to the 
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        """Remove a specific request from the queue."""
        self._heap = [(t, r) for t, r in self._heap if r != request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._heap = [(t, r) for t, r in self._heap
                      if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[Request]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            _, request = heapq.heappop(heap_copy)
            yield request

    def __reversed__(self) -> Iterator[Request]:
        """Iterate over the queue in reverse priority order."""
        return reversed(list(self))
