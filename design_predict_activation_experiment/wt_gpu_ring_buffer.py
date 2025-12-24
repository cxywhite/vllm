# gpu_ring_buffer.py

import torch
import numpy as np
import threading
import time
import sys
sys.path.append('/root/predict-schedule')
from design_predict_activation_experiment.wt_metadata import Custom_Metadata
EMPTY   = 0
WRITING = 1
READY   = 2
READING = 3


class GPURingBuffer:
    def __init__(self,
                 num_slots: int,
                 max_tokens: int,
                 hidden_dim: int,
                 device: torch.device):

        self.num_slots   = num_slots
        self.max_tokens = max_tokens
        self.hidden_dim = hidden_dim
        self.device     = device

        # ===== GPU data plane =====
        self.hidden = torch.empty(
            (num_slots, max_tokens, hidden_dim),
            device=device,
            dtype=torch.float32
        )
        self.importance = torch.empty(
            (num_slots, max_tokens),
            device=device,
            dtype=torch.float32
        )
        # ===== CPU control plane =====
        self.num_tokens = np.zeros(num_slots, dtype=np.int32)
        self.meta       = [None] * num_slots
        self.state      = np.zeros(num_slots, dtype=np.int32)
        self.batch_id   = np.zeros(num_slots, dtype=np.int64)

        self._seq  = 0
        self._lock = threading.Lock()

    # ---------------- Prefill side ----------------

    def _acquire_write_slot(self):
        while True:
            with self._lock:
                for i in range(self.num_slots):
                    if self.state[i] == EMPTY:
                        self.state[i] = WRITING
                        return i
            time.sleep(0.0005)

    def write_batch(self,
                    hidden_states: torch.Tensor,  # [N, hidden_dim]
                    importance: torch.Tensor,      # [N]
                    meta: list[Custom_Metadata]):

        n = hidden_states.shape[0]
        assert n <= self.max_tokens, \
            f"num_tokens {n} > slot capacity {self.max_tokens}"

        slot = self._acquire_write_slot()

        # GPU → GPU copy
        self.hidden[slot, :n].copy_(hidden_states)
        self.importance[slot, :n].copy_(importance)
        with self._lock:
            self.num_tokens[slot] = n
            self.meta[slot]       = meta
            self.batch_id[slot]   = self._seq
            self._seq            += 1
            self.state[slot]      = READY

    # ---------------- Predictor side ----------------

    def acquire_read_slot(self):
        with self._lock:
            ready = [
                (i, self.batch_id[i])
                for i in range(self.num_slots)
                if self.state[i] == READY
            ]
            if not ready:
                return None

            slot, _ = min(ready, key=lambda x: x[1])
            self.state[slot] = READING
            return slot

    def read_batch(self, slot: int):
        n    = int(self.num_tokens[slot])
        meta = self.meta[slot]
        return self.hidden[slot, :n], self.importance[slot,:n],meta

    def release_slot(self, slot: int):
        with self._lock:
            self.meta[slot]       = None
            self.num_tokens[slot] = 0
            self.state[slot]      = EMPTY
