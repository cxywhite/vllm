# predictor_worker_readyflag.py
import torch
import torch.nn as nn
import threading
import time
import json
import csv
from typing import Dict, Any, Optional
import torch.nn.functional as F
from transformers import BertModel,AutoTokenizer,AutoConfig
import numpy as np
import sys
# sys.path.append('/root/predict-schedule')
# from design_predict_activation_experiment.wt_metadata import Custom_Metadata
from wt_metadata import Custom_Metadata
import pickle
import os
import zmq
import logging
import queue
import atexit

class BertLengthDistributionModel_B(torch.nn.Module):
    def __init__(self, config, model_name, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(
            model_name,
            local_files_only=True,
            torch_dtype=torch.bfloat16
        )
        with torch.no_grad():
            cls_id = torch.tensor([101], device=self.bert.device)
            sep_id = torch.tensor([102], device=self.bert.device)

            self.register_buffer(
                'cls_inputs_embeds',
                self.bert.embeddings.word_embeddings(cls_id)
                    .to(dtype=torch.bfloat16)
            )
            self.register_buffer(
                'sep_inputs_embeds',
                self.bert.embeddings.word_embeddings(sep_id)
                    .to(dtype=torch.bfloat16)
            )

        
        self.proj = nn.Linear(input_dim, self.config.hidden_size)

        # 分类器，用于预测每个长度类别的概率
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_classes)
        )
        
        self.embedding_temperature = nn.Embedding(8, self.config.hidden_size)
        self.embedding_topp = nn.Embedding(6, self.config.hidden_size)
        self.embedding_topk = nn.Embedding(6, self.config.hidden_size)
        self.embedding_repetition_penalty = nn.Embedding(6, self.config.hidden_size)
        self.RMSNorm=nn.RMSNorm(normalized_shape=768, eps=1e-6)
        self.to(dtype=torch.bfloat16)
    def get_index_temperature(self,sampling_params):
        if round(float(sampling_params),2)==0.20:
            return 0
        elif round(float(sampling_params),2)==0.50:
            return 1
        elif round(float(sampling_params),2)==0.80:
            return 2
        elif round(float(sampling_params),2)==1.00:
            return 3
        elif round(float(sampling_params),2)==1.20:
            return 4
        elif round(float(sampling_params),2)==1.40:
            return 5
        elif round(float(sampling_params),2)==1.60:
            return 6
        elif round(float(sampling_params),2)==1.80:
            return 7
    def get_index_topp(self,sampling_params):
        if round(float(sampling_params),2)==0.10:
            return 0
        elif round(float(sampling_params),2)==0.30:
            return 1
        elif round(float(sampling_params),2)==0.50:
            return 2
        elif round(float(sampling_params),2)==0.70:
            return 3
        elif round(float(sampling_params),2)==0.90:
            return 4
        elif round(float(sampling_params),2)==1.00:
            return 5
    def get_index_topk(self,sampling_params):
        if int(sampling_params)==10:
            return 0
        elif int(sampling_params)==50:
            return 1
        elif int(sampling_params)==500:
            return 2
        elif int(sampling_params)==10000:
            return 3
        elif int(sampling_params)==50000:
            return 4
        elif int(sampling_params)==-1:
            return 5
    def get_index_repetition_penalty(self,sampling_params):
        if round(float(sampling_params),2)==1.00:
            return 0
        elif round(float(sampling_params),2)==1.10:
            return 1
        elif round(float(sampling_params),2)==1.20:
            return 2
        elif round(float(sampling_params),2)==1.30:
            return 3
        elif round(float(sampling_params),2)==1.40:
            return 4
        elif round(float(sampling_params),2)==1.50:
            return 5
    
    def forward(self, inputs_embeds, attention_mask, input_length,sampling_params):
        inputs_embeds = inputs_embeds.to(dtype=torch.bfloat16)
        assert inputs_embeds.dtype == self.proj.weight.dtype, \
        f"dtype mismatch: inputs {inputs_embeds.dtype}, weight {self.proj.weight.dtype}"
        inputs_embeds = self.proj(inputs_embeds)
        temperature=sampling_params['temperature']
        temperature_indices = torch.tensor([self.get_index_temperature(t) for t in temperature], dtype=torch.long, device=inputs_embeds.device)
        topp=sampling_params['topp']
        topp_indices = torch.tensor([self.get_index_topp(t) for t in topp], dtype=torch.long, device=inputs_embeds.device)
        topk=sampling_params['topk']
        topk_indices = torch.tensor([self.get_index_topk(t) for t in topk], dtype=torch.long, device=inputs_embeds.device)
        repetition_penalty = sampling_params['repetition_penalty']
        repetition_penalty_indices = torch.tensor([self.get_index_repetition_penalty(r) for r in repetition_penalty], dtype=torch.long, device=inputs_embeds.device)
        sep_indices = input_length + 1
        batch_size = inputs_embeds.size(0)
        batch_indices = torch.arange(batch_size, device=inputs_embeds.device)
        inputs_embeds[batch_indices, 0, :] = self.cls_inputs_embeds.squeeze(0)
        inputs_embeds[batch_indices, sep_indices, :] = self.sep_inputs_embeds.squeeze(0)
        outputs = self.bert(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        temp_embeds_temperature = self.embedding_temperature(temperature_indices)  # [batch_size, hidden_size]
        temp_embeds_topp = self.embedding_topp(topp_indices)  # [batch_size, hidden_size]
        temp_embeds_topk = self.embedding_topk(topk_indices)  # [batch_size, hidden_size]
        temp_embeds_repetition_penalty = self.embedding_repetition_penalty(repetition_penalty_indices)  # [batch_size, hidden_size]
        cls_output=cls_output + temp_embeds_temperature + temp_embeds_topp + temp_embeds_topk + temp_embeds_repetition_penalty
        # cls_output=self.RMSNorm(cls_output)
        logits = self.classifier(cls_output)  # [batch_size, n_classes]
        return logits

class PredictorWorker:
    def __init__(self,
                 ring_buffer,
                 max_req_len=510,
                 num_threads=8,
                 input_dim=4096
                 ):
        self.lock = threading.Lock()
        self.predict_num = 0
        self.ring = ring_buffer
        self.max_req_len = max_req_len
        self.num_threads = num_threads
        self.running = True
        self.input_dim = input_dim #llama3-8b-Instruct的hidden size qwen-2.5-7b-Instruct 3584
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Predictor
        self.hidden_dim=512
        self.num_classes=5
        self.model_dir = '/root/myshare/predict_project/input_ids_predictor_use/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir,local_files_only=True)
        self.config = AutoConfig.from_pretrained(self.model_dir,local_files_only=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.predictor = BertLengthDistributionModel_B(self.config, 'bert-base-uncased',self.input_dim, self.hidden_dim, self.num_classes).to(self.device)
        if self.input_dim == 4096:
            self.predictor.load_state_dict(torch.load('/root/myshare/predict_project/act_predictor_train/results/train_llama3-8b_act-15_max-8192_top-510_bert_H512_cls_10K_20251220-053810/model.pth'))#/root/myshare/predict_project/input_ids_predictor_use/model_weight/qwen_model.pth    
        elif self.input_dim == 3584:
            self.predictor.load_state_dict(torch.load('/root/myshare/predict_project/act_predictor_train/results/train_qwen2.5-7b-instruct_act-15_max-32768_top-510_bert_H512_cls_10K_20260123-084720_5k/model.pth'))
        else:
            raise ValueError("Unsupported hidden_dim for predictor model.")
        self.predictor = self.predictor.to(dtype=torch.bfloat16)
        self.predictor.eval()
        # 加载预训练权重（如果有）
        print(f"✅ PredictorWorker initialized on {next(self.predictor.parameters()).device}")
        print(f"   Model size: {sum(p.numel() for p in self.predictor.parameters()) / 1e6:.2f}M parameters")
        # -------- ZMQ INIT --------
        self.zmq_ctx = zmq.Context.instance()
        self.zmq_sock = self.zmq_ctx.socket(zmq.PUSH)

        # 高吞吐、低延迟配置
        self.zmq_sock.setsockopt(zmq.LINGER, 0)
        self.zmq_sock.setsockopt(zmq.SNDHWM, 4096)
        zmq_proxy_addr = "tcp://127.0.0.1:32323"  # 替换为实际的ZMQ代理地址
        self.zmq_sock.connect(zmq_proxy_addr)

        # -------- ASYNC TIMING CSV EXPORT --------
        default_csv_path = (
            f"/root/predict-schedule/vllm/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/wt_experiment/experiment_paper/4-Predict_latency/tmp/predict_timing_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        )
        self.timing_csv_path = os.getenv(
            "WT_PREDICT_TIMING_CSV", default_csv_path)
        self.verbose_timing = os.getenv("WT_TIMING_VERBOSE", "0") == "1"
        self._timing_queue: "queue.Queue[list[Any]]" = queue.Queue(maxsize=65536)
        self._timing_drop_count = 0
        self._timing_stop_event = threading.Event()
        self._timing_pending_lock = threading.Lock()
        self._timing_pending: dict[str, Custom_Metadata] = {}
        self._timing_pending_sleep_s = max(
            float(os.getenv("WT_TIMING_PENDING_CHECK_MS", "1")) / 1000.0,
            0.0005,
        )
        self._timing_writer_thread = threading.Thread(
            target=self._timing_writer_loop,
            daemon=True,
            name="wt-timing-csv-writer",
        )
        self._timing_writer_thread.start()
        self._timing_pending_thread = threading.Thread(
            target=self._timing_pending_loop,
            daemon=True,
            name="wt-timing-pending-flusher",
        )
        self._timing_pending_thread.start()
        atexit.register(self._stop_timing_writer)

    def start(self):
        for _ in range(self.num_threads):
            t = threading.Thread(
                target=self._worker_loop,
                daemon=True
            )
            t.start()
    def _worker_loop(self):
        while self.running:
            slot = self.ring.acquire_read_slot()
            if slot is None:
                time.sleep(0.0005)
                continue

            try:
                hidden_flat, importance_flat,meta = self.ring.read_batch(slot)
                self._run_predict(hidden_flat,importance_flat, meta)
            finally:
                self.ring.release_slot(slot)
    
    def _run_predict(self,
                    hidden_flat: torch.Tensor,  # [N,4096]
                    importance_flat: torch.Tensor,  # [N]
                    meta: list[Custom_Metadata]):
        lengths  = []
        active_meta: list[Custom_Metadata] = []
        token_ranges = []
        temp_list = []
        topp_list = []
        topk_list = []
        repetition_penalty_list = []
        for m in meta:
            (start, end) = m.token_range
            if end <= start:
                continue

            seq_len = end - start
            lengths.append(min(self.max_req_len, seq_len))
            active_meta.append(m)
            token_ranges.append((start, end))
            temp_list.append(float(m.temperature))
            topp_list.append(float(m.top_p))
            topk_list.append(int(m.top_k))
            repetition_penalty_list.append(float(m.repetition_penalty))
            
        if not token_ranges:
            return
        batch_size = len(token_ranges)
        max_len    = max(lengths)
        hidden_dim = hidden_flat.shape[1]
        device     = hidden_flat.device
        # [B, L+2, D]
        inputs_embeds = torch.zeros(
            batch_size,
            max_len + 2,
            hidden_dim,
            device=device,
            dtype=torch.bfloat16
            
        )
        attention_mask = torch.zeros(
            batch_size,
            max_len + 2,
            device=device,
            dtype=torch.bool
        )
        for i, (start, end) in enumerate(token_ranges):
            seq = hidden_flat[start:end]
            imp = importance_flat[start:end]
            assert seq.dim() == 2
            assert imp.dim() == 1
            assert seq.shape[0] == imp.shape[0]

            k = lengths[i]
            topk_idx = torch.topk(imp, k=k, sorted=False).indices
            topk_idx, _ = torch.sort(topk_idx)
            inputs_embeds[i, 1:1 + k] = seq[topk_idx]
            attention_mask[i, :k + 2] = True
        lengths = torch.tensor(lengths, device=device, dtype=torch.long)
        sampling_params = {
                        'temperature': temp_list,
                        'topp': topp_list,
                        'topk': topk_list,
                        'repetition_penalty': repetition_penalty_list
                    }
        attention_mask = attention_mask.to(device=inputs_embeds.device)
        with torch.no_grad():
            logits = self.predictor(inputs_embeds, attention_mask, lengths, sampling_params)
            probabilities = F.softmax(logits, dim=-1).to(torch.bfloat16)
            # [ 393. 1145. 7775. 8167.]20%, 40%, 60%, 80% 分位数
            # bucket_stats = [
            #     { "low": 0,    "high": 393,  "mean": 219 },
            #     { "low": 393,  "high": 1145, "mean": 645 },
            #     { "low": 1145, "high": 7775, "mean": 3680 },
            #     { "low": 7775, "high": 8167, "mean": 8078 },
            #     { "low": 8167, "high": 8190, "mean": 8179 },
            # ]
            # bucket_high=torch.tensor([393,1145,7775,8167,8190],device=device,dtype=torch.bfloat16)
            # bucket_mean=torch.tensor([219,645,3680,8078,8179],device=device,dtype=torch.bfloat16)
            # lambdas=torch.tensor([0.2, 0.2, 0.5, 0.8, 1.0],device=device,dtype=torch.bfloat16)
            # mu_eff = (1 - lambdas) * bucket_mean + lambdas * bucket_high
            if self.input_dim == 4096:
                bucket_low=torch.tensor([0,393,1145,7775,8167],device=device,dtype=torch.bfloat16)
                # bucket_mean=torch.tensor([219,645,3680,8078,8179],device=device,dtype=torch.bfloat16)
                # bucket两边的mean值(low+high)/2
                bucket_median=torch.tensor([196,769,4460,7971,8180],device=device,dtype=torch.bfloat16)
            elif self.input_dim == 3584:
                bucket_low=torch.tensor([0,281,704,26232,32730],device=device,dtype=torch.bfloat16)
                # bucket_mean=torch.tensor([127,464,5512,32385,32751],device=device,dtype=torch.bfloat16)
                # bucket两边的mean值(low+high)/2
                bucket_median=torch.tensor([140,482,4668,29489,32751],device=device,dtype=torch.bfloat16)
            
            # lambdas=torch.tensor([1.0, 0.8, 0.5, 0.2, 0.2],device=device,dtype=torch.bfloat16)
            # mu_eff = (1 - lambdas) * bucket_low  + lambdas * bucket_median
            # predict_len=(probabilities * mu_eff).sum(dim=-1)
            # 计算预测长度的miu
            miu=(probabilities * bucket_median).sum(dim=-1)
            sigma=torch.sqrt((probabilities * (bucket_median - miu.unsqueeze(-1))**2).sum(dim=-1))
            predict_len=miu + sigma
            # predict_len向上取整为int型整数
            predict_end_ts_ns = time.perf_counter_ns()
            timing_rows = []
            for i in range(len(active_meta)):
                assert active_meta[i].predict_output_len==None
                active_meta[i].predict_output_len = int(predict_len[i].item())
                active_meta[i].predict_end_ts_ns = predict_end_ts_ns
                row = self._build_complete_timing_row(active_meta[i])
                if row is not None:
                    timing_rows.append(row)
                else:
                    self._defer_timing_meta(active_meta[i])

            self._enqueue_timing_rows(timing_rows)
            # Opportunistically flush deferred rows without blocking hot path.
            self._drain_ready_timing_meta(max_items=64)
            self._send_metadata_to_proxy(active_meta)
            with self.lock:
                self.predict_num += batch_size
                print(f"[WT] Total predictions made: {self.predict_num} predictor_worker_readyflag.py")

    def _build_complete_timing_row(
        self,
        meta: Custom_Metadata,
    ) -> Optional[list[Any]]:
        if (meta.predict_start_ts_ns is None or
                meta.prefill_end_ts_ns is None or
                meta.predict_end_ts_ns is None):
            return None

        prefill_tail_ms = (
            meta.prefill_end_ts_ns - meta.predict_start_ts_ns) / 1e6
        predict_ms = (
            meta.predict_end_ts_ns - meta.predict_start_ts_ns) / 1e6
        covered = meta.predict_end_ts_ns <= meta.prefill_end_ts_ns

        if self.verbose_timing:
            print(
                f"[WT][Overlap] req_id={meta.req_id} "
                f"predict_ms={predict_ms:.3f} "
                f"prefill_tail_ms={prefill_tail_ms:.3f} "
                f"covered={covered}")

        return [
            meta.req_id,
            meta.predict_start_ts_ns,
            meta.prefill_end_ts_ns,
            meta.predict_end_ts_ns,
            prefill_tail_ms,
            predict_ms,
            covered,
        ]

    def _timing_pending_key(self, meta: Custom_Metadata) -> str:
        return f"{meta.req_id}::{id(meta)}"

    def _defer_timing_meta(self, meta: Custom_Metadata) -> None:
        key = self._timing_pending_key(meta)
        with self._timing_pending_lock:
            self._timing_pending[key] = meta

    def _drain_ready_timing_meta(self, max_items: int = 256) -> int:
        drained = 0
        ready_rows: list[list[Any]] = []

        with self._timing_pending_lock:
            keys = list(self._timing_pending.keys())

        for key in keys:
            if drained >= max_items:
                break

            with self._timing_pending_lock:
                meta = self._timing_pending.get(key)

            if meta is None:
                continue

            row = self._build_complete_timing_row(meta)
            if row is None:
                continue

            with self._timing_pending_lock:
                removed = self._timing_pending.pop(key, None)
            if removed is None:
                continue

            ready_rows.append(row)
            drained += 1

        if ready_rows:
            self._enqueue_timing_rows(ready_rows)

        return drained

    def _timing_pending_loop(self) -> None:
        while True:
            if self._timing_stop_event.is_set():
                self._drain_ready_timing_meta(max_items=4096)
                with self._timing_pending_lock:
                    if not self._timing_pending:
                        break
                time.sleep(self._timing_pending_sleep_s)
                continue

            drained = self._drain_ready_timing_meta(max_items=256)
            if drained == 0:
                time.sleep(self._timing_pending_sleep_s)

    def _enqueue_timing_rows(self, rows: list[list[Any]]) -> None:
        if not rows:
            return
        for row in rows:
            try:
                self._timing_queue.put_nowait(row)
            except queue.Full:
                self._timing_drop_count += 1
                if self._timing_drop_count % 1000 == 0:
                    print(
                        f"[WT][TimingCSV] dropped rows={self._timing_drop_count} "
                        "(queue full)")

    def _ensure_timing_csv_header(self) -> None:
        csv_dir = os.path.dirname(self.timing_csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        need_header = (not os.path.exists(self.timing_csv_path) or
                       os.path.getsize(self.timing_csv_path) == 0)
        if not need_header:
            return

        with open(self.timing_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "req_id",
                "predict_start_ts_ns",
                "prefill_end_ts_ns",
                "predict_end_ts_ns",
                "prefill_tail_ms",
                "predict_ms",
                "predict_hidden_by_prefill",
            ])

    def _timing_writer_loop(self) -> None:
        self._ensure_timing_csv_header()
        with open(self.timing_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            while True:
                if self._timing_stop_event.is_set() and self._timing_queue.empty():
                    break

                batch: list[list[Any]] = []
                try:
                    row = self._timing_queue.get(timeout=0.2)
                    batch.append(row)
                except queue.Empty:
                    continue

                while len(batch) < 512:
                    try:
                        batch.append(self._timing_queue.get_nowait())
                    except queue.Empty:
                        break

                writer.writerows(batch)
                f.flush()

    def _stop_timing_writer(self) -> None:
        self._timing_stop_event.set()
        if self._timing_pending_thread.is_alive():
            self._timing_pending_thread.join(timeout=1.0)
        if self._timing_writer_thread.is_alive():
            self._timing_writer_thread.join(timeout=1.0)

    def _send_metadata_to_proxy(self, meta: list[Custom_Metadata]):
        """
        Non-blocking send of predictor metadata to proxy.
        """
        try:
            meta_dicts = [m.to_dict() for m in meta]
            payload = json.dumps(meta_dicts)
            self.zmq_sock.send_string(payload, zmq.NOBLOCK)
        except zmq.Again:
            # proxy 忙，直接丢弃或计数
            print("[WT]ZMQ send failed: HWM reached predictor_worker_readyflag.py")

# if __name__ == "__main__":
#     print("[WT] hello",flush=True)
#     print("Starting PredictorWorker main")
#     worker = PredictorWorker()
#     worker.run()