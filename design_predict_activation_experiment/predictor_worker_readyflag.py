# predictor_worker_readyflag.py
import torch
import torch.nn as nn
import threading
import time
import json
from typing import Dict, Any, Optional
import torch.nn.functional as F
from transformers import BertModel,AutoTokenizer,AutoConfig
import numpy as np
import sys
sys.path.append('/root/vllm')
from examples.online_serving.disaggregated_serving_p2p_nccl_xpyd.wt_shared_gpu_buffer import SharedGPUBufferManager,write_ready_flag,read_ready_flag
sys.path.append('/root/predict-schedule')
from design_predict_activation_experiment.wt_metadata import Custom_Metadata
import pickle
import os
import zmq
class BertLengthDistributionModel_B(torch.nn.Module):
    def __init__(self, config, model_name, input_dim, hidden_dim, n_classes):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(model_name,local_files_only=True)
        self.register_buffer('cls_inputs_embeds', self.bert.embeddings.word_embeddings(torch.tensor([101])).detach())
        self.register_buffer('sep_inputs_embeds', self.bert.embeddings.word_embeddings(torch.tensor([102])).detach())
        
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
                 num_threads=8):
        self.predict_num = 0
        self.ring = ring_buffer
        self.max_req_len = max_req_len
        self.num_threads = num_threads
        self.running = True

        self.input_dim = 4096 #llama3-8b-Instruct的hidden size qwen-2.5-7b-Instruct 3584
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Predictor
        self.hidden_dim=512
        self.num_classes=5
        self.model_dir = '/root/myshare/predict_project/input_ids_predictor_use/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir,local_files_only=True)
        self.config = AutoConfig.from_pretrained(self.model_dir,local_files_only=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.predictor = BertLengthDistributionModel_B(self.config, 'bert-base-uncased',self.input_dim, self.hidden_dim, self.num_classes).to(self.device)
        self.predictor.eval()
        # 加载预训练权重（如果有）
        self.predictor.load_state_dict(torch.load('/root/myshare/predict_project/act_predictor_train/results/train_llama3-8b_act-15_max-8192_top-510_bert_H512_cls_10K_20251220-053810/model.pth'))#/root/myshare/predict_project/input_ids_predictor_use/model_weight/qwen_model.pth    
        print(f"✅ PredictorWorker initialized on {next(self.predictor.parameters()).device}")
        print(f"   Model size: {sum(p.numel() for p in self.predictor.parameters()) / 1e6:.2f}M parameters")
        # -------- ZMQ INIT --------
        self.zmq_ctx = zmq.Context.instance()
        self.zmq_sock = self.zmq_ctx.socket(zmq.PUSH)

        # 高吞吐、低延迟配置
        self.zmq_sock.setsockopt(zmq.LINGER, 0)
        self.zmq_sock.setsockopt(zmq.SNDHWM, 4096)
        zmq_proxy_addr = "tcp://0.0.0.0:32323"  # 替换为实际的ZMQ代理地址
        self.zmq_sock.connect(zmq_proxy_addr)

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
    def _send_metadata_to_proxy(self, meta: list[Custom_Metadata]):
        """
        Non-blocking send of predictor metadata to proxy.
        """
        try:
            print(f'[WT] Sending metadata to proxy via ZMQ: {meta}')
            meta_dicts = [m.to_dict() for m in meta]
            payload = json.dumps(meta_dicts)
            self.zmq_sock.send_string(payload, zmq.NOBLOCK)
        except zmq.Again:
            # proxy 忙，直接丢弃或计数
            print("[WT][Predictor] ZMQ send failed: HWM reached")

    def _run_predict(self,
                    hidden_flat: torch.Tensor,  # [N,4096]
                    importance_flat: torch.Tensor,  # [N]
                    meta: list[Custom_Metadata]):
        req_seqs = []
        lengths  = []
        temp_list = []
        topp_list = []
        topk_list = []
        repetition_penalty_list = []
        for m in meta:
            (start, end) = m.token_range

            seq = hidden_flat[start:end]
            imp = importance_flat[start:end]
            assert seq.dim() == 2
            assert imp.dim() == 1
            assert seq.shape[0] == imp.shape[0]

            k = min(self.max_req_len, seq.shape[0])
            
            topk_idx = torch.topk(imp, k=k).indices
            # print(f'WT111{req_id} seq len: {seq.shape[0]}, topk_idx: {topk_idx}.')
            selected_seq = seq[topk_idx]
            # print(f'WT222 selected_seq shape: {selected_seq.shape}.')
            req_seqs.append(selected_seq)
            lengths.append(selected_seq.shape[0])
            temp_list.append(float(m.temperature))
            topp_list.append(float(m.top_p))
            topk_list.append(int(m.top_k))
            repetition_penalty_list.append(float(m.repetition_penalty))
            
        if not req_seqs:
            return

        batch_size = len(req_seqs)
        max_len    = max(lengths)
        hidden_dim = hidden_flat.shape[1]
        device     = hidden_flat.device

        # [B, L+2, D]
        inputs_embeds = torch.zeros(
            batch_size,
            max_len + 2,
            hidden_dim,
            device=device
        )

        attention_mask = torch.zeros(
            batch_size,
            max_len + 2,
            device=device,
            dtype=torch.long
        )

        for i, seq in enumerate(req_seqs):
            L = seq.shape[0]
            inputs_embeds[i, 1:1+L] = seq
            attention_mask[i, :L+2] = 1

        lengths = torch.tensor(lengths, device=device)
        sampling_params = {
                        'temperature': temp_list,
                        'topp': topp_list,
                        'topk': topk_list,
                        'repetition_penalty': repetition_penalty_list
                    }
        with torch.no_grad():
            logits = self.predictor(inputs_embeds, attention_mask, lengths, sampling_params)
            print(f"[WT] Predictor logits shape: {logits.shape}, values: {logits}")
            probabilities = F.softmax(logits, dim=-1)
            print(f"[WT] Predictor probabilities: {probabilities}")
            # [ 393. 1145. 7775. 8167.]20%, 40%, 60%, 80% 分位数
            # bucket_stats = [
            #     { "low": 0,    "high": 393,  "mean": 219 },
            #     { "low": 393,  "high": 1145, "mean": 645 },
            #     { "low": 1145, "high": 7775, "mean": 3680 },
            #     { "low": 7775, "high": 8167, "mean": 8078 },
            #     { "low": 8167, "high": 8190, "mean": 8179 },
            # ]
            bucket_high=torch.tensor([393,1145,7775,8167,8190],device=device)
            bucket_mean=torch.tensor([219,645,3680,8078,8179],device=device)
            lambdas=torch.tensor([0.2, 0.2, 0.5, 0.8, 1.0],device=device)
            mu_eff = (1 - lambdas) * bucket_mean + lambdas * bucket_high
            
            predict_len=(probabilities * mu_eff).sum(dim=-1)
            # predict_len向上取整为int型整数
            
            for i in range(len(meta)):
                assert meta[i].predict_output_len==None
                meta[i].predict_output_len = int(predict_len[i].item())
            self._send_metadata_to_proxy(meta)
            self.predict_num += batch_size
            print(f'*'*20)
            print(f'[WT] meta:')
            print(f'{meta}')
            print(f"[WT] Total predictions made: {self.predict_num}")


# if __name__ == "__main__":
#     print("[WT] hello",flush=True)
#     print("Starting PredictorWorker main")
#     worker = PredictorWorker()
#     worker.run()