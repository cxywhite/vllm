from dataclasses import dataclass
from typing import Optional


@dataclass
class Custom_Metadata:
    req_id: str
    token_range: tuple[int, int]  # (start_idx, end_idx)
    num_tokens: int #调度token数量
    temperature: float  
    top_p: float
    top_k: int
    repetition_penalty: float
    prompt_len: Optional[int] = None  # 输入prompt长度
    predict_output_len: Optional[int] = None  # To be filled after prediction
    predict_start_ts_ns: Optional[int] = None  # 第16层(hidden+attention_score)就绪时刻
    prefill_end_ts_ns: Optional[int] = None  # 第32层输出完成时刻
    predict_end_ts_ns: Optional[int] = None  # predictor前向结束时刻
    
    def to_dict(self):
        prefill_tail_ms = None
        predict_ms = None
        predict_hidden_by_prefill = None

        if (self.predict_start_ts_ns is not None and
                self.prefill_end_ts_ns is not None):
            prefill_tail_ms = (
                self.prefill_end_ts_ns - self.predict_start_ts_ns) / 1e6

        if (self.predict_start_ts_ns is not None and
                self.predict_end_ts_ns is not None):
            predict_ms = (
                self.predict_end_ts_ns - self.predict_start_ts_ns) / 1e6

        if (self.prefill_end_ts_ns is not None and
                self.predict_end_ts_ns is not None):
            predict_hidden_by_prefill = (
                self.predict_end_ts_ns <= self.prefill_end_ts_ns)

        return {
            "req_id": self.req_id,
            "token_range": self.token_range,
            "num_tokens": self.num_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "prompt_len": self.prompt_len,
            "predict_output_len": self.predict_output_len,
            "predict_start_ts_ns": self.predict_start_ts_ns,
            "prefill_end_ts_ns": self.prefill_end_ts_ns,
            "predict_end_ts_ns": self.predict_end_ts_ns,
            "prefill_tail_ms": prefill_tail_ms,
            "predict_ms": predict_ms,
            "predict_hidden_by_prefill": predict_hidden_by_prefill,
        }
