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
    
    def to_dict(self):
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
        }
