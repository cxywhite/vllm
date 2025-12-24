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
    predict_output_len: Optional[int] = None  # To be filled after prediction
