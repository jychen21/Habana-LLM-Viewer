# https://arxiv.org/pdf/2402.16363
from tabulate import tabulate
import matplotlib.pyplot as plt


device_bw_tops = {
    "Gaudi2H": {"bf16": [2.24e12, 420e12, 22e12],
                "fp8": [2.24e12, 840e12, 22e12]},
    "Gaudi2C": {"bf16": [2.24e12, 287e12, 22e12],
                "fp8": [2.24e12, 574e12, 22e12]},
    "Gaudi2D": {"bf16": [2.24e12, 143e12, 22e12],
                "fp8": [2.24e12, 287e12, 22e12]},
}

type2bytes = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
}

device_hbm_memory = {
    "Gaudi2H": 96,
    "Gaudi2C": 96,
    "Gaudi2D": 96,
}


class Config:
    def __init__(self, batch_size, seq_len_q, seq_len_kv, hidden_size, num_heads_q, num_heads_kv, intermediate_size,
                 is_decoding, num_bytes, bw, tops, tops_tpc, with_gate, num_experts, num_layers, num_devices=1):
        self.batch_size = batch_size
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.hidden_size = hidden_size
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.intermediate_size = intermediate_size
        self.is_decoding = is_decoding
        self.num_bytes = num_bytes
        self.bw = bw
        self.tops = tops
        self.tops_tpc = tops_tpc
        self.with_gate = with_gate
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.kvcache_bucket = False
        self.hardware_ai = tops / bw
        self.hardware_ai_attn = tops / bw
        if self.is_decoding:
            self.hardware_ai_attn /= 128  # 128 for Gaudi2
        self.num_devices = num_devices # currently just for memory fit analysis
