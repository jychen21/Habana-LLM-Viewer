# https://arxiv.org/pdf/2402.16363
from tabulate import tabulate
import matplotlib.pyplot as plt
from models import *


device_bw_tops = {
    "Gaudi2H": {"bf16": [2.24e12, 420e12, 22e12],
                "fp8": [2.24e12, 840e12, 22e12]},
    "Gaudi2C": {"bf16": [1.9e12, 280e12, 22e12],
                "fp8": [1.9e12, 560e12, 22e12]},
    "Gaudi2D": {"bf16": [1.9e12, 143e12, 22e12],
                "fp8": [1.9e12, 287e12, 22e12]},
}

type2bytes = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "fp8": 1,
}

# gigabytes
device_hbm_memory = {
    "Gaudi2H": 96,
    "Gaudi2C": 96,
    "Gaudi2D": 96,
}

model_dict = {
    "Llama2-7B": model_llama2_7b,
    "Llama2-13B": model_llama2_13b,
    # "Llama2-70B": model_llama2_70b,
    "Mixtral-8x7B": model_mixtral_8x7b,
    "GLaM-1.2T": model_glam_1dot2t,
    "MoE-1.8T": model_moe_1dot8t,
}


class Config:
    def __init__(self, batch_size, seq_len_q, seq_len_kv, hidden_size, num_heads_q, num_heads_kv, intermediate_size,
                 is_decoding, mlp_with_gate, num_experts, num_layers_mlp, num_layers_moe, dtype, device, pp=1, tp=1):
        self.batch_size = batch_size
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.hidden_size = hidden_size
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.intermediate_size = intermediate_size
        self.is_decoding = is_decoding
        self.mlp_with_gate = mlp_with_gate
        self.num_experts = num_experts
        self.num_layers_mlp = num_layers_mlp
        self.num_layers_moe = num_layers_moe
        self.num_layers = self.num_layers_mlp + self.num_layers_moe
        self.kvcache_bucket = False
        self.dtype = dtype
        self.num_bytes = type2bytes[self.dtype]
        self.device = device
        self.bw = device_bw_tops[self.device][self.dtype][0]
        self.tops = device_bw_tops[self.device][self.dtype][1]
        self.tops_tpc = device_bw_tops[self.device][self.dtype][2]
        self.hardware_ai = self.tops / self.bw
        self.hardware_ai_attn = self.tops / self.bw
        if self.is_decoding:
            self.hardware_ai_attn /= 128  # for Gaudi2
        self.hardware_ai_tpc = self.tops_tpc / self.bw
        self.mme_tops_factor = 128 if (
            self.batch_size * self.seq_len_q) <= 128 else 256  # for Gaudi2
        self.device_mem = device_hbm_memory[self.device]
        self.pp = pp  # currently just for memory fit analysis
        self.tp = tp  # currently just for memory fit analysis
        self.num_devices = self.pp * self.tp  # currently just for memory fit analysis
        self.kvcache_bucket = False
