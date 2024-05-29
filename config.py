# https://arxiv.org/pdf/2402.16363
from tabulate import tabulate
import matplotlib.pyplot as plt
import json
import models


'''
references:
https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html
https://habana.ai/wp-content/uploads/2023/10/HL-225B_Datasheet_10_23.pdf
https://habana.ai/habana-labs-data-documents-and-whitepapers/
'''
HardwareParameters = {  # get from whitepaper, for reference only
    "IntelGaudi2": {
        "HBM": {"Capacity": 96e9, "Bandwidth": 2.46e12},
        "Flops": {
            "BF16": {"MME": 432e12, "Vec": 11e12},
            "FP8": {"MME": 865e12, "Vec": 11e12}
        }
    },
    "IntelGaudi3": {
        "HBM": {"Capacity": 128e9, "Bandwidth": 3.7e12},
        "Flops": {
            "BF16": {"MME": 1835e12, "Vec": 28.7e12},
            "FP8": {"MME": 1835e12, "Vec": 28.7e12}
        }
    },
}


DType2Bytes = {
    "FP32": 4,
    "FP16": 2,
    "BF16": 2,
    "FP8": 1,
}


DeviceType2Ratio = {
    "B": [1.00, 0.77, 0.14, 0.42, 0.68],
    "C": [0.65, 0.61, 0.26, 0.67, 0.88],
    "D": [0.32, 0.30, 0.67, 0.84, 0.94],
}


ModelDict = {
    "Llama2-7B": models.model_llama2_7b,
    "Llama2-13B": models.model_llama2_13b,
    # "Llama2-70B": models.model_llama2_70b,
    "Qwen-7B": models.model_qwen_7b,
    "Mixtral-8x7B": models.model_mixtral_8x7b,
    "GLaM-1.2T": models.model_glam_1dot2t,
    "MoE-1.8T": models.model_moe_1dot8t,
}


class HardwareConfig:
    def __init__(self, device, type, dtype, pp=1, tp=1):
        self.device = device
        self.type = type
        self.device_ratio = DeviceType2Ratio[self.type]
        self.dtype = dtype
        self.num_bytes = DType2Bytes[self.dtype]
        self.hbm_capacity = HardwareParameters[self.device]["HBM"]["Capacity"]
        self.hbm_bandwidth_org = HardwareParameters[self.device]["HBM"]["Bandwidth"]
        self.hbm_bandwidth = self.hbm_bandwidth_org * self.device_ratio[1]
        self.pipeline = self.device_ratio[2]
        self.flops_mme = HardwareParameters[self.device]["Flops"][self.dtype]["MME"] * \
            self.device_ratio[0]
        self.flops_vec = HardwareParameters[self.device]["Flops"][self.dtype]["Vec"]
        self.hardware_ai_mme = self.flops_mme / self.hbm_bandwidth
        self.hardware_ai_vec = self.flops_vec / self.hbm_bandwidth
        self.hardware_ai_mme_attn = self.hardware_ai_mme
        self.flops_mme_factor = 128
        self.flops_mme_factor_attn = self.flops_mme_factor
        self.pp = pp
        self.tp = tp
        self.num_devices = self.pp * self.tp


class ModelConfig:
    def __init__(self, hidden_size, num_heads_q, num_heads_kv,
                 intermediate_size, mlp_with_gate, num_experts,
                 num_layers_mlp, num_layers_moe):
        self.hidden_size = hidden_size
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.intermediate_size = intermediate_size
        self.mlp_with_gate = mlp_with_gate
        self.num_experts = num_experts
        self.num_layers_mlp = num_layers_mlp
        self.num_layers_moe = num_layers_moe
        self.num_layers = self.num_layers_mlp + self.num_layers_moe


class InputConfig:
    def __init__(self, seq_len_q, seq_len_kv, batch_size):
        self.seq_len_q = seq_len_q
        self.seq_len_kv = seq_len_kv
        self.batch_size = batch_size


class Config:
    def __init__(self, device, type, dtype, pp, tp, hidden_size, num_heads_q, num_heads_kv,
                 intermediate_size, mlp_with_gate, num_experts, num_layers_mlp,
                 num_layers_moe, seq_len_q, seq_len_kv, batch_size, is_decoding,
                 kvcache_bucket=False):
        self.hardware_config = HardwareConfig(device, type, dtype, pp, tp)
        self.model_config = ModelConfig(hidden_size, num_heads_q, num_heads_kv,
                                        intermediate_size, mlp_with_gate, num_experts,
                                        num_layers_mlp, num_layers_moe)
        self.input_config = InputConfig(seq_len_q, seq_len_kv, batch_size)

        bs = self.input_config.batch_size
        tq = self.input_config.seq_len_q
        bs_by_tq = bs * tq
        self.hardware_config.flops_mme_factor = 256 if bs_by_tq > 128 else 128
        if bs_by_tq > 128 and bs_by_tq <= 256:
            self.hardware_config.pipeline = self.hardware_config.device_ratio[3]
        elif bs_by_tq > 256:
            self.hardware_config.pipeline = self.hardware_config.device_ratio[4]

        self.is_decoding = is_decoding
        if self.is_decoding:
            self.hardware_config.hardware_ai_mme_attn = self.input_config.seq_len_q / 128

        self.bucket_perf_gain = 1
        self.kvcache_bucket = kvcache_bucket
        if self.kvcache_bucket:
            self.bucket_perf_gain = 1.3
