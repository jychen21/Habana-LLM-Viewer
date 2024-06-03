import math
from tabulate import tabulate
import matplotlib.pyplot as plt
import json
import models


ModelDict = {
    # Llama
    "Llama2-7B": models.model_llama2_7b,
    "Llama2-13B": models.model_llama2_13b,
    # "Llama2-70B": models.model_llama2_70b,
    "Llama3-8B": models.model_llama3_8b,

    # Qwen
    "Qwen-7B": models.model_qwen_7b,
    "Qwen-14B": models.model_qwen_14b,

    # MoE
    "Mixtral-8x7B": models.model_mixtral_8x7b,
    # "GLaM-1.2T": models.model_glam_1dot2t,
    # "MoE-1.8T": models.model_moe_1dot8t,

    # Falcon
    "Falcon-7B": models.model_falcon_7b,
    # "Falcon-40B": models.model_falcon_40b,

    # ChatGLM
    "ChatGLM2-6B": models.model_chatglm2_6b,
}


'''
# get from whitepaper, for reference only
https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html
https://habana.ai/wp-content/uploads/2023/10/HL-225B_Datasheet_10_23.pdf
https://habana.ai/habana-labs-data-documents-and-whitepapers/
'''
HardwareParameters = {
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


GigaBytes = 1024 * 1024 * 1024
MegaBytes = 1024 * 1024
GigaParam = 1024 * 1024 * 1024
MegaParam = 1024 * 1024
TFLOPS = 1e12
MicroSecs = 1e6
T_BW = 1e12


DeviceType2Ratio = {
    "B": [1.00, 3.38, 1.69, 0.14, 0.42, 0.68],
    "C": [0.65, 2.19, 1.10, 0.26, 0.67, 0.88],
    "D": [0.32, 1.08, 0.54, 0.67, 0.84, 0.94],
}


class HardwareConfig:
    def __init__(self, device, type, dtype, pp=1, tp=1):
        self.device = device
        self.type = type
        self.device_ratio = DeviceType2Ratio[self.type]
        self.dtype = dtype
        self.num_bytes = DType2Bytes[self.dtype]
        self.hbm_capacity = HardwareParameters[self.device]["HBM"]["Capacity"]
        self.hbm_bandwidth = HardwareParameters[self.device]["HBM"]["Bandwidth"]
        self.flops_mme = HardwareParameters[self.device]["Flops"][self.dtype]["MME"] * \
            self.device_ratio[0]
        self.flops_mme_factor = self.device_ratio[1:3]
        self.flops_mme_factor.append(self.flops_mme_factor[0])
        self.num_rounds = 1.0
        self.magic_number = 2**7
        self.pipeline = self.device_ratio[-3]
        self.flops_vec = HardwareParameters[self.device]["Flops"][self.dtype]["Vec"]
        self.hardware_ai_mme = self.flops_mme / self.hbm_bandwidth
        self.hardware_ai_vec = self.flops_vec / self.hbm_bandwidth
        self.hardware_ai_mme_attn = self.hardware_ai_mme
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
                 num_layers_moe, seq_len_q, seq_len_kv, batch_size, kvcache_bucket=None, *args, **kwargs):
        self.hardware_config = HardwareConfig(device, type, dtype, pp, tp)
        self.model_config = ModelConfig(hidden_size, num_heads_q, num_heads_kv,
                                        intermediate_size, mlp_with_gate, num_experts,
                                        num_layers_mlp, num_layers_moe)
        self.input_config = InputConfig(seq_len_q, seq_len_kv, batch_size)
        bt = self.input_config.batch_size * self.input_config.seq_len_q
        magic = self.hardware_config.magic_number
        if bt > magic:
            magic *= 2.0
            self.hardware_config.flops_mme_factor[-1] = self.hardware_config.flops_mme_factor[1]
            if bt <= magic:
                self.hardware_config.pipeline = self.hardware_config.device_ratio[-2]
            elif bt > magic:
                self.hardware_config.pipeline = self.hardware_config.device_ratio[-1]
            if bt % magic != 0:
                self.hardware_config.num_rounds = math.ceil(bt / magic)

        self.kvcache_bucket = kvcache_bucket
        self.enable_vec_bmm = kwargs.get('enable_vec_bmm', False)
