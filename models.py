model_llama2_7b = {
    "model_name": "Llama2-7B", "hidden_size": 4096, "num_heads_q": 32,
    "num_heads_kv": 32, "intermediate_size": 11008, "mlp_with_gate": True,
    "num_layers_mlp": 32, "num_layers_moe": 0, "num_experts": 1,
}

model_llama2_13b = {
    "model_name": "Llama2-13B", "hidden_size": 5120, "num_heads_q": 40,
    "num_heads_kv": 40, "intermediate_size": 13824, "mlp_with_gate": True,
    "num_layers_mlp": 40, "num_layers_moe": 0, "num_experts": 1,
}

model_llama2_70b = {
    "model_name": "Llama2-70B", "hidden_size": 8192, "num_heads_q": 64,
    "num_heads_kv": 8, "intermediate_size": 28672, "mlp_with_gate": True,
    "num_layers_mlp": 80, "num_layers_moe": 0, "num_experts": 1,
}

model_mixtral_8x7b = {
    "model_name": "Mixtral-8x7B", "hidden_size": 4096, "num_heads_q": 32,
    "num_heads_kv": 8, "intermediate_size": 14336, "mlp_with_gate": True,
    "num_layers_mlp": 0, "num_layers_moe": 32, "num_experts": 8,
}

model_glam_1dot2t = {
    "model_name": "GLaM-MoE-1.2T", "hidden_size": 13568, "num_heads_q": 64,
    "num_heads_kv": 16, "intermediate_size": 22912, "mlp_with_gate": False,
    "num_layers_mlp": 32, "num_layers_moe": 32, "num_experts": 64,
}

model_moe_1dot8t = {
    "model_name": "MoE-1.8T", "hidden_size": 13568, "num_heads_q": 64,
    "num_heads_kv": 16, "intermediate_size": 22912, "mlp_with_gate": True,
    "num_layers_mlp": 0, "num_layers_moe": 120, "num_experts": 16,
}
