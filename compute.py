# https://www.intel.com/content/www/us/en/content-details/817486/intel-gaudi-3-ai-accelerator-white-paper.html
# https://habana.ai/wp-content/uploads/2023/10/HL-225B_Datasheet_10_23.pdf
# https://habana.ai/habana-labs-data-documents-and-whitepapers/
# https://arxiv.org/pdf/2402.16363


import math
from config import *


def proj_matmul(config: HardwareConfig, m, n, k):
    num_bytes = config.num_bytes
    bw = config.hbm_bandwidth
    flops_mme = config.flops_mme
    flops_mme_factor = config.flops_mme_factor[-1]
    num_rounds = config.num_rounds
    magic = magic_number = config.magic_number
    pipeline = config.pipeline
    magic *= config.device_ratio[3]
    if m > magic_number:
        magic *= config.device_ratio[4]
        flops_mme_factor = config.flops_mme_factor[1]
        if m <= magic:
            pipeline = config.device_ratio[-2]
        elif m > magic:
            pipeline = config.device_ratio[-1]
        if m % magic != 0:
            num_rounds = math.ceil(m / magic)

    params_in_input_a = m * k
    params_in_input_b = n * k
    params_out = m * n
    params_total = params_in_input_a + params_in_input_b + params_out
    bytes_total = params_total * num_bytes
    runtime_memory = bytes_total / bw

    num_ops = m * n * k * 2
    tops = min(flops_mme, flops_mme_factor * m * 1e12)
    runtime_compute = num_ops / tops
    runtime_compute *= num_rounds

    math_ai = num_ops / bytes_total
    runtime_roofline = runtime_memory + runtime_compute * pipeline

    proj_rst = {
        "name": "matmul",
        "m": m,
        "n": n,
        "k": k,
        "operations": num_ops,
        "size": bytes_total,
        "math_ai": math_ai,
        "tops_roofline": tops,
        "mem_time": runtime_memory,
        "cmp_time": runtime_compute,
        "latency": runtime_roofline,
    }

    return proj_rst


def proj_qkvo_proj(config):
    hidden_size = config.model_config.hidden_size
    num_bytes = config.hardware_config.num_bytes
    batch_size = config.input_config.batch_size
    seq_len_q = config.input_config.seq_len_q
    bw = config.hardware_config.hbm_bandwidth
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor[-1]
    num_rounds = config.hardware_config.num_rounds
    pipeline = config.hardware_config.pipeline
    hw_ai_mme = config.hardware_config.hardware_ai_mme

    params_in_input = batch_size * seq_len_q * hidden_size
    params_in_weight = hidden_size * hidden_size
    params_out = batch_size * seq_len_q * hidden_size
    params_total = params_in_input + params_in_weight + params_out
    params_total *= 4
    bytes_total = params_total * num_bytes
    runtime_memory = bytes_total / bw

    num_ops = batch_size * seq_len_q * hidden_size * hidden_size * 2 * 4
    tops = min(flops_mme, flops_mme_factor * batch_size * seq_len_q * 1e12)
    runtime_compute = num_ops / tops
    runtime_compute *= num_rounds

    math_ai = num_ops / bytes_total
    runtime_roofline = runtime_memory + runtime_compute * pipeline
    bound = "memory" if runtime_memory > runtime_compute else "compute"

    proj_rst = {
        "name": "qkvo_proj",
        "operations": num_ops,
        "size": bytes_total,
        "hw_ai": hw_ai_mme,
        "math_ai": math_ai,
        "tops_roofline": tops,
        "bw": bw,
        "mem_time": runtime_memory,
        "cmp_time": runtime_compute,
        "latency": runtime_roofline,
        "bound": bound
    }

    return proj_rst


def proj_attn_qk(config):
    hidden_size = config.model_config.hidden_size
    num_heads_q = config.model_config.num_heads_q
    head_dim = hidden_size // num_heads_q
    num_heads_kv = config.model_config.num_heads_kv
    num_groups = num_heads_q // num_heads_kv
    num_bytes = config.hardware_config.num_bytes
    batch_size = config.input_config.batch_size
    seq_len_q = config.input_config.seq_len_q
    seq_len_kv = config.input_config.seq_len_kv
    bw = config.hardware_config.hbm_bandwidth
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor[0]
    hw_ai_mme = config.hardware_config.hardware_ai_mme

    params_in_q = batch_size * num_heads_q * seq_len_q * head_dim
    params_in_k = batch_size * num_heads_kv * seq_len_kv * head_dim
    params_out = batch_size * num_heads_q * seq_len_q * seq_len_kv
    params_total = params_in_q + params_in_k + params_out
    bytes_total = params_total * num_bytes
    runtime_memory = bytes_total / bw

    num_ops = batch_size * num_heads_q * seq_len_q * head_dim * seq_len_kv * 2
    tops = min(flops_mme, flops_mme_factor * seq_len_q * 1e12)
    if seq_len_q == 1:
        if config.enable_vec_bmm:
            tops = config.hardware_config.flops_vec
        else:
            tops *= num_groups
    runtime_compute = num_ops / tops

    math_ai = num_ops / bytes_total
    runtime_roofline = runtime_memory if runtime_memory > runtime_compute else runtime_compute
    bound = "memory" if runtime_memory > runtime_compute else "compute"

    proj_rst = {
        "name": "q@k_T",
        "operations": num_ops,
        "size": bytes_total,
        "hw_ai": hw_ai_mme,
        "math_ai": math_ai,
        "tops_roofline": tops,
        "bw": bw,
        "mem_time": runtime_memory,
        "cmp_time": runtime_compute,
        "latency": runtime_roofline,
        "bound": bound
    }

    return proj_rst


def proj_attn_softmax(config):
    num_heads_q = config.model_config.num_heads_q
    num_bytes = config.hardware_config.num_bytes
    batch_size = config.input_config.batch_size
    seq_len_q = config.input_config.seq_len_q
    seq_len_kv = config.input_config.seq_len_kv
    bw = config.hardware_config.hbm_bandwidth
    flops_vec = config.hardware_config.flops_vec
    hw_ai_vec = config.hardware_config.hardware_ai_vec

    params_in = batch_size * num_heads_q * seq_len_q * seq_len_kv
    params_out = batch_size * num_heads_q * seq_len_q * seq_len_kv

    params_total = params_in + params_out
    bytes_total = params_total * num_bytes
    runtime_memory = bytes_total / bw

    # max x-max exp(x-max) sum(exp(x-max)) x/sum(exp(x-max))
    num_ops = batch_size * num_heads_q * seq_len_q * \
        seq_len_kv * 5
    runtime_compute = num_ops / flops_vec

    runtime_roofline = runtime_memory if runtime_memory > runtime_compute else runtime_compute
    bound = "memory" if runtime_memory > runtime_compute else "compute"

    math_ai = num_ops / bytes_total

    proj_rst = {
        "name": "softmax",
        "operations": num_ops,
        "size": bytes_total,
        "hw_ai": hw_ai_vec,
        "math_ai": math_ai,
        "tops_roofline": min(flops_vec, math_ai * bw),
        "bw": bw,
        "mem_time": runtime_memory,
        "cmp_time": runtime_compute,
        "latency": runtime_roofline,
        "bound": bound
    }

    return proj_rst


def proj_attn_scorev(config):
    hidden_size = config.model_config.hidden_size
    num_heads_q = config.model_config.num_heads_q
    head_dim = hidden_size // num_heads_q
    num_heads_kv = config.model_config.num_heads_kv
    num_groups = num_heads_q // num_heads_kv
    num_bytes = config.hardware_config.num_bytes
    batch_size = config.input_config.batch_size
    seq_len_q = config.input_config.seq_len_q
    seq_len_kv = config.input_config.seq_len_kv
    bw = config.hardware_config.hbm_bandwidth
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor[1]
    hw_ai_mme = config.hardware_config.hardware_ai_mme

    params_in_score = batch_size * num_heads_q * seq_len_q * seq_len_kv
    params_in_v = batch_size * num_heads_kv * seq_len_kv * head_dim
    params_out = batch_size * num_heads_q * seq_len_q * head_dim
    params_total = params_in_score + params_in_v + params_out
    bytes_total = params_total * num_bytes
    runtime_memory = bytes_total / bw

    num_ops = batch_size * num_heads_q * seq_len_q * seq_len_kv * head_dim * 2
    tops = min(flops_mme, flops_mme_factor * seq_len_q * 1e12)
    if seq_len_q == 1:
        if config.enable_vec_bmm:
            tops = config.hardware_config.flops_vec
        else:
            tops *= num_groups

    runtime_compute = num_ops / tops

    math_ai = num_ops / bytes_total
    runtime_roofline = runtime_memory if runtime_memory > runtime_compute else runtime_compute
    bound = "memory" if runtime_memory > runtime_compute else "compute"

    proj_rst = {
        "name": "score@v",
        "operations": num_ops,
        "size": bytes_total,
        "hw_ai": hw_ai_mme,
        "math_ai": math_ai,
        "tops_roofline": tops,
        "bw": bw,
        "mem_time": runtime_memory,
        "cmp_time": runtime_compute,
        "latency": runtime_roofline,
        "bound": bound
    }

    return proj_rst


def proj_paged_attn_v1(config: HardwareConfig, num_blocks, block_size,
                       num_heads_q, num_heads_kv, head_dim, bs, seqlen_kv):
    num_bytes = config.num_bytes
    bw = config.hbm_bandwidth
    flops_mme = config.flops_mme
    flops_mme_factor = config.flops_mme_factor[-1]
    num_rounds = config.num_rounds
    pipeline = config.pipeline

    proj_rst = {}

    # proj_rst = {
    #     "name": "PA_V1",
    #     "operations": num_ops,
    #     "size": bytes_total,
    #     "math_ai": math_ai,
    #     "tops_roofline": tops,
    #     "mem_time": runtime_memory,
    #     "cmp_time": runtime_compute,
    #     "latency": runtime_roofline,
    # }

    return proj_rst


def proj_flash_attn_v1(config: HardwareConfig, block_size, num_heads_q,
                       num_heads_kv, head_dim, bs, seqlen_kv):
    num_bytes = config.num_bytes
    bw = config.hbm_bandwidth
    flops_mme = config.flops_mme
    flops_mme_factor = config.flops_mme_factor[-1]
    num_rounds = config.num_rounds
    pipeline = config.pipeline

    proj_rst = {}

    # proj_rst = {
    #     "name": "FA_V1",
    #     "operations": num_ops,
    #     "size": bytes_total,
    #     "math_ai": math_ai,
    #     "tops_roofline": tops,
    #     "mem_time": runtime_memory,
    #     "cmp_time": runtime_compute,
    #     "latency": runtime_roofline,
    # }

    return proj_rst


def proj_mlp_up_or_w1(config):
    hidden_size = config.model_config.hidden_size
    intermediate_size = config.model_config.intermediate_size
    num_bytes = config.hardware_config.num_bytes
    batch_size = config.input_config.batch_size
    seq_len_q = config.input_config.seq_len_q
    bw = config.hardware_config.hbm_bandwidth
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor[-1]
    num_rounds = config.hardware_config.num_rounds
    pipeline = config.hardware_config.pipeline
    hw_ai_mme = config.hardware_config.hardware_ai_mme

    params_in_input = batch_size * seq_len_q * hidden_size
    params_in_weight = hidden_size * intermediate_size
    params_out = batch_size * seq_len_q * intermediate_size
    params_total = params_in_input + params_in_weight + params_out
    bytes_total = params_total * num_bytes
    runtime_memory = bytes_total / bw

    num_ops = batch_size * seq_len_q * hidden_size * intermediate_size * 2
    tops = min(flops_mme, flops_mme_factor * batch_size * seq_len_q * 1e12)

    runtime_compute = num_ops / tops
    runtime_compute *= num_rounds

    math_ai = num_ops / bytes_total
    runtime_roofline = runtime_memory + runtime_compute * pipeline
    bound = "memory" if runtime_memory > runtime_compute else "compute"

    proj_rst = {
        "name": "up(w1)",
        "operations": num_ops,
        "size": bytes_total,
        "hw_ai": hw_ai_mme,
        "math_ai": math_ai,
        "tops_roofline": tops,
        "bw": bw,
        "mem_time": runtime_memory,
        "cmp_time": runtime_compute,
        "latency": runtime_roofline,
        "bound": bound
    }

    return proj_rst


def proj_mlp_down_or_w2(config):
    hidden_size = config.model_config.hidden_size
    intermediate_size = config.model_config.intermediate_size
    num_bytes = config.hardware_config.num_bytes
    batch_size = config.input_config.batch_size
    seq_len_q = config.input_config.seq_len_q
    bw = config.hardware_config.hbm_bandwidth
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor[-1]
    num_rounds = config.hardware_config.num_rounds
    pipeline = config.hardware_config.pipeline
    hw_ai_mme = config.hardware_config.hardware_ai_mme

    params_in_input = batch_size * seq_len_q * intermediate_size
    params_in_weight = intermediate_size * hidden_size
    params_out = batch_size * seq_len_q * hidden_size
    params_total = params_in_input + params_in_weight + params_out
    bytes_total = params_total * num_bytes
    runtime_memory = bytes_total / bw

    num_ops = batch_size * seq_len_q * intermediate_size * hidden_size * 2
    tops = min(flops_mme, flops_mme_factor * batch_size * seq_len_q * 1e12)
    runtime_compute = num_ops / tops
    runtime_compute *= num_rounds

    math_ai = num_ops / bytes_total
    runtime_roofline = runtime_memory + runtime_compute * pipeline
    bound = "memory" if runtime_memory > runtime_compute else "compute"

    proj_rst = {
        "name": "down(w2)",
        "operations": num_ops,
        "size": bytes_total,
        "hw_ai": hw_ai_mme,
        "math_ai": math_ai,
        "tops_roofline": tops,
        "bw": bw,
        "mem_time": runtime_memory,
        "cmp_time": runtime_compute,
        "latency": runtime_roofline,
        "bound": bound
    }

    return proj_rst


def proj_mlp_gate_or_w3(config):
    hidden_size = config.model_config.hidden_size
    intermediate_size = config.model_config.intermediate_size
    num_bytes = config.hardware_config.num_bytes
    batch_size = config.input_config.batch_size
    seq_len_q = config.input_config.seq_len_q
    bw = config.hardware_config.hbm_bandwidth
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor[-1]
    num_rounds = config.hardware_config.num_rounds
    pipeline = config.hardware_config.pipeline
    hw_ai_mme = config.hardware_config.hardware_ai_mme

    params_in_input = batch_size * seq_len_q * hidden_size
    params_in_weight = hidden_size * intermediate_size
    params_out = batch_size * seq_len_q * intermediate_size
    params_total = params_in_input + params_in_weight + params_out
    bytes_total = params_total * num_bytes
    runtime_memory = bytes_total / bw

    num_ops = batch_size * seq_len_q * hidden_size * intermediate_size * 2
    tops = min(flops_mme, flops_mme_factor * batch_size * seq_len_q * 1e12)
    runtime_compute = num_ops / tops
    runtime_compute *= num_rounds

    math_ai = num_ops / bytes_total
    runtime_roofline = runtime_memory + runtime_compute * pipeline
    bound = "memory" if runtime_memory > runtime_compute else "compute"

    proj_rst = {
        "name": "gate(w3)",
        "operations": num_ops,
        "size": bytes_total,
        "hw_ai": hw_ai_mme,
        "math_ai": math_ai,
        "tops_roofline": tops,
        "bw": bw,
        "mem_time": runtime_memory,
        "cmp_time": runtime_compute,
        "latency": runtime_roofline,
        "bound": bound
    }

    return proj_rst


def proj_attn(config):
    qk = proj_attn_qk(config)
    softmax = proj_attn_softmax(config)
    sv = proj_attn_scorev(config)
    runtime_attn = (qk["latency"] + softmax["latency"] + sv["latency"])

    return runtime_attn, (qk, softmax, sv)


def proj_mlp(config):
    mlp_with_gate = config.model_config.mlp_with_gate

    up = proj_mlp_up_or_w1(config)
    if mlp_with_gate:
        gate = proj_mlp_gate_or_w3(config)
    down = proj_mlp_down_or_w2(config)

    runtime_mlp = up["latency"] + down["latency"]
    if mlp_with_gate:
        runtime_mlp += gate["latency"]

    return runtime_mlp, (up, down, gate if mlp_with_gate else None)


def proj_moe(config):
    num_experts = config.model_config.num_experts

    runtime_mlp, mlp_items = proj_mlp(config)
    runtime_moe = runtime_mlp * num_experts

    return runtime_moe, mlp_items


def proj_single_layer(config):
    qkvo_proj = proj_qkvo_proj(config)
    runtime_attn, attn_items = proj_attn(config)
    runtime_moe, ffn_items = proj_moe(config)
    runtime_single_layer = qkvo_proj["latency"] + runtime_attn + runtime_moe

    hidden_size = config.model_config.hidden_size
    inter_size = config.model_config.intermediate_size
    num_heads_q = config.model_config.num_heads_q
    num_heads_kv = config.model_config.num_heads_kv
    head_dim = hidden_size // num_heads_q
    seq_len_q = config.input_config.seq_len_q
    seq_len_kv = config.input_config.seq_len_kv

    num_ops = qkvo_proj["operations"]
    num_ops += attn_items[0]["operations"]
    num_ops += attn_items[-1]["operations"]
    for item in ffn_items:
        if item is not None:
            num_ops += item["operations"]

    num_bytes = qkvo_proj["size"]
    num_bytes += attn_items[0]["size"]
    num_bytes += attn_items[-1]["size"]
    for item in ffn_items:
        if item is not None:
            num_bytes += item["size"]

    math_ai = num_ops / num_bytes

    single_layer_items = {
        "hidden_size": hidden_size,
        "inter_size": inter_size,
        "headsq": num_heads_q,
        "headskv": num_heads_kv,
        "tq": seq_len_q,
        "tkv": seq_len_kv,
        "headdim": head_dim,
        "qkvo": qkvo_proj,
        "attn": attn_items,
        "ffn": ffn_items,
        "operations": num_ops,
        "size": num_bytes,
        "math_ai": math_ai
    }

    return runtime_single_layer, single_layer_items


def proj_decoder(config):
    num_layers = config.model_config.num_layers

    runtime_single_layer, single_layer_items = proj_single_layer(config)
    runtime_decoder = runtime_single_layer * num_layers

    return runtime_decoder, single_layer_items


def do_model_projection(model_name, device, type, pp, tp, dtype, input, output, bs, kvcache_bucket=None, **kwargs):
    model = ModelDict[model_name]
    hidden_size = model["hidden_size"]
    num_heads_q = model["num_heads_q"]
    num_heads_kv = model["num_heads_kv"]
    intermediate_size = model["intermediate_size"]
    mlp_with_gate = model["mlp_with_gate"]
    num_layers_mlp = model["num_layers_mlp"]
    num_layers_moe = model["num_layers_moe"]
    num_experts = model["num_experts"]

    proj_rst = {}
    proj_decoding_steps = []
    proj_decoding_latency = []
    proj_decoding_ai = []

    for step in range(output):
        if step == 0:
            seq_len_q = seq_len_kv = input
            cfg = Config(device, type, dtype, pp, tp, hidden_size, num_heads_q, num_heads_kv,
                         intermediate_size, mlp_with_gate, num_experts, num_layers_mlp,
                         num_layers_moe, seq_len_q, seq_len_kv, bs, None, **kwargs)
            proj_rst["prefill"] = proj_decoder(cfg)
        else:
            seq_len_q = 1
            seq_len_kv = input + output
            if kvcache_bucket is not None:
                seq_len_kv = input + \
                    math.ceil(step / kvcache_bucket) * kvcache_bucket
            cfg = Config(device, type, dtype, pp, tp, hidden_size, num_heads_q, num_heads_kv,
                         intermediate_size, mlp_with_gate, num_experts, num_layers_mlp,
                         num_layers_moe, seq_len_q, seq_len_kv, bs, kvcache_bucket, **kwargs)
            proj_decoding_steps.append(proj_decoder(cfg))
            proj_decoding_latency.append(proj_decoding_steps[-1][0])
            proj_decoding_ai.append(proj_decoding_steps[-1][1]["math_ai"])

    proj_rst["decode"] = proj_decoding_steps

    overall_ai = round(((proj_rst["prefill"][1]["math_ai"] + sum(proj_decoding_ai)) /
                        (len(proj_decoding_ai) + 1)), 2)

    proj_rst["arithmetic_intensity"] = overall_ai

    return proj_rst


def do_op_projection(op_name, device, type, dtype, **kwargs):
    cfg = HardwareConfig(device, type, dtype)

    if op_name == "Matmul":
        m = kwargs.get('m', 1)
        n = kwargs.get('n', 4096)
        k = kwargs.get('k', 4096)

        proj_rst = proj_matmul(cfg, m, n, k)
    elif op_name == "FlashAttentionV1":
        block_size = kwargs.get("block_size", 1024)
        num_heads_q = kwargs.get("heads_q", 32)
        num_heads_kv = kwargs.get("heads_kv", 32)
        head_dim = kwargs.get("head_dim", 128)
        bs = kwargs.get("bs", 1)
        seqlen_kv = kwargs.get("seqlen_kv", 4096)

        proj_rst = proj_flash_attn_v1(cfg,
                                      block_size,
                                      num_heads_q,
                                      num_heads_kv,
                                      head_dim,
                                      bs,
                                      seqlen_kv)
    elif op_name == "PagedAttentionV1":
        num_blocks = kwargs.get("num_blocks", 1024)
        block_size = kwargs.get("block_size", 128)
        num_heads_q = kwargs.get("heads_q", 32)
        num_heads_kv = kwargs.get("heads_kv", 32)
        head_dim = kwargs.get("head_dim", 128)
        bs = kwargs.get("bs", 1)
        seqlen_kv = kwargs.get("seqlen_kv", 4096)

        proj_rst = proj_paged_attn_v1(cfg,
                                      num_blocks,
                                      block_size,
                                      num_heads_q,
                                      num_heads_kv,
                                      head_dim,
                                      bs,
                                      seqlen_kv)
    else:
        assert 0, "operation projection currently just support Matmul!"

    return proj_rst
