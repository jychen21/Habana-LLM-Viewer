from tabulate import tabulate


class Config:
    def __init__(self, batch_size, seq_len, hidden_size, num_heads_q, num_heads_kv,
                 intermediate_size, is_decoding, num_bytes, bw, tops, with_gate, num_experts, num_layers):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_heads_q = num_heads_q
        self.num_heads_kv = num_heads_kv
        self.intermediate_size = intermediate_size
        self.is_decoding = is_decoding
        self.num_bytes = num_bytes
        self.bw = bw
        self.tops = tops
        self.tops_tpc = 11e12
        self.with_gate = with_gate
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.hardware_ai = tops / bw


def proj_qkvo_proj(model_config):
    # memory (in & out)
    params_in_input = model_config.batch_size * \
        model_config.seq_len * model_config.hidden_size
    params_in_weight = model_config.hidden_size * model_config.hidden_size
    params_out = model_config.batch_size * \
        model_config.seq_len * model_config.hidden_size
    params_total = params_in_input + params_in_weight + params_out
    bytes_total = params_total * model_config.num_bytes * 4  # 4 for qkvo
    runtime_memory = bytes_total / model_config.bw

    # compute (2 for mul & add)
    num_ops = model_config.batch_size * model_config.seq_len * \
        model_config.hidden_size * model_config.intermediate_size * 2 * 4  # 4 for qkvo
    if model_config.is_decoding:
        model_config.tops = model_config.tops / 128 * model_config.batch_size
    runtime_compute = num_ops / model_config.tops

    # arithmetic intensity (#flops / #bytes)
    math_ai = num_ops / bytes_total

    return runtime_memory if runtime_memory > runtime_compute else runtime_compute
    # return bytes_total, runtime_memory if runtime_memory > runtime_compute else runtime_compute


def proj_attn_qk(model_config):
    head_dim = model_config.hidden_size // model_config.num_heads_q

    # memory (in & out)
    params_in_q = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len * head_dim
    params_in_k = model_config.batch_size * model_config.num_heads_kv * \
        model_config.seq_len * head_dim
    params_out = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len * model_config.seq_len
    params_total = params_in_q + params_in_k + params_out
    bytes_total = params_total * model_config.num_bytes
    runtime_memory = bytes_total / model_config.bw

    # compute (2 for mul & add)
    num_ops = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len * head_dim * model_config.seq_len * 2
    if model_config.is_decoding:
        model_config.tops = model_config.tops / 128 * model_config.batch_size
    runtime_compute = num_ops / model_config.tops

    # arithmetic intensity (#flops / #bytes)
    math_ai = num_ops / bytes_total

    return runtime_memory if runtime_memory > runtime_compute else runtime_compute
    # return bytes_total, runtime_memory if runtime_memory > runtime_compute else runtime_compute


def proj_attn_softmax(model_config):
    head_dim = model_config.hidden_size // model_config.num_heads_q

    # memory (in & out)
    params_in = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len * model_config.seq_len
    params_out = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len * model_config.seq_len

    params_total = params_in + params_out
    bytes_total = params_total * model_config.num_bytes
    runtime_memory = bytes_total / model_config.bw

    # compute (max, x-max, exp(x-max), sum(exp(x-max)), x/sum(exp(x-max)))
    num_ops = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len * head_dim * model_config.seq_len * 5  # 5 for traversal times
    runtime_compute = num_ops / model_config.tops_tpc

    # arithmetic intensity (#flops / #bytes)
    math_ai = num_ops / bytes_total

    return runtime_memory if runtime_memory > runtime_compute else runtime_compute
    # return bytes_total, runtime_memory if runtime_memory > runtime_compute else runtime_compute


def proj_attn_scorev(model_config):
    head_dim = model_config.hidden_size // model_config.num_heads_q

    # memory (in & out)
    params_in_score = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len * model_config.seq_len
    params_in_v = model_config.batch_size * \
        model_config.num_heads_kv * model_config.seq_len * head_dim
    params_out = model_config.batch_size * \
        model_config.num_heads_q * model_config.seq_len * head_dim
    params_total = params_in_score + params_in_v + params_out
    bytes_total = params_total * model_config.num_bytes
    runtime_memory = bytes_total / model_config.bw

    # compute (2 for mul & add)
    num_ops = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len * model_config.seq_len * head_dim * 2
    if model_config.is_decoding:
        model_config.tops = model_config.tops / 128 * model_config.batch_size
    runtime_compute = num_ops / model_config.tops

    # arithmetic intensity (#flops / #bytes)
    math_ai = num_ops / bytes_total

    return runtime_memory if runtime_memory > runtime_compute else runtime_compute
    # return bytes_total, runtime_memory if runtime_memory > runtime_compute else runtime_compute


def proj_attn(model_config):
    runtime_qk = proj_attn_qk(model_config)
    runtime_softmax = proj_attn_softmax(model_config)
    runtime_scorev = proj_attn_scorev(model_config)
    runtime_attn = runtime_qk + runtime_softmax + runtime_scorev

    return runtime_attn


def proj_mlp_gate_or_w3(model_config):
    # memory (in & out)
    params_in_input = model_config.batch_size * \
        model_config.seq_len * model_config.hidden_size
    params_in_weight = model_config.hidden_size * model_config.intermediate_size
    params_out = model_config.batch_size * \
        model_config.seq_len * model_config.intermediate_size
    params_total = params_in_input + params_in_weight + params_out
    bytes_total = params_total * model_config.num_bytes
    runtime_memory = bytes_total / model_config.bw

    # compute (2 for mul & add)
    num_ops = model_config.batch_size * model_config.seq_len * \
        model_config.hidden_size * model_config.intermediate_size * 2
    if model_config.is_decoding:
        model_config.tops = model_config.tops / 128 * model_config.batch_size
    runtime_compute = num_ops / model_config.tops

    # arithmetic intensity (#flops / #bytes)
    math_ai = num_ops / bytes_total

    return runtime_memory if runtime_memory > runtime_compute else runtime_compute
    # return bytes_total, runtime_memory if runtime_memory > runtime_compute else runtime_compute


def proj_mlp_up_or_w1(model_config):
    # memory (in & out)
    params_in_input = model_config.batch_size * \
        model_config.seq_len * model_config.hidden_size
    params_in_weight = model_config.hidden_size * model_config.intermediate_size
    params_out = model_config.batch_size * \
        model_config.seq_len * model_config.intermediate_size
    params_total = params_in_input + params_in_weight + params_out
    bytes_total = params_total * model_config.num_bytes
    runtime_memory = bytes_total / model_config.bw

    # compute (2 for mul & add)
    num_ops = model_config.batch_size * model_config.seq_len * \
        model_config.hidden_size * model_config.intermediate_size * 2
    if model_config.is_decoding:
        model_config.tops = model_config.tops / 128 * model_config.batch_size
    runtime_compute = num_ops / model_config.tops

    # arithmetic intensity (#flops / #bytes)
    math_ai = num_ops / bytes_total

    return runtime_memory if runtime_memory > runtime_compute else runtime_compute
    # return bytes_total, runtime_memory if runtime_memory > runtime_compute else runtime_compute


def proj_mlp_down_or_w2(model_config):
    # memory (in & out)
    params_in_input = model_config.batch_size * \
        model_config.seq_len * model_config.intermediate_size
    params_in_weight = model_config.intermediate_size * model_config.hidden_size
    params_out = model_config.batch_size * \
        model_config.seq_len * model_config.hidden_size
    params_total = params_in_input + params_in_weight + params_out
    bytes_total = params_total * model_config.num_bytes
    runtime_memory = bytes_total / model_config.bw

    # compute (2 for mul & add)
    num_ops = model_config.batch_size * model_config.seq_len * \
        model_config.hidden_size * model_config.intermediate_size * 2
    if model_config.is_decoding:
        model_config.tops = model_config.tops / 128 * model_config.batch_size
    runtime_compute = num_ops / model_config.tops

    # arithmetic intensity (#flops / #bytes)
    math_ai = num_ops / bytes_total

    return runtime_memory if runtime_memory > runtime_compute else runtime_compute
    # return bytes_total, runtime_memory if runtime_memory > runtime_compute else runtime_compute


def proj_mlp(model_config):
    runtime_mlp = 0
    runtime_mlp += proj_mlp_up_or_w1(model_config)
    runtime_mlp += proj_mlp_down_or_w2(model_config)
    if model_config.with_gate:
        runtime_mlp += proj_mlp_gate_or_w3(model_config)

    return runtime_mlp


def proj_moe(model_config):
    runtime_moe = proj_mlp(model_config) * model_config.num_experts

    return runtime_moe


def proj_single_layer(model_config):
    runtime_qkvo = proj_qkvo_proj(model_config)
    runtime_attn = proj_qkvo_proj(model_config)
    runtime_moe = proj_moe(model_config)
    runtime_single_layer = runtime_qkvo + runtime_attn + runtime_moe

    return runtime_single_layer


def proj_decoder(model_config):
    runtime_decoder = proj_single_layer(model_config) * model_config.num_layers

    return runtime_decoder


device_bw_tops = {
    "Gaudi2H_BF16": [2.24e12, 420e12],
    "Gaudi2H_FP8": [2.24e12, 840e12],
}

type2bytes = {
    "f32": 4,
    "bf16": 2,
    "fp8": 1,
}

item_list = ["HiddenSize", "NumHeadsQ", "NumHeadsKV", "InterSize", "IsDecoding",
             "NumExperts", "NumLayers", "SeqLength", "BatchSize", "Latency (s)", "Throughput (tokens/sec)"]

# prefill long sequence
print("projection prefill...")
data_prefill = [item_list]
for bs in [1, 2, 4, 8, 16, 32, 64]:
    model_config = Config(batch_size=bs,
                          seq_len=32000,
                          hidden_size=4096,
                          num_heads_q=32,
                          num_heads_kv=8,
                          intermediate_size=14336,
                          is_decoding=False,
                          num_bytes=type2bytes['fp8'],
                          bw=device_bw_tops["Gaudi2H_FP8"][0],
                          tops=device_bw_tops["Gaudi2H_FP8"][1],
                          with_gate=True,
                          num_experts=8,
                          num_layers=32)
    runtime_decoder = proj_decoder(model_config)
    data_prefill.append([model_config.hidden_size, model_config.num_heads_q, model_config.num_heads_kv, model_config.intermediate_size,
                        model_config.is_decoding, model_config.num_experts, model_config.num_layers, model_config.seq_len, bs, round(runtime_decoder, 2), round(1/runtime_decoder, 2)])
    # print(
    #     f"moe projection for prefill, bs: {bs}, 1st token latency: {runtime_decoder:.2f} s, 1st token throughput: {1/runtime_decoder} tokens/sec")
print(tabulate(data_prefill))
print("done!\n")

# decode
print("projection decoding...")
data_decoding = [item_list]
for bs in [1, 2, 4, 8, 16, 32, 64]:
    model_config = Config(batch_size=bs,
                          seq_len=128,
                          hidden_size=4096,
                          num_heads_q=32,
                          num_heads_kv=8,
                          intermediate_size=14336,
                          is_decoding=True,
                          num_bytes=type2bytes['fp8'],
                          bw=device_bw_tops["Gaudi2H_FP8"][0],
                          tops=device_bw_tops["Gaudi2H_FP8"][1],
                          with_gate=True,
                          num_experts=8,
                          num_layers=32)
    runtime_decoder = proj_decoder(model_config)
    data_decoding.append([model_config.hidden_size, model_config.num_heads_q, model_config.num_heads_kv, model_config.intermediate_size,
                          model_config.is_decoding, model_config.num_experts, model_config.num_layers, model_config.seq_len, bs, round(runtime_decoder, 2), round(1/runtime_decoder, 2)])
    # print(
    #     f"moe projection for decoding, bs: {bs}, 1st token latency: {runtime_decoder:.2f} s, 1st token throughput: {1/runtime_decoder} tokens/sec")
print(tabulate(data_decoding))
print("done!")
