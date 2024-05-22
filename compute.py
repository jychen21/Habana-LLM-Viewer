# https://arxiv.org/pdf/2402.16363
from tabulate import tabulate
import matplotlib.pyplot as plt
from base import *


cmp_item_list = ["Device", "HiddenSize", "HeadsQ", "HeadsKV", "InterSize", "Decoding", "Experts",
                 "Layers", "In", "Out", "DType", "BS", "Latency(s)", "Throughput(tokens/sec)"]
layer_analysis_list = ["Input", "Output", "DataType", "BatchSize", "LayerName",
                       "NumOps(e9)", "Memory(GB)", "TopsRF(TFlops)", "AI", "Bound"]


def proj_qkvo_proj(model_config):
    # memory (in & out)
    params_in_input = model_config.batch_size * \
        model_config.seq_len_q * model_config.hidden_size
    params_in_weight = model_config.hidden_size * model_config.hidden_size
    params_out = model_config.batch_size * \
        model_config.seq_len_q * model_config.hidden_size
    params_total = params_in_input + params_in_weight + params_out
    params_total *= 4  # 4 for qkvo
    bytes_total = params_total * model_config.num_bytes
    runtime_memory = bytes_total / model_config.bw

    # compute (2 for mul & add)
    # [B, T_Q, H] @ [H, H]
    num_ops = model_config.batch_size * model_config.seq_len_q * \
        model_config.hidden_size * model_config.hidden_size * 2 * 4  # 4 for qkvo
    tops = min(model_config.tops, model_config.tops *
               (model_config.batch_size * model_config.seq_len_q / 128))  # 128 for Gaudi2

    # arithmetic intensity (#flops / #bytes)
    math_ai = num_ops / bytes_total
    tops = min(tops, math_ai * model_config.bw)
    runtime_compute = num_ops / tops  # model_config.tops

    proj_rst = {
        "name": "qkvo_proj",
        "#ops": num_ops,
        "#mem": bytes_total,
        "math_ai": math_ai,
        "tops_roofline": tops,
        "latency": runtime_memory if runtime_memory > runtime_compute else runtime_compute,
        "bound": "memory" if math_ai < model_config.hardware_ai else "compute"
    }

    return proj_rst


def proj_attn_qk(model_config):
    head_dim = model_config.hidden_size // model_config.num_heads_q

    # memory (in & out)
    params_in_q = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len_q * head_dim
    params_in_k = model_config.batch_size * model_config.num_heads_kv * \
        model_config.seq_len_kv * head_dim
    params_out = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len_q * model_config.seq_len_kv
    params_total = params_in_q + params_in_k + params_out
    bytes_total = params_total * model_config.num_bytes
    runtime_memory = bytes_total / model_config.bw

    # compute (2 for mul & add)
    # [B, M, T_Q, D] @ [B, M, D, T_KV]
    num_ops = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len_q * head_dim * model_config.seq_len_kv * 2
    tops = model_config.tops
    if model_config.is_decoding:
        # 128 for Gaudi2
        # 128 for Gaudi2
        tops = min(tops, tops * (model_config.batch_size / 128))

    # arithmetic intensity (#flops / #bytes)
    math_ai = num_ops / bytes_total
    tops = min(tops, math_ai * model_config.bw)
    runtime_compute = num_ops / tops

    proj_rst = {
        "name": "q@k_T",
        "#ops": num_ops,
        "#mem": bytes_total,
        "math_ai": math_ai,
        "tops_roofline": tops,
        "latency": runtime_memory if runtime_memory > runtime_compute else runtime_compute,
        "bound": "memory" if math_ai < model_config.hardware_ai_attn else "compute"
    }

    return proj_rst


def proj_attn_softmax(model_config):
    # memory (in & out)
    params_in = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len_q * model_config.seq_len_kv
    params_out = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len_q * model_config.seq_len_kv

    params_total = params_in + params_out
    bytes_total = params_total * model_config.num_bytes
    runtime_memory = bytes_total / model_config.bw

    # compute (max, x-max, exp(x-max), sum(exp(x-max)), x/sum(exp(x-max)))
    num_ops = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len_q * model_config.seq_len_kv * 5  # 5 for traversal times
    runtime_compute = num_ops / model_config.tops_tpc

    # arithmetic intensity (#flops / #bytes)
    math_ai = num_ops / bytes_total

    proj_rst = {
        "name": "softmax",
        "#ops": num_ops,
        "#mem": bytes_total,
        "math_ai": math_ai,
        "tops_roofline": min(model_config.tops_tpc, math_ai * model_config.bw),
        "latency": runtime_memory if runtime_memory > runtime_compute else runtime_compute,
        "bound": "memory" if math_ai < model_config.hardware_ai_attn else "compute"
    }

    return proj_rst


def proj_attn_scorev(model_config):
    head_dim = model_config.hidden_size // model_config.num_heads_q

    # memory (in & out)
    params_in_score = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len_q * model_config.seq_len_kv
    params_in_v = model_config.batch_size * \
        model_config.num_heads_kv * model_config.seq_len_kv * head_dim
    params_out = model_config.batch_size * \
        model_config.num_heads_q * model_config.seq_len_q * head_dim
    params_total = params_in_score + params_in_v + params_out
    bytes_total = params_total * model_config.num_bytes
    runtime_memory = bytes_total / model_config.bw

    # compute (2 for mul & add)
    # [B, M, T_Q, T_KV] @ [B, M, T_KV, D]
    num_ops = model_config.batch_size * model_config.num_heads_q * \
        model_config.seq_len_q * model_config.seq_len_kv * head_dim * 2
    tops = model_config.tops
    if model_config.is_decoding:
        # 128 for Gaudi2
        # 128 for Gaudi2
        tops = min(tops, tops * (model_config.batch_size / 128))

    # arithmetic intensity (#flops / #bytes)
    math_ai = num_ops / bytes_total
    tops = min(tops, math_ai * model_config.bw)
    runtime_compute = num_ops / tops

    proj_rst = {
        "name": "score@v",
        "#ops": num_ops,
        "#mem": bytes_total,
        "math_ai": math_ai,
        "tops_roofline": tops,
        "latency": runtime_memory if runtime_memory > runtime_compute else runtime_compute,
        "bound": "memory" if math_ai < model_config.hardware_ai else "compute"
    }

    return proj_rst


def proj_mlp_gate_or_w3(model_config):
    # memory (in & out)
    params_in_input = model_config.batch_size * \
        model_config.seq_len_q * model_config.hidden_size
    params_in_weight = model_config.hidden_size * model_config.intermediate_size
    params_out = model_config.batch_size * \
        model_config.seq_len_q * model_config.intermediate_size
    params_total = params_in_input + params_in_weight + params_out
    bytes_total = params_total * model_config.num_bytes
    runtime_memory = bytes_total / model_config.bw

    # compute (2 for mul & add)
    # [B, T_Q, H] @ [H, H_Inter]
    num_ops = model_config.batch_size * model_config.seq_len_q * \
        model_config.hidden_size * model_config.intermediate_size * 2
    tops = min(model_config.tops, model_config.tops *
               (model_config.batch_size * model_config.seq_len_q / 128))  # 128 for Gaudi2

    # arithmetic intensity (#flops / #bytes)
    math_ai = num_ops / bytes_total
    tops = min(tops, math_ai * model_config.bw)
    runtime_compute = num_ops / tops

    proj_rst = {
        "name": "mlp_gate(w3)",
        "#ops": num_ops,
        "#mem": bytes_total,
        "math_ai": math_ai,
        "tops_roofline": tops,
        "latency": runtime_memory if runtime_memory > runtime_compute else runtime_compute,
        "bound": "memory" if math_ai < model_config.hardware_ai else "compute"
    }

    return proj_rst


def proj_mlp_up_or_w1(model_config):
    # memory (in & out)
    params_in_input = model_config.batch_size * \
        model_config.seq_len_q * model_config.hidden_size
    params_in_weight = model_config.hidden_size * model_config.intermediate_size
    params_out = model_config.batch_size * \
        model_config.seq_len_q * model_config.intermediate_size
    params_total = params_in_input + params_in_weight + params_out
    bytes_total = params_total * model_config.num_bytes
    runtime_memory = bytes_total / model_config.bw

    # compute (2 for mul & add)
    # [B, T_Q, H] @ [H, H_Inter]
    num_ops = model_config.batch_size * model_config.seq_len_q * \
        model_config.hidden_size * model_config.intermediate_size * 2
    tops = min(model_config.tops, model_config.tops *
               (model_config.batch_size * model_config.seq_len_q / 128))  # 128 for Gaudi2

    # arithmetic intensity (#flops / #bytes)
    math_ai = num_ops / bytes_total
    tops = min(tops, math_ai * model_config.bw)
    runtime_compute = num_ops / tops

    proj_rst = {
        "name": "mlp_up(w1)",
        "#ops": num_ops,
        "#mem": bytes_total,
        "math_ai": math_ai,
        "tops_roofline": tops,
        "latency": runtime_memory if runtime_memory > runtime_compute else runtime_compute,
        "bound": "memory" if math_ai < model_config.hardware_ai else "compute"
    }

    return proj_rst


def proj_mlp_down_or_w2(model_config):
    # memory (in & out)
    params_in_input = model_config.batch_size * \
        model_config.seq_len_q * model_config.intermediate_size
    params_in_weight = model_config.intermediate_size * model_config.hidden_size
    params_out = model_config.batch_size * \
        model_config.seq_len_q * model_config.hidden_size
    params_total = params_in_input + params_in_weight + params_out
    bytes_total = params_total * model_config.num_bytes
    runtime_memory = bytes_total / model_config.bw

    # compute (2 for mul & add)
    # [B, T_Q, H_Inter] @ [H_Inter, H]
    num_ops = model_config.batch_size * model_config.seq_len_q * \
        model_config.hidden_size * model_config.intermediate_size * 2
    tops = min(model_config.tops, model_config.tops *
               (model_config.batch_size * model_config.seq_len_q / 128))  # 128 for Gaudi2

    # arithmetic intensity (#flops / #bytes)
    math_ai = num_ops / bytes_total
    tops = min(tops, math_ai * model_config.bw)
    runtime_compute = num_ops / tops

    proj_rst = {
        "name": "mlp_down(w2)",
        "#ops": num_ops,
        "#mem": bytes_total,
        "math_ai": math_ai,
        "tops_roofline": tops,
        "latency": runtime_memory if runtime_memory > runtime_compute else runtime_compute,
        "bound": "memory" if math_ai < model_config.hardware_ai else "compute"
    }

    return proj_rst


def proj_attn(model_config):
    qk = proj_attn_qk(model_config)
    softmax = proj_attn_softmax(model_config)
    scorev = proj_attn_scorev(model_config)
    runtime_attn = qk["latency"] + softmax["latency"] + scorev["latency"]

    return runtime_attn, (qk, softmax, scorev)


def proj_mlp(model_config):
    up = proj_mlp_up_or_w1(model_config)
    if model_config.with_gate:
        gate = proj_mlp_gate_or_w3(model_config)
    down = proj_mlp_down_or_w2(model_config)

    runtime_mlp = up["latency"] + down["latency"]
    if model_config.with_gate:
        runtime_mlp += gate["latency"]

    return runtime_mlp, (up, down, gate if model_config.with_gate else None)


def proj_moe(model_config):
    runtime_mlp, mlp_items = proj_mlp(model_config)
    runtime_moe = runtime_mlp * model_config.num_experts

    return runtime_moe, mlp_items


def proj_single_layer(model_config):
    qkvo_proj = proj_qkvo_proj(model_config)
    runtime_attn, attn_items = proj_attn(model_config)
    runtime_moe, moe_items = proj_moe(model_config)
    runtime_single_layer = qkvo_proj["latency"] + runtime_attn + runtime_moe

    single_layer_items = {
        "qkvo": qkvo_proj,
        "attn": attn_items,
        "moe": moe_items,
    }

    return runtime_single_layer, single_layer_items


def proj_decoder(model_config):
    runtime_single_layer, single_layer_items = proj_single_layer(model_config)
    runtime_decoder = runtime_single_layer * model_config.num_layers

    return runtime_decoder, single_layer_items


def print_projection(projection_dict):
    proj_prefill, proj_decode = projection_dict["prefill"], projection_dict["decode"]
    # for key, projection in projection_dict.items():
    for (_, prefill), (_, decode) in zip(proj_prefill.items(), proj_decode.items()):
        print("prefill".center(130))
        for data in prefill:
            print(tabulate(data))
        print("decode".center(130))
        for data in decode:
            print(tabulate(data))
        print("\n\n")


def print_analysis(analysis_dict, batchsize_list):
    analysis_prefill, analysis_decode = analysis_dict["prefill"], analysis_dict["decode"]
    # for key, analysis in analysis_dict.items():
    for prefill, decode in zip(analysis_prefill, analysis_decode):
        for bs in [batchsize_list[0], batchsize_list[-1]]:
            print("prefill".center(100))
            print(tabulate(prefill[bs]))
            print("decode".center(100))
            print(tabulate(decode[bs]))
            print("\n\n")


def plot_projection(projection_dict, batchsize_list):
    for key, projection in projection_dict.items():
        if key == "decode":
            plt.figure(figsize=(20, 10))
            for dtype, proj in projection.items():
                for data in proj:
                    proj_list = []
                    for i in range(0, len(batchsize_list)):
                        proj_list.append(data[i+1][-1])
                    device, input, output = data[1][0], data[1][8], data[1][9]
                    plt.plot(batchsize_list, proj_list,
                             label=f"{device}_{dtype}_{input}_{output}")
                    for b, p in zip(batchsize_list, proj_list):
                        plt.text(b, p, p, ha='right', va='bottom', fontsize=9)
            plt.xticks(batchsize_list, batchsize_list)
            plt.tick_params(axis='x', rotation=70)
            plt.xlabel("batch size")
            plt.ylabel("tokens / s")
            plt.title("throughput")
            plt.grid(axis='x')
            plt.legend()
            plt.show()
            plt.savefig("./figure/decode_projection.png")
