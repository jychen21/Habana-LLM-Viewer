# https://arxiv.org/pdf/2402.16363
from tabulate import tabulate
import matplotlib.pyplot as plt
from base import *


cmp_item_list = ["Device", "HiddenSize", "HeadsQ", "HeadsKV", "InterSize", "Decoding", "Experts",
                 "Layers", "In", "Out", "DType", "BS", "Latency(s)", "Throughput(tokens/sec)"]
layer_analysis_list = ["Input", "Output", "DataType", "BatchSize", "LayerName",
                       "NumOps(e9)", "Memory(GB)", "TopsRF(TFlops)", "AI", "Bound"]


def proj_qkvo_proj(config):
    hidden_size = config.model_config.hidden_size
    num_bytes = config.hardware_config.num_bytes
    batch_size = config.input_config.batch_size
    seq_len_q = config.input_config.seq_len_q
    bw = config.hardware_config.hbm_bandwidth
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor
    hw_ai_mme = config.hardware_config.hardware_ai_mme

    params_in_input = batch_size * seq_len_q * hidden_size
    params_in_weight = hidden_size * hidden_size
    params_out = batch_size * seq_len_q * hidden_size
    params_total = params_in_input + params_in_weight + params_out
    params_total *= 4  # 4 for qkvo
    bytes_total = params_total * num_bytes
    runtime_memory = bytes_total / bw

    num_ops = batch_size * seq_len_q * hidden_size * hidden_size * 2 * 4  # 4 for qkvo
    tops = min(flops_mme, flops_mme *
               (batch_size * seq_len_q / flops_mme_factor))
    runtime_compute = num_ops / tops

    math_ai = num_ops / bytes_total
    runtime_roofline = runtime_memory if runtime_memory > runtime_compute else runtime_compute
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
    num_bytes = config.hardware_config.num_bytes
    batch_size = config.input_config.batch_size
    seq_len_q = config.input_config.seq_len_q
    seq_len_kv = config.input_config.seq_len_kv
    bw = config.hardware_config.hbm_bandwidth
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor
    hw_ai_mme_attn = config.hardware_config.hardware_ai_mme_attn

    params_in_q = batch_size * num_heads_q * seq_len_q * head_dim
    params_in_k = batch_size * num_heads_kv * seq_len_kv * head_dim
    params_out = batch_size * num_heads_q * seq_len_q * seq_len_kv
    params_total = params_in_q + params_in_k + params_out
    bytes_total = params_total * num_bytes
    runtime_memory = bytes_total / bw

    num_ops = batch_size * num_heads_q * seq_len_q * head_dim * seq_len_kv * 2
    tops = min(flops_mme, flops_mme * (seq_len_q / flops_mme_factor))
    runtime_compute = num_ops / tops

    math_ai = num_ops / bytes_total
    runtime_roofline = runtime_memory if runtime_memory > runtime_compute else runtime_compute
    bound = "memory" if runtime_memory > runtime_compute else "compute"

    proj_rst = {
        "name": "q@k_T",
        "operations": num_ops,
        "size": bytes_total,
        "hw_ai": hw_ai_mme_attn,
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

    # ops(max) + ops(x-max) + ops(exp(x-max)) + ops(sum(exp(x-max))) + ops(x/sum(exp(x-max)))
    num_ops = batch_size * num_heads_q * seq_len_q * \
        seq_len_kv * 5  # 5 for traversal times
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
    num_bytes = config.hardware_config.num_bytes
    batch_size = config.input_config.batch_size
    seq_len_q = config.input_config.seq_len_q
    seq_len_kv = config.input_config.seq_len_kv
    bw = config.hardware_config.hbm_bandwidth
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor
    hw_ai_mme_attn = config.hardware_config.hardware_ai_mme_attn

    params_in_score = batch_size * num_heads_q * seq_len_q * seq_len_kv
    params_in_v = batch_size * num_heads_kv * seq_len_kv * head_dim
    params_out = batch_size * num_heads_q * seq_len_q * head_dim
    params_total = params_in_score + params_in_v + params_out
    bytes_total = params_total * num_bytes
    runtime_memory = bytes_total / bw

    num_ops = batch_size * num_heads_q * seq_len_q * seq_len_kv * head_dim * 2
    tops = min(flops_mme, flops_mme * (seq_len_q / flops_mme_factor))
    runtime_compute = num_ops / tops

    math_ai = num_ops / bytes_total
    runtime_roofline = runtime_memory if runtime_memory > runtime_compute else runtime_compute
    bound = "memory" if runtime_memory > runtime_compute else "compute"

    proj_rst = {
        "name": "score@v",
        "operations": num_ops,
        "size": bytes_total,
        "hw_ai": hw_ai_mme_attn,
        "math_ai": math_ai,
        "tops_roofline": tops,
        "bw": bw,
        "mem_time": runtime_memory,
        "cmp_time": runtime_compute,
        "latency": runtime_roofline,
        "bound": bound
    }

    return proj_rst


def proj_mlp_up_or_w1(config):
    hidden_size = config.model_config.hidden_size
    intermediate_size = config.model_config.intermediate_size
    num_bytes = config.hardware_config.num_bytes
    batch_size = config.input_config.batch_size
    seq_len_q = config.input_config.seq_len_q
    bw = config.hardware_config.hbm_bandwidth
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor
    hw_ai_mme = config.hardware_config.hardware_ai_mme

    params_in_input = batch_size * seq_len_q * hidden_size
    params_in_weight = hidden_size * intermediate_size
    params_out = batch_size * seq_len_q * intermediate_size
    params_total = params_in_input + params_in_weight + params_out
    bytes_total = params_total * num_bytes
    runtime_memory = bytes_total / bw

    num_ops = batch_size * seq_len_q * hidden_size * intermediate_size * 2
    tops = min(flops_mme, flops_mme *
               (batch_size * seq_len_q / flops_mme_factor))
    runtime_compute = num_ops / tops

    math_ai = num_ops / bytes_total
    runtime_roofline = runtime_memory if runtime_memory > runtime_compute else runtime_compute
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
    flops_mme_factor = config.hardware_config.flops_mme_factor
    hw_ai_mme = config.hardware_config.hardware_ai_mme

    params_in_input = batch_size * seq_len_q * intermediate_size
    params_in_weight = intermediate_size * hidden_size
    params_out = batch_size * seq_len_q * hidden_size
    params_total = params_in_input + params_in_weight + params_out
    bytes_total = params_total * num_bytes
    runtime_memory = bytes_total / bw

    num_ops = batch_size * seq_len_q * intermediate_size * hidden_size * 2
    tops = min(flops_mme, flops_mme *
               (batch_size * seq_len_q / flops_mme_factor))
    runtime_compute = num_ops / tops

    math_ai = num_ops / bytes_total
    runtime_roofline = runtime_memory if runtime_memory > runtime_compute else runtime_compute
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
    flops_mme_factor = config.hardware_config.flops_mme_factor
    hw_ai_mme = config.hardware_config.hardware_ai_mme

    params_in_input = batch_size * seq_len_q * hidden_size
    params_in_weight = hidden_size * intermediate_size
    params_out = batch_size * seq_len_q * intermediate_size
    params_total = params_in_input + params_in_weight + params_out
    bytes_total = params_total * num_bytes
    runtime_memory = bytes_total / bw

    num_ops = batch_size * seq_len_q * hidden_size * intermediate_size * 2
    tops = min(flops_mme, flops_mme *
               (batch_size * seq_len_q / flops_mme_factor))
    runtime_compute = num_ops / tops

    math_ai = num_ops / bytes_total
    runtime_roofline = runtime_memory if runtime_memory > runtime_compute else runtime_compute
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
    scorev = proj_attn_scorev(config)
    runtime_attn = (qk["latency"] + softmax["latency"] +
                    scorev["latency"])  # *13/2

    return runtime_attn, (qk, softmax, scorev)


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

    qk = attn_items[0]
    softmax = attn_items[1]
    scorev = attn_items[2]
    up = ffn_items[0]

    hidden_size = config.model_config.hidden_size
    inter_size = config.model_config.intermediate_size
    num_heads_q = config.model_config.num_heads_q
    head_dim = hidden_size // num_heads_q
    batch_size = config.input_config.batch_size
    seq_len_q = config.input_config.seq_len_q

    single_layer_items = {
        "hidden_size": hidden_size,
        "inter_size": inter_size,
        "headsq": num_heads_q,
        "tq": seq_len_q,
        "headdim": head_dim,
        "qkvo": qkvo_proj,
        "attn": attn_items,
        "ffn": ffn_items,
    }

    '''
    print("\n")
    print("qkvo ", batch_size, qkvo_proj["memory"], round(qkvo_proj["math_ai"], 2), round(
        qkvo_proj["tops_roofline"]/1e12, 2), round(qkvo_proj["latency"]*1e6, 2), "us")
    print("qk ", batch_size, qk["memory"], round(qk["math_ai"], 2), round(
        qk["tops_roofline"]/1e12, 2), round(qk["latency"]*1e6, 2), "us")
    print("softmax ", batch_size, softmax["memory"], round(softmax["math_ai"], 2), round(
        softmax["tops_roofline"]/1e12, 2), round(softmax["latency"]*1e6, 2), "us")
    print("scorev ", batch_size, scorev["memory"], round(scorev["math_ai"], 2), round(
        scorev["tops_roofline"]/1e12, 2), round(scorev["latency"]*1e6, 2), "us")
    print("up/down/gate ", batch_size, up["memory"], round(
        up["math_ai"], 2), round(up["tops_roofline"]/1e12, 2), round(up["latency"]*1e6, 2), "us")
    print("qk+sv ", round((qk["latency"]+scorev["latency"])*1e6, 2), "us")
    print("attn ", round(runtime_attn*1e6, 2), "us")
    print("mlp ", round(runtime_moe*1e6, 2), "us")
    print("attn : all ", round(runtime_attn /
          (qkvo_proj["latency"] + runtime_attn + runtime_moe), 4))
    '''

    return runtime_single_layer, single_layer_items


def proj_decoder(config):
    num_layers = config.model_config.num_layers

    runtime_single_layer, single_layer_items = proj_single_layer(config)
    runtime_decoder = runtime_single_layer * num_layers

    return runtime_decoder, single_layer_items


def print_projection(projection_dict, model_name):
    proj_prefill, proj_decode = projection_dict["prefill"], projection_dict["decode"]
    for (_, prefill), (_, decode) in zip(proj_prefill.items(), proj_decode.items()):
        print(f"{model_name}_prefill".center(130))
        for data in prefill:
            print(tabulate(data))
        print(f"{model_name}_decode".center(130))
        for data in decode:
            print(tabulate(data))
        print("\n\n")


def print_layer_projection(layer_projection_dict, device, type, model_name, context_list, batchsize_list):
    bmm_layer_proj_item = ["Device", "DType", "BS", "HeadsQ", "SeqLenQ", "HeadDim", "Ops(QK+SV)(M)", "Size(MB)",
                           "TFLOPs", "BW(TB/s)", "Memory(us)", "Compute(us)", "ProjectLatency(us)", "Bound"]
    mm_layer_proj_item = ["Device", "DType", "BS", "SeqLenQ", "HiddenSize", "IntermediateSize", "Ops(up)(M)",
                          "Size(MB)", "TFLOPs", "BW(TB/s)", "Memory(us)", "Compute(us)", "ProjectLatency(us)", "Bound"]
    megabytes = 1024 * 1024
    megaparam = 1024 * 1024
    microsecs = 1e6
    tflops = 1e12
    tbw = 1e12

    layer_proj_prefill, layer_proj_decode = layer_projection_dict[
        "prefill"], layer_projection_dict["decode"]
    for (dtype, layer_proj_dict_prefill), (_, layer_proj_dict_decode) in \
            zip(layer_proj_prefill.items(), layer_proj_decode.items()):
        for in_out in range(len(context_list)):
            bmm_layer_proj_prefill_list = [bmm_layer_proj_item]
            bmm_layer_proj_decode_list = [bmm_layer_proj_item]
            mm_layer_proj_prefill_list = [mm_layer_proj_item]
            mm_layer_proj_decode_list = [mm_layer_proj_item]

            for bs_idx in range(len(batchsize_list)):
                layer_proj_prefill = layer_proj_dict_prefill[in_out][bs_idx]
                layer_proj_decode = layer_proj_dict_decode[in_out][bs_idx]
                bs = batchsize_list[bs_idx]
                hidden_size = layer_proj_prefill["hidden_size"]
                inter_size = layer_proj_prefill["inter_size"]
                headsq = layer_proj_prefill["headsq"]
                headdim = layer_proj_prefill["headdim"]

                # attn(bmm) prefill
                tq = layer_proj_prefill["tq"]
                qk, softmax, scorev = layer_proj_prefill["attn"]
                attn_ops_wo_softmax = round(
                    (qk["operations"] + scorev["operations"]) / megaparam, 2)
                attn_size_wo_softmax = round(
                    (qk["size"] + scorev["size"]) / megabytes, 2)
                attn_mem_time_wo_softmax = round(
                    (qk["mem_time"] + scorev["mem_time"]) * microsecs, 2)
                attn_cmp_time_wo_softmax = round(
                    (qk["cmp_time"] + scorev["cmp_time"]) * microsecs, 2)
                attn_latency_wo_softmax = round(
                    (qk["latency"] + scorev["latency"]) * microsecs, 2)
                attn_tops = round(qk["tops_roofline"] / tflops, 2)
                bw = round(qk["bw"] / tbw, 2)
                attn_bound_wo_softmax = qk["bound"]
                bmm_layer_proj_prefill_list.append([f"{device}{type}", dtype, bs, headsq, tq, headdim,
                                                    attn_ops_wo_softmax, attn_size_wo_softmax, attn_tops,
                                                    bw, attn_mem_time_wo_softmax, attn_cmp_time_wo_softmax,
                                                    attn_latency_wo_softmax, attn_bound_wo_softmax])
                # attn(bmm) decode
                tq = layer_proj_decode["tq"]
                qk, softmax, scorev = layer_proj_decode["attn"]
                attn_ops_wo_softmax = round(
                    (qk["operations"] + scorev["operations"]) / megaparam, 2)
                attn_size_wo_softmax = round(
                    (qk["size"] + scorev["size"]) / megabytes, 2)
                attn_mem_time_wo_softmax = round(
                    (qk["mem_time"] + scorev["mem_time"]) * microsecs, 2)
                attn_cmp_time_wo_softmax = round(
                    (qk["cmp_time"] + scorev["cmp_time"]) * microsecs, 2)
                attn_latency_wo_softmax = round(
                    (qk["latency"] + scorev["latency"]) * microsecs, 2)
                attn_tops = round(qk["tops_roofline"] / tflops, 2)
                bw = round(qk["bw"] / tbw, 2)
                attn_bound_wo_softmax = qk["bound"]
                bmm_layer_proj_decode_list.append([f"{device}{type}", dtype, bs, headsq, tq, headdim,
                                                   attn_ops_wo_softmax, attn_size_wo_softmax, attn_tops,
                                                   bw, attn_mem_time_wo_softmax, attn_cmp_time_wo_softmax,
                                                   attn_latency_wo_softmax, attn_bound_wo_softmax])

                # ffn_up(mm) prefill / decode
                up, down, gate = layer_proj_prefill["ffn"]
                tq = layer_proj_prefill["tq"]
                up_ops = round(up["operations"] / megaparam, 2)
                up_size = round(up["size"] / megabytes, 2)
                up_mem_time = round(up["mem_time"] * microsecs, 2)
                up_cmp_time = round(up["cmp_time"] * microsecs, 2)
                up_latency = round(up["latency"] * microsecs, 2)
                up_tops = round(up["tops_roofline"] / tflops, 2)
                bw = round(up["bw"] / tbw, 2)
                up_bound = up["bound"]
                mm_layer_proj_prefill_list.append([f"{device}{type}", dtype, bs, tq, hidden_size, inter_size, up_ops,
                                                   up_size, up_tops, bw, up_mem_time, up_cmp_time, up_latency, up_bound])
                # ffn_up(mm) decode
                up, down, gate = layer_proj_decode["ffn"]
                tq = layer_proj_decode["tq"]
                up_ops = round(up["operations"] / megaparam, 2)
                up_size = round(up["size"] / megabytes, 2)
                up_mem_time = round(up["mem_time"] * microsecs, 2)
                up_cmp_time = round(up["cmp_time"] * microsecs, 2)
                up_latency = round(up["latency"] * microsecs, 2)
                up_tops = round(up["tops_roofline"] / tflops, 2)
                bw = round(up["bw"] / tbw, 2)
                up_bound = up["bound"]
                mm_layer_proj_decode_list.append([f"{device}{type}", dtype, bs, tq, hidden_size, inter_size, up_ops,
                                                  up_size, up_tops, bw, up_mem_time, up_cmp_time, up_latency, up_bound])

            print(f"{model_name}_prefill_attn(bmm)_projection".center(150))
            print(tabulate(bmm_layer_proj_prefill_list))
            print(f"{model_name}_decode_attn(bmm)_projection".center(150))
            print(tabulate(bmm_layer_proj_decode_list), "\n\n")
            print(f"{model_name}_prefill_ffn_up(mm)_projection".center(150))
            print(tabulate(mm_layer_proj_prefill_list))
            print(f"{model_name}_decode_ffn_up(mm)_projection".center(150))
            print(tabulate(mm_layer_proj_decode_list))


def print_analysis(analysis_dict, batchsize_list, model_name):
    analysis_prefill, analysis_decode = analysis_dict["prefill"], analysis_dict["decode"]
    for prefill, decode in zip(analysis_prefill, analysis_decode):
        for bs in [batchsize_list[0], batchsize_list[-1]]:
            print(f"{model_name}_prefill".center(100))
            print(tabulate(prefill[bs]))
            print(f"{model_name}_decode".center(100))
            print(tabulate(decode[bs]))
            print("\n\n")


def plot_projection(projection_dict, batchsize_list, model_name):
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
            plt.title(f"{model_name}_throughput")
            plt.grid(axis='x')
            plt.legend()
            plt.show()
            plt.savefig("./figure/decode_projection.png")
