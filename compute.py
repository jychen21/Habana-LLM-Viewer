# https://arxiv.org/pdf/2402.16363
import os
import math
import csv
from tabulate import tabulate
import matplotlib.pyplot as plt
from config import *


def proj_qkvo_proj(config):
    hidden_size = config.model_config.hidden_size
    num_bytes = config.hardware_config.num_bytes
    batch_size = config.input_config.batch_size
    seq_len_q = config.input_config.seq_len_q
    bw = config.hardware_config.hbm_bandwidth_org
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor
    pipeline = config.hardware_config.pipeline
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
    if (batch_size * seq_len_q) % flops_mme_factor != 0:
        num_rounds = math.ceil(batch_size * seq_len_q / flops_mme_factor)
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
    num_bytes = config.hardware_config.num_bytes
    batch_size = config.input_config.batch_size
    seq_len_q = config.input_config.seq_len_q
    seq_len_kv = config.input_config.seq_len_kv
    bw = config.hardware_config.hbm_bandwidth
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor_attn
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
    flops_mme_factor = config.hardware_config.flops_mme_factor_attn
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
    bw = config.hardware_config.hbm_bandwidth_org
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor
    pipeline = config.hardware_config.pipeline
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
    if (batch_size * seq_len_q) % flops_mme_factor != 0:
        num_rounds = math.ceil(batch_size * seq_len_q / flops_mme_factor)
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
    bw = config.hardware_config.hbm_bandwidth_org
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor
    pipeline = config.hardware_config.pipeline
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
    if (batch_size * seq_len_q) % flops_mme_factor != 0:
        num_rounds = math.ceil(batch_size * seq_len_q / flops_mme_factor)
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
    bw = config.hardware_config.hbm_bandwidth_org
    flops_mme = config.hardware_config.flops_mme
    flops_mme_factor = config.hardware_config.flops_mme_factor
    pipeline = config.hardware_config.pipeline
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
    if (batch_size * seq_len_q) % flops_mme_factor != 0:
        num_rounds = math.ceil(batch_size * seq_len_q / flops_mme_factor)
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
    kvcache_bucket = config.kvcache_bucket
    bucket_perf_gain = config.bucket_perf_gain

    qk = proj_attn_qk(config)
    softmax = proj_attn_softmax(config)
    scorev = proj_attn_scorev(config)
    runtime_attn = (qk["latency"] + softmax["latency"] + scorev["latency"])

    if kvcache_bucket:
        runtime_attn /= bucket_perf_gain

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

    hidden_size = config.model_config.hidden_size
    inter_size = config.model_config.intermediate_size
    num_heads_q = config.model_config.num_heads_q
    num_heads_kv = config.model_config.num_heads_kv
    head_dim = hidden_size // num_heads_q
    seq_len_q = config.input_config.seq_len_q
    seq_len_kv = config.input_config.seq_len_kv

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
    }

    return runtime_single_layer, single_layer_items


def proj_decoder(config):
    num_layers = config.model_config.num_layers

    runtime_single_layer, single_layer_items = proj_single_layer(config)
    runtime_decoder = runtime_single_layer * num_layers

    return runtime_decoder, single_layer_items


def do_projection(model_name, device, type, pp, tp, dtype, input, output, bs, kvcache_bucket=None):
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

    for step in range(output):
        if step == 0:  # prefill
            seq_len_q = seq_len_kv = input
            cfg = Config(device, type, dtype, pp, tp, hidden_size, num_heads_q, num_heads_kv,
                         intermediate_size, mlp_with_gate, num_experts, num_layers_mlp,
                         num_layers_moe, seq_len_q, seq_len_kv, bs, is_decoding=False)
            proj_rst["prefill"] = proj_decoder(cfg)
        else:  # decoding
            seq_len_q = 1
            seq_len_kv = input + output
            if kvcache_bucket is not None:
                seq_len_kv = input + \
                    math.ceil(step / kvcache_bucket) * kvcache_bucket
            cfg = Config(device, type, dtype, pp, tp, hidden_size, num_heads_q, num_heads_kv,
                         intermediate_size, mlp_with_gate, num_experts, num_layers_mlp,
                         num_layers_moe, seq_len_q, seq_len_kv, bs, is_decoding=True)
            proj_decoding_steps.append(proj_decoder(cfg))

    proj_rst["decode"] = proj_decoding_steps

    return proj_rst


def create_data_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def plot_overall_projection(model_name, device, type, pp, tp, figure_name, proj_data, batchsize_list):
    plt.figure(figsize=(20, 10))
    for dtype, dtype_proj in proj_data.items():
        for input, input_proj in dtype_proj.items():
            for output, output_proj in input_proj.items():
                proj_list = []
                for proj in output_proj:
                    proj_list.append(proj[-1])
                plt.plot(batchsize_list, proj_list,
                         label=f"{dtype}_{input}_{output}")
                for bs, proj in zip(batchsize_list, proj_list):
                    plt.text(bs, proj, proj, ha='right',
                             va='bottom', fontsize=9)
    plt.xticks(batchsize_list, batchsize_list)
    plt.tick_params(axis='x', rotation=70)
    plt.xlabel("batch size")
    plt.ylabel("tokens / s")
    plt.title(f"{model_name}_{device}{type}_pp{pp}_tp{tp}_throughput")
    plt.grid(axis='x')
    plt.legend()
    plt.show()
    plt.savefig(figure_name)
    plt.clf()


def print_overall_projection(model_name, proj_dict, kvcache_bucket, batchsize_list, to_csv=True, plot=True):
    proj_item = ["Model", "Device", "PP", "TP", "DType", "Input", "Output", "BS", "KVCacheBucket", "Prefill(ms)",
                 "DecodeMin(ms)", "DecodeMax(ms)", "DecodeAvg(ms)", "Latency(ms)", "Throughput(tokens/sec)"]

    milli_secs = 1e3

    for device, device_proj in proj_dict.items():
        for type, type_proj in device_proj.items():
            proj_data = [proj_item]
            for pp, pp_proj in type_proj.items():
                for tp, tp_proj in pp_proj.items():
                    figure_data = {}
                    for dtype, dtype_proj in tp_proj.items():
                        figure_data[dtype] = {}
                        for input, input_proj in dtype_proj.items():
                            figure_data[dtype][input] = {}
                            for output, output_proj in input_proj.items():
                                figure_data[dtype][input][output] = []
                                for bs, proj_rst in output_proj:
                                    proj_prefill_step = proj_rst["prefill"]
                                    proj_decode_steps = proj_rst["decode"]
                                    prefill_latency = round(
                                        proj_prefill_step[0] * milli_secs, 2)
                                    decode_latency_list = [step[0]
                                                           for step in proj_decode_steps]
                                    decode_latency_min = round(
                                        decode_latency_list[0] * milli_secs, 2)
                                    decode_latency_max = round(
                                        decode_latency_list[-1] * milli_secs, 2)
                                    decode_latency_avg = round(
                                        sum(decode_latency_list)/len(decode_latency_list) * milli_secs, 2)
                                    overall_latency = round(((proj_prefill_step[0] + sum(decode_latency_list)) /
                                                            (len(decode_latency_list) + 1)) * milli_secs, 2)
                                    throughput = round(
                                        (1 / overall_latency) * bs * milli_secs, 2)
                                    proj_data.append([model_name, f"{device}{type}", pp, tp, dtype, input, output,
                                                      bs, kvcache_bucket, prefill_latency, decode_latency_min,
                                                      decode_latency_max, decode_latency_avg, overall_latency,
                                                      throughput])
                                    figure_data[dtype][input][output].append([model_name, f"{device}{type}", pp, tp, dtype, input,
                                                                              output, bs, kvcache_bucket, prefill_latency,
                                                                              decode_latency_min, decode_latency_max,
                                                                              decode_latency_avg, overall_latency, throughput])
                                proj_data.append([""] * len(proj_item))

                    if plot:
                        model_dir = f"./data/{model_name}"
                        create_data_dir(model_dir)
                        figure_name = f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_overall_projection.png"
                        plot_overall_projection(
                            model_name, device, type, pp, tp, figure_name, figure_data, batchsize_list)

            print(f"{model_name}_{device}{type}_overall_projection".center(150))
            print(tabulate(proj_data))

            if to_csv:
                model_dir = f"./data/{model_name}"
                create_data_dir(model_dir)
                with open(f"{model_dir}/{device}{type}_overall_projection.csv", "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(proj_data)


def print_overall_projection_in_detail(model_name, proj_dict, kvcache_bucket, to_csv=True):
    proj_item = ["DType", "Input", "Output", "BS", "HiddenSize", "HeadsQ", "HeadsKV", "InterSize",
                 "WithGate", "Experts", "Layers", "KVCacheBucket", "Prefill(ms)", "DecodeMin(ms)",
                 "DecodeMax(ms)", "DecodeAvg(ms)", "Latency(ms)", "Throughput(tokens/sec)"]

    milli_secs = 1e3

    model = ModelDict[model_name]
    hidden_size = model["hidden_size"]
    num_heads_q = model["num_heads_q"]
    num_heads_kv = model["num_heads_kv"]
    intermediate_size = model["intermediate_size"]
    mlp_with_gate = model["mlp_with_gate"]
    num_layers_mlp = model["num_layers_mlp"]
    num_layers_moe = model["num_layers_moe"]
    num_layers = num_layers_mlp + num_layers_moe
    num_experts = model["num_experts"]

    for device, device_proj in proj_dict.items():
        for type, type_proj in device_proj.items():
            for pp, pp_proj in type_proj.items():
                for tp, tp_proj in pp_proj.items():
                    proj_data = [proj_item]
                    for dtype, dtype_proj in tp_proj.items():
                        for input, input_proj in dtype_proj.items():
                            for output, output_proj in input_proj.items():
                                for bs, proj_rst in output_proj:
                                    proj_prefill_step = proj_rst["prefill"]
                                    proj_decode_steps = proj_rst["decode"]
                                    prefill_latency = round(
                                        proj_prefill_step[0] * milli_secs, 2)
                                    decode_latency_list = [step[0]
                                                           for step in proj_decode_steps]
                                    decode_latency_min = round(
                                        decode_latency_list[0] * milli_secs, 2)
                                    decode_latency_max = round(
                                        decode_latency_list[-1] * milli_secs, 2)
                                    decode_latency_avg = round(
                                        sum(decode_latency_list)/len(decode_latency_list) * milli_secs, 2)
                                    overall_latency = round(((proj_prefill_step[0] + sum(decode_latency_list)) /
                                                            (len(decode_latency_list) + 1)) * milli_secs, 2)
                                    throughput = round(
                                        (1 / overall_latency) * bs * milli_secs, 2)
                                    proj_data.append([dtype, input, output, bs, hidden_size, num_heads_q, num_heads_kv,
                                                      intermediate_size, mlp_with_gate, num_experts, num_layers, kvcache_bucket,
                                                      prefill_latency, decode_latency_min, decode_latency_max, decode_latency_avg,
                                                      overall_latency, throughput])
                                proj_data.append([""] * len(proj_item))
                    print(
                        f"{model_name}_{device}{type}_pp{pp}_tp{tp}_overall_projection_in_detatil".center(200))
                    print(tabulate(proj_data))

                    if to_csv:
                        model_dir = f"./data/{model_name}"
                        create_data_dir(model_dir)
                        with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_overall_projection_in_detatil.csv", "w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerows(proj_data)


def print_layer_projection(model_name, proj_dict, to_csv=True):
    bmm_layer_proj_item = ["Device", "DType", "BS", "HeadsQ", "HeadsKV", "SeqLenQ", "SeqLenKV", "HeadDim", "Ops(QK+SV)(M)",
                           "Size(MB)", "TFLOPs", "BW(TB/s)", "Memory(us)", "Compute(us)", "ProjectLatency(us)", "Bound"]
    mm_layer_proj_item = ["Device", "DType", "BS", "SeqLenQ", "SeqLenKV", "HiddenSize", "IntermediateSize", "Ops(up)(M)",
                          "Size(MB)", "TFLOPs", "BW(TB/s)", "Memory(us)", "Compute(us)", "ProjectLatency(us)", "Bound"]

    megabytes = 1024 * 1024
    megaparam = 1024 * 1024
    microsecs = 1e6
    tflops = 1e12
    tbw = 1e12

    model = ModelDict[model_name]
    hidden_size = model["hidden_size"]
    num_heads_q = model["num_heads_q"]
    headdim = hidden_size // num_heads_q
    num_heads_kv = model["num_heads_kv"]
    intermediate_size = model["intermediate_size"]

    for device, device_proj in proj_dict.items():
        for type, type_proj in device_proj.items():
            for pp, pp_proj in type_proj.items():
                for tp, tp_proj in pp_proj.items():
                    for dtype, dtype_proj in tp_proj.items():
                        bmm_layer_proj_prefill_list = [bmm_layer_proj_item]
                        mm_layer_proj_prefill_list = [mm_layer_proj_item]
                        bmm_layer_proj_decode_list = [bmm_layer_proj_item]
                        mm_layer_proj_decode_list = [mm_layer_proj_item]
                        for input, input_proj in dtype_proj.items():
                            for output, output_proj in input_proj.items():
                                for bs, proj_rst in output_proj:
                                    proj_prefill_step = proj_rst["prefill"]
                                    proj_decode_steps = proj_rst["decode"]

                                    # prefill
                                    layer_proj_prefill = proj_prefill_step[1]
                                    tq = layer_proj_prefill["tq"]
                                    tkv = layer_proj_prefill["tkv"]
                                    # attn(bmm)
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
                                    attn_tops = round(
                                        qk["tops_roofline"] / tflops, 2)
                                    bw = round(qk["bw"] / tbw, 2)
                                    attn_bound_wo_softmax = qk["bound"]
                                    bmm_layer_proj_prefill_list.append([f"{device}{type}", dtype, bs, num_heads_q, num_heads_kv, tq,
                                                                        tkv, headdim, attn_ops_wo_softmax, attn_size_wo_softmax,
                                                                        attn_tops, bw, attn_mem_time_wo_softmax, attn_cmp_time_wo_softmax,
                                                                        attn_latency_wo_softmax, attn_bound_wo_softmax])
                                    # ffn_up(mm)
                                    up, down, gate = layer_proj_prefill["ffn"]
                                    up_ops = round(
                                        up["operations"] / megaparam, 2)
                                    up_size = round(up["size"] / megabytes, 2)
                                    up_mem_time = round(
                                        up["mem_time"] * microsecs, 2)
                                    up_cmp_time = round(
                                        up["cmp_time"] * microsecs, 2)
                                    up_latency = round(
                                        up["latency"] * microsecs, 2)
                                    up_tops = round(
                                        up["tops_roofline"] / tflops, 2)
                                    bw = round(up["bw"] / tbw, 2)
                                    up_bound = up["bound"]
                                    mm_layer_proj_prefill_list.append([f"{device}{type}", dtype, bs, tq, tkv, hidden_size,
                                                                       intermediate_size, up_ops, up_size, up_tops, bw,
                                                                       up_mem_time, up_cmp_time, up_latency, up_bound])

                                    # decoding
                                    for decode_step in proj_decode_steps:
                                        layer_proj_decode = decode_step[1]
                                        tq = layer_proj_decode["tq"]
                                        tkv = layer_proj_prefill["tkv"]
                                        # attn(bmm)
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
                                        attn_tops = round(
                                            qk["tops_roofline"] / tflops, 2)
                                        bw = round(qk["bw"] / tbw, 2)
                                        attn_bound_wo_softmax = qk["bound"]
                                        bmm_layer_proj_decode_list.append([f"{device}{type}", dtype, bs, num_heads_q, num_heads_kv, tq,
                                                                           tkv, headdim, attn_ops_wo_softmax, attn_size_wo_softmax,
                                                                           attn_tops, bw, attn_mem_time_wo_softmax, attn_cmp_time_wo_softmax,
                                                                           attn_latency_wo_softmax, attn_bound_wo_softmax])
                                        # ffn_up(mm)
                                        up, down, gate = layer_proj_decode["ffn"]
                                        up_ops = round(
                                            up["operations"] / megaparam, 2)
                                        up_size = round(
                                            up["size"] / megabytes, 2)
                                        up_mem_time = round(
                                            up["mem_time"] * microsecs, 2)
                                        up_cmp_time = round(
                                            up["cmp_time"] * microsecs, 2)
                                        up_latency = round(
                                            up["latency"] * microsecs, 2)
                                        up_tops = round(
                                            up["tops_roofline"] / tflops, 2)
                                        bw = round(up["bw"] / tbw, 2)
                                        up_bound = up["bound"]
                                        mm_layer_proj_decode_list.append([f"{device}{type}", dtype, bs, tq, tkv, hidden_size,
                                                                          intermediate_size, up_ops, up_size, up_tops, bw,
                                                                          up_mem_time, up_cmp_time, up_latency, up_bound])
                                bmm_layer_proj_prefill_list.append(
                                    [""] * len(bmm_layer_proj_item))
                                mm_layer_proj_prefill_list.append(
                                    [""] * len(mm_layer_proj_item))
                                bmm_layer_proj_decode_list.append(
                                    [""] * len(bmm_layer_proj_item))
                                mm_layer_proj_decode_list.append(
                                    [""] * len(mm_layer_proj_item))

                        '''
                        print(f"{model_name}_{device}{type}_pp{pp}_tp{tp}_prefill_attn_qksv(bmm)_projection".center(200))
                        print(tabulate(bmm_layer_proj_prefill_list))
                        print(f"{model_name}_{device}{type}_pp{pp}_tp{tp}_prefill_ffn_up(mm)_projection".center(200))
                        print(tabulate(mm_layer_proj_prefill_list))
                        print(f"{model_name}_{device}{type}_pp{pp}_tp{tp}_decode_attn_qksv(bmm)_projection".center(200))
                        print(tabulate(bmm_layer_proj_decode_list))
                        print(f"{model_name}_{device}{type}_pp{pp}_tp{tp}_decode_ffn_up(mm)_projection".center(200))
                        print(tabulate(mm_layer_proj_decode_list))
                        '''

                        if to_csv:
                            model_dir = f"./data/{model_name}"
                            create_data_dir(model_dir)
                            with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_{dtype}_prefill_attn_qksv(bmm)_projection.csv", "w", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(bmm_layer_proj_prefill_list)
                            with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_{dtype}_prefill_ffn_up(mm)_projection.csv", "w", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(mm_layer_proj_prefill_list)
                            with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_{dtype}_decode_attn_qksv(bmm)_projection.csv", "w", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(bmm_layer_proj_decode_list)
                            with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_{dtype}_decode_ffn_up(mm)_projection.csv", "w", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(mm_layer_proj_decode_list)


def print_layer_analysis(model_name, proj_dict, to_csv=True):
    analysis_item = ["Model", "Device", "PP", "TP", "DType", "BS", "Input", "Output", "Operation", "SeqLenQ",
                     "SeqLenKV", "NumOps(G)", "Memory(GB)", "TopsRF(TFlops)", "AI", "Bound"]

    gigabytes = 1024 * 1024 * 1024
    gigaparam = 1024 * 1024 * 1024
    tflops = 1e12

    for device, device_proj in proj_dict.items():
        for type, type_proj in device_proj.items():
            for pp, pp_proj in type_proj.items():
                for tp, tp_proj in pp_proj.items():
                    for dtype, dtype_proj in tp_proj.items():
                        layer_analysis_prefill = [analysis_item]
                        layer_analysis_decode = [analysis_item]
                        for input, input_proj in dtype_proj.items():
                            for output, output_proj in input_proj.items():
                                for bs, proj_rst in output_proj:
                                    proj_prefill_step = proj_rst["prefill"]
                                    proj_decode_steps = proj_rst["decode"]

                                    # prefill
                                    layer_proj_prefill = proj_prefill_step[1]
                                    tq = layer_proj_prefill["tq"]
                                    tkv = layer_proj_prefill["tkv"]
                                    # qkvo / qk, softmax, scorev / up, down, gate
                                    for mod_name in ["qkvo", "attn", "ffn"]:
                                        module = layer_proj_prefill[mod_name]
                                        if isinstance(module, dict):
                                            op = module
                                            name = op["name"]
                                            num_ops = round(
                                                op["operations"] / gigaparam, 2)
                                            size = round(
                                                op["size"] / gigabytes, 2)
                                            tops = round(
                                                op["tops_roofline"] / tflops, 2)
                                            math_ai = round(
                                                op["math_ai"], 2)
                                            bound = op["bound"]
                                            layer_analysis_prefill.append([model_name, f"{device}{type}",
                                                                           pp, tp, dtype, bs, input, output,
                                                                           name, tq, tkv, num_ops, size, tops,
                                                                           math_ai, bound])
                                        elif isinstance(module, tuple):
                                            for op in module:
                                                if op is not None:
                                                    name = op["name"]
                                                    num_ops = round(
                                                        op["operations"] / gigaparam, 2)
                                                    size = round(
                                                        op["size"] / gigabytes, 2)
                                                    tops = round(
                                                        op["tops_roofline"] / tflops, 2)
                                                    math_ai = round(
                                                        op["math_ai"], 2)
                                                    bound = op["bound"]
                                                    layer_analysis_prefill.append([model_name, f"{device}{type}",
                                                                                   pp, tp, dtype, bs, input, output,
                                                                                   name, tq, tkv, num_ops, size, tops,
                                                                                   math_ai, bound])

                                    # decoding
                                    for decode_step in proj_decode_steps:
                                        layer_proj_decode = decode_step[1]
                                        tq = layer_proj_decode["tq"]
                                        tkv = layer_proj_prefill["tkv"]
                                        # qkvo / qk, softmax, scorev / up, down, gate
                                        for mod_name in ["qkvo", "attn", "ffn"]:
                                            module = layer_proj_prefill[mod_name]
                                            if isinstance(module, dict):
                                                op = module
                                                name = op["name"]
                                                num_ops = round(
                                                    op["operations"] / gigaparam, 2)
                                                size = round(
                                                    op["size"] / gigabytes, 2)
                                                tops = round(
                                                    op["tops_roofline"] / tflops, 2)
                                                math_ai = round(
                                                    op["math_ai"], 2)
                                                bound = op["bound"]
                                                layer_analysis_decode.append([model_name, f"{device}{type}",
                                                                              pp, tp, dtype, bs, input, output,
                                                                              name, tq, tkv, num_ops, size, tops,
                                                                              math_ai, bound])
                                            elif isinstance(module, tuple):
                                                for op in module:
                                                    if op is not None:
                                                        name = op["name"]
                                                        num_ops = round(
                                                            op["operations"] / gigaparam, 2)
                                                        size = round(
                                                            op["size"] / gigabytes, 2)
                                                        tops = round(
                                                            op["tops_roofline"] / tflops, 2)
                                                        math_ai = round(
                                                            op["math_ai"], 2)
                                                        bound = op["bound"]
                                                        layer_analysis_decode.append([model_name, f"{device}{type}",
                                                                                      pp, tp, dtype, bs, input, output,
                                                                                      name, tq, tkv, num_ops, size, tops,
                                                                                      math_ai, bound])
                                        layer_analysis_decode.append(
                                            [""] * len(analysis_item))

                                layer_analysis_prefill.append(
                                    [""] * len(analysis_item))
                                layer_analysis_decode.append(
                                    [""] * len(analysis_item))

                        '''
                        print(
                            f"{model_name}_{device}{type}_pp{pp}_tp{tp}_prefill_layer_analysis".center(150))
                        print(tabulate(layer_analysis_prefill))
                        print(
                            f"{model_name}_{device}{type}_pp{pp}_tp{tp}_decode_layer_analysis".center(150))
                        print(tabulate(layer_analysis_decode))
                        '''

                        if to_csv:
                            model_dir = f"./data/{model_name}"
                            create_data_dir(model_dir)
                            with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_{dtype}_prefill_layer_analysis.csv", "w", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(layer_analysis_prefill)
                            with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_{dtype}_decode_layer_analysis.csv", "w", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(layer_analysis_decode)


def print_projection(model_name, proj_dict, kvcache_bucket, batchsize_list, to_csv=True, plot=True):
    print_overall_projection(model_name, proj_dict,
                             kvcache_bucket, batchsize_list, to_csv, plot)
    print_overall_projection_in_detail(
        model_name, proj_dict, kvcache_bucket, to_csv)
    print_layer_projection(model_name, proj_dict, to_csv)
    print_layer_analysis(model_name, proj_dict, to_csv)
