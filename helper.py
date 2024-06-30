import os
import csv
import json
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

import plotly.graph_objs as go
import plotly.io as pio

from config import *


proj_cfg = {
    "model_list": ["Llama2-7B"],
    "device_list": ["IntelGaudi2"],
    "type_list": ["B"],
    "parallel": {
        "pp_list": [1],
        "tp_list": [1],
    },
    "dtype_list": ["BF16"],
    "context": {
        "input_list": [512, 1024, 2048],
        "output_list": [512],
    },
    "bs_list": [1] + [i for i in range(2, 513, 2)],
    "optims": {
        "kvcache_bucket": 256
    }
}


def dump_json(path, data):
    with open(path, "w") as fp:
        json_string = json.dumps(
            data, default=lambda o: o.__dict__, sort_keys=False, indent=4)
        fp.write(json_string)


def create_data_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def plot_steps_projection(model_name, device, type, dtype, pp, tp, input, output, kvcache_bucket,
                          batch_sizes, seq_lens, batched_latencies, batched_throughput):
    import plotly.colors as pc
    import plotly.express as px

    def assign_color_shades(batch_size, latencies):
        color_scales = {
            'Reds': pc.sequential.Reds,
            'Oranges': pc.sequential.Oranges,
            'Yellows': pc.sequential.YlOrBr,
            'Greens': pc.sequential.Greens,
            'Blues': pc.sequential.Blues,
            'Purples': pc.sequential.Purples,
            'Pinks': pc.sequential.Pinkyl
        }
        colors = list(color_scales.keys())
        color_scale_name = colors[batch_sizes.index(batch_size) % len(colors)]

        num_shades = len(latencies)
        color_shades = pc.sample_colorscale(color_scales[color_scale_name], [
                                            i/(num_shades-1) for i in range(num_shades)])

        min_latency = min(latencies)
        max_latency = max(latencies)
        normalized_latencies = [
            (latency - min_latency) / (max_latency - min_latency) for latency in latencies]

        # Assign a color shade to each normalized latency
        assigned_colors = [color_shades[int(
            norm_latency * (num_shades - 1))] for norm_latency in normalized_latencies]

        return assigned_colors

    def plot_helper():
        data = []

        for batch_size, latencies in zip(batch_sizes, batched_latencies):
            max_latency = max(latencies)
            bars = []
            cumulative_latency = 0
            color_shades = assign_color_shades(batch_size, latencies)
            for seq_len, latency, color in zip(seq_lens, latencies, color_shades):
                # color = generate_color(batch_size, max_latency, latency)
                bars.append(go.Bar(
                    x=[str(batch_size)],
                    y=[seq_len],
                    # base=cumulative_latency,
                    name=f'token_idx [{seq_len - kvcache_bucket}, {seq_len}]',
                    marker=dict(color=color),
                    hoverinfo='y',
                    hovertemplate=f'token_idx from {seq_len - kvcache_bucket} to {seq_len}\n<br>share same latency: {latency} ms',
                    opacity=0.7,
                ))
                cumulative_latency += latency
            data.extend(bars)

        line = go.Scatter(
            x=[str(batch_size) for batch_size in batch_sizes],
            y=batched_throughput,
            mode='lines+markers',
            name='Throughput',
            yaxis='y2',
            marker=dict(color='blue'),
            hovertemplate='Batch Size: %{x}<br>Throughput: %{y}',
        )

        data.append(line)

        layout = go.Layout(
            title=f'{model_name}_{device}{type}_{dtype}_pp{pp}_tp{tp}_in{input}_out{output}_{kvcache_bucket}_steps_projection',
            xaxis=dict(title='Batch Size', type='category'),
            yaxis=dict(title='Sequence Length'),
            yaxis2=dict(title='Throughput', overlaying='y',
                        side='right', showgrid=False),
            barmode='stack',
        )

        fig = go.Figure(data=data, layout=layout)

        model_dir = f"./data/{model_name}"
        create_data_dir(model_dir)
        # image_path = f"{model_dir}/{device}{type}_{dtype}_pp{pp}_tp{tp}_steps_projection.svg"
        # pio.write_image(fig, image_path)
        html_path = f"{model_dir}/{device}{type}_{dtype}_pp{pp}_tp{tp}_in{input}_out{output}_steps_projection.html"
        pio.write_html(fig, file=html_path)

    plot_helper()


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


def extract_overall_projection(proj_dict, device, type_, pp, tp, dtype, input_length, output_length, kvcache_bucket, batch_sizes):
    proj_item = ["Device", "PP", "TP", "DType", "Input", "Output", "BS", "KVCacheBucket", "Prefill(ms)",
                 "DecodeMin(ms)", "DecodeMax(ms)", "DecodeAvg(ms)", "Latency(ms)", "Throughput(tokens/sec)"]

    device_proj = proj_dict[device]
    type_proj = device_proj[type_]
    pp_proj = type_proj[pp]
    tp_proj = pp_proj[tp]
    dtype_proj = tp_proj[dtype]
    input_proj = dtype_proj[input_length]
    output_proj = input_proj[output_length]
    batch_seqlens = []
    batch_latencies = []
    batch_throughputs = []
    overall_projection_table = [proj_item]

    for bs, proj_rst in output_proj:
        compute = proj_rst["compute"]
        mem_consumed = proj_rst["memory"]["size"]
        proj_prefill_step = compute["prefill"]
        proj_decode_steps = compute["decode"]
        prefill_latency = round(proj_prefill_step[0] * MilliSecs, 2)
        step_seqlens = [input_length]
        # step_seqlens = [input_length + i for i in range(1, output_length+1)]
        decode_latency_list_full_steps = []
        decode_latency_list_bucket_only = []
        for k, v in proj_decode_steps.items():
            decode_latency_list_full_steps.extend(l[0] for l in v)
            step_seqlens.append(k)
            decode_latency_list_bucket_only.append(v[0][0])
        # step_latencies = [prefill_latency] + [round(i * MilliSecs, 2) for i in decode_latency_list_full_steps]
        step_latencies = [prefill_latency] + \
            [round(i * MilliSecs, 2) for i in decode_latency_list_bucket_only]

        assert len(step_seqlens) == len(step_latencies), \
            f"steps are not matched! seq_lens: {len(step_seqlens)} batch_latencies: {len(step_latencies)} please check!"
        # batch_latencies.append(steps_latency_bucket_only)
        batch_seqlens.append(step_seqlens)
        batch_latencies.append(step_latencies)

        decode_latency_min = round(
            decode_latency_list_full_steps[0] * MilliSecs, 2)
        decode_latency_max = round(
            decode_latency_list_full_steps[-1] * MilliSecs, 2)
        decode_latency_avg = round(
            sum(decode_latency_list_full_steps)/len(decode_latency_list_full_steps) * MilliSecs, 2)

        overall_latency = round(((proj_prefill_step[0] + sum(decode_latency_list_full_steps)) /
                                (len(decode_latency_list_full_steps) + 1)) * MilliSecs, 2)
        throughput = round((1 / overall_latency) * bs * MilliSecs, 2)
        batch_throughputs.append(throughput)
        overall_projection_table.append([f"{device}{type}", pp, tp, dtype, input_length, output_length,
                                         bs, kvcache_bucket, prefill_latency, decode_latency_min,
                                         decode_latency_max, decode_latency_avg, overall_latency,
                                         throughput if mem_consumed != "OOM" else "OOM"])

    overall_projection_dict = {
        "batch_sizes": batch_sizes,
        'batch_seqlens': batch_seqlens,
        'batch_latencies': batch_latencies,
        'batch_throughputs': batch_throughputs
    }

    return overall_projection_dict, overall_projection_table


def extract_layer_projection(proj_dict, device, type_, pp, tp, dtype, input_length, output_length, kvcache_bucket, batch_sizes):
    device_proj = proj_dict[device]
    type_proj = device_proj[type_]
    pp_proj = type_proj[pp]
    tp_proj = pp_proj[tp]
    dtype_proj = tp_proj[dtype]
    input_proj = dtype_proj[input_length]
    output_proj = input_proj[output_length]
    batch_layer_projection = []

    for _, proj_rst in output_proj:
        compute = proj_rst["compute"]
        proj_prefill_step = compute["prefill"]
        proj_decode_steps = compute["decode"]
        layer_projection = {}

        # prefill
        layer_proj_prefill = proj_prefill_step[1]
        qkvo_proj = layer_proj_prefill["qkvo"]
        qk_matmul, softmax, sv_matmul = layer_proj_prefill["attn"]
        up_proj, down_proj, gate_proj = layer_proj_prefill["ffn"]
        layer_projection["prefill"] = {
            "qkvo_proj": (qkvo_proj["math_ai"], qkvo_proj["tops_roofline"] / TFLOPS),
            "qk_matmul": (qk_matmul["math_ai"], qk_matmul["tops_roofline"] / TFLOPS),
            "softmax": (softmax["math_ai"], softmax["tops_roofline"] / TFLOPS),
            "sv_matmul": (sv_matmul["math_ai"], sv_matmul["tops_roofline"] / TFLOPS),
            "up_proj": (up_proj["math_ai"], up_proj["tops_roofline"] / TFLOPS),
            "down_proj": (down_proj["math_ai"], down_proj["tops_roofline"] / TFLOPS),
            "gate_proj": (gate_proj["math_ai"], gate_proj["tops_roofline"] / TFLOPS) if gate_proj is not None else None,
        }

        # decoding
        _, last_step = list(proj_decode_steps.items())[-1]
        layer_proj_decode = last_step[0][1]
        qkvo_proj = layer_proj_decode["qkvo"]
        qk_matmul, softmax, sv_matmul = layer_proj_decode["attn"]
        up_proj, down_proj, gate_proj = layer_proj_decode["ffn"]
        layer_projection["decode"] = {
            "qkvo_proj": (qkvo_proj["math_ai"], qkvo_proj["tops_roofline"] / TFLOPS),
            "qk_matmul": (qk_matmul["math_ai"], qk_matmul["tops_roofline"] / TFLOPS),
            "softmax": (softmax["math_ai"], softmax["tops_roofline"] / TFLOPS),
            "sv_matmul": (sv_matmul["math_ai"], sv_matmul["tops_roofline"] / TFLOPS),
            "up_proj": (up_proj["math_ai"], up_proj["tops_roofline"] / TFLOPS),
            "down_proj": (down_proj["math_ai"], down_proj["tops_roofline"] / TFLOPS),
            "gate_proj": (gate_proj["math_ai"], gate_proj["tops_roofline"] / TFLOPS) if gate_proj is not None else None,
        }

        batch_layer_projection.append(layer_projection)

    layer_projection_dict = {
        "batch_sizes": batch_sizes,
        'batch_layer_projection': batch_layer_projection
    }

    return layer_projection_dict


def print_overall_projection(model_name, proj_dict, kvcache_bucket, batchsize_list, to_csv=True, plot=True):
    proj_item = ["Model", "Device", "PP", "TP", "DType", "Input", "Output", "BS", "KVCacheBucket", "Prefill(ms)",
                 "DecodeMin(ms)", "DecodeMax(ms)", "DecodeAvg(ms)", "Latency(ms)", "Throughput(tokens/sec)"]

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
                                # [
                                #     input + i for i in range(1, output+1)]
                                batch_steps_latency = []
                                batched_throughput = []
                                for bs, proj_rst in output_proj:
                                    compute = proj_rst["compute"]
                                    mem_consumed = proj_rst["memory"]["size"]
                                    proj_prefill_step = compute["prefill"]
                                    proj_decode_steps = compute["decode"]
                                    prefill_latency = round(
                                        proj_prefill_step[0] * MilliSecs, 2)
                                    seq_lens_list = [input]
                                    decode_latency_list_full_steps = []
                                    decode_latency_list_bucket_only = []
                                    for k, v in proj_decode_steps.items():
                                        decode_latency_list_full_steps.extend(
                                            l[0] for l in v)
                                        seq_lens_list.append(k)
                                        decode_latency_list_bucket_only.append(
                                            v[0][0])
                                        # decode_latency_list_full_steps = [step[0]
                                        #                     for step in proj_decode_steps]
                                    steps_latency_bucket_only = [
                                        prefill_latency] + [round(i * MilliSecs, 2) for i in decode_latency_list_bucket_only]
                                    assert len(seq_lens_list) == len(steps_latency_bucket_only), \
                                        f"steps are not matched! seq_lens: {len(seq_lens_list)} steps_latency_bucket_only: {len(steps_latency_bucket_only)} please check!"
                                    batch_steps_latency.append(
                                        steps_latency_bucket_only)
                                    decode_latency_min = round(
                                        decode_latency_list_full_steps[0] * MilliSecs, 2)
                                    decode_latency_max = round(
                                        decode_latency_list_full_steps[-1] * MilliSecs, 2)
                                    decode_latency_avg = round(
                                        sum(decode_latency_list_full_steps)/len(decode_latency_list_full_steps) * MilliSecs, 2)
                                    overall_latency = round(((proj_prefill_step[0] + sum(decode_latency_list_full_steps)) /
                                                            (len(decode_latency_list_full_steps) + 1)) * MilliSecs, 2)
                                    throughput = round(
                                        (1 / overall_latency) * bs * MilliSecs, 2)
                                    batched_throughput.append(throughput)
                                    proj_data.append([model_name, f"{device}{type}", pp, tp, dtype, input, output,
                                                      bs, kvcache_bucket, prefill_latency, decode_latency_min,
                                                      decode_latency_max, decode_latency_avg, overall_latency,
                                                      throughput if mem_consumed != "OOM" else "OOM"])
                                    figure_data[dtype][input][output].append([model_name, f"{device}{type}", pp, tp, dtype, input,
                                                                              output, bs, kvcache_bucket, prefill_latency,
                                                                              decode_latency_min, decode_latency_max,
                                                                              decode_latency_avg, overall_latency, throughput])
                                proj_data.append([""] * len(proj_item))
                                # plot_steps_projection(model_name, device, type, dtype, pp, tp, input, output, kvcache_bucket,
                                #                       batchsize_list, seq_lens_list, batch_steps_latency, batched_throughput)

                    if plot:
                        model_dir = f"./data/{model_name}"
                        create_data_dir(model_dir)
                        figure_name = f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_overall_projection.png"
                        plot_overall_projection(
                            model_name, device, type, pp, tp, figure_name, figure_data, batchsize_list)

            print(f"{model_name}_{device}{type}_overall_projection".center(150))
            print(tabulate(proj_data))

            if to_csv:
                model_dir = f"./data/model/{model_name}"
                create_data_dir(model_dir)
                with open(f"{model_dir}/{device}{type}_overall_projection.csv", "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(proj_data)


def print_overall_projection_in_detail(model_name, proj_dict, kvcache_bucket, to_csv=True):
    proj_item = ["DType", "Input", "Output", "BS", "HiddenSize", "HeadsQ", "HeadsKV", "InterSize",
                 "WithGate", "Experts", "Layers", "KVCacheBucket", "Prefill(ms)", "DecodeMin(ms)",
                 "DecodeMax(ms)", "DecodeAvg(ms)", "Latency(ms)", "Throughput(tokens/sec)"]

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
                                    compute = proj_rst["compute"]
                                    mem_consumed = proj_rst["memory"]["size"]
                                    proj_prefill_step = compute["prefill"]
                                    proj_decode_steps = compute["decode"]
                                    prefill_latency = round(
                                        proj_prefill_step[0] * MilliSecs, 2)
                                    decode_latency_list_full_steps = []
                                    for _, v in proj_decode_steps.items():
                                        # decode_latency_list_full_steps.extend(v)
                                        decode_latency_list_full_steps.extend(
                                            l[0] for l in v)
                                        # decode_latency_list_full_steps = [step[0]
                                        #                     for step in proj_decode_steps]
                                    decode_latency_min = round(
                                        decode_latency_list_full_steps[0] * MilliSecs, 2)
                                    decode_latency_max = round(
                                        decode_latency_list_full_steps[-1] * MilliSecs, 2)
                                    decode_latency_avg = round(
                                        sum(decode_latency_list_full_steps)/len(decode_latency_list_full_steps) * MilliSecs, 2)
                                    overall_latency = round(((proj_prefill_step[0] + sum(decode_latency_list_full_steps)) /
                                                            (len(decode_latency_list_full_steps) + 1)) * MilliSecs, 2)
                                    throughput = round(
                                        (1 / overall_latency) * bs * MilliSecs, 2)
                                    proj_data.append([dtype, input, output, bs, hidden_size, num_heads_q, num_heads_kv,
                                                      intermediate_size, mlp_with_gate, num_experts, num_layers, kvcache_bucket,
                                                      prefill_latency, decode_latency_min, decode_latency_max, decode_latency_avg,
                                                      overall_latency, throughput if mem_consumed != "OOM" else "OOM"])
                                proj_data.append([""] * len(proj_item))
                    '''
                    print(
                        f"{model_name}_{device}{type}_pp{pp}_tp{tp}_overall_projection_in_detatil".center(200))
                    print(tabulate(proj_data))
                    '''

                    if to_csv:
                        model_dir = f"./data/model/{model_name}"
                        create_data_dir(model_dir)
                        with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_overall_projection_in_detatil.csv", "w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerows(proj_data)


def print_layer_projection(model_name, proj_dict, to_csv=True):
    bmm_layer_proj_item = ["Device", "DType", "BS", "HeadsQ", "HeadsKV", "SeqLenQ", "SeqLenKV", "HeadDim", "Ops(M)",
                           "Size(MB)", "TFLOPs", "BW(TB/s)", "Memory(us)", "Compute(us)", "ProjectLatency(us)", "Bound"]
    mm_layer_proj_item = ["Device", "DType", "BS", "SeqLenQ", "SeqLenKV", "HiddenSize", "IntermediateSize", "Ops(M)",
                          "Size(MB)", "TFLOPs", "BW(TB/s)", "Memory(us)", "Compute(us)", "ProjectLatency(us)", "Bound"]

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
                        qk_layer_proj_prefill_list = [bmm_layer_proj_item]
                        sv_layer_proj_prefill_list = [bmm_layer_proj_item]
                        up_layer_proj_prefill_list = [mm_layer_proj_item]
                        qk_layer_proj_decode_list = [bmm_layer_proj_item]
                        sv_layer_proj_decode_list = [bmm_layer_proj_item]
                        up_layer_proj_decode_list = [mm_layer_proj_item]
                        for input, input_proj in dtype_proj.items():
                            for output, output_proj in input_proj.items():
                                for bs, proj_rst in output_proj:
                                    compute = proj_rst["compute"]
                                    proj_prefill_step = compute["prefill"]
                                    proj_decode_steps = compute["decode"]

                                    # prefill
                                    layer_proj_prefill = proj_prefill_step[1]
                                    tq = layer_proj_prefill["tq"]
                                    tkv = layer_proj_prefill["tkv"]
                                    # attn_qk(bmm)
                                    qk, softmax, sv = layer_proj_prefill["attn"]
                                    qk_ops = round(
                                        qk["operations"] / MegaParam, 2)
                                    qk_size = round(qk["size"] / MegaBytes, 2)
                                    qk_mem_time = round(
                                        qk["mem_time"] * MicroSecs, 2)
                                    qk_cmp_time = round(
                                        qk["cmp_time"] * MicroSecs, 2)
                                    qk_latency = round(
                                        qk["latency"] * MicroSecs, 2)
                                    qk_tops = round(
                                        qk["tops_roofline"] / TFLOPS, 2)
                                    qk_bw = round(qk["bw"] / T_BW, 2)
                                    qk_bound = qk["bound"]
                                    qk_layer_proj_prefill_list.append([f"{device}{type}", dtype, bs,
                                                                       num_heads_q, num_heads_kv, tq,
                                                                       tkv, headdim, qk_ops, qk_size,
                                                                       qk_tops, qk_bw, qk_mem_time,
                                                                       qk_cmp_time, qk_latency,
                                                                       qk_bound])
                                    # attn_sv(bmm)
                                    sv_ops = round(
                                        sv["operations"] / MegaParam, 2)
                                    sv_size = round(sv["size"] / MegaBytes, 2)
                                    sv_mem_time = round(
                                        sv["mem_time"] * MicroSecs, 2)
                                    sv_cmp_time = round(
                                        sv["cmp_time"] * MicroSecs, 2)
                                    sv_latency = round(
                                        sv["latency"] * MicroSecs, 2)
                                    sv_tops = round(
                                        sv["tops_roofline"] / TFLOPS, 2)
                                    sv_bw = round(sv["bw"] / T_BW, 2)
                                    sv_bound = sv["bound"]
                                    sv_layer_proj_prefill_list.append([f"{device}{type}", dtype, bs,
                                                                       num_heads_q, num_heads_kv, tq,
                                                                       tkv, headdim, sv_ops, sv_size,
                                                                       sv_tops, sv_bw, sv_mem_time,
                                                                       sv_cmp_time, sv_latency,
                                                                       sv_bound])
                                    # ffn_up(mm)
                                    up, down, gate = layer_proj_prefill["ffn"]
                                    up_ops = round(
                                        up["operations"] / MegaParam, 2)
                                    up_size = round(up["size"] / MegaBytes, 2)
                                    up_mem_time = round(
                                        up["mem_time"] * MicroSecs, 2)
                                    up_cmp_time = round(
                                        up["cmp_time"] * MicroSecs, 2)
                                    up_latency = round(
                                        up["latency"] * MicroSecs, 2)
                                    up_tops = round(
                                        up["tops_roofline"] / TFLOPS, 2)
                                    up_bw = round(up["bw"] / T_BW, 2)
                                    up_bound = up["bound"]
                                    up_layer_proj_prefill_list.append([f"{device}{type}", dtype, bs, tq, tkv,
                                                                       hidden_size, intermediate_size, up_ops,
                                                                       up_size, up_tops, up_bw, up_mem_time,
                                                                       up_cmp_time, up_latency, up_bound])

                                    # decoding
                                    for _, decode_step in proj_decode_steps.items():
                                        layer_proj_decode = decode_step[0][1]
                                        tq = layer_proj_decode["tq"]
                                        tkv = layer_proj_decode["tkv"]
                                        # attn_qk(bmm)
                                        qk, softmax, sv = layer_proj_decode["attn"]
                                        qk_ops = round(
                                            qk["operations"] / MegaParam, 2)
                                        qk_size = round(
                                            qk["size"] / MegaBytes, 2)
                                        qk_mem_time = round(
                                            qk["mem_time"] * MicroSecs, 2)
                                        qk_cmp_time = round(
                                            qk["cmp_time"] * MicroSecs, 2)
                                        qk_latency = round(
                                            qk["latency"] * MicroSecs, 2)
                                        qk_tops = round(
                                            qk["tops_roofline"] / TFLOPS, 2)
                                        qk_bw = round(qk["bw"] / T_BW, 2)
                                        qk_bound = qk["bound"]
                                        qk_layer_proj_decode_list.append([f"{device}{type}", dtype, bs,
                                                                          num_heads_q, num_heads_kv, tq,
                                                                          tkv, headdim, qk_ops, qk_size,
                                                                          qk_tops, qk_bw, qk_mem_time,
                                                                          qk_cmp_time, qk_latency,
                                                                          qk_bound])
                                        # attn_sv(bmm)
                                        sv_ops = round(
                                            sv["operations"] / MegaParam, 2)
                                        sv_size = round(
                                            sv["size"] / MegaBytes, 2)
                                        sv_mem_time = round(
                                            sv["mem_time"] * MicroSecs, 2)
                                        sv_cmp_time = round(
                                            sv["cmp_time"] * MicroSecs, 2)
                                        sv_latency = round(
                                            sv["latency"] * MicroSecs, 2)
                                        sv_tops = round(
                                            sv["tops_roofline"] / TFLOPS, 2)
                                        sv_bw = round(sv["bw"] / T_BW, 2)
                                        sv_bound = sv["bound"]
                                        sv_layer_proj_decode_list.append([f"{device}{type}", dtype, bs,
                                                                          num_heads_q, num_heads_kv, tq,
                                                                          tkv, headdim, sv_ops, sv_size,
                                                                          sv_tops, sv_bw, sv_mem_time,
                                                                          sv_cmp_time, sv_latency,
                                                                          sv_bound])
                                        # ffn_up(mm)
                                        up, down, gate = layer_proj_decode["ffn"]
                                        up_ops = round(
                                            up["operations"] / MegaParam, 2)
                                        up_size = round(
                                            up["size"] / MegaBytes, 2)
                                        up_mem_time = round(
                                            up["mem_time"] * MicroSecs, 2)
                                        up_cmp_time = round(
                                            up["cmp_time"] * MicroSecs, 2)
                                        up_latency = round(
                                            up["latency"] * MicroSecs, 2)
                                        up_tops = round(
                                            up["tops_roofline"] / TFLOPS, 2)
                                        up_bw = round(up["bw"] / T_BW, 2)
                                        up_bound = up["bound"]
                                        up_layer_proj_decode_list.append([f"{device}{type}", dtype, bs, tq, tkv,
                                                                          hidden_size, intermediate_size, up_ops,
                                                                          up_size, up_tops, up_bw, up_mem_time,
                                                                          up_cmp_time, up_latency, up_bound])
                                        break
                                qk_layer_proj_prefill_list.append(
                                    [""] * len(bmm_layer_proj_item))
                                sv_layer_proj_prefill_list.append(
                                    [""] * len(bmm_layer_proj_item))
                                up_layer_proj_prefill_list.append(
                                    [""] * len(mm_layer_proj_item))
                                qk_layer_proj_decode_list.append(
                                    [""] * len(bmm_layer_proj_item))
                                sv_layer_proj_decode_list.append(
                                    [""] * len(bmm_layer_proj_item))
                                up_layer_proj_decode_list.append(
                                    [""] * len(mm_layer_proj_item))

                        '''
                        print(f"{model_name}_{device}{type}_pp{pp}_tp{tp}_prefill_attn_qk_projection".center(160))
                        print(tabulate(qk_layer_proj_prefill_list))
                        print(f"{model_name}_{device}{type}_pp{pp}_tp{tp}_prefill_attn_sv_projection".center(160))
                        print(tabulate(sv_layer_proj_prefill_list))
                        print(f"{model_name}_{device}{type}_pp{pp}_tp{tp}_prefill_ffn_up_projection".center(160))
                        print(tabulate(up_layer_proj_prefill_list))
                        print(f"{model_name}_{device}{type}_pp{pp}_tp{tp}_decode_attn_qk_projection".center(160))
                        print(tabulate(qk_layer_proj_decode_list))
                        print(f"{model_name}_{device}{type}_pp{pp}_tp{tp}_decode_attn_sv_projection".center(160))
                        print(tabulate(sv_layer_proj_decode_list))
                        print(f"{model_name}_{device}{type}_pp{pp}_tp{tp}_decode_ffn_up_projection".center(160))
                        print(tabulate(up_layer_proj_decode_list))
                        '''

                        if to_csv:
                            model_dir = f"./data/model/{model_name}"
                            create_data_dir(model_dir)
                            with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_{dtype}_prefill_attn_qk_projection.csv",
                                      "w", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(qk_layer_proj_prefill_list)
                            with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_{dtype}_prefill_attn_sv_projection.csv",
                                      "w", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(sv_layer_proj_prefill_list)
                            with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_{dtype}_prefill_ffn_up_projection.csv",
                                      "w", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(up_layer_proj_prefill_list)
                            with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_{dtype}_decode_attn_qk_projection.csv",
                                      "w", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(qk_layer_proj_decode_list)
                            with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_{dtype}_decode_attn_sv_projection.csv",
                                      "w", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(sv_layer_proj_decode_list)
                            with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_{dtype}_decode_ffn_up_projection.csv",
                                      "w", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(up_layer_proj_decode_list)


def print_layer_analysis(model_name, proj_dict, to_csv=True):
    analysis_item = ["Model", "Device", "PP", "TP", "DType", "BS", "Input", "Output", "Operation", "SeqLenQ",
                     "SeqLenKV", "NumOps(G)", "Memory(GB)", "TopsRF(TFlops)", "AI", "Bound"]

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
                                    compute = proj_rst["compute"]
                                    proj_prefill_step = compute["prefill"]
                                    proj_decode_steps = compute["decode"]

                                    # prefill
                                    layer_proj_prefill = proj_prefill_step[1]
                                    tq = layer_proj_prefill["tq"]
                                    tkv = layer_proj_prefill["tkv"]
                                    # qkvo / qk, softmax, sv / up, down, gate
                                    for mod_name in ["qkvo", "attn", "ffn"]:
                                        module = layer_proj_prefill[mod_name]
                                        if isinstance(module, dict):
                                            op = module
                                            name = op["name"]
                                            num_ops = round(
                                                op["operations"] / GigaParam, 2)
                                            size = round(
                                                op["size"] / GigaBytes, 2)
                                            tops = round(
                                                op["tops_roofline"] / TFLOPS, 2)
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
                                                        op["operations"] / GigaParam, 2)
                                                    size = round(
                                                        op["size"] / GigaBytes, 2)
                                                    tops = round(
                                                        op["tops_roofline"] / TFLOPS, 2)
                                                    math_ai = round(
                                                        op["math_ai"], 2)
                                                    bound = op["bound"]
                                                    layer_analysis_prefill.append([model_name, f"{device}{type}",
                                                                                   pp, tp, dtype, bs, input, output,
                                                                                   name, tq, tkv, num_ops, size, tops,
                                                                                   math_ai, bound])

                                    # decoding
                                    _, last_step = list(
                                        proj_decode_steps.items())[-1]
                                    layer_proj_decode = last_step[0][1]
                                    tq = layer_proj_decode["tq"]
                                    tkv = layer_proj_prefill["tkv"]
                                    # qkvo / qk, softmax, sv / up, down, gate
                                    for mod_name in ["qkvo", "attn", "ffn"]:
                                        module = layer_proj_prefill[mod_name]
                                        if isinstance(module, dict):
                                            op = module
                                            name = op["name"]
                                            num_ops = round(
                                                op["operations"] / GigaParam, 2)
                                            size = round(
                                                op["size"] / GigaBytes, 2)
                                            tops = round(
                                                op["tops_roofline"] / TFLOPS, 2)
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
                                                        op["operations"] / GigaParam, 2)
                                                    size = round(
                                                        op["size"] / GigaBytes, 2)
                                                    tops = round(
                                                        op["tops_roofline"] / TFLOPS, 2)
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

                        # '''
                        print(
                            f"{model_name}_{device}{type}_pp{pp}_tp{tp}_prefill_layer_analysis".center(150))
                        print(tabulate(layer_analysis_prefill))
                        print(
                            f"{model_name}_{device}{type}_pp{pp}_tp{tp}_decode_layer_analysis".center(150))
                        print(tabulate(layer_analysis_decode))
                        # '''

                        if to_csv:
                            model_dir = f"./data/model/{model_name}"
                            create_data_dir(model_dir)
                            with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_{dtype}_prefill_layer_analysis.csv",
                                      "w", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(layer_analysis_prefill)
                            with open(f"{model_dir}/{device}{type}_pp{pp}_tp{tp}_{dtype}_decode_layer_analysis.csv",
                                      "w", newline="") as csvfile:
                                writer = csv.writer(csvfile)
                                writer.writerows(layer_analysis_decode)


def print_projection(model_name, proj_dict, kvcache_bucket, batchsize_list, to_csv=True, plot=True):
    print_overall_projection(model_name, proj_dict,
                             kvcache_bucket, batchsize_list, to_csv, plot)
    print_overall_projection_in_detail(
        model_name, proj_dict, kvcache_bucket, to_csv)
    print_layer_projection(model_name, proj_dict, to_csv)
    print_layer_analysis(model_name, proj_dict, to_csv)


def print_mem_analysis(memory_dict, batchsize_list):
    for pp, pp_dict in memory_dict.items():
        for tp, tp_dict in pp_dict.items():
            for dtype, mem_data in tp_dict.items():
                print(tabulate(mem_data))
            print("\n")


def print_projected_mem_per_device(model_name, memory_dict, batchsize_list, context_list):
    proj_dict = {}
    for pp, pp_dict in memory_dict.items():
        proj_dict[pp] = {}
        for tp, tp_dict in pp_dict.items():
            proj_dict[pp][tp] = {}
            for dtype, mem_data in tp_dict.items():
                print(
                    f"Memory projection of [{model_name}] in precision [{dtype}] with PP=[{pp}] TP=[{tp}]\n")
                proj_dict[pp][tp][dtype] = []
                for data in mem_data[1:]:
                    print(data)
                    proj_dict[pp][tp][dtype].append(
                        [data[5], data[6], data[8], data[-2]])
                    # proj_dict[pp][tp][dtype][data[8]].append(data[-2])
                print(proj_dict[pp][tp][dtype])

                # print(tabulate(mem_data))
            print("\n")
    return proj_dict


def print_matmul_projection(op_name, proj_dict, to_csv=True):
    proj_item = ["Operation", "Device", "DType", "M",
                 "N", "K", "AI", "Latency(us)", "TFLOPs(1e12)"]

    for device, device_proj in proj_dict.items():
        for type, type_proj in device_proj.items():
            proj_data = [proj_item]
            for dtype, dtype_proj in type_proj.items():
                for proj_rst in dtype_proj:
                    m, n, k = proj_rst["m"], proj_rst["n"], proj_rst["k"]
                    latency = round(
                        proj_rst["latency"] * MicroSecs, 2)
                    math_ai = round(proj_rst["math_ai"], 2)

                    throughput = round(proj_rst["tops_roofline"] / TFLOPS, 2)
                    proj_data.append(
                        [op_name, f"{device}{type}", dtype, m,
                            n, k, math_ai, latency, throughput]
                    )
                proj_data.append([""] * len(proj_item))

            print(f"{op_name}_{device}{type}_projection".center(80))
            print(tabulate(proj_data))

            if to_csv:
                op_dir = f"./data/operation/{op_name}"
                create_data_dir(op_dir)
                with open(f"{op_dir}/{device}{type}_projection.csv", "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(proj_data)


def print_flashattn_projection(op_name, proj_dict, to_csv=True):
    proj_item = ["Operation", "Device", "DType", "B", "M_q", "M_kv", "T_q",
                 "T_kv", "D", "qk(us)", "sv(us)", "softmax(us)", "total(us)", "TFLOPs(1e12)"]

    for device, device_proj in proj_dict.items():
        for type, type_proj in device_proj.items():
            proj_data = [proj_item]
            for dtype, dtype_proj in type_proj.items():
                for proj_rst in dtype_proj:
                    attn_type = proj_rst["type"]
                    B, M_q, T_q, D = proj_rst["B"], proj_rst["M_q"], proj_rst["T_q"], proj_rst["D"]
                    M_kv, T_kv = proj_rst["M_kv"], proj_rst["T_kv"]
                    rt_qk, rt_sv, rt_softmax = \
                        proj_rst["rt_qk"][-1], proj_rst["rt_sv"][-1], proj_rst["rt_softmax"][-1]
                    runtime = rt_qk + rt_sv + rt_softmax
                    latency = round(runtime * MicroSecs, 2)
                    rt_qk, rt_sv, rt_softmax = \
                        round(rt_qk * MicroSecs, 2), round(rt_sv *
                                                           MicroSecs, 2), round(rt_softmax * MicroSecs, 2)

                    throughput = round(proj_rst["tops_roofline"] / TFLOPS, 2)
                    proj_data.append([attn_type, f"{device}{type}", dtype, B, M_q, M_kv, T_q,
                                      T_kv, D, rt_qk, rt_sv, rt_softmax, latency, throughput])
                proj_data.append([""] * len(proj_item))

            print(f"{op_name}_{device}{type}_projection".center(120))
            print(tabulate(proj_data))

            if to_csv:
                op_dir = f"./data/operation/{op_name}"
                create_data_dir(op_dir)
                with open(f"{op_dir}/{device}{type}_projection.csv", "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(proj_data)


def print_pagedattn_projection(op_name, proj_dict, to_csv=True):
    pass
