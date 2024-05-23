from tabulate import tabulate
import matplotlib.pyplot as plt
from base import *


mem_item_basic = ["H", "H_Inter", "Heads_Q", "HeadsKV", "Experts", "Layers(Mlp)",
                  "Layers(MoE)", "QKVO", "Up(W1)", "Gate(W3)", "Down(W2)", "1Expert",
                  "1Layer(Mlp)", "1Layer(MoE)", "ParamsTotal", "Dtype", "Weights"]
mem_item_list = ["Device", "PP", "TP", "NumDevices", "Layers", "In", "Out", "DType",
                 "BS", "Weights(GB)", "KVcache(GB)", "Activat(GB)", "Total(GB)", "Fit2Device"]


def mem_persistent_weights(model_config):
    def mem_qkvo_proj():
        # persistent memory
        params_in_weight = model_config.hidden_size * model_config.hidden_size
        params_total = params_in_weight
        params_total *= 4  # 4 for qkvo
        params_total /= model_config.num_devices  # assume only TP/PP
        bytes_total = params_total * model_config.num_bytes

        mem_rst = {
            "name": "qkvo_proj",
            "param": params_total,
            "#mem": bytes_total,
            "attr": "persistent",
            "item": {}
        }

        return mem_rst

    def mem_mlp_gate_or_w3():
        # persistent memory
        params_in_weight = model_config.hidden_size * model_config.intermediate_size
        params_total = params_in_weight
        params_total /= model_config.num_devices  # assume only TP/PP
        bytes_total = params_total * model_config.num_bytes

        mem_rst = {
            "name": "mlp_gate(w3)",
            "param": params_total,
            "#mem": bytes_total,
            "attr": "persistent",
            "item": {}
        }

        return mem_rst

    def mem_mlp_up_or_w1():
        # persistent memory
        params_in_weight = model_config.hidden_size * model_config.intermediate_size
        params_total = params_in_weight
        params_total /= model_config.num_devices  # assume only TP/PP
        bytes_total = params_total * model_config.num_bytes

        mem_rst = {
            "name": "mlp_up(w1)",
            "param": params_total,
            "#mem": bytes_total,
            "attr": "persistent",
            "item": {}
        }

        return mem_rst

    def mem_mlp_down_or_w2():
        # persistent memory
        params_in_weight = model_config.intermediate_size * model_config.hidden_size
        params_total = params_in_weight
        params_total /= model_config.num_devices  # assume only TP/PP
        bytes_total = params_total * model_config.num_bytes

        mem_rst = {
            "name": "mlp_down(w2)",
            "param": params_total,
            "#mem": bytes_total,
            "attr": "persistent",
            "item": {}
        }

        return mem_rst

    def mem_mlp_persist():
        up = mem_mlp_up_or_w1()
        if model_config.mlp_with_gate:
            gate = mem_mlp_gate_or_w3()
        down = mem_mlp_down_or_w2()

        params_total = up["param"] + down["param"]
        mem_mlp = up["#mem"] + down["#mem"]
        if model_config.mlp_with_gate:
            params_total += gate["param"]
            mem_mlp += gate["#mem"]

        item_dict = {
            "mem_up(w1)": up,
            "mem_down(w2)": down,
            "mem_gate(w3)": gate if model_config.mlp_with_gate else None
        }

        mem_rst = {
            "name": "mlp",
            "param": params_total,
            "#mem": mem_mlp,
            "attr": "persistent",
            "item": item_dict
        }

        return mem_rst

    def mem_moe_persist():
        mem_mlp = mem_mlp_persist()
        params_total = mem_mlp["param"] * model_config.num_experts
        mem_moe = mem_mlp["#mem"] * model_config.num_experts

        mem_rst = {
            "name": "moe",
            "param": params_total,
            "#mem": mem_moe,
            "attr": "persistent",
            "item": {"mem_mlp": mem_mlp}
        }

        return mem_rst

    def mem_single_layer_mlp_persist():
        params_total = 0
        mem_single_layer = 0
        ffn_name = None
        item_dict = None
        if model_config.num_layers_mlp != 0:
            mem_qkvo = mem_qkvo_proj()
            mem_ffn = mem_mlp_persist()
            params_total = mem_qkvo["param"] + mem_ffn["param"]
            mem_single_layer = mem_qkvo["#mem"] + mem_ffn["#mem"]
            ffn_name = mem_ffn["name"]

            item_dict = {
                "mem_qkvo": mem_qkvo,
                "mem_ffn": mem_ffn
            }

        mem_rst = {
            "name": f"single_layer_{ffn_name}",
            "param": params_total,
            "#mem": mem_single_layer,
            "attr": "persistent",
            "item": item_dict
        }

        return mem_rst

    def mem_single_layer_moe_persist():
        params_total = 0
        mem_single_layer = 0
        ffn_name = None
        item_dict = None
        if model_config.num_layers_moe != 0:
            mem_qkvo = mem_qkvo_proj()
            mem_ffn = mem_moe_persist()
            params_total = mem_qkvo["param"] + mem_ffn["param"]
            mem_single_layer = mem_qkvo["#mem"] + mem_ffn["#mem"]
            ffn_name = mem_ffn["name"]

            item_dict = {
                "mem_qkvo": mem_qkvo,
                "mem_ffn": mem_ffn
            }

        mem_rst = {
            "name": f"single_layer_{ffn_name}",
            "param": params_total,
            "#mem": mem_single_layer,
            "attr": "persistent",
            "item": item_dict
        }

        return mem_rst

    mem_single_layer_mlp = mem_single_layer_mlp_persist()
    mem_single_layer_moe = mem_single_layer_moe_persist()
    params_total = mem_single_layer_mlp["param"] * model_config.num_layers_mlp + \
        mem_single_layer_moe["param"] * model_config.num_layers_moe
    mem_persist = mem_single_layer_mlp["#mem"] * model_config.num_layers_mlp + \
        mem_single_layer_moe["#mem"] * model_config.num_layers_moe

    item_dict = {
        "single_layer_mlp": mem_single_layer_mlp,
        "single_layer_moe": mem_single_layer_moe
    }

    mem_rst = {
        "name": "mem_persist_weight",
        "param": params_total,
        "#mem": mem_persist,
        "attr": "persistent",
        "item": item_dict
    }

    return mem_rst


def mem_persistent_kvcache(model_config):
    assert model_config.num_heads_kv % model_config.num_devices == 0, \
        f"kv heads {model_config.num_heads_kv} need to be divisible by number of devices {model_config.num_devices}"
    head_dim = model_config.hidden_size // model_config.num_heads_q
    # persistent memory
    elements_kv = model_config.batch_size * model_config.num_heads_kv * \
        model_config.seq_len_kv * head_dim * 2
    params_total = elements_kv * 2  # 2 for kv
    params_total /= model_config.num_devices  # assume only TP/PP
    bytes_total = params_total * model_config.num_bytes

    mem_rst = {
        "name": "mem_persist_kvcache",
        "param": params_total,
        "#mem": bytes_total,
        "attr": "persistent",
        "item": {}
    }

    return mem_rst


def mem_activation(model_config):
    '''
    def mem_attn_qk():
        assert model_config.num_heads_q % model_config.num_devices == 0, \
            f"q heads {model_config.num_heads_q} need to be divisible by tp size {model_config.tp}"
        # activation memory
        params_out = model_config.batch_size * model_config.num_heads_q * \
            model_config.seq_len_q * model_config.seq_len_kv
        params_total = params_out
        bytes_total = params_total * model_config.num_bytes
        bytes_total /= model_config.num_devices # assume only TP/PP

        mem_rst = {
            "name": "q@k_T",
            "param": params_total,
            "#mem": bytes_total,
            "attr": "activation",
            "item": {}
        }

        return mem_rst
    '''

    def mem_attn_softmax_standard():
        assert model_config.num_heads_q % model_config.num_devices == 0, \
            f"q heads {model_config.num_heads_q} need to be divisible by tp size {model_config.tp}"
        # activation memory
        # prefill
        params_in = model_config.batch_size * model_config.num_heads_q * \
            model_config.seq_len_q * model_config.seq_len_q
        params_out = model_config.batch_size * model_config.num_heads_q * \
            model_config.seq_len_q * model_config.seq_len_q
        params_total = params_in + params_out
        params_total /= model_config.tp  # only TP

        bytes_total = params_total * model_config.num_bytes

        mem_rst = {
            "name": "softmax",
            "param": params_total,
            "#mem": bytes_total,
            "attr": "activation",
            "item": {}
        }

        return mem_rst

    '''
    def mem_attn_scorev():
        assert model_config.num_heads_q % model_config.num_devices == 0, \
            f"q heads {model_config.num_heads_q} need to be divisible by tp size {model_config.tp}"
        # activation memory, same with mem_attn_qk
        params_in_score = model_config.batch_size * model_config.num_heads_q * \
            model_config.seq_len_q * model_config.seq_len_kv
        params_total = params_in_score
        bytes_total = params_total * model_config.num_bytes
        bytes_total /= model_config.num_devices # assume only TP/PP

        mem_rst = {
            "name": "score@v",
            "param": params_total,
            "#mem": bytes_total,
            "attr": "activation",
            "item": {}
        }

        return mem_rst
    '''

    def mem_attn_activation():
        attn = mem_attn_softmax_standard()
        mem_attn = attn["#mem"]

        mem_rst = {
            "name": "attn",
            "param": attn["param"],
            "#mem": mem_attn,
            "attr": "activation",
            "item": {}
        }

        return mem_rst

    def mem_mlp_activation():
        params_total = 0
        # Todo
        mem_rst = {
            "name": "mlp",
            "param": params_total,
            "#mem": 0,
            "attr": "activation",
            "item": {}
        }

        return mem_rst

    attn_activat = mem_attn_activation()
    mlp_activat = mem_mlp_activation()

    params_total = max(attn_activat["param"], mlp_activat["param"])
    mem_activat = max(attn_activat["#mem"], mlp_activat["#mem"])

    mem_rst = {
        "name": "mem_activation",
        "param": params_total,
        "#mem": mem_activat,
        "attr": "activation",
        "item": {"mem_act_attn": attn_activat, "mem_act_mlp": mlp_activat}
    }
    return mem_rst


def mem_decoder(model_config):
    mem_persist_weights = mem_persistent_weights(model_config)
    mem_persist_kvcache = mem_persistent_kvcache(model_config)
    mem_activat = mem_activation(model_config)
    params_total = mem_persist_weights["param"] + \
        mem_persist_kvcache["param"] + mem_activat["param"]
    mem_total = mem_persist_weights["#mem"] + mem_activat["#mem"]

    item_dict = {
        "mem_persist_weights": mem_persist_weights,
        "mem_persist_kvcache": mem_persist_kvcache,
        "mem_activat": mem_activat
    }

    mem_rst = {
        "name": "mem_persistent",
        "param": params_total,
        "#mem": mem_total,
        "attr": "persistent",
        "item": item_dict
    }

    # mem_item_list = ["Device", "PP", "TP", "NumDevices", "Layers", "In", "Out", "DType",
    #              "BS", "Weights(GB)", "KVcache(GB)", "Activat(GB)", "Total(GB)", "Fit2Device"]
    device_gb = model_config.device_mem
    gigabytes = 1024 * 1024 * 1024
    weights_gb = mem_persist_weights["#mem"] / gigabytes
    kvcache_gb = mem_persist_kvcache["#mem"] / gigabytes
    activat_gb = mem_activat["#mem"] / gigabytes
    total_gb = mem_total / gigabytes

    mem_data = [model_config.device, model_config.pp, model_config.tp, model_config.num_devices,
                model_config.num_layers, model_config.seq_len_q, model_config.seq_len_kv, model_config.dtype,
                model_config.batch_size,  round(weights_gb, 2), round(
                    kvcache_gb, 2), round(activat_gb, 2),
                round(total_gb, 2), True if total_gb < device_gb else False]

    return mem_data


def print_mem_analysis(memory_dict, batchsize_list):
    for pp, pp_dict in memory_dict.items():
        for tp, tp_dict in pp_dict.items():
            for dtype, mem_data in tp_dict.items():
                print(tabulate(mem_data))
            print("\n")


def print_projected_mem_per_device(memory_dict, batchsize_list, in_out_token_list):
    proj_dict = {}
    for pp, pp_dict in memory_dict.items():
        proj_dict[pp] = {}
        for tp, tp_dict in pp_dict.items():
            proj_dict[pp][tp] = {}
            for dtype, mem_data in tp_dict.items():
                proj_dict[pp][tp][dtype] = {}
                for data in mem_data:
                    proj_dict[pp][tp][dtype][data[5]][data[8]] = [data[-2]]
                    # proj_dict[pp][tp][dtype][data[8]].append(data[-2])
                print(proj_dict[pp][tp][dtype])

                # print(tabulate(mem_data))
            print("\n")
