from tabulate import tabulate
import matplotlib.pyplot as plt
from base import *


# mem_item_list = ["Device", "Devices", "Layers", "In", "Out", "DType", "BS", "Weights",
#                  "KVCache", "Persist", "Runtime", "Total", "Fit2Device"]
mem_item_list = ["Device", "Devices", "Layers", "In", "Out", "DType", "BS", "Total(GB)", "Fit2Device"]


def mem_persistent(model_config):
    def mem_qkvo_proj():
        # persistent memory
        params_in_weight = model_config.hidden_size * model_config.hidden_size
        params_total = params_in_weight
        params_total *= 4  # 4 for qkvo
        bytes_total = params_total * model_config.num_bytes
        bytes_total /= model_config.num_devices # assume only TP/PP

        mem_rst = {
            "name": "qkvo_proj",
            "#mem": bytes_total,
            "attr": "persistent"
        }

        return mem_rst

    def mem_kvcache():
        assert model_config.num_heads_kv % model_config.num_devices == 0, \
            f"kv heads {model_config.num_heads_kv} need to be divisible by number of devices {model_config.num_devices}"
        head_dim = model_config.hidden_size // model_config.num_heads_q
        # persistent memory
        elements_kv = model_config.batch_size * model_config.num_heads_kv * \
            model_config.seq_len_kv * head_dim * 2
        params_total = elements_kv * 2 # 2 for kv
        bytes_total = params_total * model_config.num_bytes
        bytes_total /= model_config.num_devices # assume only TP/PP

        mem_rst = {
            "name": "kv_cache",
            "#mem": bytes_total,
            "attr": "persistent"
        }

        return mem_rst

    
    def mem_mlp_gate_or_w3():
        # persistent memory
        params_in_weight = model_config.hidden_size * model_config.intermediate_size
        params_total = params_in_weight
        bytes_total = params_total * model_config.num_bytes
        bytes_total /= model_config.num_devices # assume only TP/PP

        mem_rst = {
            "name": "mlp_gate(w3)",
            "#mem": bytes_total,
            "attr": "persistent"
        }

        return mem_rst

    def mem_mlp_up_or_w1():
        # persistent memory
        params_in_weight = model_config.hidden_size * model_config.intermediate_size
        params_total = params_in_weight
        bytes_total = params_total * model_config.num_bytes
        bytes_total /= model_config.num_devices # assume only TP/PP

        mem_rst = {
            "name": "mlp_up(w1)",
            "#mem": bytes_total,
            "attr": "persistent"
        }

        return mem_rst

    def mem_mlp_down_or_w2():
        # persistent memory
        params_in_weight = model_config.intermediate_size * model_config.hidden_size
        params_total = params_in_weight
        bytes_total = params_total * model_config.num_bytes
        bytes_total /= model_config.num_devices # assume only TP/PP

        mem_rst = {
            "name": "mlp_down(w2)",
            "#mem": bytes_total,
            "attr": "persistent"
        }

        return mem_rst

    def mem_mlp_persist():
        up = mem_mlp_up_or_w1()
        if model_config.with_gate:
            gate = mem_mlp_gate_or_w3()
        down = mem_mlp_down_or_w2()

        mem_mlp = up["#mem"] + down["#mem"]
        if model_config.with_gate:
            mem_mlp += gate["#mem"]

        mem_rst = {
            "name": "mlp",
            "#mem": mem_mlp,
            "attr": "persistent"
        }

        return mem_rst


    def mem_moe_persist():
        mem_mlp = mem_mlp_persist()
        mem_moe = mem_mlp["#mem"] * model_config.num_experts

        mem_rst = {
            "name": "moe",
            "#mem": mem_moe,
            "attr": "persistent"
        }

        return mem_rst

    def mem_single_layer_persist():
        mem_qkvo = mem_qkvo_proj()
        mem_cache = mem_kvcache()
        mem_moe = mem_moe_persist()
        mem_single_layer = mem_qkvo["#mem"] + mem_moe["#mem"]

        mem_rst = {
            "name": "single_layer",
            "#mem": mem_single_layer,
            "attr": "persistent"
        }

        return mem_rst
    
    mem_single_layer = mem_single_layer_persist()

    return mem_single_layer["#mem"] * model_config.num_layers


def mem_activation(model_config):
    def mem_attn_qk():
        assert model_config.num_heads_q % model_config.num_devices == 0, \
            f"q heads {model_config.num_heads_q} need to be divisible by number of devices {model_config.num_devices}"
        # activation memory
        params_out = model_config.batch_size * model_config.num_heads_q * \
            model_config.seq_len_q * model_config.seq_len_kv
        params_total = params_out
        bytes_total = params_total * model_config.num_bytes
        bytes_total /= model_config.num_devices # assume only TP/PP

        mem_rst = {
            "name": "q@k_T",
            "#mem": bytes_total,
            "attr": "activation"
        }

        return mem_rst

    '''
    def mem_attn_softmax():
        assert model_config.num_heads_q % model_config.num_devices == 0, \
            f"q heads {model_config.num_heads_q} need to be divisible by number of devices {model_config.num_devices}"
        # activation memory, same with mem_attn_qk
        params_out = model_config.batch_size * model_config.num_heads_q * \
            model_config.seq_len_q * model_config.seq_len_kv
        params_total = params_out
        bytes_total = params_total * model_config.num_bytes
        bytes_total /= model_config.num_devices # assume only TP/PP

        mem_rst = {
            "name": "softmax",
            "#mem": bytes_total,
            "attr": "activation"
        }

        return mem_rst

    def mem_attn_scorev():
        assert model_config.num_heads_q % model_config.num_devices == 0, \
            f"q heads {model_config.num_heads_q} need to be divisible by number of devices {model_config.num_devices}"
        # activation memory, same with mem_attn_qk
        params_in_score = model_config.batch_size * model_config.num_heads_q * \
            model_config.seq_len_q * model_config.seq_len_kv
        params_total = params_in_score
        bytes_total = params_total * model_config.num_bytes
        bytes_total /= model_config.num_devices # assume only TP/PP

        mem_rst = {
            "name": "score@v",
            "#mem": bytes_total,
            "attr": "activation"
        }

        return mem_rst
    '''

    def mem_attn_activation():
        qk = mem_attn_qk()
        mem_attn = qk["#mem"]

        mem_rst = {
            "name": "attn",
            "#mem": mem_attn,
            "attr": "activation"
        }

        return mem_rst

    '''
    def mem_mlp_activation():
        # Todo
        mem_rst = {
            "name": "mlp",
            "#mem": 0,
            "attr": "activation"
        }

        return mem_rst
    '''

    attn_activat = mem_attn_activation()
    mlp_activat = mem_attn_activation()

    return max(attn_activat["#mem"], mlp_activat["#mem"])


def mem_decoder(model_config):
    persist = mem_persistent(model_config)
    activat = mem_activation(model_config)

    return persist, activat


def print_mem_analysis(memory_dict, batchsize_list):
    for num_devices, mem_analysis in memory_dict.items():
        for dtype, mem_data in mem_analysis.items():
            print(tabulate(mem_data))
        print("\n")
