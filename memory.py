from tabulate import tabulate
import matplotlib.pyplot as plt
from config import *


mem_item_basic = ["H", "H_Inter", "Heads_Q", "HeadsKV", "Experts", "Layers(Mlp)",
                  "Layers(MoE)", "QKVO", "Up(W1)", "Gate(W3)", "Down(W2)", "1Expert",
                  "1Layer(Mlp)", "1Layer(MoE)", "ParamsTotal", "Dtype", "Weights"]
mem_item_list = ["Device", "PP", "TP", "NumDevices", "Layers", "In", "Out", "DType",
                 "BS", "Weights(GB)", "KVcache(GB)", "Activat(GB)", "Total(GB)",
                 "Fit2Device"]


def mem_persistent_weights(config):
    def mem_qkvo_proj():
        hidden_size = config.model_config.hidden_size
        num_bytes = config.hardware_config.num_bytes
        tp = config.hardware_config.tp

        assert hidden_size % tp == 0, \
            f"hidden_size {hidden_size} should be divisible by tp size {tp}!"

        params_total = hidden_size * hidden_size
        params_total *= 4  # 4 for qkvo
        params_total /= tp
        bytes_total = params_total * num_bytes

        mem_rst = {
            "name": "qkvo_proj",
            "params": params_total,
            "memory": bytes_total,
            "attr": "persistent",
            "items": None
        }

        return mem_rst

    def mem_mlp_up_or_w1():
        hidden_size = config.model_config.hidden_size
        intermediate_size = config.model_config.intermediate_size
        num_bytes = config.hardware_config.num_bytes
        tp = config.hardware_config.tp

        assert intermediate_size % tp == 0, \
            f"hidden_size {intermediate_size} should be divisible by tp size {tp}!"

        params_total = hidden_size * intermediate_size
        params_total /= tp
        bytes_total = params_total * num_bytes

        mem_rst = {
            "name": "up(w1)",
            "params": params_total,
            "memory": bytes_total,
            "attr": "persistent",
            "items": None
        }

        return mem_rst

    def mem_mlp_down_or_w2():
        hidden_size = config.model_config.hidden_size
        intermediate_size = config.model_config.intermediate_size
        num_bytes = config.hardware_config.num_bytes
        tp = config.hardware_config.tp

        assert intermediate_size % tp == 0, \
            f"hidden_size {intermediate_size} should be divisible by tp size {tp}!"

        params_total = intermediate_size * hidden_size
        params_total /= tp
        bytes_total = params_total * num_bytes

        mem_rst = {
            "name": "down(w2)",
            "params": params_total,
            "memory": bytes_total,
            "attr": "persistent",
            "items": None
        }

        return mem_rst

    def mem_mlp_gate_or_w3():
        hidden_size = config.model_config.hidden_size
        intermediate_size = config.model_config.intermediate_size
        num_bytes = config.hardware_config.num_bytes
        tp = config.hardware_config.tp

        assert intermediate_size % tp == 0, \
            f"hidden_size {intermediate_size} should be divisible by tp size {tp}!"

        params_in_weight = hidden_size * intermediate_size
        params_total = params_in_weight
        params_total /= tp
        bytes_total = params_total * num_bytes

        mem_rst = {
            "name": "gate(w3)",
            "params": params_total,
            "memory": bytes_total,
            "attr": "persistent",
            "items": None
        }

        return mem_rst

    def mem_mlp():
        mlp_with_gate = config.model_config.mlp_with_gate

        up = mem_mlp_up_or_w1()
        down = mem_mlp_down_or_w2()
        gate = None
        if mlp_with_gate:
            gate = mem_mlp_gate_or_w3()

        params_total = up["params"] + down["params"]
        mem_total = up["memory"] + down["memory"]
        if mlp_with_gate:
            params_total += gate["params"]
            mem_total += gate["memory"]

        mem_rst = {
            "name": "mlp",
            "params": params_total,
            "memory": mem_total,
            "attr": "persistent",
            "items": {
                "up": up,
                "down": down,
                "gate": gate
            }
        }

        return mem_rst

    def mem_moe():
        num_experts = config.model_config.num_experts

        mlp = mem_mlp()
        params_total = mlp["params"] * num_experts
        mem_total = mlp["memory"] * num_experts

        mem_rst = {
            "name": "moe",
            "params": params_total,
            "memory": mem_total,
            "attr": "persistent",
            "items": {
                "up": mlp["items"]["up"],
                "down": mlp["items"]["down"],
                "gate": mlp["items"]["gate"]
            }
        }

        return mem_rst

    def mem_single_layer_mlp():
        num_layers_mlp = config.model_config.num_layers_mlp

        qkvo = None
        mlp = None
        params_total = 0
        mem_total = 0
        mlp_name = None
        if num_layers_mlp != 0:
            qkvo = mem_qkvo_proj()
            mlp = mem_mlp()
            params_total = qkvo["params"] + mlp["params"]
            mem_total = qkvo["memory"] + mlp["memory"]
            mlp_name = mlp["name"]

        mem_rst = {
            "name": f"single_layer_{mlp_name}",
            "params": params_total,
            "memory": mem_total,
            "attr": "persistent",
            "items": {
                "qkvo": qkvo,
                "ffn": mlp
            }
        }

        return mem_rst

    def mem_single_layer_moe():
        num_layers_moe = config.model_config.num_layers_moe

        qkvo = None
        moe = None
        params_total = 0
        mem_total = 0
        moe_name = None
        if num_layers_moe != 0:
            qkvo = mem_qkvo_proj()
            moe = mem_moe()
            params_total = qkvo["params"] + moe["params"]
            mem_total = qkvo["memory"] + moe["memory"]
            moe_name = moe["name"]

        mem_rst = {
            "name": f"single_layer_{moe_name}",
            "params": params_total,
            "memory": mem_total,
            "attr": "persistent",
            "items": {
                "qkvo": qkvo,
                "ffn": moe
            }
        }

        return mem_rst

    num_layers_mlp = config.model_config.num_layers_mlp
    num_layers_moe = config.model_config.num_layers_moe

    single_layer_mlp = None
    single_layer_moe = None
    params_total = 0
    mem_total = 0
    if num_layers_mlp != 0:
        single_layer_mlp = mem_single_layer_mlp()
        params_total += single_layer_mlp["params"] * num_layers_mlp
        mem_total += single_layer_mlp["memory"] * num_layers_mlp
    if num_layers_moe != 0:
        single_layer_moe = mem_single_layer_moe()
        params_total += single_layer_moe["params"] * num_layers_moe
        mem_total += single_layer_moe["memory"] * num_layers_moe

    mem_rst = {
        "name": "mem_persist_weight",
        "params": params_total,
        "memory": mem_total,
        "attr": "persistent",
        "items": {
            "single_layer_mlp": single_layer_mlp,
            "single_layer_moe": single_layer_moe
        }
    }

    return mem_rst


def mem_persistent_embedding(config):
    # Todo: add memory projection of embedding
    params_total = 0
    mem_total = 0

    mem_rst = {
        "name": "mem_persist_embedding",
        "params": params_total,
        "memory": mem_total,
        "attr": "persistent",
        "items": None
    }

    return mem_rst


def mem_persistent_kvcache(config):
    hidden_size = config.model_config.hidden_size
    num_heads_q = config.model_config.num_heads_q
    num_heads_kv = config.model_config.num_heads_kv
    head_dim = hidden_size // num_heads_q
    num_bytes = config.hardware_config.num_bytes
    tp = config.hardware_config.tp
    batch_size = config.input_config.batch_size
    seq_len_kv = config.input_config.seq_len_kv

    assert num_heads_kv % tp == 0, \
        f"kv heads {num_heads_kv} should be divisible by tp size {tp}!"

    params_total = batch_size * num_heads_kv * seq_len_kv * head_dim * 2  # 2 for kv
    params_total /= tp
    bytes_total = params_total * num_bytes

    mem_rst = {
        "name": "mem_persist_kvcache",
        "params": params_total,
        "memory": bytes_total,
        "attr": "persistent",
        "items": None
    }

    return mem_rst


def mem_activation(config):
    '''
    def mem_attn_qk():
        num_heads_q = config.model_config.num_heads_q
        num_bytes = config.hardware_config.num_bytes
        tp = config.hardware_config.tp
        batch_size = config.input_config.batch_size
        seq_len_q = config.input_config.seq_len_q
        seq_len_kv = config.input_config.seq_len_kv

        assert num_heads_q % tp == 0, \
            f"q heads {num_heads_q} should be divisible by tp size {tp}!"

        params_total = batch_size * num_heads_q * seq_len_q * seq_len_kv
        params_total /= tp
        bytes_total = params_total * num_bytes

        mem_rst = {
            "name": "q@k_T",
            "params": params_total,
            "memory": bytes_total,
            "attr": "activation",
            "items": {}
        }

        return mem_rst
    '''

    def mem_attn_softmax_standard():
        num_heads_q = config.model_config.num_heads_q
        num_bytes = config.hardware_config.num_bytes
        tp = config.hardware_config.tp
        batch_size = config.input_config.batch_size
        seq_len_q = config.input_config.seq_len_q
        seq_len_kv = config.input_config.seq_len_kv

        assert num_heads_q % tp == 0, \
            f"q heads {num_heads_q} should be divisible by tp size {tp}!"

        params_in = batch_size * num_heads_q * seq_len_q * seq_len_kv
        params_out = batch_size * num_heads_q * seq_len_q * seq_len_kv
        params_total = params_in + params_out
        params_total /= tp

        bytes_total = params_total * num_bytes

        mem_rst = {
            "name": "softmax",
            "params": params_total,
            "memory": bytes_total,
            "attr": "activation",
            "items": {}
        }

        return mem_rst

    '''
    def mem_attn_scorev():
        num_heads_q = config.model_config.num_heads_q
        num_bytes = config.hardware_config.num_bytes
        tp = config.hardware_config.tp
        batch_size = config.input_config.batch_size
        seq_len_q = config.input_config.seq_len_q
        seq_len_kv = config.input_config.seq_len_kv

        assert num_heads_q % tp == 0, \
            f"q heads {num_heads_q} should be divisible by tp size {tp}!"

        params_total = batch_size * num_heads_q * seq_len_q * seq_len_kv
        params_total /= tp
        bytes_total = params_total * num_bytes

        mem_rst = {
            "name": "score@v",
            "params": params_total,
            "memory": bytes_total,
            "attr": "activation",
            "items": {}
        }

        return mem_rst
    '''

    def mem_attn_activation():
        attn = mem_attn_softmax_standard()
        mem_attn = attn["memory"]

        mem_rst = {
            "name": "attn",
            "params": attn["params"],
            "memory": mem_attn,
            "attr": "activation",
            "items": {}
        }

        return mem_rst

    def mem_ffn_activation():
        params_total = 0
        # Todo
        mem_rst = {
            "name": "ffn",
            "params": params_total,
            "memory": 0,
            "attr": "activation",
            "items": {}
        }

        return mem_rst

    attn_activat = mem_attn_activation()
    mlp_activat = mem_ffn_activation()

    params_total = max(attn_activat["params"], mlp_activat["params"])
    mem_activat = max(attn_activat["memory"], mlp_activat["memory"])

    mem_rst = {
        "name": "activation",
        "params": params_total,
        "memory": mem_activat,
        "attr": "activation",
        "items": {"attn": attn_activat, "ffn": mlp_activat}
    }
    return mem_rst


def mem_decoder(config):
    mem_persist_weights = mem_persistent_weights(config)
    mem_persist_kvcache = mem_persistent_kvcache(config)
    mem_activat = mem_activation(config)
    params_total = mem_persist_weights["params"] + \
        mem_persist_kvcache["params"] + mem_activat["params"]
    mem_total = mem_persist_weights["memory"] + mem_activat["memory"]

    mem_rst = {
        "name": "decoder",
        "params": params_total,
        "memory": mem_total,
        "attr": "persist_with_activat",
        "items": {
            "persist_weights": mem_persist_weights,
            "persist_kvcache": mem_persist_kvcache,
            "activation": mem_activat
        }
    }

    # mem_item_list = ["Device", "PP", "TP", "NumDevices", "Layers", "In", "Out", "DType",
    #              "BS", "Weights(GB)", "KVcache(GB)", "Activat(GB)", "Total(GB)", "Fit2Device"]
    device_gb = config.hardware_config.hbm_capacity
    gigabytes = 1024 * 1024 * 1024
    weights_gb = mem_persist_weights["memory"] / gigabytes
    kvcache_gb = mem_persist_kvcache["memory"] / gigabytes
    activat_gb = mem_activat["memory"] / gigabytes
    total_gb = mem_rst["memory"] / gigabytes

    device = config.hardware_config.device
    type = config.hardware_config.type
    dtype = config.hardware_config.dtype
    pp = config.hardware_config.pp
    tp = config.hardware_config.tp
    num_devices = config.hardware_config.num_devices
    num_layers = config.model_config.num_layers
    seq_len_q = config.input_config.seq_len_q
    seq_len_kv = config.input_config.seq_len_kv
    batch_size = config.input_config.batch_size

    mem_data = [f"{device}{type}", pp, tp, num_devices, num_layers, seq_len_q, seq_len_kv, dtype,
                batch_size,  round(weights_gb, 2), round(
                    kvcache_gb, 2), round(activat_gb, 2),
                round(total_gb, 2), True if total_gb < device_gb else False]

    return mem_data


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

    seq_len_q = input
    seq_len_kv = input + output

    cfg = Config(device, type, dtype, pp, tp, hidden_size, num_heads_q, num_heads_kv,
                 intermediate_size, mlp_with_gate, num_experts, num_layers_mlp,
                 num_layers_moe, seq_len_q, seq_len_kv, bs, kvcache_bucket)
    proj_rst["weights"] = mem_persistent_weights(cfg)
    proj_rst["kvcache"] = mem_persistent_kvcache(cfg)
    proj_rst["activat"] = mem_activation(cfg)
    mem_total = 0
    for _, v in proj_rst.items():
        mem_total += v["memory"]
    proj_rst["size"] = "OOM" if mem_total >= cfg.hardware_config.hbm_capacity \
        else round(mem_total / GigaBytes, 2)

    return proj_rst
