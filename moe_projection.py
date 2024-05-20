from tqdm import tqdm
from base import *


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


if __name__ == "__main__":
    hidden_size = 4096
    num_heads_q = 32
    num_heads_kv = 8
    intermediate_size = 14336
    mlp_with_gate = True
    num_experts = 8
    num_layers = 32

    dtype_list = ["bf16", "fp8"] # ["bf16"]
    # in_out_token_list = [{"in": 128, "out": 128}, {"in": 1024, "out": 1024}, {
    #     "in": 1, "out": 2048}, {"in": 32000, "out": 512}]
    in_out_token_list = [{"in": 128, "out": 128}]
    batchsize_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    # batchsize_list = [1, 4, 16, 24, 28, 32,
    #                   48, 56, 60, 64, 96, 112, 128, 256, 512]

    projection_dict = {"prefill": {}, "decode": {}}
    analysis_dict = {"prefill": [], "decode": []}

    for dtype in dtype_list:
        device = type2devices[dtype]
        num_bytes = type2bytes[dtype]
        bw = device_bw_tops[device][0]
        tops = device_bw_tops[device][1]
        tops_tpc = device_bw_tops[device][2]

        # prefill
        print(
            f"projection prefill with dtype[{dtype}], device [{device}] with seq_len: {in_out_token_list} and bs {batchsize_list}...")
        projection_dict["prefill"][dtype] = []
        prefill_projection = [item_list]
        prefill_layer_analysis = dict()
        for in_out in in_out_token_list:
            for bs in tqdm(batchsize_list):
                prefill_layer_analysis[bs] = [layer_analysis_list]
                model_config = Config(batch_size=bs, seq_len_q=in_out["in"], seq_len_kv=in_out["in"], hidden_size=hidden_size, num_heads_q=num_heads_q,
                                      num_heads_kv=num_heads_kv, intermediate_size=intermediate_size, is_decoding=False, num_bytes=num_bytes,
                                      bw=bw, tops=tops, tops_tpc=tops_tpc, with_gate=mlp_with_gate, num_experts=num_experts, num_layers=num_layers)
                runtime_decoder, single_layer_items = proj_decoder(
                    model_config)
                prefill_projection.append([device, model_config.hidden_size, model_config.num_heads_q, model_config.num_heads_kv,
                                           model_config.intermediate_size, model_config.is_decoding, model_config.num_experts,
                                           model_config.num_layers, in_out["in"], in_out["out"], dtype, bs, round(
                                               runtime_decoder, 2),
                                           round(1/runtime_decoder * model_config.batch_size, 2)])
                prefill_layer_analysis[bs].append(
                    [in_out["in"], in_out["out"], dtype, bs, single_layer_items["qkvo"]["name"], round(single_layer_items["qkvo"]["#ops"]/1e9, 2),
                     round(single_layer_items["qkvo"]["#mem"]/1024/1024/1024,
                           2), round(single_layer_items["qkvo"]["tops_roofline"]/1e12, 2),
                     round(single_layer_items["qkvo"]["math_ai"], 2), single_layer_items["qkvo"]["bound"]])
                for item in single_layer_items["attn"]:
                    prefill_layer_analysis[bs].append(
                        [in_out["in"], in_out["out"], dtype, bs, item["name"], round(item["#ops"]/1e9, 2), round(item["#mem"]/1024/1024/1024, 2),
                         round(item["tops_roofline"]/1e12, 2), round(item["math_ai"], 2), item["bound"]])
                for item in single_layer_items["moe"]:
                    prefill_layer_analysis[bs].append(
                        [in_out["in"], in_out["out"], dtype, bs, item["name"], round(item["#ops"]/1e9, 2), round(item["#mem"]/1024/1024/1024, 2),
                         round(item["tops_roofline"]/1e12, 2), round(item["math_ai"], 2), item["bound"]])
        print("done!\n")
        # print(tabulate(prefill_projection))
        # for bs in batchsize_list:
        #     print(tabulate(prefill_layer_analysis[bs]))
        projection_dict["prefill"][dtype].append(prefill_projection)
        analysis_dict["prefill"].append(prefill_layer_analysis)

        # decode
        print(
            f"projection decoding with dtype[{dtype}], device [{device}] with seq_len: {in_out_token_list} and bs {batchsize_list}...")
        projection_dict["decode"][dtype] = []
        decoding_projection = [item_list]
        decoding_layer_analysis = dict()
        for in_out in in_out_token_list:
            for bs in tqdm(batchsize_list):
                decoding_layer_analysis[bs] = [layer_analysis_list]
                model_config = Config(batch_size=bs, seq_len_q=1, seq_len_kv=in_out["out"], hidden_size=hidden_size, num_heads_q=num_heads_q,
                                      num_heads_kv=num_heads_kv, intermediate_size=intermediate_size, is_decoding=True, num_bytes=num_bytes,
                                      bw=bw, tops=tops, tops_tpc=tops_tpc, with_gate=mlp_with_gate, num_experts=num_experts, num_layers=num_layers)
                runtime_decoder, single_layer_items = proj_decoder(
                    model_config)
                decoding_projection.append([device, model_config.hidden_size, model_config.num_heads_q, model_config.num_heads_kv,
                                            model_config.intermediate_size, model_config.is_decoding, model_config.num_experts,
                                            model_config.num_layers, in_out["in"], in_out["out"], dtype, bs, round(
                                                runtime_decoder, 2),
                                            round(1/runtime_decoder * model_config.batch_size, 2)])
                decoding_layer_analysis[bs].append(
                    [in_out["in"], in_out["out"], dtype, bs, single_layer_items["qkvo"]["name"], round(single_layer_items["qkvo"]["#ops"]/1e9, 2),
                     round(single_layer_items["qkvo"]["#mem"]/1024/1024/1024,
                           2), round(single_layer_items["qkvo"]["tops_roofline"]/1e12, 2),
                     round(single_layer_items["qkvo"]["math_ai"], 2), single_layer_items["qkvo"]["bound"]]
                )
                for item in single_layer_items["attn"]:
                    decoding_layer_analysis[bs].append(
                        [in_out["in"], in_out["out"], dtype, bs, item["name"], round(item["#ops"]/1e9, 2), round(item["#mem"]/1024/1024/1024, 2),
                         round(item["tops_roofline"]/1e12, 2), round(item["math_ai"], 2), item["bound"]])
                for item in single_layer_items["moe"]:
                    decoding_layer_analysis[bs].append(
                        [in_out["in"], in_out["out"], dtype, bs, item["name"], round(item["#ops"]/1e9, 2), round(item["#mem"]/1024/1024/1024, 2),
                         round(item["tops_roofline"]/1e12, 2), round(item["math_ai"], 2), item["bound"]])
        print("done!")
        # print(tabulate(decoding_projection))
        # for bs in batchsize_list:
        #     print(tabulate(decoding_layer_analysis[bs]))
        projection_dict["decode"][dtype].append(decoding_projection)
        analysis_dict["decode"].append(decoding_layer_analysis)

    print_projection(projection_dict)
    print_analysis(analysis_dict, batchsize_list)
    # plot_projection(projection_dict, batchsize_list)
