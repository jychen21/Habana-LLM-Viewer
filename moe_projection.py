from tqdm import tqdm
from base import *
from compute import *
from memory import *


def compute_analyzer(model_name, device_list, dtype_list, batchsize_list, context_list):
    model = model_dict[model_name]
    hidden_size = model["hidden_size"]
    num_heads_q = model["num_heads_q"]
    num_heads_kv = model["num_heads_kv"]
    intermediate_size = model["intermediate_size"]
    mlp_with_gate = model["mlp_with_gate"]
    num_layers_mlp = model["num_layers_mlp"]
    num_layers_moe = model["num_layers_moe"]
    num_experts = model["num_experts"]

    projection_dict = {"prefill": {}, "decode": {}, "total": {}}
    analysis_dict = {"prefill": [], "decode": []}

    for device in device_list:
        for dtype in dtype_list:

            # prefill
            print(
                f"projection prefill with dtype[{dtype}], device [{device}] with seq_len: {context_list} and bs {batchsize_list}...")
            projection_dict["prefill"][dtype] = []
            prefill_projection = [cmp_item_list]
            prefill_layer_analysis = dict()
            for in_out in context_list:
                for bs in tqdm(batchsize_list):
                    prefill_layer_analysis[bs] = [layer_analysis_list]
                    model_config = Config(batch_size=bs, seq_len_q=in_out["in"], seq_len_kv=in_out["in"], hidden_size=hidden_size, num_heads_q=num_heads_q,
                                          num_heads_kv=num_heads_kv, intermediate_size=intermediate_size, is_decoding=False, mlp_with_gate=mlp_with_gate,
                                          num_experts=num_experts, num_layers_mlp=num_layers_mlp, num_layers_moe=num_layers_moe, dtype=dtype, device=device)
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
                f"projection decoding with dtype[{dtype}], device [{device}] with seq_len: {context_list} and bs {batchsize_list}...")
            projection_dict["decode"][dtype] = []
            decoding_projection = [cmp_item_list]
            decoding_layer_analysis = dict()
            for in_out in context_list:
                for bs in tqdm(batchsize_list):
                    decoding_layer_analysis[bs] = [layer_analysis_list]
                    model_config = Config(batch_size=bs, seq_len_q=1, seq_len_kv=in_out["out"], hidden_size=hidden_size, num_heads_q=num_heads_q,
                                          num_heads_kv=num_heads_kv, intermediate_size=intermediate_size, is_decoding=True, mlp_with_gate=mlp_with_gate,
                                          num_experts=num_experts, num_layers_mlp=num_layers_mlp, num_layers_moe=num_layers_moe, dtype=dtype, device=device)
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
    # print_analysis(analysis_dict, batchsize_list)
    plot_projection(projection_dict, batchsize_list)


def memory_analyzer(model_name, device_list, dtype_list, batchsize_list, context_list):
    model = model_dict[model_name]
    hidden_size = model["hidden_size"]
    num_heads_q = model["num_heads_q"]
    num_heads_kv = model["num_heads_kv"]
    intermediate_size = model["intermediate_size"]
    mlp_with_gate = model["mlp_with_gate"]
    num_layers_mlp = model["num_layers_mlp"]
    num_layers_moe = model["num_layers_moe"]
    num_experts = model["num_experts"]

    memory_dict = {}

    for device in device_list:
        for pp in device_pp_list:
            memory_dict[pp] = {}

            for tp in device_tp_list:
                memory_dict[pp][tp] = {}

                for dtype in dtype_list:

                    memory_dict[pp][tp][dtype] = [mem_item_list]

                    print(
                        f"memory usage with dtype[{dtype}], device [{device}] with seq_len: {context_list} and bs {batchsize_list}...\n")
                    for in_out in context_list:
                        model_config = Config(batch_size=1, seq_len_q=in_out["in"], seq_len_kv=in_out["in"]+in_out["out"],
                                              hidden_size=hidden_size, num_heads_q=num_heads_q, num_heads_kv=num_heads_kv,
                                              intermediate_size=intermediate_size, is_decoding=False, mlp_with_gate=mlp_with_gate,
                                              num_experts=num_experts, num_layers_mlp=num_layers_mlp,
                                              num_layers_moe=num_layers_moe, dtype=dtype, device=device, pp=pp, tp=tp)
                        mem_persist_weight = mem_persistent_weights(
                            model_config)
                        single_layer_name = None
                        param_layer_mlp = None
                        param_layer_moe = None
                        param_up = None
                        param_gate = None
                        param_down = None
                        if model_config.num_layers_mlp != 0:
                            single_layer_name = "single_layer_mlp"
                            param_layer_mlp = mem_persist_weight["item"][single_layer_name]["item"]["mem_ffn"]
                            param_up = mem_persist_weight["item"][single_layer_name]["item"][
                                "mem_ffn"]["item"]["mem_up(w1)"]["param"]
                            if model_config.mlp_with_gate:
                                param_gate = mem_persist_weight["item"][single_layer_name]["item"][
                                    "mem_ffn"]["item"]["mem_gate(w3)"]["param"]
                            param_down = mem_persist_weight["item"][single_layer_name]["item"][
                                "mem_ffn"]["item"]["mem_down(w2)"]["param"]
                        if model_config.num_layers_moe != 0:
                            single_layer_name = "single_layer_moe"
                            param_layer_moe = mem_persist_weight["item"][single_layer_name]["item"]["mem_ffn"]
                            param_up = mem_persist_weight["item"][single_layer_name]["item"][
                                "mem_ffn"]["item"]["mem_mlp"]["item"]["mem_up(w1)"]["param"]
                            if model_config.mlp_with_gate:
                                param_gate = mem_persist_weight["item"][single_layer_name]["item"][
                                    "mem_ffn"]["item"]["mem_mlp"]["item"]["mem_gate(w3)"]["param"]
                            param_down = mem_persist_weight["item"][single_layer_name]["item"][
                                "mem_ffn"]["item"]["mem_mlp"]["item"]["mem_down(w2)"]["param"]
                        param_qkvo = mem_persist_weight["item"][single_layer_name]["item"]["mem_qkvo"]["param"]
                        param_expert = mem_persist_weight["item"][single_layer_name]["item"]["mem_ffn"]["param"]
                        param_total = mem_persist_weight["param"]
                        mem_total = mem_persist_weight["#mem"]
                        mem_basic_data = [hidden_size, intermediate_size, num_heads_q, num_heads_kv, num_experts, num_layers_mlp,
                                          num_layers_moe, param_qkvo, param_up, param_gate, param_down, param_expert, param_layer_mlp,
                                          param_layer_moe, param_total, dtype, mem_total]
                        mem_basic_list = [mem_item_basic, mem_basic_data]
                        # print(mem_basic_list)

                        for bs in tqdm(batchsize_list):
                            model_config.batch_size = bs
                            mem_decoder_data = mem_decoder(model_config)
                            memory_dict[pp][tp][dtype].append(mem_decoder_data)
                    print("done!\n")

        print_mem_analysis(memory_dict, batchsize_list)
        # print_projected_mem_per_device(
        #     memory_dict, batchsize_list, context_list)


if __name__ == "__main__":
    # ["Llama2-7B", "Llama2-13B", "Mixtral-8x7B", "GLaM-1.2T", "MoE-1.8T"]
    model_name = "Llama2-7B"

    device_list = ["Gaudi2C"]
    device_pp_list = [1, 1, 1, 1, 1, 1, 1]
    device_tp_list = [8]  # [1, 2, 4, 8, 16]
    dtype_list = ["bf16"]  # ["bf16", "fp8"]
    batchsize_list = [1, 2, 4, 8, 16, 32, 64]  # , 128, 129, 256, 512] # 129
    context_list = [{"in": 128, "out": 2048}]
    # context_list = [{"in": 128, "out": 128}, {"in": 1024, "out": 1024}, {
    #     "in": 1, "out": 2048}, {"in": 32000, "out": 512}]

    compute_analyzer(model_name, device_list, dtype_list,
                     batchsize_list, context_list)

    context_list = [{"in": 2048, "out": 2048}]
    # len_factor = 1024
    # input_list = [2*len_factor]
    # context_length_list = [pow(2, i) * len_factor for i in range(2, 6)]
    # for input in input_list:
    #     for context_len in context_length_list:
    #         output = context_len - input
    #         context_list.append({"in": input, "out": output})
    memory_analyzer(model_name, device_list, dtype_list,
                    batchsize_list, context_list)
