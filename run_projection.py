from tqdm import tqdm
from base import *
from compute import *
from memory import *


def compute_analyzer(model_name, device_list, device_type_list, dtype_list, batchsize_list, context_list):
    model = ModelDict[model_name]
    hidden_size = model["hidden_size"]
    num_heads_q = model["num_heads_q"]
    num_heads_kv = model["num_heads_kv"]
    intermediate_size = model["intermediate_size"]
    mlp_with_gate = model["mlp_with_gate"]
    num_layers_mlp = model["num_layers_mlp"]
    num_layers_moe = model["num_layers_moe"]
    num_experts = model["num_experts"]

    for device in device_list:
        for type in device_type_list:
            projection_dict = {"prefill": {}, "decode": {}, "total": {}}
            layer_projection_dict = {"prefill": {}, "decode": {}}
            analysis_dict = {"prefill": [], "decode": []}

            for dtype in dtype_list:

                # prefill
                print(
                    f"projection prefill with dtype[{dtype}], device [{device}] with seq_len: {context_list} and bs {batchsize_list}...")
                projection_dict["prefill"][dtype] = []
                layer_projection_dict["prefill"][dtype] = []
                prefill_projection = []
                prefill_layer_projection = []
                prefill_layer_analysis = dict()
                for in_out in context_list:
                    for bs in tqdm(batchsize_list):
                        prefill_layer_analysis[bs] = [layer_analysis_list]
                        config = Config(device=device, type=type, dtype=dtype, pp=1, tp=1, hidden_size=hidden_size,
                                        num_heads_q=num_heads_q, num_heads_kv=num_heads_kv,
                                        intermediate_size=intermediate_size, mlp_with_gate=mlp_with_gate,
                                        num_experts=num_experts, num_layers_mlp=num_layers_mlp,
                                        num_layers_moe=num_layers_moe, seq_len_q=in_out["in"],
                                        seq_len_kv=in_out["in"], batch_size=bs,
                                        is_decoding=False, kvcache_bucket=False)
                        runtime_decoder, single_layer_items = proj_decoder(
                            config)
                        model_config = config.model_config
                        input_config = config.input_config
                        prefill_projection.append({"config": config, "latency": runtime_decoder})
                        # prefill_projection.append([f"{device}{type}", model_config.hidden_size, model_config.num_heads_q, model_config.num_heads_kv,
                        #                            model_config.intermediate_size, config.is_decoding, model_config.num_experts,
                        #                            model_config.num_layers, in_out["in"], in_out["out"], dtype, bs,
                        #                            round(runtime_decoder, 2),
                        #                            round(1/runtime_decoder * input_config.batch_size, 2)])
                        prefill_layer_projection.append(single_layer_items)
                        prefill_layer_analysis[bs].append(
                            [in_out["in"], in_out["out"], dtype, bs, single_layer_items["qkvo"]["name"], round(single_layer_items["qkvo"]["operations"]/1e9, 2),
                             round(single_layer_items["qkvo"]["size"]/1024/1024/1024,
                                   2), round(single_layer_items["qkvo"]["tops_roofline"]/1e12, 2),
                             round(single_layer_items["qkvo"]["math_ai"], 2), single_layer_items["qkvo"]["bound"]])
                        for items in single_layer_items["attn"]:
                            prefill_layer_analysis[bs].append(
                                [in_out["in"], in_out["out"], dtype, bs, items["name"], round(items["operations"]/1e9, 2), round(items["size"]/1024/1024/1024, 2),
                                 round(items["tops_roofline"]/1e12, 2), round(items["math_ai"], 2), items["bound"]])
                        for items in single_layer_items["ffn"]:
                            prefill_layer_analysis[bs].append(
                                [in_out["in"], in_out["out"], dtype, bs, items["name"], round(items["operations"]/1e9, 2), round(items["size"]/1024/1024/1024, 2),
                                 round(items["tops_roofline"]/1e12, 2), round(items["math_ai"], 2), items["bound"]])
                print("done!\n")
                projection_dict["prefill"][dtype].append(prefill_projection)
                layer_projection_dict["prefill"][dtype].append(
                    prefill_layer_projection)
                analysis_dict["prefill"].append(prefill_layer_analysis)

                # decode
                print(
                    f"projection decoding with dtype[{dtype}], device [{device}] with seq_len: {context_list} and bs {batchsize_list}...")
                projection_dict["decode"][dtype] = []
                layer_projection_dict["decode"][dtype] = []
                decoding_projection = []
                layer_decoding_projection = []
                decoding_layer_analysis = dict()
                for in_out in context_list:
                    for bs in tqdm(batchsize_list):
                        decoding_layer_analysis[bs] = [layer_analysis_list]
                        config = Config(device=device, type=type, dtype=dtype, pp=1, tp=1, hidden_size=hidden_size,
                                        num_heads_q=num_heads_q, num_heads_kv=num_heads_kv,
                                        intermediate_size=intermediate_size, mlp_with_gate=mlp_with_gate,
                                        num_experts=num_experts, num_layers_mlp=num_layers_mlp,
                                        num_layers_moe=num_layers_moe, seq_len_q=1,
                                        seq_len_kv=in_out["in"]+in_out["out"], batch_size=bs,
                                        is_decoding=True, kvcache_bucket=True)
                        runtime_decoder, single_layer_items = proj_decoder(
                            config)
                        model_config = config.model_config
                        input_config = config.input_config
                        throughput = 1 / runtime_decoder * input_config.batch_size
                        decoding_projection.append({"config": config, "latency": runtime_decoder})
                        # decoding_projection.append([f"{device}{type}", model_config.hidden_size, model_config.num_heads_q, model_config.num_heads_kv,
                        #                             model_config.intermediate_size, config.is_decoding, model_config.num_experts,
                        #                             model_config.num_layers, in_out["in"], in_out["out"], dtype, bs, round(
                        #                                 runtime_decoder, 2),
                        #                             round(throughput, 2)])
                        layer_decoding_projection.append(single_layer_items)
                        decoding_layer_analysis[bs].append(
                            [in_out["in"], in_out["out"], dtype, bs, single_layer_items["qkvo"]["name"], round(single_layer_items["qkvo"]["operations"]/1e9, 2),
                             round(single_layer_items["qkvo"]["size"]/1024/1024/1024,
                                   2), round(single_layer_items["qkvo"]["tops_roofline"]/1e12, 2),
                             round(single_layer_items["qkvo"]["math_ai"], 2), single_layer_items["qkvo"]["bound"]]
                        )
                        for items in single_layer_items["attn"]:
                            decoding_layer_analysis[bs].append(
                                [in_out["in"], in_out["out"], dtype, bs, items["name"], round(items["operations"]/1e9, 2), round(items["size"]/1024/1024/1024, 2),
                                 round(items["tops_roofline"]/1e12, 2), round(items["math_ai"], 2), items["bound"]])
                        for items in single_layer_items["ffn"]:
                            decoding_layer_analysis[bs].append(
                                [in_out["in"], in_out["out"], dtype, bs, items["name"], round(items["operations"]/1e9, 2), round(items["size"]/1024/1024/1024, 2),
                                 round(items["tops_roofline"]/1e12, 2), round(items["math_ai"], 2), items["bound"]])
                print("done!")
                projection_dict["decode"][dtype].append(decoding_projection)
                layer_projection_dict["decode"][dtype].append(
                    layer_decoding_projection)
                analysis_dict["decode"].append(decoding_layer_analysis)

            print_projection(projection_dict, device, type, model_name, context_list, batchsize_list)
            print_layer_projection(layer_projection_dict, device, type, model_name, context_list, batchsize_list)
            # print_analysis(analysis_dict, batchsize_list, model_name)
            # plot_projection(projection_dict, batchsize_list, model_name)


def memory_analyzer(model_name, device_list, device_type_list, pp_list, tp_list, dtype_list, batchsize_list, context_list):
    model = ModelDict[model_name]
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
        for type in device_type_list:
            for pp in pp_list:
                memory_dict[pp] = {}

                for tp in tp_list:
                    memory_dict[pp][tp] = {}

                    for dtype in dtype_list:

                        memory_dict[pp][tp][dtype] = [mem_item_list]

                        print(
                            f"memory usage with dtype[{dtype}], device [{device}] with seq_len: {context_list} and bs {batchsize_list}...\n")
                        for in_out in context_list:
                            config = Config(device=device, type=type, dtype=dtype, pp=pp, tp=tp, hidden_size=hidden_size,
                                            num_heads_q=num_heads_q, num_heads_kv=num_heads_kv,
                                            intermediate_size=intermediate_size, mlp_with_gate=mlp_with_gate,
                                            num_experts=num_experts, num_layers_mlp=num_layers_mlp,
                                            num_layers_moe=num_layers_moe, seq_len_q=in_out["in"],
                                            seq_len_kv=in_out["in"]+in_out["out"], batch_size=1,
                                            is_decoding=False, kvcache_bucket=False)
                            model_config = config.model_config
                            mem_persist_weight = mem_persistent_weights(config)
                            single_layer_name = None
                            param_layer_mlp = None
                            param_layer_moe = None
                            param_up = None
                            param_gate = None
                            param_down = None
                            if model_config.num_layers_mlp != 0:
                                single_layer_name = "single_layer_mlp"
                                param_layer_mlp = mem_persist_weight["items"][single_layer_name]["items"]["ffn"]
                                param_up = mem_persist_weight["items"][single_layer_name][
                                    "items"]["ffn"]["items"]["up"]["params"]
                                if model_config.mlp_with_gate:
                                    param_gate = mem_persist_weight["items"][single_layer_name][
                                        "items"]["ffn"]["items"]["gate"]["params"]
                                param_down = mem_persist_weight["items"][single_layer_name][
                                    "items"]["ffn"]["items"]["down"]["params"]
                            if model_config.num_layers_moe != 0:
                                single_layer_name = "single_layer_moe"
                                param_layer_moe = mem_persist_weight["items"][single_layer_name]["items"]["ffn"]
                                param_up = mem_persist_weight["items"][single_layer_name][
                                    "items"]["ffn"]["items"]["up"]["params"]
                                if model_config.mlp_with_gate:
                                    param_gate = mem_persist_weight["items"][single_layer_name][
                                        "items"]["ffn"]["items"]["gate"]["params"]
                                param_down = mem_persist_weight["items"][single_layer_name][
                                    "items"]["ffn"]["items"]["down"]["params"]
                            param_qkvo = mem_persist_weight["items"][single_layer_name]["items"]["qkvo"]["params"]
                            param_expert = mem_persist_weight["items"][single_layer_name]["items"]["ffn"]["params"]
                            param_total = mem_persist_weight["params"]
                            mem_total = mem_persist_weight["memory"]
                            mem_basic_data = [hidden_size, intermediate_size, num_heads_q, num_heads_kv, num_experts, num_layers_mlp,
                                              num_layers_moe, param_qkvo, param_up, param_gate, param_down, param_expert, param_layer_mlp,
                                              param_layer_moe, param_total, dtype, mem_total]
                            mem_basic_list = [mem_item_basic, mem_basic_data]
                            # print(mem_basic_list)

                            for bs in tqdm(batchsize_list):
                                config.model_config.batch_size = bs
                                mem_decoder_data = mem_decoder(config)
                                memory_dict[pp][tp][dtype].append(
                                    mem_decoder_data)
                        print("done!\n")

            print_mem_analysis(memory_dict, batchsize_list)
            # print_projected_mem_per_device(
            #     model_name, memory_dict, batchsize_list, context_list)


if __name__ == "__main__":
    # ["Llama2-7B", "Llama2-13B", "Mixtral-8x7B", "GLaM-1.2T", "MoE-1.8T"]
    model_name = "Mixtral-8x7B"

    device_list = ["IntelGaudi2"]
    device_type_list = ["B", "C", "D"]
    dtype_list = ["BF16", "FP8"]
    # batchsize_list = [1, 2, 4, 8, 16, 32, 64, 128, 129, 160, 192, 256, 257, 512]  # 129
    # batchsize_list = [1, 2, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512]  # 129
    batchsize_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]  # 129
    context_list = [{"in": 128, "out": 2048}]
    # context_list = [{"in": 128, "out": 128}, {"in": 1024, "out": 1024}, {
    #     "in": 1, "out": 2048}, {"in": 32000, "out": 512}]

    compute_analyzer(model_name, device_list, device_type_list,
                     dtype_list, batchsize_list, context_list)
    
    model_name = "Llama2-7B"
    compute_analyzer(model_name, device_list, device_type_list,
                     dtype_list, batchsize_list, context_list)

    dtype_list = ["FP8"]  # ["BF16", "FP8"]
    pp_list = [1, 1, 1, 1, 1, 1, 1]
    tp_list = [16]  # [1, 2, 4, 8, 16]
    context_list = [{"in": 2048, "out": 4096}]
    # len_factor = 1024
    # input_list = [2*len_factor]
    # context_length_list = [pow(2, i) * len_factor for i in range(2, 6)]
    # for input in input_list:
    #     for context_len in context_length_list:
    #         output = context_len - input
    #         context_list.append({"in": input, "out": output})
    # memory_analyzer(model_name, device_list, device_type_list, pp_list, tp_list, dtype_list,
    #                 batchsize_list, context_list)
