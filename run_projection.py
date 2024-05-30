from tqdm import tqdm
import helper
import config
import compute
import memory


def compute_analyzer(proj_cfg, to_csv=True, plot=True):
    kvcache_bucket = proj_cfg["optims"]["kvcache_bucket"]

    for model_name in proj_cfg["model_list"]:
        proj_dict = {}

        for device in proj_cfg["device_list"]:
            proj_dict[device] = {}

            for type in proj_cfg["type_list"]:
                proj_dict[device][type] = {}

                for pp in proj_cfg["parallel"]["pp_list"]:
                    proj_dict[device][type][pp] = {}

                    for tp in proj_cfg["parallel"]["tp_list"]:
                        proj_dict[device][type][pp][tp] = {}

                        for dtype in proj_cfg["dtype_list"]:
                            proj_dict[device][type][pp][tp][dtype] = {}

                            for input in proj_cfg["context"]["input_list"]:
                                proj_dict[device][type][pp][tp][dtype][input] = {}

                                for output in proj_cfg["context"]["output_list"]:
                                    proj_dict[device][type][pp][tp][dtype][input][output] = [
                                    ]

                                    for bs in tqdm(proj_cfg["bs_list"]):
                                        proj_rst = compute.do_projection(model_name, device, type, pp, tp, dtype,
                                                                         input, output, bs, kvcache_bucket)
                                        proj_dict[device][type][pp][tp][dtype][input][output].append(
                                            (bs, proj_rst))

        compute.print_projection(
            model_name, proj_dict, kvcache_bucket, proj_cfg["bs_list"], to_csv, plot)


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
                            cfg = config.Config(device=device, type=type, dtype=dtype, pp=pp, tp=tp, hidden_size=hidden_size,
                                                num_heads_q=num_heads_q, num_heads_kv=num_heads_kv,
                                                intermediate_size=intermediate_size, mlp_with_gate=mlp_with_gate,
                                                num_experts=num_experts, num_layers_mlp=num_layers_mlp,
                                                num_layers_moe=num_layers_moe, seq_len_q=in_out["in"],
                                                seq_len_kv=in_out["in"]+in_out["out"], batch_size=1,
                                                is_decoding=False, kvcache_bucket=False)
                            model_config = cfg.model_config
                            mem_persist_weight = memory.mem_persistent_weights(
                                cfg)
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
                                cfg.model_config.batch_size = bs
                                mem_decoder_data = memory.mem_decoder(cfg)
                                memory_dict[pp][tp][dtype].append(
                                    mem_decoder_data)
                        print("done!\n")

            memory.print_mem_analysis(memory_dict, batchsize_list)
            # print_projected_mem_per_device(
            #     model_name, memory_dict, batchsize_list, context_list)


if __name__ == "__main__":
    proj_cfg = {
        # ["Llama2-7B", "Llama2-13B", "Mixtral-8x7B", "GLaM-1.2T", "MoE-1.8T"]
        "model_list": ["Llama2-7B", "Mixtral-8x7B"],
        "device_list": ["IntelGaudi2"],
        "type_list": ["C"],  # ["C", "D"],
        "parallel": {
            "pp_list": [1],
            "tp_list": [1],  # [1, 2, 4, 8, 16]
        },
        "dtype_list": ["BF16"],
        "context": {
            "input_list": [128, 512, 1024, 2048],  # 32000
            "output_list": [128, 512],
        },
        # [1, 2, 4, 8, 16, 32, 64, 128, 129, 160, 192, 256, 257, 512]  # 129
        # [1] + [i for i in range(2, 513, 2)],
        # [1] + [i for i in range(2, 257, 2)],
        "bs_list": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        "optims": {
            "kvcache_bucket": 256,  # None, 1, or >= 256
            "flash_attention": False,  # Todo
        }
    }

    # compute_analyzer(model_name, device_list, type_list, pp_list, tp_list,
    #                  dtype_list, input_list, output_list, batchsize_list)
    compute_analyzer(proj_cfg)

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
