from tqdm import tqdm
import helper
import config
import compute
import memory


class Analyzer:
    def __init__(self, proj_cfg) -> None:
        self.model_list = proj_cfg["model_list"]
        self.device_list = proj_cfg["device_list"]
        self.type_list = proj_cfg["type_list"]
        self.pp_list = proj_cfg["parallel"]["pp_list"]
        self.tp_list = proj_cfg["parallel"]["tp_list"]
        self.dtype_list = proj_cfg["dtype_list"]
        self.input_list = proj_cfg["context"]["input_list"]
        self.output_list = proj_cfg["context"]["output_list"]
        self.bs_list = proj_cfg["bs_list"]
        opt_config = proj_cfg.get("optims", {})
        self.kvcache_bucket = opt_config.get("kvcache_bucket", False)
        self.enable_vec_bmm = opt_config.get("enable_vec_bmm", False)

    def analyze(self, to_csv=True, plot=True):
        for model_name in self.model_list:
            proj_dict = {}

            for device in self.device_list:
                proj_dict[device] = {}

                for type in self.type_list:
                    proj_dict[device][type] = {}

                    for pp in self.pp_list:
                        proj_dict[device][type][pp] = {}

                        for tp in self.tp_list:
                            proj_dict[device][type][pp][tp] = {}

                            for dtype in self.dtype_list:
                                proj_dict[device][type][pp][tp][dtype] = {}

                                for input in self.input_list:
                                    proj_dict[device][type][pp][tp][dtype][input] = {
                                    }

                                    for output in self.output_list:
                                        proj_dict[device][type][pp][tp][dtype][input][output] = [
                                        ]

                                        for bs in tqdm(self.bs_list):
                                            compute_projection = compute.do_projection(
                                                model_name, device, type, pp, tp, dtype, input, output, bs, self.kvcache_bucket, enable_vec_bmm=self.enable_vec_bmm)
                                            memory_projection = memory.do_projection(
                                                model_name, device, type, pp, tp, dtype, input, output, bs, self.kvcache_bucket, enable_vec_bmm=self.enable_vec_bmm)
                                            proj_rst = {
                                                "compute": compute_projection, "memory": memory_projection}
                                            proj_dict[device][type][pp][tp][dtype][input][output].append(
                                                (bs, proj_rst))

            helper.print_projection(
                model_name, proj_dict, self.kvcache_bucket, self.bs_list, to_csv, plot)


if __name__ == "__main__":
    proj_cfg = {
        # ["Llama2-7B", "Llama2-13B", "Mixtral-8x7B", "GLaM-1.2T"]
        "model_list": ["Llama2-7B", "Llama3-8B"],
        "device_list": ["IntelGaudi2"],
        "type_list": ["B"],  # ["C", "D"],
        "parallel": {
            "pp_list": [1],
            "tp_list": [1],  # [1, 2, 4, 8, 16]
        },
        "dtype_list": ["BF16"],
        "context": {
            "input_list": [512, 1024, 2048],  # 32000
            "output_list": [512],
        },
        # [1] + [i for i in range(2, 257, 2)],
        "bs_list": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
        "optims": {
            "kvcache_bucket": 256,  # None, 1, or >= 256
            "flash_attention": False,  # Todo
            "enable_vec_bmm": True,
        }
    }

    analyzer = Analyzer(proj_cfg)
    analyzer.analyze(True, True)
