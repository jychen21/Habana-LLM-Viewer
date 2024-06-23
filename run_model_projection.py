import argparse
import tabulate
from tqdm import tqdm
import helper
import config
import compute
import memory


class WebAnalyzer:
    def __init__(self, proj_cfg) -> None:
        self.device = proj_cfg["device_list"][0]
        self.type = proj_cfg["type_list"][0]
        self.model = proj_cfg["model_list"][0]
        self.dtype = proj_cfg["dtype_list"][0]
        self.pp = proj_cfg["parallel"]["pp_list"][0]
        self.tp = proj_cfg["parallel"]["tp_list"][0]
        self.bs = proj_cfg["bs_list"][0]
        self.input = proj_cfg["context"]["input_list"][0]
        self.output = proj_cfg["context"]["output_list"][0]
        opt_config = proj_cfg.get("optims", {})
        self.kvcache_bucket = opt_config.get("kvcache_bucket", False)
        self.enable_vec_bmm = opt_config.get("enable_vec_bmm", False)

    def create_table(self, proj_rst):
        proj_item = ["Model", "Device", "PP", "TP", "DType", "Input", "Output", "BS", "KVCacheBucket", "AI", "Prefill(ms)",
                     "DecodeMin(ms)", "DecodeMax(ms)", "DecodeAvg(ms)", "Latency(ms)", "Throughput(tokens/sec)"]
        proj_data = [proj_item]
        compute = proj_rst["compute"]
        arithmetic_intensity = compute["arithmetic_intensity"]
        mem_consumed = proj_rst["memory"]["size"]
        proj_prefill_step = compute["prefill"]
        proj_decode_steps = compute["decode"]
        prefill_latency = round(
            proj_prefill_step[0] * config.MilliSecs, 2)
        decode_latency_list = [step[0]
                               for step in proj_decode_steps]
        decode_latency_min = round(
            decode_latency_list[0] * config.MilliSecs, 2)
        decode_latency_max = round(
            decode_latency_list[-1] * config.MilliSecs, 2)
        decode_latency_avg = round(
            sum(decode_latency_list)/len(decode_latency_list) * config.MilliSecs, 2)
        overall_latency = round(((proj_prefill_step[0] + sum(decode_latency_list)) /
                                (len(decode_latency_list) + 1)) * config.MilliSecs, 2)
        attainable_tops = round(
            (1 / overall_latency) * self.bs * config.MilliSecs, 2)
        proj_data.append([self.model, f"{self.device}{self.type}", self.pp, self.tp, self.dtype, self.input,
                          self.output, self.bs, self.kvcache_bucket, arithmetic_intensity, prefill_latency,
                          decode_latency_min, decode_latency_max, decode_latency_avg, overall_latency,
                          attainable_tops if mem_consumed != "OOM" else "OOM"])

        return proj_data

    def analyze(self):
        compute_projection = compute.do_model_projection(self.model, self.device, self.type, self.pp, self.tp, self.dtype, self.input,
                                                         self.output, self.bs, self.kvcache_bucket, enable_vec_bmm=self.enable_vec_bmm)
        memory_projection = memory.do_model_projection(self.model, self.device, self.type, self.pp, self.tp, self.dtype, self.input,
                                                       self.output, self.bs, self.kvcache_bucket, enable_vec_bmm=self.enable_vec_bmm)
        proj_rst = {"compute": compute_projection, "memory": memory_projection}

        return self.create_table(proj_rst)


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

    # def analyze_model(self, model_name):

    # def analyze_device(self, device):

    # def analyze_device_type(self, device_type):

    def analyze(self, to_csv=True, plot=True):
        proj_model = {}
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
                                            compute_projection = compute.do_model_projection(
                                                model_name, device, type, pp, tp, dtype, input, output, bs, self.kvcache_bucket, enable_vec_bmm=self.enable_vec_bmm)
                                            memory_projection = memory.do_model_projection(
                                                model_name, device, type, pp, tp, dtype, input, output, bs, self.kvcache_bucket, enable_vec_bmm=self.enable_vec_bmm)
                                            proj_rst = {
                                                "compute": compute_projection, "memory": memory_projection}
                                            proj_dict[device][type][pp][tp][dtype][input][output].append(
                                                (bs, proj_rst))

            helper.print_projection(
                model_name, proj_dict, self.kvcache_bucket, self.bs_list, to_csv, plot)
            proj_model[model_name] = proj_dict

        return proj_model


def main(device, device_type, model, data_type, batch_size,
         context_input, context_output, kvcache_bucket):
    proj_cfg = {
        "device_list": [device],
        "type_list": [device_type],
        "model_list": [model],
        "dtype_list": [data_type],
        "parallel": {
            "pp_list": [1],
            "tp_list": [1],
        },
        "context": {
            "input_list": [context_input],
            "output_list": [context_output],
        },
        "bs_list": [batch_size],
        "optims": {
            "kvcache_bucket": kvcache_bucket,
            "flash_attention": False,  # Todo
            "enable_vec_bmm": True,
        }
    }
    analyzer = Analyzer(proj_cfg)
    analyzer.analyze(False, False)

    # proj_cfg = {
    #     "device_list": ["IntelGaudi2"],
    #     "type_list": ["B"],  # ["C", "D"],
    #     # ["Llama2-7B", "Llama2-13B", "Mixtral-8x7B", "GLaM-1.2T"]
    #     "model_list": ["Llama2-7B", "Llama3-8B"],
    #     "dtype_list": ["BF16"],
    #     "parallel": {
    #         "pp_list": [1],
    #         "tp_list": [1],  # [1, 2, 4, 8, 16]
    #     },
    #     "context": {
    #         "input_list": [512, 1024, 2048],  # 32000
    #         "output_list": [512],
    #     },
    #     # [1] + [i for i in range(2, 257, 2)],
    #     "bs_list": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    #     "optims": {
    #         "kvcache_bucket": 256,  # None, 1, or >= 256
    #         "flash_attention": False,  # Todo
    #         "enable_vec_bmm": True,
    #     }
    # }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Habana-Viewer Projection.")
    parser.add_argument("--device",
                        type=str,
                        choices=list(config.HardwareParameters.keys()),
                        default="IntelGaudi2")
    parser.add_argument("--device-type",
                        type=str,
                        choices=list(config.DeviceType2Ratio.keys()),
                        default="B")
    parser.add_argument("--model",
                        type=str,
                        choices=list(config.ModelDict.keys()),
                        default="Llama2-13B")
    parser.add_argument("--data-type",
                        type=str,
                        choices=list(config.DType2Bytes.keys()),
                        default="BF16")
    parser.add_argument("--batch-size",
                        type=int,
                        default=32)
    parser.add_argument("--context-input",
                        type=int,
                        default=512)
    parser.add_argument("--context-output",
                        type=int,
                        default=1024)
    parser.add_argument("--kvcache-bucket",
                        type=int,
                        choices=[256, 512, 1024],
                        default=256)
    args = parser.parse_args()

    main(
        device=args.device,
        device_type=args.device_type,
        model=args.model,
        data_type=args.data_type,
        batch_size=args.batch_size,
        context_input=args.context_input,
        context_output=args.context_output,
        kvcache_bucket=args.kvcache_bucket,
    )
