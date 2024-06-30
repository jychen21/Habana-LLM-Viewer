import argparse
import tabulate
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

    def analyze_bs(self, model, device, type, pp, tp, dtype, input, output):
        proj_bs = []
        for bs in tqdm(self.bs_list):
            compute_projection = compute.do_model_projection(
                model, device, type, pp, tp, dtype, input, output, bs, self.kvcache_bucket, enable_vec_bmm=self.enable_vec_bmm)
            memory_projection = memory.do_model_projection(
                model, device, type, pp, tp, dtype, input, output, bs, self.kvcache_bucket, enable_vec_bmm=self.enable_vec_bmm)
            proj_rst = {
                "compute": compute_projection, "memory": memory_projection}
            proj_bs.append((bs, proj_rst))
        return proj_bs

    def analyze_output(self, model, device, type, pp, tp, dtype, input):
        proj_output = {}
        for output in self.output_list:
            proj_output[output] = self.analyze_bs(
                model, device, type, pp, tp, dtype, input, output)
        return proj_output

    def analyze_input(self, model, device, type, pp, tp, dtype):
        proj_input = {}
        for input in self.input_list:
            proj_input[input] = self.analyze_output(
                model, device, type, pp, tp, dtype, input)
        return proj_input

    def analyze_dtype(self, model, device, type, pp, tp):
        proj_dtype = {}
        for dtype in self.dtype_list:
            proj_dtype[dtype] = self.analyze_input(
                model, device, type, pp, tp, dtype)
        return proj_dtype

    def analyze_tp(self, model, device, type, pp):
        proj_tp = {}
        for tp in self.tp_list:
            proj_tp[tp] = self.analyze_dtype(model, device, type, pp, tp)
        return proj_tp

    def analyze_pp(self, model, device, type):
        proj_pp = {}
        for pp in self.pp_list:
            proj_pp[pp] = self.analyze_tp(model, device, type, pp)
        return proj_pp

    def analyze_type(self, model, device):
        proj_type = {}
        for type in self.type_list:
            proj_type[type] = self.analyze_pp(model, device, type)
        return proj_type

    def analyze_device(self, model):
        proj_device = {}
        for device in self.device_list:
            proj_device[device] = self.analyze_type(model, device)
        return proj_device

    def analyze_model(self, print_proj=True, to_csv=False, plot=False):
        proj_model = {}
        for model in self.model_list:
            proj_model[model] = self.analyze_device(model)
            if print_proj:
                helper.print_projection(
                    model, proj_model[model], self.kvcache_bucket, self.bs_list, to_csv, plot)
        return proj_model

    def analyze(self, print_proj=True, to_csv=False, plot=False):
        return self.analyze_model(print_proj, to_csv, plot)


def main(device, device_type, model, data_type, batch_size, context_input,
         context_output, kvcache_bucket, enable_vec_bmm):
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
            "enable_vec_bmm": enable_vec_bmm,
        }
    }
    '''
    proj_cfg = {
        "device_list": ["IntelGaudi2"],
        "type_list": ["D"],  # ["C", "D"],
        "model_list": ["Llama2-13B"],
        "dtype_list": ["BF16"],
        "parallel": {
            "pp_list": [1],
            "tp_list": [1],
        },
        "context": {
            "input_list": [1024],
            "output_list": [2048],
        },
        "bs_list": [1, 2, 4, 8, 16, 32, 64, 128, 256],
        "optims": {
            "kvcache_bucket": 256,
            "enable_vec_bmm": True,
        }
    }
    '''
    analyzer = Analyzer(proj_cfg)
    analyzer.analyze()


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
    parser.add_argument("--vec-bmm", action="store_true")
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
        enable_vec_bmm=args.vec_bmm,
    )
