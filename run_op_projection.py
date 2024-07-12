from tqdm import tqdm
import csv
import argparse
import tabulate
from abc import ABC, abstractmethod
from scripts import helper, config, compute


class Analyzer:
    def __init__(self, proj_cfg) -> None:
        self.op = proj_cfg["operation"][0]
        self.device_list = proj_cfg["device_list"]
        self.type_list = proj_cfg["type_list"]
        self.dtype_list = proj_cfg["dtype_list"]
        self.input = proj_cfg["input"]

    @abstractmethod
    def analyze_input(self, device, type, dtype):
        pass

    def analyze_dtype(self, device, type):
        proj_dtype = {}
        for dtype in self.dtype_list:
            proj_dtype[dtype] = self.analyze_input(device, type, dtype)
        return proj_dtype

    def analyze_type(self, device):
        proj_type = {}
        for type in self.type_list:
            proj_type[type] = self.analyze_dtype(device, type)
        return proj_type

    def analyze_device(self):
        proj_device = {}
        for device in self.device_list:
            proj_device[device] = self.analyze_type(device)
        return proj_device

    def analyze_op(self, to_csv=False):
        proj_op = self.analyze_device()
        self.print_projection(proj_op, to_csv)
        return proj_op

    def analyze(self, to_csv=False):
        return self.analyze_op(to_csv)

    @abstractmethod
    def print_projection(self, proj_op, to_csv=False):
        pass


class MatmulAnalyzer(Analyzer):
    def __init__(self, proj_cfg) -> None:
        super().__init__(proj_cfg)
        self.m_list = self.input["m"]
        self.n_list = self.input["n"]
        self.k_list = self.input["k"]

    def analyze_input(self, device, type, dtype):
        proj_mnk = []
        for m in self.m_list:
            for k in self.k_list:
                for n in self.n_list:
                    proj_rst = compute.do_op_projection(
                        self.op, device, type, dtype, m=m, n=n, k=k)
                    proj_mnk.append(proj_rst)
        return proj_mnk

    def print_projection(self, proj_op, to_csv=False):
        helper.print_matmul_projection(self.op, proj_op, to_csv)


class FlashAttnAnalyzer(Analyzer):
    def __init__(self, proj_cfg) -> None:
        super().__init__(proj_cfg)
        self.heads_q = self.input["heads_q"]
        self.heads_kv = self.input["heads_kv"]
        self.hidden_size = self.input["hidden_size"]
        self.seq_len_kv_list = self.input["seq_len_kv"]
        self.seq_len_q = 1
        self.batch_size_list = self.input["batch_size"]

    def analyze_input(self, device, type, dtype):
        proj_fa = []
        for seq_len_kv in self.seq_len_kv_list:
            for batch_size in self.batch_size_list:
                proj_rst = compute.do_op_projection(
                    self.op, device, type, dtype, heads_q=self.heads_q,
                    heads_kv=self.heads_kv, hidden_size=self.hidden_size,
                    batch_size=batch_size, seq_len_q=self.seq_len_q,
                    seq_len_kv=seq_len_kv)
                proj_fa.append(proj_rst)
        return proj_fa

    def print_projection(self, proj_op, to_csv=False):
        helper.print_flashattn_projection(self.op, proj_op, to_csv)


class PagedAttnAnalyzer(Analyzer):
    def __init__(self, proj_cfg) -> None:
        super().__init__(proj_cfg)

    def analyze_input(self, device, type, dtype):
        proj_pa = []
        # for m in self.m_list:
        #     for k in self.k_list:
        #         for n in self.n_list:
        #             proj_rst = compute.do_op_projection(
        #                 self.op, device, type, dtype, m=m, n=n, k=k)
        #             proj_pa.append(proj_rst)
        return proj_pa

    def print_projection(self, proj_op, to_csv=False):
        helper.print_pagedattn_projection(self.op, proj_op, to_csv)


Analyzer_Mapping = {
    "Matmul": MatmulAnalyzer,
    "FlashAttentionV1": FlashAttnAnalyzer,
    "PagedAttentionV1": PagedAttnAnalyzer,
}


def main(device, device_type, op, op_version, data_type,
         batch_size_list, m_list, n_list, k_list):
    proj_cfg_matmul = {
        "operation": [op],
        "op_version": [op_version],  # v2""
        "device_list": [device],
        "type_list": [device_type],  # ["C", "D"],
        "dtype_list": [data_type],
        "input": {
            "m": m_list,
            "n": n_list,
            "k": k_list,
        },
    }
    '''
    proj_cfg_matmul = {
        "operation": ["Matmul"],
        "op_version": ["v1"], # v2""
        "device_list": ["IntelGaudi2"],
        "type_list": ["B"],  # ["C", "D"],
        "dtype_list": ["BF16"],
        "input": {
            "m": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            "n": [14336],
            "k": [4096],
        },
    }
    '''
    analyzer = Analyzer_Mapping[proj_cfg_matmul["operation"]
                                [-1]](proj_cfg_matmul)
    analyzer.analyze(False)

    # proj_cfg_flashattn = {
    #     "operation": ["FlashAttentionV1"],
    #     "op_version": ["v1"],  # v2""
    #     "device_list": ["IntelGaudi2"],
    #     "type_list": ["D"],  # ["C", "D"],
    #     "dtype_list": ["BF16"],
    #     "input": {
    #         "heads_q": 32,
    #         "heads_kv": 32,
    #         "hidden_size": 4096,
    #         "seq_len_kv": [512, 1024, 2048, 4096, 8192],
    #         "batch_size": [64, 128, 256],
    #     },
    # }

    # analyzer = Analyzer_Mapping[proj_cfg_flashattn["operation"][-1]](proj_cfg_flashattn)
    # analyzer.analyze(False)


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
    parser.add_argument("--op",
                        type=str,
                        choices=list(Analyzer_Mapping.keys()),
                        default="Matmul")
    parser.add_argument("--op_version",
                        type=str,
                        choices=["v1"],
                        default="v1")
    parser.add_argument("--data-type",
                        type=str,
                        choices=list(config.DType2Bytes.keys()),
                        default="BF16")
    # Matmul
    parser.add_argument("--m-list",
                        nargs='+',
                        type=int,
                        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    parser.add_argument("--n-list",
                        nargs='+',
                        type=int,
                        default=[14336])
    parser.add_argument("--k-list",
                        nargs='+',
                        type=int,
                        default=[4096])
    # Attn
    parser.add_argument("--batch-size-list",
                        nargs='+',
                        type=int,
                        default=[1, 4, 16, 64, 256])
    parser.add_argument("--seq-len-list",
                        nargs='+',
                        type=int,
                        default=[512, 2048, 8192])
    parser.add_argument("--num-query-heads-list",
                        nargs='+',
                        type=int,
                        default=[32, 32, 32, 32, 32, 8, 16, 32])
    parser.add_argument("--num-kv-heads-list",
                        nargs='+',
                        type=int,
                        default=[32, 2, 4, 8, 16, 1, 1, 1])
    parser.add_argument("--head-size",
                        type=int,
                        choices=[128],
                        default=128)
    parser.add_argument("--block-size",
                        type=int,
                        choices=[128, 256],
                        default=128)
    parser.add_argument("--vec-bmm", action="store_true")
    args = parser.parse_args()

    main(
        device=args.device,
        device_type=args.device_type,
        op=args.op,
        op_version=args.op_version,
        data_type=args.data_type,
        batch_size_list=args.batch_size_list,
        m_list=args.m_list,
        n_list=args.n_list,
        k_list=args.k_list,
    )
