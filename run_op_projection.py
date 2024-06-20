from tqdm import tqdm
import helper
import config
import compute


class Analyzer:
    def __init__(self, proj_cfg) -> None:
        self.op_list = proj_cfg["operation"]
        self.device_list = proj_cfg["device_list"]
        self.type_list = proj_cfg["type_list"]
        self.dtype_list = proj_cfg["dtype_list"]
        self.input = proj_cfg["input"]
        self.m_list = self.input["m"]
        self.n_list = self.input["n"]
        self.k_list = self.input["k"]

    def analyze(self, to_csv=True, plot=False):
        for op_name in self.op_list:
            proj_dict = {}

            for device in self.device_list:
                proj_dict[device] = {}

                for type in self.type_list:
                    proj_dict[device][type] = {}

                    for dtype in self.dtype_list:
                        proj_dict[device][type][dtype] = []

                        for m in self.m_list:
                            for k in self.k_list:
                                for n in self.n_list:
                                    proj_rst = compute.do_op_projection(
                                        op_name, device, type, dtype, m=m, n=n, k=k)
                                    proj_dict[device][type][dtype].append(
                                        proj_rst)

            helper.print_matmul_projection(op_name, proj_dict, to_csv)


if __name__ == "__main__":
    proj_cfg = {
        "operation": ["Matmul"],
        "device_list": ["IntelGaudi2"],
        "type_list": ["B"],  # ["C", "D"],
        "dtype_list": ["BF16"],
        "input": {
            "m": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
            "n": [14336],
            "k": [4096],
        },
    }

    analyzer = Analyzer(proj_cfg)
    analyzer.analyze(True, False)
