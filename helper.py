import json


proj_cfg = {
    "model_list": ["Llama2-7B"],
    "device_list": ["IntelGaudi2"],
    "type_list": ["C", "D"],
    "parallel": {
        "pp_list": [1],
        "tp_list": [1],
    },
    "dtype_list": ["BF16"],
    "context": {
        "input_list": [128, 512, 1024, 2048],
        "output_list": [1, 128, 512, 1024],
    },
    "bs_list": [1] + [i for i in range(2, 513, 2)],
    "optims": {
        "kvcache_bucket": 128
    }
}


def dump_json(path, data):
    with open(path, "w") as fp:
        json_string = json.dumps(data, default=lambda o: o.__dict__, sort_keys=False, indent=4)
        fp.write(json_string)