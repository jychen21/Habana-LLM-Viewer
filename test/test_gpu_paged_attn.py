# Refer to
# https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_paged_attention.py

import argparse
import random
import time
from typing import Optional, List, Tuple, Union
import csv

import torch

from vllm import _custom_ops as ops
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, create_kv_caches_with_random

NUM_BLOCKS = 1024
PARTITION_SIZE = 512


def run_cuda_benchmark(version: str, input: tuple, num_iters: int, profile: bool = False) -> float:
    output, query, key_cache, value_cache, num_kv_heads, scale, block_tables, seq_lens, block_size, \
        max_seq_len, alibi_slopes, kv_cache_dtype, exp_sums, max_logits, tmp_output = input

    torch.cuda.synchronize()
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
    start_time = time.perf_counter()

    # Using default kv_scale
    kv_scale = 1.0

    if version == "v1":
        for _ in range(num_iters):
            ops.paged_attention_v1(
                output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                kv_scale,
            )
    elif version == "v2":
        for _ in range(num_iters):
            ops.paged_attention_v2(
                output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                scale,
                block_tables,
                seq_lens,
                block_size,
                max_seq_len,
                alibi_slopes,
                kv_cache_dtype,
                kv_scale,
            )
    else:
        raise ValueError(f"Invalid version: {version}")
    torch.cuda.synchronize()

    end_time = time.perf_counter()
    if profile:
        torch.cuda.cudart().cudaProfilerStart()
    
    bs = query.size(0)
    seq_len = max_seq_len
    heads_q = query.size(1)
    heads_kv = key_cache.size(1)
    head_size = query.size(2)
    block_size = key_cache.size(-1)
    type = "MHA" if heads_q == heads_kv else (
        "GQA" if heads_kv != 1 else "MQA")
    duration = round((end_time - start_time) / num_iters * 1e6, 3)

    print(f"{duration} us")

    return (type, bs, seq_len, heads_q, heads_kv, head_size, duration)


@torch.inference_mode()
def main(
    version: str,
    num_seqs_list: List,
    seq_len_list: int,
    num_query_heads_list: int,
    num_kv_heads_list: int,
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    do_profile: bool,
    device: str = "cuda",
    kv_cache_dtype: Optional[str] = None,
    num_iters: int = 50,
) -> None:
    random.seed(seed)
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    input_dict = {}
    # heads
    for i, num_query_heads, num_kv_heads in zip(range(len(num_query_heads_list)), num_query_heads_list, num_kv_heads_list):
        query_key = f"{i}_{num_query_heads}"
        kv_key = f"{i}_{num_kv_heads}"
        input_dict[query_key] = {}
        for seq_len in seq_len_list:  # kv length
            if seq_len not in input_dict[query_key].keys():
                input_dict[query_key][seq_len] = {}
                if kv_key not in input_dict[query_key][seq_len].keys():
                    input_dict[query_key][seq_len][kv_key] = []
            for num_seqs in num_seqs_list:  # batch_size
                scale = float(1.0 / (head_size**0.5))
                query = torch.empty(num_seqs,
                                    num_query_heads,
                                    head_size,
                                    dtype=dtype,
                                    device=device)
                query.uniform_(-scale, scale)

                assert num_query_heads % num_kv_heads == 0
                alibi_slopes = None
                if use_alibi:
                    alibi_slopes = torch.randn(num_query_heads,
                                            dtype=torch.float,
                                            device=device)

                seq_lens = [seq_len for _ in range(num_seqs)]
                max_seq_len = max(seq_lens)
                seq_lens = torch.tensor(seq_lens, dtype=torch.int, device=device)

                # Create the block tables.
                max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
                block_tables = []
                for _ in range(num_seqs):
                    block_table = [
                        random.randint(0, NUM_BLOCKS - 1)
                        for _ in range(max_num_blocks_per_seq)
                    ]
                    block_tables.append(block_table)
                block_tables = torch.tensor(block_tables, dtype=torch.int, device=device)

                # Create the KV cache.
                key_caches, value_caches = create_kv_caches_with_random(NUM_BLOCKS,
                                                                        block_size,
                                                                        1,
                                                                        num_kv_heads,
                                                                        head_size,
                                                                        kv_cache_dtype,
                                                                        dtype,
                                                                        device=device)
                key_cache, value_cache = key_caches[0], value_caches[0]

                # Prepare for the paged attention kernel.
                output = torch.empty_like(query)
                input_dict[query_key][seq_len][kv_key].append([output, 
                                                               query, 
                                                               key_cache, 
                                                               value_cache,
                                                               num_kv_heads,
                                                               scale,
                                                               block_tables,
                                                               seq_lens,
                                                               block_size,
                                                               max_seq_len,
                                                               alibi_slopes,
                                                               kv_cache_dtype])
                if version == "v2":
                    num_partitions = ((max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE)
                    tmp_output = torch.empty(
                        size=(num_seqs, num_query_heads, num_partitions, head_size),
                        dtype=output.dtype,
                        device=output.device,
                    )
                    exp_sums = torch.empty(
                        size=(num_seqs, num_query_heads, num_partitions),
                        dtype=torch.float32,
                        device=output.device,
                    )
                    max_logits = torch.empty_like(exp_sums)
                    input_dict[query_key][seq_len][kv_key][-1].extend([exp_sums, max_logits, tmp_output])

    print(f"Warming up...")
    for q, q_dict in input_dict.items():
        print(f"heads_q={q[2:]}...")
        for seq_len, s_dict in q_dict.items():
            print(f"......seq_len={seq_len}...")
            for k, k_list in s_dict.items():
                print(f".........heads_kv={k[2:]}...")
                for input in k_list:
                    print(f"............bs={input[0].size(0)}...")
                    run_cuda_benchmark(version, input, num_iters=3, profile=False)

    if do_profile:
        print(f"Profiling Benchmark...")
        for i in range(num_iters):
            print(f"Iteration_{i} Start")
            for q, q_dict in input_dict.items():
                print(f"heads_q={q[2:]}...")
                for seq_len, s_dict in q_dict.items():
                    print(f"......seq_len={seq_len}...")
                    for k, k_list in s_dict.items():
                        print(f".........heads_kv={k[2:]}...")
                        for input in k_list:
                            print(f"............bs={input[0].size(0)}...")
                            run_cuda_benchmark(
                                version, input, num_iters=100, profile=False)
        print("Profiling Benchmark Finished!")
    else:
        print(f"Pure Benchmark...")
        item_list = [["Type", "Dtype", "BS", "Q_Length",
                      "KV_Length", "HeadDim", "Q_Head", "KV_Head", "A100 (us)"]]
        for q, q_dict in input_dict.items():
            print(f"heads_q={q[2:]}...")
            for seq_len, s_dict in q_dict.items():
                print(f"......seq_len={seq_len}...")
                for k, k_list in s_dict.items():
                    print(f".........heads_kv={k[2:]}...")
                    for input in k_list:
                        print(f"............bs={input[0].size(0)}...")
                        type, bs, seq_len, heads_q, heads_kv, head_size, duration = \
                            run_cuda_benchmark(version, input, num_iters=100, profile=False)
                        item_list.append(
                            [type, "FP16/BF16", bs, 1, seq_len, head_size, heads_q, heads_kv, duration])
        path = f"./benchmark_gpu_pa_{version}.csv"
        with open(path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(item_list)
        print(f"Pure Benchmark Finished! Please check data stored in {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument("--version",
                        type=str,
                        choices=["v1", "v2"],
                        default="v2")
    parser.add_argument("--seq-len-list",
                        nargs='+',
                        type=int,
                        default=[512, 1024, 2048, 4096, 8192])
    parser.add_argument("--num-query-heads-list",
                        nargs='+',
                        type=int,
                        default=[32, 32, 32, 32, 32, 8, 16, 32])
    parser.add_argument("--num-kv-heads-list",
                        nargs='+',
                        type=int,
                        default=[32, 2, 4, 8, 16, 1, 1, 1])
    parser.add_argument("--batch-size-list",
                        nargs='+',
                        type=int,
                        default=[1, 4, 16, 64, 256])
    parser.add_argument("--head-size",
                        type=int,
                        choices=[64, 80, 96, 112, 128, 192, 256],
                        default=128)
    parser.add_argument("--block-size", type=int, choices=[16, 32], default=16)
    parser.add_argument("--use-alibi", action="store_true")
    parser.add_argument("--dtype",
                        type=str,
                        choices=["half", "bfloat16", "float"],
                        default="half")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        default="auto",
        help="Data type for kv cache storage. If 'auto', will use model "
        "data type. CUDA 11.8+ supports fp8 (=fp8_e4m3) and fp8_e5m2. "
        "ROCm (AMD GPU) supports fp8 (=fp8_e4m3)")
    args = parser.parse_args()
    print(args)

    for num_query_heads, num_kv_heads in zip(args.num_query_heads_list, args.num_kv_heads_list):
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")
    main(
        version=args.version,
        num_seqs_list=args.batch_size_list,
        seq_len_list=args.seq_len_list,
        num_query_heads_list=args.num_query_heads_list,
        num_kv_heads_list=args.num_kv_heads_list,
        head_size=args.head_size,
        block_size=args.block_size,
        use_alibi=args.use_alibi,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        seed=args.seed,
        do_profile=args.profile,
        kv_cache_dtype=args.kv_cache_dtype,
    )
