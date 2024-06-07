# Refer to
# https://github.com/HabanaAI/vllm-fork/blob/habana_main/benchmarks/kernels/benchmark_paged_attention.py

import os
os.environ["PA_SPLIT_VALUE"]="1"
import argparse
import random
import time
from typing import Optional, List, Tuple, Union

import torch

from vllm.hpu import ops as hpu_ops
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, get_kv_cache_torch_dtype

import habana_frameworks.torch as htorch


NUM_BLOCKS = 1024
PARTITION_SIZE = 512


def create_kv_caches_with_random_hpu(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "hpu",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    torch.random.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    kv_cache_shape = (num_blocks, num_heads, head_size, block_size)
    key_caches = []
    value_caches = []
    for _ in range(num_layers):
        key_cache = torch.zeros(kv_cache_shape,
                    dtype=torch_dtype,
                    device=device)
        value_cache = torch.zeros(kv_cache_shape,
                dtype=torch_dtype,
                device=device)
        key_caches.append(key_cache)
        value_caches.append(value_cache)

    return key_caches, value_caches


class HPUBenchmarkModel(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, query, key_cache, value_cache, num_kv_heads, scale, block_tables,
                seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype=None):
        output = hpu_ops.paged_attention_v1(
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
        )

        return output


model = HPUBenchmarkModel()
model.eval()
import habana_frameworks.torch.hpu.graphs as htgraphs
htgraphs.wrap_in_hpu_graph(model)


def run_hpu_benchmark(input: tuple) -> float:
    query, key_cache_hpu, value_cache_hpu, num_kv_heads, scale, block_tables, \
        seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype = input

    htorch.core.mark_step()
    torch.hpu.synchronize()
    start_time = time.perf_counter()
    model(
        query,
        key_cache_hpu,
        value_cache_hpu,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
    )
    htorch.core.mark_step()
    torch.hpu.synchronize()
    end_time = time.perf_counter()
    print(f"version=v1, batch_size={query.size(0)}, seq_len={max_seq_len}, num_query_heads={query.size(1)}, num_kv_heads={key_cache_hpu.size(1)}, head_size={query.size(2)}, block_size={key_cache_hpu.size(-1)}, running time={(end_time - start_time) * 1e6:.3f}us")


@torch.inference_mode()
def main(
    version: str,
    num_seqs_list: List,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    seed: int,
    do_profile: bool,
    device: str = "hpu",
    kv_cache_dtype: Optional[str] = None,
    num_iters: int = 10,
    profiling: bool = True
) -> None:
    input_list = []
    for num_seqs in num_seqs_list:
        random.seed(seed)
        torch.random.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

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
        key_caches_hpu, value_caches_hpu = create_kv_caches_with_random_hpu(NUM_BLOCKS,
                                                                            block_size,
                                                                            1,
                                                                            num_kv_heads,
                                                                            head_size,
                                                                            kv_cache_dtype,
                                                                            dtype,
                                                                            device=device)
        key_cache_hpu, value_cache_hpu = key_caches_hpu[0], value_caches_hpu[0]
        input_list.append((query, key_cache_hpu, value_cache_hpu, num_kv_heads, scale,
                           block_tables, seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype))

    attn_type = "MHA" if query.size(1) == key_cache_hpu.size(1) else "GQA" # just ignore mqa

    print(f"Warming up {attn_type} BS={num_seqs_list}...")
    for input in input_list:
        run_hpu_benchmark(input)
        htorch.core.mark_step()
        torch.hpu.synchronize()

    print(f"Benchmark {attn_type} BS={num_seqs_list}...")
    if profiling:
        name = f"PA_v1_{attn_type}_Hq{query.size(1)}_Hkv{key_cache_hpu.size(1)}_BlockSize{block_size}_Seq{max_seq_len}"
        profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=0, warmup=3, active=3, repeat=0),
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.HPU],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(name, use_gzip=True),
            with_stack=True, with_modules=False, record_shapes=False, profile_memory=False)
        profiler.start()
    torch.hpu.synchronize()
    for i in range(num_iters):
        print(f"Iteration_{i} Start")
        for input in input_list:
            run_hpu_benchmark(input)
            htorch.core.mark_step()
            torch.hpu.synchronize()
        if profiling:
            profiler.step()
        print(f"Iteration_{i} Stop")
    if profiling:
        profiler.stop()
        print(f"Benchmark Finished! Please check profiles stored in {name}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Benchmark the paged attention kernel.")
    parser.add_argument("--version",
                        type=str,
                        choices=["v1"],
                        default="v1")
    parser.add_argument("--batch-size-list",
                        nargs='+',
                        type=int,
                        default=[1, 4, 16, 64, 256])
    parser.add_argument("--seq_len", type=int, default=8192)
    parser.add_argument("--num-query-heads", type=int, default=32)
    parser.add_argument("--num-kv-heads", type=int, default=32)
    parser.add_argument("--head-size",
                        type=int,
                        choices=[64, 80, 96, 112, 128, 256],
                        default=128)
    parser.add_argument("--block-size",
                        type=int,
                        choices=[16, 32, 128, 256],
                        default=128)
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
        choices=["auto", "fp8"],
        default="auto",
        help=
        'Data type for kv cache storage. If "auto", will use model data type. '
        'FP8_E5M2 (without scaling) is only supported on cuda version greater '
        'than 11.8. On ROCm (AMD GPU), FP8_E4M3 is instead supported for '
        'common inference criteria.')
    args = parser.parse_args()
    print(args)

    if args.num_query_heads % args.num_kv_heads != 0:
        raise ValueError("num_query_heads must be divisible by num_kv_heads")
    main(
        version=args.version,
        num_seqs_list=args.batch_size_list,
        seq_len=args.seq_len,
        num_query_heads=args.num_query_heads,
        num_kv_heads=args.num_kv_heads,
        head_size=args.head_size,
        block_size=args.block_size,
        use_alibi=args.use_alibi,
        dtype=STR_DTYPE_TO_TORCH_DTYPE[args.dtype],
        seed=args.seed,
        do_profile=args.profile,
        kv_cache_dtype=args.kv_cache_dtype,
    )
