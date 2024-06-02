# projection

## Command
1. Simpily run with **run_projection.py** and the results will be saved to folder "data".
    ```sh
    python run_projection.py
    ```
2. Run with jupyter notebook: **run_projection.ipynb** for simpily visualization.

## Example

### Compute
#### Llama2-7B
- Overall Projection [(more details in table)](./data/Llama2-7B/IntelGaudi2B_overall_projection.csv)
![Llama2-7B Overall Projection](./data/Llama2-7B/IntelGaudi2B_pp1_tp1_overall_projection.png)

- Attn Projection
[Prefill](./data/Llama2-7B/IntelGaudi2B_pp1_tp1_BF16_prefill_attn_qksv\(bmm\)_projection.csv) |
[Decode](./data/Llama2-7B/IntelGaudi2B_pp1_tp1_BF16_decode_attn_qksv\(bmm\)_projection.csv)

- FFN Projection
[Prefill](./data/Llama2-7B/IntelGaudi2B_pp1_tp1_BF16_prefill_ffn_up\(mm\)_projection.csv) |
[Decode](./data/Llama2-7B/IntelGaudi2B_pp1_tp1_BF16_decode_ffn_up\(mm\)_projection.csv)

#### Llama2-13B
- Overall Projection [(more details in table)](./data/Llama2-7B/IntelGaudi2C_overall_projection.csv)
![Llama2-13B Overall Projection](./data/Llama2-13B/IntelGaudi2B_pp1_tp1_overall_projection.png)

- Attn Projection
[Prefill](./data/Llama2-13B/IntelGaudi2B_pp1_tp1_BF16_prefill_attn_qksv\(bmm\)_projection.csv) |
[Decode](./data/Llama2-13B/IntelGaudi2B_pp1_tp1_BF16_decode_attn_qksv\(bmm\)_projection.csv)

- FFN Projection
[Prefill](./data/Llama2-13B/IntelGaudi2B_pp1_tp1_BF16_prefill_ffn_up\(mm\)_projection.csv) |
[Decode](./data/Llama2-13B/IntelGaudi2B_pp1_tp1_BF16_decode_ffn_up\(mm\)_projection.csv)

#### Qwen-7B
- Overall Projection [(more details in table)](./data/Qwen-7B/IntelGaudi2B_overall_projection.csv)
![Qwen-7B Overall Projection](./data/Qwen-7B/IntelGaudi2B_pp1_tp1_overall_projection.png)

- Attn Projection
[Prefill](./data/Qwen-7B/IntelGaudi2B_pp1_tp1_BF16_prefill_attn_qksv\(bmm\)_projection.csv) |
[Decode](./data/Qwen-7B/IntelGaudi2B_pp1_tp1_BF16_decode_attn_qksv\(bmm\)_projection.csv)

- FFN Projection
[Prefill](./data/Qwen-7B/IntelGaudi2B_pp1_tp1_BF16_prefill_ffn_up\(mm\)_projection.csv) |
[Decode](./data/Qwen-7B/IntelGaudi2B_pp1_tp1_BF16_decode_ffn_up\(mm\)_projection.csv)

#### Qwen-14B
- Overall Projection [(more details in table)](./data/Qwen-14B/IntelGaudi2B_overall_projection.csv)
![Qwen-14B Overall Projection](./data/Qwen-14B/IntelGaudi2B_pp1_tp1_overall_projection.png)

- Attn Projection
[Prefill](./data/Qwen-14B/IntelGaudi2B_pp1_tp1_BF16_prefill_attn_qksv\(bmm\)_projection.csv) |
[Decode](./data/Qwen-14B/IntelGaudi2B_pp1_tp1_BF16_decode_attn_qksv\(bmm\)_projection.csv)

- FFN Projection
[Prefill](./data/Qwen-14B/IntelGaudi2B_pp1_tp1_BF16_prefill_ffn_up\(mm\)_projection.csv) |
[Decode](./data/Qwen-14B/IntelGaudi2B_pp1_tp1_BF16_decode_ffn_up\(mm\)_projection.csv)

#### Mixtral-8x7B
- Overall Projection [(more details in table)](./data/Mixtral-8x7B/IntelGaudi2B_overall_projection.csv)
![Mixtral-8x7B Overall Projection](./data/Mixtral-8x7B/IntelGaudi2B_pp1_tp1_overall_projection.png)

- Attn Projection
[Prefill](./data/Mixtral-8x7B/IntelGaudi2B_pp1_tp1_BF16_prefill_attn_qksv\(bmm\)_projection.csv) |
[Decode](./data/Mixtral-8x7B/IntelGaudi2B_pp1_tp1_BF16_decode_attn_qksv\(bmm\)_projection.csv)

- FFN Projection
[Prefill](./data/Mixtral-8x7B/IntelGaudi2B_pp1_tp1_BF16_prefill_ffn_up\(mm\)_projection.csv) |
[Decode](./data/Mixtral-8x7B/IntelGaudi2B_pp1_tp1_BF16_decode_ffn_up\(mm\)_projection.csv)


## Todo
1. Currently only cover single card perf projection, will support multi-card / multi-node.
2. Only cover Llama2-7B, Qwen-7B and Mixtral-8x7B, will cover more models.