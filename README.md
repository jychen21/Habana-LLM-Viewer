# projection

## Command
1. Simpily run with **run_projection.py** and the results will be saved to folder "data" and "figure".
    ```sh
    python run_projection.py
    ```
2. Run with jupyter notebook: **run_projection.ipynb** for simpily visualization.

## Example

### Compute
#### Llama2-7B
- Overall Projection
![Llama2-7B 2c projection](./figure/IntelGaudi2C_compute_projection_Llama2-7B.png)

- Attention (QK + SV) Projection
![Llama2-7B 2c attn projection](./figure/IntelGaudi2C_attn_projection_Llama2-7B.png)

- FFN (UP) Projection
![Llama2-7B 2c ffn projection](./figure/IntelGaudi2C_ffn_projection_Llama2-7B.png)

#### Mixtral-8x7B
- Overall Projection
![Mixtral-8x7B 2c projection](./figure/IntelGaudi2C_compute_projection_Mixtral-8x7B.png)

- Attention (QK + SV) Projection
![Mixtral-8x7B 2c attn projection](./figure/IntelGaudi2C_attn_projection_Mixtral-8x7B.png)

- FFN (UP) Projection
![Mixtral-8x7B 2c ffn projection](./figure/IntelGaudi2C_ffn_projection_Mixtral-8x7B.png)

<!-- ##### Projection Table
![Mixtral-8x7B projection table](./figure/mixtral_proj_table.png)
##### Bound Analysis
![Mixtral-8x7B analysis table](./figure/mixtral_analysis_table.png)

### Memory
![Mixtral-8x7B memory analysis](./figure/mixtral_memory_analysis.png) -->

## Todo
1. Currently only with single card, will support multi-card / multi-node.
2. Only cover Llama2-7B, Mixtral-8x7B, will cover more models.