# projection

## Command
1. simpily run with **moe_projection.py**
    ```sh
    python moe_projection.py
    ```
2. run with jupyter notebook: **moe_projection.ipynb**

## Example

### Compute
#### Llama2-7B
![Llama2-7B 2c projection](./figure/IntelGaudi2C_compute_projection_Llama2-7B.png)


#### Mixtral-8x7B
![Mixtral-8x7B 2c projection](./figure/IntelGaudi2C_compute_projection_Mixtral-8x7B.png)
<!-- ##### Projection Table
![Mixtral-8x7B projection table](./figure/mixtral_proj_table.png)
##### Bound Analysis
![Mixtral-8x7B analysis table](./figure/mixtral_analysis_table.png)

### Memory
![Mixtral-8x7B memory analysis](./figure/mixtral_memory_analysis.png) -->

## Todo
1. Currently only with single card, will support multi-card / multi-node.
2. Only cover Llama2-7B, Mixtral-8x7B, will cover more models.