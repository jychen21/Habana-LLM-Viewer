# projection

## Command
1. simpily run with **moe_projection.py**
    ```sh
    python moe_projection.py
    ```
2. run with jupyter notebook: **moe_projection.ipynb**

## Example (Mixtral-8x7B)

### Compute
#### Llama2-7B
![Llama2-7B projection](./figure/compute_projection_llama2-7b.png)

#### Mixtral-8x7B
![Mixtral-8x7B projection](./figure/compute_projection_mixtral-8x7b.png)
<!-- ##### Projection Table
![Mixtral-8x7B projection table](./figure/mixtral_proj_table.png)
##### Bound Analysis
![Mixtral-8x7B analysis table](./figure/mixtral_analysis_table.png)

### Memory
![Mixtral-8x7B memory analysis](./figure/mixtral_memory_analysis.png) -->

## Todo
1. Currently only with single card, will support multi-card / multi-node.
2. Only cover Llama2-7B, Mixtral-8x7B, will cover more models.