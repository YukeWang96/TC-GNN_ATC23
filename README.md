# TC-GNN (Running GNN on Dense Tensor Core)

## Files and Directory.
+ `bench.py`: the benchmark file for invoking `main_tcgnn.py` for various datasets and models.
+ `main_tcgnn.py`: the main entry for running TC-GNN.
+ `count_TC_blocks.py`: counting the TC blocks without sparse-graph translation.
+ `proc_prof.py`: get the detailed GPU kernel metrics from the ncu csv output. 
+ `TCGNN_conv/`: the directory for core TC-GNN implementations, including `TCGNN_kernel.cu` and `TCGNN.cpp`.

## Running **PyG** baseline.
> +  Go to **`pyg_baseline/`** directory;
> + Pass the `--model` parameter in `pyg_main.py` with `gcn` and `gin` to profile the example GCN and GIN model, respectively;
> + `./0_bench.py| tee run_pyg.log` to run the script and the report 10 epoch runtime for all evaluated datasets. 
> + `./1_log2csv.py` to convert the `run_pyg.log` to `run_pyg.csv` for ease of analysis.

## Running **DGL** baseline.
> +  Go to **`dgl_baseline/`** directory
> +  Pass the `--model` parameter in `dgl_main.py` with `gcn` and  `gin` to profile the example GCN and GIN model, respectively;
> + `./0_bench.py| tee run_dgl.log` to run the script and the report 10 epoch runtime for all evaluated datasets. 
> + `./1_log2csv.py` to convert the `run_dgl.log` to `run_dgl.csv` for ease of visualization.

## Running **TC-GNN** .
> +  Under the current project directory 
> + `./0_bench.py| tee run_TCGNN.log` to run the script and the report 10 epoch runtime for all evaluated datasets. 
> + `./1_log2csv.py` to convert the `run_TCGNN.log` to `run_TCGNN.csv` for ease of analysis.