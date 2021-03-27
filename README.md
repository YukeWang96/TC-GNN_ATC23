# TC-GNN
Running GNN on Dense Tensor Core.

+ `bench.py` the benchmark file for invoking `main_tcgnn.py` for various datasets and models.
+ `main_tcgnn.py` the main entry for running TC-GNN.
+ `count_TC_blocks.py` the script for counting the TC blocks for the naive implementation without sparse-graph translation.
+ `proc_prof.py` get the detailed GPU kernel metrics from the ncu csv output. 
+ `TCGNN_conv/` the directory for core TC-GNN implementations, including `TCGNN_kernel.cu` and `TCGNN.cpp`.
