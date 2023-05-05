# TC-GNN Artifact for USENIX ATC'23.

+ **Cite this project and [paper](https://arxiv.org/abs/2112.02052).** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7893174.svg)](https://doi.org/10.5281/zenodo.7893174)

```
@inproceedings{TC-GNN,
  title={TC-GNN: Bridging Sparse GNN Computation and Dense Tensor Cores on GPUs},
  author={Yuke Wang and Boyuan Feng and Zheng Wang and Guyue Huang and Yufei Ding},
  booktitle={USENIX Annual Technical Conference (ATC)},
  year={2023}
}
```

+ **Clone this project**.
```
git clone git@github.com:YukeWang96/TCGNN-Pytorch.git
```

+ **OS & Compiler**: 
> + `Ubuntu 16.04+`
> + `gcc >= 7.5`
> + `cmake >= 3.14`
> + `CUDA >= 11.0` and `nvcc >= 11.0`

## Files and Directories.
+ `config.py`: the configuration file for the shape of a TC block.
+ `bench.py`: the benchmark file for invoking `main_tcgnn.py` for various datasets and models.
+ `main_tcgnn.py`: the main entry for running TC-GNN.
+ `count_TC_blocks.py`: counting the total number of TC blocks without sparse-graph translation.
+ `proc_prof.py`: get the detailed GPU kernel metrics from the ncu csv output. 
+ `TCGNN_conv/`: the directory for core TC-GNN implementations, including `TCGNN_kernel.cu` and `TCGNN.cpp`.

## Environment Setup.
### [**Method-1**] Install via Docker (Recommended).
+ Go to `docker/`
+ Run `./build.sh`
+ Run `./launch.sh`

### [**Method-2**] Install via Conda.
+ Install **`conda`** on system **[Toturial](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)**.
+ Create a **`conda`** environment: 
```
conda create -n env_name python=3.6
```
+ Install **`Pytorch`**: 
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```
or using `pip` [**Note that make sure the `pip` you use is the `pip` from current conda environment. You can check this by `which pip`**]
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
+ Install [**`Deep Graph Library (DGL)`**](https://github.com/dmlc/dgl).
```
conda install -c dglteam dgl-cuda11.0
pip install torch requests tqdm
```

+ Install [**`Pytorch-Geometric (PyG)`**](https://github.com/rusty1s/pytorch_geometric).
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric
```

### Install **`TC-GNN`**.
Go to `TCGNN_conv/`, then run
```
./0_build_tcgnn.sh
``` 
to install the TCGNN_conv modules with Pytorch binding. 
**Note that this step is required for both Docker and Conda setup.**


### Download graph datasets.
Get the preprocessed datasets
```
wget https://storage.googleapis.com/graph_dataset/tcgnn-ae-graphs.tar.gz
tar -zxvf tcgnn-ae-graphs.tar.gz && rm -rf tcgnn-ae-graphs.tar.gz
``` 

## Running **PyG** baseline.
> + Go to **`pyg_baseline/`** directory;
> + `./0_run_pyg.sh`to run all pyg experiments.
> + Check the results in **`1_bench_gcn.csv`** and **`1_bench_agnn.csv`**, which are similar as below.

| dataset         | Avg.Epoch (ms) |
|:-----------------|----------------:|
| citeseer        | 10.149         |
| cora            | 9.964          |
| pubmed          | 10.114         |
| ppi             | 13.419         |
| PROTEINS_full   | 10.908         |
| OVCAR-8H        | 72.636         |
| Yeast           | 66.644         |
| DD              | 18.972         |
| YeastH          | 118.047        |
| amazon0505      | 29.731         |
| artist          | 11.172         |
| com-amazon      | 22.476         |
| soc-BlogCatalog | 14.971         |
| amazon0601      | 26.621         |

<!-- > + Change `run_GCN=True` or  `run_GCN=False` in `0_bench.py` with `gcn` and `gin` to profile the example GCN and GIN model, respectively;
> + `./0_bench.py| tee run_pyg.log` to run the script and the report 10 epoch runtime for all evaluated datasets. 
> + `./1_log2csv.py` to convert the `run_pyg.log` to `run_pyg.csv` for ease of analysis. -->

## Running **DGL** baseline.
> +  Go to **`dgl_baseline/`** directory.
> + `./0_run_dgl.sh`to run all dgl experiments.
> + Check the results in `1_bench_gcn.csv` and `1_bench_agnn.csv`.

<!-- > +  Pass the `--model` parameter in `dgl_main.py` with `gcn` and  `gin` to profile the example GCN and GIN model, respectively;
> + `./0_bench.py| tee run_dgl.log` to run the script and the report 10 epoch runtime for all evaluated datasets. 
> + `./1_log2csv.py` to convert the `run_dgl.log` to `run_dgl.csv` for ease of visualization. -->

## Running **TC-GNN**.
> +  Go to project root directory.
> + `./0_run_tcgnn.sh`to run all dgl experiments.
> + Check the results in `1_bench_gcn.csv` and `1_bench_agnn.csv`.

<!-- > +  Under the current project directory 
> + `./0_bench.py| tee run_TCGNN.log` to run the script and the report 10 epoch runtime for all evaluated datasets. 
> + `./1_log2csv.py` to convert the `run_TCGNN.log` to `run_TCGNN.csv` for ease of analysis. -->

## Reference.
+ [**Deep Graph Library**](https://github.com/dmlc/dgl) <br>
Wang, Minjie, et al. 
**Deep graph library: A graph-centric, highly-performant package for graph neural networks.**. *The International Conference on Learning Representations (ICLR), 2019.*

+ [**Pytorch Geometric**](https://github.com/rusty1s/pytorch_geometric) <br>
Fey, Matthias, and Jan Eric Lenssen. 
**Fast graph representation learning with PyTorch Geometric.** 
*The International Conference on Learning Representations (ICLR), 2019.*

+ [**ASpT**](http://gitlab.hpcrl.cse.ohio-state.edu/chong/ppopp19_ae/-/blob/master/README.md) <br>
Hong, Changwan, et al. 
**Adaptive sparse tiling for sparse matrix multiplication.** *In Proceedings of the 24th Symposium on Principles and Practice of Parallel Programming (PPoPP), 2019*.

+ [**tSparse**](https://github.com/oresths/tSparse) <br>
Zachariadis, O., et. al. 
**Accelerating Sparse Matrix-Matrix Multiplication with GPU Tensor Cores** *Computers & Electrical Engineering (2020).

+ [**cuSPARSELt**](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/?ncid=so-twit-40939#cid=hpc06_so-twit_en-us)<br>
NVIDIA. Exploiting NVIDIA Ampere Structured Sparsity with cuSPARSELt.