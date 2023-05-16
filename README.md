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

## 1. Clone this project.
```
git clone --recursive git@github.com:YukeWang96/TCGNN-Pytorch.git
```

+ **Requirements**: 
> + `Ubuntu 16.04+`
> + `gcc >= 7.5`
> + `cmake >= 3.14`
> + `CUDA >= 11.0` and `nvcc >= 11.0`
> + NVIDIA GPU with `sm >= 80` (i.e., Ampere, like RTX3090).

## 2. Environment Setup.(Skip this if evaluation on provided server)
### 2.1 [**Method-1**] Install via Docker (Recommended).
+ Go to `docker/`
+ Run `./build.sh`
+ Run `./launch.sh`

### 2.2 [**Method-2**] Install via Conda.
+ 2.2.1 Install **`conda`** on system **[Toturial](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)**.
+ 2.2.2 Create a **`conda`** environment: 
```
conda create -n env_name python=3.6
```
+ 2.2.3 Install **`Pytorch`**: 
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```
or using `pip` [**Note that make sure the `pip` you use is the `pip` from current conda environment. You can check this by `which pip`**]
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
+ 2.2.4 Install [**`Deep Graph Library (DGL)`**](https://github.com/dmlc/dgl).
```
conda install -c dglteam dgl-cuda11.0
pip install torch requests tqdm
```

+ 2.2.5 Install [**`Pytorch-Geometric (PyG)`**](https://github.com/rusty1s/pytorch_geometric).
```
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
pip install torch-geometric
```

## 3. Install **`TC-GNN`**.
Go to `TCGNN_conv/`, then run
```
./0_build_tcgnn.sh
``` 
to install the TCGNN_conv modules with Pytorch binding. 
**Note that this step is required for both Docker and Conda setup.**


## 4. Download graph datasets.
Get the preprocessed datasets
```
wget https://storage.googleapis.com/graph_dataset/tcgnn-ae-graphs.tar.gz
tar -zxvf tcgnn-ae-graphs.tar.gz && rm -rf tcgnn-ae-graphs.tar.gz
``` 


## 5. Running **TC-GNN** in end-to-end model.
> +  Go to project root directory.
> + `./0_run_tcgnn_model.sh`to run all TC-GNN experiments.
> + Check the results in `1_bench_gcn.csv` and `1_bench_agnn.csv`.

## 6. Running **DGL** baseline (Fig-6a).
> +  Go to **`dgl_baseline/`** directory.
> + `./0_run_dgl.sh`to run all dgl experiments.
> + Check the results in `Fig_6a_dgl_gcn.csv` and `Fig_6a_dgl_agnn.csv`.

## 7. Running **PyG** baseline (Fig-6b).
> + Go to **`pyg_baseline/`** directory;
> + `./0_run_pyg.sh`to run all pyg experiments.
> + Check the results in `Fig_6b_PyG_gcn.csv` and `Fig_6b_PyG_agnn.csv`.

## 8. Running **TC-GNN** in single-kernel comparison.
> +  Go to project root directory.
> + `./0_run_tcgnn_single_kernel.sh`to run TC-GNN single kernel experiments.
> + Check the results in `1_bench_gcn.csv` and `1_bench_agnn.csv`.

## 9. cuSPARSE-bSpMM Baseline (Fig-6c) 
```
cd TCGNN-bSpmm/cusparse
./0_run_bSpMM.sh
```
+ Check the results in `Fig_6c_cuSPARSE_bSpMM.csv`.


## 10. Dense Tile Reduction (Fig-7).
```
python 3_cnt_TC_blk_SDDMM.py
python 3_cnt_TC_blk_SpMM.py
```
+ Check the results in `3_cnt_TC_blk_SDDMM.csv` and `3_cnt_TC_blk_SDDMM.csv`.

## 11. tSparse Baseline (Table-5, column-2) (**running outside the docker**).
```
cd TCGNN-tsparse/
./0_run_tSparse.sh
```
+ Check the results in `Table_5_tSparse.csv`.

## 12. Triton Baseline (Table-5, column-3)  (**running outside the docker in a `triton` conda env**).
```
cd TCGNN-trition/python/bench
conda activate triton
./0_run_triton.sh
```
+ Check the results in `1_run_triton.csv`.


## 13. Use TC-GNN as a Tool or Library for your project.

Building a new design based on TC-GNN is simple, there are only several steps:


### 13.1 Register a new PyTorch Operator. 
+ Add a compilation entry in `TCGNN.cpp` under `TCGNN_conv/`. An example is shown below.

https://github.com/YukeWang96/TC-GNN_ATC23/blob/0d40b53ccd232b396899e4bd56114e9754c9c145/TCGNN_conv/TCGNN.cpp#L63-L86

https://github.com/YukeWang96/TC-GNN_ATC23/blob/0d40b53ccd232b396899e4bd56114e9754c9c145/TCGNN_conv/TCGNN.cpp#L265

### 13.2 Build the C++ design based on our existing examples 
+ Add the operator implementation in `TCGNN_kernel.cpp` file under `TCGNN_conv/`. An example is shown below.

https://github.com/YukeWang96/TC-GNN_ATC23/blob/0d40b53ccd232b396899e4bd56114e9754c9c145/TCGNN_conv/TCGNN_kernel.cu#L175-L220

### 13.3 Build the CUDA kernel design based on our existing examples. 
+ Add a CUDA kernel design in `TCGNN_kernel.cuh`. An example is shown below.

https://github.com/YukeWang96/TC-GNN_ATC23/blob/0d40b53ccd232b396899e4bd56114e9754c9c145/TCGNN_conv/TCGNN_kernel.cu#L336-L454


### 13.4 Launch the TCGNN docker and recompile, 
+ The compiled exectuable will be located under `build/`.
```
cd docker 
./launch.sh
./0_build_tcgnn.sh
```

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
**Accelerating Sparse Matrix-Matrix Multiplication with GPU Tensor Cores** *Computers & Electrical Engineering (2020)*.

+ [**cuSPARSELt**](https://developer.nvidia.com/blog/exploiting-ampere-structured-sparsity-with-cusparselt/?ncid=so-twit-40939#cid=hpc06_so-twit_en-us)<br>
NVIDIA. **Exploiting NVIDIA Ampere Structured Sparsity with cuSPARSELt**.
