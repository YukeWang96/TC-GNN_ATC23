# TC-GNN (Running Sparse GNN on Dense Tensor Core on Ampere GPU)

+ **Cite this project and [paper](https://arxiv.org/abs/2112.02052).**
```
@inproceedings{TC-GNN,
  title={TC-GNN: Accelerating Sparse Graph Neural Network Computation Via Dense Tensor Core on GPUs},
  author={Yuke Wang and Zheng Wang and Boyuan Feng and Yufei Ding},
  booktitle={USENIX Annual Technical Conference},
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
+ Go to `Docker/`
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
./build.sh
``` 
to install the TCGNN_conv modules with Pytorch binding. 
**Note that this step is required for both Docker and Conda setup.**


### Download graph datasets.
Get the preprocessed datasets in `.npy` at [here](https://storage.googleapis.com/graph_dataset/tcgnn-ae-graphs.tar.gz), 
then run
```
wget https://storage.googleapis.com/graph_dataset/tcgnn-ae-graphs.tar.gz
tar -zxvf tcgnn-ae-graphs.tar.gz
``` 

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

## Running **TC-GNN**.
> +  Under the current project directory 
> + `./0_bench.py| tee run_TCGNN.log` to run the script and the report 10 epoch runtime for all evaluated datasets. 
> + `./1_log2csv.py` to convert the `run_TCGNN.log` to `run_TCGNN.csv` for ease of analysis.
