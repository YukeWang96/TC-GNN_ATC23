cd TCGNN_conv/
# TORCH_CUDA_ARCH_LIST="8.0" python setup.py clean --all install      # for RTX3090/RTX3070
TORCH_CUDA_ARCH_LIST="8.6" python setup.py clean --all install      # for RTX3090/RTX3070
cd ..