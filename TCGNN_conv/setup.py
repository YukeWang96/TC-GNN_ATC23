from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='TCGNN',
    ext_modules=[
        CUDAExtension('TCGNN', [
            'TCGNN.cpp',
            'TCGNN_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })