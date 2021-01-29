from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='GAcc',
    ext_modules=[
        CUDAExtension('GAcc', [
            'GAcc.cpp',
            'GAcc_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })