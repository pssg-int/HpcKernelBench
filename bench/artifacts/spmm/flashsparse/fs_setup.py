# fs_setup.py -- OUR build script for FlashSparse's torch extensions.
#
# The artifact's own source/FlashSparse/compile.sh runs
# `python setup.py install`, which builds ALL FOUR of its extensions
# (FS_SpMM, FS_SDDMM, FS_Block, FS_Block_gpu) in one shot and installs into
# the (shared) venv's site-packages. This integration only needs the SpMM
# kernel (FS_SpMM) plus its GPU preprocessing (FS_Block_gpu) -- FS_SDDMM is a
# different track (sddmm) and FS_Block is a CPU preprocessing path this
# adapter does not use (see STATUS.md) -- so this file builds only those two,
# into build/lib under THIS directory (never installed into the shared
# plexus_env site-packages, same precedent as spmv/diaq and dtcspmm/
# dtc_setup.py).

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))
FS = os.path.join(HERE, "source", "FlashSparse")
# Target SM: default sm_80 (A100), overridden to the visible GPU's arch by
# bench/env.sh's KB_SM (e.g. 90 on H100). Was hardcoded sm_80 -> SASS-only, so
# the .so failed on newer GPUs with cudaErrorNoKernelImageForDevice. Build glue.
_KB_SM = os.environ.get("KB_SM", "80")

setup(
    name="FlashSparse_kernel_subset",
    ext_modules=[
        CUDAExtension(
            "FS_SpMM",
            [os.path.join(FS, "SpMM", "src", "benchmark.cpp"),
             os.path.join(FS, "SpMM", "src", "spmmKernel.cu")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", f"-gencode=arch=compute_{_KB_SM},code=sm_{_KB_SM}"],
            },
        ),
        CUDAExtension(
            "FS_Block_gpu",
            [os.path.join(FS, "Block_gpu", "block.cpp"),
             os.path.join(FS, "Block_gpu", "block_kernel.cu")],
            include_dirs=[os.path.join(FS, "Block_gpu")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", f"-gencode=arch=compute_{_KB_SM},code=sm_{_KB_SM}"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
