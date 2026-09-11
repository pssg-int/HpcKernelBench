# fs_sddmm_setup.py -- OUR build script for FlashSparse's SDDMM extension.
#
# The artifact's own source/FlashSparse/setup.py (run via compile.sh)
# builds ALL FOUR of its bundled extensions in one shot and installs into
# the shared venv's site-packages. This integration only needs `FS_SDDMM`
# (source/FlashSparse/SDDMM/src/{benchmark.cpp,sddmmKernel.cu}, unmodified)
# and `FS_Block` (source/FlashSparse/Block/example.cpp, unmodified -- its
# `blockProcess_sddmm_balance` binding is the ONLY preprocessing path in the
# artifact that returns the 4 arrays (row_offsets, col_indices, values,
# t_window_row) FS_SDDMM's kernel needs -- see STATUS.md; FS_Block_gpu's
# `preprocess_gpu_fs`, used by the spmm track's adapter, returns only 3 and
# is NOT what SDDMM's own pybind functions take). Built here, standalone,
# same isolation precedent as ../../spmm/flashsparse/fs_setup.py -- never
# installed into the shared plexus_env site-packages.

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
    name="FlashSparse_sddmm_subset",
    ext_modules=[
        CUDAExtension(
            "FS_SDDMM",
            [os.path.join(FS, "SDDMM", "src", "benchmark.cpp"),
             os.path.join(FS, "SDDMM", "src", "sddmmKernel.cu")],
            include_dirs=[os.path.join(FS, "SDDMM", "sddmm_utils"),
                          os.path.join(FS, "SDDMM", "include")],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", f"-gencode=arch=compute_{_KB_SM},code=sm_{_KB_SM}"],
            },
        ),
        CUDAExtension(
            "FS_Block",
            [os.path.join(FS, "Block", "example.cpp")],
            extra_compile_args={
                "cxx": ["-O3", "-fopenmp"],
            },
            extra_link_args=["-fopenmp"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
