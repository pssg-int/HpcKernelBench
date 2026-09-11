# dtc_setup.py -- OUR build script for DTC-SpMM's torch extension.
#
# The artifact's own source/DTC-SpMM/setup.py links against prebuilt Sputnik
# and Glog libraries (env vars SPUTNIK_PATH/GLOG_PATH -> build/{sputnik,glog}
# dirs from a full CMake build of each). Neither is needed for the code path
# this integration wraps -- see STATUS.md's "Sputnik/Glog" section: the
# Sputnik-baseline function this repo bundles for its OWN paper comparison is
# disabled via a one-line patch (source.patch), and Glog is never referenced
# by DTCSpMM.cpp/DTCSpMM_kernel.cu at all (only listed in the upstream
# setup.py, apparently vestigial). Only Sputnik's HEADERS (not a built
# library) are needed, because the disabled function's #include lines are
# unconditional; `git submodule update --init --depth 1 third_party/sputnik`
# (run by build.sh) fetches those headers with no CMake build required.
#
# This file lives in <short>/ (not source/), per ARTIFACT_GUIDE.md rule 3:
# everything WE write lives outside source/. It builds directly from
# source/DTC-SpMM/{DTCSpMM.cpp,DTCSpMM_kernel.cu} (untouched paths, one
# one-line patch inside DTCSpMM_kernel.cu -- see source.patch) into
# build/lib/DTCSpMM*.so under THIS directory, never installed into the
# shared plexus_env site-packages (same precedent as spmv/diaq/build.sh).

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "source", "DTC-SpMM")
SPUTNIK_ROOT = os.path.join(HERE, "source", "third_party", "sputnik")
# Target SM: default sm_80 (A100), overridden to the visible GPU's arch by
# bench/env.sh's KB_SM (e.g. 90 on H100). Was hardcoded sm_80 -> SASS-only, so
# the .so failed on newer GPUs with cudaErrorNoKernelImageForDevice. Build glue.
_KB_SM = os.environ.get("KB_SM", "80")

setup(
    name="DTCSpMM",
    ext_modules=[
        CUDAExtension(
            "DTCSpMM",
            [os.path.join(SRC, "DTCSpMM.cpp"), os.path.join(SRC, "DTCSpMM_kernel.cu")],
            include_dirs=[SPUTNIK_ROOT],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", f"-gencode=arch=compute_{_KB_SM},code=sm_{_KB_SM}"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
