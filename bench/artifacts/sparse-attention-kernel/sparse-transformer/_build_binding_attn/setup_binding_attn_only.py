# Private, throwaway build script -- NOT part of the artifact's own source
# tree (lives under bench/artifacts/sparse-attention-kernel/sparse-
# transformer/_build_binding_attn/, outside source/). Mirrors
# source/src/setup.py's own extra_compile_args/arch flags exactly, scoped
# to the binding_attn extension only. See build.sh's header comment.
import os
import sys
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ops_src = sys.argv[1]
cuda_arch = sys.argv[2]
sys.argv = sys.argv[:1] + ["build_ext", "--inplace"]

extra_compile_args = {
    "nvcc": ["-O3", "-w",
             "-I/usr/local/cuda/include",
             f"-I{ops_src}/include/",
             f"-I{ops_src}/include/cutlass/include",
             f"-gencode=arch=compute_{cuda_arch},code=sm_{cuda_arch}"],
    "cxx": ["-fPIC",
            f"-I{ops_src}/include/",
            f"-I{ops_src}/include/cutlass/include"],
}

# BuildExtension.with_options(use_ninja=True) generates its own link command
# internally (build.ninja) rather than honoring the process's $LDSHARED, so
# build.sh's LDSHARED rpath override (needed on this machine's conda python,
# see build.sh's header) is silently ignored for THIS extension -- unlike the
# fused3s/gpa extensions, which use the artifact's own setup.py without
# use_ninja and so do pick up LDSHARED. Symptom: the built .so links against
# libc10.so/libtorch.so etc. (found via torch's own -L at link time) but has
# no rpath to torch/lib, so `import binding_attn` fails at runtime with
# "libc10.so: cannot open shared object file" even though the build itself
# succeeds. Fix: pass the rpath explicitly via extra_link_args, which IS
# honored by the ninja path (computed from `torch.__file__`, not hardcoded).
torch_lib = os.path.join(os.path.dirname(os.path.abspath(torch.__file__)), "lib")
extra_link_args = [f"-Wl,-rpath,{torch_lib}"]

setup(
    name="binding_attn_standalone_build",
    ext_modules=[CUDAExtension(
        name="binding_attn",
        sources=[os.path.join(ops_src, "binding_attn.cpp"),
                 os.path.join(ops_src, "binding_attn_cuda.cu")],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True, no_python_abi_suffix=True)},
)
