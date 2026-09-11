# Our own minimal torch-extension build (E.T. ships no python bindings at
# all -- only CMake C++ test binaries, README: "a few examples of encoder").
# Sources:
#  - attn_binding.cu:  the glue WE wrote, lives in this directory.
#  - source/kernels/attention.cu: the artifact's own, UNMODIFIED kernel file
#    (contains OTF_attention_kernelLauncher plus the other two launchers,
#    Prune_attention_kernelLauncher/sharedQK_attention_kernelLauncher --
#    unused by attn_binding.cu but harmless to compile alongside; they need
#    no extra sources of their own beyond utils.h, header-only).
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))
ET_SRC = os.path.join(HERE, "source")
# Target SM: default sm_80 (A100), overridden to the visible GPU's arch by
# bench/env.sh's KB_SM (e.g. 90 on an H100). This is our own build glue, not a
# kernel source edit -- the gencode below was previously hardcoded sm_80, which
# produced SASS-only binaries that fail on newer GPUs with
# cudaErrorNoKernelImageForDevice.
_KB_SM = os.environ.get("KB_SM", "80")

setup(
    name="et_attn_binding",
    ext_modules=[
        CUDAExtension(
            name="et_attn_binding",
            sources=[
                os.path.join(HERE, "attn_binding.cu"),
                os.path.join(ET_SRC, "kernels", "attention.cu"),
            ],
            include_dirs=[ET_SRC],
            extra_compile_args={
                "cxx": ["-O3"],
                # Same issue and fix as bytetransformer/setup.py: torch's
                # CUDAExtension unconditionally injects
                # -D__CUDA_NO_HALF{,2}_{OPERATORS,CONVERSIONS}__ ahead of any
                # extra_compile_args; E.T.'s own kernels/attention.cu
                # (unmodified) relies on the default half2 arithmetic
                # operators those macros disable (`sum2.x + sum2.y`,
                # `half2{...}` list-init from float literals via `(half)1.0f`,
                # `dst2[i] * one_over_sum`, `wmma::fill_fragment(c_frag,
                # (half)0.0f)`). -U undoes exactly those 4 macros for this
                # build (appears after -D on the same nvcc command line) --
                # a build-flag fix, not a change to any kernel source file.
                "nvcc": ["-O3", "--expt-relaxed-constexpr", "--extended-lambda",
                         f"-gencode=arch=compute_{_KB_SM},code=sm_{_KB_SM}",
                         "-U__CUDA_NO_HALF_OPERATORS__",
                         "-U__CUDA_NO_HALF_CONVERSIONS__",
                         "-U__CUDA_NO_HALF2_OPERATORS__",
                         "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
