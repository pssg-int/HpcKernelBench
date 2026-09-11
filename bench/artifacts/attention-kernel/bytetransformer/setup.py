# Our own minimal torch-extension build (NOT the artifact's own CMake/th_op
# build, which would additionally require CUTLASS + gemm.cu/attention_nofused
# for a full BertTransformer encoder layer -- see attn_binding.cu's docstring
# for why the attention-only boundary needs none of that). Sources:
#  - attn_binding.cu:            the glue WE wrote, lives in this directory.
#  - source/.../attention_fused.cu / attention_fused_long.cu: the artifact's
#    own, UNMODIFIED fused-attention kernel translation units.
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = os.path.dirname(os.path.abspath(__file__))
BT_SRC = os.path.join(HERE, "source", "bytetransformer")
# Target SM: default sm_80 (A100), overridden to the visible GPU's arch by
# bench/env.sh's KB_SM (e.g. 90 on an H100). This is our own build glue, not a
# kernel source edit -- the gencode below was previously hardcoded sm_80, which
# produced SASS-only binaries that fail on newer GPUs with
# cudaErrorNoKernelImageForDevice.
_KB_SM = os.environ.get("KB_SM", "80")

setup(
    name="bt_attn_binding",
    ext_modules=[
        CUDAExtension(
            name="bt_attn_binding",
            sources=[
                os.path.join(HERE, "attn_binding.cu"),
                os.path.join(BT_SRC, "src", "attention_fused.cu"),
                os.path.join(BT_SRC, "src", "attention_fused_long.cu"),
                # Attention<HALF> has a virtual destructor and two virtual
                # methods (cal_bufsize, infer) defined INLINE in attention.h;
                # instantiating/destroying an Attention<HALF> object (even
                # though we only ever CALL fused_rm_infer/fused_long_rm_infer
                # directly) forces the compiler to emit a full vtable, which
                # references nofused_infer/fused_infer/fused_long_infer too
                # (infer()'s dead else/if branches) -- their DEFINITIONS
                # (not just declarations) must therefore be link-reachable
                # even though never executed at runtime for our objects
                # (use_fused_attention_ is always true here). Confirmed by a
                # first build attempt: linker error
                # "undefined symbol ...Attention...nofused_infer...".
                # attention_nofused.cu needs gemm.h's cuBLAS-only GEMM
                # launcher (gemm.cu, NOT gemm_bias_act.cu -- that one IS
                # CUTLASS-based and is not pulled in) and
                # attention_nofused_utils.cu; variety_attention_fused.h is
                # header-only. None of these three files touch CUTLASS
                # (confirmed: `grep -rl cutlass` over the whole bytetransformer/
                # tree does not list gemm.h/gemm.cu/attention_nofused*/
                # variety_attention_fused.h), so this integration still needs
                # no CUTLASS checkout.
                os.path.join(BT_SRC, "src", "attention_nofused.cu"),
                os.path.join(BT_SRC, "src", "attention_nofused_utils.cu"),
                os.path.join(BT_SRC, "src", "gemm.cu"),
            ],
            include_dirs=[os.path.join(HERE, "source")],
            extra_compile_args={
                "cxx": ["-O3"],
                # torch's CUDAExtension unconditionally injects
                # -D__CUDA_NO_HALF{,2}_{OPERATORS,CONVERSIONS}__ ahead of any
                # extra_compile_args (to avoid ODR clashes between raw CUDA
                # half ops and ATen's own overloads) -- ByteTransformer's own
                # reduce.h (unmodified) relies on the DEFAULT implicit
                # __half<->float conversion operators those macros disable
                # (`(float)__half2add(...)`, a __half-returning helper, cast
                # to float). Since -U appears after -D on the same nvcc
                # command line, these undo exactly those 4 macros for this
                # build only -- a build-flag fix (rule 3: "CUDA-version
                # guards are fine"), not a change to any kernel source file.
                "nvcc": ["-O3", "--expt-relaxed-constexpr",
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
