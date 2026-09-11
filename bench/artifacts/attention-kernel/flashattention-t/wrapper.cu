// pybind11 boundary for FlashAttention-T's "custom API" forward kernel.
//
// NOT part of the artifact -- this is a NEW file (analogous to
// bench/artifacts/spmm/rassm/wrapper.cpp), compiled against the artifact's
// UNMODIFIED headers under source/1-figure8-main-results/flashattention-t/
// ampere/. See STATUS.md for why this boundary was chosen.
//
// What we call: `custom_mha_fwd_causal<Dtype, HeadDim,
// CtaM, CtaN, NWarps>(...)` / `..._noncausal<...>`, declared in
// cxx-tests/custom_api/flash_api_custom.cuh. That function is the artifact's
// OWN custom pybind-free C++ API (used by the paper's own
// cxx-tests/fwd_bench/benches/*.cu benchmark, see core.cuh's bench_fwd_fp16
// for the reference call pattern this file mirrors), and it in turn calls
// `FLASH_NAMESPACE::run_custom_mha_fwd<...>()`, a template FUNCTION DEFINED
// IN A HEADER (flash_fwd_launch_template.h, line ~386) -- unlike the
// standard FlashAttention-2 forward path, the custom/tensorized kernel is
// NOT split into 114 explicit per-(headdim,dtype,causal) instantiation .cu
// files. It is instantiated directly at this call site for exactly the
// (Dtype, HeadDim, CtaM, CtaN, NWarps) combinations we list below, so this
// wrapper never touches (and does not need to compile) any of
// csrc/flash_attn/src/flash_{fwd,bwd}_hdim*.cu.
//
// The FLASHATTENTION_DISABLE_* / USE_*_SOFTMAX / LOOP{1,2}_USE_ACCS_SCALE_*
// macros below are copied VERBATIM from the artifact's own
// cxx-tests/fwd_bench/benches/core.cuh -- they select which tensorized
// ILP-softmax scheduling code path compiles in (the paper's actual
// contribution); several of them gate #error checks in softmax_mma.h if
// left undefined, so this is not an optional/decorative copy.

#define FLASHATTENTION_DISABLE_LOCAL
#define FLASHATTENTION_DISABLE_ALIBI
#define FLASHATTENTION_DISABLE_DROPOUT
#define ENABLE_PRINT_CUSTOM_API_REPORT              0
#define ENABLE_PRINT_FLASH_FWD_KERNEL_TEMPLATE_ARGS 0
#define ENABLE_PRINT_DEVICE_DETAILS                 0
#define ENABLE_PRINT_FLASH_FWD_PARAMS               0
#define ENABLE_CUSTOM_DUMP_AND_PRINTF               0

#define USE_DEFAULT_NEGINF_MASK                     0
#define USE_ACC_S_LEVEL_INF_MASKING                 0
#define USE_MMA_SOFTMAX                             1

#define USE_BINARY_TREE_MAX                         0
#define USE_DEFAULT_MAX                             1

#define LOOP1_USE_ACCS_SCALE_MMA_ONLY               0
#define LOOP1_USE_ACCS_SCALE_SIMT_ONLY              0
#define LOOP1_USE_ACCS_SCALE_ILP_VERT               0
#define LOOP1_USE_ACCS_SCALE_ILP_HORI               1
#define LOOP1_ACCS_ILP_HORI_RATIO                   2

#define LOOP2_USE_ACCS_SCALE_MMA_ONLY               0
#define LOOP2_USE_ACCS_SCALE_SIMT_ONLY              0
#define LOOP2_USE_ACCS_SCALE_ILP_VERT               1
#define LOOP2_USE_ACCS_SCALE_ILP_HORI               0
#define LOOP2_ACCS_ILP_HORI_RATIO                   4

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <flash_api_custom.cuh>   // declares custom_mha_fwd_{causal,noncausal}

// One forward call, dispatching on (head_dim, causal) to the template
// instantiation the CtaM/CtaN/NWarps values match the artifact's own choice
// for that head_dim in cxx-tests/fwd_bench/benches/hdim{64,128}-{,non}causal.cu
// (hdim64: CtaM=128,CtaN=128; hdim128: CtaM=128,CtaN=64; NWarps=4 throughout).
at::Tensor fat_fwd(at::Tensor q, at::Tensor k, at::Tensor v, bool is_causal,
                    double softmax_scale, int64_t head_dim, bool is_bf16) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q,k,v must be CUDA tensors");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
              "q,k,v must be contiguous (B,S,H,d)");
  at::Tensor out = torch::empty_like(q);
  c10::optional<at::Tensor> out_opt = out;
  c10::optional<at::Tensor> alibi_opt = c10::nullopt;
  c10::optional<at::Generator> gen_opt = c10::nullopt;

  auto dispatch = [&](auto dtype_tag, int cta_m, int cta_n) {
    using T = decltype(dtype_tag);
    if (head_dim == 64) {
      if (is_causal) {
        custom_mha_fwd_causal<T, 64, 128, 128, 4>(
            q, k, v, out_opt, alibi_opt, 0.0, (float)softmax_scale, -1, -1, 0.0,
            false, gen_opt, 0);
      } else {
        custom_mha_fwd_noncausal<T, 64, 128, 128, 4>(
            q, k, v, out_opt, alibi_opt, 0.0, (float)softmax_scale, -1, -1, 0.0,
            false, gen_opt, 0);
      }
    } else if (head_dim == 128) {
      if (is_causal) {
        custom_mha_fwd_causal<T, 128, 128, 64, 4>(
            q, k, v, out_opt, alibi_opt, 0.0, (float)softmax_scale, -1, -1, 0.0,
            false, gen_opt, 0);
      } else {
        custom_mha_fwd_noncausal<T, 128, 128, 64, 4>(
            q, k, v, out_opt, alibi_opt, 0.0, (float)softmax_scale, -1, -1, 0.0,
            false, gen_opt, 0);
      }
    } else {
      TORCH_CHECK(false, "fat_fwd: only head_dim in {64,128} compiled, got ", head_dim);
    }
  };

  at::cuda::CUDAGuard device_guard{(char)q.get_device()};
  if (is_bf16) {
    dispatch(cute::bfloat16_t{}, 0, 0);
  } else {
    dispatch(cute::half_t{}, 0, 0);
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fat_fwd", &fat_fwd, "FlashAttention-T custom forward (headdim 64/128, fp16/bf16)");
}
