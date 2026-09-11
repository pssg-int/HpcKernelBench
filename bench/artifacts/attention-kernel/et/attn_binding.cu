// attn_binding.cu -- thin pybind11 glue WE wrote (not part of E.T.'s own
// repo; lives outside source/ per ARTIFACT_GUIDE rule 3) calling
// `OTF_attention_kernelLauncher` -- E.T.'s dense ("On-The-Fly") self-
// attention kernel -- DIRECTLY, one call per invocation. This is the ONE
// kernel invocation ARTIFACT_GUIDE rule 1 asks for: no Encoder_* class (that
// would additionally run the tile-pruned Q/K/V/O/MLP linear layers and
// LayerNorms -- an encoder LAYER, not a kernel), no Prune_attention (the
// structured-pruned variant this task's instructions say to skip) or
// sharedQK_attention (a fused variant that consumes an EXTERNALLY
// precomputed QK product via cublasHgemmStridedBatched -- not "self-
// attention from Q,K,V" in the sense this track measures).
//
// `OTF_attention_kernelLauncher` (kernels/attention.cu) is a plain global
// function (declared in kernels/kernels.h, no namespace) taking raw `half*`
// device pointers for ONE batch item: Q,K,V,mask all (seq_len,d_model)
// row-major (heads concatenated along d_model, matching
// bytetransformer's own convention), mask ADDITIVE (0 keep / -inf masked,
// added directly to the scaled QK^T scores before softmax -- confirmed by
// reading __kernel_multi_head_full_skew_warpSFM's mask-application line).
// There is no batch dimension in its own grid (`dim3(seq_len/16, nhead)`,
// confirmed from encoder/encoder.cu's Encoder_tile::run(), which calls it
// once per ENCODER call i.e. once per single sequence) -- adapter.py loops
// over the workload's B in Python, calling this binding once per batch
// item, matching the gpa/spfa_csr adapter's identical "loop over
// independent per-item kernel calls" precedent (nothing hidden from the
// timed region).
//
// Needs no CUTLASS: attention.cu only includes utils.h (nvcuda::wmma +
// cooperative_groups, header-only) and cuda_fp16.h/cooperative_groups.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#include "kernels/kernels.h"

torch::Tensor et_otf_attention(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                               torch::Tensor mask, int64_t seq_len, int64_t d_model,
                               int64_t nhead) {
  TORCH_CHECK(Q.is_cuda() && Q.scalar_type() == at::kHalf, "Q must be a CUDA half tensor");
  TORCH_CHECK(K.is_cuda() && K.scalar_type() == at::kHalf, "K must be a CUDA half tensor");
  TORCH_CHECK(V.is_cuda() && V.scalar_type() == at::kHalf, "V must be a CUDA half tensor");
  TORCH_CHECK(mask.is_cuda() && mask.scalar_type() == at::kHalf, "mask must be a CUDA half tensor");
  TORCH_CHECK(Q.is_contiguous() && K.is_contiguous() && V.is_contiguous() && mask.is_contiguous(),
              "Q/K/V/mask must be contiguous");
  TORCH_CHECK(seq_len % 16 == 0,
              "OTF_attention_kernelLauncher's grid is dim3(seq_len/16, nhead) with no "
              "remainder handling; seq_len must be a multiple of 16, got ", seq_len);
  TORCH_CHECK((d_model / nhead) % 16 == 0,
              "the WMMA 16x16x16 fragment tiling requires head_dim=d_model/nhead to be "
              "a multiple of 16, got head_dim=", d_model / nhead);

  auto out = torch::empty({seq_len, d_model}, Q.options());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  OTF_attention_kernelLauncher(
      reinterpret_cast<half *>(out.data_ptr<at::Half>()),
      reinterpret_cast<half *>(Q.data_ptr<at::Half>()),
      reinterpret_cast<half *>(K.data_ptr<at::Half>()),
      reinterpret_cast<half *>(V.data_ptr<at::Half>()),
      reinterpret_cast<half *>(mask.data_ptr<at::Half>()),
      (int)seq_len, (int)d_model, (int)nhead, stream);

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("otf_attention", &et_otf_attention,
        "E.T. dense on-the-fly self-attention (OTF_attention_kernelLauncher), "
        "one batch item per call");
}
