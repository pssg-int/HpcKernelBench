// attn_binding.cu -- thin pybind11 glue WE wrote (not part of ByteTransformer's
// own repo; lives outside source/ per ARTIFACT_GUIDE rule 3) that calls
// bytetransformer::Attention<OperationType::HALF>::fused_rm_infer /
// fused_long_rm_infer DIRECTLY -- the ONE kernel invocation rule 1 asks for.
//
// Why not go through bytetransformer::torch_ext::BTEncoder (th_op/) or even
// Attention<OpType>::infer()? Both pull in far more than the attention
// kernel itself:
//   - BTEncoder additionally runs the QKV projection GEMM, the output
//     projection GEMM, both LayerNorms and the whole FFN block (gemm.cu /
//     gemm_bias_act.cu, CUTLASS-backed) -- an encoder LAYER, not a kernel.
//   - Attention<OpType>::infer() is defined inline in attention.h and its
//     body unconditionally references ALL FIVE dispatch targets
//     (nofused_infer, fused_infer, fused_rm_infer, fused_long_infer,
//     fused_long_rm_infer), even though only one branch executes at
//     runtime -- calling it would force us to also compile and link
//     attention_nofused.cu (which pulls in gemm.h/variety_attention_fused.h)
//     for a code path we never take.
// Calling fused_rm_infer/fused_long_rm_infer directly (ordinary, non-virtual
// member functions) needs only their own two translation units
// (attention_fused.cu, attention_fused_long.cu) plus attention.h/reduce.h/
// common.h -- NONE of which include cutlass_attention.h -- so this
// integration needs no CUTLASS checkout at all. Confirmed by grepping the
// whole bytetransformer/ tree for "cutlass": only bert_transformer.h,
// gemm.h/gemm_bias_act.h and their .cu's reference it; the attention_fused*
// files do not.
//
// Boundary: this wraps the fused, PADDING-FREE (remove_padding) multi-head
// self-attention kernel -- softmax(QK^T/sqrt(d)) @ V for one packed batch of
// equal-length sequences, tao=1.0, zero QKV bias (the harness's
// AttentionWorkload/reference_attention model no bias term; ByteTransformer's
// own kernel always adds a QKV bias internally, so a zero bias tensor is the
// correct way to make its computed math equal ours, not a workaround).
// `is_remove_padding_=true` unconditionally: this is the padding-free path
// the paper's own headline contribution targets. The workload this adapter
// is fed has no actual padding (every sequence in a batch has the SAME
// length, per the harness's dense (B,H,S,d) shape) -- batch_idx is simply
// [0, S, 2S, ..., B*S], the degenerate "no padding to remove" case this
// kernel's own README (variable-length inference) explicitly targets as a
// special case, not a workaround around the kernel's real design point.
//
// head_size is hardcoded to 64 in EVERY WMMA_ATTENTION_RM/WMMA_ATTENTION_LONG_RM
// macro invocation in attention_fused.cu/attention_fused_long.cu (the
// SIZE_PER_HEAD template argument is the literal `64`, not `size_per_head_`)
// -- so this kernel is only ever correct for size_per_head==64 regardless of
// what's passed to the Attention<> constructor; adapter.py enforces d==64
// before calling in, matching this repo's own only-supported configuration
// (BERT-base/large's own head_dim).
//
// seq_len coverage: fused_rm_infer's own switch only has cases for
// seq_len in {16,32,48,64,80} (attention_fused.cu); fused_long_rm_infer's
// switch only has cases for split_count 6..22, i.e. seq_len in (80,352]
// (attention_fused_long.cu). Outside [1,352], NEITHER function launches any
// kernel (the switch simply falls through, leaving `output` uninitialized) --
// adapter.py raises NotImplementedError for seq_len>352 rather than let that
// happen silently.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "bytetransformer/include/attention.h"

using namespace bytetransformer;

torch::Tensor bt_fused_rm_attention(torch::Tensor qkv, torch::Tensor qkv_bias,
                                    torch::Tensor batch_idx, int64_t batch_size,
                                    int64_t seq_len, int64_t head_num, int64_t head_size) {
  TORCH_CHECK(qkv.is_cuda() && qkv.scalar_type() == at::kHalf, "qkv must be a CUDA half tensor");
  TORCH_CHECK(qkv.is_contiguous(), "qkv must be contiguous");
  TORCH_CHECK(qkv_bias.is_cuda() && qkv_bias.scalar_type() == at::kHalf,
              "qkv_bias must be a CUDA half tensor");
  TORCH_CHECK(qkv_bias.is_contiguous(), "qkv_bias must be contiguous");
  TORCH_CHECK(batch_idx.is_cuda() && batch_idx.scalar_type() == at::kInt,
              "batch_idx must be a CUDA int32 tensor");
  TORCH_CHECK(head_size == 64,
              "ByteTransformer's fused_rm_infer/fused_long_rm_infer hardcode "
              "SIZE_PER_HEAD=64 in every dispatch macro; got head_size=", head_size);
  TORCH_CHECK(seq_len >= 1 && seq_len <= 352,
              "ByteTransformer's fused (long) remove-padding kernel only "
              "dispatches for seq_len in [1,352] (attention_fused.cu's "
              "{16,32,48,64,80} buckets plus attention_fused_long.cu's "
              "split_count 6..22 buckets); got seq_len=", seq_len);

  auto output = torch::empty({qkv.size(0), head_num * head_size}, qkv.options());

  Attention<OperationType::HALF> attn(
      /*max_batch_size=*/(int)batch_size, /*head_num=*/(int)head_num,
      /*size_per_head=*/(int)head_size, /*max_seq_len=*/(int)seq_len,
      /*use_fused_attention=*/true, /*is_remove_padding=*/true, ModelType::Bert);

  AttentionParam<OperationType::HALF> param;
  param.attr_bias_QKV = reinterpret_cast<const __half *>(qkv_bias.data_ptr<at::Half>());
  param.tao = 1.0f;
  attn.initialize(param);

  AttentionInferParam<__half> infer_param{};
  infer_param.qkv = reinterpret_cast<const __half *>(qkv.data_ptr<at::Half>());
  infer_param.atten_mask = nullptr;  // dead parameter on the rm path -- see module docstring
  infer_param.attention_output = reinterpret_cast<__half *>(output.data_ptr<at::Half>());
  infer_param.buf = nullptr;  // rm kernels use no scratch buffer
  infer_param.batch_size = (int)batch_size;
  infer_param.seq_len = (int)seq_len;
  infer_param.cublas_handle = at::cuda::getCurrentCUDABlasHandle();  // unused by this kernel
  infer_param.stream = at::cuda::getCurrentCUDAStream().stream();
  infer_param.et_param = ET_Param{batch_idx.data_ptr<int>(), nullptr, 0};

  // Same threshold Attention<OpType>::infer() itself uses to choose between
  // the short (attention_fused.cu) and long (attention_fused_long.cu)
  // kernel family -- reproduced here since we call the two functions
  // directly instead of going through infer() (see module docstring).
  if (seq_len <= 80) {
    attn.fused_rm_infer(infer_param);
  } else {
    attn.fused_long_rm_infer(infer_param);
  }
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_rm_attention", &bt_fused_rm_attention,
        "ByteTransformer fused padding-free multi-head self-attention "
        "(Attention<HALF>::fused_rm_infer / fused_long_rm_infer)");
}
