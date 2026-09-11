// Thin C-linkage wrapper around Tetris's own SparseFilter/SparseConv2d entry
// points (source/Tetris/spconv2d_utils.h, source/Tetris/spconv2d_kernel.cuh),
// so kernelbench's ctypes-based adapter idiom (see
// ../../../kernelbench/impls/gpu_cuda.py's _load_lib()) can call them.
//
// NOT a modification of the artifact: this is a NEW translation unit that
// only #includes the artifact's headers and calls its existing, unmodified
// template class/function. No line of Tetris/*.{cc,cu,cuh,h} is edited.
//
// Kernel config: SparseConv2d<float,int,int, /*K=*/3,/*S=*/1,/*TILE_IC=*/128,
// /*TILE_H=*/2,/*TILE_W=*/4> -- the exact single instantiation the artifact's
// OWN "ablation" driver (source/Tetris/spf.cc) uses for its
// SparseConv2d_SPF() example, and one of the many pre-instantiated tile
// configs in source/Tetris/spconv2d_kernel.cu (see
// TETRIS_INSTANTIATE_TILED_FLOAT(SparseConv2d, 3, 1, 128, 2, 4)). Picking
// this single fixed instantiation (rather than spconv2d.cc's ~100-way
// autotuning dispatch table over tile configs) keeps the wrapper small and
// avoids re-implementing that search within this integration's budget; it
// means this adapter only supports K=3, stride=1 conv shapes, documented in
// STATUS.md and adapter.py.
//
// SparseFilter's apply_reorder=true path is used (matching spf.cc), compiled
// with -DBANK_OPTIMIZE -DREORDER_OUT_CHANNEL to match the artifact's own
// "fully optimized" build line for spf.cc/spconv2d.cc in
// evalution/script/build_all.sh -- both macros only affect the CPU-side SPF
// *packing strategy* inside SparseFilter's constructor (bank-conflict-aware
// slot assignment, out-channel reordering); the GPU kernel
// (SparseConv2dKernel in spconv2d_kernel.cu) is unaffected by them and reads
// whatever offsets/position/values/oc_permutation arrays the packer produced
// -- correctness does not depend on which macro combination packed the
// filter, only performance does.

#include <cuda_runtime.h>

#include "../source/Tetris/spconv2d_utils.h"
#include "../source/Tetris/spconv2d_kernel.cuh"

extern "C" {

// Builds the SPF (stride-packed-filter) sparse format from a dense OIRS
// (Cout, Cin, Kh, Kw) float filter -- this IS Tetris's own preprocessing
// step (CPU-side; the harness calls this once in prepare(), never in run()).
// `h_filter` must already be the PRUNED weight tensor (the zeros are what
// make this sparse -- SparseFilter only packs already-zero entries away, it
// does not decide which weights to zero).
void* tetris_pack_filter(float* h_filter, int in_channel, int out_channel,
                          int kernel_size) {
  return new SparseFilter<float, int, int>(
      h_filter, in_channel, out_channel, kernel_size,
      /*TILE_IC=*/128, /*apply_reorder=*/true, /*apply_vectorize=*/true);
}

void tetris_free_filter(void* handle) {
  delete static_cast<SparseFilter<float, int, int>*>(handle);
}

// ONE sparse-conv forward call (run()'s entire body): launches
// SparseConv2dKernel exactly once via Tetris's own SparseConv2d<...> host
// function. `d_input`/`d_output` are device pointers in NHWC layout (the
// layout SparseConv2dKernel indexes with -- see its `input(n,h,w,c)` /
// `output(n,h,w,c)` macros), already resident on the GPU; no allocation or
// H2D/D2H copy happens in this function, matching the harness's
// timing_scope ("kernel only; H2D/D2H excluded").
void tetris_conv_forward(void* handle, float* d_input, float* d_output,
                          int batch_size, int img_h, int img_w,
                          int in_channel, int out_channel, int kernel_size,
                          int stride, int padding, cudaStream_t stream) {
  auto* sf = static_cast<SparseFilter<float, int, int>*>(handle);
  SparseConv2d<float, int, int, /*K=*/3, /*S=*/1, /*TILE_IC=*/128,
               /*TILE_H=*/2, /*TILE_W=*/4>(
      d_input, sf->GetDeviceOffsets(), sf->GetDeviceStageLen(),
      sf->GetDeviceOCPermutation(), sf->GetDevicePosition(),
      sf->GetDeviceValues(), d_output, batch_size, img_h, img_w, in_channel,
      out_channel, kernel_size, stride, padding, stream);
}

}  // extern "C"
