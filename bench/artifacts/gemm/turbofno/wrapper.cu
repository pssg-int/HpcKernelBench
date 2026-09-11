/*
 * Thin extern-"C" launcher for TurboFNO's standalone complex GEMM kernel.
 *
 * TurboFNO ships `cgemm` as a raw `__global__` kernel *definition* in
 * source/fusion_variants/1D_E_baseline/cgemm.cuh -- there is no extern-"C"
 * host entry point and no grid/block-dim computation of its own; every
 * caller in the repo (fused.cu's own main(), the other 1D/2D fusion
 * variants) computes the launch geometry inline at the call site. This file
 * reproduces that same geometry (see turbofno_cgemm_launch below, copied
 * verbatim from fused.cu's own dim3 gridDim/blockDim/shmem_size lines) and
 * adds one extern-"C" entry point so ctypes can call it. Nothing here
 * reimplements or edits the kernel itself.
 *
 * #includes, UNMODIFIED, straight from the artifact:
 *   - source/utils/TurboFNO.h            the artifact's own tile-size
 *     macros (THREADBLOCK_M/N/K = 64/64/8, WARP_M/N = 32/16, THREAD_M/N =
 *     4/4, WARP_NUM_ROW/THREAD_NUM_ROW/LOAD_PER_THREAD_A/B derived from
 *     those, TID/WID/BID_X/BID_Y aliased to threadIdx.x/blockIdx.{x,y}).
 *     This is the baseline tile configuration the artifact's own CMake
 *     build feeds every 1D_* fusion variant via
 *     `include_directories(${PROJECT_ROOT}/utils)` -- not a config
 *     invented for this integration.
 *   - source/fusion_variants/1D_E_baseline/cgemm.cuh  the kernel itself.
 *
 * Compiled standalone (no TurboFFT submodule, no cuFFT/cuBLAS link):
 * cgemm.cuh only needs the macros above plus <mma.h>/<stdio.h> from the CUDA
 * toolkit (mma.h is included but unused -- 1D_E_baseline's cgemm is a plain
 * shared-memory-tiled kernel, no wmma:: calls), so this file + nvcc's own
 * headers are the entire dependency closure. No prebuilt TurboFNO .so to
 * link against -- there is none for cgemm alone.
 *
 * Kernel constraint (inherited from the artifact, not introduced here):
 * cgemm.cuh's global-memory indexing has no boundary/tail handling, so the
 * caller must pass M a multiple of THREADBLOCK_M(=64), N a multiple of
 * THREADBLOCK_N(=64), K a multiple of THREADBLOCK_K(=8) -- out-of-range
 * tiles would read/write past the buffer. All three of this project's
 * --smoke gemm shapes (256/256/256, 384x256x512, batch4-64x64x64) already
 * satisfy this; no adapter-side padding was needed to gate (see STATUS.md).
 *
 * Layout: cgemm.cuh indexes A as column-major (M,K), ld=M (`gA[row +
 * col*M]`) and B as column-major (K,N), ld=K (`gB[row + col*K]`), and
 * writes C column-major (M,N), ld=M -- exactly cuBLAS's own NN convention
 * (cross-checked against fused.cu's own reference call:
 * `cublasCgemm(..., dA, M, dB, K, ..., dC_ref, M)`). The Python adapter is
 * responsible for handing this launcher correctly laid-out buffers; this
 * file performs no transposition of its own.
 *
 * No batching: cgemm has no batch parameter. The adapter loops this
 * launcher once per batch element for batched shapes -- each call is an
 * independent, unmodified single-kernel invocation on a different pointer
 * offset, not a new fused/batched kernel.
 */

#include "TurboFNO.h"   // artifact's own macro definitions (unmodified)
#include "cgemm.cuh"    // artifact's own kernel (unmodified)
#include <cuda_runtime.h>

extern "C" {

// A, B, C: device pointers to float2 (complex64-layout) buffers, already
// laid out column-major as described above. alpha/beta given as separate
// re/im floats (ctypes has no native pass-by-value complex-struct type).
cudaError_t turbofno_cgemm_launch(int M, int N, int K,
                                   const void* A, const void* B, void* C,
                                   float alpha_re, float alpha_im,
                                   float beta_re, float beta_im) {
    dim3 gridDim((M + THREADBLOCK_M - 1) / THREADBLOCK_M,
                 (N + THREADBLOCK_N - 1) / THREADBLOCK_N, 1);
    dim3 blockDim((THREADBLOCK_M * THREADBLOCK_N / (THREAD_M * THREAD_N)), 1, 1);
    size_t shmem_size = sizeof(float2) * (THREADBLOCK_M + THREADBLOCK_N) * THREADBLOCK_K * 2;

    float2 alpha = make_float2(alpha_re, alpha_im);
    float2 beta = make_float2(beta_re, beta_im);

    cgemm<<<gridDim, blockDim, shmem_size>>>(
        M, N, K, (float2*)A, (float2*)B, (float2*)C, alpha, beta);
    return cudaGetLastError();
}

}  // extern "C"
