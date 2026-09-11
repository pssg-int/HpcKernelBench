// shim.cu -- NOT part of the YuenyeungSpTRSV artifact. Thin extern "C"
// host-side launchers around the artifact's OWN unmodified code
// (source/CUDA/tranpose.h's matrix_warp -- the host-side thread/warp
// partitioning "analysis" -- and source/CUDA/YYSpTRSV.h's
// yySpTRSV_csr_kernel), included verbatim below. Same role
// bench/artifacts/spmv/sspmv/csr_shim.cpp plays for LeSpMV. See
// adapter.py's module docstring for the full design rationale.
#include <cuda_runtime.h>
#include <cuda_fp16.h>   // force CUDA's own __shfl_down(half/half2,...,width)
                         // overloads to be declared BEFORE the scoped macro
                         // below shadows the name -- see the macro's comment.

// ---------------------------------------------------------------------
// Build-compat fix (CUDA-12 removed API), NOT a kernel algorithm change:
// YYSpTRSV.h line 91 calls the pre-CUDA-9 unmasked `__shfl_down(val,
// offset)`, removed by CUDA 9+ in favor of `__shfl_down_sync(mask, val,
// offset)`. Every thread in the warp participates unconditionally at that
// call site (no divergent branch before it in yySpTRSV_csr_kernel's
// warp-level path), so the full-warp mask 0xffffffff is exactly what the
// old unmasked form implicitly assumed on pre-Volta independent-thread-
// scheduling hardware -- same semantics, not an algorithm change. Scoped
// to ONLY the #include below (immediately #undef'd) so nothing else in
// this translation unit -- including CUDA's own genuinely-3-argument
// __shfl_down(__half/__half2, delta, width) overloads pulled in by
// <cuda_fp16.h> above -- is affected. The vendored file itself is never
// touched (see STATUS.md for the confirmed-necessary probe build without
// this macro, which fails with "identifier __shfl_down is undefined").
// ---------------------------------------------------------------------
#define __shfl_down(val, offset) __shfl_down_sync(0xffffffff, (val), (offset))
#include "tranpose.h"
#include "YYSpTRSV.h"
#undef __shfl_down

extern "C" {

// One-shot structural analysis: wraps the artifact's own matrix_warp
// (host-side, pure CPU -- no CUDA needed for this step) unmodified. The
// thread/warp-level partition border matches the artifact's own main.cu
// default (border=10). Called ONCE, from adapter.py's prepare().
void yy_warp_partition(int m, int n, int nnzL, const int *csrRowPtr,
                        const int *csrColIdx, const VALUE_TYPE *csrVal,
                        int *Len, int *warp_num, double *warp_occupy,
                        double *element_occupy) {
    matrix_warp(m, n, nnzL, csrRowPtr, csrColIdx, csrVal, /*border=*/10,
                Len, warp_num, warp_occupy, element_occupy);
}

// One genuine solve call: reset the three pieces of per-call algorithm
// state the kernel mutates (the "ready" flag array d_get_value, the
// dynamic-work-distribution atomic counter d_id_extractor, and d_x
// itself) -- matching the artifact's OWN correct per-iteration reset
// practice in YYSpTRSV_csr (source/CUDA/YYSpTRSV.h lines 213-215: exactly
// this cudaMemset trio before every timed solve), which
// benchspecs/sptrsv/spec.yaml explicitly cites and sanctions as belonging
// INSIDE the timed region ("this matches YuenyeungSpTRSV's correct
// cudaMemset-before-timing practice"). Then launch the artifact's own
// yySpTRSV_csr_kernel unmodified.
void yy_solve(const int *d_csrRowPtr, const int *d_csrColIdx,
              const VALUE_TYPE *d_csrVal, int *d_get_value, int m, int nnzL,
              const VALUE_TYPE *d_b, VALUE_TYPE *d_x, const int *d_warp_num,
              int Len, int *d_id_extractor) {
    cudaMemsetAsync(d_get_value, 0, (size_t)m * sizeof(int));
    cudaMemsetAsync(d_x, 0, (size_t)m * sizeof(VALUE_TYPE));
    cudaMemsetAsync(d_id_extractor, 0, sizeof(int));

    int num_threads = WARP_PER_BLOCK * WARP_SIZE;
    int num_blocks = (int)ceil(((double)(Len - 1) * WARP_SIZE) / (double)num_threads);
    if (num_blocks >= 1)
        yySpTRSV_csr_kernel<<<num_blocks, num_threads>>>(
            d_csrRowPtr, d_csrColIdx, d_csrVal, d_get_value, m, nnzL, d_b,
            d_x, 0, d_warp_num, Len, d_id_extractor);
}

int yy_sync() {
    cudaDeviceSynchronize();
    return (int)cudaGetLastError();
}

} // extern "C"
