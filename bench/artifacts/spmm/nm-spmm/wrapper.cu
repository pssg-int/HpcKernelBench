// wrapper.cu -- ctypes glue for NM-SpMM (bench/artifacts/spmm/nm-spmm).
//
// Touches zero lines of the artifact's own numeric/preprocessing code: links
// unmodified against source/src/preprocessing.cu and
// source/src/kernel_32x32_4x4.cu (compiled unmodified by build.sh), and only
// adds `extern "C"` entry points around three of the artifact's own
// functions (declared in source/include/NM-SpMM.h, unmodified) -- same
// "wrap the finest boundary, don't reimplement" pattern as
// rode/sspmm's wrapper.cu.
//
// Scope (see STATUS.md for the full survey of what NM-SpMM supports and why
// only this slice is wired): of the artifact's 4 kernel tile sizes (32x32,
// 32x64, 64x64, 64x128) x 2 sparsity regimes (low: {0.5,0.625} via
// PreProcessing_low_sparsity alone; high: {0.75,0.875} via transIndex +
// PreProcessing_high_sparsity + a column_info side array), this wrapper
// exposes ONLY the 32x32 tile (kernel_32x32_4x4.cu's
// nmGEMM_small_matrices_low_sparsity) at sparsity=0.5 (the artifact's own
// "16 kept of every 32" N:M block -- see adapter.py's module docstring for
// why this reduces to a 1:2 ratio at block-width 32, not the finer-grained
// 2:4 most "N:M sparsity" literature means). This keeps the wired slice
// small and auditable while still exercising a real, unmodified GPU kernel
// launch + the artifact's own preprocessing function end to end.
#include "source/include/NM-SpMM.h"

extern "C" {

// Layout-transform preprocessing for the low-sparsity path (source/src/
// preprocessing.cu::PreProcessing_low_sparsity, unmodified) -- reorders the
// "DT" index array (already populated by adapter.py's Python port of
// init_data()'s INDEX generation, see adapter.py) into the blocked layout
// the GPU kernel reads from shared memory. Runs on the CPU (host pointer),
// exactly as the artifact's own test driver calls it before any H2D copy.
void nm_preprocess_low(int* DT, int W, int Q, int Ns) {
  PreProcessing_low_sparsity(DT, W, Q, Ns);
}

// nmGEMM_small_matrices_low_sparsity (source/src/kernel_32x32_4x4.cu,
// unmodified) -- dispatches to kernel_32x32_4x4_low_sparsity<Ms=32,Ns=32,
// Ks=32,Ws=16,...> for sparsity=0.5 (the only branch this wrapper's caller
// uses; see adapter.py). All pointer arguments are DEVICE pointers except
// none -- this function itself launches the kernel and returns immediately;
// the caller must synchronize, which we do here (matching sspmm/rode's
// wrapper convention of returning a cudaError_t int rather than leaving
// synchronization to the Python side).
int nm_gemm_32x32_low(float* A, float* B, int* D, float* C,
                       int M, int N, int K, int W,
                       float sparsity, int SPLIT_K) {
  nmGEMM_small_matrices_low_sparsity(A, B, D, C, M, N, K, W, sparsity, SPLIT_K);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return (int)err;
  err = cudaDeviceSynchronize();
  return (int)err;
}

}  // extern "C"
