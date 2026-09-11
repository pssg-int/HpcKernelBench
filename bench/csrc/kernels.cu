// Baseline CUDA kernels for the spmv / spmm / sddmm tracks.
//
// These are the "known-quantity" entries of each leaderboard: straightforward
// warp-per-row formulations with no format changes and no tuning. They exist so
// that (a) the harness has a real device implementation to exercise end to end,
// and (b) every clever kernel has an honest, understandable floor to beat.
//
// Built as a shared library and driven from Python via ctypes with device
// pointers supplied by torch, so no CUDA-side allocation happens inside a timed
// region.

#include <cuda_runtime.h>
#include <cstdint>

#define WARP_SIZE 32

// ---------------------------------------------------------------- SpMV
// y[i] = sum_j A[i,j] * x[j]. One warp per row; segmented reduction in-warp.
template <typename T>
__global__ void spmv_csr_warp_kernel(int rows, const int* __restrict__ indptr,
                                     const int* __restrict__ indices,
                                     const T* __restrict__ data,
                                     const T* __restrict__ x,
                                     T* __restrict__ y) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  if (warp_id >= rows) return;

  const int lo = indptr[warp_id];
  const int hi = indptr[warp_id + 1];
  T sum = T(0);
  for (int k = lo + lane; k < hi; k += WARP_SIZE) {
    sum += data[k] * x[indices[k]];
  }
  // warp reduction
  #pragma unroll
  for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
    sum += __shfl_down_sync(0xffffffff, sum, off);
  }
  if (lane == 0) y[warp_id] = sum;
}

// ---------------------------------------------------------------- SpMM
// C[i, :] = sum_j A[i,j] * B[j, :]. One warp per row, lanes stride the N axis;
// the row's nonzeros are broadcast so B is read coalesced.
template <typename T>
__global__ void spmm_csr_warp_kernel(int rows, int N,
                                     const int* __restrict__ indptr,
                                     const int* __restrict__ indices,
                                     const T* __restrict__ data,
                                     const T* __restrict__ B,
                                     T* __restrict__ C) {
  const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  if (warp_id >= rows) return;

  const int lo = indptr[warp_id];
  const int hi = indptr[warp_id + 1];

  for (int n0 = 0; n0 < N; n0 += WARP_SIZE) {
    const int n = n0 + lane;
    T acc = T(0);
    for (int k = lo; k < hi; ++k) {
      const int col = indices[k];
      const T v = data[k];
      if (n < N) acc += v * B[(int64_t)col * N + n];
    }
    if (n < N) C[(int64_t)warp_id * N + n] = acc;
  }
}

// ---------------------------------------------------------------- SDDMM
// P[i,j] = S[i,j] * dot(A[i,:], B[j,:]) for (i,j) in nnz(S).
// One warp per nonzero: lanes split the K-length dot product.
template <typename T>
__global__ void sddmm_csr_warp_kernel(int rows, int K,
                                      const int* __restrict__ indptr,
                                      const int* __restrict__ indices,
                                      const T* __restrict__ data,
                                      const T* __restrict__ A,
                                      const T* __restrict__ B,
                                      T* __restrict__ P) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warps_per_block = blockDim.x / WARP_SIZE;
  const int warp_in_block = threadIdx.x / WARP_SIZE;

  const int lo = indptr[row];
  const int hi = indptr[row + 1];

  for (int k = lo + warp_in_block; k < hi; k += warps_per_block) {
    const int col = indices[k];
    T acc = T(0);
    for (int d = lane; d < K; d += WARP_SIZE) {
      acc += A[(int64_t)row * K + d] * B[(int64_t)col * K + d];
    }
    #pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
      acc += __shfl_down_sync(0xffffffff, acc, off);
    }
    if (lane == 0) P[k] = acc * data[k];
  }
}

// ------------------------------------------------------- C launch wrappers
extern "C" {

void spmv_csr_warp_f32(int rows, const int* indptr, const int* indices,
                       const float* data, const float* x, float* y,
                       cudaStream_t stream) {
  const int threads = 128;
  const int warps_per_block = threads / WARP_SIZE;
  const int blocks = (rows + warps_per_block - 1) / warps_per_block;
  spmv_csr_warp_kernel<float><<<blocks, threads, 0, stream>>>(
      rows, indptr, indices, data, x, y);
}

void spmv_csr_warp_f64(int rows, const int* indptr, const int* indices,
                       const double* data, const double* x, double* y,
                       cudaStream_t stream) {
  const int threads = 128;
  const int warps_per_block = threads / WARP_SIZE;
  const int blocks = (rows + warps_per_block - 1) / warps_per_block;
  spmv_csr_warp_kernel<double><<<blocks, threads, 0, stream>>>(
      rows, indptr, indices, data, x, y);
}

void spmm_csr_warp_f32(int rows, int N, const int* indptr, const int* indices,
                       const float* data, const float* B, float* C,
                       cudaStream_t stream) {
  const int threads = 128;
  const int warps_per_block = threads / WARP_SIZE;
  const int blocks = (rows + warps_per_block - 1) / warps_per_block;
  spmm_csr_warp_kernel<float><<<blocks, threads, 0, stream>>>(
      rows, N, indptr, indices, data, B, C);
}

void sddmm_csr_warp_f32(int rows, int K, const int* indptr, const int* indices,
                        const float* data, const float* A, const float* B,
                        float* P, cudaStream_t stream) {
  const int threads = 256;
  sddmm_csr_warp_kernel<float><<<rows, threads, 0, stream>>>(
      rows, K, indptr, indices, data, A, B, P);
}

int kernels_last_error() { return (int)cudaGetLastError(); }

}  // extern "C"
