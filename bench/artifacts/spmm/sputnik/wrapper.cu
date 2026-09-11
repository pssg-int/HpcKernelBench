// wrapper.cu -- ctypes glue for Sputnik (bench/artifacts/spmm/sputnik and,
// by symlink, bench/artifacts/sddmm/sputnik -- see build.sh).
//
// Touches zero lines of Sputnik's own kernel code: links unmodified against
// source/sputnik/spmm/cuda_spmm.cu.cc and source/sputnik/sddmm/
// cuda_sddmm.cu.cc (compiled unmodified by build.sh), and adds `extern "C"`
// entry points around the paper's own `sputnik::CudaSpmm` (fp32 CSR
// overload, bias=nullptr -> plain SpMM, no ReLU) and `sputnik::CudaSddmm`
// (fp32) -- same "wrap the finest boundary, don't reimplement" pattern as
// rode/sspmm/nm-spmm's wrapper.cu files.
//
// One exception, matching RoDe's own precedent exactly: `SortedRowSwizzle`
// (source/sputnik/matrix_utils.cu.cc:302-317) is REPRODUCED here byte-for-
// byte rather than compiled from that file, because matrix_utils.cu.cc's
// OTHER functions (SparseMatrix/CudaSparseMatrix, used only by Sputnik's own
// tests/benchmarks) pull in Google Glog and Abseil -- neither vendored on
// this machine, and neither needed by the actual kernel or by
// SortedRowSwizzle itself, which is a completely self-contained ~15-line
// host-side argsort using only <vector>/<algorithm>/<numeric>/<cstring>
// (verified: grepped cuda_utils.h, common.h, spmm/*.h, sddmm/*.h, and both
// cuda_*.cu.cc kernel files for glog/LOG/CHECK -- none found; only
// matrix_utils.cu.cc's UNRELATED SparseMatrix code uses glog). This is
// preprocessing the harness times in prepare(), not the kernel itself, and
// is reproduced exactly (same argsort-by-row-length algorithm, same
// std::sort comparator, same output convention) -- see STATUS.md.
#include "source/sputnik/spmm/cuda_spmm.h"
#include "source/sputnik/sddmm/cuda_sddmm.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

extern "C" {

// Ported byte-for-byte from source/sputnik/matrix_utils.cu.cc's
// SortedRowSwizzle (see header comment above for why it is not compiled
// from that file directly). Host-side; row_offsets/row_indices are HOST
// pointers, called before any H2D copy, matching the artifact's own usage
// (SparseMatrix's constructor calls this exact function the same way).
void sputnik_sorted_row_swizzle(int rows, const int* row_offsets,
                                int* row_indices) {
  std::vector<int> swizzle_staging(rows);
  std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

  std::sort(swizzle_staging.begin(), swizzle_staging.end(),
           [&row_offsets](int idx_a, int idx_b) {
             int length_a = row_offsets[idx_a + 1] - row_offsets[idx_a];
             int length_b = row_offsets[idx_b + 1] - row_offsets[idx_b];
             return length_a > length_b;
           });

  std::memcpy(row_indices, swizzle_staging.data(), sizeof(int) * rows);
}

// sputnik::CudaSpmm (fp32 CSR-times-dense; bias=nullptr, no ReLU) --
// dispatches internally on n%4/n%2 residue handling (source/sputnik/spmm/
// cuda_spmm.cu.cc:415-497), so any N is supported directly, no adapter-side
// tile restriction needed. All pointer args are DEVICE pointers.
int sputnik_spmm_run(int m, int k, int n, int nonzeros,
                     const int* row_indices, const float* values,
                     const int* row_offsets, const int* column_indices,
                     const float* dense_matrix, float* output_matrix) {
  cudaError_t err = sputnik::CudaSpmm(
      m, k, n, nonzeros, row_indices, values, row_offsets, column_indices,
      dense_matrix, output_matrix, /*stream=*/0);
  if (err != cudaSuccess) return (int)err;
  err = cudaDeviceSynchronize();
  return (int)err;
}

// sputnik::CudaSddmm (fp32; sampled dense-dense product, S-pattern-only --
// output_values[nz] = dot(lhs_matrix[row], rhs_matrix[col]), NOT scaled by
// any input sparse value -- Sputnik's own signature has no `values` operand
// at all). Dispatches internally on k%4/k%2 (source/sputnik/sddmm/
// cuda_sddmm.cu.cc:169-195), so any K is supported directly. All pointer
// args are DEVICE pointers.
int sputnik_sddmm_run(int m, int k, int n, int nonzeros,
                      const int* row_indices, const int* row_offsets,
                      const int* column_indices, const float* lhs_matrix,
                      const float* rhs_matrix, float* output_values) {
  cudaError_t err = sputnik::CudaSddmm(
      m, k, n, nonzeros, row_indices, row_offsets, column_indices,
      lhs_matrix, rhs_matrix, output_values, /*stream=*/0);
  if (err != cudaSuccess) return (int)err;
  err = cudaDeviceSynchronize();
  return (int)err;
}

}  // extern "C"
