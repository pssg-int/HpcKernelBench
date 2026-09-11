// wrapper.cu -- ctypes glue for Sputnik's SDDMM kernel
// (bench/artifacts/sddmm/sputnik). `source/` here is a SYMLINK to
// ../../spmm/sputnik/source (see source.provenance) -- read-only reuse of
// the spmm track's already-cloned checkout, same convention as
// artifacts/sddmm/rode/wrapper.cu.
//
// Touches zero lines of Sputnik's own kernel code: links unmodified against
// source/sputnik/sddmm/cuda_sddmm.cu.cc (compiled unmodified by build.sh),
// and adds an `extern "C"` entry point around `sputnik::CudaSddmm` (fp32).
// `SortedRowSwizzle` is ported the SAME way as ../../spmm/sputnik/wrapper.cu
// (byte-for-byte from source/sputnik/matrix_utils.cu.cc:302-317; see that
// file's header comment for why it is not compiled from there directly --
// Glog/Abseil, unavailable here, needed only by that file's UNRELATED
// SparseMatrix test/benchmark code, not by SortedRowSwizzle or the kernel).
#include "source/sputnik/sddmm/cuda_sddmm.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <vector>

extern "C" {

// Identical to ../../spmm/sputnik/wrapper.cu's copy (duplicated rather than
// shared via a header, to keep this directory's build.sh self-contained and
// independent of the spmm directory's own build products).
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

// sputnik::CudaSddmm (fp32; sampled dense-dense product, S-PATTERN only --
// output_values[nz] = dot(lhs_matrix[row], rhs_matrix[col]), NOT scaled by
// any sparse value; the harness's own reference_sddmm multiplies by the
// original matrix's value afterward, matching this codebase's existing
// TorchSDDMM/CustomSDDMM convention -- see adapter.py). Dispatches
// internally on k%4/k%2 (source/sputnik/sddmm/cuda_sddmm.cu.cc:169-195), so
// any K is supported directly. All pointer args are DEVICE pointers.
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
