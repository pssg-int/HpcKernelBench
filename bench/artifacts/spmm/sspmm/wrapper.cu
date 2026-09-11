// wrapper.cu -- ctypes glue for SSpMM's own contributed kernel
// (bench/artifacts/spmm/sspmm). Touches zero lines of kernel code: links
// against sspmm.o, compiled unmodified from source/SSpMM/src/sspmm.cu by
// build.sh (via the artifact's own Makefile_ampere compile rule), and only
// declares an `extern "C"` entry point around the artifact's own
// `spmm::SSpMM(...)` host function (declared in source/SSpMM/include/
// sspmm.cuh, unmodified) -- same "wrap the finest boundary, don't
// reimplement" pattern as inferfast/mp-spmm's wrapper.cu.
//
// `spmm::SSpMM` is the paper's own contribution -- the test driver
// (sspmm_test.cpp) calls it "Ro-SpMM" and it is what this paper adds beyond
// the vectorSparse baseline also vendored in this repo (spmm::SpMM,
// "mma884", NOT wrapped here -- see STATUS.md's selection note).
//
// vec_length is fixed by this adapter to 8 (spmm::SSpMM's own
// `switch(vec_length)` only has cases 8 and 16 -- any other value hits its
// `default: printf("Unsupported Vector Length!\n");` and launches NOTHING,
// mirrored here by returning -2 without calling into the artifact at all
// for any other value, so the gate is never handed a not-really-computed
// buffer).
#include "source/SSpMM/include/sspmm.cuh"

extern "C" {

// All array pointers are DEVICE pointers. Uses spmm::SSpMM's mixed-precision
// overload (half*half -> fp32 accumulate), matching the spmm-tensorcore-fp16
// spec variant this is gated under.
int sspmm_run(int m_vec, int vec_length, int N, int K,
              const void* row_indices, const void* row_offsets,
              const void* column_indices, const void* values,
              const void* rhs_matrix, void* output_matrix) {
  if (vec_length != 8 && vec_length != 16) return -2;

  cudaError_t err = spmm::SSpMM(
      m_vec, vec_length, N, K,
      (const int*)row_indices, (const int*)row_offsets,
      (const int*)column_indices, (const half*)values,
      (const half*)rhs_matrix, (float*)output_matrix);
  if (err != cudaSuccess) return (int)err;
  err = cudaDeviceSynchronize();
  return (int)err;
}

} // extern "C"
