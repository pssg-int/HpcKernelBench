// wrapper.cu -- ctypes glue for MP-SpMM's sparse-tensor-core (2:4) SpMM
// kernel (bench/artifacts/spmm/mp-spmm). Touches ZERO lines of the
// artifact's kernel code (ARTIFACT_GUIDE.md rule 1/3): it #includes
// kernels.cu UNMODIFIED (exactly the way the artifact's own
// source/mpspmm/SpMM/spmm_sp_new.cu does: "#include kernels.cu"), so the
// __global__ kernels compiled here are the artifact's own, byte for byte.
//
// The artifact ships no library -- source/mpspmm/SpMM/spmm_sp_new.cu is a
// standalone `main()` that reads a preprocessed binary from disk, launches
// one of two kernels depending on N, and prints a GFLOP/s number. There is
// no separately-linkable object to declare `extern "C"` against (InferFast's
// pattern), so this file instead compiles kernels.cu itself into a small
// .so and exposes two thin functions:
//
//   mpspmm_pack_metadata(...)  -- device-independent bit-packing of the
//       preprocessing tool's raw per-element 2-bit selector array into the
//       32-bit words `mma.sp.sync`'s metadata operand expects. This is a
//       BYTE-FOR-BYTE copy of `valueToStorage` / `storeArrayInUint32` /
//       `storeArrayInUint32ForLargeMatrix` from
//       source/mpspmm/SpMM/spmm_sp_new.cu (lines ~24-70), only with the
//       `std::vector<int>` parameter changed to a raw pointer+length pair
//       so ctypes can call it (spmm_sp_new.cu defines these functions
//       inline in the same file as its own main(), so they cannot be
//       linked against directly -- same rationale as InferFast's wrapper.cu
//       re-declaring `reorder_matrices`, ARTIFACT_GUIDE.md rule 1's
//       "wrap at the finest boundary available").
//   mpspmm_run(...)  -- launches sparse_mma_kernel_base_Bhalf2_Cfloat4
//       (N=32) or sparse_mma_kernel_base_Buint64_Cfloat4 (N=128), with the
//       EXACT SAME grid/block/cudaFuncSetAttribute call spmm_sp_new.cu's
//       main() uses. For any other N it returns -1 without launching
//       anything -- this mirrors spmm_sp_new.cu's own `if (N==32) ... else
//       if (N==128) ...` with no `else` branch (the artifact itself has no
//       kernel for other N; adapter.py turns this into a NotImplementedError
//       from prepare(), per ARTIFACT_GUIDE.md rule 8, rather than silently
//       handing the gate an all-zero buffer).
#include "source/mpspmm/SpMM/kernels.cu"
#include <cstdint>
#include <cstring>

// ---- verbatim from source/mpspmm/SpMM/spmm_sp_new.cu (see header above) ----
namespace mpspmm_meta {

int valueToStorage(int value) {
    switch (value) {
        case 0: return 0b00;
        case 1: return 0b01;
        case 2: return 0b10;
        case 3: return 0b11;
        default: return 0;
    }
}

void storeArrayInUint32(const int *metadata_old, uint32_t *metadata_new,
                         int block_row, int block_col, int block_start) {
  int start_row = (block_start + block_col) * tile_M * tile_K / 2;
  for (int r = 0; r < 16 / 2; ++r) {
      for (int c = 0; c < 8 / 2; ++c) {
          int value_lower0 = metadata_old[start_row + r *  tile_K/2  + c * 2];
          int value_lower1 = metadata_old[start_row + r *  tile_K/2 + c * 2 + 1];
          int value_upper0 = metadata_old[start_row + (r+8) * tile_K/2 + c * 2];
          int value_upper1 = metadata_old[start_row + (r+8) *  tile_K/2 + c * 2 + 1];
          int storageValue_lower0 = valueToStorage(value_lower0);
          int storageValue_lower1 = valueToStorage(value_lower1);
          int storageValue_upper0 = valueToStorage(value_upper0);
          int storageValue_upper1 = valueToStorage(value_upper1);
          int bit_pos_lower0 = c * 4;
          int bit_pos_lower1 = c * 4 + 2;
          int bit_pos_upper0 = c * 4 + 16;
          int bit_pos_upper1 = c * 4 + 16 + 2;
          int idx = r + (block_start + block_col) * 8;
          metadata_new[idx] |= (storageValue_lower0 << bit_pos_lower0);
          metadata_new[idx] |= (storageValue_lower1 << bit_pos_lower1);
          metadata_new[idx] |= (storageValue_upper0 << bit_pos_upper0);
          metadata_new[idx] |= (storageValue_upper1 << bit_pos_upper1);
      }
  }
}

// Original signature took `std::vector<int> tcblocks_offset`; changed to
// (pointer, length) so this is ctypes-callable -- logic body unchanged.
void storeArrayInUint32ForLargeMatrix(const int *metadata_old, uint32_t *metadata_new,
                                       const int *tcblocks_offset, int tcblocks_offset_len) {
  for (int block_row = 0; block_row < tcblocks_offset_len - 1; block_row++) {
    int block_start = tcblocks_offset[block_row];
    int block_count_each_window = tcblocks_offset[block_row + 1] - tcblocks_offset[block_row];
    for (int block_col = 0; block_col < block_count_each_window; ++block_col) {
      storeArrayInUint32(metadata_old, metadata_new, block_row, block_col, block_start);
    }
  }
}

} // namespace mpspmm_meta

extern "C" {

// metadata_old: length metadata_old_len (int, values 0..3). metadata_new_out:
// caller-allocated, zeroed, length metadata_old_len/16 (uint32_t) -- matches
// spmm_sp_new.cu main()'s `malloc` + `memset(...,0,...)` before calling
// storeArrayInUint32ForLargeMatrix (the packing uses `|=`, so it depends on
// starting from zero).
void mpspmm_pack_metadata(const int* metadata_old, int metadata_old_len,
                           const int* tcblocks_offset, int tcblocks_offset_len,
                           uint32_t* metadata_new_out) {
  (void)metadata_old_len;
  mpspmm_meta::storeArrayInUint32ForLargeMatrix(metadata_old, metadata_new_out,
                                                 tcblocks_offset, tcblocks_offset_len);
}

// Launches the artifact's own kernel for N==32 or N==128, identical
// grid/block/cudaFuncSetAttribute call to spmm_sp_new.cu's main(). All
// pointers are device pointers. Returns 0 (cudaSuccess) on success, -1 for
// unsupported N (no kernel launched -- see header comment), or a
// cudaError_t code otherwise.
int mpspmm_run(int M, int N, int K,
               const void* opd_A_dev, const void* metadata_dev,
               const void* tcblocks_offset_dev, const void* old_col_all_dev,
               void* B_dev, void* C_dev) {
  int num_row_windows = (M + tile_M - 1) / tile_M;
  dim3 grid(num_row_windows, 1, 1);
  int warpCount = (N + tile_N - 1) / tile_N;

  if (N == 32) {
    dim3 block_half2(WARP_SIZE, warpCount / 2, 1);
    cudaFuncSetAttribute(sparse_mma_kernel_base_Bhalf2_Cfloat4,
                          cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    sparse_mma_kernel_base_Bhalf2_Cfloat4<<<grid, block_half2>>>(
        M, N, K, (const half*)opd_A_dev, (const uint32_t*)metadata_dev,
        (const int*)tcblocks_offset_dev, (const int*)old_col_all_dev,
        (half*)B_dev, (float*)C_dev);
  } else if (N == 128) {
    dim3 block_uint64(WARP_SIZE, warpCount / 4, 1);
    cudaFuncSetAttribute(sparse_mma_kernel_base_Buint64_Cfloat4,
                          cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    sparse_mma_kernel_base_Buint64_Cfloat4<<<grid, block_uint64>>>(
        M, N, K, (const half*)opd_A_dev, (const uint32_t*)metadata_dev,
        (const int*)tcblocks_offset_dev, (const int*)old_col_all_dev,
        (half*)B_dev, (float*)C_dev);
  } else {
    return -1;  // mirrors spmm_sp_new.cu main(): no kernel exists for this N
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return (int)err;
  err = cudaDeviceSynchronize();
  return (int)err;
}

} // extern "C"
