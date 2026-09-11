// wrapper.cu -- ctypes-facing glue for Magicube's 16-bit quantized WMMA SDDMM kernel.
//
// Kernel wrapped (ARTIFACT_GUIDE.md rule 1): `sddmm::wmmaSddmm_16b`
// (source/SDDMM/SDDMM/src/wmma_sddmm.cu + include/wmma_sddmm.cuh) --
// compiled and linked here completely UNMODIFIED.
#include <cuda_runtime.h>
#include "wmma_sddmm.cuh"

extern "C" {

int magicube_sddmm_16b(int m_vec, int k, int n,
                        const int* row_indices, const int* row_offsets,
                        const int* col_indices,
                        const int* lhs_matrix, const int* rhs_matrix,
                        int* output_values, int vec_length) {
    cudaError_t err = sddmm::wmmaSddmm_16b(
        m_vec, k, n, row_indices, row_offsets, col_indices,
        lhs_matrix, rhs_matrix, output_values, vec_length);
    return (int)err;
}

}  // extern "C"
