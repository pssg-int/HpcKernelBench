// wrapper.cu -- ctypes-facing glue for RoDe's SpMM kernel.
//
// Kernel wrapped (ARTIFACT_GUIDE.md rule 1): `RoDeSpmm_n32` / `RoDeSpmm_n128`
// from source/RoDe_SpMM/RoDeSpmm.cu + RoDeSpmm.h -- compiled and linked here
// completely UNMODIFIED. These two host launcher functions (and everything
// they call: RoDeSpmmKernel, RoDeComputeKernel1/2, SparseKernel) take only
// raw pointers/ints -- confirmed by RoDeSpmm.cu's own #include list
// (basic_utils.h, cuda_runtime.h, common_utils.h -- NOT matrix_utils.h),
// i.e. the actual numeric kernel has zero dependency on the SPC::SparseMatrix
// class hierarchy or its abseil/glog/gflags toolchain.
//
// Preprocessing note (ARTIFACT_GUIDE.md rule 2/3): RoDe's row-decomposition
// preprocessing -- the paper's actual contribution -- lives in
// SPC::SparseMatrix::RowDivide2Segment (source/utils/matrix_utils.cu:989-1041).
// That class DOES pull in abseil-cpp (random) + Glog + gflags (see
// source/CMakeLists.txt's cmake/Dependencies.cmake: `find_package(Glog
// REQUIRED)`, `add_subdirectory(third_party/abseil-cpp)`) purely for
// unrelated host-side logging (CHECK_LE/CHECK_GE) and a random-fill
// constructor we would only overwrite anyway (this adapter supplies REAL
// matrix values, not RoDe's own random-fill path). Rather than vendor and
// build three extra dependency chains for host-side bookkeeping the numeric
// kernel never touches, `row_divide_to_segment` below is a byte-for-byte
// port of RowDivide2Segment's algorithm (identical loop structure, same
// SegmentLength/vectorLen/KBLOCK semantics, same output arrays) operating
// directly on our CSR row_offsets array -- no GPU, no randomness, no
// logging. This is the artifact's own preprocessing ALGORITHM, timed as
// preprocessing in adapter.py's prepare() exactly per the harness contract;
// only the class scaffolding around it (which exists to serve RoDe's OWN
// benchmark driver reading .mtx files with random fallback values) is not
// linked. See STATUS.md for the full rationale.
//
// rode_run() calls the real, compiled RoDeSpmm_n32/n128 for the actual
// SpMM computation -- nothing about the numeric kernel is reimplemented.

#include <cuda_runtime.h>
#include <vector>
#include <cstring>

#include "RoDeSpmm.h"  // source/RoDe_SpMM/RoDeSpmm.h, unmodified

extern "C" {

struct RodeHandle {
    int rows, cols, nnz;
    int n_segs, n_segs_residue;
    int* row_offsets_d;
    int* col_indices_d;
    float* values_d;
    int* seg_row_indices_d;         // device, length n_segs
    int* seg_st_offsets_d;          // device, length n_segs+1
    int* seg_row_indices_residue_d; // device, length n_segs_residue (maybe null)
};

// Verbatim port of SPC::SparseMatrix::RowDivide2Segment
// (source/utils/matrix_utils.cu:989-1041) -- see header comment above.
static void row_divide_to_segment(
        int rows, const int* row_offsets, int SegmentLength, int vectorLen, int KBLOCK,
        std::vector<int>& seg_row_indices, std::vector<int>& seg_st_offsets,
        std::vector<int>& seg_row_indices_residue) {
    for (int i = 0; i < rows; ++i) {
        int row_offset = row_offsets[i];
        int n_padding = row_offset % vectorLen;
        int nnz = row_offsets[i + 1] - row_offset + n_padding;

        if (nnz > SegmentLength) {
            seg_row_indices.push_back(i);
            seg_st_offsets.push_back(row_offset);
            row_offset = (row_offset + SegmentLength) - n_padding;
            nnz -= SegmentLength;
        }
        while (nnz > SegmentLength) {
            seg_row_indices.push_back(i);
            seg_st_offsets.push_back(row_offset);
            row_offset += SegmentLength;
            nnz -= SegmentLength;
        }
        if (nnz > 0) {
            if (nnz >= KBLOCK) {
                seg_row_indices.push_back(i);
                seg_st_offsets.push_back(row_offset);
            }
            if (nnz % KBLOCK) {
                seg_row_indices_residue.push_back(i);
            }
        }
    }
    seg_st_offsets.push_back(row_offsets[rows]);
}

// rows/cols/nnz + host CSR arrays -> device handle with RoDe's own
// row-decomposition metadata built (SegmentLength=512, vectorLen=4,
// KBLOCK=32 -- literally RoDe's own eval driver's call:
// `sm1.RowDivide2Segment(SEG_LENGTH,4,32)` with `#define SEG_LENGTH 512` in
// source/eval/eval_spmm_f32_n128.cu).
void* rode_prepare(int rows, int cols, int nnz,
                    const int* row_offsets_h, const int* col_indices_h,
                    const float* values_h) {
    std::vector<int> seg_idx, seg_off, seg_res;
    row_divide_to_segment(rows, row_offsets_h, 512, 4, 32, seg_idx, seg_off, seg_res);

    RodeHandle* h = new RodeHandle();
    h->rows = rows; h->cols = cols; h->nnz = nnz;
    h->n_segs = (int)seg_idx.size();
    h->n_segs_residue = (int)seg_res.size();

    cudaMalloc((void**)&h->row_offsets_d, sizeof(int) * (rows + 1));
    cudaMalloc((void**)&h->col_indices_d, sizeof(int) * nnz);
    cudaMalloc((void**)&h->values_d, sizeof(float) * nnz);
    cudaMemcpy(h->row_offsets_d, row_offsets_h, sizeof(int) * (rows + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(h->col_indices_d, col_indices_h, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(h->values_d, values_h, sizeof(float) * nnz, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&h->seg_row_indices_d, sizeof(int) * h->n_segs);
    cudaMalloc((void**)&h->seg_st_offsets_d, sizeof(int) * (h->n_segs + 1));
    cudaMemcpy(h->seg_row_indices_d, seg_idx.data(), sizeof(int) * h->n_segs, cudaMemcpyHostToDevice);
    cudaMemcpy(h->seg_st_offsets_d, seg_off.data(), sizeof(int) * (h->n_segs + 1), cudaMemcpyHostToDevice);

    if (h->n_segs_residue > 0) {
        cudaMalloc((void**)&h->seg_row_indices_residue_d, sizeof(int) * h->n_segs_residue);
        cudaMemcpy(h->seg_row_indices_residue_d, seg_res.data(),
                   sizeof(int) * h->n_segs_residue, cudaMemcpyHostToDevice);
    } else {
        h->seg_row_indices_residue_d = nullptr;
    }
    return (void*)h;
}

int rode_n_segs(void* handle) { return ((RodeHandle*)handle)->n_segs; }
int rode_n_segs_residue(void* handle) { return ((RodeHandle*)handle)->n_segs_residue; }
int rode_last_cuda_error() { return (int)cudaGetLastError(); }

// ONE SpMM kernel invocation -- N in {32, 128} only (the only two
// instantiations RoDe ships, RoDeSpmm_n32/RoDeSpmm_n128). Returns
// cudaError_t (0 = success) from cudaGetLastError() after both streams'
// kernel launches (Kernel1 = segmented rows, Kernel2 = residue tail; RoDe's
// own main() launches both on separate streams -- s1/s2 here may be the
// same stream (0) or two real streams created by the caller).
int rode_run(void* handle, int N, const float* B, float* C,
             cudaStream_t s1, cudaStream_t s2) {
    RodeHandle* h = (RodeHandle*)handle;
    if (N == 32) {
        RoDeSpmm_n32(h->n_segs, h->n_segs_residue, h->cols, N,
                     h->values_d, h->col_indices_d, h->row_offsets_d,
                     h->seg_row_indices_d, h->seg_row_indices_residue_d, h->seg_st_offsets_d,
                     B, C, s1, s2);
    } else if (N == 128) {
        RoDeSpmm_n128(h->n_segs, h->n_segs_residue, h->cols, N,
                      h->values_d, h->col_indices_d, h->row_offsets_d,
                      h->seg_row_indices_d, h->seg_row_indices_residue_d, h->seg_st_offsets_d,
                      B, C, s1, s2);
    } else {
        return -1;
    }
    cudaError_t err = cudaGetLastError();
    // RoDeSpmmKernel (source/RoDe_SpMM/RoDeSpmm.cu:492-506, unmodified)
    // always launches BOTH Kernel1 (main segments, grid.x = ceil(m1/kBlockItemsY))
    // and Kernel2 (residue, grid.x = ceil(m2/kBlockItemsY)) unconditionally.
    // For matrices where every row has fewer nonzeros than KBLOCK=32 (e.g.
    // this track's GNN-degree graphs), RowDivide2Segment produces n_segs==0
    // (m1=0) and everything lands in the residue set -- confirmed by direct
    // ctypes testing (see STATUS.md): Kernel1's launch with grid=(0,*,*)
    // deterministically raises cudaErrorInvalidConfiguration(9) on this
    // CUDA/driver combination, but Kernel2 (a VALID, non-empty launch on the
    // same stream) still executes and produces numerically correct output
    // regardless (verified: max abs err ~2e-6 against an fp64 CPU reference,
    // both with and without this condition). This is RoDe's OWN kernel
    // launcher's behavior for an edge case its own eval driver never
    // exercises (SuiteSparse .mtx matrices with min row nnz >= 32); it is
    // not something this adapter introduces, and the numeric result is
    // unaffected -- so this ONE specific, structurally-predictable error
    // code is treated as expected/benign here rather than surfaced as a
    // kernel failure. Any OTHER error code, or this same code when
    // n_segs != 0 (i.e. not explained by an empty grid), is still reported.
    if (err == cudaErrorInvalidConfiguration && h->n_segs == 0) {
        return 0;
    }
    return (int)err;
}

void rode_free(void* handle) {
    RodeHandle* h = (RodeHandle*)handle;
    cudaFree(h->row_offsets_d);
    cudaFree(h->col_indices_d);
    cudaFree(h->values_d);
    cudaFree(h->seg_row_indices_d);
    cudaFree(h->seg_st_offsets_d);
    if (h->seg_row_indices_residue_d) cudaFree(h->seg_row_indices_residue_d);
    delete h;
}

}  // extern "C"
