// wrapper.cu -- ctypes-facing glue for RoDe's SDDMM kernel.
//
// Kernel wrapped (ARTIFACT_GUIDE.md rule 1): `RoDeSDDMM_n32` / `RoDeSDDMM_n128`
// from source/RoDe_SDDMM/RoDeSddmm.cu + RoDeSddmm.h -- compiled and linked
// here completely UNMODIFIED. Confirmed by direct grep of RoDeSddmm.cu: it
// contains zero references to SPC::SparseMatrix/CudaSparseMatrix/absl:: --
// the two host launcher functions (and everything they call:
// SDDMMKernel4Block, SDDMMKernel4Residue) take only raw pointers/ints, same
// situation as RoDeSpmm_n32/n128 in the sibling spmm/rode adapter. The one
// build-time complication is that RoDeSddmm.cu itself (unlike RoDeSpmm.cu)
// has a dead `#include "matrix_utils.h"` line that would otherwise drag in
// abseil-cpp (unavailable here -- see shim_include/matrix_utils.h's header
// comment and STATUS.md for the full story); build.sh resolves it via an
// include-path substitution, not a source patch.
//
// Preprocessing (rule 2/3): RoDe's own eval driver for SDDMM calls the
// EXACT SAME row-decomposition as its SpMM eval driver --
// source/eval/eval_sddmm_f32_n32.cu: `sm1.RowDivide2Segment(SEG_LENGTH,4,32)`
// with `#define SEG_LENGTH 512` -- i.e. `SparseMatrix::RowDivide2Segment`
// (source/utils/matrix_utils.cu:989-1041), same algorithm, same parameters,
// same reason to port rather than link (that class pulls in abseil+Glog+
// gflags for host-side logging/random-fill the numeric kernel never uses).
// `row_divide_to_segment` below is the identical byte-for-byte port already
// used by ../../spmm/rode/wrapper.cu (copied here rather than shared via a
// cross-track header, per this task's "do not touch other tracks" rule --
// each artifact directory is self-contained). See that file's header
// comment / bench/artifacts/spmm/rode/STATUS.md for the full derivation;
// not re-derived here.
//
// sddmm_run() calls the real, compiled RoDeSDDMM_n32/RoDeSDDMM_n128 for the
// actual SDDMM computation -- nothing about the numeric kernel is
// reimplemented. Unlike RoDeSpmm's kernels (which atomicAdd into C),
// RoDeSDDMM's kernels Store() directly into `out` at the nonzero's own CSR
// position (`out + row_offset + n_idx` / `out + row_offset + threadIdx.x`,
// confirmed by reading RoDeSddmm.cu) and already multiply by the sparse
// value themselves (`accumulator_fragment[x] * values_tile[x]`) -- so the
// output is already S[i,j]*dot(A[i,:],B[j,:]) in the caller's original CSR
// nnz order, no post-hoc reindex or "* S.data" step needed in adapter.py
// (unlike the fused3s adapter's TC-block layout decode / unweighted-mask
// handling -- RoDe's SDDMM kernel is neither of those).

#include <cuda_runtime.h>
#include <vector>
#include <cstring>

#include "RoDeSddmm.h"  // source/RoDe_SDDMM/RoDeSddmm.h, unmodified

extern "C" {

struct RodeSddmmHandle {
    int rows, cols, nnz;
    int n_segs, n_segs_residue;
    int* row_offsets_d;
    int* col_indices_d;
    float* values_d;
    int* seg_row_indices_d;         // device, length n_segs (row_indices_block)
    int* seg_st_offsets_d;          // device, length n_segs+1
    int* seg_row_indices_residue_d; // device, length n_segs_residue (maybe null)
};

// Verbatim port of SPC::SparseMatrix::RowDivide2Segment
// (source/utils/matrix_utils.cu:989-1041) -- identical to
// ../../spmm/rode/wrapper.cu's copy; see that file's header comment for the
// full derivation against unmodified artifact source.
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
// row-decomposition metadata built. `K` selects the SegmentLength constant:
// RoDe's own eval drivers use a DIFFERENT SEG_LENGTH per dense-width variant
// -- confirmed by direct grep of both drivers:
//   source/eval/eval_sddmm_f32_n32.cu:  `#define SEG_LENGTH 512`
//   source/eval/eval_sddmm_f32_n128.cu: `#define SEG_LENGTH 32`
// each feeding `sm1.RowDivide2Segment(SEG_LENGTH,4,32)`, and each matching
// that variant's own kernel-internal grid_dim1.y computation
// (RoDeSDDMMKernel_n32<...,512> vs RoDeSDDMMKernel_n128<...,32> --
// grid_dim1.y = SEG_LENGTH/kBlockItemsX). Passing the WRONG SegmentLength
// here for K=128 (e.g. reusing n32's 512) makes RowDivide2Segment produce
// segments the n128 kernel's actual grid does not cover -- confirmed by
// direct testing: entries beyond the first 32 of any >32-wide segment are
// silently never written (left at whatever the caller pre-zeroed `out`
// to), which reads as a ~1e-1 relative-error correctness failure that is
// this wrapper's own preprocessing-parameter bug, not a defect in RoDe's
// compiled kernel itself.
void* sddmm_prepare(int rows, int cols, int nnz, int K,
                     const int* row_offsets_h, const int* col_indices_h,
                     const float* values_h) {
    const int seg_length = (K == 32) ? 512 : 32;  // K==128 -> 32, per above
    std::vector<int> seg_idx, seg_off, seg_res;
    row_divide_to_segment(rows, row_offsets_h, seg_length, 4, 32, seg_idx, seg_off, seg_res);

    RodeSddmmHandle* h = new RodeSddmmHandle();
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

int sddmm_n_segs(void* handle) { return ((RodeSddmmHandle*)handle)->n_segs; }
int sddmm_n_segs_residue(void* handle) { return ((RodeSddmmHandle*)handle)->n_segs_residue; }

// ONE SDDMM kernel invocation -- K (dense feature dim) in {32, 128} only,
// the only two instantiations RoDe ships (RoDeSDDMM_n32/RoDeSDDMM_n128,
// each with the feature dim baked into the template as a compile-time
// constant -- see RoDeSddmm.cu:
// `RoDeSDDMMKernel_n32<float4,4,32,32,8,32,512>` / `..._n128<...,128,32>`).
// `A` is the row-indexed dense operand (M x K, "lhs_matrix"), `B` the
// column-indexed dense operand (N x K, "rhs_matrix") -- same convention as
// cpu_ref.reference_sddmm (out = dot(A[row],B[col]) * S.data). Returns
// cudaError_t (0 = success) from cudaGetLastError() after both streams'
// kernel launches (Kernel-block = segmented rows, Kernel-residue = residue
// tail; RoDe's own eval driver launches both on separate streams -- s1/s2
// here may be the same stream (0) or two real streams created by the caller).
int sddmm_run(void* handle, int K, const float* A, const float* B, float* out,
              cudaStream_t s1, cudaStream_t s2) {
    RodeSddmmHandle* h = (RodeSddmmHandle*)handle;
    if (K == 32) {
        RoDeSDDMM_n32(h->n_segs, h->n_segs_residue, h->cols, K,
                      h->seg_row_indices_d, h->seg_row_indices_residue_d, h->seg_st_offsets_d,
                      h->row_offsets_d, h->col_indices_d, h->values_d,
                      A, B, out, s1, s2);
    } else if (K == 128) {
        RoDeSDDMM_n128(h->n_segs, h->n_segs_residue, h->cols, K,
                       h->seg_row_indices_d, h->seg_row_indices_residue_d, h->seg_st_offsets_d,
                       h->row_offsets_d, h->col_indices_d, h->values_d,
                       A, B, out, s1, s2);
    } else {
        return -1;
    }
    cudaError_t err = cudaGetLastError();
    // Same structurally-explained edge case as RoDeSpmm (see
    // bench/artifacts/spmm/rode/wrapper.cu / STATUS.md): for matrices where
    // every row has fewer nonzeros than KBLOCK=32 (this track's low-degree
    // GNN-style smoke graphs), RowDivide2Segment produces n_segs==0 and
    // RoDeSDDMM_n32/n128's block-kernel grid becomes
    // `dim3((0+3)/4, SEG_LENGTH/kBlockItemsX, 1)` i.e. grid.x==0 -- a zero-
    // block launch that deterministically raises
    // cudaErrorInvalidConfiguration(9) on this CUDA/driver pair, while the
    // residue kernel (a valid, non-empty launch on the same stream) still
    // executes and produces the full, correct output regardless (every
    // nonzero lands in EITHER the block set or the residue set, never
    // both -- RowDivide2Segment's own invariant). This is RoDe's own kernel
    // launcher's behavior for an input shape its own eval driver (always
    // real, higher-degree SuiteSparse .mtx files) never exercises; the
    // numeric result is unaffected. Any OTHER error code, or this same code
    // when n_segs != 0, is still propagated and would fail the gate --
    // nothing about the correctness gate itself is loosened.
    if (err == cudaErrorInvalidConfiguration && h->n_segs == 0) {
        return 0;
    }
    return (int)err;
}

void sddmm_free(void* handle) {
    RodeSddmmHandle* h = (RodeSddmmHandle*)handle;
    cudaFree(h->row_offsets_d);
    cudaFree(h->col_indices_d);
    cudaFree(h->values_d);
    cudaFree(h->seg_row_indices_d);
    cudaFree(h->seg_st_offsets_d);
    if (h->seg_row_indices_residue_d) cudaFree(h->seg_row_indices_residue_d);
    delete h;
}

}  // extern "C"
