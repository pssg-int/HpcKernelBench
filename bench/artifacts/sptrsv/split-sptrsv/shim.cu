// shim.cu -- NOT part of the Split_SpTRSV artifact. Thin extern "C"
// host-side launchers around the artifact's OWN unmodified synchronization-
// free CUDA kernels (source/include/sptrsv_syncfree_cuda.cuh), included
// verbatim below -- same role bench/artifacts/spmv/sspmv/csr_shim.cpp plays
// for LeSpMV. See adapter.py's module docstring for the full design
// rationale (per-call state reset, CSC diagonal-first layout).
#include "sptrsv_syncfree_cuda.cuh"

extern "C" {

// One-shot structural analysis (a pure function of L's sparsity pattern,
// unrelated to any particular solve call): wraps the artifact's own
// sptrsv_syncfree_cuda_analyser kernel unmodified. Called ONCE, from
// adapter.py's prepare().
void split_sptrsv_indegree(const ind_type *d_cscRowIdx, int m, int nnz,
                            int *d_indegree) {
    cudaMemset(d_indegree, 0, (size_t)m * sizeof(int));
    int threads = 128;
    int blocks = (nnz + threads - 1) / threads;
    if (blocks >= 1)
        sptrsv_syncfree_cuda_analyser<<<blocks, threads>>>(
            d_cscRowIdx, (sz_type)m, (sz_type)nnz, d_indegree);
}

// One genuine solve call: reset the two pieces of per-call algorithm state
// the executor kernel mutates in place during a solve (the in-degree
// synchronization counter, the partial-sum accumulator -- see
// adapter.py's module docstring for why this reset belongs INSIDE the
// timed region, per benchspecs/sptrsv/spec.yaml's own timing_scope
// wording), then launch the artifact's own sptrsv_syncfree_cuda_executor
// kernel unmodified (forward substitution, SUBSTITUTION_FORWARD).
void split_sptrsv_solve(const ind_type *d_cscColPtr, const ind_type *d_cscRowIdx,
                         const val_type *d_cscVal, const int *d_indegree_template,
                         int *d_indegree_work, val_type *d_left_sum, int m,
                         const val_type *d_b, val_type *d_x) {
    cudaMemcpyAsync(d_indegree_work, d_indegree_template, (size_t)m * sizeof(int),
                     cudaMemcpyDeviceToDevice);
    cudaMemsetAsync(d_left_sum, 0, (size_t)m * sizeof(val_type));
    int threads = WARP_PER_BLOCK * WARP_SIZE;
    int blocks = (m + (threads / WARP_SIZE) - 1) / (threads / WARP_SIZE);
    if (blocks >= 1)
        sptrsv_syncfree_cuda_executor<<<blocks, threads>>>(
            d_cscColPtr, d_cscRowIdx, d_cscVal, d_indegree_work, d_left_sum,
            (sz_type)m, SUBSTITUTION_FORWARD, d_b, d_x);
}

int split_sptrsv_sync() {
    cudaDeviceSynchronize();
    return (int)cudaGetLastError();
}

} // extern "C"
