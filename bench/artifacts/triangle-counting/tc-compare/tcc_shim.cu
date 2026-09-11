// Thin extern "C" shim around TC-Compare's GroupTC kernel
// (source/approach/GroupTC/tc.cu) -- same role as
// bench/artifacts/triangle-counting/tot/tot_shim.cu and
// bench/artifacts/spmv/sspmv/csr_shim.cpp: a small wrapper exposing a
// ctypes-callable boundary around the artifact's OWN, UNMODIFIED kernel.
//
// GroupTC is the paper's own proposed algorithm (the "representative GPU
// intersection kernel" chosen from TC-Compare's 9-algorithm comparative
// suite -- see STATUS.md for why GroupTC over the other 8).
//
// tc.cu is #include'd verbatim (unmodified, zero patches) rather than
// compiled+linked separately, so its `grouptc` __global__ kernel, its
// `graph`/`edge_count`/`vertex_count`/`grid_size`/`block_bucketnum` globals
// are all directly usable from this translation unit. tc.cu's own `main()`
// becomes dead code inside this shared library (never called) -- harmless,
// a .so may contain an unused function literally named `main`.
//
// tcc_count() below reproduces EXACTLY tc.cu's own gpu_run()'s single-
// iteration body (cudaMemset -> grouptc<<<>>> -> cudaDeviceSynchronize ->
// thrust::reduce, source/approach/GroupTC/tc.cu:159-171) with the outer
// 100x loop and the printf-only reporting removed -- i.e. this IS the
// artifact's own per-call kernel invocation, just returned as a value
// instead of printed, so the harness can gate/time ONE call at a time.
#include "source/approach/GroupTC/tc.cu"

#include <vector>

extern "C" {

struct TccHandle {
    vertex_t* d_source;
    vertex_t* d_adj;
    index_t* d_offset;
    unsigned long long* d_results;
    int n_results;
};

// n_vertices/n_edges describe an already-ORIENTED DAG (each undirected
// edge kept exactly once, source_list[i] < adj_list[i] under SOME total
// vertex order -- ties/order choice do not affect the triangle count, only
// performance, matching GroupTC's own rid_dcsr preprocessing, whose vertex
// degree-based relabeling is a locality optimization, not a correctness
// requirement, see STATUS.md); adj_list sorted ascending within each row
// (required by grouptc's binary-search intersection). h_source[i] is the
// row (source) of edge i, redundant with h_offset but required by the
// kernel's own indexing (matches tc.cu's `graph` struct layout exactly:
// begin.bin/source.bin/adjacent.bin).
void* tcc_prepare(int n_vertices, long long n_edges,
                   const int* h_source, const int* h_adj,
                   const long long* h_offset) {
    vertex_count = n_vertices;
    edge_count = (int)n_edges;

    long edge_size = sizeof(vertex_t) * (long)edge_count;
    long offset_size = sizeof(index_t) * (long)(vertex_count + 1);

    std::vector<index_t> h_offset_idx((size_t)vertex_count + 1);
    for (int i = 0; i <= vertex_count; i++) {
        h_offset_idx[(size_t)i] = (index_t)h_offset[i];
    }

    TccHandle* h = new TccHandle();
    cudaMalloc(&h->d_source, edge_size);
    cudaMalloc(&h->d_adj, edge_size);
    cudaMalloc(&h->d_offset, offset_size);
    h->n_results = grid_size * block_bucketnum;
    cudaMalloc(&h->d_results, (size_t)h->n_results * sizeof(unsigned long long));

    cudaMemcpy(h->d_source, h_source, edge_size, cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_adj, h_adj, edge_size, cudaMemcpyHostToDevice);
    cudaMemcpy(h->d_offset, h_offset_idx.data(), offset_size, cudaMemcpyHostToDevice);

    return (void*)h;
}

// One kernel-only triangle count, mirroring gpu_run()'s per-iteration body.
long long tcc_count(void* handle) {
    TccHandle* h = (TccHandle*)handle;
    cudaMemset(h->d_results, 0, (size_t)h->n_results * sizeof(unsigned long long));
    grouptc<<<grid_size, block_bucketnum>>>(
        h->d_source, h->d_adj, h->d_offset, edge_count, vertex_count, h->d_results);
    cudaDeviceSynchronize();
    thrust::device_ptr<unsigned long long> ptr(h->d_results);
    unsigned long long count = thrust::reduce(ptr, ptr + h->n_results);
    return (long long)count;
}

void tcc_free(void* handle) {
    TccHandle* h = (TccHandle*)handle;
    cudaFree(h->d_source);
    cudaFree(h->d_adj);
    cudaFree(h->d_offset);
    cudaFree(h->d_results);
    delete h;
}

}  // extern "C"
