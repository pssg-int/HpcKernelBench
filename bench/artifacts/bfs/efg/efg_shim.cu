// Thin extern "C" shim around efg's own CSR -> EFLayout -> EFGraph ->
// CUEFGraph -> BFS pipeline (src/main.cu's exact sequence, minus its CLI
// parsing and boost::program_options dependency, which this shim never
// needs). NOT part of the artifact -- same role as
// bench/artifacts/triangle-counting/tot/tot_shim.cu. Zero lines of
// source/ are modified.
//
// efg_prepare() mirrors main.cu lines 68-86 exactly:
//   CSR csr(in_dir)                              -- load CSR (see below)
//   EFLayout<0,512> ef_layout(csr)                -- compute EF byte sizes
//   CUEFGraph<0,512> cu_ef_graph(ef_layout, mode)  -- ENCODE (host, folly's
//                                                     real EliasFanoEncoder,
//                                                     see ef_graph.h) +
//                                                     upload to device
//   BFS<0,512> bfs(cu_ef_graph, sort_frontier)     -- allocate device
//                                                     scratch buffers
// kSkipQuantum=0, kForwardQuantum=512 are main.cu's own compile-time
// defaults (BFS<> is templated on them, so they cannot be runtime
// parameters without exploding the number of template instantiations built
// -- using the paper's own CLI defaults is the honest choice here).
//
// CSR's own constructor (source/src/csr.cpp) is file-based ONLY (reads
// vertex.CSR/edge.CSR binary files from a directory) -- there is no
// in-memory constructor in the artifact. Rather than touch csr.{h,cpp} (a
// kernel-adjacent file we'd rather not patch, per ARTIFACT_GUIDE.md rule
// 3), the adapter (adapter.py) writes the graph's CSR to a fresh temp
// directory as vertex.CSR/edge.CSR (uint64, exactly the format
// source/README.md documents) and hands US that directory -- this IS
// "the artifact's own format conversion", i.e. still entirely inside
// prepare() by ARTIFACT_GUIDE.md rule 2, just routed through a temp
// directory instead of an in-memory array because that's the only loading
// path CSR actually offers.
//
// efg_run() is the kernel-only call: BFS<>::traverse(source) ONLY -- the
// GPU top-down frontier expansion (src/bfs.cuh / bfs_kernels.cuh), which
// decodes the Elias-Fano-compressed adjacency ENTIRELY ON-DEVICE with a
// hand-rolled bit-select routine (byte_select.cuh's dSelectInByte lookup
// table + cub::BFE) -- folly is NOT used on the device side at all, only
// during the host-side ENCODE in CUEFGraph's constructor above (verified
// by reading bfs_kernels.cuh: no folly:: symbol appears in any __device__/
// __global__ function). traverse() returns the LEVEL/distance array
// directly (std::vector<size_t>, UINT64_MAX sentinel for unreached) --
// this artifact was ALREADY level-array-native, unlike blest/bit-graphblas.
#include "csr.h"
#include "cu_ef_graph.cuh"
#include "ef_layout.h"
#include "bfs.cuh"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

constexpr size_t kSkipQuantum = 0;
constexpr size_t kForwardQuantum = 512;

struct EfgHandle {
    CSR* csr;
    EFLayout<kSkipQuantum, kForwardQuantum>* ef_layout;
    CUEFGraph<kSkipQuantum, kForwardQuantum>* cu_ef_graph;
    BFS<kSkipQuantum, kForwardQuantum>* bfs;
    size_t num_vertices;
};

extern "C" {

// csr_dir: directory containing vertex.CSR / edge.CSR (uint64, written by
// adapter.py -- see file docstring above). use_uvm: 0 = plain device
// memory (main.cu's default), 1 = UVM (CUEFGraph's alloc_mode=1).
// sort_frontier: 1 = main.cu's default (!no_sort), 0 = disable the
// partial-frontier-sorting optimisation (README: needed only for graphs
// that barely fit in memory; kept as a passthrough for completeness).
// Returns an opaque handle (free with efg_free), or nullptr on failure.
void* efg_prepare(const char* csr_dir, int use_uvm, int sort_frontier) {
    try {
        EfgHandle* h = new EfgHandle();
        h->csr = new CSR(std::string(csr_dir));
        h->num_vertices = h->csr->get_num_vertices();

        h->ef_layout = new EFLayout<kSkipQuantum, kForwardQuantum>(*h->csr);

        int alloc_mode = use_uvm ? 1 : 0;
        h->cu_ef_graph = new CUEFGraph<kSkipQuantum, kForwardQuantum>(*h->ef_layout, alloc_mode);

        h->bfs = new BFS<kSkipQuantum, kForwardQuantum>(*h->cu_ef_graph, sort_frontier != 0);

        return (void*)h;
    } catch (const std::exception& e) {
        fprintf(stderr, "efg_prepare failed: %s\n", e.what());
        return nullptr;
    }
}

unsigned long long efg_num_vertices(void* handle) {
    return (unsigned long long)((EfgHandle*)handle)->num_vertices;
}

unsigned long long efg_csr_storage_bytes(void* handle) {
    return (unsigned long long)((EfgHandle*)handle)->csr->get_storage_size();
}

unsigned long long efg_csr_optimal_bytes(void* handle) {
    return (unsigned long long)((EfgHandle*)handle)->csr->get_optimal_size();
}

unsigned long long efg_ef_storage_bytes(void* handle) {
    return (unsigned long long)((EfgHandle*)handle)->ef_layout->get_storage_size();
}

// source: vertex id. out_distances: caller-allocated host buffer of length
// num_vertices (uint64, UINT64_MAX sentinel = unreached, matching efg's own
// convention exactly -- traverse() already returns per-vertex distances,
// no remapping needed, unlike blest's reordered-index space).
int efg_run(void* handle, unsigned long long source, unsigned long long* out_distances) {
    EfgHandle* h = (EfgHandle*)handle;
    if (!h) return -1;
    std::vector<size_t> distances = h->bfs->traverse((size_t)source);
    for (size_t i = 0; i < h->num_vertices; ++i) {
        out_distances[i] = (unsigned long long)distances[i];
    }
    return 0;
}

void efg_free(void* handle) {
    EfgHandle* h = (EfgHandle*)handle;
    if (!h) return;
    delete h->bfs;
    delete h->cu_ef_graph;
    delete h->ef_layout;
    delete h->csr;
    delete h;
}

}  // extern "C"
