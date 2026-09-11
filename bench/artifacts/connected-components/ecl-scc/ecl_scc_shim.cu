// Thin ctypes-callable wrapper around ECL-SCC's own kernels (globalInit,
// propagateMax, removeEdges, localInit -- the max-ID/max-out-degree
// "signature" propagation SCC algorithm from source/source/ECL-SCC_10.cu),
// refactored out of that file's CLI main() into three reusable extern "C"
// entry points. This file is NOT part of the artifact -- same role as
// bench/artifacts/triangle-counting/tot/tot_shim.cu and
// bench/artifacts/spmv/sspmv/csr_shim.cpp.
//
// ECL-SCC_10.cu is #included UNMODIFIED below: its four __global__ kernels
// and CheckCuda()/Device/ThreadsPerBlock are reused byte-for-byte from the
// SAME translation unit; only main()'s own file-I/O / argument-parsing /
// printf driver is bypassed (present in the object file, never called).
//
// Mapping (connected-components track, task C): SCC is only meaningful for
// DIRECTED graphs, but ECL-SCC's algorithm runs unconditionally on whatever
// CSR it is given. On a SYMMETRIZED graph (every edge u->v also has v->u),
// every weakly-connected component is trivially strongly connected too --
// SCC(symmetrize(G)) == CC(symmetrize(G)). adapter.py's prepare() performs
// the symmetrization (host side, before calling ecl_scc_prepare below) and
// feeds the symmetrized CSR here; ECL-SCC's kernels never know or care that
// the graph happens to be symmetric -- they run the identical algorithm they
// would run on any directed input. See adapter.py's module docstring for the
// full mapping writeup and the correctness-gate argument.
#include "source/source/ECL-SCC_10.cu"

#include <cstdint>
#include <vector>

struct EclSccHandle {
    ECLgraph d_g;
    int2* d_wl1;
    int2* d_wl2;
    int2* d_iomax;
    int* d_wl2size;
    bool* d_goagain;
    int blocks;
    int nodes;
    int edges;
};

extern "C" {

// Build the device-resident graph + working buffers (the artifact's own
// "copy graph to GPU" + allocation step in main()). CSR arrays are int32,
// matching ECLgraph's own nindex/nlist layout exactly -- no format
// conversion beyond H2D copy is needed since kernelbench's Matrix/Graph CSR
// is already what ECLgraph expects.
void* ecl_scc_prepare(int n, long long nnz, const int* row_ptr, const int* col_idx) {
    EclSccHandle* h = new EclSccHandle();
    h->nodes = n;
    h->edges = static_cast<int>(nnz);

    cudaSetDevice(Device);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, Device);
    const int SMs = deviceProp.multiProcessorCount;
    const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
    h->blocks = SMs * (mTpSM / ThreadsPerBlock);

    h->d_g.nodes = n;
    h->d_g.edges = h->edges;
    h->d_g.eweight = nullptr;
    cudaMalloc((void**)&h->d_g.nindex, (n + 1) * sizeof(int));
    cudaMemcpy(h->d_g.nindex, row_ptr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&h->d_g.nlist, h->edges * sizeof(int));
    cudaMemcpy(h->d_g.nlist, col_idx, h->edges * sizeof(int), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&h->d_iomax, n * sizeof(int2));
    cudaMalloc((void**)&h->d_wl1, h->edges * sizeof(int2));
    cudaMalloc((void**)&h->d_wl2, h->edges * sizeof(int2));
    cudaMalloc((void**)&h->d_wl2size, sizeof(int));
    cudaMalloc((void**)&h->d_goagain, sizeof(bool));
    CheckCuda(__LINE__);
    return h;
}

// The kernel itself: EXACTLY main()'s "start time ... end time" region from
// ECL-SCC_10.cu (globalInit, then the propagateMax/removeEdges/localInit
// fixed-point loop), unmodified. Re-runs globalInit at the top of every call
// so repeated invocations (warmup + timed reps) each start from ECL-SCC's
// own initial state -- the algorithm mutates d_wl1/d_iomax/d_wl2size in
// place, so without this reset a second call would resume from the first
// call's already-converged (empty-worklist) state and do no work.
void ecl_scc_run(void* handle) {
    EclSccHandle* h = static_cast<EclSccHandle*>(handle);

    int wl1size;
    globalInit<<<h->blocks, ThreadsPerBlock>>>(h->d_g, h->d_wl1, h->d_wl2size, h->d_iomax);
    cudaMemcpy(&wl1size, h->d_wl2size, sizeof(int), cudaMemcpyDeviceToHost);

    bool goagain;
    do {
        do {
            cudaMemsetAsync(h->d_goagain, 0, sizeof(bool));
            propagateMax<<<h->blocks, ThreadsPerBlock>>>(h->d_wl1, wl1size, h->d_iomax, h->d_goagain);
            cudaMemcpy(&goagain, h->d_goagain, sizeof(bool), cudaMemcpyDeviceToHost);
        } while (goagain);

        cudaMemsetAsync(h->d_wl2size, 0, sizeof(int));
        removeEdges<<<h->blocks, ThreadsPerBlock>>>(h->d_wl1, wl1size, h->d_wl2, h->d_wl2size, h->d_iomax);
        int2* tmp = h->d_wl1;
        h->d_wl1 = h->d_wl2;
        h->d_wl2 = tmp;
        cudaMemcpyAsync(&wl1size, h->d_wl2size, sizeof(int), cudaMemcpyDeviceToHost);

        cudaMemsetAsync(h->d_goagain, 0, sizeof(bool));
        localInit<<<h->blocks, ThreadsPerBlock>>>(h->nodes, h->d_iomax, h->d_goagain);
        cudaMemcpy(&goagain, h->d_goagain, sizeof(bool), cudaMemcpyDeviceToHost);
    } while (goagain);
    cudaDeviceSynchronize();
}

// D2H copy of the converged per-vertex signature (iomax[v].x is ECL-SCC's
// own component-representative id, per main()'s "count[iomax[v].x]++"
// aggregation). Called once by adapter.py's to_host(), never inside the
// timed loop -- matches cc-scc-kernel's timing_scope, which excludes output
// transfer.
void ecl_scc_labels(void* handle, long long* out) {
    EclSccHandle* h = static_cast<EclSccHandle*>(handle);
    std::vector<int2> iomax(h->nodes);
    cudaMemcpy(iomax.data(), h->d_iomax, h->nodes * sizeof(int2), cudaMemcpyDeviceToHost);
    for (int v = 0; v < h->nodes; v++) out[v] = iomax[v].x;
}

void ecl_scc_free(void* handle) {
    EclSccHandle* h = static_cast<EclSccHandle*>(handle);
    if (!h) return;
    cudaFree(h->d_g.nindex);
    cudaFree(h->d_g.nlist);
    cudaFree(h->d_iomax);
    cudaFree(h->d_wl1);
    cudaFree(h->d_wl2);
    cudaFree(h->d_wl2size);
    cudaFree(h->d_goagain);
    delete h;
}

}  // extern "C"
