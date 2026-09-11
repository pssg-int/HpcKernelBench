// Thin extern "C" shim around BLEST's own CSC -> reorder -> BVSS pipeline
// and per-source BFS kernel (BFSKernel::singleSourceRun). NOT part of the
// artifact -- same role as bench/artifacts/triangle-counting/tot/tot_shim.cu.
// Zero lines of source/ are modified.
//
// blest_prepare() mirrors main.cu's Benchmark::run() preprocessing exactly:
// CSC construction -> reorder(sliceSize=8) -> BVSS bit-sliced construction.
// It deliberately EXCLUDES main.cu's own file I/O, on-disk .bin caching, and
// the SuiteSparseMatrixDownloader/libcurl dependency (which only exists to
// fetch matrices from the SuiteSparse website) -- this project's graphs come
// from kernelbench.matrices' own cache, so the CSC struct is built directly
// from an in-memory CSR handed to us by the adapter, and the classes we need
// (CSC, BVSS, BFSKernel) are reachable without ever including Benchmark.cuh
// / main.cu, which is the only place libcurl is required.
//
// blest_run() mirrors main.cu's per-source remapping: BLEST internally
// reorders vertices for locality (Benchmark::run's `reorder()` call), so a
// source id and the returned level array must be translated through
// inversePermutation both ways, exactly as Benchmark::run()'s "BFS" branch
// does (see BLEST/source/Benchmark.cuh, the `else if (kernelName == "BFS")`
// block this shim's blest_run mirrors line for line for the remap).
#include "BFS/BFSKernel.cuh"

#include <cstdint>
#include <cstdio>
#include <cstring>

struct BlestHandle {
    BFSKernel* kernel;
    BVSS* bvss;
    unsigned n;
    unsigned* inversePermutation;  // original vertex id -> BLEST-internal id
};

extern "C" {

// n: vertex count; nnz: number of directed CSR entries; row_ptr has n+1
// entries, col_idx has nnz entries (both uint32, host memory). The input is
// already self-loop-free with sorted per-row indices (see
// kernelbench.domains.graph._clean_graph) -- required by
// BVSS::constructFromCSCMatrix ("adjacency list must be sorted", CSC.cuh).
// Returns an opaque handle (free with blest_free), or nullptr on failure
// (message on stderr).
void* blest_prepare(unsigned n, unsigned long long nnz,
                     const unsigned* row_ptr, const unsigned* col_idx) {
    try {
        // BLEST's "CSC" is, despite the name, exactly our CSR (m_ColPtrs[j]
        // is the offset of vertex j's OUT-edges into m_Rows, which holds
        // destination ids) -- see CSC::CSC(filename,...)'s own construction
        // (`adjList[j].emplace_back(i)` for an edge j->i). Copy into
        // freshly-owned buffers since CSC's destructor deletes them.
        unsigned* colPtrs = new unsigned[(size_t)n + 1];
        std::memcpy(colPtrs, row_ptr, sizeof(unsigned) * ((size_t)n + 1));
        unsigned* rows = new unsigned[(size_t)nnz];
        std::memcpy(rows, col_idx, sizeof(unsigned) * (size_t)nnz);

        CSC* csc = new CSC();
        csc->getN() = n;
        csc->getNNZ() = (unsigned)nnz;
        csc->getColPtrs() = colPtrs;
        csc->getRows() = rows;

        // CSC's own "is this a social network" classifier
        // (CSC::socialNetworkHelper, a power-law-tail degree-distribution
        // test called from isSocialNetwork()'s computation in the file
        // constructor) is PRIVATE and only ever run from the file-reading
        // constructor we bypass here. Reproduce just the cheap half of its
        // own OR-condition (average degree > SOCIAL_THRESHOLD, Common.cuh)
        // as a documented simplification: this choice ONLY steers which
        // reordering heuristic reorder() below picks (Jaccard-window vs.
        // RCM) and which kernel variant (FULL_PADDING) runs at kernel-
        // selection time in BFSKernel.cuh -- never correctness, since every
        // reorder()/kernel branch produces a valid permutation / correct
        // BFS levels (see adapter.py docstring for the full argument).
        double avgDeg = n ? (double)nnz / (double)n : 0.0;
        bool social = avgDeg > (double)SOCIAL_THRESHOLD;
        csc->isSocialNetwork() = social;
        FULL_PADDING = !social;   // matches Benchmark::run's own assignment
        JACKARD_ON = true;        // matches README's recommended default
        WINDOW_SIZE = 65536;      // matches README's recommended default

        constexpr unsigned sliceSize = 8;
        constexpr unsigned noMasks = 32 / sliceSize;

        // preprocessing: vertex reordering (Jaccard-window or RCM, chosen
        // by CSC::reorder() itself based on isSocialNetwork() above)
        unsigned* inversePermutation = csc->reorder(sliceSize);

        // preprocessing: bit-sliced tensor-core-friendly layout (BVSS)
        BVSS* bvss = new BVSS(sliceSize, noMasks);
        bvss->constructFromCSCMatrix(csc);
        delete csc;  // BVSS keeps its own transposed copy (getCSR()); the
                     // reordered-space CSC we built is no longer needed.

        BFSKernel* kernel = new BFSKernel(dynamic_cast<BitMatrix*>(bvss));

        BlestHandle* h = new BlestHandle();
        h->kernel = kernel;
        h->bvss = bvss;
        h->n = n;
        h->inversePermutation = inversePermutation;
        return (void*)h;
    } catch (const std::exception& e) {
        fprintf(stderr, "blest_prepare failed: %s\n", e.what());
        return nullptr;
    }
}

// source: ORIGINAL vertex id. out_levels: caller-allocated host buffer of
// length n (unsigned, UNSIGNED_MAX sentinel = unreached, matching BLEST's
// own convention) -- filled with BLEST's raw level array translated back to
// original vertex ids, matching Benchmark::run()'s own BFS-branch remap
// exactly:
//   newLevels[old] = result.levels[inversePermutation[old]]
// This is the kernel-only call: BFSKernel::singleSourceRun's device buffer
// alloc + H2D upload of the (already-reordered, already on the *host* only
// as `bvss`) adjacency happens INSIDE singleSourceRun, before its own
// `start = omp_get_wtime()` -- i.e. excluded from BLEST's *own* reported
// per-search time, but NOT excluded from what this harness's CudaEventTimer
// measures around the whole blest_run() ctypes call (see adapter.py
// docstring: BLEST's public API has no finer boundary that reuses uploaded
// device state across repeated single-source calls without also taking the
// whole 64-root list upfront via multiSourceRun, which would violate this
// harness's one-search-per-timed-call contract -- documented contamination,
// not a bug, per ARTIFACT_GUIDE.md rule 1).
int blest_run(void* handle, unsigned source, unsigned* out_levels) {
    BlestHandle* h = (BlestHandle*)handle;
    if (!h || source >= h->n) return -1;
    unsigned internalSource = h->inversePermutation[source];
    BFSResult result = h->kernel->singleSourceRun(internalSource, /*switching=*/false);
    for (unsigned old = 0; old < h->n; ++old) {
        out_levels[old] = result.levels[h->inversePermutation[old]];
    }
    delete[] result.levels;
    return 0;
}

void blest_free(void* handle) {
    BlestHandle* h = (BlestHandle*)handle;
    if (!h) return;
    delete h->kernel;
    delete h->bvss;
    delete[] h->inversePermutation;
    delete h;
}

}  // extern "C"
