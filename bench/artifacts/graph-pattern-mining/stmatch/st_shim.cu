// STMatch bridge (this directory, NOT part of the artifact).
//
// Paper: "STMatch: Accelerating Graph Pattern Matching on GPU with
// Stack-Based Loop Optimizations" (SC'22, conf/sc/WeiJ22).
// Artifact: https://github.com/HPC-Research-Lab/STMatch
//
// Headline pattern wrapped: exact k-clique COUNTING via STMatch's general
// stack-based subgraph-matching kernel (`_parallel_match`,
// `source/src/gpu_match.cu`), driven by a complete-graph K_k `Pattern`
// (STMatch's own `v`/`e` text pattern format -- see `_write_pattern_file`
// in adapter.py). Matches bench/kernelbench/domains/graph.py's
// graph-pattern-mining headline (`params['clique_k']`, default 4).
//
// The artifact's own driver, `cu_test.cu`'s `main()`, is monolithic exactly
// like GLumin's CliqueSolver/TileSpGEMM's tilespgemm()/GraphSet's
// pattern_matching_init(): it builds the device Graph/Pattern/JobQueue/
// CallStack representations AND launches `_parallel_match` AND reads back
// the per-warp counts, with no public sub-boundary. Per ARTIFACT_GUIDE.md
// rule 1, this file reproduces `main()`'s own body VERBATIM, split into:
//
//   st_prepare() -- builds the artifact's own `Graph` struct DIRECTLY from
//                   our CSR (no file I/O, bypassing GraphPreprocessor's
//                   binary .meta.txt/.vertex.bin/.edge.bin format
//                   entirely), then calls the artifact's own
//                   `PatternPreprocessor` (source/src/pattern.h, reads the
//                   small auto-generated K_k pattern text file -- this IS
//                   the artifact's own schedule/set-operation compiler,
//                   pattern-only preprocessing) and `JobQueuePreprocessor`
//                   (source/src/job_queue.h, builds the initial 2-vertex
//                   job list from the graph+pattern, unmodified), then
//                   reproduces `main()`'s own device setup (Graph/Pattern/
//                   JobQueue H2D copies, CallStack/slot_storage/idle_warps/
//                   global_mutex allocation+init) verbatim -- exactly
//                   `main()`'s own pre-launch scope.
//   st_run()      -- resets every piece of PER-CALL mutable device state
//                   (CallStack array, job-queue cursor/mutex, per-warp
//                   result buffer, idle-warp bookkeeping) to its pristine
//                   prepare()-time value -- needed because this harness
//                   reuses one handle across warmup+reps, unlike the
//                   artifact's own single-shot CLI, which only initializes
//                   these once before its one launch -- then launches
//                   `_parallel_match<<<>>>` (the artifact's own unmodified
//                   `__global__` kernel, declared in the unmodified
//                   `gpu_match.cuh` and defined in the unmodified
//                   `gpu_match.cu`, compiled together with this file via
//                   `-rdc=true`, the SAME separate-compilation scheme the
//                   artifact's own Makefile already uses for gpu_match.cu
//                   -- see build.sh) and reads back the per-warp counts,
//                   summing and multiplying by `PatternMultiplicity`
//                   exactly as `main()` does for an unlabeled pattern.
//   st_free()     -- releases everything.
//
// Zero lines of source/ were modified. `_parallel_match` and everything it
// calls (`match`, `trans_skt`, `trans_layer`, ...) remain exclusively in
// the unmodified source/src/gpu_match.cu.
//
// Counting semantics (checked empirically, see STATUS.md): STMatch's own
// `main()` computes `tot_count = sum(per-warp counts) * PatternMultiplicity`
// when the pattern is unlabeled (`!LABELED`, this project's config). For a
// complete graph K_k, EVERY permutation of the k vertices is a valid
// automorphism, so `PatternMultiplicity == k!`; STMatch's own partial-order
// pruning (`PatternPreprocessor::get_partial_order`) restricts the raw
// per-warp search to exactly one canonical vertex ordering per embedding,
// so the raw sum (before multiplying by k!) is already the graph's own
// distinct-subset clique count -- verified directly against the other
// three adapters' cross-validated K4 count on ca-HepPh (150281372, see
// STATUS.md). This adapter therefore returns the RAW per-warp sum, NOT
// STMatch's own final printed number (which would overcount by k!).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "graph.h"
#include "pattern.h"
#include "job_queue.h"
#include "callstack.h"
#include "gpu_match.cuh"

using namespace STMatch;

#define ST_CUDA_CHECK(x)                                                        \
    do {                                                                        \
        cudaError_t _e = (x);                                                   \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "st_shim CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_e));                                    \
        }                                                                       \
    } while (0)

extern "C" {

struct StHandle {
    Graph *g;                 // host Graph (fields we own)
    Graph *gpu_graph;
    PatternPreprocessor *pat;
    Pattern *gpu_pattern;
    JobQueuePreprocessor *jq;
    JobQueue *gpu_queue;
    CallStack *gpu_callstack;
    graph_node_t *slot_storage;
    size_t *gpu_res;
    int *idle_warps;
    int *idle_warps_count;
    int *global_mutex;
    std::vector<CallStack> initial_stk;  // pristine per-warp state, replayed each run()
    long long multiplicity;
};

void *st_prepare(int32_t n, int64_t e_cnt, const int64_t *rowptr,
                  const int32_t *colidx, const char *pattern_path) {
    // Artifact's own Graph struct (source/src/graph.h), built directly from
    // our CSR -- no file I/O (GraphPreprocessor's binary .meta.txt/
    // .vertex.bin/.edge.bin format is bypassed entirely). vertex_label is
    // set to (1 << 1) for every vertex, matching the UNLABELED convention
    // PatternPreprocessor::readfile() itself uses (`vertex_labels.
    // push_back(1)` when `!LABELED`) -- deliberately NOT reproducing
    // GraphPreprocessor::read_bin_file()'s own default-label path, which
    // fills its `lb` array via `memset(lb, 1, n*sizeof(int))` (a
    // byte-value-1 memset on an int array, i.e. every int becomes
    // 0x01010101, not the integer 1) -- that only "works" by relying on
    // x86's shift-count-mod-32 behavior (0x01010101 % 32 == 1). We
    // construct the INTENDED value directly instead of relying on that
    // coincidence.
    Graph *g = new Graph();
    g->nnodes = n;
    g->nedges = e_cnt;
    g->rowptr = new graph_edge_t[n + 1];
    g->colidx = new graph_node_t[e_cnt];
    g->vertex_label = new bitarray32[n];
    for (int32_t i = 0; i <= n; i++) g->rowptr[i] = (graph_edge_t)rowptr[i];
    for (int64_t i = 0; i < e_cnt; i++) g->colidx[i] = (graph_node_t)colidx[i];
    for (int32_t i = 0; i < n; i++) g->vertex_label[i] = (bitarray32)(1u << 1);

    // Graph -> GPU: mirrors GraphPreprocessor::to_gpu() (source/src/
    // graph.h) exactly, reproduced here since we build Graph ourselves
    // rather than through the file-based GraphPreprocessor.
    Graph gcopy = *g;
    ST_CUDA_CHECK(cudaMalloc(&gcopy.vertex_label, sizeof(bitarray32) * n));
    ST_CUDA_CHECK(cudaMalloc(&gcopy.rowptr, sizeof(graph_edge_t) * (n + 1)));
    ST_CUDA_CHECK(cudaMalloc(&gcopy.colidx, sizeof(graph_node_t) * e_cnt));
    ST_CUDA_CHECK(cudaMemcpy(gcopy.vertex_label, g->vertex_label, sizeof(bitarray32) * n, cudaMemcpyHostToDevice));
    ST_CUDA_CHECK(cudaMemcpy(gcopy.rowptr, g->rowptr, sizeof(graph_edge_t) * (n + 1), cudaMemcpyHostToDevice));
    ST_CUDA_CHECK(cudaMemcpy(gcopy.colidx, g->colidx, sizeof(graph_node_t) * e_cnt, cudaMemcpyHostToDevice));
    Graph *gpu_graph;
    ST_CUDA_CHECK(cudaMalloc(&gpu_graph, sizeof(Graph)));
    ST_CUDA_CHECK(cudaMemcpy(gpu_graph, &gcopy, sizeof(Graph), cudaMemcpyHostToDevice));

    // Pattern: artifact's own PatternPreprocessor(filename) (source/src/
    // pattern.h), unmodified -- reads the small auto-generated K_k pattern
    // text file adapter.py writes, then compiles the matching order/
    // partial order/set-ops schedule (STMatch's own contribution).
    PatternPreprocessor *pat = new PatternPreprocessor(std::string(pattern_path));
    Pattern *gpu_pattern = pat->to_gpu();

    // JobQueuePreprocessor: artifact's own initial-job-list builder
    // (source/src/job_queue.h), unmodified.
    JobQueuePreprocessor *jq = new JobQueuePreprocessor(*g, *pat);
    JobQueue *gpu_queue = jq->to_gpu();

    // ---- everything below is cu_test.cu's own main(), reproduced
    // verbatim, up to (NOT including) the _parallel_match<<<>>> launch. ----
    CallStack *gpu_callstack;
    graph_node_t *slot_storage;
    ST_CUDA_CHECK(cudaMalloc(&slot_storage,
                              sizeof(graph_node_t) * NWARPS_TOTAL * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE));

    std::vector<CallStack> stk(NWARPS_TOTAL);
    for (int i = 0; i < NWARPS_TOTAL; i++) {
        auto &s = stk[i];
        memset(s.iter, 0, sizeof(s.iter));
        memset(s.slot_size, 0, sizeof(s.slot_size));
        s.slot_storage = (graph_node_t(*)[UNROLL][GRAPH_DEGREE])(
            (char *)slot_storage + i * sizeof(graph_node_t) * MAX_SLOT_NUM * UNROLL * GRAPH_DEGREE);
    }
    ST_CUDA_CHECK(cudaMalloc(&gpu_callstack, NWARPS_TOTAL * sizeof(CallStack)));
    ST_CUDA_CHECK(cudaMemcpy(gpu_callstack, stk.data(), sizeof(CallStack) * NWARPS_TOTAL, cudaMemcpyHostToDevice));

    size_t *gpu_res;
    ST_CUDA_CHECK(cudaMalloc(&gpu_res, sizeof(size_t) * NWARPS_TOTAL));

    int *idle_warps;
    ST_CUDA_CHECK(cudaMalloc(&idle_warps, sizeof(int) * GRID_DIM));
    int *idle_warps_count;
    ST_CUDA_CHECK(cudaMalloc(&idle_warps_count, sizeof(int)));
    int *global_mutex;
    ST_CUDA_CHECK(cudaMalloc(&global_mutex, sizeof(int) * GRID_DIM));
    // cu_test.cu also allocates `stk_valid` (bool[GRID_DIM]) but never
    // passes it to _parallel_match (confirmed against the kernel's own
    // parameter list, gpu_match.cuh) -- dead code in the artifact's own
    // driver, not reproduced here.

    StHandle *h = new StHandle();
    h->g = g;
    h->gpu_graph = gpu_graph;
    h->pat = pat;
    h->gpu_pattern = gpu_pattern;
    h->jq = jq;
    h->gpu_queue = gpu_queue;
    h->gpu_callstack = gpu_callstack;
    h->slot_storage = slot_storage;
    h->gpu_res = gpu_res;
    h->idle_warps = idle_warps;
    h->idle_warps_count = idle_warps_count;
    h->global_mutex = global_mutex;
    h->initial_stk = std::move(stk);
    h->multiplicity = pat->PatternMultiplicity;
    return h;
}

unsigned long long st_run(void *handle) {
    StHandle *h = (StHandle *)handle;

    // Reset every piece of per-call mutable device state to its pristine
    // prepare()-time value -- needed because this harness reuses one
    // handle across warmup+measured reps, unlike the artifact's own
    // single-shot CLI (cu_test.cu), which only initializes these once
    // before its one launch.
    ST_CUDA_CHECK(cudaMemcpy(h->gpu_callstack, h->initial_stk.data(),
                              sizeof(CallStack) * NWARPS_TOTAL, cudaMemcpyHostToDevice));
    ST_CUDA_CHECK(cudaMemset(h->gpu_res, 0, sizeof(size_t) * NWARPS_TOTAL));
    ST_CUDA_CHECK(cudaMemset(h->idle_warps, 0, sizeof(int) * GRID_DIM));
    ST_CUDA_CHECK(cudaMemset(h->idle_warps_count, 0, sizeof(int)));
    ST_CUDA_CHECK(cudaMemset(h->global_mutex, 0, sizeof(int) * GRID_DIM));

    // Job-queue cursor/mutex reset, round-tripped through host memory so
    // the device-resident job-array pointer (never touched after
    // prepare()) is preserved rather than clobbered by a host pointer
    // value.
    JobQueue qsnap;
    ST_CUDA_CHECK(cudaMemcpy(&qsnap, h->gpu_queue, sizeof(JobQueue), cudaMemcpyDeviceToHost));
    qsnap.cur = 0;
    qsnap.mutex = 0;
    ST_CUDA_CHECK(cudaMemcpy(h->gpu_queue, &qsnap, sizeof(JobQueue), cudaMemcpyHostToDevice));

    _parallel_match<<<GRID_DIM, BLOCK_DIM>>>(h->gpu_graph, h->gpu_pattern, h->gpu_callstack,
                                              h->gpu_queue, h->gpu_res, h->idle_warps,
                                              h->idle_warps_count, h->global_mutex);
    ST_CUDA_CHECK(cudaPeekAtLastError());
    ST_CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<size_t> res(NWARPS_TOTAL);
    ST_CUDA_CHECK(cudaMemcpy(res.data(), h->gpu_res, sizeof(size_t) * NWARPS_TOTAL, cudaMemcpyDeviceToHost));
    unsigned long long tot = 0;
    for (int i = 0; i < NWARPS_TOTAL; i++) tot += res[i];

    // NOTE: deliberately NOT multiplying by h->multiplicity here -- see
    // this file's header comment ("Counting semantics") and STATUS.md for
    // the empirical verification that the RAW per-warp sum is the exact
    // distinct-subset clique count this track's domain gates on, while
    // STMatch's own main()'s `tot_count * PatternMultiplicity` convention
    // (which this adapter exposes via `multiplicity` for anyone auditing
    // this choice) would overcount a complete-graph pattern by k!.
    return tot;
}

long long st_debug_info(void *handle, long long *job_queue_length) {
    StHandle *h = (StHandle *)handle;
    *job_queue_length = (long long)h->jq->q.length;
    return h->multiplicity;
}

void st_free(void *handle) {
    StHandle *h = (StHandle *)handle;
    // Free device Graph's sub-allocations: h->gpu_graph is a device pointer
    // to a Graph struct whose OWN fields (rowptr/colidx/vertex_label) are
    // themselves device pointers -- fetch that struct back to host memory
    // first (we kept no separate record of those sub-pointers) so cudaFree
    // gets valid host-side copies of the device addresses to free, rather
    // than dereferencing a device pointer from host code.
    Graph gcopy;
    cudaMemcpy(&gcopy, h->gpu_graph, sizeof(Graph), cudaMemcpyDeviceToHost);
    cudaFree(gcopy.vertex_label);
    cudaFree(gcopy.rowptr);
    cudaFree(gcopy.colidx);
    cudaFree(h->gpu_graph);
    cudaFree(h->gpu_pattern);

    JobQueue qcopy;
    cudaMemcpy(&qcopy, h->gpu_queue, sizeof(JobQueue), cudaMemcpyDeviceToHost);
    cudaFree(qcopy.q);
    cudaFree(h->gpu_queue);

    cudaFree(h->gpu_callstack);
    cudaFree(h->slot_storage);
    cudaFree(h->gpu_res);
    cudaFree(h->idle_warps);
    cudaFree(h->idle_warps_count);
    cudaFree(h->global_mutex);

    delete h->jq;
    delete h->pat;
    delete[] h->g->rowptr;
    delete[] h->g->colidx;
    delete[] h->g->vertex_label;
    delete h->g;
    delete h;
}

}  // extern "C"
