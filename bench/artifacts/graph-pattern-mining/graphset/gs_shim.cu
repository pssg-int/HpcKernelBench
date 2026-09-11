// GraphSet bridge (this directory, NOT part of the artifact).
//
// Paper: "GraphSet: High Performance Graph Mining through Equivalent Set
// Transformations" (SC'23, conf/sc/ShiZWCZHYC23).
// Artifact: https://github.com/sth1997/GraphSet
//
// Headline pattern wrapped: exact k-clique COUNTING, matching bench/
// kernelbench/domains/graph.py's graph-pattern-mining headline
// (`params['clique_k']`, default 4). GraphSet ships a DEDICATED clique
// driver, `source/gpu/gpu_clique.cu`'s `main()`: it builds a Pattern that
// IS a complete graph K_k (`for i: for j>i: add_edge(i,j)`), compiles it
// via `Schedule_IEP` -- the equivalent-set-transformation schedule search
// that IS GraphSet's actual paper contribution (in-exclusion-optimized
// prefix sharing across the pattern's automorphisms) -- and launches
// `gpu_pattern_matching`, the same general pattern-matching kernel every
// GraphSet application (pattern matching / clique / motif counting) uses,
// specialized only by the compiled Schedule_IEP it is handed. This is
// therefore GraphSet's own "clique as its own pattern, adjacency-matrix
// driven" path, not a hand-adapted general-pattern-matching call.
//
// `pattern_matching_init()` (gpu_clique.cu) is monolithic exactly like
// GLumin's `CliqueSolver()` and TileSpGEMM's `tilespgemm()`: it builds the
// device Graph representation AND the device `GPUSchedule` AND launches
// the kernel AND reads back the count, all in one function, with no public
// sub-boundary. Per ARTIFACT_GUIDE.md rule 1, this file reproduces that
// function's own body VERBATIM (same calls, same order, copied not
// rewritten), split into two:
//
//   gs_prepare() -- builds the artifact's own `Graph` struct DIRECTLY from
//                   our CSR (no file I/O, no DataLoader's binary .g
//                   format), calls the artifact's own
//                   `reduce_edges_for_clique()` (== `erase_edge()`,
//                   source/src/graph.cpp, unmodified) for the clique-
//                   specific DAG orientation, builds the K_k `Pattern` +
//                   `Schedule_IEP` (schedule compilation, pattern-only,
//                   graph-independent), then reproduces
//                   `pattern_matching_init()`'s own H2D setup (edge_from
//                   computation, dev_edge/dev_edge_from/dev_vertex/dev_tmp
//                   allocation+copy) and its entire `GPUSchedule` device-
//                   struct construction (all `cudaMallocManaged`+`cudaMemcpy`
//                   calls) verbatim -- exactly `pattern_matching_init()`'s
//                   own pre-launch scope.
//   gs_run()     -- resets `dev_sum`/`dev_cur_edge` to 0 (needed because
//                   this harness reuses one handle across warmup+reps,
//                   unlike the artifact's own single-shot CLI, which only
//                   zeroes them once via `cudaMemcpyToSymbol` before its
//                   one call -- same idiom GLumin/fringe-sgc's adapters in
//                   this repo already use), launches
//                   `gpu_pattern_matching<<<>>>` (the artifact's own
//                   unmodified `__global__` kernel, DEFINED in
//                   source/gpu/gpu_clique.cu and declared `extern` here --
//                   built together with `-rdc=true` so the device symbols
//                   resolve across translation units), then reads back
//                   `dev_sum` and divides by
//                   `get_in_exclusion_optimize_redundancy()` -- exactly
//                   `pattern_matching_init()`'s own post-launch scope.
//   gs_free()    -- releases everything.
//
// Zero lines of source/ were modified. `gpu_pattern_matching`,
// `GPU_pattern_matching_func`, `GPUVertexSet`, `intersection2`,
// `do_intersection`, `dev_sum`, `dev_cur_edge` all remain exclusively in
// the unmodified source/gpu/gpu_clique.cu, compiled verbatim alongside
// this file (build.sh) and referenced here only via `extern` declarations.

#include <cstdint>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>

#include "graph.h"
#include "pattern.h"
#include "schedule_IEP.h"
#include "vertex_set.h"
#include "component/gpu_schedule.cuh"

constexpr int THREADS_PER_BLOCK = 256;
constexpr int THREADS_PER_WARP = 32;
constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / THREADS_PER_WARP;

// Defined, unmodified, in source/gpu/gpu_clique.cu; resolved at device-link
// time via -rdc=true (see build.sh). NOT redefined here.
extern __device__ unsigned long long dev_sum;
extern __device__ unsigned int dev_cur_edge;
extern __global__ void gpu_pattern_matching(uint32_t edge_num, uint32_t buffer_size,
                                             uint32_t *edge_from, uint32_t *edge,
                                             e_index_t *vertex, uint32_t *tmp,
                                             const GPUSchedule *schedule);

// Exact layout mirror of gpu_clique.cu's own (file-local, non-header)
// `GPUVertexSet` class -- used ONLY to compute `sizeof()` for the launch's
// dynamic shared-memory size, matching pattern_matching_init()'s own
// `sizeof(GPUVertexSet)` computation. `GPUVertexSet` has no virtual
// functions and exactly these two data members in this order (verified by
// reading gpu_clique.cu directly), so the Itanium C++ ABI (both host g++
// and nvcc's device compiler use it) guarantees an identical size/
// alignment to the real class. This mirror is never instantiated as an
// actual GPUVertexSet nor crosses the ctypes boundary -- the real class,
// with all its methods, stays exclusively inside the unmodified
// gpu_clique.cu.
struct GpuVertexSetSizeofMirror {
    uint32_t size;
    uint32_t *data;
};

#define GS_CUDA_CHECK(x)                                                        \
    do {                                                                        \
        cudaError_t _e = (x);                                                   \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "gs_shim CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_e));                                    \
        }                                                                       \
    } while (0)

extern "C" {

struct GsHandle {
    Graph *g;
    Pattern *pat;
    Schedule_IEP *sched;
    uint32_t *dev_edge;
    uint32_t *dev_edge_from;
    e_index_t *dev_vertex;
    uint32_t *dev_tmp;
    GPUSchedule *dev_schedule;
    int num_blocks;
    uint32_t buffer_size;
    size_t block_shmem_size;
    long long redundancy;
};

void *gs_prepare(int32_t n, int64_t e_cnt, const int64_t *vertex_ptr,
                  const int32_t *edge_idx, int32_t k) {
    // Artifact's own Graph struct (source/include/graph.h), built directly
    // from our CSR (BOTH directions present -- reduce_edges_for_clique
    // below requires a full symmetric adjacency, the same convention
    // source/src/dataloader.cpp's own binary .g loader assumes). new[]-
    // allocated to match Graph::~Graph()'s delete[].
    Graph *g = new Graph();
    g->v_cnt = n;
    g->e_cnt = e_cnt;
    g->tri_cnt = -1;
    g->vertex = new e_index_t[n + 1];
    g->edge = new v_index_t[e_cnt];
    g->edge_from = nullptr;
    for (int32_t i = 0; i <= n; i++) g->vertex[i] = (e_index_t)vertex_ptr[i];
    for (int64_t i = 0; i < e_cnt; i++) g->edge[i] = (v_index_t)edge_idx[i];

    // VertexSet::max_intersection_size: normally baked into the binary .g
    // header by the artifact's own offline dataset preparation
    // (dataloader.cpp's dump_graph, from the SECOND-largest degree). We
    // bypass that file format entirely, so set it here directly from the
    // actual MAX out-degree of the (pre-reduction) input -- a safe, at
    // least as conservative, bound: the intersection of any two vertices'
    // neighbor sets can never exceed the smaller set's own size, which is
    // bounded above by the largest degree in the graph.
    int32_t max_deg = 0;
    for (int32_t i = 0; i < n; i++) {
        int64_t d = g->vertex[i + 1] - g->vertex[i];
        if (d > max_deg) max_deg = (int32_t)d;
    }
    VertexSet::max_intersection_size = max_deg;

    // reduce_edges_for_clique() == erase_edge() (source/src/graph.cpp,
    // unmodified): keeps, for every undirected edge {u,v}, exactly the
    // single directed entry max(u,v) -> min(u,v) -- a DAG total-ordered by
    // vertex id. This IS the artifact's own clique-specific preprocessing:
    // its CPU clique_test.cpp driver calls this exact function before
    // building the schedule too.
    reduce_edges_for_clique(*g);

    // Pattern/Schedule_IEP construction: EXACTLY gpu_clique.cu's own
    // main() -- a complete graph K_k as the Pattern, compiled via
    // Schedule_IEP's equivalent-set-transformation schedule search (this
    // IS GraphSet's paper contribution). Pure combinatorial preprocessing
    // over the PATTERN only (independent of the runtime graph).
    Pattern *pat = new Pattern(k);
    for (int i = 0; i < k; i++)
        for (int j = i + 1; j < k; j++)
            pat->add_edge(i, j);
    Schedule_IEP *schedule_iep = new Schedule_IEP(pat->get_adj_mat_ptr(), k);
    schedule_iep->set_in_exclusion_optimize_redundancy(1);

    // ---- everything below is pattern_matching_init()'s own body
    // (source/gpu/gpu_clique.cu), reproduced verbatim up to (NOT
    // including) the gpu_pattern_matching<<<>>> launch. ----
    int num_blocks = 1024;
    int num_total_warps = num_blocks * WARPS_PER_BLOCK;

    size_t size_edge = (size_t)g->e_cnt * sizeof(v_index_t);
    size_t size_vertex = (size_t)(g->v_cnt + 1) * sizeof(e_index_t);
    size_t size_tmp = (size_t)VertexSet::max_intersection_size * sizeof(uint32_t) *
                       num_total_warps * (schedule_iep->get_total_prefix_num() + 2);

    uint32_t *edge_from_h = new uint32_t[g->e_cnt];
    for (v_index_t i = 0; i < g->v_cnt; ++i)
        for (e_index_t j = g->vertex[i]; j < g->vertex[i + 1]; ++j)
            edge_from_h[j] = i;

    uint32_t *dev_edge, *dev_edge_from, *dev_tmp;
    e_index_t *dev_vertex;
    GS_CUDA_CHECK(cudaMalloc((void **)&dev_edge, size_edge));
    GS_CUDA_CHECK(cudaMalloc((void **)&dev_edge_from, size_edge));
    GS_CUDA_CHECK(cudaMalloc((void **)&dev_vertex, size_vertex));
    GS_CUDA_CHECK(cudaMalloc((void **)&dev_tmp, size_tmp));

    GS_CUDA_CHECK(cudaMemcpy(dev_edge, g->edge, size_edge, cudaMemcpyHostToDevice));
    GS_CUDA_CHECK(cudaMemcpy(dev_edge_from, edge_from_h, size_edge, cudaMemcpyHostToDevice));
    GS_CUDA_CHECK(cudaMemcpy(dev_vertex, g->vertex, size_vertex, cudaMemcpyHostToDevice));
    delete[] edge_from_h;

    GPUSchedule *dev_schedule;
    GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule, sizeof(GPUSchedule)));
    int schedule_size = schedule_iep->get_size();
    int max_prefix_num = schedule_size * (schedule_size - 1) / 2;

    bool *only_need_size = new bool[max_prefix_num];
    for (int i = 0; i < max_prefix_num; ++i)
        only_need_size[i] = schedule_iep->get_prefix_only_need_size(i);

    int in_exclusion_optimize_vertex_id_size = schedule_iep->in_exclusion_optimize_vertex_id.size();
    int in_exclusion_optimize_array_size = schedule_iep->in_exclusion_optimize_coef.size();

    int *in_exclusion_optimize_vertex_id = new int[in_exclusion_optimize_vertex_id_size];
    bool *in_exclusion_optimize_vertex_flag = new bool[in_exclusion_optimize_vertex_id_size];
    int *in_exclusion_optimize_vertex_coef = new int[in_exclusion_optimize_vertex_id_size];
    int *in_exclusion_optimize_coef = new int[in_exclusion_optimize_array_size];
    bool *in_exclusion_optimize_flag = new bool[in_exclusion_optimize_array_size];
    int *in_exclusion_optimize_ans_pos = new int[in_exclusion_optimize_array_size];

    for (int i = 0; i < in_exclusion_optimize_vertex_id_size; ++i) {
        in_exclusion_optimize_vertex_id[i] = schedule_iep->in_exclusion_optimize_vertex_id[i];
        in_exclusion_optimize_vertex_flag[i] = schedule_iep->in_exclusion_optimize_vertex_flag[i];
        in_exclusion_optimize_vertex_coef[i] = schedule_iep->in_exclusion_optimize_vertex_coef[i];
    }
    for (int i = 0; i < in_exclusion_optimize_array_size; ++i) {
        in_exclusion_optimize_coef[i] = schedule_iep->in_exclusion_optimize_coef[i];
        in_exclusion_optimize_flag[i] = schedule_iep->in_exclusion_optimize_flag[i];
        in_exclusion_optimize_ans_pos[i] = schedule_iep->in_exclusion_optimize_ans_pos[i];
    }

    if (in_exclusion_optimize_vertex_id_size > 0) {
        GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->in_exclusion_optimize_vertex_id,
                                         sizeof(int) * in_exclusion_optimize_vertex_id_size));
        GS_CUDA_CHECK(cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_id,
                                  in_exclusion_optimize_vertex_id,
                                  sizeof(int) * in_exclusion_optimize_vertex_id_size,
                                  cudaMemcpyHostToDevice));
        GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->in_exclusion_optimize_vertex_flag,
                                         sizeof(bool) * in_exclusion_optimize_vertex_id_size));
        GS_CUDA_CHECK(cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_flag,
                                  in_exclusion_optimize_vertex_flag,
                                  sizeof(bool) * in_exclusion_optimize_vertex_id_size,
                                  cudaMemcpyHostToDevice));
        GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->in_exclusion_optimize_vertex_coef,
                                         sizeof(int) * in_exclusion_optimize_vertex_id_size));
        GS_CUDA_CHECK(cudaMemcpy(dev_schedule->in_exclusion_optimize_vertex_coef,
                                  in_exclusion_optimize_vertex_coef,
                                  sizeof(int) * in_exclusion_optimize_vertex_id_size,
                                  cudaMemcpyHostToDevice));
    }
    if (in_exclusion_optimize_array_size > 0) {
        GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->in_exclusion_optimize_coef,
                                         sizeof(int) * in_exclusion_optimize_array_size));
        GS_CUDA_CHECK(cudaMemcpy(dev_schedule->in_exclusion_optimize_coef,
                                  in_exclusion_optimize_coef,
                                  sizeof(int) * in_exclusion_optimize_array_size,
                                  cudaMemcpyHostToDevice));
        GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->in_exclusion_optimize_flag,
                                         sizeof(bool) * in_exclusion_optimize_array_size));
        GS_CUDA_CHECK(cudaMemcpy(dev_schedule->in_exclusion_optimize_flag,
                                  in_exclusion_optimize_flag,
                                  sizeof(bool) * in_exclusion_optimize_array_size,
                                  cudaMemcpyHostToDevice));
        GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->in_exclusion_optimize_ans_pos,
                                         sizeof(int) * in_exclusion_optimize_array_size));
        GS_CUDA_CHECK(cudaMemcpy(dev_schedule->in_exclusion_optimize_ans_pos,
                                  in_exclusion_optimize_ans_pos,
                                  sizeof(int) * in_exclusion_optimize_array_size,
                                  cudaMemcpyHostToDevice));
    }

    GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->adj_mat,
                                     sizeof(int) * schedule_size * schedule_size));
    GS_CUDA_CHECK(cudaMemcpy(dev_schedule->adj_mat, schedule_iep->get_adj_mat_ptr(),
                              sizeof(int) * schedule_size * schedule_size, cudaMemcpyHostToDevice));

    GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->father_prefix_id, sizeof(int) * max_prefix_num));
    GS_CUDA_CHECK(cudaMemcpy(dev_schedule->father_prefix_id, schedule_iep->get_father_prefix_id_ptr(),
                              sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->last, sizeof(int) * schedule_size));
    GS_CUDA_CHECK(cudaMemcpy(dev_schedule->last, schedule_iep->get_last_ptr(),
                              sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

    GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->next, sizeof(int) * max_prefix_num));
    GS_CUDA_CHECK(cudaMemcpy(dev_schedule->next, schedule_iep->get_next_ptr(),
                              sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->only_need_size, sizeof(bool) * max_prefix_num));
    GS_CUDA_CHECK(cudaMemcpy(dev_schedule->only_need_size, only_need_size,
                              sizeof(bool) * max_prefix_num, cudaMemcpyHostToDevice));

    GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->break_size, sizeof(int) * max_prefix_num));
    GS_CUDA_CHECK(cudaMemcpy(dev_schedule->break_size, schedule_iep->get_break_size_ptr(),
                              sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->loop_set_prefix_id, sizeof(int) * schedule_size));
    GS_CUDA_CHECK(cudaMemcpy(dev_schedule->loop_set_prefix_id, schedule_iep->get_loop_set_prefix_id_ptr(),
                              sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

    GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->restrict_last, sizeof(int) * schedule_size));
    GS_CUDA_CHECK(cudaMemcpy(dev_schedule->restrict_last, schedule_iep->get_restrict_last_ptr(),
                              sizeof(int) * schedule_size, cudaMemcpyHostToDevice));

    GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->restrict_next, sizeof(int) * max_prefix_num));
    GS_CUDA_CHECK(cudaMemcpy(dev_schedule->restrict_next, schedule_iep->get_restrict_next_ptr(),
                              sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    GS_CUDA_CHECK(cudaMallocManaged((void **)&dev_schedule->restrict_index, sizeof(int) * max_prefix_num));
    GS_CUDA_CHECK(cudaMemcpy(dev_schedule->restrict_index, schedule_iep->get_restrict_index_ptr(),
                              sizeof(int) * max_prefix_num, cudaMemcpyHostToDevice));

    dev_schedule->in_exclusion_optimize_array_size = in_exclusion_optimize_array_size;
    dev_schedule->in_exclusion_optimize_vertex_id_size = in_exclusion_optimize_vertex_id_size;
    dev_schedule->size = schedule_iep->get_size();
    dev_schedule->total_prefix_num = schedule_iep->get_total_prefix_num();
    dev_schedule->basic_prefix_num = schedule_iep->get_basic_prefix_num();
    dev_schedule->total_restrict_num = schedule_iep->get_total_restrict_num();
    dev_schedule->in_exclusion_optimize_num = schedule_iep->get_in_exclusion_optimize_num();

    size_t block_shmem_size =
        (size_t)(schedule_iep->get_total_prefix_num() + 2) * WARPS_PER_BLOCK * sizeof(GpuVertexSetSizeofMirror) +
        (size_t)in_exclusion_optimize_vertex_id_size * WARPS_PER_BLOCK * sizeof(int);
    dev_schedule->ans_array_offset =
        block_shmem_size - (size_t)in_exclusion_optimize_vertex_id_size * WARPS_PER_BLOCK * sizeof(int);

    delete[] only_need_size;
    delete[] in_exclusion_optimize_vertex_id;
    delete[] in_exclusion_optimize_vertex_flag;
    delete[] in_exclusion_optimize_vertex_coef;
    delete[] in_exclusion_optimize_coef;
    delete[] in_exclusion_optimize_flag;
    delete[] in_exclusion_optimize_ans_pos;

    GsHandle *h = new GsHandle();
    h->g = g;
    h->pat = pat;
    h->sched = schedule_iep;
    h->dev_edge = dev_edge;
    h->dev_edge_from = dev_edge_from;
    h->dev_vertex = dev_vertex;
    h->dev_tmp = dev_tmp;
    h->dev_schedule = dev_schedule;
    h->num_blocks = num_blocks;
    h->buffer_size = VertexSet::max_intersection_size;
    h->block_shmem_size = block_shmem_size;
    h->redundancy = schedule_iep->get_in_exclusion_optimize_redundancy();
    return h;
}

unsigned long long gs_run(void *handle) {
    GsHandle *h = (GsHandle *)handle;

    // Explicit reset before every call -- pattern_matching_init()'s own CLI
    // driver only ever runs once per process (so it only zeroes these once,
    // implicitly via static __device__ initializers), but this harness
    // reuses one handle across warmup+measured reps (same idiom GLumin's/
    // fringe-sgc's adapters in this repo already document and use).
    unsigned long long zero_sum = 0;
    unsigned int zero_edge = 0;
    GS_CUDA_CHECK(cudaMemcpyToSymbol(dev_sum, &zero_sum, sizeof(zero_sum)));
    GS_CUDA_CHECK(cudaMemcpyToSymbol(dev_cur_edge, &zero_edge, sizeof(zero_edge)));

    gpu_pattern_matching<<<h->num_blocks, THREADS_PER_BLOCK, h->block_shmem_size>>>(
        (uint32_t)h->g->e_cnt, h->buffer_size, h->dev_edge_from, h->dev_edge,
        h->dev_vertex, h->dev_tmp, h->dev_schedule);
    GS_CUDA_CHECK(cudaPeekAtLastError());
    GS_CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long sum = 0;
    GS_CUDA_CHECK(cudaMemcpyFromSymbol(&sum, dev_sum, sizeof(sum)));
    sum /= h->redundancy;
    return sum;
}

void gs_free(void *handle) {
    GsHandle *h = (GsHandle *)handle;
    cudaFree(h->dev_edge);
    cudaFree(h->dev_edge_from);
    cudaFree(h->dev_vertex);
    cudaFree(h->dev_tmp);
    if (h->dev_schedule) {
        cudaFree(h->dev_schedule->adj_mat);
        cudaFree(h->dev_schedule->father_prefix_id);
        cudaFree(h->dev_schedule->last);
        cudaFree(h->dev_schedule->next);
        cudaFree(h->dev_schedule->only_need_size);
        cudaFree(h->dev_schedule->break_size);
        cudaFree(h->dev_schedule->loop_set_prefix_id);
        cudaFree(h->dev_schedule->restrict_last);
        cudaFree(h->dev_schedule->restrict_next);
        cudaFree(h->dev_schedule->restrict_index);
        if (h->dev_schedule->in_exclusion_optimize_vertex_id_size > 0) {
            cudaFree(h->dev_schedule->in_exclusion_optimize_vertex_id);
            cudaFree(h->dev_schedule->in_exclusion_optimize_vertex_flag);
            cudaFree(h->dev_schedule->in_exclusion_optimize_vertex_coef);
        }
        if (h->dev_schedule->in_exclusion_optimize_array_size > 0) {
            cudaFree(h->dev_schedule->in_exclusion_optimize_coef);
            cudaFree(h->dev_schedule->in_exclusion_optimize_flag);
            cudaFree(h->dev_schedule->in_exclusion_optimize_ans_pos);
        }
        cudaFree(h->dev_schedule);
    }
    delete h->sched;
    delete h->pat;
    delete h->g;
    delete h;
}

}  // extern "C"
