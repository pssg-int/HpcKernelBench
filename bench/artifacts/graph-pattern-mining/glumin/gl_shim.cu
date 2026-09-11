// GLumin shim for the graph-pattern-mining track (headline pattern: exact
// k-clique counting, k=4 default -- see bench/kernelbench/domains/graph.py's
// module docstring deviation (4)).
//
// NOT part of the artifact (same role as bench/artifacts/triangle-counting/
// tot/tot_shim.cu and bench/artifacts/graph-pattern-mining/graphfold/
// gf_shim.cu). GLumin wraps the "+LUT" variant of G2Miner's own
// CliqueSolver (src/clique/clique_GM_LUT.cu's `void CliqueSolver(Graph&,
// int k, uint64_t&, int, int)`) -- this is GLumin's actual paper
// contribution (its README/survey.md: "GLumin's contribution is a drop-in
// connectivity-check [LUT] accelerator" applied on top of G2Miner/GraphFold/
// AutoMine; the plain, non-LUT `clique_GM.cu` is just the G2Miner baseline).
//
// CliqueSolver() itself is monolithic: it builds `GraphGPU gg(g);
// gg.init_edgelist(g);` (H2D transfer) AND allocates its kernel workspace
// (frontier_list, BinaryEncode LUT buffer, d_total) INSIDE the same function
// call that launches the counting kernel, with only its own internal
// wall-clock Timer (not exposed to a caller) separating "preprocessing" from
// "kernel" -- there is no public sub-boundary to call into. Per
// ARTIFACT_GUIDE.md rule 1 ("wrap at the finest boundary available"), this
// shim reproduces CliqueSolver's OWN setup code (unmodified: same GraphGPU/
// BinaryEncode/maximum_residency calls, same launch-config arithmetic,
// literally copied from clique_GM_LUT.cu) split across two functions instead
// of one:
//
//   gl_prepare() -- builds a Graph DIRECTLY from our own CSR arrays (via the
//                   artifact's own allocateFrom()/fixEndEdge()/
//                   constructEdge() public API -- the SAME pattern the
//                   artifact's own src/common/graph_partition.cc uses to
//                   build a Graph from arrays rather than a file), calls
//                   g.orientation() (DAG construction, same as passing
//                   USE_DAG=true to the file-based constructor), then
//                   GraphGPU(g)/init_edgelist(g) (H2D) and all of
//                   CliqueSolver's own workspace allocation (frontier_list,
//                   BinaryEncode<> LUT buffer, d_total, launch-config
//                   arithmetic) -- ALL of this is CliqueSolver's own
//                   pre-Timer.Start() code, untimed by the artifact itself.
//   gl_count()   -- launches ONLY the appropriate cliqueK_warp_*_subgraph
//                   kernel (K=4..7, the artifact's own unmodified __global__
//                   functions) + D2H readback of the count -- CliqueSolver's
//                   own Timer.Start()/Timer.Stop() region. One addition not
//                   in the original (documented, not a kernel change): the
//                   accumulator `d_total` is explicitly reset to 0 before
//                   each launch, because our harness calls run() repeatedly
//                   (warmup+reps) on the SAME handle, whereas the artifact's
//                   own CLI driver only ever calls CliqueSolver() once per
//                   process (so it only zeroes d_total once, before its
//                   single call) -- without this reset, counts would
//                   accumulate across repetitions and fail the exact-count
//                   gate on the very first (correctness-gated) call already
//                   being repeated internally by warmup.
//   gl_free()    -- releases everything.
//
// Zero lines of source/ were modified; the kernel functions themselves
// (clique4_warp_vertex_subgraph etc., in src/clique/gpu_kernels/*.cuh) are
// #include'd and called verbatim.

#include <algorithm>
#include <cstdint>

#include <cub/cub.cuh>
#include "graph.h"
#include "graph_gpu.h"
#include "binary_encode.h"
#include "operations.cuh"
#include "cutil_subset.h"
#include "cuda_launch_config.hpp"

#define BLK_SZ BLOCK_SIZE
typedef cub::BlockReduce<AccType, BLK_SZ> BlockReduce;
#include "clique4_warp_vertex_subgraph.cuh"
#include "clique5_warp_edge_subgraph.cuh"
#include "clique6_warp_edge_subgraph.cuh"
#include "clique7_warp_edge_subgraph.cuh"

struct GLHandle {
  Graph *g;
  GraphGPU *gg;
  vidType *frontier_list;
  AccType *d_total;
  BinaryEncode<> *sub_graph;
  size_t nblocks;
  size_t nthreads;
  vidType nv;
  eidType ne;
  vidType max_degree;
  int k;
};

extern "C" {

void *gl_prepare(int n, long long nnz_in, const int *row_ptr, const int *col_idx, int k) {
  // Build the Graph directly from our own CSR (row_ptr/col_idx), via the
  // artifact's own allocateFrom/fixEndEdge/constructEdge API -- the default
  // Graph() ctor zero-inits `nnz` (needed so Graph::init_edgelist()'s
  // `if (nnz != 0) return;` guard below doesn't short-circuit on garbage),
  // then allocateFrom() allocates the CSR buffers we fill in.
  Graph *g = new Graph();
  g->allocateFrom((vidType)n, (eidType)nnz_in);
  for (int v = 0; v < n; ++v) {
    g->fixEndEdge((vidType)v, (eidType)row_ptr[v + 1]);
  }
  for (long long e = 0; e < nnz_in; ++e) {
    g->constructEdge((eidType)e, (vidType)col_idx[e]);
  }
  g->orientation(); // DAG orientation -- required for k>3 clique kernels

  GraphGPU *gg = new GraphGPU(*g); // init(g): H2D copy of oriented CSR
  gg->init_edgelist(*g);           // builds device-side COO edge list

  vidType nv = g->num_vertices();
  eidType ne = g->num_edges();
  vidType md = g->get_max_degree();

  // --- everything below verbatim from CliqueSolver (clique_GM_LUT.cu) ---
  size_t nwarps = WARPS_PER_BLOCK;
  size_t nthreads = BLK_SZ;
  size_t nblocks = (size_t)(ne - 1) / WARPS_PER_BLOCK + 1;
  size_t per_block_vlist_size = nwarps * size_t(k - 3) * size_t(md) * sizeof(vidType);
  if (nblocks > 65536) nblocks = 65536;

  size_t memsize = get_gpu_mem_size(0);
  size_t mem_graph = size_t(nv + 1) * sizeof(eidType) + size_t(2) * size_t(ne) * sizeof(vidType);
  size_t nb = (memsize - mem_graph) / per_block_vlist_size;
  if (nb < nblocks) nblocks = nb;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int max_blocks_per_SM = 0;
  if (k == 4) max_blocks_per_SM = maximum_residency(clique4_warp_vertex_subgraph, nthreads, 0);
  else if (k == 5) max_blocks_per_SM = maximum_residency(clique5_warp_edge_subgraph, nthreads, 0);
  else if (k == 6) max_blocks_per_SM = maximum_residency(clique6_warp_edge_subgraph, nthreads, 0);
  else if (k == 7) max_blocks_per_SM = maximum_residency(clique7_warp_edge_subgraph, nthreads, 0);
  size_t max_blocks = (size_t)max_blocks_per_SM * deviceProp.multiProcessorCount;
  nblocks = std::min(max_blocks, nblocks);

  size_t list_size = nblocks * per_block_vlist_size;
  vidType *frontier_list;
  CUDA_SAFE_CALL(cudaMalloc((void **)&frontier_list, list_size));

  AccType *d_total;
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_total, sizeof(AccType)));

  BinaryEncode<> *sub_graph = new BinaryEncode<>((int)(nblocks * nwarps), (int)md, (int)md);
  // --- end verbatim CliqueSolver setup ---

  GLHandle *h = new GLHandle{g, gg, frontier_list, d_total, sub_graph,
                              nblocks, nthreads, nv, ne, md, k};
  return h;
}

unsigned long long gl_count(void *handle) {
  GLHandle *h = static_cast<GLHandle *>(handle);
  AccType h_total = 0;
  // reset accumulator before every call -- see file header comment
  CUDA_SAFE_CALL(cudaMemcpy(h->d_total, &h_total, sizeof(AccType), cudaMemcpyHostToDevice));

  if (h->k == 4) {
    clique4_warp_vertex_subgraph<<<h->nblocks, h->nthreads>>>(
        h->nv, *h->gg, h->frontier_list, h->max_degree, h->d_total, *h->sub_graph);
  } else if (h->k == 5) {
    clique5_warp_edge_subgraph<<<h->nblocks, h->nthreads>>>(
        h->ne, *h->gg, h->frontier_list, h->max_degree, h->d_total, *h->sub_graph);
  } else if (h->k == 6) {
    clique6_warp_edge_subgraph<<<h->nblocks, h->nthreads>>>(
        h->ne, *h->gg, h->frontier_list, h->max_degree, h->d_total, *h->sub_graph);
  } else if (h->k == 7) {
    clique7_warp_edge_subgraph<<<h->nblocks, h->nthreads>>>(
        h->ne, *h->gg, h->frontier_list, h->max_degree, h->d_total, *h->sub_graph);
  }
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  CUDA_SAFE_CALL(cudaMemcpy(&h_total, h->d_total, sizeof(AccType), cudaMemcpyDeviceToHost));
  return (unsigned long long)h_total;
}

void gl_free(void *handle) {
  GLHandle *h = static_cast<GLHandle *>(handle);
  if (h) {
    cudaFree(h->frontier_list);
    cudaFree(h->d_total);
    delete h->sub_graph;
    delete h->gg;
    delete h->g;
    delete h;
  }
}

} // extern "C"
