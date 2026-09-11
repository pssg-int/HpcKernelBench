// GraphFold shim for the graph-pattern-mining track (headline pattern:
// exact k-clique counting, k=4 default -- see bench/kernelbench/domains/
// graph.py's module docstring deviation (4) for why K4 is the domain's
// headline).
//
// NOT part of the artifact (same role as bench/artifacts/triangle-counting/
// tot/tot_shim.cu). GraphFold's own entry points are CLI drivers
// (test/test_cliquefinding.cu) that always read a Matrix-Market FILE with
// their own Loader, then call Engine::RunCF() which bundles orientation +
// H2D copy + the CFSolver kernel into one call. Per ARTIFACT_GUIDE.md rule 1
// ("wrap the kernel, not the paper's benchmark script"), this shim instead:
//
//   gf_prepare() -- builds a project_GraphFold::Graph<int,int> DIRECTLY from
//                   our own CSR arrays (row_ptr/col_idx), bypassing
//                   GraphFold's own Loader/_load_mtx entirely (Loader just
//                   reads (v0,v1) pairs off disk into the exact same
//                   std::vector<std::pair<VID,VID>> edge list Init() wants
//                   -- see src/graph/io.h's Loader::Build/_load_mtx and
//                   src/graph/graph.h's Graph::Init -- so handing Init() the
//                   CSR's own (row,col) pairs directly is equivalent, minus
//                   a redundant file round-trip). Then calls hg->orientation()
//                   (DAG construction) and hg->copyToDevice() (H2D) -- EXACTLY
//                   Engine::RunCF()'s own preprocessing steps
//                   (src/Engine.h: "orientation here" comment), done ONCE.
//                   Our CSR is already cleaned + symmetrized (both edge
//                   directions present) before this call, matching what
//                   GraphFold's own .mtx graphs provide to the same Init()
//                   path (their _load_mtx does NOT symmetrize -- the mtx
//                   files it ships already list both directions).
//   gf_count()   -- calls CFSolver(*hg, k, result, 1, e_centric) ONLY: the
//                   k-clique-counting CUDA kernel itself (clique4/5/6/7_
//                   graphfold in src/gpu_kernels/clique_kernels.cuh), ToT's
//                   own "CFk matching time" timed region (src/CFSolver.cuh).
//                   This is the kernel-only call run() makes.
//   gf_free()    -- releases the Graph.
//
// Zero lines of source/ were modified.

#include <utility>
#include <vector>
#include <cstdint>

#include "src/Engine.h"
#include "src/graph/graph.h"

using namespace project_GraphFold;

struct GFHandle {
  Graph<int, int> *hg;
  int k;
};

extern "C" {

void *gf_prepare(int n, long long nnz, const int *row_ptr, const int *col_idx,
                  int k) {
  std::vector<int> vids(n);
  for (int i = 0; i < n; ++i) vids[i] = i;
  std::vector<int> vlabels; // empty: use_label=false throughout

  std::vector<std::pair<int, int>> edges;
  edges.reserve((size_t)nnz);
  for (int u = 0; u < n; ++u) {
    for (int j = row_ptr[u]; j < row_ptr[u + 1]; ++j) {
      edges.emplace_back(u, col_idx[j]);
    }
  }

  Graph<int, int> *hg = new Graph<int, int>();
  hg->Init(vids, vlabels, edges, 1, false);
  // preprocessing -- matches Engine::RunCF()'s own two calls before CFSolver
  hg->orientation();
  hg->copyToDevice(0, hg->get_enum(), 1, false, false);

  GFHandle *h = new GFHandle{hg, k};
  return h;
}

unsigned long long gf_count(void *handle) {
  GFHandle *h = static_cast<GFHandle *>(handle);
  uint64_t result = 0;
  CFSolver(*h->hg, h->k, result, 1, e_centric);
  return (unsigned long long)result;
}

void gf_free(void *handle) {
  GFHandle *h = static_cast<GFHandle *>(handle);
  if (h) {
    delete h->hg;
    delete h;
  }
}

} // extern "C"
