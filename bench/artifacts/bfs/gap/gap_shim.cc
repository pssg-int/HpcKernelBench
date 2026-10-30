// gap_shim.cc -- NOT part of the artifact. Exposes the GAP Benchmark Suite's
// direction-optimizing BFS (DOBFS in source/src/bfs.cc, unmodified) through a
// small extern "C" API for ctypes. bfs.cc is compiled in with its main()
// renamed, so DOBFS, InitParent, TDStep, BUStep are GAP's own code.
//
//   gap_prepare()      host CSR (and CSC for directed graphs) -> GAP CSRGraph,
//                      built the way GAP's own Builder lays it out (one
//                      neighbor array + GenIndex pointer index). Preprocessing.
//   gap_run()          DOBFS(g, source) only, timed internally with GAP's own
//                      Timer around exactly the call GAP's BenchmarkKernel
//                      times (so *internal_ms matches GAP's "Trial Time").
//   gap_copy_parent()  the parent array of the last run (untimed)
//   gap_free()
//
// Parent encoding (GAP's): parent[source] = source, parent[v] = -1 when
// unreached; the adapter turns the tree into BFS levels.
#define main gapbs_bfs_main
#include "bfs.cc"
#undef main

#include <omp.h>

#include <cstdint>
#include <cstring>
#include <exception>

namespace {
struct GapHandle {
  Graph g;
  pvector<NodeID> parent;
  int64_t n = 0;
};

NodeID* copy_neighbors(int64_t nnz, const int32_t* idx) {
  NodeID* neighs = new NodeID[nnz];
  std::memcpy(neighs, idx, sizeof(NodeID) * nnz);
  return neighs;
}

pvector<SGOffset> copy_offsets(int64_t n, const int64_t* off) {
  pvector<SGOffset> offsets(n + 1);
  for (int64_t i = 0; i <= n; i++) offsets[i] = off[i];
  return offsets;
}
}  // namespace

extern "C" {

void* gap_prepare(int64_t n, const int64_t* out_off, const int32_t* out_idx,
                  const int64_t* in_off, const int32_t* in_idx, int directed) {
  try {
    auto* h = new GapHandle();
    h->n = n;
    NodeID* out_neighs = copy_neighbors(out_off[n], out_idx);
    NodeID** out_index = Graph::GenIndex(copy_offsets(n, out_off), out_neighs);
    if (directed) {
      NodeID* in_neighs = copy_neighbors(in_off[n], in_idx);
      NodeID** in_index = Graph::GenIndex(copy_offsets(n, in_off), in_neighs);
      h->g = Graph(n, out_index, out_neighs, in_index, in_neighs);
    } else {
      h->g = Graph(n, out_index, out_neighs);
    }
    return h;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gap_prepare: %s\n", e.what());
    return nullptr;
  }
}

int gap_run(void* handle, int32_t source, int alpha, int beta, double* internal_ms) {
  try {
    auto* h = static_cast<GapHandle*>(handle);
    Timer t;
    t.Start();
    h->parent = DOBFS(h->g, source, false, alpha, beta);
    t.Stop();
    if (internal_ms) *internal_ms = t.Millisecs();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "gap_run: %s\n", e.what());
    return 1;
  }
}

int gap_copy_parent(void* handle, int32_t* out) {
  auto* h = static_cast<GapHandle*>(handle);
  if (h->parent.size() != static_cast<size_t>(h->n)) return 1;
  std::memcpy(out, h->parent.data(), sizeof(NodeID) * h->n);
  return 0;
}

int gap_num_threads() { return omp_get_max_threads(); }

void gap_free(void* handle) { delete static_cast<GapHandle*>(handle); }

}  // extern "C"
