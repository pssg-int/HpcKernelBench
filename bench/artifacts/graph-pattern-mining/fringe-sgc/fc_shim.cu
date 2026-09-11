// Fringe-SGC shim for the graph-pattern-mining track (headline pattern:
// exact k-clique counting; this artifact only reaches k=4 -- see below).
//
// NOT part of the artifact (same role as the other two adapters in this
// track). Fringe-SGC's own paper premise is counting small "core" motifs
// (vertex/edge/triangle/wedge, <=3 vertices) WITH additional degree-1
// "fringe" vertices attached -- the survey (benchspecs/graph-pattern-mining/
// survey.md) describes only those 4 predefined core shapes, which looks
// incompatible with this track's K4-clique headline at first glance.
// BUT src/fringePreprocess.cpp's `determineCore()` is fully general (greedy
// degree-based core/fringe peeling, core up to MaxCoreSize=6 -- see
// src/fringes.h), not hardcoded to 4 shapes; those 4 are just the paper's
// worked examples. Fed a K4 edge list, `determineCore` peels ONE vertex off
// as a fringe (all 4 K4 vertices have degree 3, so the peeling's x=3 pass
// picks one) whose anchor set is ALL 3 remaining (triangle) core vertices --
// i.e. K4 = triangle-core + one fringe vertex anchored to all 3 corners.
// This is verified BY RUNNING the artifact's own unmodified
// `fringePreprocess` binary with a K4 edge list (see build.sh): its own
// output prints "3 core nodes" / "anchor sets: A B C: 1", and
// src/fringeCount.cu's `triangleCore()` has a DEDICATED specialized path for
// exactly this case (`f[iuvw]==1, others 0` -> its own `clique<<<...>>>`
// kernel, literally named "clique" in their source) -- this is a real,
// intentionally-supported case in the artifact, not a fragile coincidence.
// k=5,6,7 do NOT reduce this way (peeling K5/K6/K7 does not terminate at a
// <=6-node "fringe+core" shape the way K4 does), so this adapter supports
// k=4 ONLY -- see adapter.py's `available()`/`prepare()` for the explicit
// k-range guard.
//
// fringePreprocess (src/fringePreprocess.cpp) is graph-INDEPENDENT: it only
// depends on the pattern (K4), never on the data graph being benchmarked.
// build.sh therefore builds and runs the artifact's own UNMODIFIED
// fringePreprocess binary ONCE (`./fringePreprocess 4 0 1 0 2 0 3 1 2 1 3
// 2 3`) to produce `k4.mo` (a serialized `MatchingOrder` struct, exactly
// fringeCount's own `main()` reads via `fread(&mo, 1, sizeof(MatchingOrder),
// f)`) -- this is pattern-level preprocessing done once, analogous to a
// query-compilation step, not a per-graph cost.
//
// fringeCount.cu's own `main()` (src/fringeCount.cu) does: readECLgraph (CPU
// load) -> H2D copy of the graph (untimed) -> `occurrences(mo, d_g, d_wl,
// SMs, mTpSM)` wrapped in the artifact's own CPUTimer (its own "kernel-only"
// timed region per survey.md) -> print count. Per ARTIFACT_GUIDE.md rule 1,
// this shim reproduces exactly that split instead of shelling out to the
// CLI binary:
//
//   fc_prepare() -- builds an ECLgraph directly from our own CSR arrays (no
//                   file I/O -- ECLgraph is just {nodes,edges,nindex,nlist},
//                   the same layout as a CSR with doubled/symmetric edges,
//                   which our workload already provides), then does the SAME
//                   H2D copy + GPU worklist allocation (`d_wl`) main() does
//                   BEFORE its CPUTimer.start(). Also loads the K4
//                   MatchingOrder produced once by fringePreprocess.
//   fc_count()   -- calls `occurrences(mo, d_g, d_wl, SMs, mTpSM)` ONLY --
//                   the artifact's own unmodified dispatcher, which for our
//                   K4-as-triangle-core+fringe motif routes to
//                   `triangleCore()` -> its own `clique<<<>>>` kernel. This
//                   IS fringeCount's own CPUTimer-timed region.
//                   One addition not in the original (documented, not a
//                   kernel change): `total`/`wlsize`/`wlpos`/`maxdeg`
//                   (`static __device__` globals fringeCount.cu itself
//                   zero-initializes ONCE via `= 0` and never resets except
//                   inside its OWN automorphism-count recursive sub-call,
//                   src/fringeCount.cu lines ~2235-2239, and inside
//                   edgeCore/vertexCore before their own kernel launches --
//                   triangleCore/generalCore do NOT reset them, relying on
//                   the process-lifetime static initializer since the
//                   artifact's CLI only ever calls occurrences() once per
//                   process) are explicitly reset to 0 before every call,
//                   using the EXACT same 4-symbol reset idiom the artifact's
//                   own automorphism sub-call already uses at another call
//                   site -- necessary because our harness reuses one handle
//                   across warmup+reps, unlike the artifact's single-shot
//                   CLI, which only ever needs one clean run.
//   fc_free()    -- releases everything.
//
// Zero lines of source/ were modified. fringeCount.cu is #include'd
// directly (its core-search functions, e.g. `occurrences`/`triangleCore`,
// have `static` linkage -- internal to the translation unit -- so this is
// the only way to call them without patching their linkage; fringeCount.cu's
// own unused `main()` is compiled into this shared library too, harmlessly).

#include <cstring>
#include <cstdint>

#define main fringecount_cli_main_unused // avoid any ambiguity about which
                                          // main a -shared build exports;
                                          // fringeCount.cu's own main() is
                                          // never called by this shim.
#include "fringeCount.cu"
#undef main

struct FCHandle {
  ECLgraph d_g;
  int2 *d_wl;
  MatchingOrder mo;
  int SMs;
  int mTpSM;
};

extern "C" {

void *fc_prepare(int n, long long nnz, const int *row_ptr, const int *col_idx,
                  const unsigned char *mo_bytes, int mo_bytes_len) {
  if ((size_t)mo_bytes_len != sizeof(MatchingOrder)) return nullptr;
  MatchingOrder mo;
  std::memcpy(&mo, mo_bytes, sizeof(MatchingOrder));

  ECLgraph g;
  g.nodes = n;
  g.edges = (int)nnz;
  g.nindex = new int[n + 1];
  g.nlist = new int[nnz];
  g.eweight = nullptr;
  for (int i = 0; i <= n; ++i) g.nindex[i] = row_ptr[i];
  for (long long i = 0; i < nnz; ++i) g.nlist[i] = col_idx[i];

  ECLgraph d_g = g;
  cudaMalloc((void **)&d_g.nindex, sizeof(int) * (size_t)(g.nodes + 1));
  cudaMalloc((void **)&d_g.nlist, sizeof(int) * (size_t)g.edges);
  cudaMemcpy(d_g.nindex, g.nindex, sizeof(int) * (size_t)(g.nodes + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(d_g.nlist, g.nlist, sizeof(int) * (size_t)g.edges, cudaMemcpyHostToDevice);
  delete[] g.nindex;
  delete[] g.nlist;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  int SMs = deviceProp.multiProcessorCount;
  int mTpSM = deviceProp.maxThreadsPerMultiProcessor;

  int2 *d_wl;
  size_t wl_bytes = std::max(sizeof(int2) * (size_t)g.edges, sizeof(int) * (size_t)g.nodes);
  cudaMalloc((void **)&d_wl, wl_bytes);

  FCHandle *h = new FCHandle{d_g, d_wl, mo, SMs, mTpSM};
  return h;
}

unsigned long long fc_count(void *handle) {
  FCHandle *h = static_cast<FCHandle *>(handle);
  // reset accumulator/worklist globals before every call -- see file header
  ull zero_u = 0;
  int zero_i = 0;
  cudaMemcpyToSymbol(total, &zero_u, sizeof(total));
  cudaMemcpyToSymbol(wlsize, &zero_i, sizeof(wlsize));
  cudaMemcpyToSymbol(wlpos, &zero_i, sizeof(wlpos));
  cudaMemcpyToSymbol(maxdeg, &zero_i, sizeof(maxdeg));

  ull count = occurrences(h->mo, h->d_g, h->d_wl, h->SMs, h->mTpSM);
  cudaDeviceSynchronize();
  return (unsigned long long)count;
}

void fc_free(void *handle) {
  FCHandle *h = static_cast<FCHandle *>(handle);
  if (h) {
    cudaFree(h->d_g.nindex);
    cudaFree(h->d_g.nlist);
    cudaFree(h->d_wl);
    delete h;
  }
}

} // extern "C"
