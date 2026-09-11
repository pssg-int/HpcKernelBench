# ecl-scc (ECL-SCC) — connected-components

**Status: BUILT+GATED**

- Paper: "A GPU Algorithm for Detecting Strongly Connected Components"
  (SC'23). `PAPER_KEY = conf/sc/AlabandiSBB23` (matched by title +
  artifact_url in `../../../output/included.json`).
- Artifact: https://github.com/burtscher/ECL-SCC
- Commit cloned: `8e67732687d06f75cdd602f36e1fe54429ba4f99`, `git clone --depth 1`.
- Toolchain: `nvcc` 12.9 (`/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda/12.9`),
  host compiler `g++` 14.3.0 (`/opt/cray/pe/gcc-native/14/bin/g++`),
  `-arch=sm_80` (A100), C++17. Python
  `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`.
  `LD_PRELOAD=/usr/lib64/libstdc++.so.6` needed at *run* time (this venv's
  Python links an older libstdc++ than what `nvcc`-compiled code expects;
  same fix as `bench/artifacts/triangle-counting/tot`).

## Track thinness

connected-components is one of two "thin tracks" in this integration pass:
this track's own survey (`benchspecs/connected-components/survey.md`) covers
exactly 4 papers, only 2 of them graph-domain GPU candidates — ECL-SCC (this
artifact, BUILT+GATED) and FAST-BCC-on-GPUs (`../fast-bcc/`, SKIPPED — see
its own STATUS.md: no clean CC-only entry point, its `CC()` function is the
terminal labeling step of the BCC pipeline itself). The 4th paper,
**YACCLAB** (Allegretti, Bolelli, Grana, TPDS'20) is NOT integrated and never
attempted: it performs 2D/3D binary-IMAGE connected-component LABELING on a
regular pixel/voxel grid under a fixed 8-/26-connectivity rule (OpenCV-loaded
image datasets, `journals/tpds/AllegrettiBG20`) — a fundamentally different
input structure from the sparse-graph `Graph`/CSR workload this track's
`connected-components` kernel and every one of its 4 spec variants operate
on (`cc-image-ccl-2d3d` is its own separate spec variant precisely because
of this domain mismatch, per `benchspecs/connected-components/spec.yaml`'s
own `notes_on_fairness`). With both real graph-domain GPU candidates
resolved (1 integrated, 1 SKIPPED) and YACCLAB out of scope by domain, this
track's GPU-candidate list is exhausted.

## What the artifact actually is

`source/source/ECL-SCC_10.cu`: a single-file CUDA implementation of
max-ID/max-out-degree "signature propagation" SCC detection (Alabandi,
Sands, Biros, Burtscher). Four `__global__` kernels
(`globalInit`/`propagateMax`/`removeEdges`/`localInit`) iterate a
fixed-point loop over a 32-bit binary CSR (`ECLgraph`, `source/source/
ECLgraph.h`) until every remaining edge's endpoints share the same
`(inMax, outMax)` signature pair, at which point `iomax[v].x` is the SCC
representative id for vertex `v`. The file's only entry point is a CLI
`main()` that reads a binary `.egr` file and prints an SCC-size histogram —
no reusable library API is exposed.

## The shim (rule 1: wrap the kernel, not the driver)

`ecl_scc_shim.cu` (this directory, **NOT** part of the artifact — same role
as `bench/artifacts/triangle-counting/tot/tot_shim.cu`) `#include`s
`source/source/ECL-SCC_10.cu` **unmodified** and refactors `main()`'s
"copy graph to GPU / run kernels / copy result back" sections into three
reusable `extern "C"` entry points that reuse the artifact's own
`__global__` kernels, `CheckCuda()`, `Device`, and `ThreadsPerBlock` from the
SAME translation unit, byte-for-byte:

- `ecl_scc_prepare(n, nnz, row_ptr, col_idx)` — H2D copy of the (already
  0/1-pattern, self-loop-free) CSR into an `ECLgraph`, plus working-buffer
  allocation (`d_wl1`/`d_wl2`/`d_iomax`/`d_wl2size`/`d_goagain`) — exactly
  `main()`'s own allocation block.
- `ecl_scc_run(handle)` — **exactly** `main()`'s timed region: `globalInit`
  once, then the `propagateMax`/`removeEdges`/`localInit` fixed-point loop.
  Re-runs `globalInit` at the top of every call (the algorithm mutates
  `d_wl1`/`d_iomax` in place, so a second call without this would resume
  from the first call's already-converged, empty-worklist state and do
  nothing — same "reset before each call" requirement
  `InsumSpMM.run()`'s `C.zero_()` documents for a different artifact).
- `ecl_scc_labels(handle, out)` — D2H copy of the converged `iomax[v].x`
  signature array, called once by `adapter.py`'s `to_host()`, never inside
  the timed loop (matches `cc-scc-kernel`'s `timing_scope`, which excludes
  output transfer).

Zero lines of `source/` were modified.

## Mapping: SCC-on-directed-graphs artifact -> this track's CC gate (task C)

`kernelbench.domains.graph` (this harness's own reference/gate module) only
ever produces ONE `Graph` workload per run and gates it against ONE
connectivity notion: strong (Tarjan, for directed input) or weak (union-find,
for already-symmetric input) — never both from the same call, and it
explicitly does not implement full BCC or image-CCL (see its own docstring).
ECL-SCC's actual contribution is strong connectivity on *directed* graphs, a
problem this harness has no independent non-scipy reference for.

The reduction used here (directed by the task instructions): **on a
SYMMETRIZED graph, SCC degenerates to plain connectivity** — if every edge
u->v also has v->u, any directed path implies its reverse, so every weakly-
connected component is trivially strongly connected too, i.e.
`SCC(symmetrize(G)) == CC(symmetrize(G)) == CC(G)` (symmetrizing never
changes which vertices are mutually reachable via undirected paths).
`adapter.py`'s `prepare()` symmetrizes the workload's CSR (union of
forward+reverse edges — the SAME convention `kernelbench.domains.graph.
smoke_workloads()` and `SpgemmTriangleCount.prepare()` already use) and
feeds that symmetrized CSR into ECL-SCC's real, unmodified kernel. The
kernel itself never sees or special-cases the symmetry — it runs the
identical signature-propagation algorithm it would run on any directed
input; this is a genuine reduction, not a relabeling shortcut.

`params["connectivity_mode"]` is forced to `"weak"` so the domain's
`reference_cc()` computes the matching quantity (plain connectivity) instead
of its own directed/undirected auto-detection, and both sides are
canonicalized by the identical "first order of appearance" relabeling
(`_canonical_labels`, duplicated here rather than imported, logically
identical to `kernelbench.domains.graph._canonical_labels`) — so the
"exact, up to label permutation" gate genuinely checks component
MEMBERSHIP, per the connected-components spec's requirement, never loosened.

## Gate verification (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
LD_PRELOAD=/usr/lib64/libstdc++.so.6 $PY -m kernelbench.runner \
    --kernel connected-components --variant cc-scc-kernel \
    --impl ecl-scc-cc-via-symmetrized-scc --smoke
```

Result: **2/2 runs valid**, `err = 0.00e+00` (exact) on both synthetic smoke
graphs (`graph-smoke-uniform` 1500x1500/23926 edges, `graph-smoke-powerlaw`
1500x1500/17968 edges — both already symmetric, `symmetrized_in_prepare:
False`).

Additionally cross-checked on a real cached SuiteSparse graph (`ca-HepPh`,
12008 vertices / 236978 directed entries, symmetrized in `prepare()`
— `symmetrized_in_prepare: True`) against the domain's independent CPU
reference (`kernelbench.domains.graph.reference_cc`, union-find, NOT scipy):

```
$PY -m kernelbench.runner --kernel connected-components \
    --variant cc-scc-kernel --impl ecl-scc-cc-via-symmetrized-scc \
    --matrices ca-HepPh --warmup 1 --reps 3
```
`1/1 runs valid`, `err = 0.00e+00` (exact match).

Reduced-protocol numbers only (warmup=1 or 5, reps=3 or 20, shared
login-node GPU) — explicitly non-conforming (`conforming: False`), not a
timing claim, per ARTIFACT_GUIDE.md rule 5. No timing sweep was run.

## Build-system patches (rule 3: none needed)

None. `nvcc -shared` compiles `ecl_scc_shim.cu` (which `#include`s the
artifact's own `.cu` file) directly with no CMake/Makefile involvement —
the artifact ships no build system of its own for this single-file kernel,
so there was nothing to patch.

## Not done

- No sweep across the spec's SuiteSparse/MFEM-mesh `recommended_subset`
  (cage14, soc-LiveJournal1, wiki-Talk, web-Google, circuit5M, beam-hex/
  toroid-wedge/twist-hex mesh families) — out of scope for the login-node
  budget; a functional/gate check only.
- `cc-scc-kernel`'s `algorithm_family_disclosure` and `graph_regime_
  disclosure` REQUIRED fields are echoed into `params` as
  `algorithm_family` but a full per-graph regime classification (power-law
  vs. structured/mesh) was not run across a matrix sweep.
- The MFEM radiative-transfer mesh graphs (ECL-SCC's own domain-specific
  workload family) were not regenerated — out of scope for this pass.

## Verdict

`ecl-scc-cc-via-symmetrized-scc: BUILT+GATED, exact match on 2 smoke graphs
+ 1 real graph (ca-HepPh) via the SCC(symmetrized G) == CC(G) reduction; no
artifact bugs found; zero lines of the artifact's own source touched.`

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, NVIDIA A100-SXM4-40GB (sm_80, gres `a100_1g.5gb`
  request landed a full card this run), login-node build.
- Toolchain: nvcc 12.8.93 (conda-forge CUDA 12.8), g++ 13.4.0 (conda-forge),
  torch 2.8.0+cu128, Python 3.12.14, no cmake involved (single `nvcc -shared`
  call); `-arch=sm_80`.
- Build: OK. Build-system changes: none (`build.sh` already reads
  `CXX`/`CUDA_HOME` via `${VAR:-<Perlmutter default>}`, overridden cleanly by
  `bench/env.sh`/`toolchain.sh`'s exports).
- Gate: `cc-scc-kernel`/`ecl-scc-cc-via-symmetrized-scc`: PASS, 2/2 runs
  valid, err 0.00e+00 (exact) on both smoke graphs (`graph-smoke-uniform`,
  `graph-smoke-powerlaw`); non-conforming (synthetic smoke, reduced
  protocol), as expected.
- Deviation from the recorded ruling: none.
- Verdict here: BUILT+GATED — equals the recorded ruling.
