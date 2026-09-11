# Sparse Tensor Contraction — Evaluation Methodology Survey

Track input: `data/track_inputs/sparse-tensor-contraction.json` (3 papers, all 3 surveyed via
fulltext and/or artifact code — 2 via full paper PDF, 1 via arXiv fulltext + repo code
inspection).

For cross-track consistency this survey was written after reading
`benchspecs/tensor-contraction/spec.yaml` (general einsum/tensor-network contraction track)
and `benchspecs/mttkrp/spec.yaml` (sparse-tensor x dense-factor contraction track). The
FROSTT tensor subset below is deliberately the same named subset used by the mttkrp track,
since this track's one FROSTT-based paper (parTC) uses tensors from that same corpus
(`nips.tns` is parTC's own README example and is also nips in the mttkrp subset).

---

## Insum (ASPLOS 2026 / arXiv:2510.17505, "Sparse GPU Kernels Simplified and Optimized with
Indirect Einsums")

- **workloads/inputs**: 4 sparse-GPU application case studies, each with its own input set
  (no shared cross-application suite):
  - Structured SpMM: synthetic 4096x4096 sparse matrix, 90% uniform sparsity, 32x32 dense
    blocks, dense operand N unspecified in accessible text (ablation only)
  - Unstructured SpMM: real-world matrices from the "TC-GNN datasets" (not individually
    named in the accessible 6-page workshop-length text), output dimension N=128
  - Point Cloud Sparse Convolution: 7 real indoor point clouds from S3DIS (Area 6), 5cm
    voxel quantization; headline single-input comparison table uses the `conferenceRoom`
    scene, fp16, channel size 128
  - Equivariant Tensor Product: batch size 10,000, real Clebsch-Gordan coefficients, l_max
    swept as a hyperparameter
- **timing protocol**: NOT explicitly stated. No warmup-iteration count, repetition count, or
  summary statistic (mean/median/min) is given anywhere in the accessible text (this is a
  6-page paper — likely a PLDI-Sparse'25 workshop precursor of the full ASPLOS 2026 paper;
  the full paper may contain this detail but was not accessible). Table 2's single-number
  "Runtime (ms)" figures carry no stated statistic.
- **timing scope**: kernel runtime is reported separately from (a) compile+autotune time
  (9.9s compile + 4.9s autotune = 14.8s total, one-time, explicitly said to be "easily
  amortized" over repeated inference calls) and (b) format-conversion time (0.55ms,
  compared against TACO's 0.47ms and SparseTIR's 13.47ms CPU-side conversion). This
  three-way split (compile / format-convert / kernel) is the most explicit preprocessing
  breakdown in this track's 3 papers.
- **precision & correctness**: fp32 and fp16 (Tensor-Core path) are both used across the 4
  case studies; Table 2's headline point-cloud number is fp16. No correctness-validation
  method (reference implementation, tolerance) is stated in the accessible text.
- **metric**: primarily speedup vs. a named hand-written/compiler baseline (Table 1: 1.14x
  to 3.81x across the 4 applications) plus lines-of-code reduction (202x-4491x); Table 2
  reports absolute ms for one specific input/config. No GFLOP/s number is reported anywhere
  in the accessible text — this is a speedup-and-LoC paper, not a throughput paper.
- **baselines**: TorchBSR (202 LoC, structured SpMM), Sputnik (1918 LoC, unstructured SpMM),
  e3nn (225 LoC, equivariant tensor product), TorchSparse (4491 LoC, sparse conv), cuSPARSE
  (mentioned as a comparison point for unstructured SpMM), TACO and SparseTIR (compared
  specifically on compile/autotune/format-convert/runtime cost for point-cloud conv, Table 2).
- **source**: arXiv fulltext (`arxiv.org/abs/2510.17505`, full paper PDF read directly —
  6 pages, matches the PLDI 2025 "Sparse" workshop listing of the same title); repo
  `github.com/nullplay/IndirectEinsum` was also inspected but contains only a minimal
  `indirecteinsum.py` compiler example and a single `blockspmm.py`/`example.py`, not the
  paper's actual benchmark harness — the evaluation numbers above come from the paper PDF,
  not from runnable repo code.

---

## parTC / SpGETT (TPDS 2025, "Efficient Parallel Sparse Tensor Contraction", Singh & Uçar)

- **workloads/inputs**: real-life sparse tensors from FROSTT (Formidable Repository of Open
  Sparse Tensors and Tools) — the repo's own worked README example uses `nips.tns` (the
  2.5K x 2.9K x 14K x 17, 3.1M-nnz NIPS tensor, also in this project's mttkrp-track FROSTT
  subset). The second contraction operand is NOT an independent second tensor: the code
  (`sparse_tensor_generator.hpp`) synthetically derives tensor B from A by copying A's
  nonzero list with a `sparsification_factor` (set to 1, i.e. B has the same number of
  nonzeros as A) and a shuffled/copy ordering — this is a self-contraction-style synthetic
  benchmark, not a benchmark over independently-sparse tensor pairs. The exact list of
  FROSTT tensors used in the paper's own Table of results (beyond the one README example)
  was not visible from the accessible sources (paper PDF blocked by Anubis bot-protection on
  hal.science for both the `hal-04659658` and `hal-05047235` HAL deposit IDs; only the
  abstract, via a search-engine indexed excerpt, was retrievable).
- **timing protocol**: the driver (`main.cpp`) takes a `numTests`/"number of rounds" CLI
  argument, but a full-repo grep of every `.cpp`/`.hpp` file shows this argument is parsed
  and stored but **never used anywhere else in the code** — there is no in-binary
  repetition loop, no warmup call, and no averaging/statistic computed by the program
  itself. Each invocation performs exactly one measured run using
  `std::chrono::high_resolution_clock`, printed at nanosecond resolution
  (`fksCuckooTC.hpp`, `fksCuckoo.hpp`). If the paper itself repeats runs and averages, that
  averaging happens in an external (not-in-repo) driver script that was not accessible.
- **timing scope**: the code explicitly separates two phases with separate printed timers:
  "Total time for insertion" (hash-table construction/preprocessing, `fksCuckoo.hpp`) and
  "Total time for tensor contraction" (`fksCuckooTC.hpp`) — preprocessing is never silently
  folded into the contraction number in the reference implementation.
- **precision & correctness**: values are `double` (fp64) throughout (`Tensor::val_array` is
  `double*`). A correctness-check block exists in `main.cpp` (sorts and elementwise-compares
  the contraction output against a second matricized-SpGEMM-based reference implementation)
  but is wrapped in `#if 0` / `#endif` — **disabled by default** in the shipped code, so a
  literal build of this artifact does not validate correctness unless a user manually
  re-enables the block.
- **metric**: wall-clock time (seconds); the abstract states aggregate results as percentage
  reductions vs. baselines ("~21-25%"), i.e. speedup-vs-baseline rather than an absolute
  GFLOP/s throughput figure. `main.cpp` also prints "Total number of nonzeros in the output
  tensor", confirming the contraction output is itself a sparse tensor with a data-dependent
  (not a priori known) nonzero count — the code allocates `Tensor O` from the hash-based
  contraction routine without a visibly inspectable two-pass symbolic/numeric split from the
  accessible headers, so whether output-array sizing is pre-counted (two-pass) or grown
  dynamically was not confirmed.
- **baselines**: per the abstract, "current state-of-the-art SpGEMM-based approaches,
  existing SpGETT approaches, and a carefully implemented SpGETT approach with a new
  fine-tuned [hashing method]" — specific system names beyond this were not visible in the
  accessible abstract text.
- **source**: repo `github.com/ssomesh/parTC` — `README.md`, `main.cpp`,
  `sparse_tensor_generator.hpp`, `fksCuckoo.hpp`, `fksCuckooTC.hpp` read directly via
  `gh api .../contents/<path>`; paper abstract via web search snippet (HAL deposit IDs
  `hal-04659658` and `hal-05047235`, both blocked by Anubis anti-bot protection for direct
  fulltext fetch, including the direct-file URL `hal.science/hal-05047235/file/spgettHAL.pdf`).

---

## SparseLNR (ICS 2022, arXiv:2205.11622, "SparseLNR: accelerating sparse tensor
computations using loop nest restructuring")

- **workloads/inputs**: 14 sparse 2D matrices (SuiteSparse / Network Repository: `cora`,
  `bcsstk17`, `pdb1HYS`, `circuit5M`, `webbase-1M`, `rma10`, `cant`, `consph`, `cop20k_A`,
  `shipsec1`, `scircuit`, `mac_econ_fwd500`, `amazon`, plus one more not separately named in
  the accessible summary) spanning ~5.4K to 59.5M nnz, and 5 sparse 3-mode tensors from
  FROSTT/DARPA (`flickr-3d` 112.9M nnz, `nell-2` 76.9M nnz, `nell-1` 143.6M nnz,
  `vast-2015-mc1-3d`, `darpa1998`) — the FROSTT tensors overlap with this project's
  mttkrp-track subset (`flickr-4d`/`nell-2`/`nell-1` share the same underlying corpus).
  Dense operands are randomly generated with small inner dimensions (k, l in {32, 64, 128}
  except in a dedicated sensitivity sweep).
- **timing protocol**: NOT explicitly stated in the accessible fulltext — no warmup count,
  repetition count, or statistic (mean/median/min) is given; no timer mechanism (wall clock
  vs. CPU-cycle counter) is named. Compiled with `gcc 7.5.0 -O3 -ffast-math`.
- **timing scope**: implied kernel-only (the paper separately discusses TACO code-generation
  as a compile-time step, distinct from the measured execution numbers), but this is not
  stated with the explicit rigor of e.g. Insum's compile/convert/kernel three-way split.
- **precision & correctness**: not discussed in the accessible fulltext — no fp32/fp64
  statement, no numerical-error/tolerance validation method against a reference is
  described.
- **metric**: speedup ratios and raw time comparisons across 6 composite fused-kernel
  expressions (⟨SDDMM,SpMM⟩, ⟨SpMMH,GEMM⟩, ⟨SpMM,GEMM⟩, ⟨SDDMM,SpMM,GEMM⟩ [3-way fusion],
  ⟨MTTKRP,GEMM⟩, ⟨SpTTM,SpTTM⟩) plus a thread-count scaling study (Fig. 8); no GFLOP/s
  figure is reported.
- **baselines**: two TACO configurations only — "TACO Original" (single perfectly-nested
  loop for the whole fused expression, higher asymptotic complexity) and "TACO Separate"
  (manual decomposition into per-kernel temporaries; **the paper explicitly states "when
  multiple decomposition options existed, we evaluate all combinations and report best
  execution time"** — an explicit best-of-N practice for the baseline). No third-party
  compiler (StripeMine, COGENT, etc.) is compared. The paper also discloses that "in
  multithreaded execution, TACO does not generate code when output is sparse format.
  Following prior work, we manually added multithreading" for its own multithreaded
  numbers — a hand-patched-baseline caveat relevant to any reproduction with sparse-output
  kernels.
- **hardware**: single-socket 64-core AMD Ryzen Threadripper 3990X @ 2.2GHz, OpenMP 4.5,
  64 threads (all cores) for the multithreaded numbers; single-threaded numbers also
  reported for the base comparisons.
- **source**: `ar5iv.labs.arxiv.org/html/2205.11622` (arXiv fulltext, HTML-rendered);
  repo `github.com/adhithadias/SparseLNR` (a fork of TACO — `README.md`/directory listing
  inspected; it is the full TACO codebase plus SparseLNR's scheduling-directive extension,
  not a small standalone artifact, so the paper's own numeric tables were the primary
  source, not a benchmark-script read).

---

## Divergences

- **Kernel shape is not shared across the 3 papers.** Insum evaluates GPU sparse-*dense*
  kernels (SpMM, sparse convolution, equivariant tensor product) expressed as an "indirect
  einsum" over fixed-length formats (GroupCOO/BlockGroupCOO); parTC evaluates true
  sparse-tensor x sparse-tensor contraction (SpGETT) on FROSTT N-mode tensors, CPU-only;
  SparseLNR evaluates *composite, fused* multi-kernel sparse-tensor-algebra expressions
  (chains like ⟨SDDMM,SpMM,GEMM⟩) built from TACO scheduling directives, CPU-only, over
  both 2D matrices and FROSTT tensors. No single "sparse tensor contraction" input suite or
  reference kernel spans all 3 — this is the same structural situation the sibling
  `tensor-contraction` track spec documents for its 4 shape families, and this spec follows
  the same one-variant-per-family resolution rather than forcing an artificial common suite.
- **Timing rigor is the weakest of any track surveyed so far.** None of the 3 papers state
  an explicit warmup-iteration count, repetition count, or summary statistic in their
  accessible text. parTC's own repo code goes further: it accepts a repetition-count CLI
  argument that is silently unused, so the reference implementation is a bare single-shot
  timer with zero warmup. This spec fixes warmup/reps/statistic explicitly for all variants
  rather than inheriting any paper's (non-existent) protocol.
- **Correctness validation is disabled-by-default or unstated in all 3 papers' accessible
  material.** parTC ships a correctness-check block wrapped in `#if 0` (present in the code,
  disabled by default); Insum and SparseLNR state no numerical tolerance or reference
  implementation at all in the accessible text. This spec adds an explicit tolerance to
  every variant rather than leaving correctness unchecked before timing counts, per the
  track-wide fairness principle used in the sibling tensor-contraction/mttkrp specs.
  and per the project's fairness principles.
- **Output sparsity/allocation strategy is unclear in all 3 papers.** parTC's driver prints
  a data-dependent output-nnz count ("Total number of nonzeros in the output tensor"),
  confirming SpGETT's output is itself sparse with a size unknown until computed, but the
  accessible headers do not clearly show whether output storage is two-pass
  (symbolic-count-then-numeric-fill, as in classic SpGEMM) or grown dynamically during the
  hash-join. SparseLNR's own text flags a related fairness issue: TACO does not
  multithread-codegen when the *output* is itself sparse-format, requiring a hand-patched
  baseline. Insum's SpMM/convolution kernels instead produce dense (or a fixed
  pre-allocated fixed-length-format) output, sidestepping the issue by format choice. This
  spec requires the output-allocation strategy (two-pass vs. over-allocate-and-compact vs.
  dynamic growth) to be stated explicitly and its cost reported, since none of the 3 papers
  do so.
- **Flop accounting has no data-independent formula for the sparse-sparse case.** Unlike
  SpMM (2*nnz*N) or MTTKRP (2*R*nnz), the flop count for sparse-tensor x sparse-tensor
  contraction (parTC's SpGETT) is intrinsically data-dependent: it equals twice the number
  of *matched* nonzero-coordinate pairs actually found during the hash-join/intersection,
  which cannot be predicted from nnz(A) and nnz(B) alone without knowing the sparsity
  pattern overlap. This spec requires the matched-pair count to be measured and reported
  alongside GFLOP/s, not estimated from a closed-form input-size formula (same conservative
  principle the sibling tensor-contraction track applies to its quantum-circuit variant's
  flop counting).
- **Precision differs by paper's target hardware, not by a stated design choice.** parTC
  (CPU) and SparseLNR (CPU) do not name a hardware-driven precision restriction (SparseLNR:
  unstated entirely; parTC: fp64 always). Insum (GPU, Tensor-Core-targeted) uses fp16 for
  its Tensor-Core-fused headline numbers and fp32 elsewhere — this is a genuine
  Tensor-Core-availability-driven divergence, not an oversight, and the spec keeps fp32 as
  default with fp16 as a labeled Tensor-Core-path secondary result, mirroring how the
  sibling tensor-contraction track treats FastKron's fp32/fp64 split.
