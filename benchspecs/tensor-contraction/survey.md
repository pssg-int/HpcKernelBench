# Tensor-contraction track — evaluation-methodology survey

Surveyed 4/4 papers, all via fulltext (2 via arXiv PDF; 1 via ACM-poster +
GitHub source inspection because the paper itself is ACM-paywalled with no
arXiv copy; 1 via NSF PAR open-access mirror since the paper has no arXiv
copy).

## Einsum Trees — Breuer, Blacher, Engel, Giesen, Heinecke, Klaus, Remke,
"Einsum Trees: An Abstraction for Optimizing the Execution of Tensor
Expressions" (ASPLOS'25)

Sources: ASPLOS'25 poster PDF (`scalable.uni-jena.de`, full evaluation
figures + captions) + the `scalable-analyses/einsum_ir` GitHub repo
(`src/bench_expression.cpp` benchmark driver, `samples/tccg/*`,
`samples/einsum_benchmark/*`). Paper itself is ACM-paywalled with no arXiv
mirror found.

- **Workloads**: two families.
  1. **TCCG benchmark** (binary tensor contractions, Springer & Bientinesi
     2018) in a "modified/adjusted" configuration — the repo's
     `samples/tccg/onednn_adjusted.txt` lists ~24 explicit contraction cases
     as (einsum string, dimension sizes, contraction path), e.g.
     `"efbad,cf->abcde" "48,36,24,36,48,36" "(0,1)"`, ranging from small
     2-operand contractions up to `"ac,cb->ab" "7248,7240,7248"`.
  2. **Multi-tensor contraction trees** from real tensor-network/tensorized-
     ML applications, evaluated as complete trees (not decomposed into
     pairwise TCCG cases): SYN (synthetic tree, Fig.1 of the paper), TT
     (tensor-train decomposition), FCTN (fully-connected tensor-network
     decomposition), TW (tensor-wheel decomposition), GETD (tensor ring of
     a generalized Tucker model), TRN (first layer of a tensor-ring
     network), MERA (multi-scale entanglement renormalization ansatz), and
     TNLM (tensor-network language-model inference query). Sample configs
     for some of these (`lm_first_last_brackets_4_16d.cfg`,
     `str_nw_mera_open_26.cfg`, `synthetic.cfg`) are checked into the repo
     with explicit einsum strings/paths and per-dimension sizes (e.g. the
     MERA config lists 45 explicit per-dimension sizes ranging 3-79).
- **Precision**: figures are explicitly labeled "**FP32 GFLOPS**"; the
  benchmark driver (`bench_expression.cpp`) supports FP32/FP64/CPX_FP32/
  CPX_FP64 as a runtime flag (default FP32), so the reported headline
  numbers use fp32 even though fp64 is implemented and available.
- **Permutation/transpose cost**: **included in the timed kernel, by
  design** — the paper's central contribution is the Einsum Tree IR's
  `swap` / `reorder-index-string` / `insert-permutation-node`
  transformations that minimize permutations; permutation nodes are
  first-class IR nodes that get lowered and executed as part of the same
  timed contraction, not stripped out or amortized separately. All
  baselines (ATen, opt_einsum+PyTorch backend "OE-Torch", TVM-Ansor,
  TBLIS backend) likewise perform any needed transpose/permute inside
  their own timed call.
- **Flop accounting**: the driver counts flops via `l_einsum_exp.num_ops()`
  (an internal op counter over the compiled contraction plan), then
  `GFLOPS = 1e-9 * num_flops / time`. Two GFLOPS numbers are computed per
  run: `gflops (eval)` uses only the timed evaluation call; `gflops
  (total)` divides by `time_compile + time_eval` (i.e., includes the
  one-time contraction-path-lowering/compile cost in the denominator once,
  not amortized over multiple calls).
- **Timing protocol**: from `bench_expression.cpp` — **exactly one warmup
  `eval()` call, then exactly one timed `eval()` call**, measured with
  `std::chrono::steady_clock`. This is a **single-shot** measurement, not a
  median/min-of-many statistic — a real methodology weakness relative to
  the fairness principles in these instructions (see Divergences).
- **Baselines**: ET-TPP (their own IR lowered to LIBXSMM Tensor Processing
  Primitives), ET-TBLIS (their own IR lowered to the TBLIS tensor-
  contraction library), TVM-Ansor, ATen (native PyTorch einsum path),
  OE-Torch (opt_einsum path-finder with PyTorch backend execution).
- **Hardware**: CPU-only evaluation — NVIDIA Grace CPU (72 ARM Neoverse V2
  cores) and Intel Xeon Platinum 8488C (48 cores); no GPU results reported
  for this paper despite the platform metadata listing `arm-cpu`/`cpu`.
- **Correctness**: no explicit numerical-tolerance statement found in the
  accessible poster/source; the CLI exposes a `store_lock` flag ("if 1 all
  einsum_ir input tensors are stored and locked before evaluation") that
  hints at internal validation infrastructure not surfaced in the fetched
  materials.
- **Metric**: GFLOP/s (primary, both eval-only and total-incl.-compile
  variants).

## cuKE — Hu, Li, Jiang, "cuKE: An Efficient Code Generator for Score
Function Computation in Knowledge Graph Embedding" (IPDPS'24)

Source: NSF PAR open-access PDF mirror (`par.nsf.gov/servlets/purl/
10530462`; paper has no arXiv copy and IEEE Xplore is paywalled).

- **Workloads**: KGE (knowledge-graph-embedding) "score function"
  computations — batched, small, irregular tensor-contraction-like fused
  ops (vector-vector inner product, batched matrix-vector, batched vector
  outer product, scalar-vector) rather than the large dense contractions of
  TCCG/tensor-network work. Evaluated on **4 real knowledge graphs**
  (Table IV, exact sizes given): FB15k (14,951 nodes / 592,213 edges /
  1,345 relations), FB15k-237 (14,541 / 310,116 / 237), BioKG (93,773 /
  5,088,434 / 51), WN18 (40,943 / 151,442 / 18). 5 score functions
  (contraction shapes) benchmarked: **TransE, TransH, TransR, TransF,
  RESCAL** (Table II gives each one's exact tensor-algebra formula; TransR
  and RESCAL involve batched matrix operands, i.e. genuine 3-operand
  contractions, while TransE/TransH/TransF are vector-only).
- **Shape sweep**: batch_size in {4096, 8192} (also 16,384 for a
  memory-only measurement), embedding dimension `dim` in {512, 1024}.
- **Precision**: single-precision (fp32) throughout — sizes computed via
  `sizeof(float)` in the memory-footprint analysis (Table V).
- **Permutation/transpose cost**: the paper's own contribution
  (per-edge redundant-index "runtime inspection") is a **fused,
  in-kernel** dedup/index-remap step, not a separate preprocessing pass —
  it is generated as part of the same CUDA kernel and is included in every
  reported timing. No explicit transpose step exists in this workload
  (contractions are along small embedding/relation dims via
  shared-memory gather, not global transpose).
- **Flop accounting**: not given as an explicit formula; GFLOPS bars in
  Fig. 9/10 are computed from measured global memory traffic (GMT, labeled
  on each bar in MB/GB) and time, implying flops are derived from the
  known score-function arithmetic count per edge × batch size (not stated
  precisely in the fetched text — open question).
- **Timing protocol**: "The execution time was measured as the **average of
  3000 training iterations**." No explicit separate warmup phase is
  described (early iterations are implicitly part of the 3000-iteration
  average, effectively acting as their own mild warmup dilution — a
  fairness-relevant point since it's not the same as fixed warmup + median
  over separately-repeated timed runs).
- **Baselines**: PyTorch/TorchScript (DGL-KE's native cuBLAS-based
  implementation — the better of compiled/uncompiled reported), TVM (Relay
  DSL), and a **hand-optimized CUDA baseline** manually fused the same way
  as cuKE's compiler output but *without* the runtime-inspection dedup
  optimization (isolates the value of the paper's specific contribution).
- **Hardware**: single NVIDIA RTX3090 (82 SMs, 1.4GHz, 24GB, 936GB/s BW),
  NVCC 11.7.
- **Correctness**: not explicitly stated as a numeric tolerance in the
  fetched text (implicit correctness is argued via matching DGL-KE's
  training loss / end-to-end accuracy over 1.3-2.3x-faster KGE training,
  Section VII-C, rather than a per-kernel tolerance check before timing).
- **Metric**: GFLOP/s, global memory traffic (MB/GB, always labeled
  alongside GFLOPS bars so memory-vs-compute tradeoffs are visible),
  execution time (ms) per training iteration, memory usage (GB, Table V).

## FastKron — Jangda, Yadav, "Fast Kronecker Matrix-Matrix Multiplication
on GPUs" (PPoPP'24), arXiv:2401.10187

Source: arXiv fulltext, Sections 6 (Evaluation) + Artifact Appendix A.

- **Workloads**: Kronecker Matrix-Matrix Multiplication (Kron-Matmul),
  X_{M x P^N} times N Kronecker factors F_{P x Q} — a structurally distinct
  contraction shape from general TCCG-style einsum contractions (it is a
  chain of N identical small-factor "sliced multiplies", not an arbitrary
  binary-contraction tree). Two workload sets:
  1. **Microbenchmarks**: M=1024, P (factor size) a power of two from 8 to
     128, N (number of factors) swept to the two largest allocatable
     values of P^N (Fig. 9); also M=16 with float/double compared (Table 3).
  2. **28 real-world Kron-Matmul sizes** (Table 4, exact M and factor
     shapes given) drawn from published work in LSTM/RNN compression,
     ML weight compression, hybrid Kronecker-product decomposition
     (HyPA), Kronecker graph models, computational biology, drug-target
     interaction prediction, and Gaussian-Process kernel interpolation
     (SKI/SKIP/LOVE) — N ranges from 2 to 11 factors, M values are odd
     and non-power-of-two in several cases.
- **Precision**: **both float and double** explicitly compared (Table 3:
  achieved TFLOPS for float vs. double at M=16, largest P^N, across all 4
  systems compared).
- **Permutation/transpose cost**: **explicitly included in the reported
  time for baselines, and explicitly absent by design for FastKron.** The
  shuffle-algorithm baseline (GPyTorch/PyKronecker)'s transpose step is
  measured to take **up to 80% of total execution time** (Table 1 breaks
  out GPyTorch's Matmul-only vs. Trans.-only vs. Total time explicitly).
  FTMMT-based baselines (COGENT, cuTensor) fuse transpose internally.
  FastKron's algorithm mathematically avoids transpose/reshape altogether
  (its core contribution), so its reported time has zero transpose
  component by construction, not by exclusion from timing.
- **Flop accounting**: algorithm-level complexity given explicitly as
  O(MP * sum_{i=1}^{N} Q^{N-i} P^i) computations; the achieved-TFLOPS
  metric is derived from this operation count divided by measured time
  (exact per-run flop-counting code not shown in the fetched text, but the
  complexity formula is explicit, unlike cuKE/SWQsim).
- **Timing protocol**: "We use CUDA 12.2 on Ubuntu 22.04 and report the
  **average TFLOPS of 100 runs after a warmup of 10 runs**." This is the
  most precisely and completely specified timing protocol of the four
  papers in this track (explicit warmup count AND explicit repetition
  count), though it uses **average**, not median.
- **Baselines**: single-GPU — GPyTorch (shuffle algorithm via cuBLAS),
  COGENT (FTMMT via generated tensor-contraction code), cuTensor (FTMMT,
  NVIDIA's tensor-contraction library, used as the strongest FTMMT
  baseline since it matches manually-tuned CUTLASS); multi-GPU — CTF
  (Cyclops Tensor Framework, shuffle via distributed matmul/transpose),
  DISTAL (FTMMT via a user-specified distributed schedule).
- **Hardware**: single NVIDIA DGX-2 node, 16x Tesla V100 (32GB, NVLink 2),
  dual 24-core Intel Xeon host CPUs.
- **Correctness**: not explicitly described as a numeric tolerance in the
  fetched text (open question — likely present in the artifact's test
  suite, `tests/run-tests.py`, but not verified here).
- **Metric**: achieved TFLOPS (primary), speedup vs. baseline, execution
  time (ms, Table 1 breakdown), shared-memory load/store transaction
  counts (Table 2, a microarchitectural secondary metric), end-to-end
  Gaussian-Process training-time speedup (case study, Table 5).

## SWQsim — Liu et al., "Closing the 'Quantum Supremacy' Gap: Achieving
Real-Time Simulation of a Random Quantum Circuit Using a New Sunway
Supercomputer" (SC'21), arXiv:2110.14502

Source: arXiv fulltext (includes SC'21 paper + an appendix added post-
publication), Sections 6 (Performance Results) + Appendix A.

- **Workloads**: tensor-network contraction for random-quantum-circuit
  (RQC) simulation — structurally the most extreme-scale, most irregular
  contraction shape in this track. Two named target circuits: the paper's
  own **10x10x(1+40+1) qubit circuit** (their new milestone, contracted via
  a custom PEPS-based slicing scheme with near-uniform rank~5-6 /
  dimension-32 pairwise contractions) and **Google Sycamore** (53-qubit,
  20-cycle circuit, contracted via a CoTenGra-searched path that yields
  highly **imbalanced** rank-30-vs-rank-4, dimension-2 pairwise
  contractions). A third circuit, 20x20x(1+16+1), is used for strong-
  scaling and validation experiments only.
- **Precision**: **single-precision (FP32) and mixed-precision
  (FP32/FP16, adaptive per-tensor scaling)** — no fp64 is used anywhere in
  this paper (explicit design choice for a memory-bandwidth-bound
  workload at exascale). This is the only paper across both tracks that
  never uses fp64.
- **Permutation/transpose cost**: **the paper's headline architectural
  contribution is fusing permutation with multiplication** ("a fused
  permutation and multiplication design that improves the compute
  efficiency for a wide range of tensor contraction scenarios" — abstract).
  Two distinct fused kernels are used and BOTH fully include permutation
  cost in the reported performance: (1) a CPE-array cooperative
  diagonal-broadcast scheme for the compute-dense PEPS contractions
  (Section 5.4, Fig. 8), and (2) a TTGT (Transpose-Transpose-GEMM-
  Transpose) fused kernel for the memory-bound, rank-imbalanced Sycamore
  contractions (Fig. 9). Permutation is never excluded or separately
  amortized — it is the thing being optimized.
- **Flop accounting**: explicitly stated methodology (Section 6.1) — flops
  are measured **two ways**: (1) counting all floating-point arithmetic
  instructions needed for permutation+multiplication, and (2) hardware
  floating-point-operation counters. Hardware counters read **10-20%
  higher** than the instruction count "due to the generation of temporary
  floating-point operations along the way," so the paper explicitly
  chooses the **lower, counted number as "a conservative and fair Flop
  measurement."** This is the most explicit, self-critical flop-accounting
  discussion found in either track.
- **Timing protocol**: "The performance is measured using the **average
  time recorded for running the same case for three times**." Only 3
  repetitions, no stated warmup — the weakest repetition count of any
  paper surveyed across both tracks, though understandable given the
  extreme cost of each run (up to ~42M cores, minutes-to-hours per run at
  the largest scale).
  Metric reported: sustained PFLOPS/EFLOPS at 6 strong-scaling points from
  199,680 to 41,932,800 cores (Fig. 13).
- **Correctness**: two distinct validation methods, neither a simple
  fixed-tolerance numeric check against a reference tensor: (1)
  statistical — computed-amplitude-probability distribution compared
  against the theoretical Porter-Thomas distribution for both single- and
  mixed-precision (Fig. 11); (2) convergence — mixed-precision relative
  error vs. single-precision baseline, shown to converge to <1% once
  ~300 blocks (27,000 contraction paths) are aggregated (Fig. 10), with an
  explicit note that <2% of mixed-precision runs are discarded due to
  under/overflow.
- **Baselines**: not a same-machine kernel-level comparison but a
  literature comparison of achieved sustained performance and Sycamore
  sampling time against prior classical RQC simulators — qFlex/Summit
  (281 Pflops for a smaller 7x7x(1+40+1) circuit), a Summit hard-disk
  approach (Pednault et al., 2.55 days estimated), an AliCloud estimate
  (19.3 days), and a 60-GPU approach (Pan & Zhang, 5 days) — summarized in
  the paper's own Table 1.
- **Hardware**: new-generation Sunway supercomputer, up to 107,520 nodes /
  41,932,800 SW26010P cores.
- **Metric**: sustained PFLOPS/EFLOPS (primary), memory bandwidth
  utilization (GB/s vs. peak, Fig. 12), wall-clock time-to-solution for
  the Sycamore sampling task (304s headline number).

## Divergences

- **Contraction-shape suite: four genuinely different problem families,
  zero overlap.** Einsum Trees benchmarks general dense binary
  contractions (TCCG) plus named multi-operand tensor-network application
  trees; cuKE benchmarks small, batched, irregular KGE score-function
  fused ops; FastKron benchmarks the structurally distinct Kronecker-
  product-chain shape; SWQsim benchmarks extreme-scale, highly imbalanced
  quantum-circuit tensor-network contraction. No single benchmark corpus
  spans all four. Unlike the MTTKRP track (where a real gap exists but a
  plausible "general vs. symmetric" split covers it), this track's four
  papers are near-orthogonal in what they mean by "tensor contraction."
  The spec below therefore defines **4 variants, one per shape family**,
  rather than trying to force a single unified suite — collapsing them
  would hide, not resolve, the divergence.
- **Permutation/transpose-cost inclusion is actually the one point of
  strong agreement.** All four papers include permutation/transpose cost
  inside the timed kernel for every system compared (Einsum Trees'
  permutation-node IR, FastKron's explicit Table-1 transpose breakdown for
  baselines even though FastKron itself has none, SWQsim's headline "fused
  permutation and multiplication" contribution, cuKE's fused runtime-
  inspection). No paper amortizes or excludes transpose separately. The
  spec keeps this as a firm rule for all four variants: transpose/
  permutation is always in-kernel, never a separate preprocessing step.
- **Precision: fp32-only (cuKE, SWQsim, Einsum Trees' headline figures)
  vs. fp32-and-fp64-both (FastKron).** SWQsim never uses fp64 at all
  (explicit bandwidth-driven design choice); Einsum Trees' evaluation
  figures are FP32-only even though the implementation supports FP64;
  cuKE is FP32-only; only FastKron directly compares FP32 vs FP64 in the
  same table. The spec makes fp32 the default per-variant precision
  (matching 3 of 4 papers) and requires fp64 to be reported as a labeled
  secondary result only where the underlying system supports it (Einsum
  Trees, FastKron), rather than assuming fp64 is universally available
  the way the MTTKRP track's papers mostly do.
- **Flop-accounting rigor varies enormously.** SWQsim is the most rigorous
  and self-critical (explicitly compares 2 counting methods, picks the
  conservative one, and explains why). FastKron gives an explicit
  algorithm-level complexity formula. Einsum Trees uses an internal
  `num_ops()` counter without describing its exact definition in the
  accessible materials. cuKE gives no explicit flop formula at all (GFLOPS
  bars are backed out from memory-traffic figures, not a stated flop
  count). The spec requires each variant to state its flop formula
  explicitly (following SWQsim's model) rather than accepting an opaque
  internal counter.
- **Timing rigor ranges from single-shot to well-specified.** Einsum
  Trees: literally 1 warmup + 1 timed run (no repetition statistic at
  all). SWQsim: average of only 3 runs. cuKE: average of 3000 *training*
  iterations (not repeated measurements of one isolated kernel call).
  FastKron: the only paper with an explicit, standard warmup=10/reps=100
  protocol. The spec adopts FastKron's protocol shape (explicit warmup +
  reps + statistic) as the fixed default across all four variants, since
  it is both the most rigorous and the most standard-looking of the four,
  while flagging Einsum Trees' single-shot measurement as a fairness
  issue this benchmark explicitly does not want to reproduce.
- **Correctness gates are essentially absent from all four papers' visible
  text.** SWQsim is the partial exception, with a domain-specific
  statistical/convergence check (Porter-Thomas fit, <1% mixed-precision
  drift) rather than a fixed numeric tolerance against a reference tensor.
  No paper states a plain "max relative error < X" correctness gate before
  timing counts. The spec adds one explicitly for all variants except the
  quantum-circuit variant, where SWQsim's own statistical/convergence
  checks are adopted instead (a fixed elementwise tolerance is not
  meaningful for stochastically-sampled quantum amplitudes).
