# Tensor Network — Evaluation Methodology Survey

Track input: `data/track_inputs/tensor-network.json` (5 papers, all 5 surveyed — 3 via full
paper PDF, 1 via arXiv preprint PDF, 1 via artifact-repo scripts since the paper itself was
inaccessible). As the track's `one_liner`s note, this track is "largely quantum-circuit
simulation on classical HPC" rather than literal tensor-network *contraction* in every case:
4 of the 5 papers (SliQSim, HyQuas, SV-Sim, DM-Sim) are gate-by-gate state-vector or
density-matrix propagation simulators, and only 1 (SWQsim) performs genuine tensor-network
*contraction* (slicing + contraction-path search over a tensor network derived from the
circuit). This structural split is preserved as the main axis of the spec below rather than
papered over. For cross-track consistency this survey was written after reading
`benchspecs/tensor-contraction/spec.yaml`, which already carries a dedicated
`quantum-circuit-contraction-kernel-fp32-mixed` variant using SWQsim as its evidence; this
track's `rqc-tensor-network-contraction` variant is written to be consistent with that
variant's protocol (same flop-counting convention, same permutation-always-included rule)
rather than duplicating it independently.

---

## SWQsim (SC 2021, arXiv:2110.14502, "Closing the 'Quantum Supremacy' Gap")

- **workloads/inputs**: two fixed, published, extreme-scale random-quantum-circuit (RQC)
  tensor networks — (1) a 10x10-qubit 2D lattice, depth (1+40+1) [i.e. 1 initial layer + 40
  cycles + 1 final layer], PEPS-sliced contraction; (2) the Google Sycamore circuit, 53
  qubits, 20 cycles, CoTenGra-searched contraction path. A third circuit, 20x20 qubits depth
  (1+16+1), is used only for the strong-scaling figure, not as a headline
  accuracy/throughput target.
- **qubit count and depth**: 10x10=100 qubits / depth 42 (headline target: "the largest RQC
  that gets simulated using a classic supercomputer" per the paper's own claim); Sycamore
  53 qubits / 20 cycles (the published quantum-supremacy circuit, used for a direct,
  literature-standard comparison point); 20x20=400 qubits / depth 18 (scaling-only).
- **contraction-order search time vs. execution time (kept separate)**: YES, explicitly
  separated by construction. The PEPS-slicing scheme for the 10x10 lattice is an *analytic*
  heuristic (closed-form choice of slice count S and contraction order, Section 5.1 — not a
  searched result, effectively zero search cost), while the Sycamore circuit's contraction
  path is found via CoTenGra's hyper-optimized search (Section 5.2) with a stated
  multi-objective loss (complexity + compute-density). The paper's Fig. 6 plots
  computational-complexity vs. estimated-simulation-time for several path choices (a "bad"
  CoTenGra path, PEPS+qFlex, AC-QDP/stem-optimization, this paper's PEPS+optimized-slicing,
  this paper's architecture-aware CoTenGra search) but does NOT give an explicit wall-clock
  number for how long the CoTenGra search itself took to run — only the resulting
  complexity and the *execution* time-to-solution are reported numerically. This is a
  genuine open question (see below), not resolved by the accessible text.
- **memory ceiling**: hard per-node ceiling of 96GB (SW26010P node), 16GB per core-group
  (CG), sliced-tensor size explicitly derived to just touch the ~16GB single-CG bound
  (L^(N+b) x 8B = 16GB for N=5, b=1, L=32 in the amplitude representation), forcing the CG-
  pair (32GB combined) parallelization scheme in Section 5.3. This memory-driven
  slicing-depth choice is the paper's own stated design constraint, not incidental.
- **fidelity/accuracy target**: NOT a fixed elementwise numerical tolerance. Two distinct
  checks: (1) the computed-amplitude probability distribution is checked against the
  theoretical Porter-Thomas distribution (Fig. 11, both fp32-only and mixed-precision runs
  overlaid against the theoretical curve); (2) mixed-precision vs. fp32-reference relative
  error is shown to converge below 1% once ~300 aggregated contraction-path "blocks"
  (27,000 individual paths) are summed, with <2% of individual mixed-precision paths
  discarded for under/overflow (Fig. 10). Separately, the paper reports the XEB
  (linear-cross-entropy-benchmarking) fidelity of its own computed correlated-bitstring
  bunch for the Sycamore case (0.741) as a comparison point against Sycamore hardware's own
  reported XEB of 0.2%.
- **single-node vs. distributed**: full-machine distributed only for the headline numbers —
  up to 107,520 nodes / 41,932,800 cores (strong-scaling swept from 199,680 to 41,932,800
  cores, Fig. 13, near-linear). No single-node baseline number is given (the per-CG-pair
  compute-density/roofline analysis in Section 6.3 is the closest single-node-level
  measurement, reported as sustained TFlop/s per 2-CG-pair, not as a standalone single-node
  benchmark).
- **timing protocol**: "the average time recorded for running the same case for three
  times" — 3 repetitions, mean statistic, no stated warmup.
- **precision & correctness**: fp32 (single) and mixed fp32/fp16 (adaptive per-tensor
  scaling) — fp64 is explicitly never used anywhere in this paper, a genuine
  memory-bandwidth-driven design choice (not an oversight) since the workload is
  memory-bound by the paper's own roofline analysis.
- **metric**: sustained PFLOP/s-to-EFLOP/s throughput (headline: 1.2 EFlop/s single-
  precision, 4.4 EFlop/s mixed-precision, for the 10x10x(1+40+1) circuit; 60.4 PFlop/s /
  102 PFlop/s respectively for Sycamore, at markedly lower ~4%/1.7% peak-efficiency due to
  imbalanced rank-30-vs-rank-4 tensor pairs in the CoTenGra-searched path) AND, separately,
  wall-clock time-to-solution for the sampling task (304 seconds to compute a 2^21
  correlated-amplitude bunch for Sycamore, vs. the physical Sycamore device's own 200
  seconds and vs. 10,000 years / 2.55 days / 19.3 days / 5 days for four prior classical
  reproduction estimates/measurements it cites).
- **flop-counting method**: EXPLICITLY dual-method and the paper explicitly picks the
  conservative one — "the number of floating point operations are measured using two
  different methods, by counting all floating point arithmetic instructions needed... and
  by monitoring the floating-point operation hardware counters... As the hardware counters
  generally provide a number that is 10~20% larger (due to the generation of temporary
  floating-point operations along the way), we use the counted number as the basis for a
  conservative and fair Flop measurement."
- **baselines**: literature-reported (not same-machine-reproduced) prior systems only —
  qFlex on NASA Pleiades/Electra (20 PFlop/s) and on Summit (281 PFlop/s, 7x7x(1+40+1)
  circuit), a Summit hard-disk-based estimate (2.55 days for Sycamore), an AliCloud estimate
  (19.3 days), and a 60-GPU measured result (5 days).
- **source**: arXiv fulltext PDF (`arxiv.org/pdf/2110.14502`, full 20-page paper including
  the appendix, read directly).

---

## SV-Sim (SC 2021, "SV-Sim: Scalable PGAS-Based State Vector Simulation of Quantum
Circuits", PNNL/Microsoft)

- **workloads/inputs**: QASMBench suite (the authors' own prior benchmark-suite paper). 8
  "medium" circuits (11-15 qubits: `seca`, `sat`, `cc`, `multiply`, `bv`, `qf21`, `qft`,
  `multiplier`) for single-device and single-node-multi-device evaluation; 8 "large"
  circuits (16-23 qubits: `dnn`, `bigadder`, `cc`, `square_root`, `bv`, `qft`, `cat_state`,
  `ghz_state`) for multi-node scale-out evaluation. Circuit families include a Bernstein-
  Vazirani (`bv`), QFT (`qft`), and application-inspired circuits (VQE-UCCSD H2 chemistry,
  a QNN power-grid-classification prototype) discussed as separate case studies outside the
  main latency tables.
- **qubit count and depth**: 11-23 qubits across the two main tables, up to 24 qubits /
  2.3M gates for the VQE-UCCSD case-study circuit (the deepest circuit in the paper).
- **contraction-order search time vs. execution time**: NOT APPLICABLE — this is a
  gate-by-gate state-vector propagation simulator, not a tensor-network-contraction
  simulator; there is no contraction path to search.
- **memory ceiling**: implicit O(2^n) state-vector memory (16 bytes/amplitude, complex
  double), never stated as a single hard number since the paper's largest reported qubit
  count (24) fits comfortably in every tested platform's memory; the paper frames the
  motivating problem (Section 1) around this exponential wall but does not push up against
  it as a headline result the way SWQsim does.
- **fidelity/accuracy target**: not addressed as a distinct validation step in the
  accessible text — correctness is implied via deterministic gate arithmetic (fp64
  throughout) rather than checked against a stated tolerance or statistical test.
- **single-node vs. distributed**: BOTH, explicitly split into three evaluation categories:
  single-device, single-node-multi-device (scale-up, 2 to 256 CPU cores / 1 to 16 GPUs, on
  6 named platforms), and multi-node (scale-out, up to 1024 CPU cores or 1024 GPUs on OLCF
  Summit).
- **timing protocol**: "CUDA and HIP events for the time measurement" (GPU); "system time
  measurement approach" (CPU, unspecified further); "reported results are the average
  values of 10 times' execution" — 10 repetitions, mean statistic, no stated warmup.
- **precision & correctness**: fp64 (complex double) throughout — the `ValType` state-vector
  storage type is double-precision by design; no explicit correctness-tolerance statement
  found.
- **metric**: absolute and relative latency (ms), no GFLOP/s figure anywhere in the
  accessible text — this is a pure wall-clock/latency-centric paper, unlike SWQsim's
  sustained-throughput framing.
- **baselines**: default simulators of 3 mainstream frameworks on the same machine — Qiskit
  v0.26.2 (IBM), Cirq v0.11.0 (Google), Q# v0.17.2105.143879 (Microsoft), each run on a
  single CPU core or single V100 GPU; SV-Sim reports ~10x average speedup.
- **hardware**: 6 platforms spanning CPU/GPU/Xeon-Phi and NVIDIA/AMD/IBM vendors — Intel
  Xeon Platinum-8276M (224-core server), an AMD EPYC-7742-hosted 8xA100 DGX server, an Intel
  Xeon-P8168-hosted 16xV100 DGX-2, OLCF Spock (36 nodes, 4xMI100/node), OLCF Summit (4608
  nodes, 6xV100/node), ALCF Theta (4392 Xeon-Phi-7230 KNL nodes).
- **source**: repo `github.com/pnnl/SV-Sim`, `doc/paper_sc21.pdf` fetched via raw GitHub URL
  and read directly as a PDF (full 13-page SC'21 paper).

---

## DM-Sim (SC 2020, "Density Matrix Quantum Circuit Simulation via the BSP Machine on
Modern GPU Clusters", PNNL)

- **workloads/inputs**: (a) a synthetic circuit generator (random unitary gates, qubit
  count n and gate count m both swept independently); (b) 13 named real routines from
  QASMBench (`deutsch`, `grover`, `wstate`, `iswap`, `pea3pi8`, `qec`, `qv5`, `w3test`,
  `adder` at 9 and 18 qubits, `sat`, `bv` at 15/17 qubits, `cc` at 15/17 qubits, `qft` at
  15/17 qubits). Note this is density-matrix simulation: an n-qubit density-matrix
  simulation is computationally equivalent to a 2n-qubit state-vector simulation (the
  paper's own stated equivalence, used throughout to compare against state-vector
  baselines), so its largest headline case (15-qubit density matrix) is "equivalent to a
  30-qubit state-vector simulation."
- **qubit count and depth**: 3-15 qubits (density-matrix count) across the main tables (=
  equivalent to 6-30 state-vector qubits); deep-simulation stress test up to 1 million
  gates at 15 qubits (5,645s / 1.5hr wall-clock, 5.65ms/gate average).
- **contraction-order search time vs. execution time**: NOT APPLICABLE — gate-by-gate
  density-matrix propagation, no contraction path. The paper's own algebraic reformulation
  (Eq. 3, reducing the number of required adjoint/transpose communication operations from m
  [one per gate] to 2) is the closest analog to a "preprocessing" step, and it is a
  closed-form algebraic rewrite applied once per circuit, not a searched schedule — its
  cost is effectively zero and not separately timed.
- **memory ceiling**: O(4^n) density-matrix memory, explicitly the paper's headline
  motivating constraint (density matrix scales as the *square* of the state-vector's O(2^n)
  — the paper's Fig. 9-A/B/C roofline analysis and its "arithmetic intensity" framing treat
  this as the fundamental bound); the paper explicitly does NOT chase larger qubit counts
  the way state-vector papers do, targeting circuit *depth* (up to 1M gates) as its scaling
  axis instead of qubit count.
- **fidelity/accuracy target**: not addressed via a stated numerical tolerance; validated
  implicitly by matching a known expected output bit-pattern for the worked adder-circuit
  example, and via the Roofline-model cross-check (achieved GFLOPS vs. theoretical/empirical
  peak) as an indirect sanity check on kernel correctness rather than a numerical-accuracy
  gate.
- **single-node vs. distributed**: BOTH — single-node multi-GPU scale-up (up to 16 GPUs on
  DGX-2, 5 platforms total: SLI/DGX-1P/DGX-1V/DGX-2/RTX-2080-SLI) and multi-node scale-out
  on OLCF Summit (64 to 1024 GPUs, MPI all-to-all with GPUDirect-RDMA).
- **timing protocol**: "we use CUDA Event for the timing measurement. The reported values
  are average of five times' execution" — 5 repetitions, mean statistic, no stated warmup.
- **precision & correctness**: fp64 (`double` `ValType`) throughout, matching SV-Sim; no
  explicit numerical-tolerance correctness gate stated.
- **metric**: GFLOP/s IS reported here (unlike SV-Sim), via an explicit Roofline-model
  analysis (Section IV-B) using the Empirical-Roofline-Toolkit (ERT) for theoretical/
  empirical peak FLOPS and DRAM bandwidth, and per-gate flop counts derived from the
  algorithm's own known operation-count formula (12x8^n double-precision ops for the full
  density-matrix-gate multiply, per the paper's own Section III-A derivation) rather than
  from hardware performance counters. Also reports ms/gate and strong-scaling speedup.
- **baselines**: 10 other literature quantum simulators (not same-machine-reproduced),
  normalized to 20-qubit and 26-qubit-equivalent state-vector systems for comparison
  (Fig. 13), showing >10x speedup on average.
- **hardware**: 5 platforms — NVIDIA Tesla-P100 DGX-1, Tesla-V100 DGX-1, Tesla-V100 DGX-2,
  RTX-2080 SLI (2 GPUs), and ORNL Summit (up to 1024 GPUs).
- **source**: repo `github.com/pnnl/DM-Sim`, `doc/paper_sc20.pdf` fetched via raw GitHub URL
  and read directly as a PDF (full 14-page SC'20 paper, nominated for SC'20 Best Paper per
  its own README).

---

## SliQSim (DAC 2021, arXiv:2007.09304, "Bit-Slicing the Hilbert Space")

- **workloads/inputs**: 4 distinct benchmark sets, and this is the one paper in the track
  whose evaluation is NOT throughput-oriented at all:
  1. Randomly generated circuits, 40/80/120/160/200/300/400/500 qubits, gate:qubit ratio
     fixed at 3:1, 10 circuits generated per size, drawn from a fixed 10-gate library
     (X, Y, Z, H, S, T, CNOT, CZ, Toffoli, Fredkin — excluding Rx(pi/2)/Ry(pi/2) as
     "similar effects" to H).
  2. RevLib reversible-circuit benchmarks (classical-function circuits converted to
     quantum), both as-given and modified by inserting H-gates on unspecified inputs to
     force superposition (14 named benchmarks, 130-923 qubits: `_443`, `add64_184`,
     `apex2_289`, `callif_32_439`, `cps_292`, `cpu_alu_16bit_400`, `cpu_control_unit_402`,
     `cpu_register_32_405`, `e64-bdd_295`, `ex5p_296`, `hwb9_304`, `lu_326`,
     `nestedif2_32_445`, `pdc_307`, `spla_315`, `varops_32_447`).
  3. Quantum-algorithm circuits: entanglement/Bell-state preparation (80-10,000 qubits) and
     Bernstein-Vazirani (80-10,000 qubits, up to 29,999 gates).
  4. Google's published random-quantum-supremacy circuits (from the public GRCS repo,
     `inst/rectangular/cz_v2` directory), 16-90 qubits, depth reduced from the published 10
     to 5 (the full depth-10 circuits being "too difficult to simulate" for this evaluation).
- **qubit count and depth**: this is the paper's PRIMARY axis — not a throughput number but
  a scalability frontier. Random circuits up to 500 qubits (depth = 3x qubit count by
  construction); entanglement circuits up to 10,000 qubits; BV up to 10,000 qubits / 29,999
  gates; Google-supremacy circuits up to 90 qubits at reduced depth 5.
- **contraction-order search time vs. execution time**: NOT APPLICABLE — BDD
  (binary-decision-diagram)-based exact symbolic state representation with bit-sliced
  algebraic-integer amplitudes, not a tensor-network-contraction method at all.
- **memory ceiling**: explicit hard limit enforced as part of the protocol itself — a
  2GB memory-out (MO) limit and a 7200-second time-out (TO) limit per test case, both
  applied identically to the baseline (DDSIM) and to this paper's own method; results are
  reported as counts of TO/MO/numerical-error/segfault cases alongside average runtime
  over only the *successful* cases (a survivorship-biased average, since failed cases are
  excluded from the mean rather than contributing an infinite/censored value).
- **fidelity/accuracy target**: the paper's headline technical claim is EXACT arithmetic
  (an algebraic representation of complex numbers under the Clifford+T-superset gate set
  used, with zero precision loss by construction, verified via `#error` detection: DDSIM's
  own floating-point-based QMDD approach is shown to produce outright numerical-error and
  segfault failures at scale, which this paper's method structurally cannot produce). There
  is no elementwise numerical tolerance to state because the target is exactness, not
  approximation.
- **single-node vs. distributed**: single machine only — "a server with Intel(R) Xeon(R)
  Silver 4210 CPU @ 2.20GHz, 30.7GB RAM," with no stated core/thread count used (implicitly
  single-threaded, since CUDD's dynamic-reordering BDD engine is not described as
  parallelized anywhere in the accessible text).
- **timing protocol**: reported runtime is "only averaged over success cases" per size/
  circuit; no warmup is described (BDD/CUDD state persists and grows across gate
  applications within a single run by design, so a conventional "warmup iteration" concept
  does not apply the way it does to a steady-state numerical kernel).
- **precision & correctness**: exact (see above); explicitly contrasted against DDSIM
  (QMDD-based), which is shown to accumulate "error" cases (defined as: total state
  probability failing to sum to 1 due to precision loss) at scale — DDSIM had 13 MO and 30
  numerical-error cases across the random-circuit set where this paper's method had zero.
- **metric**: wall-clock time (seconds) and memory usage (MB, Google-supremacy set only) —
  no GFLOP/s or any throughput-per-second metric; the headline comparison metric is really
  "how many of the 10 random instances at this qubit count could be simulated at all within
  the TO/MO budget" (e.g. at 49 qubits on the Google-supremacy set, this paper's method
  simulates 9/10 vs. DDSIM's 4/10).
- **baselines**: DDSIM v1.0.1a (QMDD-based, the paper's stated prior state-of-the-art), plus
  a stabilizer-circuit-only comparison against CHP for the entanglement-circuit subset
  (CHP exploits stabilizer-circuit structure DDSIM/this-paper cannot).
- **source**: arXiv fulltext PDF (`arxiv.org/pdf/2007.09304`, the DAC 2021 preprint, full
  9-page paper including all result tables, read directly).

---

## HyQuas (ICS 2021, "HyQuas: Hybrid Partitioner Based Quantum Circuit Simulation System on
GPU", THU-PACMAN)

- **workloads/inputs**: the paper's own `dl.acm.org` PDF and every mirror tried (including a
  proxy text-extraction service) were blocked or unreadable; this entry is built entirely
  from the artifact repo's `README.md`, `scripts/init.sh`, `scripts/check.sh`, and the
  `tests/input/` directory listing. The default circuit set (`tests_28` in `init.sh`) is 7
  named circuit families at 28 qubits each: `basis_change`, `bv` (Bernstein-Vazirani),
  `hidden_shift`, `qaoa`, `qft`, `quantum_volume`, `supremacy` — with `basis_change` also
  swept 24-30 qubits for a weak-scaling set (`tests_scale`), and the full 7-family set also
  available at 25 and 30 qubits (`tests_25`, `tests_30`). This is the only paper in the
  track whose accessible material names a QAOA-family circuit explicitly (the sibling
  tensor-contraction track's already-written variants do not).
- **qubit count and depth**: 24-30 qubits across the accessible test-circuit set (depth not
  independently stated per circuit in the accessible repo material — depth is implicit in
  each named circuit-generator's own construction, not exposed as a separate parameter).
- **contraction-order search time vs. execution time**: PARTIALLY analogous but not a
  literal tensor-network path search — HyQuas's own abstract states it "supports both
  single-GPU... methods, *OShareMem* and *TransMM*... it can select the better simulation
  method for different parts of a given quantum circuit according to its pattern," and the
  repo has a dedicated `evaluator-preprocess/` directory building a "database for the time
  predictor" (`benchmark/preprocess.sh`) — i.e. a per-circuit-segment method-selection
  decision informed by a pre-built performance-prediction database, conceptually similar in
  spirit to a scheduling/path search but for *simulation-method choice* rather than tensor-
  contraction order. Whether this decision's own cost is reported separately from execution
  could not be confirmed from the accessible repo material (open question).
- **memory ceiling**: not stated in accessible material (implicit O(2^n) state-vector
  memory per the paper's title/abstract framing as a state-vector simulator).
- **fidelity/accuracy target**: `scripts/check.sh` diffs each circuit's output log against a
  reference `tests/output/*.log` file (`diff -q -B`) — implying an exact/bitwise-match
  correctness check rather than a stated numerical tolerance, but the exact comparison
  granularity (full state vector vs. summary statistics) could not be confirmed from the
  accessible material.
- **single-node vs. distributed**: BOTH, explicitly — the repo supports single-GPU,
  single-node-multi-GPU, and multi-node-multi-GPU execution (`run-single.sh`,
  `run-multi-GPU.sh`, `run-multi-node.sh`), and the abstract's own headline numbers state
  "up to 10.71x speedup on a single GPU and 227x speedup on a GPU cluster."
- **timing protocol**: the build system prints "Time Cost" lines that `check.sh`/
  `bench_backend.sh` grep out into a log file; no warmup/repetition/statistic could be
  confirmed from the accessible scripts (the reference build/run scripts appear to invoke
  each circuit exactly once per configuration, with no visible repetition loop) — flagged
  as an open question, consistent with this track's general pattern of weakly-specified
  timing protocols outside the fully-accessible papers.
- **precision & correctness**: `-DUSE_DOUBLE=on` is the default build flag in every provided
  script (`check.sh`, `run-single.sh`, `bench_backend.sh`), i.e. fp64 by default, matching
  SV-Sim/DM-Sim rather than SWQsim's fp32/mixed-only choice.
- **metric**: not confirmed beyond "Time Cost" (implies wall-clock time, not GFLOP/s) from
  the accessible material; the abstract's own headline claims are speedup ratios (10.71x
  single-GPU, 227x cluster) vs. unnamed "state-of-the-art quantum circuit simulation
  systems," not absolute throughput numbers.
- **baselines**: not confirmed from accessible material beyond the paper's own abstract
  referring generically to "state-of-the-art quantum circuit simulation systems" and the
  three internal HyQuas backend variants compared against each other in `run-single.sh`
  (`group`, `blas`, `mix`).
- **source**: repo `github.com/thu-pacman/HyQuas` — `README.md`, `scripts/init.sh`,
  `scripts/check.sh`, `scripts/check_wrapper.sh`, `scripts/run-single.sh`,
  `benchmark/bench_backend.sh`, and the `tests/input/` directory listing, all read directly
  via `gh api .../contents/<path>`. The ICS'21 paper PDF itself
  (`dl.acm.org/doi/pdf/10.1145/3447818.3460357`) could not be fetched (blocked/unparseable
  as binary content); no arXiv preprint exists for this paper.

---

## Divergences

- **Two structurally different simulation paradigms are mixed in this track's paper set.**
  4 of 5 papers (SliQSim, HyQuas, SV-Sim, DM-Sim) are gate-by-gate state-vector or
  density-matrix propagation simulators — there is no "contraction order" to search for
  any of them, so the "contraction-order search time vs. execution time" axis this survey
  was asked to investigate is simply not applicable to 80% of the track's papers. Only
  SWQsim performs genuine tensor-network contraction with a real path-search step
  (CoTenGra). This spec resolves the mismatch by giving state-vector/density-matrix
  simulation and tensor-network contraction separate variants rather than forcing every
  paper's evaluation into a single "contraction" template.
- **Precision defaults split cleanly along the same paradigm line.** SWQsim (tensor-network
  contraction, extreme scale, memory-bandwidth-bound) never uses fp64 — fp32 and mixed
  fp32/fp16 only, a deliberate design choice justified by its own roofline analysis.
  SliQSim, SV-Sim, DM-Sim, and HyQuas (all gate-by-gate) default to fp64 (SliQSim uses exact
  algebraic-integer arithmetic, which is a stronger guarantee than any floating-point
  choice; SV-Sim/DM-Sim/HyQuas all use `double`). This spec keeps fp64 as the
  gate-by-gate-variant default and fp32/mixed as the contraction-variant default,
  mirroring the same fp64-vs-fp32 divergence the sibling tensor-contraction track already
  documents between its non-quantum and quantum-circuit variants.
- **Metric choice diverges just as sharply.** SWQsim and DM-Sim both report sustained
  GFLOP/s-scale throughput (via, respectively, a dual hardware-counter/instruction-count
  method and an Empirical-Roofline-Toolkit-based analysis); SV-Sim and HyQuas report only
  wall-clock latency/ms, never a throughput-per-second figure; SliQSim reports neither
  throughput nor pure latency as its headline metric, but rather a scalability-frontier
  metric (how many of N random instances complete within a fixed time/memory budget at a
  given qubit count). This spec's three variants (see spec.yaml) each keep the metric that
  is actually meaningful for the claim being made, rather than forcing a single throughput
  number onto SliQSim's fundamentally different scalability claim.
- **Correctness/fidelity validation methods are entirely non-overlapping across the 5
  papers.** SliQSim: exact arithmetic, no tolerance needed by construction. HyQuas: exact
  bitwise log-diff against a reference (implied, not confirmed in detail). SV-Sim/DM-Sim:
  no explicit tolerance stated at all in either accessible paper. SWQsim: NOT a fixed
  tolerance at all — a statistical goodness-of-fit test (Porter-Thomas distribution match)
  plus a convergence criterion for its mixed-precision approximation. This spec's
  correctness clauses reflect each variant's own paradigm rather than inventing a single
  numeric tolerance that would misrepresent SliQSim's exactness claim or SWQsim's
  fundamentally statistical validation method.
- **Timing-protocol rigor varies from "fully specified" (SWQsim, SV-Sim, DM-Sim, all state
  an explicit repetition count and timer mechanism) to "unconfirmable from accessible
  material" (HyQuas, entirely repo-script-based) to "not applicable in the conventional
  sense" (SliQSim, where a circuit is simulated exactly once per instance and "warmup" has
  no clear meaning for a BDD engine that accumulates state across gate applications by
  design). None of the 5 papers state a warmup-iteration count. This spec adds an explicit
  warmup count to every throughput-oriented variant and explicitly waives it (with
  justification) for the scalability-frontier variant.
- **Confidence note on HyQuas.** This survey's HyQuas entry required unusually heavy
  artifact-repo cross-referencing since the paper itself was entirely inaccessible (blocked
  ACM PDF, no arXiv preprint). Every claim in that entry is repo-sourced and cross-checked
  against the paper's own public abstract for consistency, but should be treated as
  lower-confidence than the other 4 entries, all of which are backed by a full paper-PDF
  read.
