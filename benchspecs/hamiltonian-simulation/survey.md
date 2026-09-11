# Hamiltonian-simulation track — evaluation-methodology survey

## Scope note (read this first)

`output/benchmark_groups.json["other:hamiltonian-simulation"]` (directory
`hamiltonian-simulation`) has exactly 1 paper. This is a genuinely thin
track and this survey covers that 1 paper (below the instructions' "≥5"
bar, but the instructions explicitly permit "all, if fewer"). Unlike the
other two singleton tracks filled in alongside this one
(`particle-in-cell`, `amr-kernel`), this track is unusually
**well-sourced on all three fronts at once**: arXiv fulltext was fully
reachable (`arxiv.org/html/2606.16959`, with detailed Evaluation-section
content extracted), the artifact repo is verified and its actual benchmark
driver script (`benchmark_apps/app2_hamlib_latest.py`) was read in full
(not just the README), and the README itself documents exact CLI usage,
hardware, and per-microbenchmark flags. This is the best-grounded of the
three singleton specs written in this batch.

---

## Chundury et al. — Diagonal-Budgeted Trotterization for Efficient Quantum Hamiltonian Simulation
ICS 2026, `conf/ics/ChunduryBLSM26`, arXiv `2606.16959`, DOI
`10.1145/3797905.3807869`; artifact `github.com/srikarchundury/diaq_for_hamsim`
(`artifact_status: verified`); companion library repo
`github.com/srikarchundury/diaq` (DiaQ, the diagonal-sparse-matrix engine
this repo builds on, referenced but not itself read in detail).

- **What it does**: classical (non-quantum-hardware) simulation of quantum
  Hamiltonian time evolution `e^{-iHt}|psi_0>` via Trotterization. The
  paper's contribution, "diagonal-budgeted Trotterization," picks the
  SMALLEST number of Trotter steps `r` such that the per-step propagator
  `e^{-iH(t/r)}` still fits within a user-set diagonal-count budget
  `D_max`, rather than fixing `r` a priori from a generic error bound —
  this adaptive-`r` search is confirmed, read directly, in
  `estimate_min_timesteps_diaq()` in `app2_hamlib_latest.py`: a
  binary/exponential search over candidate step counts, run ONCE per
  (Hamiltonian, budget) pair as a one-time preprocessing step (matches the
  README's own framing: "a one-time preprocessing step per Hamiltonian and
  budget... negligible compared to the overall simulation cost"). The
  per-step propagator itself is then applied via `HamSim`'s diagonal-sparse
  data layout (`DiaQ` library) with C++/CUDA kernels, bypassing generic CSR
  overhead.
- **Hardware** (confirmed from arXiv fulltext): CPU results — "AMD EPYC
  8124P 16-Core Processor with 192 GB of DDR5 4800 MHz ECC memory,"
  compiled with "GCC 12.3 with the -O3 optimization flag," using "all 32
  hardware threads with simultaneous multithreading (SMT)" (i.e. 16
  physical / 32 logical cores). GPU results — "NVIDIA H100 NVL PCIe Gen5
  GPU (94 GB HBM3, SM 9.0)," CUDA kernels compiled with `nvcc -O3`. (Note:
  the README separately mentions "AMD EPYC 8124P (32 threads)" for its own
  headline numbers, consistent with the fulltext's 16-core/32-SMT-thread
  description.)
- **Workloads/inputs** (confirmed from arXiv fulltext, Table 3): all drawn
  from **HamLib** (Sawaya et al. 2024, a public benchmark suite of quantum
  Hamiltonians as Pauli strings in HDF5, hosted on the NERSC public
  portal), across 4 families:
  - **TSP** (Traveling Salesman): 8–16 qubits, 6 instances, purely diagonal
    (1 active diagonal) — a binary-optimization Hamiltonian.
  - **MaxCut**: 10–16 qubits, 8 instances, purely diagonal (1 active
    diagonal) — binary optimization.
  - **Heisenberg** model: 6–14 qubits, 7 instances, ~19 active diagonals at
    10 qubits — condensed-matter.
  - **TFIM** (Transverse Field Ising Model): 7–16 qubits, 6 instances, ~21
    active diagonals at 10 qubits — condensed-matter.
  Table 3 (per the fulltext extraction) lists exact diagonal counts
  (1–33 active diagonals across all instances) and sparsity percentages
  (93.75%–99.9985%) per instance — i.e. this is a genuinely
  structure-characterized, not just size-characterized, input suite.
- **Timing protocol**: the paper's own reported protocol (fulltext) is
  **"measured, averaged over 10 runs, with whiskers of bar charts providing
  the standard deviation"** — mean + stddev over 10 runs, no explicit
  warmup mentioned in the extracted text. This is DIFFERENT from the
  artifact script's own DEFAULT: `app2_hamlib_latest.py`'s `ITRS` CLI
  argument (repetition count) defaults to **5**, not 10, when unspecified
  — confirmed directly from the script's argument-parsing block
  (`ITRS = int(sys.argv[6]) if len(sys.argv) > 6 else 5`). This means the
  paper's actual reported numbers used `ITRS=10` explicitly on the command
  line, not the script's bare default; flagged as a discrepancy a
  reproducer must not miss.
- **Timing scope / loop structure** (confirmed directly from reading
  `benchmark_apps/app2_hamlib_latest.py` end-to-end): for the timed method,
  the FIRST iteration's `perf_counter()`-measured wall time is printed
  immediately (not discarded as warmup), then `ITRS-1` additional
  repetitions are run and each individually printed as a separate CSV-like
  line (`method,num_qubits,key,time`) — **there is no discarded warmup
  iteration in the public script as read**, matching this project's
  standing observation (see e.g. the `nbody` track's survey) that several
  otherwise-careful artifacts skip a warmup discard. The diagonal-budget
  search itself (`estimate_min_timesteps_diaq`) runs once, BEFORE the timed
  region, and is not included in any of the `ITRS` per-method timings —
  i.e. the one-time preprocessing claim in the README is directly
  confirmed structurally in the code, not just asserted in prose.
  Aggregation (mean/median/stddev across the `ITRS` printed lines) is done
  DOWNSTREAM in separate Jupyter analysis notebooks
  (`1.1_analysis_hamlib_results*.ipynb`), not inside the timing script
  itself — the script's own job is only to print raw per-iteration times.
- **Precision & correctness**: complex128 (double-precision complex)
  throughout (confirmed from `dtype=np.complex128` in the Pauli-matrix
  construction in the script). Fidelity is computed via Qiskit's
  `state_fidelity()` against a NumPy-dense reference state, itself computed
  via `scipy.linalg.expm` applied to the FULL (non-Trotterized) Hamiltonian
  when `TEST_FIDELITY=True` (confirmed from `run_numpy()`); this reference
  computation is automatically SKIPPED for `num_qubits >= 16` "to avoid
  memory issues" (confirmed directly from the script's own guard) — i.e.
  the correctness check itself is scale-limited by construction, not just
  by choice, and this limitation is baked into the artifact, not an
  omission this survey is inferring.
- **Baselines** (confirmed from both fulltext and script): NumPy dense
  reference, SciPy CSR + `expm_multiply`, SciPy DIA format, Qiskit-Aer
  (1st-order Trotter, CPU statevector simulator), Qiskit-Aer-GPU (cuQuantum
  backend, `AerSimulator(method='statevector', device='GPU')` confirmed
  from script). All 6 methods are implemented as CLI-selectable branches in
  the SAME driver script (`METHOD` argument), i.e. a genuinely
  apples-to-apples same-harness comparison, not separately-authored
  benchmark scripts per baseline.
- **Metric**: raw wall-clock time per run (seconds), from which the paper
  computes speedup-vs-baseline ratios. Reported speedups (fulltext, CPU):
  TSP 182–1,269x vs. Aer, 90–3,900x vs. `expm_multiply`; MaxCut 543–1,269x
  vs. Aer, 7,800–120,000x vs. CSR; Heisenberg 4.8–841x vs. `expm_multiply`
  (with Aer's OWN fidelity as low as 0.18, vs. HamSim's ~1.0 — i.e. part of
  the paper's claim is that the CHEAPER baseline is also LESS ACCURATE,
  not merely slower); TFIM 3.2–841x speedup (Aer fidelity as low as 0.03).
  GPU: 7.8–37.1x on 8–10 qubits, up to 178x on 12–16-qubit instances.
- **Microbenchmarks** (confirmed from repo file listing + README): 3
  isolated-kernel scripts separate from the end-to-end HamLib application —
  `microbench_spmv.py` (diagonal-sparse SpMV throughput across qubit
  sizes/diagonal fractions/formats: DiaQ, DIA, CSR, NumPy),
  `microbench_spgemm.py` (SpGEMM with independently controlled density for
  both operands), `microbench_expm.py` (matrix-exponential runtime vs.
  evolution time `t` and diagonal count) — each accepts `--qubits`,
  `--diag-frac`, `--format` CLI flags per the README and writes
  per-combination results to text files. These isolate the core
  diagonal-sparse linear-algebra kernel from the Trotter-step orchestration
  and HamLib I/O overhead the end-to-end app also pays.
- **Source**: arXiv fulltext (`arxiv.org/html/2606.16959`, full
  Evaluation-section extraction); repo `README.md` (full read, both the
  algorithm description and benchmark-application usage sections); repo
  file `benchmark_apps/app2_hamlib_latest.py` (read in full via `gh api
  ... -H "Accept: application/vnd.github.raw+json"` — not just described,
  the actual timing-loop code was read line by line).

---

## Divergences

Only 1 paper is available, so there is no cross-paper divergence in the
usual sense of this survey template. The substantive divergences are
between different SOURCES for this one paper, and between the paper/repo
and the parent task's assumed axes:

1. **README default (`ITRS=5`) vs. paper's actual reported protocol
   (10 runs)**: the script's own default repetition count does NOT match
   what the paper's fulltext says was used to generate its own reported
   numbers. This spec uses 10 reps (the paper's own reported number),
   flagged explicitly as differing from the artifact's bare default so a
   reproducer running the script with no `ITRS` argument does not silently
   under-sample relative to the paper.
2. **No discarded warmup iteration exists in the script as read** — the
   first of the `ITRS` timed iterations is included in the reported set,
   not discarded. This spec adds an explicit warmup discard (see
   notes_on_fairness), a deviation from both the paper's stated protocol
   and the script's actual behavior.
3. **"Timestep count" in the parent task's suggested axes maps to TWO
   distinct numbers in this paper**, not one: (a) `FINAL_TIME` (the
   physical evolution time simulated, script default 1.8) and (b)
   `NUM_TIMESTEPS`/`r` (the number of Trotter steps), which is itself
   DERIVED from the diagonal-budget search rather than fixed independently
   — i.e. "timestep count" here is an OUTPUT of the algorithm being
   benchmarked (for the `diaq` method), not purely an independent input
   knob, for the paper's own method. Baselines that don't do a diagonal-
   budget search (Qiskit-Aer, dense NumPy) use whatever `NUM_TIMESTEPS`
   the diaq search produced for a given `(H, D_max)`, held fixed across
   methods for a fair apples-to-apples step count — confirmed from the
   script structure (all methods share the same `exact_times` list built
   once from the resolved `NUM_TIMESTEPS`).
4. **"Fidelity/energy-conservation gate" maps directly and well** onto this
   paper's own state-fidelity-vs-NumPy-dense-reference check — no
   substitution or invention needed here, unlike the `amr-kernel` and (in
   part) `particle-in-cell` tracks surveyed alongside this one. The one
   caveat is scale: the reference itself is infeasible above ~15 qubits by
   construction (dense `2^n x 2^n` state), so the correctness gate cannot
   cover the paper's own largest-scale (16-qubit) performance-only claims.

