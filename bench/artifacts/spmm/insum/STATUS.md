# insum (Insum / IndirectEinsum) — spmm

**Status: BUILT+GATED**

- Paper: "Insum: Sparse GPU Kernels Simplified and Optimized with Indirect
  Einsums", ASPLOS'26. `PAPER_KEY = conf/asplos/WonAAE26` (matched by title
  in `../../output/included.json`).
- Artifact: https://github.com/nullplay/IndirectEinsum
- Commit cloned: `56074f2afa1387a2e08360b17b211c1dd47d36bd` (2026-01-13),
  `git clone --depth 50`.
- Toolchain: python `/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python`,
  torch `2.8.0+cu128`, triton `3.4.0`, CUDA runtime `12.8` (torch-bundled).
  No `nvcc`/sm_80 compile step — the artifact is pure Python
  (`torch.compile` -> `torch._inductor` -> Triton codegen), no CUDA
  extension to build.

## What the artifact actually is

`Insum(einsum_str, **tensors)` (`source/indirecteinsum.py`) parses a small
indirect-einsum DSL (gather/scatter with index indirection and `+=`
accumulation) and lowers it, via `torch.compile`, to a fused Triton kernel.
`source/example.py` demonstrates exactly an SpMM kernel:

```
C[Row[p],n] += Val[p] * B[Col[p],n]
```

i.e. COO-format sparse-times-dense. This is a real kernel entry point, not
the paper's benchmark script — the adapter calls `Insum(...)` directly
(ARTIFACT_GUIDE.md rule 1).

## Patch (rule 3: minimal, recorded)

`source/example.py` line 20 unconditionally sets
`torch._inductor.config.triton.native_matmul = True`. That config knob is
new in PyTorch 2.10 (per the repo's own README: "In order to get the maximum
performance on GPU, you will need PyTorch with `native_matmul` enabled...
released as PyTorch 2.10"). This machine's torch is `2.8.0+cu128`, where the
attribute does not exist and setting it raises `AttributeError`, so
`example.py` could not even run unmodified.

Patched to a one-line version guard:

```python
if hasattr(torch._inductor.config.triton, "native_matmul"):
    torch._inductor.config.triton.native_matmul = True
```

No kernel code touched — only whether the (unavailable) fast-path knob gets
set. `torch.compile` still lowers the same `Insum(...)` call to a working
Triton kernel via its default path; confirmed by running the artifact's own
correctness check (`torch.testing.assert_close(A_dense @ In, out)` in
`example.py`), which passes. `git diff --stat` inside `source/`:
`example.py | 10 +++++++++-`.

**Consequence**: this integration measures Insum's default `torch.compile`
Triton lowering of the indirect-einsum SpMM, not the paper's fastest
`native_matmul`-enabled configuration (unavailable until torch 2.10 ships).
Any throughput numbers taken from this adapter should be read with that
caveat; the correctness/functional result is unaffected.

## adapter.py

- `KERNEL = "spmm"`, `IMPL_NAME = "insum-spmm-coo"`, `PRECISIONS = ["fp32"]`.
- `prepare()`: the artifact's own format conversion — `matrix.csr.tocoo()`
  (CSR -> COO, the format Insum's SpMM expression consumes) — then H2D to
  CUDA tensors (`rows`/`cols` int64, `values` fp32). This is timed as
  preprocessing per the harness contract, not folded into `run()`.
- `run()`: ONE `Insum("C[Row[p],n] += Val[p] * B[Col[p],n]", ...)` call
  (zeroing `C` first since Insum's scatter is `+=`-accumulating, so repeated
  timed calls must each start from a clean output).
- **Finding, not an Insum bug**: `B` is generated here with
  `numpy.random.default_rng(seed).uniform(-1,1,(K,N))`, matching
  `kernelbench.impls.cpu_ref.reference_spmm`'s `_dense_operand` bit-for-bit
  in distribution/seed — NOT `kernelbench/impls/gpu_cuda.py`'s `_dense`
  helper (a `torch.Generator`-seeded draw). Those two RNGs produce different
  sequences from the same integer seed. Reproduced independently while
  building this adapter: running the harness's own built-in GPU baseline —
  `python -m kernelbench.runner --kernel spmm --variant spmm-gpu-kernel-f32
  --impl cusparse-csr-spmm --matrices cant --dims 128 --warmup 1 --reps 3`
  — **also fails the correctness gate** (`max_scaled_err=3.207e+04` vs tol
  `1e-4`), for exactly this reason: `TorchSpMM.prepare()` builds `B` with the
  torch-RNG `_dense` helper, while `cpu_ref.reference_spmm` (the gate's
  reference) builds its own `B` with numpy's RNG — two different matrices
  multiplied against the same `A`, gate fails by construction, independent
  of kernel correctness. This is a **pre-existing harness/built-in-impl
  issue**, out of scope for this artifact-integration task to fix in
  `gpu_cuda.py`; flagging it here since it would otherwise look like an
  Insum-specific problem. This adapter sidesteps it by matching the
  reference's own operand generation directly.
- `timer()` reuses `kernelbench.impls.gpu_cuda.CudaEventTimer` (one event
  pair per iteration, per the specs) — no artifact code duplicated for
  device timing.

## Gate verification (login node, functional check only)

```
PY=/pscratch/sd/c/cunyang/gnn/plexus_env/bin/python
$PY -m kernelbench.runner --kernel spmm --variant spmm-gpu-kernel-f32 \
    --impl insum-spmm-coo --matrices cant --dims 128 --warmup 1 --reps 3
```

Result: **1/1 runs valid**. `max_scaled_err = 4.52e-07 <= tol 1e-4`.
`11.000 ms`, `93.26 GFLOP/s` — reduced-protocol numbers only (warmup=1,
reps=3, shared login-node GPU), explicitly marked non-conforming by the
runner (`conforming: False`); not to be read as a timing result, per
ARTIFACT_GUIDE.md rule 5. No timing sweep was run.

## Not done

- No sweep across matrices/dims — out of scope per the task's login-node
  budget; a single gate check only.
- The `blockspmm.py` script (BSR-blocked variant with grouped/padded COO,
  closer to the paper's fastest configuration) was read but not wired — the
  plain COO `Insum` call in `example.py` is the simplest genuine SpMM entry
  point and was sufficient to gate; `blockspmm.py`'s block-sparse format
  doesn't match this track's plain-CSR `Matrix` workload without additional
  block-structure assumptions the input matrices don't have.


## Baseline role (2026-09-05 selection-rule revision)

**Competitor, not a SOTA baseline.** Under the revised rule (core kernel papers
evaluated on the track's own input regime first; `kernel-papers/output/
baseline_selection.md`), this artifact would not have been selected:
- kernel centrality rated `component` (the spmm kernel is not this paper's headline, kernel-level contribution).
Rating rationale (`output/kernel_centrality.json`): Insum is a general indirect-Einsum compiler (SpMM/SDDMM/sparse-einsum all generated by the same lowering); README shows an SpMM-COO example but no named benchmark matrix suite, so it generates this kernel among others rather than being an SpMM-specific implementation.
It stays in the registry and runs under the same gate as every other
implementation, but Phase 3 does not treat it as the human-SOTA reference for
`spmm`.

## Reproduction on zaratan (2026-09-09)

- Machine: UMD zaratan, A100 MIG 1g.5gb slice (sm_80), GPU node for both the
  build check and the gate (`build.sh` itself needs CUDA to run
  `source/example.py`; the login node here has no GPU).
- Toolchain: torch 2.8.0+cu128, triton 3.4.0, Python 3.12.14, CUDA runtime
  12.8 (torch-bundled); no `nvcc`/compile step (pure Python/Triton, same as
  recorded).
- Build: OK — `source/example.py`'s own SpMM-COO example ran end to end and
  its own `torch.testing.assert_close` correctness check passed;
  `native_matmul` confirmed **NOT** available on this torch 2.8.0+cu128
  (the recorded one-line `hasattr` guard is exercised exactly as intended).
  Build-system changes: none this pass.
- Gate: `spmm-gpu-kernel-f32 --matrices cant --dims 128`: 1/1 valid,
  err=4.42e-07 <= tol 1e-4, PASS. `24.848 ms, 41.29 GFLOP/s` — reduced
  -protocol, non-conforming numbers only (A100 MIG 1g.5gb slice: 14 SMs,
  ~5 GB — much smaller than the GPU used for the recorded run), not a timing
  comparison.
- Deviation from the recorded ruling: none — err (4.42e-07) is the same
  order of magnitude as recorded (4.52e-07), consistent with ordinary fp32
  torch.compile/Triton reduction-order nondeterminism; the correctness gate
  passes cleanly either way.
- Verdict here: BUILT+GATED — same as the recorded ruling.
