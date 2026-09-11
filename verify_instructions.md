# Verify-pass instructions — HPC kernel-optimization survey

You are re-checking borderline verdicts from a first classification pass. First read
`classify_instructions.md` in this directory for the include/exclude criteria and
category ids — those rules are authoritative. Then re-judge each paper in your
input chunk INDEPENDENTLY from title+abstract, and only after forming your own
judgment compare with `current_verdict`.

Each input item has `mode`:
- `confirm-include`: currently included with low/medium confidence. Flip to
  excluded if the contribution is NOT a runnable software kernel implementation
  (e.g. it is really a runtime, scheduler, serving system, simulated accelerator,
  whole application, or perf model).
- `confirm-exclude`: currently excluded but suspicious (kernel-y title/abstract).
  Flip to included if the primary contribution IS an optimized kernel
  implementation / kernel-producing codegen per the criteria.

## Boundary rulings (apply uniformly; they override first-pass judgment)
1. **NTT / crypto kernels**: GPU/CPU NTT, polynomial multiplication for FHE or
   post-quantum crypto ARE kernel work → include under `fft_spectral`
   (add `primitives` if sampling/hash kernels dominate). Pure protocol/security
   papers without a kernel contribution stay excluded.
2. **GNN aggregation / sparse-dense fusion**: include under `sparse_la` +
   `ml_kernels` when the SpMM/SDDMM-like kernel is the contribution; a GNN
   training *system* stays excluded.
3. **FPGA**: HLS/software-programmable implementations of standard kernels are
   included (platform `fpga`); RTL/architecture proposals are `hardware-accel`.
4. **Database/query kernels** (joins, learned indexes, skyline): excluded as
   `other` unless the contribution is a generic parallel primitive (sort, hash
   table, set intersection) evaluated as such.
5. **Bayesian networks, branch-and-bound, SAT/SMT**: excluded (`application` /
   `other`) unless recast as one of the canonical kernel categories.
6. **Quantum-circuit simulation on classical HPC hardware**: include
   (usually `tensor` or `dense_la`); quantum-circuit compilation/EDA excluded.

## Output — write a raw JSON array (no fences), one object per input, same order:
```json
{
  "key": "<copied exactly>",
  "is_kernel_opt": true,
  "confidence": "high|medium|low",
  "categories": ["..."], "kernels": ["..."], "platform": ["..."],
  "approach": "manual|codegen|autotuning|library|algorithmic",
  "excluded_reason": null,
  "one_liner": "...",
  "changed": true
}
```
- Emit the FULL corrected verdict (not a diff). If you agree with the current
  verdict, copy it, set confidence "high" if you are sure, and `changed`: false.
- `changed`: true iff you flipped is_kernel_opt or materially changed categories.
Write to the output path given in your task prompt. Reply with one line:
"F flipped, C confirmed / M total".
