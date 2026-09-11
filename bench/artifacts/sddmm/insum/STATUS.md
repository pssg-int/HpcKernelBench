# Insum / IndirectEinsum — sddmm — STATUS: SKIPPED

- Paper: Insum: Sparse GPU Kernels Simplified and Optimized with Indirect
  Einsums (ASPLOS'26). `PAPER_KEY = conf/asplos/WonAAE26`
- Repo: https://github.com/nullplay/IndirectEinsum, commit
  `56074f2afa1387a2e08360b17b211c1dd47d36bd` (2026-01-13). `source/` is a
  symlink to `../../spmm/insum/source` — a sibling agent integrating this
  same repo for the spmm track cloned it first; reused here rather than
  cloning a second copy, per the task instructions.

## Evidence

The entire repository is 4 files: `indirecteinsum.py` (the `Insum(...)` DSL
— parses an einsum-like string with gather/scatter indices, e.g.
`"C[Row[p],n] += Val[p] * B[Col[p],n]"`, and lowers it through
`torch._inductor`/`torch.compile` into a Triton kernel), `blockspmm.py` (a
block-sparse SpMM benchmark script built on top of that DSL), `example.py`
(a single COO SpMM demo), and `README.md`. There is:

- no SDDMM example, benchmark, or reference anywhere in the repo (`grep -rin
  "sddmm\|sampled\|dot(A\|attention" **/*.py` returns nothing);
- no dataset, run script, or artifact-evaluation harness of any kind for
  ANY kernel — `blockspmm.py`/`example.py` are the only "benchmarks" and
  both are SpMM;
- confirmed by the earlier benchspecs survey pass
  (`benchspecs/sddmm/spec.yaml`'s `evidence.conf/asplos/WonAAE26` entry,
  written before this integration task and independently reaching the same
  conclusion): "Artifact repo is a single unrelated SpMM-COO Triton-codegen
  demo ... no benchmark/dataset/run scripts."

## Why this isn't "wrap the kernel, just for SDDMM instead of SpMM"

`Insum` is a general indirect-einsum DSL — one could, in principle, write an
SDDMM as an einsum expression in this syntax (something like
`"P[p] += A[Row[p],k] * B[Col[p],k]"` reduced over `k`, scattering to a COO
output instead of a dense one). But the repo ships no such expression, no
verification that the DSL/compiler pipeline actually handles a
reduction-into-scatter pattern like that (every example is `+=` over a
*gather* into a dense output, not a reduction *contracted out* into a sparse
one — a materially different lowering path through `torch._inductor` that
is untested anywhere in this repo), and no benchmark protocol to run it
under. Writing that expression ourselves would mean authoring the SDDMM
kernel's actual computation from scratch using their primitive — which is
squarely what ARTIFACT_GUIDE rule 1 ("wrap the kernel, not the paper's
benchmark script") and rule 3 ("if the kernel itself must change to run,
mark SKIPPED") are guarding against, just one level removed (writing new
*use* of a DSL, rather than editing existing kernel code, but the effect —
us being the ones who determine whether the result is even correct, with no
artifact-side reference to check against — is the same).

## Disposition

SKIPPED per ARTIFACT_GUIDE rule 7 ("If the artifact genuinely does not
implement the track's kernel ... mark SKIPPED with evidence and move to the
next candidate"). No `build.sh`/`adapter.py` written — there is no kernel
entry point to wrap. Moved to the next candidate (RASSM), and then Fused3S
after RASSM was gated — both BUILT+GATED, see their own STATUS.md.
