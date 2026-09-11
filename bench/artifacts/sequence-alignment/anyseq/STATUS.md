# anyseq (AnySeq) — sequence-alignment

**Status: SKIPPED — toolchain unavailable (AnyDSL/Impala partial-evaluation compiler)**

- Paper: "AnySeq: A High Performance Sequence Alignment Library based on Partial
  Evaluation" (IPDPS'20). `PAPER_KEY = conf/ipps/MullerS0MLKH20`.
  Artifact: https://github.com/AnyDSL/anyseq. Rated core / regime matches /
  single-GPU path exists (`output/kernel_centrality.json`).
- Not cloned. AnySeq is written in Impala and needs the AnyDSL toolchain
  (Impala + Thorin + AnyDSL runtime, an LLVM-based compiler stack) to produce
  CUDA code. `module spider anydsl`, `module spider impala` and `command -v
  impala` all come up empty on this machine and no prebuilt release exists;
  building the AnyDSL stack from source is a multi-hour LLVM build, judged out
  of budget for one baseline (ruling recorded in `../README.md`, 2026-09-05).
- Re-attempt recipe: install AnyDSL per https://github.com/AnyDSL/anydsl into
  `<short>/deps/`, then wrap AnySeq's generated GPU Smith-Waterman for
  `seqalign-exact-pairwise-kernel` (scores gate, `kernelbench/domains/alignment.py`).
