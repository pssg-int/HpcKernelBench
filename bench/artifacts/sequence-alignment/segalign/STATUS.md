# segalign (SegAlign) — sequence-alignment

**Status: SKIPPED — whole-genome pipeline (e2e variant, PLANNED) with unavailable dependencies**

- Paper: "SegAlign: a scalable GPU-based whole genome aligner" (SC'20).
  `PAPER_KEY = conf/sc/GoenkaTPH20`. Artifact: https://github.com/gsneha26/SegAlign.
  Rated core / regime matches / single-GPU path exists.
- Not cloned. Two independent reasons (ruling in `../README.md`, 2026-09-05):
  1. Its GPU kernel is seed-hit filtering (ungapped extension over seed hits),
     which belongs to the track's `seqalign-e2e-whole-genome-alignment` variant
     — a pipeline variant that `kernelbench/domains/alignment.py` deliberately
     leaves PLANNED; it is not a DP/GCUPS kernel for the two implemented variants.
  2. Its build needs Intel TBB, Boost, LASTZ and kentUtils; `module spider` finds
     no TBB/Boost here and LASTZ/kentUtils are absent.
- Re-attempt only together with the e2e variant; then TBB/Boost via an
  artifact-local build and LASTZ as its baseline.
