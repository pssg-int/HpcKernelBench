# bit-graphblas (Bit-GraphBLAS: Bit-Level Optimizations of Matrix-Centric Graph Processing) — bfs

**Status: SKIPPED**

- Paper: "Bit-GraphBLAS: Bit-Level Optimizations of Matrix-Centric Graph
  Processing on GPU" (IPDPS'22). `PAPER_KEY = conf/ipps/ChenSSTBL22`
  (matched by title + artifact_url in `../../../output/included.json`).
- Artifact: https://github.com/hsung2/Bit-GraphBLAS
- Commit cloned: `32f495c3b9a8a3656f6d5e80af5db02ff47aa0f4`, `git clone --depth 1`.
- No build was attempted (nothing to build against for this track — see
  evidence below); no `adapter.py`/`build.sh`/shim were written, per
  ARTIFACT_GUIDE.md rule 7 ("cheap skip, no build attempt").

## Evidence: the public artifact does not implement BFS

The paper's headline claim ("GraphBLAS-based BFS accelerate up to 433x")
describes an **application built on top of** the paper's bit-level
GraphBLAS primitives, not code shipped in this repository. The entire
repository tree (bounded `find`, depth 3, `.git` excluded):

```
.
./bmv_eval
./bmv_eval/backend
./bmv_eval/backend/bsrbmv.cu
./bmv_eval/backend/csr2bsr_batch_bsrbmv.cu
./bmv_eval/backend/mmio.hpp
./bmv_eval/backend/readMtx.hpp
./bmv_eval/backend/utility.cu
./bmv_eval/Makefile
./bmv_eval/test_bin_bin_bin.cu
./bmv_eval/test_bin_bin_full.cu
./bmv_eval/test_bin_full_full.cu
./bmv_eval/testbmv.sh
./bmv_eval/test_spmv_baseline_a100.cu
./README.md
```

`README.md` is a single line ("# Bit-GraphBLAS") — no further documentation
exists in the repo. The entire codebase is `bmv_eval/`: a **BMV (Boolean
Matrix-Vector multiply) microbenchmark** — `backend/bsrbmv.cu` implements a
single bit-packed BSR-format boolean matrix-vector product (the paper's
"bit-level SpMV" primitive), swept across ~700 named SuiteSparse matrices
by `testbmv.sh` and 4 tile-dimension/format variants
(`test_bin_bin_full.cu`, `test_bin_full_full.cu`, `test_bin_bin_bin.cu`,
plus a cuSPARSE baseline in `test_spmv_baseline_a100.cu`). `Makefile`
confirms this: it builds exactly `baseline`, `bmv{4,8,16,32}_bin_bin_full`,
`bmv{4,8,16,32}_bin_full_full`, `bmv{4,8,16,32}_bin_bin_bin` — 13 BMV
binaries, nothing else.

A repo-wide search (bounded to this cloned tree, `.cu`/`.h`/`.hpp`/`.cpp`
files) for `bfs`, `breadth.first`, `frontier`, or `traversal` (case-
insensitive) returns **zero matches**. There is no frontier data
structure, no level/distance array, no multi-iteration traversal loop, no
visited-bitmap update logic anywhere in the repository — only the single
masked/unmasked matrix-vector product that a GraphBLAS-style BFS would need
to be *composed from* (BFS-as-repeated-masked-BMV is the standard
GraphBLAS formulation), but that composition itself was never published in
this artifact.

This matches `benchspecs/bfs/survey.md`'s own finding verbatim: *"the
public artifact ships only the BMV microbenchmark ... not a full BFS
driver"* and `spec.yaml`'s `open_questions`: *"Bit-GraphBLAS's BFS-specific
protocol ... could not be confirmed from available sources ... the public
artifact ships only the underlying BMV microbenchmark, not a full BFS
driver script."* Independent re-inspection of the actual cloned repository
for this integration pass confirms that finding exactly — nothing new was
discovered that changes the conclusion.

## Why this is SKIPPED, not "wrap a BFS built from BMV ourselves"

One could, in principle, compose a GraphBLAS-style level-synchronous BFS by
calling `bsrbmv` (the real, unmodified kernel) repeatedly with a one-hot
source vector and a growing visited mask, incrementing a level counter
between calls. This was deliberately **not** done, for two reasons:

1. **ARTIFACT_GUIDE.md rule 7 is explicit and directly on point**: *"If the
   artifact genuinely does not implement the track's kernel ... mark
   SKIPPED with evidence and move to the next candidate."* This is exactly
   that situation, not an edge case needing interpretation.
2. **The level-loop control flow would be entirely ours, not the
   artifact's.** ARTIFACT_GUIDE.md rule 1 says to wrap the artifact's own
   kernel invocation "at the finest boundary available" when only an
   end-to-end binary is shipped — that rule presumes the artifact *has* an
   implementation of the track's kernel somewhere in its call graph. Here
   there is no BFS call graph to find a boundary inside of: assembling one
   from the BMV primitive would mean writing a new BFS driver ourselves and
   presenting its performance/correctness as "bit-graphblas's BFS," which
   the artifact's authors never published, tuned, or validated as such.
   That crosses from "wrap the kernel" into "write a new kernel that calls
   theirs" — the same distinction ARTIFACT_GUIDE.md rule 3 draws when it
   says patching kernel code (as opposed to build-system plumbing) means
   the result should be marked SKIPPED, not silently authored around.

## Note for a different track

`bmv_eval/backend/bsrbmv.cu`'s bit-packed BSR boolean matrix-vector product
IS a real, complete, buildable GPU kernel — just not a BFS kernel. It would
be a legitimate candidate for this project's **spmv** (or sddmm/spgemm,
depending on how the bit-packed format is read) track instead, where the
kernel-per-call boundary the artifact actually ships (`bsrbmv`) matches
what that track needs. Out of scope for this bfs integration pass; noted
here so it isn't lost.

## Verdict

`bit-graphblas: SKIPPED — public artifact (commit 32f495c) contains only a
Boolean Matrix-Vector (BMV) microbenchmark (bmv_eval/), no BFS driver,
frontier structure, or traversal loop anywhere in the repository; the
paper's BFS speedup claim describes an application never published in this
artifact. No adapter/build attempted, per ARTIFACT_GUIDE.md rule 7. The
underlying bsrbmv kernel may be worth revisiting for the spmv track.`
