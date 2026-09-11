# lossy-compression — artifact integrations

Domain module: `kernelbench/domains/compression.py`. Spec:
`benchspecs/lossy-compression/spec.yaml`. Correctness gate: `max_abs_err`
against the workload's own resolved absolute error bound (`eb`); paired
with compression ratio, never reported alone.

## Baseline selection rule (user decision, 2026-09-05)

Baselines are chosen from `kernel-papers/output/baseline_selection.md`,
produced by `select_baselines.py` from the per-(paper, track) ratings in
`output/kernel_centrality.json`:

1. **kernel centrality** `core` (the kernel IS the paper's headline
   contribution) > `component` (part of a bigger pipeline) — `tangential`
   is never a baseline;
2. **regime match** with the track spec's inputs: `matches` > `partial` >
   `mismatch`;
3. **single-NVIDIA-GPU path** required (current scope);
4. **recency** only as the tiebreak; up to 5 per track.

Already-integrated adapters the rule would not have picked stay in the
registry as competitors (still run under the same gate) but are labelled
`component`/off-regime here and are not the "human SOTA" reference in
Phase 3.

## Directories

| dir | paper (PAPER_KEY) | centrality | regime | status |
|---|---|---|---|---|
| `pfpl` | PFPL, IPDPS'25 (`conf/ipps/FallinADCB25`) | core | matches | BUILT+GATED |
| `cuszp` | cuSZp3/"VGC", SC'25 (`conf/sc/HuangDLC25`) | core | matches | BUILT+GATED |
| `cuszp-v1` | cuSZp, SC'23 (`conf/sc/HuangD0LC23`) | core | matches | BUILT+GATED |
| `fzgpu` | FZ-GPU, HPDC'23 (`conf/hpdc/ZhangTD0F0TC23`) | core | matches | BUILT+GATED |
| `lscomp` | lsCOMP, SC'25 (`conf/sc/HuangDUMCLSC25`) | core | **mismatch** | SKIPPED (integer-only API, no abs/rel eb axis — see its STATUS.md; the SAME paper's kernel IS wrapped for the lossless-compression track, `../lossless-compression/lscomp/`, where its own documented lossless mode fits that track's bit-exact contract) |
| `ffcz` | FFCz, IPDPS'26 (`conf/ipps/RenUDKLYCG26`) | component | partial | SKIPPED (post-hoc correction stage atop another compressor's output, not a standalone `compress(D,eb)->C`) |
| `packkv` | PackKV, IPDPS'26 (`conf/ipps/JiangYLHDJ26`) | component | mismatch | SKIPPED (no general GPU float-array compressor API; LLM-KV-cache-specific) |

`cuszp` and `cuszp-v1` share one upstream repository
(`https://github.com/szcompressor/cuSZp`) spanning 3 SC papers (SC'23/
SC'24/SC'25) — see `cuszp/STATUS.md`'s "PAPER_KEY disambiguation" and
`cuszp-v1/STATUS.md` for the full trail of which directory wraps which
paper's actual kernel code (confirmed genuinely distinct code paths, not
just a version bump).

## New this pass

- **`cuszp-v1`**: the SC'23 original single-mode kernel
  (`SZp_compress_deviceptr_f32`, no dim/mode parameters) — a distinct code
  path from `cuszp/`'s SC'25 dim×mode-versatile kernel, confirmed by
  checking out tag `cuSZp-V1.1` as a git worktree of the same local clone.
  BUILT+GATED, err≤eb on 3/3 smoke workloads, ratio ~4.08x (markedly lower
  than SC'25's 54.00x on the same smoke field — real evidence of the
  paper-to-paper improvement, not a bug).
