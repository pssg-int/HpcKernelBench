# cg-krylov — artifact integration outcomes

## Baseline selection rule (user decision, 2026-09-05)

Baselines are chosen from `kernel-papers/output/baseline_selection.md`,
produced by `select_baselines.py` from the per-(paper, track) ratings in
`output/kernel_centrality.json`: (1) kernel **centrality** `core` (the
kernel IS the paper's headline contribution) beats `component`
(`tangential` is never a baseline); (2) **regime match** with the track
spec's inputs (`matches` > `partial` > `mismatch`); (3) **single-NVIDIA-GPU
path** required; (4) **recency** only as the tiebreak, up to 5 per track.
Already-integrated adapters the rule would not have picked stay in the
registry as competitors (still gated) but are labelled `component`/
off-regime here rather than treated as the human-SOTA reference.

## Directory listing (`kernel_centrality.json` keys are `cg-krylov|<paper_key>`)

| dir | paper | key | centrality/regime | verdict |
|---|---|---|---|---|
| `spcg/` | SPCG (SC'25) | `conf/sc/MaACSH25` | core / matches — this variant's own artifact-discipline requirement is generalized from SPCG | BUILT+GATED (`cg-e2e-ilu0-to-convergence`, fp32; 3/3 smoke valid, `cant` correctly FAILS — genuine solver stagnation, not an adapter bug) |
| `millefeuille/` | Mille-feuille (SC'24) | `conf/sc/YangZNJS0T024` | core / matches (this track's `cg-mixed-precision-cascade` variant names this paper directly) | BUILT+GATED (`cg-kernel-fixed-iter`; 3/3 smoke valid) |
| `perks/` | PERKS (ICS'23) | `conf/ics/ZhangWCMWEM23` | core / matches — this track's `cg-kernel-fixed-iter` variant quotes PERKS's own research question verbatim | BUILT+GATED |
| `bootcmatchgx/` | **BootCMatchGX (TPDS'23)** | `journals/tpds/BernaschiCVD23` | **core** / matches | **BUILT+GATED** (`cg-e2e-ilu0-to-convergence`, fp64, AMG-preconditioned FCG; 3/3 smoke valid — see below) |

`spcg`/`millefeuille`/`perks` predate this integration pass. This pass
added `bootcmatchgx/`, reusing the already-built, already-gated
`../multigrid/bootcmatchgx/` integration's binary and clone rather than a
second clone/compile.

## BootCMatchGX as AMG-preconditioned FCG (TPDS'23) — `bootcmatchgx/` — **BUILT+GATED**

`PAPER_KEY = journals/tpds/BernaschiCVD23`. **Shared build**: `source/` and
`fcg_bcmg.properties` in this directory are symlinks to
`../multigrid/bootcmatchgx/`'s own files — the same clone (commit
`420223a88d2dfef8b28571db2c9392c1af8e146c`), the same two build-system/I/O
patches (`config_mk.patch`: cray-libsci substitution; `vector_print_
precision.patch`: `Vector::print`'s `%g`→`%.17g`, the fix that lets an
independently-recomputed fp64 residual gate pass at all), and the same
compiled `bin/example/driverSolve`. `build.sh` here verifies that shared
binary rather than rebuilding it (see `bootcmatchgx/STATUS.md`'s
"Shared-build relationship" section, and `../multigrid/bootcmatchgx/
STATUS.md`, both updated to record this relationship).

**Why this variant, not a multigrid-only one**: `cg-e2e-ilu0-to-
convergence`'s own claim text generalizes SPCG's artifact discipline into
a track-wide "whole-solve to a relative-residual convergence gate"
competition (not literally ILU0-restricted); the sibling
`cg-hpcg-generator-amg` variant explicitly cross-references this same
gate when describing its own AMG-preconditioned loop. This adapter adds
BootCMatchGX's AMG-preconditioned FCG as a second competitor on the SAME
protocol `spcg` already competes on (ILU0-preconditioned PCG) — the
variant's own `protocol.precision` text names both papers by name: fp32
primary (SPCG), fp64 secondary (BootCMatchGX).

**Gate discipline — deliberately different from the multigrid sibling's
gate**: `to_host()` uses cg-krylov's own to-convergence formula
(`gate_tol = max(10*rtol, 1e-9)`, mirroring `spcg/adapter.py` and
`solvers.ScipyCG`'s own "to-convergence" branch EXACTLY), not the
multigrid sibling's harder no-slack `relres < rtol` gate — so this
adapter's pass/fail decision is directly comparable to spcg's own numbers
on this track, not silently importing a stricter bar from a different one.

**Gate result**: 3/3 smoke runs valid. Relative residual 3.9e-8 to 7.4e-7
across the 3 synthetic Poisson smoke matrices (6-7 FCG iterations each),
comfortably inside the 10x-slack gate (1e-5) and in fact inside the bare
rtol=1e-6 too.
