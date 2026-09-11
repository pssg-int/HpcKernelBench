# string-regex-matching — artifact integration summary

5 papers in this track (`output/benchmark_groups.json["string-regex-matching"]`).
Scope ruling (ARTIFACT_GUIDE.md): NVIDIA GPU, single-card only. Newest 3
single-GPU-eligible papers were the integration targets; the other 2 are
cheap SKIPs (platform out of scope), per the task brief.

| Paper | Venue/Year | Key | Platform | Outcome |
|---|---|---|---|---|
| Cicero (MLIR + domain-specific architecture for regex matching) | CGO'25 | `conf/cgo/SomainiCASC25` | FPGA / domain-specific architecture ("other") | **SKIP (scope)** — not NVIDIA GPU, single-card. No build attempted. |
| **ngAP** (Non-blocking Large-scale Automata Processing on GPUs) | ASPLOS'24 | `conf/asplos/GeZ024` | NVIDIA GPU | **BUILT+GATED** — see `ngap/STATUS.md` |
| iMFAnt / MFSA (One Automaton to Rule Them All) | CGO'24 | `conf/cgo/CicoliniCSC24` | CPU | **SKIP (scope)** — CPU-only. No build attempted. |
| GSpecPal (Speculation-Centric FSM Parallelization on GPUs) | IPDPS'22 | `conf/ipps/WangWQW22` | NVIDIA GPU | **SKIPPED (evidence-backed)** — see `gspecpal/STATUS.md`. DFA-table workload model doesn't fit this track's ANML/NFA abstraction; also hit cascading, only partially-resolved build-portability failures on this machine's toolchain (gcc-14/nvcc 12.9). Build attempted, several classes of fix applied, stopped before completion — not a cheap skip. |
| **gpunfa-artifact** (Why GPUs are Slow at Executing NFAs...) | ASPLOS'20 | `conf/asplos/0002PJ20` | NVIDIA GPU | **BUILT+GATED** — see `gpunfa/STATUS.md` |

## Newest-3 accounting

Ordered newest first: Cicero (2025, SKIP/scope) -> ngAP (2024, **BUILT+GATED**)
-> iMFAnt (2024, SKIP/scope) -> GSpecPal (2022, SKIPPED/evidence) -> gpunfa
(2020, **BUILT+GATED**). Of the 3 single-GPU-eligible papers (ngAP, GSpecPal,
gpunfa-artifact — the task's own "third candidate" list), 2 are BUILT+GATED
and 1 (GSpecPal) is SKIPPED after a genuine, evidence-backed attempt. Cicero
and iMFAnt are out of scope by platform (FPGA / CPU respectively) and were
never candidates for this integration pass regardless of recency.

## Gate summary (both BUILT+GATED artifacts)

Both `ngap-nonblockingallgroups` and `gpunfa-obat2` pass
`string-regex-matching`'s exact structural correctness gate
(`regex-kernel-throughput-precompiled`, CORRECTNESS_MODE="exact") against
`kernelbench/domains/automata.py`'s independent reference simulator, on
this module's own synthetic smoke automata, with non-trivial match sets
(6 and 8 reports respectively, identical counts to the CPU floor impl on
the same workloads — not a vacuous zero-match pass). Both artifacts are
monolithic CLIs with no separable prepare/run library API, so both
adapters wrap at the CLI-invocation boundary and document harness-timer
contamination (ANML load + NFA build land inside the timed region) in
their own STATUS.md, alongside the artifact's own clean, matching-only
self-reported throughput/elapsed numbers captured into each result
record's `params`.

Both artifacts' report conventions (offset base, state/report-id encoding)
were reverse-engineered from source and independently verified on a
hand-checkable tiny automaton (`apple.anml`, pattern "apple", expected
matches at offsets 5/47/64/75) before trusting either adapter's output on
larger synthetic workloads — see each STATUS.md's "Report-convention
mapping" section for the full derivation and cross-check.

## Toolchain notes common to both builds

Both `ngap` and `gpunfa` needed the host compiler bumped from the system
default (gcc 7.5.0, too old for C++17 `<execution>`/gcc-14-style STL
internals used by their shared VASim-lineage `commons/` code) to
`/opt/cray/pe/gcc-native/14/bin/g++` (via `-DCMAKE_CXX_COMPILER`/
`-DCMAKE_CUDA_HOST_COMPILER`), plus one or two `#include <cstdint>`
additions where gcc-14's libstdc++ no longer transitively leaks that
header the way gcc-7 did. Both also needed their hardcoded `-arch=sm_86`
(their own RTX 3090 dev-box default) overridden to `-arch=sm_80` for this
machine's A100. All patches are one-line, build-system/portability class
(ARTIFACT_GUIDE rule 3); no kernel logic was touched in either artifact.
`bench/artifacts/toolchain.sh` (nvcc 12.9 pin) was sourced by every
`build.sh`, per ARTIFACT_GUIDE's mandatory toolchain-pin rule.
