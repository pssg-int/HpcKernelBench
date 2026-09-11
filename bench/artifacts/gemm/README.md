# gemm — artifact integration outcomes

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

## Directory listing (`kernel_centrality.json` keys are `gemm|<paper_key>`)

| dir | paper | key | centrality/regime | verdict |
|---|---|---|---|---|
| `turbofno/` | TurboFNO (SC'25) | `conf/sc/WuZDZHC25` | **component** / partial (FFT-GEMM-iFFT fusion for FNOs; GEMM is one stage) | BUILT+GATED (fp32 real-embedded-in-complex `cgemm`, err 1.76e-07..2.37e-07) |
| `moonpoly/` | MoonPoly-dev (ASPLOS'24) | `conf/asplos/YuLZCFX24` | **component** / partial (dynamic-shape DL tensor-program compiler; GEMM is one of several operator types) | BUILT+GATED fp32 (err 1.62e-07..2.55e-07); fp16 BUILT, gate blocked (fp16-rounding vs. a fp64-calibrated tolerance) |
| `hexcute/` | Hexcute (CGO'26) | `conf/cgo/ZhangDSHSP26` | **component** / partial (layout-synthesis compiler targeting GEMM/FlashAttention/MoE; DL-shaped) | **BUILT+GATED** (fp16 kernel, harness now applies the spec's per-precision fp16 bound `1e-2`; err 1.08e-04..2.31e-04, **3/3 valid** — see "Dependency isolation (2026-09-05)" in `hexcute/STATUS.md`; previously reported as gate-blocked against a stale fp64-calibrated `1e-6` bound before `spec.py` grew `tolerance_by_precision`) |
| `ftgemm/` | **FT-GEMM (ICS'23)** | `conf/ics/WuZLHJWC23` | **core** / matches — this track's own `gemm-square-kernel` variant is explicitly SOURCED from this paper's square sweep | **BUILT** (fused-ABFT `ft_sgemm_large`, fp32; gate FAILS under the harness's precision-agnostic tolerance parser, but PASSES the spec's own stated fp32 bound with 5.6x-33x margin — see below) |

`turbofno`/`moonpoly`/`hexcute` predate this integration pass. All three
are rated `component`/`partial` in `kernel_centrality.json` — none is the
track's canonical, spec-sourced GEMM paper. This pass added `ftgemm/`,
which the centrality data rates `core`/`matches` — the track's spec text
itself cites FT-GEMM's own 1024-6144-step-512 square sweep as the direct
basis for `gemm-square-kernel`'s `recommended_subset`.

## FT-GEMM (ICS'23) — `ftgemm/` — **BUILT**

`PAPER_KEY = conf/ics/WuZLHJWC23`. Wraps `ft_sgemm_large`
(`source/kernel/ft_sgemm/include_code_gen/ft_sgemm_large.cuh`) through a
new `bridge.cu` — the "large"-tile (ms=ns=64, ks=8) fused-ABFT
(Algorithm-Based Fault Tolerance) SGEMM kernel, WITH fault tolerance ON:
the paper's own headline contribution (online error-detection/correction
via row/column checksums accumulated in shared memory as the kernel goes,
self-contained — same 8-argument signature as the plain kernel, no
external checksum buffers). The plain fault-tolerance-OFF counterpart
(`sgemm_large`, same tile config) is also exposed in `bridge.cu` and was
used to cross-check the operand-layout convention, but is not separately
registered (per this task's "disclose which" framing — the FT-on kernel
is the paper's actual point).

**Layout**: A is column-major `(M,K)`, B is row-major `(K,N)` (matches
this domain's own operand convention with NO transform needed), C is
written column-major `(M,N)` — verified empirically against
`dense.py::reference_gemm` on a random 128x192x64 shape before wiring into
`adapter.py`.

**Gate result — a tolerance-PARSER finding, not a kernel defect** (full
evidence in `ftgemm/STATUS.md`): `benchspecs/gemm/spec.yaml`'s
`gemm-square-kernel.correctness` text states *"< 1e-6 for fp64 ... < 1e-3
for fp32 ... < 1e-2 for fp16"*, and this artifact's own measured error
(3.0e-5 to 1.8e-4 across the 3 `--smoke` shapes) is comfortably inside the
spec's STATED fp32 bound (5.6x-33x margin). But `kernelbench/spec.py`'s
free-text tolerance parser extracts a single number from that field
without disambiguating by declared precision — `variant.tolerance` is
`1e-6` regardless of whether the impl runs fp64, fp32, or fp16 (confirmed
directly by loading the spec). Gated against that parsed `1e-6`, the
FT-on kernel's `--smoke` run is `0/3 valid`. Per ARTIFACT_GUIDE.md rule 4
the gate is NOT loosened to compensate; this is recorded as BUILT with the
measured error (precedent: spmm/inferfast), not silently hidden or
force-passed. The disclosed FT-off counterpart passes the SAME 1e-6 gate
cleanly (1.3e-7 to 2.2e-7). This tolerance-parsing gap (same CLASS of
finding as `cg-krylov/spcg`'s own documented `runner.py` precision-text
heuristic issue) is flagged for whoever owns `spec.py`'s parser next, not
patched in this pass (`ARTIFACT_GUIDE.md`: "no `kernelbench/` changes
unless a genuine harness bug is found" — noted, not fixed, per task scope).
