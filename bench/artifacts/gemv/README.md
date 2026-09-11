# gemv track — artifact integration summary

Three candidates attempted, all resolved:

- **marlin** (PPoPP'25, `conf/ppopp/FrantarCCHA25`) — **BUILT+GATED**,
  `gemv-quantized-weight-kernel`, err 3.60e-05..6.37e-04 (<= 1e-3), W4A16
  decode GEMV. See `marlin/STATUS.md`.
- **packkv** (IPDPS'26, `conf/ipps/JiangYLHDJ26`) — **BUILT+GATED**,
  `gemv-e2e-compressed-format`, err 7.87e-05..1.44e-03 (<= 1e-2), KV-cache
  compressed-K x q fused decode+dot-product. See `packkv/STATUS.md`.
- **gpu-dpf** (ASPLOS'24, `conf/asplos/LamJ0MGLLLRLRW024`) — **SKIPPED**,
  no separable plain-fp/int GEMV: its shipped kernel is PRF-evaluation
  (compute-bound, no table ever read from memory), and the one component
  that IS a materialized `C=A@B` reduction (`GEMM128`) computes wraparound
  uint128 modular arithmetic, not representable via this suite's fp64
  reference gate, and isn't even part of the artifact's actual shipped
  Python interface. See `gpu-dpf/STATUS.md`.

## Newest-three verification (against `benchspecs/gemv/survey.md`)

`survey.md` lists 5 input papers with publication years: QIGen (CGO'26),
PackKV (IPDPS'26), MARLIN (PPoPP'25), GPU-DPF (ASPLOS'24), TLR-MVM
(SC'23). By venue year alone, the three newest are {QIGen, PackKV, MARLIN}
— **not** the three actually attempted above (which include GPU-DPF,
2024, instead of QIGen, 2026).

**QIGen was not attempted, and correctly so**: it is CPU-only.
`survey.md`'s own entry states its hardware as "single-socket AMD EPYC
7742 (64 cores/64 threads, AVX2)", compiled with `gcc ... -mavx -mavx2
-mfma -march=native`, OpenMP-threaded — there is no CUDA/GPU code
anywhere in its own survey entry or repo listing. Per `ARTIFACT_GUIDE.md`'s
scope ruling (2026-08-07): "Integration targets NVIDIA GPU, single-card
implementations only for now. CPU-only... artifacts are SKIPPED with a
one-line reason (platform out of current scope) — cheap skip, no build
attempt." QIGen was never a GPU-eligible candidate, so it correctly drops
out of consideration regardless of how recent it is; no artifact directory
was created for it (a cheap-skip candidate does not need one — this note
in this README constitutes the "one-line reason" the guide asks for).

Restricting to GPU-eligible candidates only, the three newest are exactly
{PackKV (2026), MARLIN (2025), GPU-DPF (2024)} — the three attempted
above. **TLR-MVM** (SC'23, GPU-capable via its own cuBLAS dense-comparison
harness) is the oldest of the 5 and is correctly excluded by the
"three newest" rule on chronology alone, not scope; it is also the
survey's one remaining open question (its SC'23 PDF text could not be
extracted by any tool available during the spec survey, and its own
compressed-format datasets require Zenodo registration/download), so even
absent the recency argument it would not have been the next natural pick
over the three already integrated.
