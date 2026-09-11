# Winograd track survey

SINGLETON track: 1 paper (`journals/tpds/LanMHSDWLQF20`, FeatherCNN). Sources
used: fulltext PDF at `hpcc.siat.ac.cn/jintao/jintao/paper/tpds_feathercnn.pdf`
(a co-author's institutional mirror of the TPDS-published version — the
arXiv/OA link field in this project's own `included.json` was empty for this
paper, so this mirror was located via web search and text-extracted locally),
plus the artifact repo `Tencent/FeatherCNN` (README only — `src/` has no
standalone benchmark/timing script, only the library itself; the paper text
is the load-bearing source for every protocol value below).

## FeatherCNN: Fast Inference Computation with TensorGEMM on ARM Architectures (TPDS 2020)

FeatherCNN accelerates 3x3 convolution via Winograd minimal-filtering
algorithms, re-expressed as a custom "TensorGEMM" batched-GEMM formulation
that keeps Winograd-transformed tiles in registers with an ARM-NEON-tuned
packing scheme.

- **Tile sizes F(m,r)**: exactly two configurations are evaluated,
  **F(2x2,3x3)** (u = (m+r-1)^2 = 16 transform-domain elements per tile;
  primary/default scheme, used in essentially every reported number) and
  **F(6x6,3x3)** (u=64; compared only against NNPACK, a narrower comparison).
  Both target r=3x3 kernels only (the paper does not evaluate F(m,5) or other
  kernel sizes). An x86_64/AVX512 F(6x6,3x3) sketch is mentioned (u=64, L=16,
  4 passes) but not benchmarked — ARM is the only hardware family with
  reported numbers.
- **Layer shapes / networks**: Table 3 in the paper gives an EXACT,
  machine-extractable list of 9 named VGG-16 conv-layer shapes (used for the
  step-wise Winograd-vs-GEMM comparison) — C(in), K(out), H(in, square),
  output-tile-count H'xW' for both F(2,3) and F(6,3), and the reference
  GFLOP count per layer. Four full networks are used for network-level
  throughput/scalability numbers: VGG-16, GoogLeNet, ResNet-50,
  MobileNet-V1 (the last has zero 3x3 convs and so exercises the
  non-Winograd/general-conv/depthwise path instead, used to test layer
  fusion rather than Winograd itself).
- **Timing protocol**: for the single-threaded step-wise evaluation (Fig. 6,
  input/TensorGEMM/output-transform breakdown), "**results are averaged over
  20 runs**" — no explicit warmup phase or min/max/median reporting is
  described; mean is the only statistic given. For the network-level
  strong-scaling test (Fig. 8, ARM server), the same lack of an explicit
  warmup/rep-count disclosure holds — thread count and network are the swept
  axes, not repetition count.
- **Timing scope / preprocessing boundary**: this paper is explicit and
  precise about what is included: "effective" GFLOPS = (reference/standard
  conv's FLOP count, Table 3's last column) / compute_time, where
  compute_time = input_transform + TensorGEMM + output_transform, and the
  **filter transformation is explicitly EXCLUDED** because "it is performed
  at initialization" (i.e., a one-shot, amortized-away preprocessing step —
  functionally identical to this project's "format conversion" concept for
  sparse kernels). For the competing plain-GEMM baseline, **im2col time is
  also explicitly excluded** ("to reflect bare-metal performance").
- **Precision**: fp32 throughout (Table 1's peak-performance table is
  explicitly labeled "Theoretical Peak SINGLE-Precision Performance"); no
  fp16/int8 variant is evaluated for Winograd itself (FP16/INT8 are mentioned
  only as a hypothetical future extension in Sec 3.1, not implemented or
  measured).
- **Numerical-accuracy validation**: **NOT PERFORMED.** The paper never
  reports a Winograd-vs-direct-convolution numerical error/tolerance check,
  nor a downstream classification-accuracy delta, despite it being
  well-established in the broader Winograd literature (outside this track's
  1-paper survey) that larger tile sizes — especially F(6x6,3x3), which this
  paper DOES evaluate — lose meaningfully more fp32 precision than F(2x2,3x3)
  due to the larger dynamic range of the transform matrices. This is a real
  gap in the track's own paper, not a survey omission — flagged as
  Divergence 1 / the primary justification for this spec's mandatory
  numerical-accuracy gate.
- **Metric / hardware**: GFLOPS (effective, per above), milliseconds
  (decomposed per Fig. 6), images/sec (network-level, not separately
  reported per layer). Hardware: 2 mobile (Samsung Galaxy S8 / Snapdragon
  835; Apple iPhone 7 Plus / A10 Fusion), 2 ARM servers (Huawei D05,
  2-socket x 32-core Cortex-A72; Phytium FT1500A, 16-core FTC660), 1
  embedded dev board (Firefly-RK3399). Table 4 lists 6 devices total
  (including 3 more phones — Vivo iQOO/Snapdragon 855, Xiaomi 8SE/Snapdragon
  710, Huawei Mate 10/Kirin 970 — used for the peak-GFLOPS reference table
  only, not the step-wise Winograd evaluation).
- **Baselines**: NNPACK (Winograd-specific comparison, F(2,3) AND F(6,3));
  Caffe+OpenBLAS, Caffe2+Eigen (network-level, VGG-16 on ARM server); Eigen,
  Apple Accelerate Framework (plain-GEMM-routine comparison, not
  Winograd-specific); TensorFlow Lite (mobile, whole-network). FeatherCNN
  Winograd F(2,3) beats its OWN plain-GEMM implementation by 21.5-80.6%
  (iPhone) / 50.4-100.3% (Galaxy S8), and beats NNPACK by 36-183% (iPhone) /
  42-212% (Galaxy S8).

Source: fulltext PDF (`hpcc.siat.ac.cn/jintao/jintao/paper/tpds_feathercnn.pdf`,
local text extraction, Sections 3.1/3.2 and 4/4.1/4.2/4.3); `Tencent/FeatherCNN`
README (via `gh api`, confirms no standalone benchmark script exists in the
repo — the paper is the only source of protocol detail).

## Divergences / gaps this spec has to resolve

1. **No numerical-accuracy gate in the source paper**, despite Winograd
   (especially the F(6x6,3x3) configuration this paper itself evaluates)
   being known to lose fp32 precision as tile size grows. Per the task's own
   framing ("numerical-accuracy gate since Winograd loses precision"), this
   spec makes this gate MANDATORY rather than optional, and ties its
   tolerance to tile size (looser for F(6,3) than F(2,3)) since a single
   flat tolerance would either be too loose to catch F(6,3) issues or too
   tight to legitimately pass F(2,3).
2. **Flop-accounting caveat**: the paper's own "effective GFLOPS" metric
   already does the right thing (divides the ALGORITHM-INVARIANT reference
   FLOP count by Winograd's actual, much-lower compute time), and is
   explicit that this can exceed a chip's theoretical peak GFLOPS — but it
   does NOT also report the actual, reduced Winograd-arithmetic op count
   (executed multiply-adds), so a reader cannot tell, from the paper alone,
   how much of the reported GFLOPS gain is "real hardware efficiency" vs.
   "Winograd does ~2.25x fewer multiplies for F(2,3)" algorithmic reduction.
   This spec requires both numbers be reported (mirroring this project's
   `convolution` track's existing primary/secondary GFLOP/s split).
3. **Preprocessing (filter transform) is excluded from timing but its own
   cost is never separately quantified** in the reachable text — the paper
   states it "is performed at initialization" but gives no absolute time or
   fraction. This spec requires it be measured and reported, not merely
   asserted to be negligible.
4. **No warmup/min/max disclosure** — "averaged over 20 runs" is the only
   statistic given anywhere in the paper; there is no discussion of
   discarding a warmup period or reporting variance. This spec upgrades to
   median+min/max per this project's standing fairness principle, noting the
   source paper's own practice falls short of it.
