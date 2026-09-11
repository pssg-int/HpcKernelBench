# Survey — moe-kernel (singleton track: 1 paper)

## conf/cgo/ZhangDSHSP26 — Hexcute: A Compiler Framework for Automating Layout Synthesis in GPU Programs (CGO 2026)

- **What is measured**: Hexcute is a GPU-kernel-layout-synthesis compiler (constraint-programming
  + type-inference over tensor layouts). Among its evaluated kernel classes (GEMM, Attention, FP8
  GEMM, warp-specialized kernels) is a **mixed-type Mixture-of-Experts (MoE) operator** — the paper
  states "for mixed-type mixture-of-experts (MoE) operators, Hexcute achieves an average speedup
  of 6.46x over Triton" and separately reports up to 2.60x end-to-end speedup on DeepSeek-R1-AWQ
  (a real MoE model) inside vLLM.
- **workloads/inputs**: not enumerated as explicit (expert-count, top-k, hidden-dim) shapes in the
  abstract or the fetched top-level README; the artifact's `scripts/run_moe.sh` invokes
  `hidet/examples/cute/test_moe.py` (two vLLM environments, v9.2 and v8.2) — the actual MoE shapes
  live in that Hidet example script, which was not fetched (out of scope for this repo's top-level
  listing). The paper's end-to-end target model, DeepSeek-R1-AWQ, is a real large-scale MoE (256
  routed experts, top-8 routing, shared experts) — used here as the anchor for canonical shapes in
  `recommended_subset` below since the artifact's own concrete shape list could not be recovered.
  "Mixed-type" means activation and weight are in different low-precision formats (the paper's own
  FP8-GEMM and AWQ-quantization context strongly implies weight-quantized (e.g. INT4/AWQ or FP8)
  x bf16-activation, the same asymmetric-precision pattern as `benchspecs/quantized-gemm`).
- **timing protocol**: `scripts/run_a100.sh` / `run_h100.sh` / `run_moe.sh` all call
  `nvidia-smi -lgc 2000` (GPU clock **locked** before benchmarking, restored with `-rgc` after) —
  the artifact requires `--privileged` Docker specifically for this. Per-iteration warmup/repeat
  counts for the MoE benchmark specifically are not visible in the top-level README; the general
  kernel-benchmark scripts (`run_a100.sh`, 5 hours for the full A100 sweep) imply many repeated
  configs but not a stated warmup/rep constant.
- **timing scope**: kernel-only speedup number (6.46x vs Triton) is separate from the end-to-end
  vLLM number (2.60x on DeepSeek-R1-AWQ) — the paper itself keeps these two claims distinct, which
  this spec's kernel/e2e variant split follows.
- **precision & correctness**: mixed-type (quantized weight, bf16/fp16 activation); no explicit
  correctness-gate tolerance stated in the fetched README (the repo's `hidet` submodule likely
  carries Hexcute's correctness-test suite, not audited here).
- **metric**: speedup (dimensionless x) vs Triton for the MoE kernel; end-to-end tokens/s implied
  for the vLLM DeepSeek-R1-AWQ/Mamba numbers (not itemized in the fetched text).
- **baselines**: Triton (kernel-level MoE); vLLM's own DeepSeek-R1-AWQ / Mamba-based-model serving
  path (end-to-end).
- **source**: abstract (`output/included.json`); repo README + `scripts/run_moe.sh`
  (`gh api repos/hexcute/hexcute-bench/contents/...`). Note: this same paper/repo is also surveyed
  under `benchspecs/attention-kernel` (key `conf/cgo/ZhangDSHSP26`) for its GEMM/Attention layout-
  synthesis claims — this track covers only the MoE-specific operator, a structurally distinct
  grouped/routed-GEMM problem from dense attention.

## Divergences

- Single-paper track: no cross-paper divergence to resolve. The main gap is that the artifact's
  concrete MoE shape suite (expert count, top-k, hidden/intermediate dims) is not recoverable from
  the top-level repo listing fetched — `recommended_subset` below is therefore grounded in publicly
  documented real MoE model configs (DeepSeek-V2/V3/R1, Mixtral) that match the paper's own named
  end-to-end target (DeepSeek-R1-AWQ) and the general "moe-kernel" track brief (expert count/top-k
  routing, token-distribution imbalance, batch), not lifted verbatim from `test_moe.py`.
- The paper reports GPU-clock-locked kernel speedup (6.46x) and un-locked, real-serving end-to-end
  speedup (2.60x) as two different regimes; this spec keeps them as two variants rather than
  conflating into one number, matching the paper's own separation.
