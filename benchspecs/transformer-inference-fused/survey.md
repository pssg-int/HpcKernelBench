# Survey — transformer-inference-fused (singleton track: 1 paper)

## conf/ipps/ZhaiJWJZCLZ23 — ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs (IPDPS 2023)

- **What is measured**: a fully fused BERT-style transformer **encoder layer** on GPU — not just
  the attention core (that narrower kernel is already covered by `benchspecs/attention-kernel`,
  which also surveys this same paper for its MHA-only number). This track's scope is the whole
  fused pipeline: padding-free QKV projection + bias, multi-head attention, softmax, feed-forward
  network (2 GEMMs + activation), and layer-norm, all executed without padding variable-length
  sequences to a fixed max length.
- **workloads/inputs**: standard BERT-base encoder (12 layers, head_num=12, head_size=64);
  `benchmark/bert_bench.sh` sweeps batch_size in {1,2,4,8,16}, seqlen in
  {32,64,128,192,256,320,384,448,512,576,640,704,768,832,896,960,1024}, at a fixed
  `avg_seqlen_percent = 60` (i.e., the *average* sequence length within a padded batch is fixed at
  60% of the nominal seqlen, modeling realistic length variance) — a full batch x seqlen grid of
  16 x 5 = 80 configurations, fp16 only in the released script. The paper also demonstrates
  applicability to ALBERT, DistilBERT, and DeBERTa (architecturally similar BERT variants) but the
  released benchmark script targets BERT only.
- **timing protocol**: `bert_bench.sh` calls `bert_transformer_test.py <batch> <seqlen> <head_num>
  <head_size> --avg_seqlen=... --n_layers=12 --dtype=fp16` and greps the **last line** of that
  script's stdout for a single "BT Latency (ms)" number per config (`tail -n 1 ${tmp_log} | ... |
  awk '{print $3}'`) — i.e. one reported number per (batch, seqlen) cell, no visible warmup/repeat
  count or min/max/median spread at the shell-script level (the internal Python driver may loop
  more, but that is not exposed in the released orchestration script).
- **timing scope**: full 12-layer BERT encoder forward pass (`n_layers=12` hardcoded in the bench
  script) — i.e. this is a stacked multi-layer latency number, not a single fused-layer kernel time
  in isolation; a single-layer variant requires dividing by layer count or a separate harness
  invocation (`bert_transformer_test.py` with `--n_layers=1` is exposed as a CLI flag).
  Padding-free execution removes computation on pad tokens; this is itself the paper's structural
  claim and is folded into every reported latency (not a separately-reportable preprocessing cost).
- **precision & correctness**: fp16 default in the released harness; unit tests
  (`bert_transformer_test.py ... --export_data` then `bin/bert_transformer_test`) exist as a
  correctness check but are a SEPARATE code path from the benchmark script — the benchmark driver
  itself does not call the correctness check inline (same disconnect pattern noted for MEATTEN in
  `benchspecs/attention-kernel`).
- **metric**: latency (ms) per (batch, seqlen) cell; the paper's headline numbers are relative
  speedups vs. baselines at those cells (e.g. 6.13x vs PyTorch for the fused-MHA-only number,
  87%/131%/138%/74%/55% faster end-to-end vs PyTorch-JIT/TF-XLA/TurboTransformer/DeepSpeed-
  Inference/FasterTransformer respectively for the full encoder).
- **baselines**: PyTorch, TensorFlow (XLA), Tencent TurboTransformer, Microsoft DeepSpeed-
  Inference, NVIDIA FasterTransformer (with and without its own remove-padding optimization).
- **source**: abstract (`output/included.json`); repo README
  (`gh api repos/bytedance/ByteTransformer/contents/README.md`) and
  `benchmark/bert_bench.sh` (`gh api .../contents/benchmark/bert_bench.sh`).

## Divergences

- Single-paper track: the main internal tension is scope, not cross-paper disagreement — the
  paper's own headline numbers mix a **kernel-only MHA claim** (6.13x vs PyTorch, already the
  subject of `benchspecs/attention-kernel`'s `attn-prefill-kernel-fp16bf16` variant) with a
  **12-layer end-to-end encoder claim** (the %-faster-than-framework numbers). This track's spec
  deliberately targets the latter (the whole fused encoder layer, the "fusion scope" this track's
  brief calls out), to avoid duplicating the attention-only variant already specified elsewhere.
- The released benchmark script reports a single latency number per config with no visible
  repeat/median (same gap flagged for this paper in `benchspecs/attention-kernel`'s
  notes_on_fairness); this spec fixes it the same way, requiring >=20 repetitions with device-side
  timing.
