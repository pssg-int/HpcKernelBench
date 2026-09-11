# lossless-compression — artifact integrations

Domain module: `kernelbench/domains/compression.py`. Spec:
`benchspecs/lossless-compression/spec.yaml`. Correctness gate: `exact`
(bit-for-bit round trip, non-negotiable — no tolerance); paired with
compression ratio and compress/decompress throughput, never reported alone.

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
| `zipserv` | ZipServ, ASPLOS'26 (`conf/asplos/FanYPLLW0026`) | core | matches | BUILT+GATED, gate FAILS (genuine bit-exactness bug in ZipServ's own decompress kernel — see its STATUS.md) |
| `mans` | MANS, SC'25 (`conf/sc/HuangYYLGLJWFHT25`) | core | matches | BUILT, gate CRASHES (CUDA runtime fault inside MANS's own kernel — see its STATUS.md) |
| `gpulz` | GPULZ, ICS'23 (`conf/ics/ZhangTDYSTC23`) | core | matches | BUILT+GATED |
| `lscomp` | lsCOMP, SC'25 (`conf/sc/HuangDUMCLSC25`) | core | matches | BUILT+GATED |
| `opthuffdec` | OptHuffDec, IPDPS'22 (`conf/ipps/RiveraDTYTC22`) | core | matches | BUILT+GATED |

## New this pass

- **`lscomp`**: light-source-detector integer compressor, wrapped in its
  own documented lossless mode (`quantBins=1,1,1,1`, `poolingTH=1` —
  README: "makes lsCOMP operate in a lossless mode"). fp32 field
  bit-reinterpreted as **uint16** symbols (not uint32 — see below).
  BUILT+GATED, bit-exact on 3/3 smoke workloads.

  **Real bug found**: the uint32 entry points have a genuine 5-bit
  metadata-field overflow (`temp_rate` needs 6 bits, 0..32, but is packed
  into a 5-bit field; `temp_rate==32` aliases to 0) that silently drops
  compressed data for any 64-element block containing a value `>= 2**31`
  — triggered immediately by any negative float's bit pattern, never by
  lsCOMP's own light-source-photon-count target data. Root-caused down to
  the exact source lines and reproduced in isolation
  (`lscomp/debug_roundtrip.py`); worked around (not patched) by using the
  uint16 entry points instead, which the same reproduction script confirms
  are immune (max value ≤16 bits, never approaches the 32-bit collision
  boundary). Full accounting in `lscomp/STATUS.md`.

- **`opthuffdec`**: `opt-gap-array` variant (one of the paper's two
  optimized decoders). The artifact ships its own usable reference encoder
  (`llhuff::LLHuffmanEncoder`), so no independent encoder was needed.
  fp32 field bit-reinterpreted as uint16 cuSZ-quant-code-style symbols.
  BUILT+GATED, bit-exact on 3/3 smoke workloads, ratio ~1.15x (high-entropy
  bit-reinterpreted floats — not comparable to the paper's own
  quant-code-native ratios, disclosed in the adapter).
