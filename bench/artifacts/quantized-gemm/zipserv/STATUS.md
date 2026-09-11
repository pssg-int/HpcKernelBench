# zipserv (ZipServ, fused decompress+GEMM path) — quantized-gemm

**Status: NOT ATTEMPTED (deferred) — the fused ZipGEMM path consumes ZipServ's compressed weight stream, whose decompress kernel fails the bit-exactness gate in `../../lossless-compression/zipserv/` (confirmed artifact bug: uint8 underflow + integer-promotion sign flip)**

- Paper: "ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware
  Lossless Compression" (ASPLOS'26). `PAPER_KEY = conf/asplos/FanYPLLW0026`.
  Rated core / regime partial for quantized-gemm (lossless bit-exact weight
  compression + fused decompress-GEMM, not integer quantization).
- `source/` is a symlink to the lossless-compression clone (same repository,
  same commit; see `source.provenance`); its build exists there.
- To attempt: wrap `BF16TripleBitmap_MM_API` (fused path) with a new bridge,
  gate against `ml.reference_qgemm` via the `dequantized_W_override` hook
  carrying the ORIGINAL bf16 weights (lossless ⇒ exact values). Expect the
  gate to fail wherever the decompress bug corrupts a block; that would be the
  same finding as the lossless-compression track's, recorded once already.
