# FFCz — STATUS

**Outcome: SKIPPED — does not implement the track's kernel standalone**

Paper: "FFCz: Fast Fourier Correction for Spectrum-Preserving Lossy
Compression of Scientific Data" (arXiv 2601.01596, 2026).
PAPER_KEY = `conf/ipps/RenUDKLYCG26`.
Repo: `https://github.com/rcrcarissa/FFCz`
(commit `96649b36b2a155075682d86fbabeb925e7d9e29a`, 2026-06-13;
`git clone --depth 50` into `./source/`).

## Why SKIPPED (evidence)

FFCz is **not** a standalone error-bounded compressor implementing
`compress(D, eb) -> C`. Per its own README ("Correction (editing) stage"):

```
ffcz (-f | -d) -i <original_path> -e <base_path> -z <compressed_path> \
     -M REL <epsilon> -F REL <delta> (-1 N | -2 Nx Ny | -3 Nx Ny Nz)
```

`-e <base_path>` is documented in `GPU/main.cu` (`--help` text, line 33-34)
as "the reconstructed file of base compressor" — i.e. FFCz requires the
**already-decompressed output of some other, external compressor** (the
paper uses SZ3/ZFP/SPERR as "base" compressors) as a mandatory input
alongside the original data. FFCz's own algorithm (`projection_algorithm.cu`,
`decompression.cu`) is a dual-domain (spatial+frequency) *correction* pass
applied on top of that base reconstruction — it projects the base
compressor's *existing* spatial error vector into the intersection of the
spatial and frequency error-bound regions, then re-encodes the correction
(Huffman+ZSTD, `HuffmanZSTDCoder.cu`) into the final compressed stream. It
does not compress the original data from scratch.

Checked `GPU/main.cu` end-to-end (grep for `SZ3`/`ZFP`/`SPERR`/`baseDecompFile`
usage): no base compressor is vendored or invoked internally — the base
decompressed file must be produced by a separate external tool not present in
this repo, then handed to `ffcz` as a file path. Wrapping FFCz's kernel for
this benchmark would therefore require standing up and calling a *second*
artifact (SZ3, ZFP, or SPERR) first, none of which are vendored here, and
even then the "kernel" under test would be a two-tool pipeline, not FFCz's
own compress-from-scratch path — which does not exist.

This matches ARTIFACT_GUIDE.md rule 7 ("If the artifact genuinely does not
implement the track's kernel ... mark SKIPPED with evidence and move to the
next candidate"): FFCz's actual contribution is a correction/refinement
stage layered on an external base compressor's output, not a general
float-array `compress(D, eb) -> C` API.

No build attempted (cheap skip per rule 7 — decision made from README +
`main.cu` inspection alone, no `cmake`/`nvcc` invocation needed to reach this
conclusion).

## Verdict

`ffcz: SKIPPED (requires external base-compressor output as input; not a standalone compress(D,eb)->C kernel)`
