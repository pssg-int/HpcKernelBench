# PackKV — STATUS

**Outcome: SKIPPED — no general GPU float-array compressor API**

Paper: "PackKV: Reducing KV Cache Memory Footprint through LLM-Aware Lossy
Compression" (IPDPS 2026, arXiv 2512.24449). PAPER_KEY = `conf/ipps/JiangYLHDJ26`.
Repo: `https://github.com/BoJiang03/PackKV`
(commit `da3c00d986d1e37b5c3a7e2ce9403bbf2ab6397f`, 2026-01-30;
`git clone --depth 50` into `./source/`).

## Why SKIPPED (evidence)

Inspected the CUDA extension's full exported surface
(`packkv_cuda_ext/export_packkv.cpp`, the pybind11 module, plus the headers
it binds against — `include/kernels.h`, `include/fused_kernels.h`,
`include/cpu_compress.h`):

```
m.def("kq_mat_vec_mul", ...);   // matrix-vector mult reading compressed K
m.def("wv_mat_vec_mul", ...);   // matrix-vector mult reading compressed V
m.def("k_encode_cpu",   ...);   // -> k_encode_cpu_pyi   (CPU, cpu_compress.h)
m.def("k_decode_cpu",   ...);   // -> k_decode_cpu_pyi   (CPU, cpu_compress.h)
m.def("v_encode_cpu",   ...);   // -> v_encode_cpu_pyi   (CPU, cpu_compress.h)
m.def("v_decode_cpu",   ...);   // -> v_decode_cpu_pyi   (CPU, cpu_compress.h)
m.def("fused_kq",       ...);   // -> fused_kq_launcher  (GPU, fused_kernels.h)
m.def("fused_wv",       ...);   // -> fused_wv_launcher  (GPU, fused_kernels.h)
```

The actual lossy-compression **encode** step (`k_encode_cpu_pyi`,
`v_encode_cpu_pyi` in `include/cpu_compress.h`) is CPU-only by construction
— the function names, the file it lives in (`cpu_compress.h`/
`src/k_encode_cpu.cpp`/`src/v_encode_cpu.cpp`), and the pybind binding all
agree there is no GPU counterpart. The GPU-side kernels
(`kq_mat_vec_mul`/`wv_mat_vec_mul`, `include/kernels.h`,
`src/kernels.cu`) do not compress or decompress an array to/from a
standalone buffer at all — they read the already-compressed
`compressed_buffer`/`block_info_buffer` directly inside a fused
matrix-vector multiply against attention Q/W tensors (`kq_mat_vec_mul(...,
q, kq_out, ...)`, `wv_mat_vec_mul(..., w, wv_out, ...)`). `fused_kq`/
`fused_wv` (`include/fused_kernels.h`) are a *different*, simpler
zero-point/scale quantization path (`k_quant_zero`, `k_quant_scale`) fused
directly into the KQ/WV matvec, again never materializing a decompressed
array on the GPU.

So there is no `compress(D, eb) -> C` / `decompress(C) -> D'` GPU kernel
pair anywhere in this artifact: compression happens on CPU, and everything
on the GPU is a fused decode-and-multiply specific to transformer
attention's K/V cache layout (parameters are `ctx_len`, `hidden_dim`,
`ctx_len_block_size`, `hidden_dim_block_size`, `bits_len` — an attention
tensor block layout, not an n-D scientific field with an abs/rel error
bound). This is exactly the case the integration brief anticipated
("KV-cache-specific; only if a general float-array compressor API exists;
else SKIP with evidence") — no such general API exists here, on GPU or
otherwise (even the CPU encode API is KV-cache-block-shaped, not a general
n-D array API).

No build attempted (cheap skip per ARTIFACT_GUIDE.md rule 7 — decision made
from header/binding inspection alone).

## Verdict

`packkv: SKIPPED (compression encode is CPU-only; GPU kernels are fused attention decode+matvec, not a general compress(D,eb)->C API)`
