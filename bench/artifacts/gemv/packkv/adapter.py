"""
PackKV adapter for the gemv track.

Paper: "PackKV: Reducing KV Cache Memory Footprint through LLM-Aware Lossy
Compression", IPDPS'26 (arXiv 2512.24449). `PAPER_KEY` set below from
`kernel-papers/output/included.json`.
Artifact: https://github.com/BoJiang03/PackKV -- an ahead-of-time CUDA
extension (source/packkv_cuda_ext/) implementing (1) a bit-packed encoder
for INT4-quantized KV-cache blocks (`k_encode_cpu`/`k_decode_cpu`, CPU-side
bit-packing with a further adaptive per-round bit-length reduction -- see
"BUG FOUND" below for what happens outside its supported shapes) and (2) a
fused decompression+dot-product kernel that reads the COMPRESSED buffer
directly and produces the K^T@q attention-score dot product without ever
materializing a decompressed dense K matrix (`kq_mat_vec_mul` + `fused_kq`).

VARIANT: `gemv-e2e-compressed-format` (benchspecs/gemv/spec.yaml) -- "GEMV
embedded in a compressed/structured storage format where decompression is
NOT separable from the per-call kernel by the algorithm's own design". The
spec's own survey.md names this artifact for exactly this variant ("PackKV's
own kernel-vs-cuBLAS comparison times fused decompression+compute together
BY CONSTRUCTION... there is no already-decompressed state to start the
timer at"). Preprocessing here (timed once, in `prepare()`) is genuinely
just: quantize K into codes, and encode those codes into the compressed
block format (`k_encode_cpu`). The decode+dot-product chain
(`kq_mat_vec_mul` -> `fused_kq`) is the timed kernel call in `run()`,
matching this variant's protocol field ("PackKV has no separable
preprocessing to report [beyond quantize+encode]").

SHAPE MAPPING (this domain's y[M]=A[M,K]@x[K] <-> PackKV's own K^T@q): A is
the (uncompressed, per-request) K cache [ctx_len=M rows, hidden_dim=K
columns] and x is the query vector q[K] -- the literal mapping given in the
task brief, and exactly what `evaluation.py::k_cpu_compress_gpu_mat_vec_mul`
computes for one real attention head (`target_kq[h,:] = K[h,:,:] @ q[h,:]`,
verified against this file's own numpy reference below).

THE ARTIFACT'S KERNEL ONLY SUPPORTS TWO EXACT (hidden_dim) VALUES -- a real,
load-bearing constraint discovered by reading `kernels.cu` (NOT
`evaluation.py`, which never exercises an unsupported shape): both
`kq_mat_vec_mul_impl` (kernels.cu ~L564-589) and `fused_kq_launcher_impl`
(fused_kernels.cu ~L108-150) are `if/else if` ladders hardcoded to EXACTLY
`hidden_dim/head_num in {1024/8, 5120/40}` (matching two of the paper's own
eval models' KV-head counts -- Qwen3-8B has 8 KV heads, Phi-4 has 40
attention heads, both head_dim=128) with NO general case. `kq_mat_vec_mul`'s
"no such case" branch (see BUG below) and `fused_kq_launcher_impl`'s
`throw std::runtime_error("Unsupported kernel configuration")` both confirm
this is not a parameter this adapter could just pass differently -- the
GPU code genuinely does not exist for any other hidden_dim. This domain's
gemv shapes were not chosen with this floor in mind (none of `_gemv_shapes
()`'s K values equal 1024 or 5120), so an arbitrary K must be realized by
CHUNKING it into fixed HEADS_PER_CHUNK=8 (hidden_dim=1024) segments and
calling PackKV's own, UNMODIFIED per-chunk kernel chain once per chunk;
`run()` sums the chunks' partial dot products to recover the full-K
`sum_{k=0}^{K-1} A[ctx,k]*x[k]` -- elementary linear algebra (a
column-block-partitioned matrix product is the sum of its block partial
products), never a kernel edit. In real PackKV usage there is only ever ONE
chunk per call (hidden_dim already equals the model's true KV-head count);
chunking an arbitrary K into several independent 1024-wide calls is this
adapter's own boundary choice for satisfying "K cache as an A matrix, q as
the x vector" at K values no real model produces, documented here rather
than silently assumed. HEADS_PER_CHUNK=8 (not 40) is used throughout since
it is the smaller of the two supported sizes and evenly divides more of
this domain's K values without padding.

QUANTIZATION / FAIRNESS: PackKV's own K-path (`evaluation.py`'s
`quant_ints`, `QuantMethod.PackKV = (TokenQuant, TokenQuant)`,
`QUANT_DIM[TokenQuant] = [1,4]`) is an asymmetric per-row (per ctx-len
POSITION, reduced across the full per-call hidden-state width -- i.e. one
scale/zero-point PER CHUNK PER ROW here, shared across that chunk's 8
heads) affine quantizer:
    scale[row]     = (max(row) - min(row)) * quant_scale_rel
    zero_int[row]  = round(min(row) / scale[row])
    code_stored[row,c] = clip(round(A[row,c]/scale[row]) - zero_int[row], 0, qmax)
This adapter's own private `_quantize_row` reproduces this EXACT formula
(never imported from PackKV's `utils/compute.py`, and never called by
`reference_gemv` -- see below). One algebraic fact this adapter relies on,
verified by direct substitution and independently confirmed by the GPU-vs-
numpy check in this file's own inline smoke path (see STATUS.md's gate
numbers): PackKV's own reconstruction `(code_stored + zero_int) * scale`
recovers `round(A/scale) * scale` exactly -- the zero-point is a CODE-
STORAGE-only offset (keeps stored codes non-negative for nibble-packing)
that cancels out of the reconstructed VALUE. `fused_kq`'s CUDA formula
(`result = scale * (t + zero_int * sum_c(q[c]))`) is exactly this identity
distributed across the codes-dot-q computation `t` that `kq_mat_vec_mul`
produces -- so the reconstructed dot product is provably `dequant(A) @ x`
with `dequant(A) = round(A/scale)*scale`, regardless of which zero-point
convention was used to store the codes.

This scheme is NOT expressible via the marlin-precedent
`quantize_dequantize_groupwise` helper (SYMMETRIC, zero-point-free, groups
along COLUMNS -- PackKV's scheme has an explicit zero-point and groups
along ROWS: a different quantization FAMILY, not just a different axis).
Per the task brief's explicit fallback, this adapter therefore does NOT
reuse that helper; it uses the new `dense.py::reference_gemv` hook
`params["dequantized_A_override"]` instead (see that function's docstring):
an (M,K) fp64 array `reference_gemv` uses VERBATIM in place of its own
freshly-generated operand, with NO quantization logic of any kind inside
`dense.py`. This adapter computes that array with `_quantize_row` --
applied PER CHUNK, using ONLY that chunk's real (unpadded) columns, exactly
mirroring what happens on the kernel side (see PADDING below) -- and never
hands `reference_gemv` anything it would have to trust blindly: the array
is finished VALUES, computed independently of `packkv_cuda`'s actual C++/
CUDA-executed functions, which remain the real subject under test. A bug in
any of them (wrong nibble unpacking, wrong scale/zero application, wrong
q-transform permutation) changes `run()`'s real, GPU-computed output
relative to this independently-computed ground truth -- exactly what a
meaningful gate requires (DOMAIN_GUIDE.md "Reference independence" ruling).

PADDING (K-columns per chunk, and M-rows overall) -- needed because this
domain's gemv shapes were not chosen with PackKV's 1024/64 tile floors in
mind, same situation as gemv/marlin's own tile padding, but with one added
subtlety from the row-shared (not column-grouped) scale:
  - K columns within a chunk: a chunk's scale/zero are shared across the
    WHOLE 1024-column chunk width, so naively zero-padding a short final
    chunk would let the padding value become that row's new min or max,
    corrupting the REAL columns' own scale/zero too. Fixed by computing
    scale/zero from the chunk's REAL columns ONLY, then padding the STORED
    CODE array (not the raw values) with a duplicate of one of those real
    columns' own code (already a valid code in [0,qmax] by construction,
    so it perturbs nothing) and padding x with an actual 0 in the matching
    lanes -- the padded columns' contribution to the dot product is
    provably zero (multiplied by x=0) regardless of which real code was
    duplicated, while the real columns' codes (and hence dequantized
    values) are IDENTICAL to what an unpadded, exactly-1024-wide chunk
    would have produced. Proven exact, not approximate.
  - M rows: ctx_len must be an exact multiple of PackKV's own hardcoded
    `ctx_len_block_size=64` for the K path. Extra rows are filled with a
    DUPLICATE of the last real row (never all-zero -- an all-zero row
    would make that row's own max-min=0, i.e. scale=0, a divide-by-zero in
    PackKV's own formula that padding must never trigger); those rows are
    computed by the kernel like any other but sliced off in `to_host()`,
    exactly like marlin's own M-padding.

PRECISION: `PRECISIONS = ["fp16"]` -- `fused_kq_launcher`/`kq_mat_vec_mul`
only dispatch on `torch.float16`/`torch.bfloat16` (fused_kernels.cu,
kernels.cu), never a continuous fp32/fp64 K cache; this matches the spec's
own `gemv-e2e-compressed-format` dense_operand field ("PackKV: FP16/BF16").
`dense.py`'s `DEFAULT_PRECISION["gemv"]` is unconditionally "fp64" (see that
module's docstring point 3), so running this impl REQUIRES an explicit
`--precision fp16` on the CLI, same as marlin -- documented in STATUS.md's
exact verified command.

BUG FOUND IN THE ARTIFACT (see STATUS.md for the full writeup): `kernels.cu`
`kq_mat_vec_mul_impl`'s `else` branch (unsupported `hidden_dim`) does
`std::cout << "no such case"; return 0.0;` -- it neither throws nor sets an
error flag; the caller's `kq_out` buffer is left exactly as initialized
(zeros), and the immediately-following `fused_kq` call would then silently
compute a WRONG (garbage-looking-plausible, not obviously-zero) result from
that all-zero `t`, with NO exception anywhere in the chain. This adapter
therefore hard-asserts `hidden_dim in (1024, 5120)` itself before calling
`kq_mat_vec_mul` on every chunk (see `_CHUNK_HIDDEN_DIM` below), rather than
trusting the extension to fail loudly if this adapter's own chunk-sizing
logic ever had a bug.
"""

from __future__ import annotations

import os

import numpy as np

KERNEL = "gemv"
IMPL_NAME = "packkv-kq-compressed-gemv"
PAPER_KEY = "conf/ipps/JiangYLHDJ26"
PRECISIONS = ["fp16"]

HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(HERE, "vendor")

_BITS = 4                       # K path's own hardcoded bits_len (assert k_bit_len==4)
_QMAX = (1 << _BITS) - 1         # 15
_QUANT_SCALE_REL = 1.0 / _QMAX   # natural full-range 4-bit affine quantizer scale
_CTX_BLOCK = 64                  # K path's own hardcoded ctx_len_block_size
_HIDDEN_BLOCK = 128              # K path's own hardcoded hidden_dim_block_size (== head_dim)
_HEADS_PER_CHUNK = 8             # one of the ONLY two kq_mat_vec_mul/fused_kq-supported sizes
_CHUNK_HIDDEN_DIM = _HEADS_PER_CHUNK * _HIDDEN_BLOCK   # 1024
_B_OFFSET = 1_000_003            # dense.py's own arbitrary seed offset for x, replicated verbatim


def available() -> tuple[bool, str]:
    if not os.path.isdir(_VENDOR) or not any(
            f.startswith("packkv_cuda") for f in os.listdir(_VENDOR)):
        return False, "vendor/ not built -- run build.sh"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    try:
        import sys
        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)
        import packkv_cuda  # noqa: F401  -- must import AFTER torch, see build.sh
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if "CXXABI" in msg or "GLIBCXX" in msg:
            msg = ("libstdc++ ABI mismatch -- run with "
                   "LD_PRELOAD=/usr/lib64/libstdc++.so.6 (see STATUS.md)")
        return False, msg
    return True, ""


def create(precision: str):
    return PackkvGemv(precision)


def _rng_operand(rows: int, cols: int, seed: int, dtype) -> np.ndarray:
    """Verbatim copy of kernelbench.domains.dense._rng_operand (private to
    that module; replicated here per this project's own cross-adapter
    convention -- see gemv/marlin/adapter.py's identical helper)."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=(rows, cols)).astype(dtype)


def _rng_vector(n: int, seed: int, dtype) -> np.ndarray:
    """Verbatim copy of kernelbench.domains.dense._rng_vector."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=n).astype(dtype)


def _quantize_row(A_cols: np.ndarray):
    """
    This adapter's own, independent numpy re-implementation of PackKV's
    K-path affine quantizer (`evaluation.py`'s `quant_ints`, low-precision-
    zero-point branch, `QuantMode.TokenQuant`: per-row min/max reduced over
    the ENTIRE column width passed in). NEVER imported by, or shared with,
    `kernelbench.domains.dense.reference_gemv` -- see this module's
    docstring's "Reference independence" paragraph.

    `A_cols`: (rows, W) fp64, W <= _CHUNK_HIDDEN_DIM, the REAL (unpadded)
    columns of one chunk. Returns (code_stored: int64 (rows,W) in
    [0,_QMAX], zero_int: fp64 (rows,), scale: fp64 (rows,), dequant: fp64
    (rows,W) == (code_stored+zero_int)*scale == round(A_cols/scale)*scale,
    the value-level identity this module's docstring derives).
    """
    row_min = A_cols.min(axis=1)
    row_max = A_cols.max(axis=1)
    scale = (row_max - row_min) * _QUANT_SCALE_REL
    # guard only: a real (non-duplicated) row of continuous U(-1,1) data has
    # max>min almost surely; this exists so a pathological all-equal row
    # (not expected for this domain's synthetic operands) fails safe rather
    # than dividing by zero, mirroring quantize_dequantize_groupwise's own
    # amax==0 guard in dense.py.
    scale = np.where(scale == 0.0, 1.0, scale)
    zero_int = np.round(row_min / scale)
    code_math = np.round(A_cols / scale[:, None])
    code_stored = np.clip(code_math - zero_int[:, None], 0, _QMAX)
    dequant = (code_stored + zero_int[:, None]) * scale[:, None]
    return code_stored.astype(np.int64), zero_int, scale, dequant


class PackkvGemv:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp16"):
        if precision != "fp16":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp16 (PackKV's fused_kq/"
                f"kq_mat_vec_mul dispatch on torch.float16/bfloat16 only, "
                f"fused_kernels.cu/kernels.cu -- never a continuous fp32/"
                f"fp64 K cache); requested {precision!r}. dense.py's "
                f"DEFAULT_PRECISION['gemv'] is 'fp64', so pass "
                f"--precision fp16 explicitly on the CLI -- see this "
                f"adapter's module docstring / STATUS.md for the exact "
                f"verified command.")
        self.precision = precision
        self._M_true = None

    def prepare(self, workload, params: dict):
        import sys
        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)
        import torch
        import packkv_cuda as pk

        w = workload
        M, K = w.M, w.K
        device = "cuda"

        # --- regenerate A, x exactly as reference_gemv will (generate at
        # the RUN's own precision -- fp16 -- first, THEN widen to fp64; see
        # dense.py's module docstring for why this order is load-bearing).
        seed = params.get("seed", w.seed)
        A16 = _rng_operand(M, K, seed, np.float16)
        A64 = A16.astype(np.float64)
        x16 = _rng_vector(K, seed + _B_OFFSET, np.float16)
        x64 = x16.astype(np.float64)

        # --- M-row padding to PackKV's ctx_len_block_size=64 floor (see
        # module docstring): duplicate the LAST real row, never zero-pad
        # (an all-zero row makes that row's own scale=0, a genuine
        # divide-by-zero in PackKV's own formula, not something padding
        # should ever trigger).
        M_pad = ((M + _CTX_BLOCK - 1) // _CTX_BLOCK) * _CTX_BLOCK
        if M_pad > M:
            A_rowpad = np.vstack([A64, np.tile(A64[-1:, :], (M_pad - M, 1))])
        else:
            A_rowpad = A64

        # --- per-chunk (<=1024 real columns) quantization; see module
        # docstring's PADDING section for why K-column padding must reuse
        # a REAL column's own code rather than an arbitrary value.
        dequant_true = np.zeros((M, K), dtype=np.float64)
        chunks = []   # list of (code_full: (M_pad,1024) uint8, x_full: (1024,) fp16)
        col = 0
        while col < K:
            W = min(_CHUNK_HIDDEN_DIM, K - col)
            code_stored, zero_int, scale, dequant = _quantize_row(A_rowpad[:, col:col + W])
            dequant_true[:, col:col + W] = dequant[:M, :]
            if W < _CHUNK_HIDDEN_DIM:
                pad_w = _CHUNK_HIDDEN_DIM - W
                code_full = np.concatenate(
                    [code_stored, np.tile(code_stored[:, :1], (1, pad_w))], axis=1)
                x_full = np.concatenate(
                    [x64[col:col + W], np.zeros(pad_w)]).astype(np.float16)
            else:
                code_full = code_stored
                x_full = x64[col:col + W].astype(np.float16)
            chunks.append((code_full.astype(np.uint8), x_full, zero_int, scale))
            col += W

        # --- fairness hook (see module docstring): reference_gemv (called
        # AFTER prepare(), same params dict, per harness.run_variant) gates
        # against these SAME dequantized values, computed independently of
        # packkv_cuda above.
        params["dequantized_A_override"] = dequant_true

        # --- encode each chunk (this IS the artifact's own preprocessing
        # -- format conversion into its compressed block layout -- timed
        # once via prepare(), per ARTIFACT_GUIDE rule 2).
        ctx_len_block_num = M_pad // _CTX_BLOCK
        hidden_dim_block_num = _CHUNK_HIDDEN_DIM // _HIDDEN_BLOCK  # == _HEADS_PER_CHUNK
        gpu_chunks = []
        for code_full, x_full, zero_int, scale in chunks:
            assert code_full.shape == (M_pad, _CHUNK_HIDDEN_DIM)
            block = torch.from_numpy(code_full).contiguous()
            block_info_buffer = torch.zeros(
                ctx_len_block_num * hidden_dim_block_num, 2, dtype=torch.uint32)
            compressed_buffer = torch.zeros_like(block)
            pk.k_encode_cpu(
                block, compressed_buffer, block_info_buffer,
                M_pad, _CHUNK_HIDDEN_DIM, _CTX_BLOCK, _HIDDEN_BLOCK, _BITS)

            # cheap lossless-roundtrip self-check (mirrors evaluation.py's
            # own `assert (block == decompress_tensor).all()`); a real
            # encode/decode bug would otherwise surface only as a
            # downstream numeric mismatch, harder to diagnose.
            decompress_tensor = torch.zeros_like(block)
            pk.k_decode_cpu(compressed_buffer, block_info_buffer, decompress_tensor,
                             M_pad, _CHUNK_HIDDEN_DIM, _CTX_BLOCK, _HIDDEN_BLOCK, _BITS)
            if not torch.equal(block, decompress_tensor):
                raise RuntimeError(
                    f"{IMPL_NAME}: k_encode_cpu/k_decode_cpu roundtrip "
                    f"mismatch -- see STATUS.md")

            q = torch.from_numpy(x_full).to(device).view(_HEADS_PER_CHUNK, _HIDDEN_BLOCK)
            gpu_chunks.append({
                "compressed_buffer": compressed_buffer.to(device),
                "block_info_buffer": block_info_buffer.to(device),
                "q": q,
                "k_quant_zero": torch.from_numpy(zero_int.astype(np.float16)).to(device),
                "k_quant_scale": torch.from_numpy(scale.astype(np.float16)).to(device),
            })

        self._M_true = M
        self._M_pad = M_pad
        return gpu_chunks

    def run(self, chunks):
        import torch
        import sys
        if _VENDOR not in sys.path:
            sys.path.insert(0, _VENDOR)
        import packkv_cuda as pk

        M_pad = self._M_pad
        y_total = torch.zeros(M_pad, dtype=torch.float32, device="cuda")
        for c in chunks:
            q = c["q"]
            # PackKV's own nibble-packing-aligned reordering of q -- see
            # module docstring; a permutation, invariant under the sum
            # fused_kq needs and cancels correctly in the dot product
            # `kq_mat_vec_mul` computes against the SAME reordering applied
            # implicitly to how codes are packed.
            q_reshaped = q.view(_HEADS_PER_CHUNK, _HIDDEN_BLOCK // 4, 4)
            part1 = q_reshaped[:, :, :2].contiguous().view(_HEADS_PER_CHUNK, -1)
            part2 = q_reshaped[:, :, 2:].contiguous().view(_HEADS_PER_CHUNK, -1)
            q_transform = torch.cat([part1, part2], dim=1)

            hidden_dim = _CHUNK_HIDDEN_DIM
            if hidden_dim not in (1024, 5120):
                # defensive: kq_mat_vec_mul's own unsupported-shape branch
                # silently no-ops (see module docstring's BUG FOUND) rather
                # than raising -- never let that happen unnoticed.
                raise RuntimeError(
                    f"{IMPL_NAME}: hidden_dim={hidden_dim} is not one of "
                    f"kq_mat_vec_mul's two supported values (1024, 5120) "
                    f"-- see module docstring's BUG FOUND section")

            t = torch.zeros(_HEADS_PER_CHUNK, M_pad, dtype=torch.float16, device="cuda")
            pk.kq_mat_vec_mul(
                c["compressed_buffer"], c["block_info_buffer"], q_transform, t,
                M_pad, hidden_dim, _CTX_BLOCK, _HIDDEN_BLOCK, _BITS)

            our_kq = torch.empty_like(t)
            pk.fused_kq(our_kq, q, c["k_quant_zero"], c["k_quant_scale"], t)
            y_total += our_kq.sum(dim=0).float()
        return y_total

    def to_host(self, out):
        import torch
        y = out.detach().to("cpu", dtype=torch.float64).numpy()
        return y[: self._M_true]

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, chunks):
        import torch
        chunks.clear()
        torch.cuda.empty_cache()
