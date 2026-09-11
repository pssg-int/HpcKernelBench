"""
BLCO adapter for the mttkrp track.

Paper: "Efficient, Out-of-Memory Sparse MTTKRP on Massively Parallel
Architectures" (ICS'22). `PAPER_KEY = conf/ics/NguyenHCLTSRPC22` (matched by
title in `../../../output/included.json`).
Artifact: https://github.com/jeewhanchoi/blocked-linearized-coordinate

BLCO (Blocked Linearized CoOrdinate) is the paper's own sparse-tensor
storage format for GPU MTTKRP: nonzero coordinates are bit-interleaved into
one linearized ALTO-style index (source/include/alto.hpp's `gen_alto`), then
re-split into fixed-size, GPU-resident "blocks" (source/include/blco.hpp).

## Kernel entry point wrapped (ARTIFACT_GUIDE.md rule 1)

`source/src/main.cpp`'s own `-p`/`--bench` CLI path (BLCO's MTTKRP
benchmark mode) is: `gen_blcotensor_host<LIType>(X, max_block_size)` ->
`CreateKruskalModel` + `KruskalModelRandomInit` -> `mttkrp_alto_dev<IType>`,
where `mttkrp_alto_dev` is itself just a thin loop around
`mttkrp_alto_dev_onemode<IType>` (source/include/alto_dev.hpp: "Single
MTTKRP execution, used by both benchmark MTTKRP and CPD driver") called
`iters` times with CUDA-event timing around EACH call. `wrapper.cu` (this
directory, NOT under `source/` -- see its header comment for exact
provenance of every call) exposes this exact sequence as 5 `extern "C"`
functions, letting this adapter call `mttkrp_alto_dev_onemode<IType>`
DIRECTLY, exactly once per `run()`, instead of going through BLCO's own CLI
argument parsing / `.tns` file I/O / iters-loop.

`wrapper.cu` touches ZERO lines of `source/`; every function it calls
(`gen_blcotensor_host`, `gen_blcotensor_device`, `send_blcotensor_over`,
`send_masks_over`, `make_device_copy(KruskalModel*)`,
`mttkrp_alto_dev_onemode<IType>`, `CreateSparseTensor`, `CreateKruskalModel`)
is already-defined, unmodified artifact code.

## Kernel variant chosen: BLCO Level-1 (kernel_id=1), not AUTO(10)

BLCO ships 4 on-format MTTKRP kernel variants (`-k 1/2/3/13`) plus an
"automatic" selector (`-k 10`, the README's own recommended default) that
picks Level-3 (a partial-matrix/tree-reduction kernel, needs
`create_da_mem_dev`/`zero_partials_Async`/`partial_matrix_reduction`
plumbing) whenever the target mode's length is <= 100, else Level-1. This
adapter always uses **Level-1** (`kernel_id=1`,
`mttkrp_lvl1<IType>`/`mttkrp_lvl1_3d_kernel`/`mttkrp_lvl1_4d_kernel`) --
BLCO's simplest, always-valid GPU kernel over the BLCO format, needing no
partial-matrix allocation/reduction regardless of tensor shape. This is a
choice AMONG the artifact's own already-implemented kernel variants (like
picking `torch-mttkrp` vs a hypothetical `torch-mttkrp-shared-mem` would
be), not an avoidance of the paper's contribution: BLCO's actual claim is
the FORMAT (blocked linearized coordinate storage + the resulting
uniform-across-modes traversal), which Level-1 exercises identically to
Level-3/AUTO -- only the per-block reduction strategy differs. Level-3/AUTO
were skipped here purely to bound this integration's scope (see STATUS.md);
a future adapter revision could add `kernel_id` as a `params` knob.

## Preprocessing split (rule 2): prepare() = COO -> BLCO -> GPU

`prepare()` calls, in order: `blco_build()` (host `SparseTensor` ->
`gen_blcotensor_host<IType>`, i.e. ALTO linearization+sort followed by
BLCO's own relinearize+block pass -- THIS is BLCO's format conversion, the
artifact's own preprocessing), `blco_set_factor()` once per non-target mode
(copies this workload's own seeded factor matrices in, see "Correctness
gate" below), then `blco_upload()` (host->device transfer of the BLCO
tensor + factor matrices, `gen_blcotensor_device`+`send_blcotensor_over`+
`send_masks_over`+`make_device_copy(KruskalModel*)` -- matches every other
GPU adapter in this harness building its device-resident operands inside
prepare()). `run()` then calls ONLY `blco_run()` (one
`mttkrp_alto_dev_onemode<IType>` call) -- the timed kernel is pure MTTKRP
compute over an already-resident BLCO tensor, no data movement.

## In-memory path (task brief: "use the in-memory path if available")

BLCO's own out-of-memory streaming mode (`--stream-data`, `stream_data=
true`) re-transfers each block from host to device fresh on EVERY
`mttkrp_alto_dev_onemode` call (source/src/alto_dev.cu line ~1985:
`if (stream_data) send_block_over(...)`), trading device memory for
per-call PCIe transfer -- the paper's actual headline contribution for
tensors that overflow GPU memory (Amazon/Patents/Reddit in Table 2). This
adapter wraps the **in-memory (non-streaming) path** instead: `blco_build()`
uses `max_block_size = nnz` (ONE block, matching
`source/src/main.cpp`'s own `if (!stream_data) max_block_size = X->nnz;`)
and `blco_upload()`/`blco_run()` both hardcode `stream_data=false` --
matching `mttkrp_alto_dev`'s own non-streaming branch exactly (num_streams =
at_host->block_count, one synchronous `send_blcotensor_over` upfront, never
re-transferred). This is the correct choice for this integration's
single-GPU/synthetic-smoke scope (ARTIFACT_GUIDE.md's own scope ruling), but
means the wrapper does NOT exercise BLCO's out-of-memory contribution --
recorded plainly in STATUS.md, not hidden.

## Correctness gate (rule 4): matches the harness's OWN seeded factors

`kernelbench.domains.tensor.reference_mttkrp` generates its factor matrices
via `_make_factors(w.shape, R, seed, np.float64)` (`np.random.default_rng
(seed + m).uniform(-1, 1, ...)` per mode). This adapter calls the SAME
`_make_factors` helper (imported directly, not reimplemented) so its own
factor matrices are numerically IDENTICAL to the reference's, then feeds
them into BLCO's `KruskalModel` via `blco_set_factor()` -- BLCO's own
`KruskalModelRandomInit` (a `rand_r`-based generator, source/src/
kruskal_model.cpp) is never called. This matters: `kernelbench.impls.
gpu_cuda.TorchMTTKRP` (this domain's OWN CUDA wiring, "never instantiated/
run" per its docstring) generates factors via a DIFFERENT RNG (`torch.
Generator`/`torch.rand`, `gpu_cuda._dense_nd`) than `_make_factors`'s numpy
`default_rng` -- had it ever been run through the gate as-is, its MTTKRP
output would be computed against DIFFERENT factor values than
`reference_mttkrp` used, and the gate would fail for a reason having
nothing to do with MTTKRP correctness. Worth flagging as a latent
harness-side bug (not a BLCO artifact bug); this adapter avoids it by
construction.

PRECISION: fp64 only. BLCO's `FType` (source/include/common.hpp) is
compile-time `double` throughout (`#if 1 ... typedef double FType ...`, the
`#else float` branch is dead code, never selected by any build flag) --
matches the paper's own fp64 usage (survey.md: "double-precision values,
64-bit integers") and the spec's fp64 default; `PRECISIONS = ["fp64"]`.

MODE / MAX_NUM_MODES: this build targets `cpd64` (`ALTO_MASK_LENGTH=64`,
`LIT=IType=unsigned long long` throughout, matching BLCO's default `make`
target) with `MAX_NUM_MODES=5` -- sufficient for every workload in this
harness's `mttkrp` smoke set (order 3-4) and the general-kernel variant's
FROSTT recommended_subset (order 3-4). A real FROSTT tensor whose modes need
>64 total linearization bits (e.g. nell-1, per BLCO's own paper Table 2)
would need the `cpd128` build (`LIT=unsigned __int128`) -- not built here,
see STATUS.md.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "mttkrp"
IMPL_NAME = "blco-mttkrp-lvl1"
PAPER_KEY = "conf/ics/NguyenHCLTSRPC22"
PRECISIONS = ["fp64"]

_KERNEL_ID_LVL1 = 1   # see module docstring: "Kernel variant chosen"

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB_PATH = os.path.join(_HERE, "libblco_wrapper.so")

_lib = None


def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(_LIB_PATH):
        raise FileNotFoundError(f"{_LIB_PATH} not built -- run build.sh")
    lib = ctypes.CDLL(_LIB_PATH)
    p = ctypes.c_void_p
    u64p = ctypes.POINTER(ctypes.c_uint64)
    f64p = ctypes.POINTER(ctypes.c_double)
    i32 = ctypes.c_int
    i64 = ctypes.c_int64

    lib.blco_build.argtypes = [i32, u64p, i64, u64p, f64p, i64, i32, i32]
    lib.blco_build.restype = p
    lib.blco_set_factor.argtypes = [p, i32, f64p]
    lib.blco_set_factor.restype = None
    lib.blco_upload.argtypes = [p]
    lib.blco_upload.restype = None
    lib.blco_run.argtypes = [p]
    lib.blco_run.restype = None
    lib.blco_get_output.argtypes = [p, f64p]
    lib.blco_get_output.restype = None
    lib.blco_free.argtypes = [p]
    lib.blco_free.restype = None
    _lib = lib
    return _lib


def available() -> tuple[bool, str]:
    if not os.path.exists(_LIB_PATH):
        return False, "libblco_wrapper.so not built -- run build.sh"
    try:
        import torch
    except Exception as e:
        return False, f"torch import failed: {type(e).__name__}: {e}"
    if not torch.cuda.is_available():
        return False, "CUDA not available on this host"
    try:
        _load_lib()
    except Exception as e:
        return False, f"ctypes load failed: {type(e).__name__}: {e}"
    return True, ""


def create(precision: str):
    return BlcoMTTKRP(precision)


def _as_u64p(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.uint64)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def _as_f64p(arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr, arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


class BlcoMTTKRP:
    """
    prepare() = COO -> BLCO (host) -> GPU upload (see module docstring for
    the full breakdown); run() = ONE mttkrp_alto_dev_onemode<IType> call
    (BLCO Level-1, in-memory path); to_host() = D2H copy of the (now
    MTTKRP-result-holding) target-mode factor buffer.
    """

    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"{IMPL_NAME} only wired for fp64 (BLCO's FType is 'double' "
                f"throughout, no other value type); requested {precision}")
        self.precision = precision
        self._dim_mode = 0
        self._rank = 0

    def prepare(self, w, params: dict):
        # Local import: kernelbench.domains.tensor.DEFAULT_RANK / _make_factors
        # -- reused verbatim (not reimplemented) so this adapter's factor
        # matrices are numerically identical to reference_mttkrp's own, see
        # module docstring's "Correctness gate" section.
        from kernelbench.domains import tensor as tensor_domain

        lib = _load_lib()

        R = int(params.get("R", tensor_domain.DEFAULT_RANK))
        params["R"] = R
        mode = int(params.get("mode", 0)) % w.order
        params["mode"] = mode
        seed = params.get("seed", 42)

        factors = tensor_domain._make_factors(w.shape, R, seed, np.float64)

        dims, dims_p = _as_u64p(np.asarray(w.shape, dtype=np.uint64))
        indices, indices_p = _as_u64p(w.indices)          # (nnz, order), row-major
        values, values_p = _as_f64p(w.values)

        # --- artifact's own COO -> BLCO conversion, timed as preprocessing --
        handle = lib.blco_build(w.order, dims_p, int(w.nnz), indices_p,
                                values_p, R, mode, _KERNEL_ID_LVL1)

        for m in range(w.order):
            if m == mode:
                continue
            fm, fm_p = _as_f64p(factors[m])
            lib.blco_set_factor(handle, m, fm_p)

        # --- host -> device transfer, still preprocessing (rule 2) ---------
        lib.blco_upload(handle)

        self._dim_mode = int(w.shape[mode])
        self._rank = R
        return {"handle": handle}

    def run(self, h):
        lib = _load_lib()
        lib.blco_run(h["handle"])
        return h["handle"]

    def to_host(self, out):
        lib = _load_lib()
        buf = np.empty(self._dim_mode * self._rank, dtype=np.float64)
        lib.blco_get_output(out, buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return buf.reshape(self._dim_mode, self._rank)

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        lib = _load_lib()
        lib.blco_free(h["handle"])
        h.clear()
        import torch
        torch.cuda.empty_cache()
