"""
TurboFFT adapter (PPoPP 2025, "TurboFFT: Co-Designed High-Performance and
Fault-Tolerant Fast Fourier Transform on GPUs", conf/ppopp/WuZ0HJDDCC25).
Repo: https://github.com/shixun404/TurboFFT

First pass (2026-08-08) recorded BUILD-FAILED: include/TurboFFT.h's
ARCH_SM==80 block #includes one code-generated .cuh kernel file per logN
(1..25), but the repo as cloned only shipped generated files for logN in
{7,8,9,10}, and even those defined a stale, differently-signed kernel
(`fft_10(float2*, float2*, int)`) than the one TurboFFT.h actually calls
(`fft_radix_2<float2, N, dim_id, if_thread_ft, if_ft, if_err_injection>`,
6 args). TurboFFT is explicitly a CODE-GENERATED FFT library (see its own
readme.md's "Workflow Overview": Environment Setup -> Code Generation ->
Compilation -> Benchmarking -> Plotting) -- the first pass evidently never
ran the generator.

This second pass DOES run the generator
(source/TurboFFT/include/code_gen/scripts/fft_codegen.py) and confirms it
fixes BOTH problems: see STATUS.md for the full evidence, including a
4m8s, ZERO-error compile of the complete, unmodified include/TurboFFT.h
(all logN 1..25, both float2/double2, all 4 if_thread_ft/if_ft/
if_err_injection combinations TurboFFT.h's ARCH_SM==80 block
unconditionally instantiates) once the codegen has been run for the 8 flag
combinations run_A100.sh's own (mostly commented-out) script enumerates.

This adapter does NOT build/wrap that full header, though -- per
ARTIFACT_GUIDE.md rule 1 ("wrap the kernel, not the paper's benchmark
script"), it wraps ONLY the plain (no fault tolerance, no error injection)
forward-C2C kernel via a thin extern "C" shim (turbofft_shim.cu, this dir,
NOT part of the artifact -- same pattern as artifacts/bfs/efg/efg_shim.cu),
restricted to TurboFFT's own SINGLE-kernel-launch param-table rows (logN
1..13, N in [2, 8192]; see param_float2.csv rows 1-13, `row[1]==1`
"num decomposition stages"). Rows 14+ decompose one big 1D transform into
2-3 sequential kernel launches (Cooley-Tukey four-step) -- correctly
wrapping that needs its own scratch-buffer bookkeeping and is out of scope
for a gate-sized shim (see turbofft_shim.cu's docstring); N above 8192
therefore raises NotImplementedError here, not a silent wrong answer.

Coverage against this domain's 8-workload smoke set
(kernelbench/domains/spectral.py `_FFT_SMOKE`): TurboFFT's codegen'd
kernels are radix-2 ONLY (power-of-two N), C2C ONLY (no R2C/C2R codegen
path anywhere in the repo), and 1D-only (the "multi-dim" param-table rows
are multi-STAGE decompositions of one big 1D transform, not spatial 2D/3D
grids). Of the 8 smoke workloads this adapter therefore supports exactly 3:
  smoke-1d-pow2-c2c-fwd-oop        (N=1024, batch=4,  forward, oop)
  smoke-1d-pow2-c2c-fwd-inplace    (N=512,  batch=1,  forward, in-place)
  smoke-1d-pow2-c2c-inverse-oop    (N=1024, batch=1,  inverse, oop)
and clearly rejects (NotImplementedError, not a crash/wrong-answer) the
other 5: smoke-1d-mixedradix56-c2c-fwd-oop (N=56 not power-of-two),
smoke-1d-r2c-fwd-oop / smoke-1d-r2c-inverse-oop (real_input=True), and
smoke-2d-c2c-fwd-oop / smoke-3d-c2c-fwd-oop (ndim != 1).

`kernelbench/runner.py`'s `--smoke` path calls
`domain.smoke_workloads(kernel="fft")` unconditionally -- it is NOT
variant-aware for this domain (unlike e.g. annsearch.py's opt-in
`smoke_workloads(variant=...)`), so it always builds the FULL 8-workload
list regardless of which `--variant` was requested, and
`kernelbench/harness.py::run_variant`'s `impl.prepare()` call is not
wrapped in a try/except anywhere in the runner's per-workload loop. This
means a straight `--smoke` run over ALL 8 workloads for this impl surfaces
this adapter's NotImplementedError as an UNCAUGHT exception on the very
first unsupported workload (index 1, the mixed-radix-56 point) rather than
skipping it and continuing to the 3 supported points later in the list --
see STATUS.md "gate run" for the exact traceback observed and the
substitute gate command used instead (same precedent as
artifacts/spmv/diaq/STATUS.md: a direct small script over
`harness.run_variant`, restricted to the workloads this artifact's own
public API can do). This is a pre-existing spectral.py/runner.py
interaction, not something introduced by this adapter, and is not touched
here per the task's "do not modify kernelbench/ unless a genuine harness
bug is found" + "never touch other tracks" scope.

Direction: TurboFFT's codegen'd kernel set has NO inverse-FFT path at all
(no `ifft_radix_2` anywhere in TurboFFT.h/TurboFFT_radix_2_template.h --
confirmed by grep, see STATUS.md). `direction="inverse"` is therefore
implemented here as the standard, kernel-code-untouched identity
IFFT(x) = conj(FFT(conj(x))) / N -- pure host-side (well, device-side
elementwise conj via torch, which is metadata-only until `.conj_physical()`
materializes it) pre/post-processing around the SAME unmodified forward
kernel, not a new kernel.

Layout: "in-place" is honored the same way NumpyFFT/ScipyFFT already do in
this file's sibling classes (see their docstrings) -- TurboFFT's kernel
signature takes separate input/output device pointers (main.cu's own
`inputs[]`/`outputs[]` split), so this adapter runs out-of-place into a
scratch buffer and copies the result back into the operand buffer,
honoring the in-place BYTE-MOVEMENT convention without claiming an
allocation saving the kernel doesn't actually provide.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

KERNEL = "fft"
IMPL_NAME = "turbofft"
PAPER_KEY = "conf/ppopp/WuZ0HJDDCC25"
PRECISIONS = ["fp32"]  # this shim only wraps the float2 (fp32) codegen path; see build.sh

_HERE = os.path.dirname(os.path.abspath(__file__))
_SO_PATH = os.path.join(_HERE, "turbofft_shim.so")

# TurboFFT's own per-N threadblock batch size (param_float2.csv rows 1-13),
# duplicated from turbofft_shim.cu's TB_BS table so prepare() can validate
# `batch % tb_bs == 0` with a clear message before ever touching the GPU
# (the shim itself also checks this and returns -2, belt-and-suspenders).
_TB_BS = {1: 32, 2: 16, 3: 8, 4: 4, 5: 4, 6: 2, 7: 4, 8: 4,
          9: 1, 10: 1, 11: 1, 12: 1, 13: 1}
_MIN_LOGN, _MAX_LOGN = 1, 13


def available() -> tuple[bool, str]:
    try:
        if not os.path.exists(_SO_PATH):
            return False, f"not built: {_SO_PATH} missing (run build.sh)"
        try:
            import torch
        except Exception as e:
            return False, f"torch import failed: {type(e).__name__}: {e}"
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def _load_lib():
    lib = ctypes.CDLL(_SO_PATH)
    lib.turbofft_forward_c2c_f32.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong, ctypes.c_longlong, ctypes.c_int]
    lib.turbofft_forward_c2c_f32.restype = ctypes.c_int
    return lib


def _logn_of(n: int) -> int | None:
    if n <= 0:
        return None
    logn = n.bit_length() - 1
    return logn if (1 << logn) == n else None


class TurboFFTImpl:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision != "fp32":
            raise NotImplementedError(
                f"{IMPL_NAME}: only the float2 (fp32) codegen path is wrapped; "
                f"requested precision={precision!r}")
        self.precision = precision
        self.lib = _load_lib()

    def prepare(self, w, params: dict):
        import torch

        if w.real_input:
            raise NotImplementedError(
                f"{IMPL_NAME}: TurboFFT's codegen'd kernels are C2C only (no R2C/C2R "
                f"codegen path anywhere in the repo); workload {w.name!r} is real-valued")
        if w.ndim != 1:
            raise NotImplementedError(
                f"{IMPL_NAME}: TurboFFT's param-table 'multi-dim' rows are multi-STAGE "
                f"decompositions of one big 1D transform, not spatial grids; workload "
                f"{w.name!r} is {w.ndim}D")
        if w.direction not in ("forward", "inverse"):
            raise NotImplementedError(f"{IMPL_NAME}: unknown direction {w.direction!r}")

        n = w.dims[0]
        logn = _logn_of(n)
        if logn is None or not (_MIN_LOGN <= logn <= _MAX_LOGN):
            raise NotImplementedError(
                f"{IMPL_NAME}: N={n} is outside this shim's wrapped range "
                f"(power-of-two, logN in [{_MIN_LOGN}, {_MAX_LOGN}], i.e. N in "
                f"[{1 << _MIN_LOGN}, {1 << _MAX_LOGN}] -- TurboFFT's own "
                f"single-kernel-launch param-table rows); workload {w.name!r}")

        tb_bs = _TB_BS[logn]
        if w.batch % tb_bs != 0:
            raise NotImplementedError(
                f"{IMPL_NAME}: TurboFFT's own threadblock batch size for N={n} is "
                f"{tb_bs} (param_float2.csv row {logn}); batch={w.batch} is not a "
                f"multiple of it; workload {w.name!r}")

        # SAME operand construction as reference_fft/NumpyFFT (numpy RNG,
        # not torch's -- torch-fft's own docstring notwithstanding, using a
        # different RNG family here would transform DIFFERENT random data
        # than what reference_fft regenerates from the same seed, making
        # the correctness gate compare apples to oranges).
        rng = np.random.default_rng(params.get("seed", w.seed))
        shape = (w.batch,) + tuple(w.dims)
        re = rng.uniform(-1.0, 1.0, size=shape)
        im = rng.uniform(-1.0, 1.0, size=shape)
        x_np = (re + 1j * im).astype(np.complex64)

        x = torch.from_numpy(x_np).contiguous().cuda()
        scratch = torch.empty_like(x)
        return {"x": x, "scratch": scratch, "N": n, "bs": w.batch, "workload": w}

    def run(self, h):
        import torch

        w, x, scratch, N, bs = h["workload"], h["x"], h["scratch"], h["N"], h["bs"]
        src = x if w.direction == "forward" else torch.conj_physical(x)

        rc = self.lib.turbofft_forward_c2c_f32(
            ctypes.c_void_p(src.data_ptr()), ctypes.c_void_p(scratch.data_ptr()),
            ctypes.c_longlong(N), ctypes.c_longlong(bs), ctypes.c_int(1))
        if rc != 0:
            raise RuntimeError(f"{IMPL_NAME}: turbofft_forward_c2c_f32 failed, rc={rc}")

        result = scratch if w.direction == "forward" else torch.conj_physical(scratch) / N
        if w.layout == "in-place":
            x.copy_(result)
            result = x
        return result

    def to_host(self, out):
        import torch
        o = out.detach().to("cpu")
        o = torch.view_as_real(o)
        return o.to(torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        h.clear()


def create(precision: str = "fp32"):
    return TurboFFTImpl(precision)
