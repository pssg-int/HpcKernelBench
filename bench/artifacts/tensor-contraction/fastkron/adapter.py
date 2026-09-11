"""
FastKron adapter for the tensor-contraction track.

Paper: "Fast Kronecker Matrix-Matrix Multiplication on GPUs" (PPoPP'24;
`conf/ppopp/JangdaY24` in output/included.json).
Artifact: https://github.com/abhijangda/fastkron

FastKron's core primitive is Generalized Matrix-Kronecker-Matrix
Multiplication (GeMKM): Z = alpha * X @ (F_1 (x) F_2 (x) ... (x) F_N) +
beta * Y, for X (M, P^N) and N Kronecker factors F_i (P, Q). The spec's own
`kron-matmul-kernel-fp32-fp64` variant IS this exact primitive.

## Why this adapter maps onto the GENERIC 2-operand contraction, not the
## paper's own N-factor microbenchmark grid

`kernelbench.domains.tensor` (the harness's tensor-contraction domain
module) implements exactly ONE workload/reference pathway --
`ContractionWorkload`, a strictly 2-OPERAND einsum contraction
(`A[in_a],B[in_b]->C[out]`) with no notion of "N-factor Kronecker chain" at
all. `runner.py` always builds this SAME workload type (from
`smoke_workloads(kernel="tensor-contraction")` or `load_workload(name)`)
regardless of which spec variant id is passed on the command line, and gates
every impl against `reference_tensor_contraction`'s `np.einsum(w.equation,
A, B)` on THOSE SAME two operands. There is no way for an artifact adapter
to inject a genuine multi-factor Kronecker-chain workload into this
harness's tensor-contraction path -- the domain module structurally cannot
represent one (see its own docstring: "multi-operand contraction trees ...
out of scope").

The mapping used here is FastKron's own **N=1 case**: for N=1, GeMKM
degenerates to Z = alpha * X @ F + beta * Y -- an ordinary GEMM,
mathematically IDENTICAL to any 2-operand einsum contraction after reducing
it to (free_a, contracted) x (contracted, free_b) via transpose+reshape --
exactly what `kernelbench.domains.tensor.NumpyTensordotContraction` already
does for the CPU baseline in this same domain module (this adapter's
`prepare()` reuses that same axes/shape derivation verbatim). This is a
real, non-degenerate invocation of FastKron's own compiled
`sgemkm`/`dgemkm` CUDA kernel -- not a relabeling trick and not calling
cuBLAS directly under FastKron's name (see "Avoiding a real FastKron
packaging bug" below for how this is confirmed).

**Scope caveat, stated plainly**: this does NOT exercise FastKron's actual
novel contribution (the shuffle-avoiding N>1 Kronecker-CHAIN algorithm,
Table 1's up-to-80%-of-time transpose elimination for baselines). It
exercises FastKron's real N=1 GEMM kernel, correctly, on the same 2-operand
contraction shape every other impl in this domain is gated on. A genuine
multi-factor benchmark of FastKron's headline contribution is not reachable
through this harness's tensor-contraction domain module as it stands; see
STATUS.md for a standalone (non-harness-gated) N=3 functional self-check
that DOES exercise the real multi-factor kernel path and confirms it works.

The transpose/reshape glue (mapping the equation's arbitrary index order
onto FastKron's expected (free, contracted) x (contracted, free) GEMM
layout) is done via torch ops INSIDE `run()` -- i.e. included in-kernel, per
this track's own `notes_on_fairness` ("permutation is the one point of
universal agreement... never strip it out of a timed kernel"). It is
explicitly NOT part of FastKron's own zero-transpose contribution (that
claim is specific to N>1 chains it structurally never needs to transpose
for -- an N=1 GEMM has nothing to "avoid" in the first place; any 2-operand
contraction needs SOME reduction to canonical GEMM layout, by any
algorithm).

## Avoiding a real FastKron packaging bug (recorded in STATUS.md)

pyfastkron's `supportedProcessor()` (`FastKronBase`, `pyfastkron/
fastkronbase.py`) is:

    def supportedProcessor(self):
        return (platform.machine() in ("x86_64", "AMD64")) and self.x86 == True

i.e. it unconditionally requires the **X86** backend to have been built and
imported (`self.x86 = x86_requested and fastkronX86 is not None`),
REGARDLESS of whether the call in question is for a CUDA tensor on the CUDA
backend. This function gates not just the top-level `fastkrontorch.gemkm()`
convenience wrapper's fallback-to-shuffle decision, but also
`FastKronBase.gekmmSizes()` -- which, on a CUDA-only build (`self.x86 ==
False`), silently takes an "unsupported" branch that sets the temp-buffer
size `ts = -1` as a sentinel. That -1 then flows straight into
`x.new_empty(ts)` inside `FastKronTorch.gemkm()` and CRASHES with
`RuntimeError: Trying to create tensor with negative dimension -1`, even
when calling the instance method directly (bypassing the outer
`isSupported()`-gated convenience wrapper does NOT avoid this, since the
crash is inside a DIFFERENT method the instance method itself calls).
Confirmed independently of this integration's own build flags: this is a
genuine bug in pyfastkron's Python wrapper -- a CUDA-only build (no X86
extension at all) can never successfully call `gemkm`/`gekmm` through the
documented Python API, for ANY input, not just this adapter's.

The workaround used here (a one-line RUNTIME attribute flip on the
module-level singleton, not a source-file patch): after importing
`pyfastkron.fastkrontorch.fastkrontorch`, set `.x86 = True`. This makes
`supportedProcessor()`/`supportedSystem()` return `True` so `gekmmSizes()`
takes its correct branch and calls the real
`fastkronCUDA.libFastKron.gekmmSizes` C function for the temp-buffer sizes,
after which `xgemkm()` dispatches to `libFastKron.sgemkm`/`dgemkm` purely
via `x.device.type == "cuda"` -- `self.x86` is never actually read again for
dispatch (only for this one gate), and `fastkronX86` stays `None` throughout
(never touched, since every tensor here is CUDA-resident). Verified directly
(see STATUS.md) against both the N=1 case this adapter uses AND a genuine
N=3 multi-factor Kronecker-chain call (`torch.kron`-based reference),
confirming the real CUDA kernel runs correctly, not a coincidentally-passing
shuffle fallback.
"""

from __future__ import annotations

import os
import sys

KERNEL = "tensor-contraction"
IMPL_NAME = "fastkron-gemkm-n1"
PAPER_KEY = "conf/ppopp/JangdaY24"
PRECISIONS = ["fp32", "fp64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_RUNTIME_PKG = os.path.join(_HERE, "pyfastkron_runtime")


def _ensure_path():
    if _RUNTIME_PKG not in sys.path:
        sys.path.insert(0, _RUNTIME_PKG)


def available() -> tuple[bool, str]:
    try:
        if not os.path.isdir(os.path.join(_RUNTIME_PKG, "pyfastkron")):
            return False, f"not built: {_RUNTIME_PKG}/pyfastkron missing (run build.sh)"
        try:
            import torch
        except Exception as e:
            return False, f"torch import failed: {type(e).__name__}: {e}"
        if not torch.cuda.is_available():
            return False, "no CUDA device visible"
        _ensure_path()
        try:
            from pyfastkron.fastkronbase import fastkronCUDA
        except Exception as e:
            return False, f"pyfastkron import failed: {type(e).__name__}: {e}"
        if fastkronCUDA is None:
            return False, "pyfastkron built without a working CUDA backend (fastkronCUDA is None)"
        return True, ""
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


class FastKronGeMKM:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp32"):
        if precision not in PRECISIONS:
            raise NotImplementedError(
                f"{IMPL_NAME}: precision {precision!r} not in {PRECISIONS}")
        self.precision = precision

    def prepare(self, w, params: dict):
        _ensure_path()
        import numpy as np
        import torch
        from pyfastkron.fastkrontorch import fastkrontorch as _fkt

        # Runtime workaround for the packaging bug documented in this
        # module's docstring ("Avoiding a real FastKron packaging bug"):
        # gekmmSizes() gates its correct-sizing branch on self.x86, which is
        # False on a CUDA-only build regardless of device. Flipping this
        # attribute is enough -- self.x86 is never read again for dispatch
        # (only device_type(x) is), and fastkronX86 (None on this build) is
        # never touched since every tensor here is CUDA-resident. Idempotent;
        # safe to set on every prepare() call.
        _fkt.x86 = True
        params["fastkron_packaging_workaround"] = (
            "fastkrontorch.x86 forced True at runtime to bypass a "
            "supportedProcessor()-gated -1 temp-buffer-size bug on "
            "CUDA-only builds; see adapter.py docstring")

        np_dtype = np.float32 if self.precision == "fp32" else np.float64
        torch_dtype = torch.float32 if self.precision == "fp32" else torch.float64

        # SAME seeded generator reference_tensor_contraction's fp64 gate
        # uses (w.operands(dtype=np.float64)) -- just cast to this impl's
        # working precision, so there is no operand-generation mismatch of
        # the kind documented in bench/artifacts/spmm/insum/STATUS.md.
        A_np, B_np = w.operands(dtype=np_dtype)
        A = torch.as_tensor(A_np, device="cuda", dtype=torch_dtype)
        B = torch.as_tensor(B_np, device="cuda", dtype=torch_dtype)

        # Transpose/reshape derivation copied from kernelbench.domains.
        # tensor.NumpyTensordotContraction.prepare() (this domain's own CPU
        # baseline uses the identical axes/shape math to reduce an arbitrary
        # 2-operand contraction to a canonical GEMM) -- see module docstring.
        in_a, in_b = w.input_specs
        contracted = w.contracted_letters
        free_a = [l for l in in_a if l not in contracted]
        free_b = [l for l in in_b if l not in contracted]
        axes_a = [in_a.index(l) for l in free_a] + [in_a.index(l) for l in contracted]
        axes_b = [in_b.index(l) for l in contracted] + [in_b.index(l) for l in free_b]
        m = 1
        for l in free_a:
            m *= A.shape[in_a.index(l)]
        n = 1
        for l in free_b:
            n *= B.shape[in_b.index(l)]
        k = 1
        for l in contracted:
            k *= A.shape[in_a.index(l)]
        natural_order = free_a + free_b
        natural_shape = tuple(A.shape[in_a.index(l)] for l in free_a) + \
            tuple(B.shape[in_b.index(l)] for l in free_b)
        out_perm = [natural_order.index(l) for l in w.output_spec]

        return {"A": A, "B": B, "axes_a": axes_a, "axes_b": axes_b,
                "m": m, "n": n, "k": k, "natural_shape": natural_shape,
                "out_perm": out_perm}

    def run(self, h):
        import torch
        from pyfastkron.fastkrontorch import fastkrontorch as _fkt

        # Transpose+reshape into FastKron's expected (free, contracted) x
        # (contracted, free) GEMM layout, INSIDE run() -- glue, included
        # in-kernel per the track's permutation-always-in-kernel fairness
        # rule; NOT FastKron's own zero-transpose contribution (see module
        # docstring).
        A2d = torch.permute(h["A"], h["axes_a"]).reshape(h["m"], h["k"]).contiguous()
        B2d = torch.permute(h["B"], h["axes_b"]).reshape(h["k"], h["n"]).contiguous()

        # N=1 GeMKM: Z = A2d @ (Kronecker product of the single factor B2d)
        # = A2d @ B2d, computed by FastKron's own libFastKron.sgemkm/dgemkm.
        # Calls the FastKronTorch INSTANCE method directly, bypassing the
        # top-level fastkrontorch.gemkm() convenience wrapper's buggy
        # isSupported() gate (see module docstring).
        z, _zs = _fkt.gemkm(False, A2d, [B2d])

        out2d = z.reshape(h["natural_shape"])
        return torch.permute(out2d, h["out_perm"])

    def to_host(self, out):
        import torch
        return out.detach().contiguous().to("cpu", dtype=torch.float64).numpy()

    def timer(self):
        from kernelbench.impls.gpu_cuda import CudaEventTimer
        return CudaEventTimer()

    def free(self, h):
        import torch
        h.clear()
        torch.cuda.empty_cache()


def create(precision: str):
    return FastKronGeMKM(precision)
