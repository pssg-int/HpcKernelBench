"""
cuSPARSE ILU(0)-preconditioned CG -- the vendor-library BASELINE for SPCG
(SC'25) in the to-convergence group (`cg-e2e-ilu0-to-convergence`).

SPCG's own claim is "faster than the same ILU(0) PCG without sparsification",
and that un-sparsified solver is cuSPARSE csrilu02 + SpSV + SpMV, i.e. NVIDIA's
conjugateGradientPrecond sample. This is that solver, with the spec's rules
instead of SPCG's hard-coded ones:
  - stop at RELATIVE residual ||r|| / ||b|| < 1e-6, cap 2000 steps (SPCG
    stops on an absolute squared residual < 1e-24 with a 1000 cap, which in
    fp32 it almost never reaches);
  - ILU(0) zero pivots are an error, not silent NaN;
  - every run() = fresh ILU(0) factorization + solve, timed in-process with
    CUDA events (no process start-up or file parsing). Setup and solve times
    are also recorded separately (cusparse_setup_ms_internal /
    cusparse_solve_ms_internal), matching SPCG's own "Preconditioning Time"
    vs "PCG Time" split.
fp32 is the primary precision (SPCG is fp32-only); fp64 is available.

Shares ../cusparse-cg/bridge.so and its CusparseCG class; build that first.
"""
from __future__ import annotations

import importlib.util
import os

KERNEL = "cg-krylov"
IMPL_NAME = "cusparse-ilu0-cg"
PAPER_KEY = "conf/sc/MaACSH25"   # SPCG: speedup over non-sparsified ILU(0) PCG
ROLE = "baseline"
PRECISIONS = ["fp32", "fp64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_ADAPTER = os.path.join(_HERE, "..", "cusparse-cg", "adapter.py")


def _base():
    spec = importlib.util.spec_from_file_location("kb_cusparse_cg_base", _BASE_ADAPTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def available() -> tuple[bool, str]:
    try:
        return _base().available()
    except Exception as e:  # available() must never raise
        return False, f"{type(e).__name__}: {e}"


def create(precision: str):
    base = _base()

    class CusparseIlu0CG(base.CusparseCG):
        name = IMPL_NAME
        PRECOND = 1
        MODE = "to-convergence"

    return CusparseIlu0CG(precision)
