"""
PERKS's own non-persistent baseline: the same cg_perks.exe binary as
../perks, run with `--baseline` instead of `--cmat --cvec`.

PERKS (ICS'23) claims that turning an iterative solver into one persistent
kernel that keeps the matrix/vectors in on-chip memory (registers + shared
memory) makes it faster. `--baseline` is the artifact's own conventional
version of the same CG (no persistent kernel, no caching), compiled into the
same binary. Running both on the same input isolates exactly what PERKS
adds, independent of how good any outside library is. Same protocol, b,
gate and timers as ../perks (see that adapter's docstring); only
perks_variant differs.

Shares ../perks's build (no separate build step; build.sh checks it).
"""
from __future__ import annotations

import importlib.util
import os

KERNEL = "cg-krylov"
IMPL_NAME = "perks-cg-baseline"
PAPER_KEY = "conf/ics/ZhangWCMWEM23"
ROLE = "baseline"
PRECISIONS = ["fp64", "fp32"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_BASE_ADAPTER = os.path.join(_HERE, "..", "perks", "adapter.py")


def _base():
    spec = importlib.util.spec_from_file_location("kb_perks_base", _BASE_ADAPTER)
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

    class PerksCgBaseline(base.PerksCgFixedIter):
        name = IMPL_NAME
        DEFAULT_VARIANT = "baseline"

    return PerksCgBaseline(precision)
