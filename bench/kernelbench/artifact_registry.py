"""
Discover paper-artifact adapters under bench/artifacts/<kernel>/<name>/adapter.py
and expose them to the runner as additional implementations.

An adapter that is present but unbuilt (available() -> (False, reason)) is
listed with its reason instead of silently vanishing — partial integration
status is part of the trial's output, not something to hide.
"""

from __future__ import annotations

import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS = os.path.normpath(os.path.join(HERE, "..", "artifacts"))


def _pin_toolchain() -> None:
    """Adapters may import torch extensions or runtime-JIT DSLs on import or
    in available(); make sure CUDA_HOME/PATH already point at the toolkit
    matching torch's CUDA major (idempotent; see env.pin_cuda_toolchain)."""
    try:
        from kernelbench import env
        env.pin_cuda_toolchain()
    except Exception:
        pass


def _load_adapter(path: str):
    spec = importlib.util.spec_from_file_location(
        f"artifact_adapter_{abs(hash(path))}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover(kernel: str) -> dict:
    """
    Return {impl_name: {"factory": callable(precision)->Implementation,
                        "paper_key": str, "precisions": [..], "path": str}}
    for every AVAILABLE adapter of this kernel.
    Unavailable ones land in discover_status() instead.
    """
    _pin_toolchain()
    out = {}
    kdir = os.path.join(ARTIFACTS, kernel)
    if not os.path.isdir(kdir):
        return out
    for short in sorted(os.listdir(kdir)):
        ap = os.path.join(kdir, short, "adapter.py")
        if not os.path.exists(ap):
            continue
        try:
            mod = _load_adapter(ap)
            ok, _reason = mod.available()
            if ok:
                out[mod.IMPL_NAME] = {
                    "factory": mod.create,
                    "paper_key": getattr(mod, "PAPER_KEY", ""),
                    "precisions": getattr(mod, "PRECISIONS", ["fp32"]),
                    "path": os.path.dirname(ap),
                }
        except Exception:
            # a broken adapter must not take the runner down; it shows up in
            # discover_status() with its error
            continue
    return out


def discover_status(kernel: str | None = None, isolate: bool = True) -> list[dict]:
    """Full status of every adapter (available or not), for reporting.

    With `kernel=None` (a whole-registry scan) each kernel directory is scanned
    in its OWN Python subprocess when `isolate` is true. Reason (2026-09-06):
    adapters put artifact-local packages first on sys.path (e.g. two different
    `hidet` packages: upstream for convolution/hidet, the HexCute fork for
    gemm/hexcute); scanned in one process, whichever imports first wins and
    the other's available() reports a false negative -- and two runtime
    libraries of the same name loaded into one process aborted the
    interpreter at exit (`free(): invalid pointer`). The runner itself only
    ever loads ONE kernel's adapters per process (discover(kernel)), so this
    only concerned reporting; per-kernel subprocesses make the report match
    what a real run sees. A single-kernel call stays in-process.
    """
    if kernel is None and isolate:
        return _discover_status_isolated()
    _pin_toolchain()
    rows = []
    kernels = [kernel] if kernel else (
        sorted(os.listdir(ARTIFACTS)) if os.path.isdir(ARTIFACTS) else [])
    for k in kernels:
        kdir = os.path.join(ARTIFACTS, k)
        if not os.path.isdir(kdir):
            continue
        for short in sorted(os.listdir(kdir)):
            ap = os.path.join(kdir, short, "adapter.py")
            row = {"kernel": k, "name": short, "impl": "", "available": False,
                   "reason": ""}
            if not os.path.exists(ap):
                row["reason"] = "no adapter.py"
                rows.append(row)
                continue
            try:
                mod = _load_adapter(ap)
                ok, reason = mod.available()
                row["impl"] = getattr(mod, "IMPL_NAME", "?")
                row["available"] = bool(ok)
                row["reason"] = reason
            except Exception as e:
                row["reason"] = f"adapter error: {type(e).__name__}: {e}"
            rows.append(row)
    return rows


def _discover_status_isolated() -> list[dict]:
    """One subprocess per kernel directory; each runs discover_status(kernel)
    in-process and prints JSON. Failures of a whole subprocess are reported as
    rows too, never swallowed."""
    import json
    import subprocess
    import sys
    rows: list[dict] = []
    kernels = sorted(os.listdir(ARTIFACTS)) if os.path.isdir(ARTIFACTS) else []
    bench_root = os.path.normpath(os.path.join(HERE, ".."))
    for k in kernels:
        if not os.path.isdir(os.path.join(ARTIFACTS, k)):
            continue
        code = ("import json, sys; sys.path.insert(0, %r); "
                "from kernelbench import artifact_registry as ar; "
                "print(json.dumps(ar.discover_status(%r, isolate=False)))" % (bench_root, k))
        try:
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                               timeout=600, cwd=bench_root)
            # the JSON is the LAST stdout line (adapters may print during import)
            out = [ln for ln in r.stdout.splitlines() if ln.startswith("[")]
            if out:
                rows.extend(json.loads(out[-1]))
                continue
            reason = (r.stderr.strip().splitlines() or ["no output"])[-1][:200]
        except subprocess.TimeoutExpired:
            reason = "scan timed out (600 s)"
        except Exception as e:  # noqa: BLE001
            reason = f"{type(e).__name__}: {e}"
        for short in sorted(os.listdir(os.path.join(ARTIFACTS, k))):
            if os.path.exists(os.path.join(ARTIFACTS, k, short, "adapter.py")):
                rows.append({"kernel": k, "name": short, "impl": "?", "available": False,
                             "reason": f"scan subprocess failed: {reason}"})
    return rows
