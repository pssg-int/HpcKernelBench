"""
Environment capture. A throughput number without the machine it came from is not
a result, so every record carries this block.
"""

from __future__ import annotations

import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:
        return ""


def capture() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "node_type": "login" if "login" in socket.gethostname() else "unknown",
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": {
            "model": next((l.split(":", 1)[1].strip()
                           for l in open("/proc/cpuinfo").read().splitlines()
                           if l.startswith("model name")), ""),
            "logical_cores": os.cpu_count(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"),
            "omp_proc_bind": os.environ.get("OMP_PROC_BIND", "unset"),
        },
        "slurm": {k: v for k, v in os.environ.items()
                  if k.startswith("SLURM_JOB") or k == "SLURM_NODELIST"},
    }
    try:
        import numpy
        info["numpy"] = numpy.__version__
    except Exception:
        pass
    try:
        import scipy
        info["scipy"] = scipy.__version__
    except Exception:
        pass
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                "count": torch.cuda.device_count(),
                "cuda_runtime": torch.version.cuda,
                "driver": _sh(["nvidia-smi", "--query-gpu=driver_version",
                               "--format=csv,noheader"]).split("\n")[0],
                "clocks_locked": None,   # set by the runner if it locks clocks
            }
    except Exception:
        info["torch"] = None
    nvcc = _sh(["nvcc", "--version"])
    if nvcc:
        info["nvcc"] = nvcc.splitlines()[-1]
    info["cuda_home"] = os.environ.get("CUDA_HOME", "unset")
    info["cuda_toolchain_pin"] = _LAST_PIN_NOTE
    return info


def warn_if_unsuitable(info: dict) -> list[str]:
    """Flag conditions under which a timing number should not be trusted."""
    w = []
    if info.get("node_type") == "login":
        w.append("running on a LOGIN node: CPU and GPU are shared with other "
                 "users — timings are indicative only, never publishable")
    if not info.get("slurm"):
        w.append("no SLURM job environment detected: no exclusive allocation")
    gpu = info.get("gpu") or {}
    if gpu and not gpu.get("clocks_locked"):
        w.append("GPU clocks not locked; boost-clock drift will inflate variance")
    rt = (gpu.get("cuda_runtime") or "").split(".")[0]
    nv = _nvcc_major_from_banner(info.get("nvcc", ""))
    if rt and nv and rt != nv:
        w.append(f"nvcc on PATH is CUDA {nv}.x but torch was built for CUDA {rt}.x: "
                 f"JIT/runtime-compiled kernels will be built with a mismatched "
                 f"toolkit (set KB_CUDA_HOME or CUDA_HOME to a {rt}.x toolkit)")
    return w


# ---------------------------------------------------------------------------
# CUDA toolchain pinning.
#
# Why this exists (2026-09-04): the login environment's default module flipped
# from cudatoolkit/12.9 to cudatoolkit/13.2 between integration batches, so
# CUDA_HOME/NVHPC_CUDA_HOME and the nvcc on PATH silently became 13.2 while the
# harness's torch is still cu128. torch.utils.cpp_extension resolves nvcc from
# CUDA_HOME first, and every runtime-JIT artifact (tilus, qfactory, hidet, the
# flashattention-t wrapper) follows the same lookup -- so an unpinned run would
# rebuild or fail against the wrong toolkit and, worse, invalidate JIT caches
# built with the right one. The harness therefore pins CUDA_HOME/PATH to a
# toolkit whose MAJOR version matches torch.version.cuda before any
# implementation is constructed, and records what it did in the env block.
#
# Lookup is bounded to one known prefix (depth-2 os.listdir of the NVIDIA HPC
# SDK root), never a filesystem search -- see the project's filesystem rules.
# ---------------------------------------------------------------------------

_HPC_SDK_ROOT = "/opt/nvidia/hpc_sdk/Linux_x86_64"   # known prefix on Perlmutter
_LAST_PIN_NOTE = "not attempted"


def _nvcc_major_from_banner(banner: str) -> str:
    m = re.search(r"release\s+(\d+)\.", banner or "")
    return m.group(1) if m else ""


def _toolkit_major(prefix: str) -> str:
    """CUDA major version of a toolkit prefix, from its version.json (or the
    directory name as a fallback). '' if it does not look like a toolkit."""
    if not prefix or not os.path.isfile(os.path.join(prefix, "bin", "nvcc")):
        return ""
    vj = os.path.join(prefix, "version.json")
    try:
        import json
        with open(vj) as fh:
            v = json.load(fh).get("cuda", {}).get("version", "")
        if v:
            return v.split(".")[0]
    except Exception:
        pass
    base = os.path.basename(prefix.rstrip("/"))
    return base.split(".")[0] if base[:1].isdigit() else ""


def _find_matching_toolkit(major: str) -> str:
    """Newest toolkit under _HPC_SDK_ROOT whose major matches. Bounded:
    os.listdir at depth 1 and 2 of one known prefix, nothing recursive."""
    best, best_key = "", ()
    try:
        sdks = os.listdir(_HPC_SDK_ROOT)
    except OSError:
        return ""
    for sdk in sdks:
        cdir = os.path.join(_HPC_SDK_ROOT, sdk, "cuda")
        try:
            versions = os.listdir(cdir)
        except OSError:
            continue
        for v in versions:
            prefix = os.path.join(cdir, v)
            if _toolkit_major(prefix) != major:
                continue
            try:
                key = tuple(int(x) for x in v.split("."))
            except ValueError:
                key = (0,)
            if key > best_key:
                best, best_key = prefix, key
    return best


def pin_cuda_toolchain() -> str:
    """Make CUDA_HOME and PATH agree with the CUDA major torch was built for.

    Order of precedence: $KB_CUDA_HOME (explicit user override) > an already
    matching $CUDA_HOME > the newest matching toolkit under _HPC_SDK_ROOT.
    Idempotent; returns a one-line note (also stored for env.capture()). Never
    raises: if nothing matches, the environment is left alone and
    warn_if_unsuitable() flags the mismatch at run time instead.
    """
    global _LAST_PIN_NOTE
    try:
        import torch
        rt = (torch.version.cuda or "").split(".")[0]
    except Exception:
        rt = ""
    if not rt:
        _LAST_PIN_NOTE = "skipped: torch has no CUDA runtime version"
        return _LAST_PIN_NOTE

    override = os.environ.get("KB_CUDA_HOME", "")
    current = os.environ.get("CUDA_HOME", "")
    if override:
        chosen, why = override, "KB_CUDA_HOME override"
    elif _toolkit_major(current) == rt:
        chosen, why = current, "CUDA_HOME already matches torch"
    else:
        chosen = _find_matching_toolkit(rt)
        why = (f"CUDA_HOME was {current or 'unset'} (major "
               f"{_toolkit_major(current) or '?'}), torch needs {rt}.x")
        if not chosen:
            _LAST_PIN_NOTE = f"unresolved: {why}; no {rt}.x toolkit under {_HPC_SDK_ROOT}"
            return _LAST_PIN_NOTE

    os.environ["CUDA_HOME"] = chosen
    # NVHPC's module exports this too and some build systems read it first.
    os.environ["NVHPC_CUDA_HOME"] = chosen
    binp = os.path.join(chosen, "bin")
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if parts[:1] != [binp]:
        os.environ["PATH"] = os.pathsep.join([binp] + [p for p in parts if p != binp])
    scrubbed = _scrub_foreign_toolkit_paths(rt)
    added = _add_matching_includes(chosen)
    _LAST_PIN_NOTE = f"CUDA_HOME={chosen} ({why})" + (
        f"; dropped {scrubbed} CPATH/LIBRARY_PATH entries of other CUDA majors"
        if scrubbed else "") + (f"; CPATH += {added}" if added else "")
    return _LAST_PIN_NOTE


def _add_matching_includes(chosen: str) -> str:
    """Put the chosen toolkit's own headers AND its HPC-SDK math_libs headers
    (cusparse.h, cublas_v2.h, ... -- torch's ATen/cuda headers include them)
    on CPATH, exactly what `module load cudatoolkit/<ver>` exports. Without
    the math_libs entry a torch-extension compile dies with
    "fatal error: cusparse.h: No such file or directory" (observed
    2026-09-04 after scrubbing the 13.2 CPATH). Returns what was prepended."""
    entries = []
    # <root>/<sdk>/cuda/<ver>  ->  <root>/<sdk>/math_libs/<ver>/include
    cuda_dir = os.path.dirname(chosen)
    ver = os.path.basename(chosen.rstrip("/"))
    if os.path.basename(cuda_dir) == "cuda":
        ml = os.path.join(os.path.dirname(cuda_dir), "math_libs", ver, "include")
        if os.path.isfile(os.path.join(ml, "cusparse.h")):
            entries.append(ml)
    inc = os.path.join(chosen, "include")
    if os.path.isdir(inc):
        entries.append(inc)
    # Link-time and run-time library paths for the same two trees (the math
    # libraries -lcublas/-lcusparse/... live only under math_libs/<ver>/lib64;
    # nvcc adds only cuda/<ver>/lib64 implicitly). Affects child processes
    # (nvcc, runtime-JIT DSLs), which is exactly where it is needed.
    libs = []
    if os.path.basename(cuda_dir) == "cuda":
        mll = os.path.join(os.path.dirname(cuda_dir), "math_libs", ver, "lib64")
        if os.path.isdir(mll):
            libs.append(mll)
    if os.path.isdir(os.path.join(chosen, "lib64")):
        libs.append(os.path.join(chosen, "lib64"))
    for var in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        cur = [e for e in os.environ.get(var, "").split(os.pathsep) if e]
        merged = libs + [e for e in cur if e not in libs]
        if merged != cur:
            os.environ[var] = os.pathsep.join(merged)
    if not entries:
        return ""
    cur = [e for e in os.environ.get("CPATH", "").split(os.pathsep) if e]
    new = entries + [e for e in cur if e not in entries]
    if new != cur:
        os.environ["CPATH"] = os.pathsep.join(new)
    return os.pathsep.join(entries)


_SCRUB_VARS = ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH")


def _scrub_foreign_toolkit_paths(major: str) -> int:
    """Drop include/library search entries that belong to a CUDA toolkit (or
    the HPC SDK's matching math_libs) of a DIFFERENT major than torch's.

    The cudatoolkit/13.2 module exports
    CPATH=<sdk>/26.5/math_libs/13.2/include:<sdk>/26.5/cuda/13.2/include, and
    gcc/nvcc honour CPATH after -I/-isystem, so even a 12.9 nvcc ends up
    compiling against 13.2's cudaTypedefs.h (which dropped the unversioned
    PFN_* aliases CUTLASS relies on) -- observed 2026-09-04 as a hard
    compile failure of the flashattention-t wrapper. Only entries under
    _HPC_SDK_ROOT whose '<cuda|math_libs>/<ver>' component has another major
    are removed; everything else is left untouched. Returns the count."""
    pat = re.compile(re.escape(_HPC_SDK_ROOT) + r"/[^/]+/(cuda|math_libs)/(\d+)(?:\.\d+)*(?:/|$)")
    dropped = 0
    for var in _SCRUB_VARS:
        val = os.environ.get(var)
        if not val:
            continue
        keep = []
        for entry in val.split(os.pathsep):
            m = pat.match(entry)
            if m and m.group(2) != major:
                dropped += 1
                continue
            keep.append(entry)
        if dropped and len(keep) != len(val.split(os.pathsep)):
            if keep:
                os.environ[var] = os.pathsep.join(keep)
            else:
                os.environ.pop(var, None)
    return dropped
