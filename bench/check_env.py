#!/usr/bin/env python3
"""Check a machine against what HPC-KernelBench needs, before building anything.

    source bench/env.sh && $PY bench/check_env.py

Prints one line per requirement (OK / WARN / MISSING) and ends with a summary of
which artifact directories this machine cannot build (REQUIRES_GPU markers,
optional MPI/BLAS/HIP dependencies). Read-only; never installs or downloads.
Bounded: looks only inside this repository and at the paths in KB_* variables.
"""
import importlib
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
rows = []


def rec(status, what, detail=""):
    rows.append((status, what, detail))


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


# ---- Python + packages
v = sys.version_info
rec("OK" if v >= (3, 10) else "MISSING", f"python {v.major}.{v.minor}", "need >= 3.10 (reference 3.11.7)")
need = {"numpy": "2.3", "scipy": "1.16", "yaml": "6.0", "torch": "2.8"}
for m, ref in need.items():
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        rec("OK", f"{m} {ver}", f"reference {ref}.x")
    except Exception as e:  # noqa: BLE001
        rec("MISSING" if m != "torch" else "WARN", m, f"{type(e).__name__}; torch is needed for every GPU path" if m == "torch" else "pip install -r bench/requirements.txt")

# ---- torch <-> CUDA toolkit agreement
torch_cuda = ""
gpu_cap = None
try:
    import torch
    torch_cuda = torch.version.cuda or ""
    rec("OK" if torch.cuda.is_available() else "WARN", f"torch CUDA runtime {torch_cuda}",
        "GPU visible" if torch.cuda.is_available() else "no GPU visible to torch (fine on a build-only/CPU node)")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        gpu_cap = cap
        rec("OK", f"GPU {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}", "")
except Exception:
    pass

cuda_home = os.environ.get("KB_CUDA_HOME") or os.environ.get("CUDA_HOME", "")
nvcc = os.path.join(cuda_home, "bin", "nvcc") if cuda_home else shutil.which("nvcc") or ""
banner = sh([nvcc, "--version"]) if nvcc and os.path.exists(nvcc) else ""
m = re.search(r"release (\d+)\.(\d+)", banner)
if m:
    nv = f"{m.group(1)}.{m.group(2)}"
    same = (not torch_cuda) or torch_cuda.split(".")[0] == m.group(1)
    rec("OK" if same else "MISSING", f"nvcc {nv} at {nvcc}",
        "" if same else f"CUDA major must match torch's ({torch_cuda}); set KB_CUDA_HOME to a {torch_cuda.split('.')[0]}.x toolkit")
else:
    rec("MISSING", "nvcc", "set KB_CUDA_HOME (bench/env.sh) or put nvcc on PATH")

# math libs (cuBLAS/cuSPARSE headers) reachable?
ml = os.environ.get("KB_CUDA_MATHLIBS", "")
cands = [os.path.join(cuda_home, "include", "cusparse.h")] + ([os.path.join(ml, "include", "cusparse.h")] if ml else [])
rec("OK" if any(os.path.exists(c) for c in cands) else "MISSING", "cusparse.h (cuBLAS/cuSPARSE headers)",
    "" if any(os.path.exists(c) for c in cands) else "standard toolkit: under $CUDA_HOME/include; HPC SDK: set KB_CUDA_MATHLIBS")

# ---- compilers and tools
for var, ref in (("KB_CC", "gcc 12-14"), ("KB_CXX", "g++ 12-14")):
    path = os.environ.get(var) or shutil.which(var[3:].lower()) or ""
    ver = (sh([path, "--version"]).splitlines() or [""])[0] if path and os.path.exists(path) else ""
    mv = re.search(r"\b(\d+)\.\d+\.\d+", ver)
    ok = bool(mv) and 12 <= int(mv.group(1)) <= 14
    rec("OK" if ok else ("WARN" if mv else "MISSING"), f"{var}={path or '?'}", ver or f"need {ref} (nvcc 12.x rejects gcc >= 15)")
hc = os.environ.get("KB_HOST_COMPILER_BIN", "")
rec("OK" if hc and os.path.exists(os.path.join(hc, "g++")) else "WARN", "gcc/g++ 12 for -ccbin (KB_HOST_COMPILER_BIN)",
    "" if hc and os.path.exists(os.path.join(hc, "g++")) else "14 artifacts need gcc/g++ 12 (nvcc -ccbin or their own build); see ENVIRONMENT.md")
_gxx12 = next((c for c in ((os.path.join(hc, "g++") if hc else ""), "/usr/bin/g++-12", shutil.which("g++-12") or "") if c and os.path.exists(c)), "")
rec("OK" if _gxx12 else "WARN", f"gcc 12 pair resolves to {_gxx12 or '<none>'} (KB_GXX12/KB_GCC12)",
    "" if _gxx12 else "toolchain.sh falls back to CXX for the gcc12-only artifacts; they may fail to build")
for tool, why in (("cmake", "28 artifacts"), ("ninja", "torch JIT extensions"), ("git", "cloning artifacts"), ("make", "most artifacts")):
    p = shutil.which(tool)
    rec("OK" if p else ("MISSING" if tool != "ninja" else "WARN"), f"{tool} {(sh([p, '--version']).splitlines() or [''])[0] if p else ''}".strip(), "" if p else f"needed by {why}")

# ---- optional stacks
mpi = os.environ.get("KB_MPI_ROOT", "")
rec("OK" if mpi and os.path.exists(os.path.join(mpi, "include", "mpi.h")) else "WARN", "MPI headers (KB_MPI_ROOT)",
    "" if mpi and os.path.exists(os.path.join(mpi, "include", "mpi.h")) else "only amgt/bootcmatchgx/hicma-x/exageostat/graphset link MPI")
blas = os.environ.get("KB_BLAS_LIBDIR", "")
rec("OK" if blas and os.path.isdir(blas) else "WARN", "BLAS/LAPACK libdir (KB_BLAS_LIBDIR)",
    "" if blas and os.path.isdir(blas) else "only cholesky/* and multigrid/bootcmatchgx need it")
rec("OK" if shutil.which("hipcc") else "WARN", "hipcc", "" if shutil.which("hipcc") else "only spmv/spmv-acc (HIP-on-CUDA) needs it")

# ---- libstdc++ ABI trap
pre = os.environ.get("LD_PRELOAD", "")
sys_lib = "/usr/lib64/libstdc++.so.6"
has = os.path.exists(sys_lib) and "CXXABI_1.3.15" in sh(["strings", sys_lib])
rec("OK" if (pre or not os.path.exists(sys_lib)) else "WARN", f"LD_PRELOAD={pre or '<unset>'}",
    "torch CUDA extensions built with g++>=13 need a libstdc++ with CXXABI_1.3.15 loaded first; "
    + ("system libstdc++ has it -> export KB_LD_PRELOAD=/usr/lib64/libstdc++.so.6 if an extension import fails" if has else "verify with any *-torch-extension build.sh"))

# ---- artifacts this machine cannot build; checkouts still to fetch; sm_80 pins
deferred, missing, pinned_sm80 = [], [], 0
for k in sorted(os.listdir(ART)) if os.path.isdir(ART) else []:
    kd = os.path.join(ART, k)
    if not os.path.isdir(kd):
        continue
    for s in sorted(os.listdir(kd)):
        d = os.path.join(kd, s)
        if not os.path.isdir(d):
            continue
        rq = os.path.join(d, "REQUIRES_GPU")
        if os.path.exists(rq):
            deferred.append((f"{k}/{s}", open(rq).read().strip()[:90]))
        for f in os.listdir(d):
            if f.endswith(".provenance") and not os.path.exists(os.path.join(d, f[:-len(".provenance")])):
                missing.append(f"{k}/{s}/{f[:-len('.provenance')]}")   # dangling shared symlinks count too
        b = os.path.join(d, "build.sh")
        if os.path.exists(b) and re.search(r'sm_80|compute_80|ARCH_LIST="?8\.0|ARCHITECTURES="?80', open(b).read()):
            pinned_sm80 += 1

# ---- report
width = max(len(w) for _, w, _ in rows)
for st, what, det in rows:
    print(f"[{st:7s}] {what:<{width}}  {det}")
n_missing = sum(1 for st, _, _ in rows if st == "MISSING")
print()
print(f"{n_missing} MISSING, {sum(1 for st,_,_ in rows if st=='WARN')} WARN.  "
      + ("Core harness + GPU artifact builds should work." if n_missing == 0 else "Fix MISSING items before building GPU artifacts (CPU smoke_all needs only python+numpy+scipy+pyyaml)."))
if deferred:
    print("\nArtifacts that need a newer GPU than sm_80 (DEFERRED-HARDWARE; build these on such a machine):")
    for d, r in deferred:
        print(f"  {d:36s} {r}")
print(f"\n{len(missing)} artifact checkouts missing (git-ignored clones): each build.sh re-creates its own from "
      "<name>.provenance on first run; bench/artifacts/fetch_sources.sh prefetches all of them.")
if gpu_cap and tuple(gpu_cap) != (8, 0):
    print(f"\nGPU is sm_{gpu_cap[0]}{gpu_cap[1]} but {pinned_sm80} build.sh files pin sm_80 (A100) -- ENVIRONMENT.md section 8: "
          "-arch=sm_80 embeds PTX and JITs (works); -gencode ...,code=sm_80 and TORCH_CUDA_ARCH_LIST=\"8.0\" give "
          "SASS-only binaries that will not load -> take the arch from KB_SM / KB_TORCH_ARCH (exported by toolchain.sh).")
sys.exit(1 if n_missing else 0)
