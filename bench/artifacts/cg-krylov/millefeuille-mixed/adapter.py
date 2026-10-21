"""
Mille-feuille (SC'24), the paper's actual method: `main-cg-mixed`.

Paper: "Mille-feuille: A Tile-Grained Mixed Precision Single-Kernel Conjugate
Gradient Solver on GPUs". PAPER_KEY = conf/sc/YangZNJS0T024. Artifact:
https://github.com/SuperScientificSoftwareLaboratory/Mille-feuille @ 1421c29.

Why a second Mille-feuille adapter: ../millefeuille wraps `main-cg` (the
Makefile's `cg` target). For nnz >= 100K that binary runs cg_solve_reduce, an
ordinary multi-kernel fp64 CG -- not the paper's design. The paper's method
(one persistent kernel for the whole solve, tiled sparse format, per-tile
precision chosen at run time among fp64/fp32/fp16/fp8) is only in
src/main-cg-mixed.cu (`make cg-mix`), which always calls cg_solve_inc with
the *_mix_precision kernels, whatever the nnz. This adapter wraps that.
Keep both: the gap between them is the paper's contribution.

What the binary does (read from the source, not changed):
  - N is truncated to a multiple of 16 (BLOCK_SIZE); every cg-common input
    already is one.
  - Exactly 100 CG steps inside the kernel (`for (iter = 1; iter <= 100;
    iter++)`, residual stop commented out upstream). So this adapter is
    fixed-iteration with maxiter = 100 regardless of params -- the spec's
    k = 50 cannot be honoured without editing kernel code. Compare per-step
    times (millefeuille_reported_ms_per_iter), not whole-run times.
  - Precision per tile is switched on the ABSOLUTE per-step change of p
    (thresholds 1e-5 / 1e-6 / 1e-7 / 1e-8), so the scale of b changes how
    much low precision is used. The paper used b = A*ones; this adapter
    uses the track's b = U(-1,1), seed 42, like every cg-krylov impl.
  - The work split must fit in < 6912 warps (an A100: 108 SMs x 64). If it
    cannot, the shipped code returns silently without solving; mixed.patch
    makes it print MF_MIXED_SKIPPED, reported here as UNSUPPORTED.

mixed.patch (I/O only, no kernel code): MF_RHS_FILE (external b),
MF_X_OUT_FILE (x dump), one MF_MIXED_RESULT stdout line with the artifact's
own time_cg (gettimeofday around the single solve kernel) and time_format
(CSR -> tile conversion), and the MF_MIXED_SKIPPED message.

Timing boundary: one subprocess per run(), timed by a wall clock, so the
harness number includes process start, .mtx parsing and the tile
conversion. Use millefeuille_reported_ms / _per_iter (the artifact's own
time_cg) for speed comparisons.

Gate: fixed-iteration rule, same as ../millefeuille and ScipyCG: relres =
||b - A x|| / ||b|| recomputed in fp64, pass iff finite and < 1. Because
this method computes in lower precision, compare its relres after 100
steps with an fp64 CG's after 100 steps; the record keeps it.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

KERNEL = "cg-krylov"
IMPL_NAME = "millefeuille-cg-mixed"
PAPER_KEY = "conf/sc/YangZNJS0T024"
PRECISIONS = ["fp64"]   # fp64 interface; tiles drop to fp32/fp16/fp8 internally

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, "source")
_BINARY = os.path.join(_SOURCE, "main-cg-mixed")
_BLOCK_SIZE = 16          # src/common.h
_KERNEL_ITERS = 100       # hard-coded loop in the *_mix_precision kernels
_RESULT_RE = re.compile(
    r"MF_MIXED_RESULT\s+time_cg=([-+0-9.eE]+)\s+iterations=(\d+)\s+"
    r"time_format=([-+0-9.eE]+)\s+l2_norm=(\S+)\s+snew=(\S+)")


def available() -> tuple[bool, str]:
    if not os.path.exists(_BINARY):
        return False, "source/main-cg-mixed missing -- run build.sh first"
    if not os.access(_BINARY, os.X_OK):
        return False, "source/main-cg-mixed exists but is not executable"
    return True, ""


def create(precision: str):
    return MilleFeuilleMixedCG(precision)


class MilleFeuilleMixedCG:
    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        if precision != "fp64":
            raise NotImplementedError(
                f"{IMPL_NAME}: interface is fp64 (MAT_VAL_TYPE=double in src/common.h); "
                f"mixed precision is internal and automatic; requested {precision}")
        self.precision = precision
        self._tmpdir = None
        self._last_stdout = ""

    def prepare(self, matrix, params: dict):
        import numpy as np
        from scipy.io import mmwrite

        A = matrix.csr.astype(np.float64)
        n = A.shape[0]
        n_trunc = (n // _BLOCK_SIZE) * _BLOCK_SIZE or n

        params["mode"] = "fixed-iter"
        if int(params.get("maxiter", _KERNEL_ITERS)) != _KERNEL_ITERS:
            params["millefeuille_maxiter_note"] = (
                f"requested maxiter={params['maxiter']} ignored: the kernel runs a "
                f"hard-coded {_KERNEL_ITERS} steps")
        params["maxiter"] = _KERNEL_ITERS
        params["millefeuille_code_path"] = "cg_solve_inc + *_mix_precision kernels (main-cg-mixed)"
        params["internal_precision"] = "per-tile dynamic fp64/fp32/fp16/fp8"
        params["millefeuille_n_original"] = int(n)
        params["millefeuille_n_truncated"] = int(n_trunc)
        if n_trunc != n:
            params["millefeuille_truncation_note"] = (
                f"N={n} is not a multiple of {_BLOCK_SIZE}: the binary drops the last "
                f"{n - n_trunc} rows; gated against the top-left {n_trunc}x{n_trunc} block")

        rng = np.random.default_rng(params.get("seed", 42))
        b = rng.uniform(-1.0, 1.0, size=n_trunc).astype(np.float64)

        tmpdir = tempfile.mkdtemp(prefix="millefeuille_mixed_")
        # the binary appends to ./data/cg_performance.csv; run in tmpdir so
        # the shared clone stays clean
        os.makedirs(os.path.join(tmpdir, "data"), exist_ok=True)
        mtx_path = os.path.join(tmpdir, "matrix.mtx")
        mmwrite(mtx_path, A, symmetry="general")
        rhs_path = os.path.join(tmpdir, "rhs.bin")
        b.tofile(rhs_path)

        self._tmpdir = tmpdir
        self._params = params
        self._A_trunc = A[:n_trunc, :n_trunc].tocsr()
        self._b = b
        self._n_trunc = n_trunc
        return {"mtx": mtx_path, "rhs": rhs_path,
                "x_out": os.path.join(tmpdir, "x_out.bin")}

    def run(self, h):
        env = dict(os.environ)
        env["MF_RHS_FILE"] = h["rhs"]
        env["MF_X_OUT_FILE"] = h["x_out"]
        if os.path.exists(h["x_out"]):
            os.remove(h["x_out"])
        proc = subprocess.run([_BINARY, h["mtx"]], cwd=self._tmpdir, env=env,
                              capture_output=True, text=True, timeout=1800)
        self._last_stdout = proc.stdout
        if "MF_MIXED_SKIPPED" in proc.stdout:
            raise NotImplementedError(
                f"{IMPL_NAME}: work split does not fit in 6912 warps (A100-sized "
                f"persistent kernel); binary skipped the solve: "
                f"{proc.stdout.strip().splitlines()[-1]}")
        if proc.returncode != 0 or not os.path.exists(h["x_out"]):
            raise RuntimeError(
                f"{IMPL_NAME}: main-cg-mixed exited {proc.returncode}, x dump "
                f"{'present' if os.path.exists(h['x_out']) else 'missing'}\n"
                f"stdout(tail)={proc.stdout[-1000:]}\nstderr(tail)={proc.stderr[-1000:]}")
        m = _RESULT_RE.search(proc.stdout)
        p = self._params
        if m:
            p["millefeuille_reported_ms"] = float(m.group(1))
            p["iterations_actual"] = int(m.group(2))
            p["millefeuille_reported_ms_per_iter"] = float(m.group(1)) / int(m.group(2))
            p["millefeuille_format_ms"] = float(m.group(3))
            p["millefeuille_reported_relres"] = m.group(4)
        else:
            p["millefeuille_stdout_parse_failed"] = True
            p["iterations_actual"] = _KERNEL_ITERS
        return h["x_out"]

    def to_host(self, out):
        import numpy as np

        x = np.fromfile(out, dtype=np.float64)
        p = self._params
        if x.size != self._n_trunc:
            relres, finite = float("inf"), False
            p["millefeuille_x_dump_size_mismatch"] = f"expected {self._n_trunc}, got {x.size}"
        else:
            b_norm = max(float(np.linalg.norm(self._b)), 1e-300)
            relres = float(np.linalg.norm(self._b - self._A_trunc @ x) / b_norm)
            finite = bool(np.isfinite(relres))
        passed = finite and relres < 1.0
        p["relres_achieved"] = relres
        p["gate"] = "finite-and-decreased (fixed-iter; full convergence not required)"
        p["gate_passed"] = bool(passed)
        return np.array([1.0 if passed else 0.0])

    def timer(self):
        from kernelbench.harness import Timer
        return Timer()

    def free(self, h):
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None
