"""
BootCMatchGX adapter for the cg-krylov track's `cg-e2e-ilu0-to-convergence`
variant, competing as an AMG-preconditioned FCG alternative to spcg's own
ILU0-preconditioned PCG on the SAME variant (the variant's own claim text:
"generalizes SPCG's own artifact discipline... into a mandatory
requirement for every submission" -- i.e. this slot is the track's general
"e2e Krylov solve to a relative-residual convergence gate" competition, not
literally ILU0-only; `cg-hpcg-generator-amg`'s correctness text explicitly
cross-references "same relative-residual gate as cg-e2e-ilu0-to-
convergence" when describing its OWN AMG-preconditioned loop, confirming
AMG-preconditioned solvers are meant to be comparable on this protocol).
This variant's own `protocol.precision` text names BOTH papers directly:
"fp32 (primary -- matches SPCG)... fp64 (secondary -- matches BootCMatchGX's
preconditioned-CG convention...)" -- so PRECISIONS = ["fp64"] here is not a
guess, it is what the spec itself expects for this exact artifact.

Paper: "A Multi-GPU Aggregation-Based AMG Preconditioner for Iterative
Linear Solvers", TPDS'23. PAPER_KEY = "journals/tpds/BernaschiCVD23".

SHARED BUILD -- reuses ../../multigrid/bootcmatchgx/'s clone and compiled
binary verbatim. `source/` and `fcg_bcmg.properties` in THIS directory are
symlinks to that directory's files (see build.sh) -- no second clone, no
second compile, no drift between the two integrations' solver config. The
multigrid track wraps `driverSolve --settings fcg_bcmg.properties` for the
`mg-gpu-e2e-pcg` variant (a hard, no-slack `relres < rtol` gate, per
multigrid's own spec text: "failing to converge within the cap is a hard
FAIL"); this adapter wraps the EXACT SAME binary/config/subprocess pattern
for the cg-krylov track's `cg-e2e-ilu0-to-convergence` variant instead,
which uses a DIFFERENT (softer) gate -- see "Gate discipline" below. Only
the correctness-gate formula, KERNEL/IMPL_NAME/PAPER_KEY, and the
params bookkeeping keys differ; the subprocess-wrapping logic itself
(mtx/rhs writing, `--info` log parsing, independent fp64 residual recompute)
is deliberately near-identical to the multigrid sibling, since it is
literally the same underlying artifact call.

Gate discipline (per this task's own instruction: mirror spcg's to-
convergence gate here, not multigrid's hard gate) -- read directly from
`kernelbench/domains/solvers.py`'s `ScipyCG.to_host()` "to-convergence"
branch and `../../cg-krylov/spcg/adapter.py`'s own `to_host()`, both of
which use `gate_tol = max(10*rtol, 1e-9)`, `passed = finite and relres <=
gate_tol` (a 10x SLACK over rtol, not the bare `relres < rtol` the
multigrid sibling uses for its own harder-gated variant). Applying THIS
gate (not multigrid's) is what makes this adapter's pass/fail decision
directly comparable to spcg's own numbers on the same cg-krylov variant.

No MPI launcher (identical situation to the multigrid sibling, not
repeated in full here): Cray MPICH ships no `mpirun`/`mpiexec`; verified
empirically that `driverSolve` initializes fine as a single-rank MPI
"singleton" when executed directly. `run()` execs the binary with no
launcher prefix.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

KERNEL = "cg-krylov"
IMPL_NAME = "bootcmatchgx-cg-fcg-bcmg"
PAPER_KEY = "journals/tpds/BernaschiCVD23"
PRECISIONS = ["fp64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, "source")   # symlink -> ../../multigrid/bootcmatchgx/source
_BINARY = os.path.join(_SOURCE, "BCMGX", "bin", "example", "driverSolve")
_SETTINGS = os.path.join(_HERE, "fcg_bcmg.properties")  # symlink, same file multigrid uses


def available() -> tuple[bool, str]:
    if not os.path.exists(_BINARY):
        return False, (f"binary not built: {_BINARY} missing -- this "
                        "directory reuses ../../multigrid/bootcmatchgx/'s "
                        "build (symlinked source/); run that sibling's "
                        "build.sh first")
    if not os.access(_BINARY, os.X_OK):
        return False, f"binary present but not executable: {_BINARY}"
    return True, ""


def create(precision: str):
    return BootCMatchGxCgFcgBcmg(precision)


_ITER_RE = re.compile(r"Total iterations\s*:\s*(\d+)")
_INIT_RES_RE = re.compile(r"Initial residual\s*:\s*([0-9.eE+-]+)")
_FINAL_RES_RE = re.compile(r"Final residual\s*:\s*([0-9.eE+-]+)")


class BootCMatchGxCgFcgBcmg:
    """One `run()` call = one fresh `driverSolve` subprocess (no MPI
    launcher) = one complete BCMG setup + FCG-to-convergence solve --
    structurally identical to the multigrid sibling's own class (same
    binary, same settings file, same mtx/rhs writing); only `to_host()`'s
    gate formula differs (see module docstring's "Gate discipline")."""

    name = IMPL_NAME
    platform = "cuda"

    def __init__(self, precision: str = "fp64"):
        self.precision = precision
        self._workdir = None
        self._params = None
        self._A64 = None
        self._b64 = None
        self._mtx_path = None
        self._rhs_path = None
        self._x_out_path = None
        self._info_path = None

    def prepare(self, matrix, params: dict):
        import numpy as np
        import scipy.io as sio

        n = matrix.shape[0]
        rng = np.random.default_rng(params.get("seed", 42))
        # spec's vector_operand: b dense, U(-1,1), fixed seed=42 -- same RNG
        # call as solvers.ScipyCG.prepare()/spcg's own adapter, so this
        # artifact is gated against the identical b any other cg-krylov
        # implementation in this harness would see.
        b64 = rng.uniform(-1.0, 1.0, size=n)
        self._A64 = matrix.csr.astype(np.float64)
        self._b64 = b64

        self._workdir = tempfile.mkdtemp(prefix="bcmgx_cg_")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", matrix.name) or "matrix"
        self._mtx_path = os.path.join(self._workdir, f"{safe_name}.mtx")
        # BootCMatchGX's own format-conversion-equivalent step (it only
        # reads .mtx): timed as preprocessing, excluded from the timed
        # window -- same discipline as spcg's adapter.
        sio.mmwrite(self._mtx_path, matrix.csr.tocoo(), symmetry="general")

        self._rhs_path = os.path.join(self._workdir, "rhs.txt")
        np.savetxt(self._rhs_path, b64, fmt="%.17g")

        self._x_out_path = os.path.join(self._workdir, "x_out.txt")
        self._info_path = os.path.join(self._workdir, "info.txt")

        # resolved knobs written back so they are explicit, auditable
        # fields in RunResult.params -- matches spcg's own discipline of
        # recording BOTH the artifact's own internal cap/tol AND the
        # spec's, never conflated.
        params["mode"] = "e2e-fcg-bcmg-to-convergence"
        params["rtol"] = float(params.get("rtol", 1e-6))
        params["spec_max_iter"] = 2000   # this variant's own cap
        # BootCMatchGX's own hardcoded cap/tol (fcg_bcmg.properties, shared
        # with the multigrid sibling) -- recorded for audit, never adopted
        # as this adapter's own gate.
        params["bcmgx_own_itnlim"] = 1000
        params["bcmgx_own_rtol"] = 1e-6

        self._params = params
        return {"mtx": self._mtx_path}

    def run(self, h):
        for p in (self._x_out_path, self._info_path):
            if os.path.exists(p):
                os.remove(p)

        env = dict(os.environ)
        env["MPICH_GPU_SUPPORT_ENABLED"] = "0"  # same Cray-MPICH quirk as
        # the multigrid sibling's adapter.py.
        # Zaratan-specific finding (2026-09-09 reproduction): driverSolve
        # links MPI and does MPI_Init as a singleton (see module docstring
        # -- "no MPI launcher needed" held on Perlmutter's Cray-MPICH). On
        # this cluster (Slurm + PMIx + OpenMPI), the FIRST such singleton
        # subprocess inside one srun job step succeeds, but every SUBSEQUENT
        # one deterministically deadlocks: OpenMPI's singleton bootstrap
        # inherits SLURM_*/PMIX_*/OMPI_* env vars from the surrounding srun
        # step and tries to attach to that step's PMIx namespace instead of
        # doing a genuinely independent singleton init, and only one client
        # can attach. Confirmed empirically: 4 sequential driverSolve calls
        # in one process hang on call #2 with inherited env (reproduced on
        # both an a100_1g.5gb MIG slice and a full A100 -- not a MIG/memory
        # issue), and succeed 4/4 (6-20s each) once SLURM_*/PMIX_*/OMPI_*
        # are stripped from the child's env. Since the harness calls run()
        # many times per matrix (1 correctness check + warmup + reps), this
        # is required for the gate to complete at all here. Stripping these
        # is a no-op on a machine that doesn't set them (e.g. Perlmutter).
        env = {k: v for k, v in env.items()
               if not (k.startswith("SLURM_") or k.startswith("PMIX_")
                       or k.startswith("OMPI_"))}

        cmd = [_BINARY,
              "--matrix", self._mtx_path,
              "--rhs", self._rhs_path,
              "--settings", _SETTINGS,
              "--out", self._x_out_path,
              "--info", self._info_path]
        proc = subprocess.run(cmd, cwd=self._workdir, env=env,
                              capture_output=True, text=True, timeout=900)

        p = self._params
        p["bcmgx_subprocess_returncode"] = proc.returncode
        if not os.path.exists(self._x_out_path):
            raise RuntimeError(
                f"driverSolve produced no --out file (returncode={proc.returncode}); "
                f"stdout tail:\n{proc.stdout[-2000:]}\n"
                f"stderr tail:\n{proc.stderr[-2000:]}")

        if os.path.exists(self._info_path):
            with open(self._info_path) as f:
                info_text = f.read()
            m = _ITER_RE.search(info_text)
            if m:
                p["iterations_actual"] = int(m.group(1))
                p["bcmgx_num_iterations"] = int(m.group(1))
            m = _INIT_RES_RE.search(info_text)
            if m:
                p["bcmgx_initial_residual_internal"] = float(m.group(1))
            m = _FINAL_RES_RE.search(info_text)
            if m:
                p["bcmgx_final_residual_internal"] = float(m.group(1))
            if "bcmgx_initial_residual_internal" in p and "bcmgx_final_residual_internal" in p:
                init_r = p["bcmgx_initial_residual_internal"]
                if init_r > 0:
                    p["bcmgx_relative_residual_internal"] = p["bcmgx_final_residual_internal"] / init_r

        return self._x_out_path

    def to_host(self, out):
        import numpy as np

        n = self._A64.shape[0]
        x = np.loadtxt(out).reshape(-1)
        if x.size != n:
            relres = float("inf")
            finite = False
        else:
            b_norm = max(float(np.linalg.norm(self._b64)), 1e-300)
            relres = float(np.linalg.norm(self._b64 - self._A64 @ x) / b_norm)
            finite = bool(np.isfinite(relres))

        p = self._params
        rtol = float(p.get("rtol", 1e-6))
        # cg-krylov's OWN to-convergence gate (10x slack over rtol) --
        # mirrors solvers.ScipyCG.to_host()'s "to-convergence" branch and
        # spcg/adapter.py's own to_host() EXACTLY, so this artifact's
        # pass/fail decision is directly comparable to both on this same
        # variant. Deliberately NOT the multigrid sibling's harder,
        # no-slack `relres < rtol` gate -- see module docstring.
        gate_tol = max(10.0 * rtol, 1e-9)
        passed = finite and relres <= gate_tol
        p["relres_achieved"] = relres
        p["gate"] = f"relres<={gate_tol:.3e} (to-convergence, 10x slack over rtol)"
        p["gate_passed"] = bool(passed)
        return np.array([1.0 if passed else 0.0])

    def timer(self):
        # plain CPU wall-clock, NOT CudaEventTimer -- the solve runs in a
        # separate subprocess with its own CUDA context (same posture as
        # spcg's and the multigrid sibling's adapters).
        from kernelbench.harness import Timer
        return Timer()

    def free(self, h):
        if self._workdir and os.path.isdir(self._workdir):
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None
