"""
BootCMatchGX adapter for the multigrid track's `mg-gpu-e2e-pcg` variant.

Paper: "A Multi-GPU Aggregation-Based AMG Preconditioner for Iterative
Linear Solvers", TPDS'23. PAPER_KEY = "journals/tpds/BernaschiCVD23"
(title-matched in ../../../output/benchmark_groups.json's "multigrid" list).
Artifact: https://github.com/bootcmatch/BootCMatchGX, cloned fresh into
`source/` (see STATUS.md for commit/build provenance -- this is a NEW
integration, unlike amgt/ which reuses ../spgemm/amgt/'s existing build).

What is wrapped
----------------
BootCMatchGX's own library exposes a clean setup/solve C++ API
(`src/preconditioner/prec_setup.h`: `prec_setup(handles*, CSR*,
Preconditioner*, InputParameters&)`; `src/solver/solve.h`: `solve(handles*,
CSR*, vector<vtype>*, vector<vtype>*, InputParameters&, CurrentParameters&,
Preconditioner&, SolverOut*)`), but it is a templated C++ API built around
the library's own `CSR`/`vector<T>`/`handles` structs -- there is no
extern-"C" boundary at that level, and writing one from scratch (matching
this session's bounded login-node budget) was judged higher-risk than
wrapping the artifact's own end-to-end CLI driver. Per ARTIFACT_GUIDE.md
rule 1 ("if the artifact only ships an end-to-end binary that loads
matrices itself, wrap at the finest boundary available and document the
contamination"), this adapter wraps the COMPILED BINARY
`bin/example/driverSolve` as a subprocess -- the SAME pattern
`../../cg-krylov/spcg/adapter.py` already uses in this codebase for an
analogous situation (a driver that loads its own matrix/rhs from files and
runs a complete setup+solve internally).

This is also the CORRECT choice, not just the easiest one, for THIS
variant specifically: mg-gpu-e2e-pcg's own protocol is "one-shot AMG setup
+ PCG iterations to a real stopping tolerance" -- exactly what
`driverSolve --settings fcg_bcmg.properties` does in one process
invocation (FCG = Flexible CG, BCMG = BootCMatchGX's own aggregation AMG
preconditioner -- the paper's own primary configuration, matching
`src/test/data/settings/FCG_BCMG.properties` in the artifact's own repo).

Configuration (`fcg_bcmg.properties`, this directory): rtol=1e-6,
itnlim=1000 -- BootCMatchGX's own exact tolerance/cap
(benchspecs/multigrid/spec.yaml's mg-gpu-e2e-pcg correctness text, sourced
from this paper); prerelax_sweeps=4, postrelax_sweeps=4,
relaxnumber_coarse=20 -- the artifact's own DEFAULTS
(src/test/data/settings/sample.properties), reproduced explicitly rather
than left implicit, and matching the spec's own protocol.hierarchy_config
text verbatim ("4 pre-/post-smoothing l1-Jacobi sweeps ... 20 sweeps at the
coarsest level").

Single-process, single-GPU: BootCMatchGX is designed multi-GPU/MPI-first,
but its own README documents single-MPI-node operation as a first-class
mode ("the library supports different low-level frameworks for computing
the SpMM product ... on a single MPI node"). Cray MPICH on this machine
ships NO `mpirun`/`mpiexec` binary at all (only `srun`, compute-node-only,
out of scope for a login-node build/gate check) -- verified empirically
that `driverSolve` run DIRECTLY (no launcher) initializes fine as an MPI
"singleton" (`MPI_Init` with `nprocs=1`, `myid=0`), so `run()` execs the
binary with no wrapper: one process, one rank, one GPU, `ilower=0,
iupper=n-1` (the whole matrix local to that one rank), so no cross-rank
halo communication ever runs.

File formats (read directly from BootCMatchGX's own source, not guessed):
`--matrix` is Matrix Market general format (`read_local_matrix_from_mtx`,
src/utility/input.cu -- no symmetric-storage auto-expansion found, so
`scipy.io.mmwrite(..., symmetry="general")` is used, same convention
`../../cg-krylov/spcg/adapter.py` already uses for the identical reason);
`--rhs`/`--out` are plain text, ONE VALUE PER LINE
(`Vector::load`/`Vector::print`, src/datastruct/vector.cu); `--info` is a
human-readable "Label : value" log (`SolverOut.cu`'s `dump()`), parsed here
via regex for `Total iterations`, `Initial residual`, `Final residual`
(never trusted directly as the correctness gate -- see to_host()).

Correctness gate discipline (mirrors solvers.AmgE2ePcgCPU.to_host() so the
two are apples-to-apples comparable -- same kernelbench.domains.solvers
module): to_host() independently recomputes, in fp64,
relres = ||b - A@x|| / ||b|| using the x read back from `--out` and the
SAME b that was actually written to the rhs file -- never trusting the
driver's own internal `SolverOut::exitRes` bookkeeping (recorded into
params for audit only). Hard fail (per spec, unlike the fixed-iter
variant) if relres >= rtol. This gate is NEVER loosened to compensate for
the precision-ceiling finding below -- the same rtol=1e-6 the spec demands
is compared against, and the resulting (mechanical, explained) failure is
reported honestly rather than hidden.

DECISIVE FINDING (2026-09-05, gate investigation): --out's ONLY solution-
export path in this artifact is `Vector::print()` (src/datastruct/
vector.cu:445), which writes every value via `fprintf(fp, "%g\n", ...)` --
C's `%g` defaults to 6 SIGNIFICANT DIGITS, nowhere near fp64's ~16. Every
`--matrix`-path smoke run's SOLVE itself is genuinely correct: BootCMatchGX's
own internal (pre-truncation, full-fp64) bookkeeping shows final/initial
residual ratios of 3.9e-8 / 1.8e-7 / 7.4e-7 across the 3 smoke matrices
(all comfortably under rtol=1e-6 -- see params["bcmgx_*_residual_internal"]
and note these are ABSOLUTE residual norms, not pre-normalized: BootCMatchGX
does not report a ready-made relative figure, so divide final by initial
yourself). But to_host()'s INDEPENDENT recompute of relres from the
6-sig-fig-truncated exported x lands at 5.0e-6 / 1.2e-5 / 2.6e-6 --
ABOVE rtol=1e-6, failing the gate. A control experiment proves this is
pure text-I/O quantization, not a solver defect: taking the TRUE solution
x* (scipy spsolve, relres ~1e-16) and rounding it to 6 significant digits
with Python's own "%.6g" (simulating BootCMatchGX's fprintf exactly)
reproduces near-identical residuals -- 5.14e-6 / 1.26e-5 / 2.39e-6 -- from
a PERFECT solution that never touched BootCMatchGX at all. I.e. no x this
artifact could possibly export through --out (however well it actually
solved the system) can pass an independent rtol=1e-6 gate: the boundary
this adapter is forced to use for the solution vector has a precision
ceiling below what the spec's own tolerance demands.

RESOLUTION: `%g` -> `%.17g` in `Vector::print` (`vector.cu:447`,
`vector_print_precision.patch`, applied by `../build.sh`) IS an I/O-only,
zero-numerical-effect format-string change -- it controls how many
`fprintf` digits get written for a `double`, nothing else on the line, no
solver/kernel arithmetic. ARTIFACT_GUIDE.md rule 3 permits build-system/IO
fixes and forbids only kernel/solver-code changes, so this patch is
in-scope (an earlier draft of this docstring/STATUS.md incorrectly treated
the whole file as off-limits and left the gate failing; corrected once the
rule was re-read against this specific line). With the patch applied, the
gate PASSES on all 3 smoke matrices (relres 3.9e-8/1.8e-7/7.4e-7, matching
BootCMatchGX's own internal bookkeeping to 6-8 significant digits -- see
STATUS.md's "Gate outcome (current)" section for the full table).

Timing-boundary caveat (documented prominently here AND in STATUS.md, same
posture as spcg's adapter): timer() returns the plain CPU-wall-clock
kernelbench.harness.Timer, NOT a CUDA-event timer -- the solve runs in a
separate subprocess with its own CUDA context, so there is no shared CUDA
context to bracket with events from this process. The harness's own
per-rep number therefore includes process/CUDA-context startup and
Matrix-Market parsing on top of the actual setup+solve work -- coarser
than BootCMatchGX's own in-process timers. This adapter does NOT currently
parse BootCMatchGX's own tsetup/tsolve breakdown (would need the
`--detailed-prof`/`--summary-prof` machinery, not wired here -- see
STATUS.md "Not done"); only the combined subprocess wall time is reported.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile

KERNEL = "multigrid"
IMPL_NAME = "bootcmatchgx-fcg-bcmg"
PAPER_KEY = "journals/tpds/BernaschiCVD23"
PRECISIONS = ["fp64"]

_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(_HERE, "source")
_BINARY = os.path.join(_SOURCE, "BCMGX", "bin", "example", "driverSolve")
_SETTINGS = os.path.join(_HERE, "fcg_bcmg.properties")
# No MPI launcher: Cray MPICH ships no mpirun/mpiexec (only srun on compute
# nodes), and a single-rank MPI program initializes fine as a "singleton" when
# executed directly -- verified 2026-09-05 on the login node (MPI_Init +
# CUDA device selection succeed without a launcher). So run() executes the
# binary directly: one process, one rank, one GPU.


def available() -> tuple[bool, str]:
    if not os.path.exists(_BINARY):
        return False, f"binary not built: {_BINARY} missing (run build.sh)"
    if not os.access(_BINARY, os.X_OK):
        return False, f"binary present but not executable: {_BINARY}"
    return True, ""


def create(precision: str):
    return BootCMatchGxFcgBcmg(precision)


_ITER_RE = re.compile(r"Total iterations\s*:\s*(\d+)")
_INIT_RES_RE = re.compile(r"Initial residual\s*:\s*([0-9.eE+-]+)")
_FINAL_RES_RE = re.compile(r"Final residual\s*:\s*([0-9.eE+-]+)")


class BootCMatchGxFcgBcmg:
    """One `run()` call = one fresh `driverSolve` subprocess (no MPI
    launcher, see module docstring) = one complete BCMG setup + FCG-to-
    convergence solve. `prepare()` writes the workload's CSR to a .mtx file
    and the spec's U(-1,1) rhs to a text file once; `run()` launches the
    subprocess and parses its own `--info` log for audit (never for the
    gate); `to_host()` performs the independent fp64 correctness gate (see
    module docstring's "DECISIVE FINDING" for why this gate mechanically
    fails on every input despite BootCMatchGX's own solve being correct)."""

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

        params["mg_phase"] = "e2e"
        n = matrix.shape[0]
        rng = np.random.default_rng(params.get("seed", 42))
        b64 = rng.uniform(-1.0, 1.0, size=n)
        self._A64 = matrix.csr.astype(np.float64)
        self._b64 = b64

        self._workdir = tempfile.mkdtemp(prefix="bcmgx_")
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", matrix.name) or "matrix"
        self._mtx_path = os.path.join(self._workdir, f"{safe_name}.mtx")
        # BootCMatchGX's own format-conversion-equivalent step (it only
        # reads .mtx): timed as preprocessing is N/A here since setup+solve
        # both happen inside run() (see module docstring's e2e discipline,
        # matching solvers.AmgE2ePcgCPU) -- this mmwrite happens in
        # prepare(), excluded from the timed window, matching the spec's
        # preprocessing_reported text (host-side matrix generation/parsing
        # counted separately, once).
        sio.mmwrite(self._mtx_path, matrix.csr.tocoo(), symmetry="general")

        self._rhs_path = os.path.join(self._workdir, "rhs.txt")
        np.savetxt(self._rhs_path, b64, fmt="%.17g")

        self._x_out_path = os.path.join(self._workdir, "x_out.txt")
        self._info_path = os.path.join(self._workdir, "info.txt")

        params["mode"] = "e2e-fcg-bcmg"
        params["rtol"] = 1e-6       # matches fcg_bcmg.properties; informational
        params["itnlim"] = 1000     # matches fcg_bcmg.properties; informational

        self._params = params
        return {"mtx": self._mtx_path}

    def run(self, h):
        for p in (self._x_out_path, self._info_path):
            if os.path.exists(p):
                os.remove(p)

        env = dict(os.environ)
        env["MPICH_GPU_SUPPORT_ENABLED"] = "0"  # same Cray-MPICH quirk as
        # ../amgt/adapter.py -- single-rank, no GPU-direct MPI transfers.
        # driverSolve does MPI_Init as a singleton. On a Slurm+PMIx+OpenMPI
        # site (e.g. zaratan) inheriting the surrounding srun job step's
        # SLURM_*/PMIX_*/OMPI_*/PMI_* env makes OpenMPI's singleton bootstrap
        # try to attach to that namespace and DEADLOCK; strip them from the
        # child's env (no-op on Cray where these caused no problem). Same fix
        # as ../../cg-krylov/bootcmatchgx/adapter.py (identical binary).
        env = {k: v for k, v in env.items()
               if not (k.startswith("SLURM_") or k.startswith("PMIX_")
                       or k.startswith("OMPI_") or k.startswith("PMI_"))}

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
                p["bcmgx_num_iterations"] = int(m.group(1))
            m = _INIT_RES_RE.search(info_text)
            if m:
                p["bcmgx_initial_residual_internal"] = float(m.group(1))
            m = _FINAL_RES_RE.search(info_text)
            if m:
                p["bcmgx_final_residual_internal"] = float(m.group(1))
            # BootCMatchGX's own "Initial"/"Final residual" fields are
            # ABSOLUTE norms (||r0||=||b|| at x0=0, confirmed empirically:
            # they print as O(10), not 1.0), not a ready-made relative
            # figure -- divide ourselves purely for AUDIT/DISCLOSURE
            # (never for the gate; see module docstring's "never trusting
            # the driver's own internal bookkeeping" and "DECISIVE FINDING"
            # -- this is reported so a reader can see the solve itself
            # converged even though the independent gate below fails for
            # an unrelated, I/O-precision reason).
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
        # hard fail if not converged -- matches spec's mg-gpu-e2e-pcg
        # correctness text ("failing to converge within the cap is a hard
        # FAIL for this variant"), same criterion as
        # solvers.AmgE2ePcgCPU.to_host(). NEVER loosened for the precision-
        # ceiling finding below -- the same rtol the spec demands is
        # compared against, unconditionally.
        passed = finite and relres < rtol
        p["relres_achieved"] = relres
        p["gate"] = f"relres<{rtol:.1e} within {p.get('itnlim', 1000)} iters (hard fail if not reached)"
        p["gate_passed"] = bool(passed)
        if not passed and finite:
            # Disclosed, evidence-backed diagnostic (see module docstring's
            # "DECISIVE FINDING"): this artifact's ONLY solution-export path
            # (--out -> Vector::print -> fprintf "%g", 6 significant
            # digits) makes an independent rtol=1e-6 gate structurally
            # unreachable regardless of solve quality. Not a claim made on
            # faith -- p["bcmgx_relative_residual_internal"] (BootCMatchGX's
            # own full-precision, pre-truncation bookkeeping, set in run()
            # if the --info log parsed) typically already satisfies rtol
            # here, which is the actual evidence this is an I/O ceiling,
            # not a solver defect.
            p["gate_fail_reason"] = (
                "independent-recompute uses x from --out's 6-sig-fig %g text "
                "dump (BootCMatchGX's own Vector::print, unmodified per "
                "ARTIFACT_GUIDE.md rule 3); this caps achievable relres well "
                "above rtol=1e-6 regardless of solver quality -- see "
                "STATUS.md's control experiment (rounding the EXACT solution "
                "to 6 sig figs alone reproduces this same relres magnitude)")
        return np.array([1.0 if passed else 0.0])

    def timer(self):
        # plain CPU wall-clock, NOT CudaEventTimer -- see module docstring's
        # "Timing-boundary caveat" (separate subprocess/CUDA context).
        from kernelbench.harness import Timer
        return Timer()

    def free(self, h):
        if self._workdir and os.path.isdir(self._workdir):
            shutil.rmtree(self._workdir, ignore_errors=True)
        self._workdir = None
