# Reproducing HPC-KernelBench on another machine: prompt for a coding agent

> **On zaratan (UMD) the environment is already set up** — `source
> bench/env.sh` and skip STEP 1. See `bench/ENVIRONMENT.md` §9 and
> `bench/REPRODUCTION_zaratan.md`; every artifact already has a "Reproduction on
> zaratan" section in its `STATUS.md`. Use this prompt for a THIRD machine, or
> to re-run the whole build+gate sweep from scratch.

Paste the block below into a coding agent (Claude Code or similar) started in a
fresh clone of this repository on the target machine. Fill in the two
`<...>` placeholders. Everything the agent needs to know about the machine is
read from `bench/ENVIRONMENT.md` and `bench/env.sh`; everything it needs to
know about the rules is in `bench/ARTIFACT_GUIDE.md`. One agent can do the
whole job serially (about a day of wall-clock for 98 adapters, dominated by
builds); an orchestrating agent that hands one artifact directory to each
sub-agent, ten at a time, finishes in a few hours. Bulk sub-agents do not need
the strongest model; the orchestrator should be the strongest available.

---

```text
You are working in a fresh clone of HPC-KernelBench (this repository, root =
kernel-papers/). Machine: <one line: site, GPU model, how to get a GPU shell>.
Enable timing runs: <yes/no>.

GOAL. Reproduce on this machine the functional state recorded in the repo:
build every integrated paper artifact, run each through the harness's
correctness gate, and record per artifact what differs from the recorded
outcome. Report; do not "fix" the benchmark to make things pass.

READ FIRST, in this order: CLAUDE.md; bench/README.md; bench/ENVIRONMENT.md
(sections 1-8); bench/ARTIFACT_GUIDE.md (rules 1-10 are binding);
bench/DOMAIN_GUIDE.md point 4. The recorded outcome of every artifact is the
first bold line of bench/artifacts/<track>/<short>/STATUS.md; the summary
page is output/benchmark_status.html.

GROUND RULES
- Never loosen a correctness gate, tolerance, or protocol (benchspecs/,
  bench/kernelbench/). A failed gate is a result, not a bug to work around.
- Never edit kernel code under any source/ checkout. Build-system fixes only
  (arch flags, include/link paths, CUDA-version guards); after such a fix run
  `git -C source diff > source.patch` in that artifact directory and set
  `patch=source.patch` in source.provenance. If the kernel itself would have
  to change to run, record SKIPPED with the evidence and move on.
- Every file you write for an artifact lives in bench/artifacts/<track>/<short>/,
  never only inside source/ (source/ is git-ignored and re-created from
  provenance).
- No machine-specific paths in scripts: use the KB_* variables from
  bench/env.sh and `${VAR:-<default>}`; record the versions actually used in
  STATUS.md.
- Install nothing into a shared Python environment. Per-artifact
  `pip install --target=<artifact>/pylibs` only, as the existing build.sh
  files do.
- Never commit source/ checkouts, build products (.so/.o/.a/ELF binaries),
  logs, data caches, or any file over 1 MB. Check `git status` and
  `git diff --cached --stat` before every commit.
- Bound every filesystem search to this repository. Never run find, grep -r,
  du, tree, or a recursive script over /, /opt, /usr, /home, /global,
  /scratch, or any other shared root. Locate software with `command -v`,
  `module spider`, package metadata, or the KB_* paths.
- Read a directory's STATUS.md before touching it; do not rewrite history in
  it (see step 3d).

STEP 1: ENVIRONMENT (once)
1. Python >= 3.10 in a venv. Install torch 2.8 from the wheel index whose CUDA
   major matches the toolkit you will use (`--index-url
   https://download.pytorch.org/whl/cu128` for CUDA 12.x), then
   `pip install -r bench/requirements.txt`.
2. Edit ONLY the KB_* lines in bench/env.sh: KB_PY, KB_CUDA_HOME (same CUDA
   major as torch), KB_CUDA_MATHLIBS (only for the NVIDIA HPC SDK layout),
   KB_CC/KB_CXX (gcc 12-14; nvcc 12.x rejects 15), KB_HOST_COMPILER_BIN (a
   directory holding gcc/g++ 12; 14 artifacts need it), KB_MPI_ROOT,
   KB_BLAS_LIBDIR (plus KB_BLAS_LIB if the library is not libsci_gnu.so),
   KB_LD_PRELOAD (empty unless a torch extension import fails with
   CXXABI_1.3.15). Then `source bench/env.sh` and run
   `$PY bench/check_env.py` until it reports 0 MISSING. Keep its WARN lines:
   they say which artifacts this machine cannot build and how many build.sh
   files pin sm_80.
3. `cd bench && ./smoke_all.sh` must end with `kernels green: 35 ; failing: 0`
   (CPU only, no GPU needed). If it does not, report the failing kernel and
   the traceback; do not patch the harness.
4. Optional: `bench/artifacts/fetch_sources.sh` pre-fetches all git-ignored
   artifact checkouts (~6 GB) from their .provenance files and reports what
   it cannot fetch. Otherwise every build.sh fetches its own on first run.

STEP 2: GPU ARCHITECTURE
`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`. If it is not 8.0
(A100), read bench/ENVIRONMENT.md section 8 before building anything: 78
build.sh files pin sm_80. `-arch=sm_80` embeds PTX and JIT-compiles on a newer
GPU (works; say so in STATUS.md); `-gencode arch=compute_80,code=sm_80` and
`TORCH_CUDA_ARCH_LIST="8.0"` produce SASS-only binaries that fail with
cudaErrorNoKernelImageForDevice. toolchain.sh exports KB_SM and KB_TORCH_ARCH
from the visible GPU: change the flag in OUR build.sh to `sm_${KB_SM}` /
`${KB_TORCH_ARCH}`; a pin inside the artifact's own Makefile/CMake/setup.py is
a build-system patch (record it in source.patch). Never touch the harness's
own kernels for this.

STEP 3: ARTIFACTS, one directory at a time
Work through every bench/artifacts/<track>/<short>/ that contains adapter.py
(`ls bench/artifacts/*/*/adapter.py`, 98 directories). Parallelize across
directories, never inside one. For each:
 a. Read STATUS.md: the recorded ruling, the gate numbers, the toolchain used,
    and the run command it quotes.
 b. `bench/artifacts/<track>/<short>/build.sh` (exit 0 = built; idempotent).
    It sources toolchain.sh, which re-creates source/ from source.provenance
    and re-applies source.patch on first run. If the build fails: compare with
    the toolchain versions in STATUS.md, fix build-system issues only, and if
    it still fails record BUILD-FAILED with the first real error line.
 c. Functional gate:
      $PY -m kernelbench.runner --kernel <track> --list
        (spec variants + discovered adapters and their IMPL_NAME)
      $PY -m kernelbench.runner --kernel <track> --variant <variant> --smoke --impl <IMPL_NAME>
        for every variant/precision STATUS.md names (LD_PRELOAD comes from
        env.sh). The outcome is the PASS/FAIL per (variant, precision), the
        reported error, and the `N/M runs valid (k unsupported)` line. A
        UNSUPPORTED line is neither pass nor fail.
 d. Append to STATUS.md a section
      ## Reproduction on <machine> (<YYYY-MM-DD>)
    with: GPU and compute capability; nvcc, torch, gcc versions; build outcome;
    gate outcome per variant/precision with the max error; every deviation
    from the recorded ruling and your evidence for its cause. Never delete or
    rewrite earlier text. If the ruling itself changes (a GATE FAILS that now
    passes, a DEFERRED-HARDWARE artifact that builds and gates here), put the
    new ruling in the first bold line and keep the old one after
    "Earlier outcome, kept for the record:" (the status page parses this).
 e. Directories with a REQUIRES_GPU file (spmm/voltrix, quantized-gemm/mxblas)
    build only on the architecture the file names. If this machine has it,
    build and gate them like any other and update STATUS.md as in (d); leave
    the REQUIRES_GPU file in place, it states the requirement.
 f. Directories with STATUS.md but no adapter.py are SKIPPED with a documented
    reason. Do not integrate them unless asked.
 Expectations: most builds take under 5 minutes; cholesky/exageostat takes
 about an hour (builds StarPU and Chameleon); cholesky/*,
 multigrid/{amgt,bootcmatchgx}, spgemm/amgt and graph-pattern-mining/graphset
 link MPI and/or BLAS (KB_MPI_ROOT, KB_BLAS_LIBDIR; link only, nothing is
 launched with mpirun); spmv/spmv-acc needs hipcc (HIP-on-CUDA) and is
 BUILD-FAILED on the reference machine; spmm/mp-spmm downloads its code from
 Zenodo. Torch JIT extensions cache under each artifact directory.

STEP 4: REPORT
- From the repo root: `LD_PRELOAD=$KB_LD_PRELOAD $PY make_status_page.py`
  regenerates output/benchmark_status.html from the STATUS.md files.
- Write bench/REPRODUCTION_<machine>.md: one row per adapter with recorded
  ruling vs observed, grouped as: same / now passes / now fails / build
  failed here / not attempted (why). List every build-system patch and every
  arch-flag change you made, by file.
- Commit in small commits (one per track), never the checkouts or build
  products. Commit messages: what changed and why, one paragraph.

TIMING (only if enabled above). A conforming measurement needs an exclusive
GPU allocation (SLURM_JOB_* set, no other processes on the GPU), locked
clocks, no --smoke, and no --warmup/--reps overrides; the runner prints
`conforming: true` only then and writes bench/results/*.json. Spec input sets
(SuiteSparse matrices, graphs, SIFT1M, ...) download on first use into the
data directories named in each domain module. Without an exclusive GPU, do
the functional gates only.
```
