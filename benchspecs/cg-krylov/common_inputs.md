# cg-krylov — common inputs all four artifacts can run fairly

Goal: one matrix set that PERKS, Mille-feuille, SPCG and BootCMatchGX can all
solve under the same conditions, so "which is fastest" has a real answer.
Researched 2026-10-15 from the artifact source (pinned commits in
`bench/artifacts/cg-krylov/*/source.provenance`) and the papers themselves.

## 1. What each paper actually evaluated on

| | PERKS (ICS'23) | Mille-feuille (SC'24) | SPCG (SC'25) | BootCMatchGX (TPDS'23) |
|---|---|---|---|---|
| Matrices | 20 SuiteSparse SPD, Table 4 (D1–D20), also in `conjugateGradient/data/filelist.txt` | all 230 SPD matrices in SuiteSparse | 114 SuiteSparse SPD (`script_src/python_scripts/prep/matrix_download.py`) | **only generated 3D Poisson, 7-point**: 300³ strong scaling, 130³ per GPU weak scaling. No SuiteSparse |
| Precision | fp64 | fp64 + tile-wise fp32/fp16/fp8 | fp32 only | fp64 |
| Preconditioner | none | none | ILU(0) | AMG (BCMG), FCG |
| How timed | fixed 10,000 iterations | fixed 100 iterations | to-convergence, 1000 cap | to-convergence, rtol 1e-6, 1000 cap |
| RHS / x0 | ones / 0 | A·ones / 0 | ones / 0 | f ≡ 1 (Poisson) |
| GPUs | A100, V100 | A100, MI210 | A100, V100 | P100 (Piz Daint) |
| Baseline | Ginkgo | cuSPARSE, PETSc, Ginkgo | cuSPARSE ILU(0)/ILU(K) PCG | AmgX, hypre |

Sources: PERKS arXiv 2204.02064 §6.2.2 + Table 4; Mille-feuille
ssslab.cn/assets/papers/2024-yang-Millefeuille-final.pdf §V-A; BootCMatchGX
arXiv 2303.02352 §5 + App. A–C. SPCG paper is paywalled (ACM DOI
10.1145/3712285.3759796); its row is from the artifact code only.

## 2. Hard input limits found in the code

**Mille-feuille** (`src/main-cg.cu`, `src/main-cg-mixed.cu`)
- `n = (n / 16) * 16` in `main()`: rows past the last multiple of 16 are
  silently dropped (column count kept), so the solved system is NOT the input
  system unless N % 16 == 0.
- `main-cg` (the binary our harness builds) picks a path by nnz:
  < 10K `cg_solve_inc`, 10K–100K `cg_solve_sync` (single-kernel, fp64),
  >= 100K `cg_solve_reduce` (ordinary multi-kernel fp64 CG, host loop
  `while (iterations < 10000)`). So for nnz >= 100K, `main-cg` runs neither the
  single-kernel design nor mixed precision.
- The paper's method (single kernel + tile-wise mixed precision) is only in
  `main-cg-mixed` (`make cg-mix`), which always calls `cg_solve_inc` with the
  `*_mix_precision` kernels, whatever the nnz. **Not built by our harness.**
- Mixed version: fixed `for (iter = 1; iter <= 100; iter++)` inside the
  kernel, convergence check commented out. Precision per tile is chosen from
  the ABSOLUTE per-step change of p (thresholds 1e-5/1e-6/1e-7/1e-8), so
  matrix/RHS scale changes how much low precision is used.
- Persistent single kernel: work is split to fit `< 6912` warps
  (= 108 SMs × 64, an A100). All blocks must be resident at once; GPUs with
  fewer SMs than an A100 risk deadlock. A100/H100 are fine.

**PERKS** (`conjugateGradient/`)
- No size or shape limits: on-chip caching uses whatever shared memory and
  registers are left over (`cg_dispatcher.cuh`), and the launch is sized from
  occupancy. Reads symmetric `.mtx` correctly (mirrors off-diagonals).
- `--staticiter` = fixed count; without it, it stops on the absolute
  `r1 < tol*tol`.

**SPCG** (`gpu_src/ilu0_gpu/nonsp/main.cpp`)
- fp32 throughout; stop rule `r1 > tol*tol` with `tol = 1e-12f` on the
  absolute, unnormalized squared residual and `max_iter = 1000`. With b = ones,
  fp32 essentially never gets that low, so it normally runs all 1000
  iterations; our patch does not change this.
- No `cusparseXcsrilu02_zeroPivot` check: an ILU(0) breakdown gives silent
  NaN/Inf. Rejects pattern-only `.mtx` files.

**BootCMatchGX**
- Reads symmetric `.mtx` correctly (`readMTXDouble`, `CSR.cu`). Its paper only
  validates AMG on Poisson-type (diagonally dominant, M-matrix) problems;
  ℓ1-Jacobi smoothing is guaranteed to converge only for SPD.
- H100 run in `bench/H100_baselines.md` timed out (rc=124); cause not yet
  known.

## 3. Checking the proposed criteria

| Proposed criterion | Verdict |
|---|---|
| SPD, needed by all four | Correct. |
| N a multiple of 16, for Mille-feuille | Correct. Alternative: pad any matrix with identity rows (diag 1, b 0) up to a multiple of 16 and give the SAME padded file to every code. Solution and SPD are preserved. |
| ≥100K nnz so Mille-feuille runs its real code | **Wrong.** In `main-cg`, ≥100K nnz selects the plain multi-kernel fp64 path. To run the paper's method at any size, build and wrap `main-cg-mixed`. |
| Not too ill-conditioned, for unpreconditioned PERKS/MF | Correct for a to-convergence comparison. PERKS's own paper chose "SPD matrices that can converge in a CG solver". |
| Poisson-like for BootCMatchGX | Correct: its paper used nothing else. |
| Easy enough for fp32 to reach the target, for SPCG | Correct, but SPCG's stop rule must also be changed to a relative tolerance, or it always runs 1000 iterations. |
| 128³ 3D Poisson | Good. N = 2,097,152 = 16 × 131,072; nnz = 14,581,760 (7-point). κ ≈ 6.7e3, so unpreconditioned CG reaches 1e-6 in a few hundred iterations. Closest to BootCMatchGX's own 130³ per GPU (130³ = 2,197,000 is NOT a multiple of 16, so 128³ is the right adjustment). |
| thermomech_dM | **Fails the multiple-of-16 rule**: N = 204,316 = 16 × 12,769 + 12 (sparse.tamu.edu). Usable only with padding. Otherwise a good pick: in PERKS's and SPCG's lists, SPD, 1.42M nnz. |

## 4. Proposed common set

All verified on sparse.tamu.edu (rows, nnz, SPD) and N % 16 == 0.

| Matrix | N | nnz | In which papers' lists | Why |
|---|---|---|---|---|
| Poisson 3D 7-pt 64³ (generated) | 262,144 | 1,810,432 | BCMG-style (also F3R's hpcg grids) | small size, where PERKS's on-chip caching matters most |
| Poisson 3D 7-pt 128³ (generated) | 2,097,152 | 14,581,760 | BCMG-style | headline case, all four fair |
| jnlbrng1 | 40,000 | 199,200 | SPCG, MF | 2D 5-point stencil (AMG-friendly), κ ≈ 183 |
| torsion1 | 40,000 | 197,608 | SPCG, MF | 2D 5-point stencil, κ ≈ 41 |
| finan512 | 74,752 | 596,992 | PERKS, SPCG, MF | only real matrix in 3 papers' lists that passes /16; not Poisson-like (AMG stress case) |
| shallow_water2 | 81,920 | 327,680 | PERKS, SPCG, MF | same as above, CFD |
| apache1 *(hard)* | 80,800 | 542,184 | SPCG, MF, our spec | 3D structural; expect many unpreconditioned iterations |
| cfd1 *(hard)* | 70,656 | 1,825,580 | SPCG, MF, our spec | ill-conditioned; separates preconditioned from not |
| thermomech_dM *(padded to 204,320)* | 204,316 | 1,423,116 | PERKS, SPCG, MF | user's pick, only with identity padding |

Wired into the harness as `sweep:cg-common` (`bench/kernelbench/domains/solvers.py`
`SWEEPS`, mirrored in `spec.yaml` `common_subset`; thermomech_dM left out until
padding is implemented):

    $PY -m kernelbench.runner --kernel cg-krylov --variant cg-kernel-fixed-iter \
        --matrices sweep:cg-common

No real SuiteSparse matrix is in all four papers' lists: BootCMatchGX's paper
used only generated Poisson. The generated Poisson grids are the only inputs
that every paper's own evaluation style covers.

## 5. What must change before the numbers are comparable

1. Wrap `main-cg-mixed` for Mille-feuille (the paper's method), not only `main-cg`.
   *Done 2026-10-21 (not yet run): `bench/artifacts/cg-krylov/millefeuille-mixed/`,
   impl `millefeuille-cg-mixed`.*
2. Use one rule for every code: stop at relative residual ‖r‖/‖r0‖ < 1e-6,
   max 1000 iterations, then verify ‖b − Ax‖/‖b‖ independently in fp64.
   Needs small patches: SPCG (relative tol), Mille-feuille (re-enable the
   commented-out residual stop), PERKS (non-static mode with a relative tol).
3. Same b for everyone (b = A·ones matches Mille-feuille's and BootCMatchGX's
   convention), x0 = 0.
4. Report two numbers per matrix: time per iteration (all four) and
   time-to-solution with setup separated (ILU(0) / AMG setup / tile-format
   conversion each reported on its own line).
5. Time inside the process (each code's own CUDA-event/solve timer), not
   subprocess wall time; see the timing caveats in each `STATUS.md`.
6. Add cuSPARSE CG (fp64) as the common vendor baseline. All three papers
   that name one compare against cuSPARSE or Ginkgo.
