# Homotopy continuation — evaluation-methodology survey

Track: `homotopy-continuation` (1 paper with artifact). GPU path-tracking for
systems of polynomial equations, here in its multiview-geometry role
(camera-pose estimation inside RANSAC).

Evidence basis: no arXiv preprint and no open-access fulltext (IPDPS, paywalled);
grounded in the artifact repository read via `gh api`, whose README and YAML
configuration are unusually explicit about the experimental setup.

## Improved GPU-HC — IPDPS'25, `conf/ipps/ChienAK25`
*Accelerating Homotopy Continuation with GPUs: Application to Trifocal Pose Estimation*
Artifact: <https://github.com/Brown-LEMS/Trifocal_Pose_Estimation_using_Improved_GPUHC>
(verified — README names both this IPDPS'25 paper and the companion BMVC'24 one)

**Problem instance** (`problems/trifocal_2op1p_30x30/gpuhc_settings.yaml`, verbatim)
- `problem_name: trifocal_2op1p_30x30` — trifocal relative pose from 2 oriented
  points + 1 point, formulated as **30 polynomial equations in 30 unknowns**.
- `Num_Of_Vars: 30`, `Num_Of_Params: 33`, **`Num_Of_Tracks: 312`** — 312
  homotopy paths per solve. This is the unit of work.
- Solver step controls: `GPUHC_Max_Steps: 80`,
  `GPUHC_Max_Correction_Steps: 3`, `GPUHC_Num_Of_Steps_to_Increase_Delta_t: 4`.
- Problem-structure constants: `dHdx_Max_Terms: 8`, `dHdx_Max_Parts: 5`,
  `dHdt_Max_Terms: 16`, `dHdt_Max_Parts: 6`, `Max_Order_Of_T: 2`,
  `Num_Of_Coeffs_From_Params: 37`.
- Start system and target parameters ship as data (`start_params.txt`,
  `start_sols.txt`, `target_params.txt`), plus a Julia monodromy script that
  generated the formulation — so the instance is fully reproducible.

**Dataset**
- `RANSAC_Dataset: Synthetic` — the released data is a **noiseless synthetic
  curve dataset** (synthcurves-multiview-3d-dataset). The README states the
  real-world dataset "shall be released soon", i.e. the noisy regime that
  RANSAC exists for is *not* in the public artifact.

**Configuration knobs that change the work performed**
- `Abort_RANSAC_by_Good_Sol: false` in the shipped default. The README
  identifies early RANSAC termination as an **IPDPS-paper contribution** (absent
  from the earlier BMVC version). Turning it on changes how many solves happen.
- The README lists four speedup sources: (i) direct parameter-homotopy
  evaluation, (ii) GPU utilization improvements (register pressure, instruction
  cache misses), (iii) **pruning homotopy paths**, (iv) **early termination of
  RANSAC**. Items (iii) and (iv) reduce work; items (i) and (ii) increase
  efficiency at fixed work. The headline speedups mix all four.
- `Num_Of_GPUs: 1` default, multi-GPU supported.
- `Num_Of_Cores: 4` for the CPU-HC reference path.

**Reported results** (README)
- 9.17×, 13.75×, 17.06×, 17× over the original GPU-HC on V100, A100, H100,
  GH200 respectively. Baseline is the authors' own prior GPU-HC.
- `arxived_GPU_code/` retains earlier kernel variants specifically so the
  incremental-improvement ablation can be recreated — good practice, and the
  natural basis for separating efficiency gains from work-reduction gains.

**Timing protocol**
- No warmup/repetition/statistic constants are exposed in the YAML or README;
  the main file is `cmd/magmaHC-main.cpp`. The README's operational advice
  ("use as many CPU cores as possible ... CPU-HC could take a long time")
  concerns the reference path, not the measured GPU path.
- **Warmup count, repetition count and statistic are therefore not recoverable**
  from the public artifact without reading the driver in depth; treat as unknown.

**Correctness**
- Solutions are written to `Output_Write_Files/`. The natural gate — how many of
  the 312 tracked paths reach valid finite solutions, and whether the recovered
  pose matches ground truth — is available in principle (synthetic data has
  exact ground truth) but no automated pass/fail check is documented.

**Dependencies**
- MAGMA ≥2.5.4 (complex arithmetic, host and device), OpenBLAS 0.3.x,
  YAML-CPP, CUDA 11.x/12.x, CMake 3.2x+.

## Divergences

Single-paper track; the tensions are between the artifact's framing and what a
fair benchmark must separate:

1. **Work reduction is conflated with kernel efficiency.** Path pruning and
   RANSAC early termination make the solver do *less*; register-pressure and
   parameter-evaluation improvements make it do the *same work faster*. A single
   wall-clock speedup over the previous GPU-HC cannot be attributed, and a
   future implementation could "win" purely by pruning more aggressively at the
   cost of solution quality.
2. **Success rate is the hidden variable.** HC path tracking can fail (paths
   diverge, correction fails to converge within `GPUHC_Max_Correction_Steps`).
   Speed at a lower success rate is not a speedup. Nothing in the artifact
   makes success rate a reported quantity.
3. **Noiseless synthetic data flatters early termination.** With clean
   correspondences, RANSAC finds a good hypothesis almost immediately, so
   `Abort_RANSAC_by_Good_Sol: true` looks free. On noisy real data — precisely
   the data not yet released — the same setting trades accuracy for time.
4. **Step-control constants are performance-relevant and instance-specific.**
   `GPUHC_Max_Steps: 80` and `GPUHC_Max_Correction_Steps: 3` bound the work per
   path; changing them changes both time and success rate. They must be pinned
   in any comparison.
5. **Baseline is self-referential.** The comparison is against the authors'
   earlier GPU-HC. That is legitimate for an ablation but does not establish
   standing against general-purpose HC solvers.
