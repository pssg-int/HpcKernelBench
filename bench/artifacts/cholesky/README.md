# cholesky — artifact integration outcomes

8 papers surveyed (`kernel-papers/output/benchmark_groups.json`, group
`cholesky`), backed by 3 distinct repositories. Per ARTIFACT_GUIDE.md's
2026-08-07 scope ruling, only NVIDIA GPU / single-card artifacts are
attempted; CPU-only, multi-GPU/distributed-only, and FPGA artifacts are
SKIPPED with a one-line reason and no clone.

## Newest-GPU-eligible accounting

Per repository, ranked by publication year, marking which papers actually
carry a `nvidia-gpu` platform tag (not just "the newest paper on this
repo"):

- **`ecrc/hicma-x`** backs 3 papers: LtaiefACRSKDBAK24 (SC'24,
  **nvidia-gpu**), CaoAPBLKD22 (IPDPS'22, cpu-only), CaoPABLKD21 (IPDPS'21,
  cpu-only). Newest GPU-eligible: **LtaiefACRSKDBAK24 (SC'24)** — attempted,
  **BUILT+GATED**.
- **`ecrc/exageostat`** backs 4 papers: CaoAAPNBDGKLS22 (SC'22, arm-cpu +
  distributed, **no** nvidia-gpu tag despite being the newest on this repo),
  AbdulahCPBDGKLS22 (TPDS'22, **nvidia-gpu**), MondalAL0GK22 (IPDPS'22,
  cpu-only), SalvanaAHLSGK21 (TPDS'21, cpu-only). Newest GPU-eligible:
  **AbdulahCPBDGKLS22 (TPDS'22)** — attempted, **BUILT, GATE-FAILED** (GPU
  numerics bug isolated and documented, see `exageostat/STATUS.md`).
- **`anonymousSC21/SC21`** (KwasniewskiKBZS21, SC'21) — cpu + distributed
  only, no GPU tag at all on this repo's single paper. SKIPPED, no clone.

## Per-paper outcomes

| Paper (dblp key) | Venue | Platform | Repo | Outcome |
|---|---|---|---|---|
| `conf/sc/LtaiefACRSKDBAK24` | SC'24 | nvidia-gpu, distributed | ecrc/hicma-x | **BUILT+GATED (self-reported residual)** — `hicma-x/` adapter `hicma-x-potrf-tlr`, `cholesky-dense-single-node-fp64-kernel` gate PASSES (err 4.11e-03 / 1.87e-03 <= tol 30.0) but the residual is the artifact's own `--check` print on its own internally generated STARS-H operand (no operand/factor export → no independent recomputation possible). See `hicma-x/STATUS.md`. |
| `conf/ipps/CaoAPBLKD22` | IPDPS'22 | cpu, distributed | ecrc/hicma-x (same repo) | SKIP — platform out of scope (no nvidia-gpu tag); no separate clone (same repo already integrated above for the GPU-eligible paper on this lineage). |
| `conf/ipps/MondalAL0GK22` | IPDPS'22 | cpu, distributed | ecrc/exageostat | SKIP — platform out of scope (no nvidia-gpu tag); repo already integrated above for its GPU-eligible paper. |
| `conf/sc/CaoAAPNBDGKLS22` | SC'22 | arm-cpu, distributed | ecrc/exageostat | SKIP — platform out of scope (arm-cpu + distributed, no nvidia-gpu tag despite being the newest paper on this repo). |
| `journals/tpds/AbdulahCPBDGKLS22` | TPDS'22 | distributed, nvidia-gpu, cpu | ecrc/exageostat | **BUILT, GATE-FAILED** (GPU path) — `exageostat/` adapter `exageostat-chameleon-dpotrf`; Chameleon's own `--check` reliably passes on CPU (`--gpus 0`) and reliably fails on GPU (`--gpus 1`) at every N tried, isolated (via a differential `dgemm` check with a printed residual) to a genuine GPU-numerics bug, not a build or gate-script defect. Recorded per rule 4, not patched. Note: this paper's own text describes a **PaRSEC**-based Cholesky; the mapped repo (`ecrc/exageostat`) is StarPU/Chameleon-based — see `exageostat/STATUS.md`'s "Provenance mismatch" section (the PaRSEC lineage this paper describes appears to now live in `hicma-x`'s own "HiCMA-PaRSEC", which cites this exact paper in its README). |
| `conf/ipps/CaoPABLKD21` | IPDPS'21 | cpu, distributed | ecrc/hicma-x (same repo) | SKIP — platform out of scope (no nvidia-gpu tag). |
| `conf/sc/KwasniewskiKBZS21` | SC'21 | cpu, distributed | anonymousSC21/SC21 (unmaintained; spec.yaml's own `artifact_note` names the maintained successor `eth-cscs/conflux`, also cpu/distributed-only) | SKIP — platform out of scope (no nvidia-gpu tag on either the archived snapshot or its maintained successor); no clone. |
| `journals/tpds/SalvanaAHLSGK21` | TPDS'21 | distributed, cpu | ecrc/exageostat (same repo) | SKIP — platform out of scope (no nvidia-gpu tag). |

## Summary

- 2 of 8 papers were GPU-eligible (one per repository lineage, the newest
  GPU-tagged paper on each); both attempted.
- `hicma-x`: **BUILT+GATED**. A substantial, real HiCMA-PaRSEC (PaRSEC/DPLASMA
  in-tree superbuild) stack, requiring a nontrivial LAPACKE header-guard
  collision fix (a 3-function compatibility shim, force-included, zero
  submodule source touched) plus two real Cray-environment/CUDA-toolchain
  bugs (documented in `hicma-x/STATUS.md`) — none of them specific to this
  paper's own numerics. `cholesky-dense-single-node-fp64-kernel` gates
  clean at machine-precision-adjacent residuals.
- `exageostat`: **BUILT**, but the GPU-accelerated correctness gate
  genuinely **fails** — isolated to a real numerical bug in this Chameleon
  build's CUDA path (confirmed via Chameleon's own unit-test binary,
  cross-validated on two different operations), not a build defect or an
  artifact of this adapter's own gate script. An honest negative result per
  ARTIFACT_GUIDE.md rule 4.
- 6 of 8 papers were SKIPPED, all for the same reason (CPU/distributed-only
  platform, no NVIDIA GPU tag) — cheap skips, no clone, consistent with the
  2026-08-07 scope ruling.

## dense.py change

`kernelbench/domains/dense.py`'s `reference_cholesky` gained one optional,
documented hook (`params["external_A_fro_norm"]`) used by both adapters
above, since neither artifact's driver accepts an externally-supplied
operand (both generate their own SPD matrix internally). Default behavior
(hook absent) is unchanged for every existing caller
(`scipy-cholesky`/`torch-cholesky`) — verified via
`--variant cholesky-dense-single-node-fp64-kernel --impl scipy-cholesky
--smoke` before and after, identical PASS results (see either artifact's
`STATUS.md` for the exact transcript).
