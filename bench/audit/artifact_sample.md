# Artifact link audit — fixed sample of 40 (verified/likely), seed 20260807

Method: for every GitHub URL, pulled `{full_name,size,language,archived,description,pushed_at}`
via `gh api repos/OWNER/REPO`; for the one GitLab URL, hit the GitLab REST API + raw README.
For any repo with no description, an empty/generic top-level README, or a name that didn't
obviously match the paper title, fetched the README (and, where needed, drilled into
subdirectories / cited source files) to confirm the code actually corresponds to *this* paper
rather than a same-named or reused project.

Verdict key: OK = repo exists, has real code, plausibly the artifact for this paper.
DEAD = repo/page gone or empty. WRONG-REPO = code is for a different project.
SUSPICIOUS = exists but real doubt about correspondence remains after investigation.

| # | Status (pre) | URL | Verdict | Notes |
|---|---|---|---|---|
| 1 | verified | https://github.com/scale-snu/cheddar-fhe | OK | Repo description is the paper title verbatim; active, C++, non-trivial size. |
| 2 | verified | https://github.com/SuperScientificSoftwareLaboratory/TileSpGEMM | OK | Description cites PPoPP'22 TileSpGEMM paper with matching author list. |
| 3 | verified | https://github.com/IST-DASLab/marlin | OK | FP16xINT4 LLM inference kernel — matches MARLIN abstract directly. |
| 4 | verified | https://github.com/ParCIS/Magicube | OK | Description: quantized sparse SpMM/SDDMM on Tensor Cores — matches title exactly. |
| 5 | verified | https://github.com/AIS-SNU/PathWeaver | OK | Description is the paper title verbatim. |
| 6 | verified | https://github.com/jnyao/ozHOPE | OK | Description is the paper title verbatim. |
| 7 | verified | https://github.com/appl-lab/CuTS | OK | README header = "AE cuTS: Scaling Subgraph Isomorphism on Distributed Multi-GPU Systems Using Trie Based Data Structure", exact match. |
| 8 | verified | https://github.com/xiaoyeli/superlu_dist | OK | General SuperLU_DIST repo (not paper-specific), but its own reference list cites "Unified Communication Optimization Strategies for Sparse Triangular Solver on CPU and GPU Clusters", SC23 — this is the target paper; its algorithms live in this codebase. |
| 9 | likely | https://github.com/jeun-990806/NPD | OK | README: "NPD (Novel PIM Decoder)... JPEG decoder designed to run on PIM-integrated systems" — matches title exactly. |
| 10 | verified | https://github.com/SuperScientificSoftwareLaboratory/PanguLU | OK | Description is the paper title verbatim. |
| 11 | verified | https://github.com/HicrestLaboratory/Popcorn | OK | README: "Popcorn: Accelerating Kernel K-means on GPU using Sparse Linear Algebra", PPoPP 2025 — exact match. |
| 12 | verified | https://github.com/necst/iMFAnt | OK | README title/abstract matches exactly; Zenodo DOI + CGO'24 AE badge present. |
| 13 | likely | https://github.com/hgl71964/cuasmrl | OK (messy) | Repo is archived and top-level README is an unmodified copy of OpenAI Triton's (author built the tool inside a Triton checkout). Actual work lives in `cuasmrl/cuasmrl/` (`ppo.py`, `drl.py`, `autotuner.py`, `selection.py`, `simulated_annealing.py`) — package name `cuasmrl`, and README references cloning `hgl71964/CuAssembler`. Content strongly matches "CuAsmRL: Optimizing GPU SASS Schedules via Deep RL" despite the misleading surface presentation. |
| 14 | likely | https://github.com/YangletLiu/cuTensor_CUDA_Library_for_Transform_based_Tensors | OK | Shared library repo for a family of papers by the same authors; README explicitly lists as ref [2] "T. Zhang, X.-Y. Liu, X. Wang. High Performance GPU tensor completion with tubal-sampling pattern. IEEE TPDS, 2020" — exact title match to the paper being audited. |
| 15 | verified | https://github.com/FFFLJL/Tensile-FGPS | OK | README opens with a link to the exact IEEE paper "A Fine-grained Prefetching Scheme for DGEMM Kernels on GPU with Auto-tuning Compatibility". |
| 16 | verified | https://github.com/JohndeVostok/APE | OK | README: "APE is a method of emulating high-bitwidth computation with low-bitwidth data types" — matches title exactly; real src/include/test dirs. |
| 17 | verified | https://github.com/hexcute/hexcute-bench | OK | README: benchmark scripts for "Hexcute: A Compiler Framework for Automating Layout Synthesis in GPU Programs" — exact match. |
| 18 | verified | https://github.com/ZCCLorg/zccl | OK | README explicitly states ZCCL is based on both IPDPS'24 ZCCL and SC'24 hZCCL papers; source contains `ZCCL_ring_ho.c` ("ho" = homomorphic, matching "Co-Designed Homomorphic Compression" in the hZCCL title) — confirms hZCCL-specific code is present. |
| 19 | verified | https://github.com/lsl036/STM-Multifrontal-QR-Factorization-Empowered-by-GCN | OK | Description is the paper title verbatim. |
| 20 | likely | https://github.com/Edgarrrrrr/T2-RELION | OK | Root README.md is empty, but repo contains `relion.h`, `src/`, `scripts-AE/` with `1_2_build_relion_branch.sh` and a Zenodo fetch script — consistent with "T2-RELION: Task Parallelism, Tensor Core Accelerated RELION" artifact evaluation package. |
| 21 | verified | https://github.com/anonymousSC21/SC21 | OK | README: anonymized submission for "On the Parallel I/O Optimality of Linear Algebra Kernels: Near-Optimal Matrix Factorizations" — exact match. |
| 22 | verified | https://github.com/PASSIONLab/CombBLAS | OK | Description matches Combinatorial BLAS concept; same project, description just predates the "2.0" version bump in the paper title. |
| 23 | verified | https://gitlab.com/pgrete/kathena | OK | GitLab project exists; description/README: "K-Athena is a performance portable structured grid finite volume magnetohydrodynamics code..." — exact match (project itself now points to successor AthenaPK, but the K-Athena code and repo remain intact). |
| 24 | verified | https://github.com/shibatch/sleef | OK | Description is the paper title verbatim (SLEEF vectorized libm). |
| 25 | verified | https://github.com/shengguolsg/PSMMA | OK | Description cites the exact TPDS paper title. |
| 26 | verified | https://github.com/facebookresearch/GPU-DPF | OK | Archived but description matches exactly ("GPU-based Distributed Point Functions and 2-server PIR"); code present (2.7MB, Python). |
| 27 | verified | https://github.com/Daisyforest/SWQsim | OK | README: "Repository for simulation data for 'Closing the "Quantum Supremacy" Gap: Achieving Real-Time Simulation of a Random Quantum Circuit Using a New Sunway Supercomputer'" — exact match. |
| 28 | likely | https://github.com/lsl036/SSpMV | OK | Description matches title concept (sparsity-aware adaptive SpMV format/algorithm selection). |
| 29 | verified | https://github.com/UCLA-VAST/Odyssey | OK | README cites "Suhail Basalama, Jie Wang, Jason Cong. A Comprehensive Automated Exploration Framework for Systolic Array Designs. DAC 2023" — exact match. |
| 30 | likely | https://github.com/prittt/YACCLAB | OK | Author `prittt` is the actual YACCLAB author; project is specifically a connected-components-labeling benchmark, matching "Optimized Block-Based Algorithms to Label Connected Components" closely. |
| 31 | verified | https://github.com/neozhang307/PERKS | OK | Description: "persistent kernel sample implementation of iterative stencil solver and conjugate gradient solver" — matches "PERKS: a Locality-Optimized Execution Model for Iterative Methods". |
| 32 | verified | https://github.com/hpdps-group/ICS23-GPULZ | OK | Description is the paper title verbatim. |
| 33 | verified | https://github.com/microsoft/ConvStencil | OK | README: "ConvStencil: Transform Stencil Computation to Matrix Multiplication on Tensor Cores" — exact match; has real `src/` despite small repo size. |
| 34 | verified | https://github.com/sth1997/GraphSet | OK | README is a generic "AE of SC23" page (no explicit "GraphSet" restatement) but repo is literally named GraphSet and content (pattern matching, clique counting, motif counting, FSM) matches "High Performance Graph Mining through Equivalent Set..." topic; substantial real code (gpu/, src/, code_gen/). |
| 35 | verified | https://github.com/hpcde/spmv-acc | OK | README explicitly cites the exact bibtex entry for "Efficient Algorithm Design of Optimizing SpMV on GPU", HPDC'23. |
| 36 | verified | https://github.com/vortexlab-uclouvain/flups | OK | Description is the paper title verbatim (FLUPS Fourier-based Poisson solver library). |
| 37 | verified | https://github.com/stobis/euler-meets-cuda | OK | README describes Euler Tour, list rank, bridges, LCA CUDA implementations — matches "Euler Meets GPU: Practical Graph Algorithms with Theoretical Guarantee". |
| 38 | verified | https://github.com/srikarchundury/diaq_for_hamsim | OK | Description "Diagonal-Budgeted Hamiltonian Simulation" matches "Diagonal-Budgeted Trotterization for Efficient Quantum Hamiltonian Simulation". |
| 39 | verified | https://github.com/johnpzh/liteform_ae | OK | Description states explicitly it is "the artifact repo for the paper LiteForm: Lightweight and Automatic Format Composition for Sparse Matrix-Matrix Multiplication on GPUs". |
| 40 | verified | https://github.com/GPUPeople/ParallelBatchRCM | OK | Description is the paper title verbatim. |

## Summary

- 40/40 OK — repo exists, has real code, and content plausibly corresponds to the sampled paper.
- 0 DEAD, 0 WRONG-REPO, 0 SUSPICIOUS.
- Two repos are GitHub-archived but still contain intact code: `hgl71964/cuasmrl`, `facebookresearch/GPU-DPF`.
- Weakest-evidence cases (real but indirect correspondence, resolved OK only after drilling past the top-level README/description): `hgl71964/cuasmrl` (unmodified Triton README at root; actual RL scheduler code found one level down), `Edgarrrrrr/T2-RELION` (empty root README, confirmed via `scripts-AE/` and `relion.h`), `xiaoyeli/superlu_dist` and `YangletLiu/cuTensor_CUDA_Library_for_Transform_based_Tensors` (general/shared library repos serving multiple papers, confirmed via in-repo citation lists), `ZCCLorg/zccl` (umbrella repo for the ZCCL paper family, confirmed hZCCL-specific source present), `sth1997/GraphSet` (generic AE README, confirmed by repo name + code content).
