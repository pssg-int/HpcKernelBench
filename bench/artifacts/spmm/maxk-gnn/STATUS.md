# maxk-gnn (MaxK-GNN) — spmm

**Status: SKIPPED — same artifact defect as `../../gnn-aggregation/maxk-gnn/STATUS.md`: the custom SpMM/SSpMM kernels have no Python entry point and the paper's own training pipeline never calls them**

- Paper: "MaxK-GNN: Extremely Fast GPU Kernel Design for Accelerating Graph Neural
  Networks Training" (ASPLOS'24). `PAPER_KEY = conf/asplos/PengXSHZHKKD24`.
  Rated core / regime partial for spmm. `source/` is a symlink to the
  gnn-aggregation clone (same repository, same commit; see `source.provenance`).
- The SpMM ruling is identical to the gnn-aggregation one: the kernels exist
  as CUDA sources but are unreachable from any shipped binding; wrapping them
  would mean writing the missing host code around an untested kernel (rule 3/7).
