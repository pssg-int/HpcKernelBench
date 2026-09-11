# hp-spmm-sddmm (HP-SpMM-SDDMM) — sddmm

**Status: SKIPPED — no SDDMM code in the released artifact (rule 7)**

- Paper: "Fast Sparse GPU Kernels for Accelerated Training of Graph Neural
  Networks", IPDPS'23. `PAPER_KEY = conf/ipps/FanWC23`.
- Artifact: https://github.com/fan1997/HP-SpMM-SDDMM, commit
  `e5a35ed65e2a083cf233a81933359386a519aadc` (2023-12-31 per the repo's own
  commit date), `git clone --depth 1` into `source/` (not built; kept as
  evidence for this SKIPPED verdict).

## Verification (per this task's instruction: verify before building)

`benchspecs/sddmm/spec.yaml`'s own `evidence`/`open_questions` entries for
`conf/ipps/FanWC23` already flagged this as unresolved ("current artifact
snapshot contains only an SpMM benchmark path, no SDDMM code despite
paper/repo title"). Re-verified directly against the freshly cloned repo:

1. **Directory structure** (top-level, bounded listing): `cmake/`,
   `dataset/`, `include/{spmm,starml}/`, `spmm/{mm,starml,utils}/`, `test/`.
   No `sddmm/` directory anywhere, at any depth — the ENTIRE 58-file
   `.cu`/`.cpp`/`.h`/`.cuh` source tree lives under `spmm/`/`include/spmm/`.

2. **The repo's own README.md** (`source/README.md`), verbatim: "The code
   contains high-performance **FP32 SpMM implementations** (for Ampere and
   Hopper Arch)." No SDDMM claim anywhere in the README — only the repo
   *name* and the paper *title* reference SDDMM.

3. **Case-insensitive grep for "sddmm" across every `.cu/.cpp/.h/.cuh/.py/
   .md` file** (bounded to this cloned `source/` tree) turns up exactly two
   hits:
   - `README.md` (the title/citation block only, no code reference).
   - `include/spmm/mm/spmm_common.h:130-135`: six struct fields
     (`sddmm_cs_time_`, `sddmm_cs_throughput_`, `sddmm_dgl_time_`,
     `sddmm_dgl_throughput_`, `sddmm_hp_time_`, `sddmm_hp_throughput_`) —
     **declared but never referenced anywhere else in the codebase**
     (`grep -rn "sddmm_hp_time_\|sddmm_cs_time_\|sddmm_dgl_time_" --include=
     "*.cu" --include="*.cpp" --include="*.h" .` outside this one file: zero
     hits). Vestigial bookkeeping fields for a benchmark-comparison table
     column that was apparently planned but never implemented or wired to
     any kernel — not a real, callable SDDMM path.

Conclusion: this artifact snapshot genuinely does not implement SDDMM
despite the paper/repo title, exactly as `benchspecs/sddmm/spec.yaml`'s
`open_questions` suspected. Per ARTIFACT_GUIDE.md rule 7, SKIPPED with this
file-level evidence; no build attempted. (The repo's real, working `spmm/`
kernel is a candidate for the **spmm** track instead, out of scope here.)
