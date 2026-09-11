# RASSM — STATUS

**Outcome: SKIPPED — no SpMV in the artifact**

Paper: "RASSM: Residue-based Acceleration of Single Sparse Matrix Computation
via Adaptive Tiling", ASPLOS 2025. PAPER_KEY = `conf/asplos/JainGC25`.
Repo: `https://github.com/gt-tinker/RASSM`
(commit `HEAD` at clone time; `git clone --depth 50`).

This was flagged in advance ("KNOWN from our spmv survey: the public
artifact implements SpMM/SDDMM ONLY, no SpMV") and verified quickly per the
integration brief rather than spending build budget on it.

## Evidence

```
$ grep -ril "spmv" .          # from the repo root, case-insensitive, all files
(zero hits, .git excluded)

$ find code/include -maxdepth 1 -type d
code/include/spmm
code/include/matrices
code/include/utils
code/include/sddmm
```

`code/include/` has kernel implementations for `spmm/` (jstream.h,
kstream.h, simple.h, taco.h) and `sddmm/` (simple.h, tiled.h) — no `spmv/`
directory, no SpMV-named function, type, or file anywhere in the tree.

The repo's own README confirms the same scope: "Once you have built the
executable `rassm`, use the provided
`$RASSM_HOME/scripts/run_basic_test.sh` script ... This should run the
`rassm` program for all the baselines and `rassm` itself for **the SpMM
kernel**." SpMV is never mentioned.

## Verdict

`rassm: SKIPPED (no SpMV kernel in the artifact -- code/include/ has spmm/
and sddmm/ only, README's own basic test is scoped to "the SpMM kernel",
grep -ril "spmv" over the full tree returns zero hits; matches the
pre-flagged known finding from our spmv survey)`
