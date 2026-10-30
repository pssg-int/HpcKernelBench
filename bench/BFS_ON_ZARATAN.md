# BFS benchmark on zaratan: BLEST and efg vs their own baselines

Paste-ready steps for a coding agent (or a person) on zaratan. Read
`CLAUDE.md` and `bench/ENVIRONMENT.md` §9 first. All GPU work goes through
`bench/gpu_run.sh`. Do not edit adapters to make a number look better;
report failures as results.

## What is compared

| paper | its claim | baselines integrated here |
|---|---|---|
| BLEST (ICS'26), `blest-tensorcore-bfs` | 4.64x over Gunrock, 13.25x over GAP (H200) | `gunrock-bfs` (Gunrock v2.1.0), `gap-dobfs` (GAP v1.5, CPU) |
| efg (IPDPS'23), `efg-elias-fano-bfs` | 0.82x of cuGraph when the graph fits in GPU memory; 3.8x over out-of-core CSR when it does not | `cugraph-bfs` (pylibcugraph 25.06) |

Not integrated: BerryBees, GSWITCH (BLEST); CGR, Ligra+ (efg).

All implementations use the same 64 fixed-seed roots from the giant
component, the same exact level-array gate, and time one search per call.
Adapters with their own timer record it per call (`gunrock_internal_ms`,
`gap_internal_ms`, `cugraph_internal_ms`); `artifacts/bfs/bfs_summary.py`
uses those.

## 1. Build (once)

```bash
cd bench && source env.sh
bash artifacts/bfs/gap/build.sh                     # CPU only, login node is fine
bash artifacts/bfs/cugraph/build.sh                 # pip venv, needs internet, login node
./gpu_run.sh -t 30 -- 'bash artifacts/bfs/gunrock/build.sh'
./gpu_run.sh -t 30 -- 'bash artifacts/bfs/blest/build.sh && bash artifacts/bfs/efg/build.sh'
```

If a build fails, record the error, do not patch around it silently.

## 2. Smoke gate (small graphs, MIG slice)

```bash
./gpu_run.sh -t 30 -- '$PY -m kernelbench.runner --kernel bfs --variant bfs-kernel-real-graphs --smoke \
  --impl blest-tensorcore-bfs,efg-elias-fano-bfs,cugraph-bfs,gunrock-bfs,gap-dobfs,scipy-bfs \
  --keep-going --out results/bfs_smoke.json'
```

Every implementation must pass the gate before step 3.

## 3. Runs

The inputs are the papers' own graphs, as named sweeps in
`kernelbench/domains/graph.py` (`--matrices sweep:<id>`):

| sweep | graphs | from |
|---|---|---|
| `bfs-quick` | com-Orkut (234M edges), GAP-road (57M) | efg; BLEST |
| `bfs-large` | uk-2005 (936M), twitter7 (1.47B), GAP-web (1.9B) | BLEST; efg; both |
| `bfs-shared` | GAP-web (1.9B), com-Friendster (3.6B) | both papers |
| `bfs-papers` | all six | |

GPU implementations, full A100. Quick set first:

```bash
./gpu_run.sh -g a100 -m 128G -t 120 -- '$PY -m kernelbench.runner --kernel bfs --variant bfs-kernel-real-graphs \
  --impl blest-tensorcore-bfs,efg-elias-fano-bfs,cugraph-bfs,gunrock-bfs \
  --matrices sweep:bfs-quick --keep-going --out results/bfs_gpu_quick.json'
```

Large graphs: one graph per job, `-m 256G -t 240`, same impl list,
`--matrices uk-2005` (then `twitter7`, `GAP-web`),
`--out results/bfs_gpu_<graph>.json`.

com-Friendster last, only if the node has about 200 GB of host RAM free;
drop `gunrock-bfs` (int32 ids, 3.6B edges is over its limit; it would only
be reported as unsupported).

GAP is a CPU code: run it on a whole CPU node, not through `gpu_run.sh`
(which gives 4 cores). Record the node type and thread count:

```bash
srun -N 1 -c 128 -p standard -t 180 --mem=200G bash -c 'cd $PWD && source env.sh && \
  OMP_NUM_THREADS=128 OMP_PROC_BIND=spread $PY -m kernelbench.runner --kernel bfs --variant bfs-kernel-real-graphs \
  --impl gap-dobfs --matrices sweep:bfs-papers --keep-going --out results/bfs_gap.json'
```

(Check the partition name and cores per node with `sinfo` first.)

## 4. Summary

```bash
$PY artifacts/bfs/bfs_summary.py results/bfs_gpu_*.json results/bfs_gap_*.json
```

Report, per paper: the measured speedup per graph and its geomean next to
the paper's claim, which graphs failed or were unsupported and why, and
these caveats:
- BLEST's harness time includes its per-call graph re-upload
  (`blest/STATUS.md`), so BLEST reads slower than in its paper.
- BLEST ran on an H200 in its paper; we use an A100.
- efg's claim is about graphs that do NOT fit in GPU memory; on graphs that
  fit, its own paper expects cuGraph to win.
- GAP's thread count and CPU differ from BLEST's 2x Xeon Gold 6152.
- Direction optimization: GAP yes (alpha 15, beta 18); BLEST, efg, Gunrock,
  cuGraph (default) no.
