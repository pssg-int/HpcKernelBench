# FeatGraph — gnn-aggregation — STATUS: SKIPPED (evidence below)

**Outcome: SKIPPED — TVM itself DOES install into an artifact-local target
without building from source, but FeatGraph's own kernel-definition modules
immediately fail against every pip-installable TVM release, which is ~6
years and multiple full internal rewrites newer than the TVM v0.7 this
artifact's README requires.**

Paper: "FeatGraph: A Flexible and Efficient Backend for Graph Neural
Network Systems" (SC'20). `PAPER_KEY = conf/sc/HuYWYZL0ZW20`. Selected as
**core**, regime-**matches** under the revised kernel-centrality rule
(2026-09-05) -- `output/kernel_centrality.json`'s
`gnn-aggregation|conf/sc/HuYWYZL0ZW20` entry: "FeatGraph's whole
contribution is optimizing the SpMM-like GNN aggregation operator via
composable sparse templates and UDFs." Repo:
https://github.com/amazon-science/FeatGraph, commit
`f0a380f276f27cdc00f8bc1706a722faf8342474` (2021-06-17). Cloned into
`source/` (`git clone --depth 1`).

## Task condition and how it was evaluated

Per this task's own instruction ("TVM-based; only if TVM installs into an
artifact-local target without building TVM from source; otherwise SKIP with
evidence"), this was tested directly rather than assumed:

**Step 1 -- does TVM install without building from source? YES.**
```
pip install --target=<artifact-local dir> apache-tvm
```
resolves and installs cleanly from prebuilt wheels (`apache-tvm-0.26.0`,
pulling its own `tvm_ffi` dependency the same way), no compilation, no
network access beyond PyPI. This step of the condition is satisfied.

**Step 2 -- does the artifact's own code run against that TVM? NO --
confirmed by direct import, not assumed.** `source/README.md` states
plainly: **"TVM v0.7 is required"** (`git clone -b v0.7 ...`). PyPI's
`apache-tvm` package only publishes recent releases --
`pip install apache-tvm==0.7` / `==0.14` both fail with "Could not find a
version that satisfies the requirement" (only `0.25.0rc0/rc1/0.25.0/
0.25.0.post1/0.26.0rc0/0.26.0` exist). Importing FeatGraph's own kernel
module against the newest installable release fails immediately:

```
PYTHONPATH=source/python:<tvm-target-dir> python -c "import featgraph.module.spmm"
...
File "source/python/featgraph/module/sddmm.py", line 4, in <module>
    from tvm.topi.util import get_const_tuple
ModuleNotFoundError: No module named 'tvm.topi.util'
```

`tvm.topi.util` (FeatGraph's v0.7-era import, used in BOTH
`module/spmm.py`'s import chain -- `module/__init__.py` imports `sddmm`
before `spmm` -- and directly in `benchmark/bench_vanilla_spmm.py`) was
renamed to `tvm.topi.utils` at some point between v0.7 (2020) and 0.26.0
(current); this single rename is fixable, but it is only the FIRST
statement reached in a 6-year-old TE-based scheduling module, and it lives
in the artifact's own kernel-DEFINITION code (`python/featgraph/module/
{spmm,sddmm}.py`, `python/featgraph/op/vanilla_spmm.py`), not a build
script -- ARTIFACT_GUIDE.md rule 3 puts editing kernel code out of scope.
TVM's internals changed far more substantially than one renamed module
over this span (TE-based scheduling -> TensorIR -> Relax, `tvm.build`'s
signature, device/context APIs (`.ctx` vs `.device`), target
specification) -- there is no confidence that patching this one import
would not immediately surface several more, deeper incompatibilities in
the actual schedule/compute construction logic (which WOULD constitute
touching kernel code), and verifying that safely is outside this
integration's remaining budget.

## Not done

- No further attempt to patch past the confirmed import failure (would
  require editing `python/featgraph/module/*.py`, the kernel-definition
  code itself, per rule 3).
- The artifact-local TVM install used for this test
  (`pip install --target=...`) was removed after the evidence above was
  collected -- no `build.sh`/`adapter.py` is being shipped for this
  artifact, so no persistent TVM target tree is needed.
- `python/featgraph/op/vanilla_sddmm.py` and the SDDMM path were not
  probed separately; `module/__init__.py`'s import order means both SpMM
  and SDDMM fail at the SAME first import regardless.
