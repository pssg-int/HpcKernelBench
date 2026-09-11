# fast-bcc (FAST-BCC-on-GPUs) — connected-components — STATUS

**Outcome: SKIPPED — no clean CC-only entry point; the artifact's only
"CC"-named function is the FINAL LABELING STEP of the BCC pipeline itself,
requiring the full upstream spanning-tree + Euler-tour + tag-computation
state as input**

Paper: "GPU Algorithms for Biconnected Components on Large Graphs"
(IPDPS'26). `PAPER_KEY = conf/ipps/SahuARKB26` (matched by title +
`artifact_url` in `../../../output/included.json`).
Artifact: https://github.com/Abhijeetkumar96/FAST-BCC-on-GPUs (commit
`147db596535064ee6ea43efbcab315d5b224a1a6`, `git clone --depth 1` into
`./source/`).

## Task framing

Biconnected components (BCC) is a genuinely different problem from this
track's `connected-components` kernel (cut-vertex/cut-edge structure on an
undirected graph vs. plain weak/strong connectivity). Per the task brief:
"evaluate briefly whether the artifact also exposes a plain CC/spanning-forest
stage wrappable at a clean boundary; if not, SKIP with one-paragraph
evidence, no build."

## Why SKIPPED (evidence)

Both of the artifact's two in-memory implementations —
`source/gpu/with_filter/cc.hxx` (Fast-BCC-Filter, the paper's proposed
method) and `source/gpu/without_filter/cc.hxx` (the GPU port of multicore
Fast-BCC) — expose a function literally named `CC()`:

```cpp
float CC(graph_data& d_input) { ... }
```

but `graph_data` (defined in `common.hxx`, identical shape in both variants)
is **not** a raw graph — it is the BCC pipeline's internal skeleton state:

```cpp
class graph_data {
public:
    int V; long E; int root;
    int *parent;     // rooted Spanning Tree parent array
    int *first, *last;  // Euler-tour first/last occurrence numbering
    int *flag;        // fence-edge / back-edge classification (without_filter)
    // or: uint64_t *sf_edges + int *low, *high (with_filter's tag-based variant)
    int *label;       // output -- "Final BCC Labels" (without_filter's own comment)
    int *comp_head;   // component-head array
    uint64_t *edgelist;
};
```

`CC()` consumes `d_input.parent`/`.first`/`.last`/`.flag` (or
`.sf_edges`/`.low`/`.high` in `with_filter`) — a rooted spanning tree plus
Euler-tour tags that only exist after running the algorithm's OWN earlier,
BCC-specific stages (spanning-tree construction, `euler.cu`'s Euler-tour
numbering, `tags.cu`'s low/high tag computation). There is no entry point
that takes an arbitrary CSR/edge-list and returns connected-components
labels without first running essentially the entire BCC pipeline up to its
last kernel launch. Confirming this is not accidental naming: `with_filter/
cc.hxx` writes `d_input.label = label.release(); // final bcc numbers for
all vertices` — the function's OUTPUT is explicitly documented as BCC
labels, not a general graph-connectivity result; it is called "CC" because
its internal technique is union-find/label-propagation (the same primitive
family Shiloach-Vishkin/Afforest belong to), not because it solves the
`connected-components` track's problem.

Wrapping `CC()` at a "clean boundary" would therefore require building and
running the artifact's full spanning-tree + Euler-tour + tag pipeline just
to reach a function whose output is BCC membership, not CC — i.e., there is
no boundary here that is both (a) clean (a single, independently-meaningful
kernel call, per ARTIFACT_GUIDE.md rule 1) and (b) actually testing plain
connectivity rather than the BCC algorithm's own internal machinery. This
was checked in BOTH `with_filter` and `without_filter` (the two variants
this repo ships) with the same result.

No build was attempted (task's own "SKIP with one-paragraph evidence, no
build" instruction).

## Verdict

`fast-bcc: SKIPPED (both in-memory variants' only "CC"-named function is the
terminal labeling step of the BCC pipeline itself, operating on a rooted-
spanning-tree + Euler-tour-tag skeleton that only exists after running the
algorithm's own upstream BCC-specific stages -- no independently-wrappable
plain-connectivity entry point exists). No build attempted.`
