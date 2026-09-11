# gspecpal (GSpecPal) — string-regex-matching

**Status: SKIPPED (evidence-backed; partial build attempted, not completed)**

- Paper: "GSpecPal: Speculation-Centric Finite State Machine Parallelization
  on GPUs" (IPDPS'22, `conf/ipps/WangWQW22` in `../../../output/included.json`).
- Artifact: https://github.com/l1ghtWang/GSpecPal
- Commit cloned: `git clone --depth 1` (see `source/.git`).

## Why this is SKIPPED rather than gated

Two independent, compounding reasons, both investigated in depth before
deciding not to finish:

### 1. Input format is a DFA transition table, not ANML/NFA — architectural
mismatch with this track's workload model

GSpecPal is the track's only DFA-based paper (per spec.yaml's own
`notes_on_fairness`). Its `data/<ruleset>/*_DFA<N>.table` +
`*_AC<N>.txt` files are, CONFIRMED FROM SOURCE (not guesswork —
`src/fsm.cpp::Table::readFromFile`, `src/input.cpp::MappingRule`,
`test/testFramework_0902.cpp`):

- `*_AC<N>.txt`: a whitespace-separated list of int **accept state IDs**.
- `*_DFA<N>.table`: N lines of `MAXSYMBOL` space-separated ints each — the
  DFA's `next_state = table[state][symbol]` transition function.
- `MAXSYMBOL` comes from a `MappingRule` (`src/input.cpp`); the driver
  actually used (`testFramework_0902.cpp` line 25:
  `MappingRule::defineMappingRule()`, no key) is the **identity, 256-symbol**
  rule (`mRules[i]=i` for all bytes) — so for the rulesets this track cares
  about (clamAV/newSnort/powerEN), the table IS a plain, fully-specified
  `nstate x 256` byte-indexed DFA transition table. This part of the format
  is NOT actually undocumented once read from source (spec.yaml's own
  open_questions note, written from a shallower read, is superseded by this
  finding).

The real obstacle is not the file format but the workload MODEL: this
track's domain module (`kernelbench/domains/automata.py`) represents every
workload as an `Automaton` of ANML-style STEs (symbol-set test + activate-
on-match edges, homogeneous/position-automaton semantics) built for the
NFA-family papers (ngAP, gpunfa, and this module's own regex->NFA
generator). A classical DFA's `table[state][byte]->next_state` function
does NOT map onto one STE per state the way an NFA's activate-on-match
graph does: one ANML STE fires on ONE symbol-set test and then activates
ALL its outputs together, but a DFA state must FAN OUT to a DIFFERENT
successor depending on WHICH byte matched. Representing this faithfully
would need, per DFA state, one STE per distinct (successor state, byte
subset) pair — a real conversion (byte-class partition by target state),
not a copy — plus a decision on the reporting semantics GSpecPal's DFA
tables encode implicitly (does the accept-state list mean "report every
cycle in that state" or "report only on the transition INTO the accept
state"? not confirmed from the reachable driver code) and, per the paper's
actual contribution, GSpecPal's own SPECULATIVE chunk-boundary parallel
execution (4 policies: merge/end/round/first, in `script_release/*.sh`) —
its own report/offset reconciliation semantics across those policies were
not chased down in the time available. Building an independent reference
for this shape of workload, correctly, was judged to need meaningfully
more time than the remaining bounded budget allowed, given two artifacts
(gpunfa, ngap) were already built+gated and this is a genuinely different
data model, not a drop-in third NFA-family engine.

### 2. Real, compounding build-portability failures on THIS machine's toolchain

A build attempt WAS made (per ARTIFACT_GUIDE, cheap to try even given (1)
above, in case it turned out easy): `build.sh` present, using the same
`toolchain.sh` pin (nvcc 12.9) and `gcc-native/14` host compiler as
ngAP/gpunfa. Patches attempted (all build-system/portability class, per
ARTIFACT_GUIDE rule 3 — no kernel logic touched):

1. `source/CMakeLists.txt`: `-gencode arch=compute_86,code=sm_86` ->
   `compute_80,code=sm_80` and `CMAKE_CUDA_ARCHITECTURES 86` -> `80`
   (A100, same class of fix as ngAP/gpunfa).
2. `source/CMakeLists.txt`: `CMAKE_CXX_STANDARD 11` -> `17` and both
   `-std=c++11` occurrences -> `-std=c++17` — the artifact's original
   C++11 mode does not compile at all against gcc-14's libstdc++ under
   nvcc 12.9 (`__enable_if_t`/`__is_pair`-class internal STL errors in
   `<bits/alloc_traits.h>`/`<bits/hashtable.h>`, a known class of
   incompatibility between very new host compilers and older CUDA C++11
   mode) — switching to C++17 (matching ngAP's own successful choice)
   cleared this class of error.
3. Four kernel files (`src/kernels_full.cu`, `kernels_hash.cu`,
   `kernels_transform.cu`, `kernels_transform_dump.cu`) were missing
   `#include <cstdio>`, needed for device-side `printf` under gcc-14 —
   patched (one line each).

After all three fixes, a NEW, WIDER wave of missing-`<cstdint>` errors
surfaced, rooted in `include/input.h` (used almost everywhere:
`uint8_t*`/`uint8_t` undeclared, cascading into `src/input.cpp`,
`src/fsm.cpp`, `src/analyzer.cpp` — over a dozen further errors spanning
symbol/pointer arithmetic, not just bare type names, suggesting still more
undiscovered breakage beyond a single missing include). This is the same
gcc-14-stopped-leaking-`<cstdint>`-transitively root cause hit (and fixed
cheaply, 1-2 files each) in both `ngap` and `gpunfa` — but here it recurs
across enough of the codebase that continuing to chase it file-by-file was
judged not worth the remaining time, ON TOP OF the unresolved architectural
question in (1). The build was stopped at this point rather than pushed
further.

## What was NOT attempted

- No further build-fix iteration beyond the point above.
- No adapter.py / DFA-to-STE conversion attempted (see (1)).
- No attempt to run any of the `script_release/*.sh` scripts.

## Disposition

Per ARTIFACT_GUIDE.md rule 7 ("if the artifact genuinely does not fit,
mark SKIPPED with evidence and move to the next candidate") and this
track's own README accounting (5 papers total; 2 CPU/FPGA papers already
out of scope; ngAP and gpunfa-artifact are BUILT+GATED as the newest
2 single-GPU-eligible papers) — GSpecPal is the 3rd newest single-GPU-
eligible candidate and was investigated in real depth (input format fully
decoded from source, two classes of build fix attempted and partially
successful) before being SKIPPED, rather than cheaply skipped without
evidence.
