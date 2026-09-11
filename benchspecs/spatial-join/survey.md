# Survey — spatial-join (singleton track: 1 paper)

## conf/ics/GengLZ24 — RayJoin: Fast and Precise Spatial Join (ICS 2024)

- **What is measured**: two spatial-join primitives accelerated via GPU ray-tracing hardware
  (NVIDIA RT Cores / OptiX BVH): **LSI** (line segment intersection — given two sets of polyline
  chains R and S, find every crossing pair of segments) and **PIP** (point-in-polygon test — given
  a point set and a polygon set, determine polygon membership per point), plus a **polygon overlay**
  operation (`polyover_exec`) combining LSI+PIP results. The paper's own two headline technical
  contributions, independent of the RT-mapping trick itself, are: (1) recovering high PRECISION
  despite the RT core's inherently limited (approximately fp32-ish) precision support, and (2)
  reducing the BVH (Bounding Volume Hierarchy) index BUILD cost, which is otherwise the dominant
  overhead of RT-accelerated approaches.
- **workloads/inputs**: real US geospatial polygon/polyline datasets — USCounty, Zipcode,
  BlockGroup, WaterBodies (all from ArcGIS Hub) and Lakes-and-Parks (SpatialHadoop). Format: a
  custom **CDB** (chain database) format storing polygons as topologically-linked polyline chains
  with neighbor-face IDs (enables O(1)-ish PIP via chain-neighbor lookups); conversion scripts
  (`misc/wkt2shp.py`, `misc/shp2cdb.py`) are provided for WKT/shapefile sources. Preprocessed CDB
  datasets are hosted externally (Dryad) rather than bundled. The README's updated (Aug 2024)
  cuSpatial-comparison table names 8 specific dataset PAIRS used for evaluation: County⋈Zipcode,
  Block⋈Water, and 6 lake/park (LK*)⋈(PK*) region pairs (AF/AS/AU/EU/NA/SA — apparently continent
  codes).
- **timing protocol**: `query_exec` exposes `-warmup <N>` and `-repeat <N>` CLI flags directly —
  "N warmup rounds" then "N rounds to evaluate, **average time is reported**" (mean, not
  median/min/max) — this is the artifact's own explicit, documented protocol (unusually complete
  compared to most repos surveyed in this project).
- **timing scope**: `-mode grid/lbvh/rt` selects the query algorithm; for `rt` mode specifically,
  BVH index construction cost is real and the artifact's own `-check` flag exists to validate `rt`
  results against the `grid` (presumably ground-truth/brute-force-adjacent) implementation — but
  whether the reported query time INCLUDES or excludes BVH build is not stated in the fetched
  README at the `query_exec` level (the `polyover_exec` tool separately exposes `-serialize` to
  cache a CDB-to-binary conversion, implying format conversion is understood by the authors as a
  distinct, cacheable cost from the query itself).
- **precision & correctness**: `-check` (only functional for `mode=rt`) compares RT-mode results
  against the `grid` implementation's output — i.e. **grid is the artifact's own designated
  reference/ground-truth implementation** for correctness checking, not an external tool. The
  paper's own stated technical challenge is recovering full precision despite RT hardware's limited
  native precision — i.e. correctness here is specifically about NOT losing precision relative to a
  double-precision-exact computational-geometry reference, not merely "close enough."
- **metric**: query execution time (ms); the paper reports speedups from 3.0x to 28.3x over "any
  existing highly optimized method... in high precision," and states RayJoin can join millions of
  polygons in under 460ms.
- **baselines**: `grid` (uniform-grid spatial index) and `lbvh` (a linear/software BVH,
  non-RT-hardware-accelerated — via the `ToruNiina/lbvh` library credited in the README) are BOTH
  compiled into the SAME RayJoin binary as alternative `-mode` values, an apples-to-apples same-
  codebase comparison; **cuSpatial** (NVIDIA's RAPIDS GPU spatial-analytics library) is compared
  externally, with the README documenting a specific correctness/memory bug workaround needed for
  cuSpatial 23.12 and updated PIP timing numbers for cuSpatial 24.06 (a fixed version) — the
  updated table shows cuSpatial 24.06's raw PIP times ranging 236ms-38799ms across the 8 named
  dataset pairs, i.e. RayJoin's own comparison methodology already accounts for a known
  version-dependent correctness bug in one of its baselines, a genuine fairness precedent worth
  preserving in this spec.
- **source**: abstract (`output/included.json`); repo README
  (`gh api repos/pwrliang/RayJoin/contents/README.md`).

## Divergences

- Single-paper track: no cross-paper divergence. The main internal design point is that RayJoin
  itself already provides three query-mode baselines (grid/lbvh/rt) in one binary PLUS an external
  cuSpatial comparison with a documented version-dependent correctness caveat — this spec treats
  all three internal modes as required comparison points (index-build-cost and precision are the
  paper's own stated axes of contribution, best demonstrated by comparing modes within the same
  codebase) and keeps cuSpatial as a required external baseline but pinned to the version (>=24.06)
  the README itself confirms is not affected by the OOM/precision-relevant bug.
