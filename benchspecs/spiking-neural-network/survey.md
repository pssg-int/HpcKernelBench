# Survey — spiking-neural-network (singleton track: 1 paper)

## journals/tpds/QuZFZ20 — High Performance Simulation of Spiking Neural Network on GPGPUs (TPDS 2020)

- **What is measured**: BSim, a code-generation framework for simulating Spiking Neural Networks
  (SNN, Leaky-Integrate-and-Fire/LIF neuron model at the moment the README was fetched — the
  abstract also claims a general fine-grained-IR approach) on single- and multi-GPGPU. Contributed
  optimizations: (1) a fine-grained network IR, (2) cross-population/-projection parallelism
  exploration, (3) sparsity-aware load balancing across populations/synapses, (4) multi-GPU
  support.
- **workloads/inputs**: the repo's `test/gpu/standard_test.cpp` synthetic benchmark: a "CUBA IF"
  (current-based leaky-integrate-and-fire) network — a forward chain of populations where ~80% of
  connections are local (front/rear neighbor populations only) and ~20% are long-range/remote,
  with configurable `number_of_populations`, `number_of_neurons_per_population`, and 4 connection-
  probability-like float parameters (0.7/0.5/0.6/0.3 in the README's low-rate example) plus a
  final integer parameter (`6`) controlling simulated duration/steps. Three named firing-rate
  regimes are given as canonical test points: 100 Hz, 500 Hz, 2000 Hz (achieved by varying the
  4 float parameters, e.g. `0.7 1.3 1 2 1 50` for 2000 Hz). No SuiteSparse-style named benchmark
  suite exists for SNN topologies in this artifact — the standard_test generator IS the benchmark.
- **timing protocol**: not specified in the top-level README (no explicit warmup/repeat/timer
  convention visible); the paper's own reported result is a **speedup factor vs. GeNN** (a
  well-known GPU SNN simulator), 1.41x-9.33x depending on configuration — implying at minimum one
  timed run per (topology, firing-rate) configuration was compared against a matching GeNN run,
  but the exact protocol (wall-clock vs. device timer, single-run vs. averaged) is not recoverable
  from the fetched source.
- **timing scope**: whole-simulation wall time for a fixed simulated duration (the SNN literature's
  standard metric is real-time factor: simulated-biological-time / wall-clock-time, or simply total
  wall time for N simulated ms) — the exact scope (does it include network-construction/IR-
  compilation time, or steady-state timestep loop only) is not stated in the README.
- **precision & correctness**: `build.sh release float` / `build.sh release double` toggle
  single/double-precision; a "release" build additionally logs the "overall firing rate of each
  neuron" (`GFire.log`) and a "log" build further records membrane voltage (`g_v.data`) — these are
  the artifact's own correctness/inspection outputs, but there is no visible automated pass/fail
  correctness gate (e.g. spike-train match vs. a reference simulator) wired into the benchmark
  driver.
- **metric**: speedup (dimensionless x) vs. GeNN, reported per configuration (1.41x-9.33x range).
- **baselines**: GeNN (GPU-accelerated SNN simulator, the field's standard comparison point); the
  paper's abstract also states BSim "outperforms other simulators much more" without naming them
  in the fetched abstract text.
- **source**: abstract (`output/included.json`); repo README
  (`gh api repos/CRAFT-THU/BSim/contents/README.md`).

## Divergences

- Single-paper track: no cross-paper divergence. The main gap is that the artifact's own timing
  protocol (warmup, repetitions, statistic, exact timer) is not documented in the top-level README
  — this spec must therefore specify a protocol from field convention (SNN-simulator benchmarking
  standard: wall time for a fixed simulated duration, repeated across independent random seeds
  since spike generation involves the topology's built-in randomization) rather than one lifted
  directly from BSim's own numbers.
