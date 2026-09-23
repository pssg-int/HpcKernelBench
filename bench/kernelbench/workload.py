"""
Workloads and their cost model.

A workload is whatever a kernel consumes — a sparse matrix, a dense shape, a
grid, a transform length, a graph, a data file. All the harness needs from one
is:

  * `describe()` -> a JSON-able dict that identifies it in the result record
  * a registered cost rule so `flops` / `bytes` can be computed for the kernel

Domain modules (sparse.py, dense.py, stencil.py, ...) register their rules here,
which is what keeps harness.py from knowing anything about any particular kernel.
"""

from __future__ import annotations

from typing import Callable, Protocol

# kernel -> fn(workload, params) -> (flop_count, byte_count)
_COST_RULES: dict[str, Callable] = {}
# kernel -> unit label for the primary metric, e.g. "GFLOP/s", "GCUP/s"
_UNITS: dict[str, str] = {}


class Workload(Protocol):
    name: str

    def describe(self) -> dict: ...


def register_cost(kernel: str, rule: Callable, unit: str = "GFLOP/s") -> None:
    """
    rule(workload, params) -> (flops:int, bytes:int)

    `bytes` is the compulsory-traffic lower bound (each datum counted once); the
    harness labels it as such so it is never mistaken for measured traffic.
    """
    _COST_RULES[kernel] = rule
    _UNITS[kernel] = unit


def cost(kernel: str, workload, params: dict) -> tuple[int, int]:
    if kernel not in _COST_RULES:
        raise KeyError(
            f"no cost rule registered for kernel {kernel!r}; "
            f"have {sorted(_COST_RULES)} — the domain module must call "
            "workload.register_cost() at import time")
    return _COST_RULES[kernel](workload, params)


def unit(kernel: str) -> str:
    return _UNITS.get(kernel, "GFLOP/s")


# kernel -> fn(impl_name, workload, params, stats_ms, statistic) -> dict | None
_NATIVE_RULES: dict[str, Callable] = {}


def register_native_metrics(kernel: str, rule: Callable) -> None:
    """
    Optional per-kernel hook for each paper's OWN performance metric(s),
    reported ALONGSIDE the registered primary unit, never replacing it.

    rule(impl_name, workload, params, stats_ms, statistic) -> dict | None
    (None when the implementation has no paper attached, e.g. numpy baselines).
    `workload` is the object after impl.prepare() ran, so any documented
    in-place mutation an adapter made (timesteps, shape) is already visible.
    """
    _NATIVE_RULES[kernel] = rule


def native_metrics(kernel: str, impl_name: str, workload, params: dict,
                   stats_ms: dict, statistic: str) -> dict | None:
    rule = _NATIVE_RULES.get(kernel)
    return rule(impl_name, workload, params, stats_ms, statistic) if rule else None


def registered() -> list[str]:
    return sorted(_COST_RULES)
