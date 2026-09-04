"""Micro-benchmark for per-evaluation overhead with telemetry disabled.

Measures the two paths a hot server loop actually exercises, with no tracking
callback, no usage callback, and no deferred buffer configured — the
worst-case for the telemetry plumbing added in 3.1 (EvaluationContext carries
the callback fields; eval_feature reports usage through a wrapper):

  * default-value path: the cheapest possible evaluation
  * experiment-rule path: a realistic hashed assignment

Run against two checkouts to compare (best of 5 rounds each):

    python tests/scripts/benchmark_eval_overhead.py [N]
    (cd /path/to/main-checkout && python tests/scripts/... )
"""
import sys
import time

sys.path.insert(0, ".")

from growthbook import GrowthBook  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000

gb = GrowthBook(attributes={"id": "u1"}, features={
    "flag": {"defaultValue": True},
    "exp": {"defaultValue": 0, "rules": [{"key": "e", "variations": [0, 1], "coverage": 1}]},
})


def best_of(key, rounds=5):
    best = float("inf")
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(N):
            gb.eval_feature(key)
        best = min(best, time.perf_counter() - start)
    return best


for key, label in (("flag", "default-value"), ("exp", "experiment-rule")):
    took = best_of(key)
    print(f"{label:16s} {took:.3f}s / {N} evals  ({took / N * 1e6:.3f} us/eval)")

gb.destroy()
