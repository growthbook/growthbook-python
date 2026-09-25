# Saved group references v2 verification

Measured on 2026-09-25. No material performance regression was detected in the
tested existing workloads after deferring cycle-set allocation and keeping legacy
arrays on a direct lookup path. This is a local comparison, not proof of zero
regression on every workload or interpreter.

## Reference and correctness

- Implementation prompt: [Notion](https://app.notion.com/p/3dda857e74a0808499d6cfba00a18196).
- Merged reference: [growthbook#6942](https://github.com/growthbook/growthbook/pull/6942).
- Checked monorepo `main` at `e613b6149809c71f5155f6af6cd92a437b447c01`.
- `mongrule.ts`, `core.ts`, and the JS corpus are identical to reference commit
  `023c21040b8878650f0e2cf0eb7ca44ed4b6d2f1`, which the Python port used.
- Corpus comparison: spec 0.9.0, no missing or drifted cases; 41 Python-local extras.
- Both clients run all feature and experiment v2 cases, including prerequisites.
  Additional tests cover plaintext/encrypted payload injection and fetching,
  refreshed membership, malformed groups, and concurrent async evaluations.
- Final verification: 1,154 tests passed, including the MyPy and Pyright typing
  regressions. Standalone MyPy, Pyright, and CI's blocking lint check also pass.
  Pytest reports one existing `event_loop_policy` deprecation warning.

The updated prompt permits capability registration before publication only with
`prerelease: true` on a new version. Remove that flag and regenerate
`CAPABILITIES.md` when publishing. Python has no such entry yet. Remote evaluation
depends on updating the proxy's JS SDK dependency; it does not require a separate
operator implementation in the proxy.

## Method

- Baseline: PR parent `e6e7773cfbe7aa00202708d3fa49bca849797602`.
- Candidate: `75c1c1a1789dd53f55cdecf6dac23084ba8058f8`.
- Python 3.13.15; macOS-26.6.2-arm64-arm-64bit-Mach-O. Same interpreter and installed dependencies.
- Three independent runs per checkout, ordered base/change, change/base,
  base/change. All benchmark processes ran sequentially, without concurrent tests.
- Targeted benchmark: 1,000 warmup evaluations, then seven samples of 20,000
  evaluations per client/workload per run. Reported values are medians of the
  three per-run medians. Client construction, payload loading, result checks,
  and cleanup are excluded. Both value and result source are checked before timing.
- Existing sync benchmark: best of five samples of 100,000 evaluations per run;
  report the median of the three runs.
- Existing async benchmark: 1,000 requests, concurrency 100, simulated 1 ms sticky
  storage I/O; report the median of three runs. Each request evaluates two features.
- Raw samples and candidate source hash: [JSONL](saved-group-references-v2.jsonl).

## Existing behavior: targeted benchmark

Times are microseconds per evaluation; positive changes mean slower.

| Workload | Sync base | Sync change | Delta | Async base | Async change | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| default | 0.915 | 0.950 | +3.9% | 0.999 | 1.016 | +1.7% |
| experiment | 4.653 | 4.590 | -1.4% | 4.537 | 4.683 | +3.2% |
| plain-condition | 1.535 | 1.581 | +3.0% | 1.600 | 1.615 | +0.9% |
| nested-condition | 2.694 | 2.783 | +3.3% | 2.757 | 2.857 | +3.6% |
| legacy-list-10 | 1.646 | 1.701 | +3.3% | 1.764 | 1.807 | +2.4% |
| legacy-list-1000 | 6.273 | 6.199 | -1.2% | 6.226 | 6.295 | +1.1% |
| legacy-array-1000 | 8.757 | 8.485 | -3.1% | 8.687 | 8.772 | +1.0% |
| legacy-not-in-group | 1.615 | 1.664 | +3.1% | 1.763 | 1.759 | -0.2% |

All changes were between -3.1% and +3.9%. Even the unchanged default-value
control varied by +3.9%, so differences of this size should not be interpreted
as a proven speedup or slowdown. The initial implementation showed about
+0.13 µs (+7.5%) in sync `$notInGroup`; the final comparison is +0.05 µs (+3.1%).

## V2 payload cost

These compare equivalent decisions on the candidate checkout: a v2 list vs.
the same legacy list, or a v2 condition group vs. its inline condition.
The old SDK cannot evaluate v2 references, so timing its no-match path would
produce a misleading comparison.

| Equivalent workload | Client | Legacy/inline µs | V2 µs | Added µs |
| --- | --- | ---: | ---: | ---: |
| v2-list-10 | sync | 1.701 | 1.567 | -0.134 |
| v2-list-10 | async | 1.807 | 1.680 | -0.127 |
| v2-list-1000 | sync | 6.199 | 6.047 | -0.152 |
| v2-list-1000 | async | 6.295 | 6.251 | -0.044 |
| v2-condition | sync | 1.581 | 1.750 | +0.169 |
| v2-condition | async | 1.615 | 1.878 | +0.264 |
| v2-nested-condition | sync | 2.783 | 2.992 | +0.209 |
| v2-nested-condition | async | 2.857 | 3.093 | +0.236 |

## Existing benchmark scripts

| Sync workload | Base µs | Change µs | Delta |
| --- | ---: | ---: | ---: |
| default-value | 0.930 | 0.911 | -2.0% |
| experiment-rule | 4.146 | 4.145 | -0.0% |

| Async workload | Base req/s | Change req/s | Throughput delta | Base p95 ms | Change p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-sticky-service | 95,458.9 | 95,855.1 | +0.4% | 0.01 | 0.01 |
| sync-sticky-service | 4,923.1 | 4,813.8 | -2.2% | 21.62 | 21.84 |
| async-sticky-service | 26,321.0 | 26,272.2 | -0.2% | 3.93 | 4.12 |
| async-hot-user | 37,538.6 | 36,426.1 | -3.0% | 2.78 | 2.84 |

Async throughput differences were within about 3%. Event-loop lag measurements
are retained in the raw data; short runs and OS scheduling make maxima noisy.
These existing scripts do not exercise saved-group conditions, which is why
the targeted benchmark is included.

## Reproduce

Activate the development environment. Extract or check out the baseline separately,
then run the same benchmark script against each SDK path:

```bash
python tests/scripts/benchmark_saved_groups.py --sdk-path /path/to/baseline --legacy-only
python tests/scripts/benchmark_saved_groups.py --sdk-path "$PWD" --legacy-only
python tests/scripts/benchmark_saved_groups.py  # includes v2 workloads
python tests/scripts/benchmark_eval_overhead.py 100000
PYTHONPATH=. python tests/scripts/benchmark_async_client.py
```

For the existing scripts, run from each checkout with the same absolute path to
the virtual environment's Python, and set `PYTHONPATH=.` so editable-install
resolution does not accidentally benchmark the wrong checkout. Alternate the
checkout order and repeat; do not run the two benchmarks concurrently.
