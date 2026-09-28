# Saved group references v2 benchmarks

Measured 2026-09-25 on Python 3.13.15, macOS 26.6.2 ARM64:

- Baseline: `e6e7773cfbe7aa00202708d3fa49bca849797602`.
- Candidate: `75c1c1a1789dd53f55cdecf6dac23084ba8058f8`.
- Three sequential runs per checkout, alternating order with identical dependencies.
- Saved-group benchmarks: 1,000 warmups, seven samples of 20,000 evaluations;
  results are medians of the three per-run medians. Setup and loading are excluded.
- Existing sync benchmark: best of five 100,000-evaluation samples per run.
  Async benchmark: 1,000 requests, concurrency 100, simulated 1 ms sticky storage.

| Measurement | Result |
| --- | --- |
| Existing conditions through both clients | Evaluation time changed -3.1% to +3.9% |
| Existing sync default/experiment benchmark | Evaluation time changed -2.0% / approximately 0% |
| Existing async throughput benchmark | Throughput changed -3.0% to +0.4% |
| V2 list groups vs equivalent legacy lists | No slowdown measured |
| V2 condition groups vs equivalent inline conditions | Added 0.17–0.26 µs per evaluation |

No material regression was detected locally. The unchanged default-value control
also varied by +3.9%; these results do not establish zero regression on every
workload or Python version.

## Reproduce

Use the same virtual environment's Python for both checkouts. Run sequentially,
alternate checkout order, and repeat three times:

```bash
python tests/scripts/benchmark_saved_groups.py --sdk-path /path/to/baseline --legacy-only
python tests/scripts/benchmark_saved_groups.py --sdk-path /path/to/candidate --legacy-only
python tests/scripts/benchmark_saved_groups.py --sdk-path /path/to/candidate
```

Run the existing benchmarks from each checkout with `PYTHONPATH=.` and the same
absolute path to Python:

```bash
PYTHONPATH=. python tests/scripts/benchmark_eval_overhead.py 100000
PYTHONPATH=. python tests/scripts/benchmark_async_client.py
```
