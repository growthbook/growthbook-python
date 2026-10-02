"""Measure condition evaluation through both public clients, without network I/O.

Compare the same workloads on two checkouts using the same interpreter:

    python tests/scripts/benchmark_saved_groups.py --sdk-path /path/to/base --legacy-only
    python tests/scripts/benchmark_saved_groups.py --sdk-path /path/to/change --legacy-only
    python tests/scripts/benchmark_saved_groups.py  # also measure v2 payloads

Run comparisons sequentially and alternate checkout order to reduce interference
and thermal bias. JSON output includes the imported SDK path and every sample.
Payload loading, correctness checks, and cleanup are outside the timed region.
This measures CPU cost per evaluation; benchmark_async_client.py measures
concurrent requests with simulated sticky-bucket I/O and event-loop lag.
"""

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path


def scenarios():
    """Return equivalent legacy/v2 conditions plus ordinary evaluation controls."""
    attributes = {"id": "u999", "plan": "pro", "country": "US", "tags": ["u998", "u999"]}
    plain = {"plan": "pro", "country": "US"}
    nested = {"$and": [plain, {"$not": {"plan": "free"}}, {"$or": [{"country": "CA"}, plain]}]}
    cases = [
        ("default", None, {}, False),
        ("experiment", None, {}, False),
        ("plain-condition", plain, {}, False),
        ("nested-condition", nested, {}, False),
    ]
    for size in (10, 1000):
        values = [f"u{i}" for i in range(1000 - size, 1000)]
        cases.extend([
            (f"legacy-list-{size}", {"id": {"$inGroup": "g"}}, {"g": values}, False),
            (f"v2-list-{size}", {"$savedGroup": {"id": "g"}}, {
                "g": {"type": "list", "attributeKey": "id", "values": values},
            }, True),
        ])
    values = [f"u{i}" for i in range(1000)]
    cases.extend([
        ("legacy-array-1000", {"tags": {"$inGroup": "g"}}, {"g": values}, False),
        ("legacy-not-in-group", {"id": {"$notInGroup": "g"}}, {"g": ["other"]}, False),
        ("v2-condition", {"$savedGroup": {"id": "g"}}, {
            "g": {"type": "condition", "condition": plain},
        }, True),
        ("v2-nested-condition", {"$savedGroup": {"id": "g"}}, {
            "g": {"type": "condition", "condition": nested},
        }, True),
    ])
    for name, condition, groups, v2 in cases:
        feature = {"defaultValue": True}
        if name == "experiment":
            feature = {"defaultValue": False, "rules": [{"key": "bench", "variations": [True, True], "coverage": 1}]}
        elif condition is not None:
            feature = {"defaultValue": False, "rules": [{"condition": condition, "force": True}]}
        yield name, attributes, {"features": {"flag": feature}, "savedGroups": groups}, v2


async def benchmark(args):
    """Load the selected checkout before importing either client."""
    sdk_path = Path(args.sdk_path).resolve()
    sys.path.insert(0, str(sdk_path))
    import growthbook
    from growthbook import GrowthBook, GrowthBookClient, Options, UserContext

    assert Path(growthbook.__file__).resolve().parent == sdk_path / "growthbook"
    for name, attributes, payload, v2 in scenarios():
        if args.legacy_only and v2:
            continue
        sync_client = GrowthBook(attributes=deepcopy(attributes))
        async_client = GrowthBookClient(Options())
        user = UserContext(attributes=deepcopy(attributes))
        try:
            sync_client.set_payload(payload)
            await async_client.set_payload(payload)
            for client_name in ("sync", "async"):
                expected_source = "defaultValue" if name == "default" else "experiment" if name == "experiment" else "force"
                if client_name == "sync":
                    result = sync_client.eval_feature("flag")
                else:
                    result = await async_client.eval_feature("flag", user)
                assert result.value is True and result.source == expected_source, (name, client_name, result.to_dict())

                samples = []
                for round_index in range(args.rounds + 1):
                    count = min(args.iterations, 1000) if round_index == 0 else args.iterations
                    start = time.perf_counter_ns()
                    if client_name == "sync":
                        for _ in range(count):
                            sync_client.eval_feature("flag")
                    else:
                        for _ in range(count):
                            await async_client.eval_feature("flag", user)
                    elapsed = time.perf_counter_ns() - start
                    if round_index:
                        samples.append(elapsed / count / 1000)
                print(json.dumps({
                    "scenario": name,
                    "client": client_name,
                    "sdk_path": str(sdk_path),
                    "python": platform.python_version(),
                    "iterations": args.iterations,
                    "median_us": statistics.median(samples),
                    "min_us": min(samples),
                    "samples_us": samples,
                }), flush=True)
        finally:
            sync_client.destroy()
            await async_client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk-path", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    if args.iterations < 1 or args.rounds < 1:
        parser.error("--iterations and --rounds must be positive")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
