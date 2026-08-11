"""Benchmark GrowthBookClient under high asyncio concurrency.

Not collected by pytest — run manually, like the other scripts here:

    python tests/scripts/benchmark_async_client.py            # all scenarios
    python tests/scripts/benchmark_async_client.py sync-sticky-service

Simulates an asyncio web service: CONCURRENCY request handlers in flight,
each evaluating an experiment feature (sticky bucket write path) and a plain
flag for a user, against a sticky bucket service with simulated network
latency. No real network involved; feature loading is mocked.

Scenarios (each runs in-process against the installed/current growthbook):

  no-sticky-service        Control. Pure evaluation cost — establishes the
                           CPU floor and that eval itself needs no async.
  sync-sticky-service      Existing AbstractStickyBucketService with blocking
                           I/O. The SDK offloads it to a thread executor.
  async-sticky-service     AbstractAsyncStickyBucketService awaited natively
                           (a well-built one batches get_all_assignments).
  async-hot-user           Async service, every request is the SAME user —
                           exercises the prefetch cache hit + coalescing path.

Metrics:
  throughput_rps    completed requests per second
  eval_p50/p95/p99  per-request latency (two evaluations per request)
  loop_lag_max/mean scheduling delay observed by an unrelated coroutine on
                    the same loop. THE health metric for asyncio services:
                    if this grows, every coroutine in the process is starving,
                    not just GrowthBook calls.
"""
import asyncio
import json
import statistics
import sys
import time
from unittest.mock import patch, AsyncMock

from growthbook import (
    AbstractAsyncStickyBucketService,
    InMemoryStickyBucketService,
)
from growthbook.common_types import Options, UserContext
from growthbook.growthbook_client import GrowthBookClient

CONCURRENCY = 100
TOTAL_REQUESTS = 1000
SERVICE_LATENCY = 0.001  # 1ms simulated Redis round-trip

FEATURES = {
    "features": {
        "checkout-experiment": {
            "defaultValue": "control",
            "rules": [{
                "key": "checkout-exp",
                "variations": ["control", "treatment"],
                "weights": [0.5, 0.5],
                "meta": [{"key": "0"}, {"key": "1"}],
            }],
        },
        "plain-flag": {"defaultValue": True},
    },
    "savedGroups": {},
}


class BlockingRedisLikeService(InMemoryStickyBucketService):
    """Sync service with blocking network latency."""

    def get_all_assignments(self, attributes):
        time.sleep(SERVICE_LATENCY)
        return super().get_all_assignments(attributes)

    def save_assignments(self, doc):
        time.sleep(SERVICE_LATENCY)
        super().save_assignments(doc)


class AsyncRedisLikeService(AbstractAsyncStickyBucketService):
    """Async service; get_all_assignments batched like a Redis MGET."""

    def __init__(self):
        self.docs = {}

    async def get_assignments(self, attributeName, attributeValue):
        return self.docs.get(self.get_key(attributeName, attributeValue))

    async def get_all_assignments(self, attributes):
        await asyncio.sleep(SERVICE_LATENCY)
        docs = {}
        for name, value in attributes.items():
            doc = self.docs.get(self.get_key(name, value))
            if doc:
                docs[self.get_key(name, value)] = doc
        return docs

    async def save_assignments(self, doc):
        await asyncio.sleep(SERVICE_LATENCY)
        self.docs[self.get_key(doc["attributeName"], doc["attributeValue"])] = doc


SCENARIOS = {
    "no-sticky-service": (None, "distinct"),
    "sync-sticky-service": (BlockingRedisLikeService, "distinct"),
    "async-sticky-service": (AsyncRedisLikeService, "distinct"),
    "async-hot-user": (AsyncRedisLikeService, "same"),
}


async def run_scenario(name):
    service_factory, user_mode = SCENARIOS[name]
    service = service_factory() if service_factory else None

    opts = Options(
        api_host="https://localhost.growthbook.io",
        client_key=f"bench-{name}",  # unique key: avoids singleton reuse across scenarios
        sticky_bucket_service=service,
    )

    lags = []
    stop = asyncio.Event()

    async def lag_monitor(interval=0.005):
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            start = loop.time()
            await asyncio.sleep(interval)
            lags.append(loop.time() - start - interval)

    latencies = []

    async def request_handler(client, i):
        user_id = f"user-{i}" if user_mode == "distinct" else "user-hot"
        user = UserContext(attributes={"id": user_id, "country": "US"})
        t0 = time.perf_counter()
        await client.eval_feature("checkout-experiment", user)
        await client.is_on("plain-flag", user)
        latencies.append(time.perf_counter() - t0)

    with patch("growthbook.FeatureRepository.load_features_async",
               new_callable=AsyncMock, return_value=FEATURES), \
         patch("growthbook.growthbook_client.EnhancedFeatureRepository.start_feature_refresh",
               new_callable=AsyncMock), \
         patch("growthbook.growthbook_client.EnhancedFeatureRepository.stop_refresh",
               new_callable=AsyncMock):
        client = GrowthBookClient(opts)
        await client.initialize()
        monitor = asyncio.ensure_future(lag_monitor())

        sem = asyncio.Semaphore(CONCURRENCY)

        async def bounded(i):
            async with sem:
                await request_handler(client, i)

        wall0 = time.perf_counter()
        await asyncio.gather(*[bounded(i) for i in range(TOTAL_REQUESTS)])
        wall = time.perf_counter() - wall0

        stop.set()
        await monitor
        await client.close()

    def pct(data, p):
        return statistics.quantiles(data, n=100)[p - 1] if data else 0.0

    return {
        "scenario": name,
        "throughput_rps": round(TOTAL_REQUESTS / wall, 1),
        "eval_p50_ms": round(pct(latencies, 50) * 1000, 2),
        "eval_p95_ms": round(pct(latencies, 95) * 1000, 2),
        "eval_p99_ms": round(pct(latencies, 99) * 1000, 2),
        "loop_lag_max_ms": round(max(lags) * 1000, 2) if lags else 0,
        "loop_lag_mean_ms": round(statistics.mean(lags) * 1000, 3) if lags else 0,
        "sticky_docs_persisted": len(service.docs) if service else None,
    }


def main():
    names = sys.argv[1:] or list(SCENARIOS)
    for name in names:
        if name not in SCENARIOS:
            sys.exit(f"unknown scenario {name!r}; choose from {list(SCENARIOS)}")
        print(json.dumps(asyncio.run(run_scenario(name))))


if __name__ == "__main__":
    main()
