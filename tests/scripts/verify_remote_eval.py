#!/usr/bin/env python3
"""verify_remote_eval.py — manual end-to-end verification for remote eval.

Two modes:

  python3 tests/scripts/verify_remote_eval.py
      Default. Spawns an in-process fake proxy on a random localhost port and
      runs every scenario against it. Zero external setup. The fake proxy
      implements the actual wire contract (POST /api/eval/{key} → thin
      {features, savedGroups} payload with rule.tracks where appropriate).

  python3 tests/scripts/verify_remote_eval.py --real
      Runs against a real growthbook-proxy. Requires:
          GB_PROXY_URL    e.g. http://localhost:3300
          GB_CLIENT_KEY   SDK Connection key from the GrowthBook UI
      Some scenarios that depend on specific feature shapes are skipped in
      real mode (the proxy returns whatever features exist in your GB account,
      not the fixtures the fake proxy serves).

See ./README.md for proxy bootstrap and the running friction log.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import socket
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

# Make the in-tree growthbook package importable when run from a checkout.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from growthbook import (  # noqa: E402
    FeatureRepository,
    GrowthBook,
    InMemoryStickyBucketService,
    feature_repo,
)
from growthbook.common_types import Options, UserContext  # noqa: E402
from growthbook.growthbook_client import (  # noqa: E402
    EnhancedFeatureRepository,
    GrowthBookClient,
    SingletonMeta,
)


# ----------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------

_NO_COLOR = os.environ.get("NO_COLOR") or not sys.stdout.isatty()
G = "" if _NO_COLOR else "\033[32m"
R = "" if _NO_COLOR else "\033[31m"
Y = "" if _NO_COLOR else "\033[33m"
D = "" if _NO_COLOR else "\033[2m"
B = "" if _NO_COLOR else "\033[1m"
X = "" if _NO_COLOR else "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {G}✓{X} {msg}")


def _fail(msg: str, detail: str = "") -> None:
    suffix = f"  {D}{detail}{X}" if detail else ""
    print(f"  {R}✗{X} {msg}{suffix}")


def _skip(msg: str, reason: str = "") -> None:
    suffix = f"  {D}({reason}){X}" if reason else ""
    print(f"  {Y}–{X} {msg}{suffix}")


def _info(msg: str) -> None:
    print(f"  {D}{msg}{X}")


def _section(title: str) -> None:
    print(f"\n{B}[{title}]{X}")


# ----------------------------------------------------------------------------
# POST counter (works in both fake and real modes — wraps the SDK's network
# methods at the class level so every instance is observed).
# ----------------------------------------------------------------------------


class PostCounter:
    def __init__(self) -> None:
        self.count = 0
        self.bodies: List[Dict[str, Any]] = []
        self._installed = False

    def reset(self) -> None:
        self.count = 0
        self.bodies.clear()

    def install(self) -> None:
        if self._installed:
            return
        self._installed = True

        orig_sync = FeatureRepository._post
        counter = self

        def wrapped_sync(self, url, payload, headers=None):  # type: ignore[no-untyped-def]
            counter.count += 1
            counter.bodies.append(payload)
            return orig_sync(self, url, payload, headers)

        FeatureRepository._post = wrapped_sync  # type: ignore[assignment]

        orig_async = EnhancedFeatureRepository._fetch_and_decode_post_async

        async def wrapped_async(self, api_host, client_key, payload):  # type: ignore[no-untyped-def]
            counter.count += 1
            counter.bodies.append(payload)
            return await orig_async(self, api_host, client_key, payload)

        EnhancedFeatureRepository._fetch_and_decode_post_async = wrapped_async  # type: ignore[assignment]


COUNTER = PostCounter()


# ----------------------------------------------------------------------------
# State reset between scenarios
# ----------------------------------------------------------------------------


def _reset_state() -> None:
    # Drop both the metaclass attr and any shadow class attr (test_growthbook_client.py
    # uses `EnhancedFeatureRepository._instances = {}` which would shadow the metaclass).
    EnhancedFeatureRepository._instances = {}
    SingletonMeta._instances.clear()
    feature_repo.clear_cache()
    feature_repo._etag_cache.clear()
    feature_repo._feature_update_callbacks = []
    COUNTER.reset()


# ----------------------------------------------------------------------------
# Fake proxy (in-process aiohttp server in a background thread)
# ----------------------------------------------------------------------------


def _filtered_features(attrs: Dict[str, Any]) -> Dict[str, Any]:
    """Mimic what the real growthbook-proxy emits: a thin {features, savedGroups}
    payload with server-side filtering already applied. Rules carry only `force`
    and optional `tracks` (deferred experiment tracking)."""
    features: Dict[str, Any] = {}

    # verify-string: matches id=="u1" only.
    if attrs.get("id") == "u1":
        features["verify-string"] = {
            "defaultValue": "off",
            "rules": [{"id": "rule-u1", "force": "on-for-u1"}],
        }
    else:
        features["verify-string"] = {"defaultValue": "off"}

    # verify-bool: matches country=="DE" only.
    if attrs.get("country") == "DE":
        features["verify-bool"] = {
            "defaultValue": False,
            "rules": [{"id": "rule-de", "force": True}],
        }
    else:
        features["verify-bool"] = {"defaultValue": False}

    # verify-tracks: always experiment-backed. Used by the rule.tracks scenario.
    features["verify-tracks"] = {
        "defaultValue": False,
        "rules": [
            {
                "force": True,
                "tracks": [
                    {
                        "experiment": {"key": "track-exp", "variations": [0, 1]},
                        "result": {
                            "variationId": 1,
                            "inExperiment": True,
                            "value": True,
                            "hashUsed": True,
                            "hashAttribute": "id",
                            "hashValue": str(attrs.get("id", "unknown")),
                            "featureId": "verify-tracks",
                            "key": "1",
                        },
                    }
                ],
            }
        ],
    }

    return {"features": features, "savedGroups": {}}


class FakeProxy:
    """Background-thread aiohttp server implementing the /api/eval wire contract."""

    def __init__(self) -> None:
        self.client_key = "sdk-fake-1234"
        self.host = "127.0.0.1"
        self.port = self._pick_port()
        self.url = f"http://{self.host}:{self.port}"
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._runner: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _pick_port() -> int:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    async def _handle(self, request):  # type: ignore[no-untyped-def]
        from aiohttp import web

        body = await request.json()
        # Allow scenarios to artificially slow a POST by passing _slow in attrs.
        attrs = body.get("attributes") or {}
        if attrs.get("_slow"):
            await asyncio.sleep(0.05)
        return web.json_response(_filtered_features(attrs))

    def start(self) -> None:
        from aiohttp import web

        ready = threading.Event()

        def run() -> None:
            try:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
                app = web.Application()
                app.router.add_post("/api/eval/{client_key}", self._handle)
                self._runner = web.AppRunner(app)
                self._loop.run_until_complete(self._runner.setup())
                site = web.TCPSite(self._runner, self.host, self.port)
                self._loop.run_until_complete(site.start())
                ready.set()
                self._loop.run_forever()
            except Exception:
                ready.set()
                traceback.print_exc()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        ready.wait(timeout=5)
        # Probe until the socket accepts a connection (server might be a tick
        # behind the event flag).
        deadline = time.time() + 3
        while time.time() < deadline:
            try:
                s = socket.create_connection((self.host, self.port), timeout=0.1)
                s.close()
                return
            except OSError:
                time.sleep(0.02)
        raise RuntimeError("Fake proxy never accepted a connection")

    def stop(self) -> None:
        if not self._loop or not self._loop.is_running():
            return
        # Gracefully shut down the AppRunner from inside its own loop so aiohttp
        # closes its idle keep-alive handlers (otherwise CPython prints
        # "Task was destroyed but it is pending!" warnings at exit).
        async def _shutdown() -> None:
            if self._runner is not None:
                await self._runner.cleanup()

        fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
        try:
            fut.result(timeout=2)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=2)


@dataclass
class ProxyHandle:
    url: str
    client_key: str
    mode: str  # "fake" or "real"


# ----------------------------------------------------------------------------
# Scenario registry
# ----------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    fn: Callable[..., Any]
    needs_fake: bool  # True if the scenario depends on fake-proxy fixture content
    is_async: bool


SCENARIOS: List[Scenario] = []


def sync_scenario(name: str, needs_fake: bool = False) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        SCENARIOS.append(Scenario(name, fn, needs_fake, is_async=False))
        return fn

    return deco


def async_scenario(name: str, needs_fake: bool = False) -> Callable[[Callable], Callable]:
    def deco(fn: Callable) -> Callable:
        SCENARIOS.append(Scenario(name, fn, needs_fake, is_async=True))
        return fn

    return deco


# ----------------------------------------------------------------------------
# Sync scenarios
# ----------------------------------------------------------------------------


@sync_scenario("Initial POST on construction with correct body shape", needs_fake=False)
def s_initial(proxy: ProxyHandle) -> None:
    _reset_state()
    gb = GrowthBook(
        api_host=proxy.url,
        client_key=proxy.client_key,
        attributes={"id": "u1", "country": "US"},
        forced_variations={"exp-1": 2},
        url="/checkout",
        remoteEval=True,
    )
    assert COUNTER.count == 1, f"expected 1 POST, got {COUNTER.count}"
    body = COUNTER.bodies[0]
    assert body == {
        "attributes": {"id": "u1", "country": "US"},
        "forcedFeatures": [],
        "forcedVariations": {"exp-1": 2},
        "url": "/checkout",
    }, f"unexpected body: {json.dumps(body)}"
    gb.destroy()


@sync_scenario("Response parsed, eval returns rule-forced value", needs_fake=True)
def s_eval_uses_response(proxy: ProxyHandle) -> None:
    _reset_state()
    gb = GrowthBook(
        api_host=proxy.url,
        client_key=proxy.client_key,
        attributes={"id": "u1"},
        remoteEval=True,
    )
    val = gb.get_feature_value("verify-string", "fallback")
    assert val == "on-for-u1", f"expected 'on-for-u1', got {val!r}"
    gb.destroy()


@sync_scenario("Same payload → cache hit (no extra POST)")
def s_cache_hit(proxy: ProxyHandle) -> None:
    _reset_state()
    gb = GrowthBook(
        api_host=proxy.url,
        client_key=proxy.client_key,
        attributes={"id": "u1"},
        remoteEval=True,
    )
    initial = COUNTER.count
    gb.set_attributes({"id": "u1"})  # identical payload
    assert COUNTER.count == initial, f"expected 0 new POSTs, got {COUNTER.count - initial}"
    gb.destroy()


@sync_scenario("Different attributes → cache miss, new POST")
def s_cache_miss(proxy: ProxyHandle) -> None:
    _reset_state()
    gb = GrowthBook(
        api_host=proxy.url,
        client_key=proxy.client_key,
        attributes={"id": "u1"},
        remoteEval=True,
    )
    gb.set_attributes({"id": "u2"})
    assert COUNTER.count == 2, f"expected 2 POSTs (init + setter), got {COUNTER.count}"
    assert COUNTER.bodies[-1]["attributes"] == {"id": "u2"}
    gb.destroy()


@sync_scenario("set_url triggers refetch")
def s_set_url(proxy: ProxyHandle) -> None:
    _reset_state()
    gb = GrowthBook(
        api_host=proxy.url, client_key=proxy.client_key,
        attributes={"id": "u1"}, remoteEval=True,
    )
    before = COUNTER.count
    gb.set_url("/page-A")
    assert COUNTER.count == before + 1, f"expected +1 POST, got +{COUNTER.count - before}"
    assert COUNTER.bodies[-1]["url"] == "/page-A"
    gb.destroy()


@sync_scenario("set_forced_variations triggers refetch")
def s_set_forced(proxy: ProxyHandle) -> None:
    _reset_state()
    gb = GrowthBook(
        api_host=proxy.url, client_key=proxy.client_key,
        attributes={"id": "u1"}, remoteEval=True,
    )
    before = COUNTER.count
    gb.set_forced_variations({"exp-1": 1})
    assert COUNTER.count == before + 1, f"expected +1 POST, got +{COUNTER.count - before}"
    assert COUNTER.bodies[-1]["forcedVariations"] == {"exp-1": 1}
    gb.destroy()


@sync_scenario("cache_key_attributes narrows the cache key")
def s_cache_key_attrs(proxy: ProxyHandle) -> None:
    _reset_state()
    gb = GrowthBook(
        api_host=proxy.url, client_key=proxy.client_key,
        attributes={"id": "u1", "country": "US"},
        cacheKeyAttributes=["id"],
        remoteEval=True,
    )
    before = COUNTER.count
    # country is NOT in cacheKeyAttributes → must be a cache hit
    gb.set_attributes({"id": "u1", "country": "FR"})
    assert COUNTER.count == before, (
        f"expected 0 new POSTs (country not in cache_key_attributes), got {COUNTER.count - before}"
    )
    gb.destroy()


@sync_scenario("All 5 validation guards raise at construction")
def s_validation(proxy: ProxyHandle) -> None:
    checks = [
        ("missing client_key",
         dict(api_host=proxy.url, remoteEval=True),
         "client_key for remote eval"),
        ("missing/empty api_host",
         dict(client_key=proxy.client_key, remoteEval=True),
         "Must specify api_host"),
        ("decryption_key + remote_eval",
         dict(api_host=proxy.url, client_key=proxy.client_key,
              decryption_key="x", remoteEval=True),
         "Encryption is not available"),
        ("sticky_bucket_service + remote_eval",
         dict(api_host=proxy.url, client_key=proxy.client_key,
              sticky_bucket_service=InMemoryStickyBucketService(), remoteEval=True),
         "sticky_bucket_service is not compatible"),
        ("stale_while_revalidate + remote_eval",
         dict(api_host=proxy.url, client_key=proxy.client_key,
              stale_while_revalidate=True, remoteEval=True),
         "stale_while_revalidate is not compatible"),
        ("cloud host + remote_eval",
         dict(api_host="https://cdn.growthbook.io", client_key=proxy.client_key,
              remoteEval=True),
         "Cloud host does not support remote eval"),
    ]
    for label, kwargs, match in checks:
        try:
            GrowthBook(**kwargs)
        except ValueError as e:
            assert match in str(e), f"{label}: error message changed: {e!r}"
        else:
            raise AssertionError(f"{label}: expected ValueError")


@sync_scenario("rule.tracks fires trackingCallback", needs_fake=True)
def s_tracks(proxy: ProxyHandle) -> None:
    _reset_state()
    tracked: List[Any] = []

    def cb(experiment, result, user_context):
        tracked.append((experiment.key, result.variationId))

    gb = GrowthBook(
        api_host=proxy.url, client_key=proxy.client_key,
        attributes={"id": "u1"},
        on_experiment_viewed=cb,
        remoteEval=True,
    )
    gb.eval_feature("verify-tracks")
    assert tracked == [("track-exp", 1)], f"unexpected tracks: {tracked}"
    gb.destroy()


@sync_scenario("Two instances with same client_key see their own filtered results", needs_fake=True)
def s_no_cross_pollution(proxy: ProxyHandle) -> None:
    _reset_state()
    gb1 = GrowthBook(
        api_host=proxy.url, client_key=proxy.client_key,
        attributes={"id": "u1"}, remoteEval=True,
    )
    gb2 = GrowthBook(
        api_host=proxy.url, client_key=proxy.client_key,
        attributes={"id": "u2"}, remoteEval=True,
    )
    v1 = gb1.get_feature_value("verify-string", "fallback")
    v2 = gb2.get_feature_value("verify-string", "fallback")
    assert v1 == "on-for-u1", f"gb1 expected 'on-for-u1', got {v1!r}"
    assert v2 == "off", f"gb2 expected 'off' (rule not visible), got {v2!r}"
    gb1.destroy()
    gb2.destroy()


# ----------------------------------------------------------------------------
# Async scenarios
# ----------------------------------------------------------------------------


async def _new_async_client(proxy: ProxyHandle, **opts: Any) -> GrowthBookClient:
    _reset_state()
    client = GrowthBookClient(Options(
        api_host=proxy.url,
        client_key=proxy.client_key,
        remote_eval=True,
        refresh_strategy=None,
        **opts,
    ))
    await client.initialize()
    return client


@async_scenario("Initial fetch happens on first eval, not in initialize()")
async def a_lazy_fetch(proxy: ProxyHandle) -> None:
    client = await _new_async_client(proxy)
    # initialize() should NOT have POSTed
    assert COUNTER.count == 0, f"initialize() unexpectedly POSTed {COUNTER.count} time(s)"
    await client.is_on("verify-string", UserContext(attributes={"id": "u1"}))
    assert COUNTER.count == 1, f"expected 1 POST after first eval, got {COUNTER.count}"
    await client.close()


@async_scenario("Same UserContext → cache hit on subsequent evals")
async def a_cache_hit(proxy: ProxyHandle) -> None:
    client = await _new_async_client(proxy)
    uc = UserContext(attributes={"id": "u1"})
    await client.is_on("verify-string", uc)
    await client.is_on("verify-string", uc)
    await client.is_on("verify-string", uc)
    assert COUNTER.count == 1, f"expected 1 POST total, got {COUNTER.count}"
    await client.close()


@async_scenario("Different UserContext → cache miss + new POST")
async def a_cache_miss(proxy: ProxyHandle) -> None:
    client = await _new_async_client(proxy)
    await client.is_on("verify-string", UserContext(attributes={"id": "u1"}))
    await client.is_on("verify-string", UserContext(attributes={"id": "u2"}))
    assert COUNTER.count == 2, f"expected 2 POSTs, got {COUNTER.count}"
    await client.close()


@async_scenario("preload warms the cache; subsequent eval = pure-local")
async def a_preload(proxy: ProxyHandle) -> None:
    client = await _new_async_client(proxy)
    uc = UserContext(attributes={"id": "u1"})
    await client.preload_remote_eval(uc)
    assert COUNTER.count == 1, f"expected 1 POST during preload, got {COUNTER.count}"
    await client.is_on("verify-string", uc)
    assert COUNTER.count == 1, f"expected 0 extra POSTs after preload, got {COUNTER.count - 1}"
    await client.close()


@async_scenario("Inflight coalescing: 20 concurrent evals = 1 POST", needs_fake=True)
async def a_coalescing(proxy: ProxyHandle) -> None:
    client = await _new_async_client(proxy)
    # The fake proxy slows responses when attrs._slow is truthy, which gives the
    # 20 coroutines time to all observe the inflight future before it resolves.
    uc = UserContext(attributes={"id": "u-coalesce", "_slow": True})
    results = await asyncio.gather(*[
        client.is_on("verify-string", uc) for _ in range(20)
    ])
    assert COUNTER.count == 1, f"expected 1 POST for 20 concurrent evals, got {COUNTER.count}"
    # All 20 results must agree (each got the same response).
    assert len(set(results)) == 1, f"results disagreed: {set(results)}"
    await client.close()


@async_scenario("LRU eviction at remote_eval_cache_size")
async def a_lru(proxy: ProxyHandle) -> None:
    client = await _new_async_client(proxy, remote_eval_cache_size=3)
    for uid in ("A", "B", "C", "D"):  # 4th evicts A
        await client.is_on("verify-string", UserContext(attributes={"id": uid}))
    await client.is_on("verify-string", UserContext(attributes={"id": "A"}))  # re-POST
    assert COUNTER.count == 5, f"expected 5 POSTs (4 distinct + 1 re-fetch), got {COUNTER.count}"
    await client.close()


@async_scenario("Validation guards on the async client")
async def a_validation(proxy: ProxyHandle) -> None:
    checks = [
        ("missing client_key",
         Options(api_host=proxy.url, remote_eval=True),
         "client_key for remote eval"),
        ("empty api_host",
         Options(api_host="", client_key=proxy.client_key,
                 remote_eval=True, refresh_strategy=None),
         "Must specify api_host"),
        ("None api_host",
         Options(api_host=None, client_key=proxy.client_key,
                 remote_eval=True, refresh_strategy=None),
         "Must specify api_host"),
        ("decryption_key + remote_eval",
         Options(api_host=proxy.url, client_key=proxy.client_key,
                 decryption_key="x", remote_eval=True),
         "Encryption is not available"),
        ("sticky_bucket_service + remote_eval",
         Options(api_host=proxy.url, client_key=proxy.client_key,
                 sticky_bucket_service=InMemoryStickyBucketService(), remote_eval=True),
         "sticky_bucket_service is not compatible"),
        ("STALE_WHILE_REVALIDATE + remote_eval (the default refresh strategy)",
         Options(api_host=proxy.url, client_key=proxy.client_key,
                 remote_eval=True),
         "STALE_WHILE_REVALIDATE is not compatible"),
        ("cloud host + remote_eval",
         Options(api_host="https://cdn.growthbook.io", client_key=proxy.client_key,
                 remote_eval=True, refresh_strategy=None),
         "Cloud host does not support remote eval"),
    ]
    for label, options, match in checks:
        try:
            GrowthBookClient(options)
        except ValueError as e:
            assert match in str(e), f"{label}: error message changed: {e!r}"
        else:
            raise AssertionError(f"{label}: expected ValueError")


@async_scenario("rule.tracks fires trackingCallback on the async client", needs_fake=True)
async def a_tracks(proxy: ProxyHandle) -> None:
    tracked: List[Any] = []

    def cb(experiment, result, user_context):
        tracked.append((experiment.key, result.variationId))

    _reset_state()
    client = GrowthBookClient(Options(
        api_host=proxy.url, client_key=proxy.client_key,
        on_experiment_viewed=cb,
        remote_eval=True,
        refresh_strategy=None,
    ))
    await client.initialize()
    await client.eval_feature("verify-tracks", UserContext(attributes={"id": "u1"}))
    assert tracked == [("track-exp", 1)], f"unexpected tracks: {tracked}"
    await client.close()


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------


def _run_one(scenario: Scenario, proxy: ProxyHandle, loop: asyncio.AbstractEventLoop) -> bool:
    try:
        if scenario.is_async:
            loop.run_until_complete(scenario.fn(proxy))
        else:
            scenario.fn(proxy)
    except AssertionError as e:
        _fail(scenario.name, str(e))
        return False
    except Exception as e:
        _fail(scenario.name, f"{type(e).__name__}: {e}")
        if os.environ.get("VERBOSE"):
            traceback.print_exc()
        return False
    _ok(scenario.name)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real", action="store_true",
                        help="Run against a real growthbook-proxy (requires GB_PROXY_URL + GB_CLIENT_KEY).")
    parser.add_argument("--only", help="Run only scenarios whose names contain this substring.")
    args = parser.parse_args()

    print(f"{B}GrowthBook Python SDK — remote_eval verification{X}")
    fake_proxy: Optional[FakeProxy] = None
    if args.real:
        url = os.environ.get("GB_PROXY_URL")
        key = os.environ.get("GB_CLIENT_KEY")
        if not url or not key:
            print(f"{R}--real requires GB_PROXY_URL and GB_CLIENT_KEY{X}", file=sys.stderr)
            return 2
        proxy = ProxyHandle(url=url, client_key=key, mode="real")
        print(f"Mode: {B}REAL{X}  →  {url}  (client_key=…{key[-4:]})")
        print(f"{D}Scenarios marked 'fixture-dependent' will be skipped — the real{X}")
        print(f"{D}proxy returns whatever features exist in your GB account.{X}")
    else:
        fake_proxy = FakeProxy()
        fake_proxy.start()
        proxy = ProxyHandle(url=fake_proxy.url, client_key=fake_proxy.client_key, mode="fake")
        print(f"Mode: {B}FAKE{X}  →  {fake_proxy.url}  (in-process)")

    COUNTER.install()

    sync_scenarios = [s for s in SCENARIOS if not s.is_async]
    async_scenarios = [s for s in SCENARIOS if s.is_async]

    def maybe_filter(items: List[Scenario]) -> List[Scenario]:
        if not args.only:
            return items
        return [s for s in items if args.only in s.name]

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    passed = failed = skipped = 0
    t0 = time.time()
    try:
        for label, items in (("sync", maybe_filter(sync_scenarios)),
                             ("async", maybe_filter(async_scenarios))):
            if not items:
                continue
            _section(label)
            for s in items:
                if s.needs_fake and proxy.mode != "fake":
                    _skip(s.name, "fixture-dependent; fake mode only")
                    skipped += 1
                    continue
                if _run_one(s, proxy, loop):
                    passed += 1
                else:
                    failed += 1
    finally:
        loop.close()
        if fake_proxy:
            fake_proxy.stop()

    print()
    total = passed + failed
    elapsed = time.time() - t0
    if failed == 0:
        print(f"{G}{B}{passed}/{total} passed{X}  ({skipped} skipped) in {elapsed:.1f}s")
        return 0
    print(f"{R}{B}{failed} failed{X}, {passed} passed ({skipped} skipped) in {elapsed:.1f}s")
    return 1


if __name__ == "__main__":
    sys.exit(main())
