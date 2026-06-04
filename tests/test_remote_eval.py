"""Tests for remote evaluation support.

Remote eval mode POSTs the user's targeting context to
`/api/eval/{client_key}` on a self-hosted proxy. The server returns the same
`{features, savedGroups}` shape (rules filtered server-side) and the SDK runs
its normal local `eval_feature` over the result.
"""

import asyncio
import json
import pytest
from unittest.mock import MagicMock, patch

from growthbook import GrowthBook, feature_repo, FeatureRepository, InMemoryStickyBucketService
from growthbook.common_types import Options, UserContext
from growthbook.growthbook_client import GrowthBookClient, SingletonMeta, EnhancedFeatureRepository


def _make_post_response(body):
    """Build a urllib3-shaped HTTPResponse mock with JSON body."""
    resp = MagicMock()
    resp.status = 200
    resp.data = json.dumps(body).encode("utf-8")
    return resp


# Default response body used by most async tests — a single boolean flag
# defaulting to True. Tests that care about specific feature shapes (rule.tracks,
# cross-pollution, etc.) inline their own payload.
DEFAULT_BODY = {"features": {"flag1": {"defaultValue": True}}, "savedGroups": {}}


def _reset_repo():
    """Clear shared singleton repo state between tests."""
    feature_repo.clear_cache()
    # Drop registered callbacks so prior test instances don't fire on new ones
    feature_repo._feature_update_callbacks = []
    # Drop ETag entries from prior CDN-mode tests
    feature_repo._etag_cache.clear()


# ---------------------------------------------------------------------------
# Sync GrowthBook tests
# ---------------------------------------------------------------------------


class TestRemoteEvalSyncValidation:
    def setup_method(self):
        _reset_repo()

    def test_missing_client_key_raises(self):
        with pytest.raises(ValueError, match="client_key for remote eval"):
            GrowthBook(api_host="https://proxy.example.com", remoteEval=True)

    def test_missing_api_host_raises(self):
        """sync `api_host: str = ""` defaults to empty — must reject in
        remote_eval mode, otherwise `_get_remote_eval_url` silently falls
        through to the Cloud CDN, surfacing as an opaque 404."""
        with pytest.raises(ValueError, match="api_host .* for remote eval"):
            GrowthBook(client_key="sdk-test", remoteEval=True)  # api_host omitted

    def test_decryption_key_raises(self):
        with pytest.raises(ValueError, match="Encryption is not available"):
            GrowthBook(
                api_host="https://proxy.example.com",
                client_key="k",
                decryption_key="d",
                remoteEval=True,
            )

    def test_sticky_bucket_service_raises(self):
        with pytest.raises(ValueError, match="sticky_bucket_service is not compatible"):
            GrowthBook(
                api_host="https://proxy.example.com",
                client_key="k",
                sticky_bucket_service=InMemoryStickyBucketService(),
                remoteEval=True,
            )

    def test_stale_while_revalidate_raises(self):
        with pytest.raises(ValueError, match="stale_while_revalidate is not compatible"):
            GrowthBook(
                api_host="https://proxy.example.com",
                client_key="k",
                stale_while_revalidate=True,
                remoteEval=True,
            )

    def test_cloud_host_raises(self):
        with pytest.raises(ValueError, match="Cloud host does not support remote eval"):
            GrowthBook(
                api_host="https://cdn.growthbook.io",
                client_key="k",
                remoteEval=True,
            )

    def test_cloud_subdomain_raises(self):
        with pytest.raises(ValueError, match="Cloud host does not support remote eval"):
            GrowthBook(
                api_host="https://anything.growthbook.io",
                client_key="k",
                remoteEval=True,
            )


class TestRemoteEvalSyncWireFormat:
    def setup_method(self):
        _reset_repo()

    def test_post_url_and_body_shape(self):
        body = DEFAULT_BODY
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1", "country": "US"},
                forced_variations={"exp-1": 2},
                url="/checkout",
                remoteEval=True,
            )

        assert mock_post.call_count == 1
        # _post(self, url, payload, headers)
        call = mock_post.call_args
        assert call.args[0] == "https://proxy.example.com/api/eval/sdk-test"
        assert call.args[1] == {
            "attributes": {"id": "u1", "country": "US"},
            "forcedFeatures": [],
            "forcedVariations": {"exp-1": 2},
            "url": "/checkout",
        }
        assert call.args[2]["Content-Type"] == "application/json"

    def test_eval_uses_response_features(self):
        body = DEFAULT_BODY
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)):
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            assert gb.is_on("flag1") is True

    def test_user_agent_header_present(self):
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            GrowthBook(
                api_host="https://proxy.example.com",
                client_key="abcd1234",
                attributes={"id": "u1"},
                remoteEval=True,
            )
        headers = mock_post.call_args.args[2]
        assert headers["User-Agent"].startswith("Gb-Python")


class TestRemoteEvalSyncRefetch:
    def setup_method(self):
        _reset_repo()

    def test_set_attributes_triggers_refetch(self):
        body = DEFAULT_BODY
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            assert mock_post.call_count == 1
            gb.set_attributes({"id": "u2"})
            assert mock_post.call_count == 2

    def test_set_forced_variations_triggers_refetch(self):
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            assert mock_post.call_count == 1
            gb.set_forced_variations({"exp-1": 1})
            assert mock_post.call_count == 2

    def test_user_ctx_synced_from_instance_at_eval_time(self):
        """Every field that has an instance-level counterpart must flow into
        _user_ctx at construction, via the setter, AND as a safety-net resync
        at eval time — so callbacks (on_feature_usage, on_experiment_viewed,
        event_logger) always see the instance's current state."""
        body = DEFAULT_BODY
        seen: list = []

        def feature_usage_cb(key, result, user_context):
            seen.append({
                "forced_features": dict(user_context.forced_features),
                "forced_variations": dict(user_context.forced_variations),
                "attributes": dict(user_context.attributes),
                "url": user_context.url,
            })

        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)):
            # 1) Constructor wiring — every relevant field on _user_ctx is populated.
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                forced_features={"a": 1},
                forced_variations={"exp-1": 0},
                url="/start",
                on_feature_usage=feature_usage_cb,
                remoteEval=True,
            )
            assert gb._user_ctx.forced_features == {"a": 1}
            assert gb._user_ctx.forced_variations == {"exp-1": 0}
            assert gb._user_ctx.url == "/start"

            # 2) Setters keep _user_ctx in sync.
            gb.set_forced_features({"b": 2})
            assert gb._user_ctx.forced_features == {"b": 2}
            gb.set_forced_variations({"exp-1": 1})
            assert gb._user_ctx.forced_variations == {"exp-1": 1}

            # 3) Eval-time safety net: direct mutations bypassing the setters
            #    still get surfaced through the centralized sync helper.
            gb._forcedFeatures = {"c": 3}
            gb._forcedVariations = {"exp-1": 2}
            gb._attributes = {"id": "u9"}
            gb._url = "/checkout"
            gb.eval_feature("flag1")

            assert seen[-1] == {
                "forced_features": {"c": 3},
                "forced_variations": {"exp-1": 2},
                "attributes": {"id": "u9"},
                "url": "/checkout",
            }, "_sync_user_ctx_from_instance() must propagate ALL fields to callbacks"

    def test_forced_features_wire_format_and_cache_key_exclusion(self):
        """forced_features ships as [[k, v], ...] in the POST body but is
        deliberately excluded from the cache key (matches JS — the proxy
        doesn't filter on it). So `set_forced_features` alone is a cache
        hit (no new POST); the new value ships on the next actual refetch
        triggered by an attribute/url/forced_variations change."""
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                forced_features={"banner": "v2"},  # constructor wiring
                remoteEval=True,
            )
            # 1) Constructor sends forcedFeatures on the first POST.
            assert mock_post.call_args.args[1]["forcedFeatures"] == [["banner", "v2"]]
            # 2) set_forced_features doesn't force a network round-trip
            #    (forcedFeatures isn't in the cache key) — it just updates state.
            gb.set_forced_features({"banner": "v3", "promo": True})
            assert mock_post.call_count == 1
            # 3) When a cache-busting setter fires, the NEW forced_features
            #    value ships in the next POST body.
            gb.set_attributes({"id": "u2"})
            assert mock_post.call_count == 2
            assert mock_post.call_args.args[1]["forcedFeatures"] == [["banner", "v3"], ["promo", True]]

    def test_set_url_triggers_refetch(self):
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            assert mock_post.call_count == 1
            gb.set_url("/different-page")
            assert mock_post.call_count == 2


class TestRemoteEvalSyncCaching:
    def setup_method(self):
        _reset_repo()

    def test_same_payload_hits_cache(self):
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            initial = mock_post.call_count
            # set_attributes with same value: cache hit
            gb.set_attributes({"id": "u1"})
            assert mock_post.call_count == initial

    def test_different_attributes_cache_miss(self):
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            gb.set_attributes({"id": "u2"})
            assert mock_post.call_count == 2

    def test_cache_key_attributes_narrows(self):
        """If cacheKeyAttributes=['id'], changing a non-listed attr is a cache HIT."""
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1", "country": "US"},
                cacheKeyAttributes=["id"],
                remoteEval=True,
            )
            initial = mock_post.call_count
            # 'country' is not in cacheKeyAttributes — should be cache hit
            gb.set_attributes({"id": "u1", "country": "DE"})
            assert mock_post.call_count == initial


class TestRemoteEvalSyncRobustness:
    def setup_method(self):
        _reset_repo()

    def test_missing_saved_groups_doesnt_crash(self):
        body = {"features": {"flag1": {"defaultValue": True}}}  # no savedGroups key
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)):
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="k",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            assert gb.is_on("flag1") is True

    def test_empty_features_returns_off(self):
        body = {"features": {}, "savedGroups": {}}
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)):
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="k",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            # Unknown features return a falsy value -> is_on is False
            assert gb.is_on("nonexistent") is False

    def test_no_cross_pollution_between_instances(self):
        """Two GrowthBook instances with same client_key but different attributes
        must each see their own filtered features (the singleton repo's global
        callback path is skipped in remote-eval mode to prevent leakage)."""
        responses = {
            ("u1",): {"features": {"flag1": {"defaultValue": "for-u1"}}, "savedGroups": {}},
            ("u2",): {"features": {"flag1": {"defaultValue": "for-u2"}}, "savedGroups": {}},
        }

        def fake_post(self, url, payload, headers):
            return _make_post_response(responses[(payload["attributes"]["id"],)])

        with patch.object(FeatureRepository, "_post", fake_post):
            gb1 = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="shared",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            gb2 = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="shared",
                attributes={"id": "u2"},
                remoteEval=True,
            )
            assert gb1.get_feature_value("flag1", "fallback") == "for-u1"
            assert gb2.get_feature_value("flag1", "fallback") == "for-u2"


class TestRemoteEvalSyncSSE:
    """SSE wiring for the sync class. The proxy emits parameter-less
    `features-updated` events in remote-eval mode as a cache-invalidation
    signal — they must reach _dispatch_sse_event and trigger load_features()."""

    def setup_method(self):
        _reset_repo()

    def test_dispatch_handles_event_with_no_data_field(self):
        """_dispatch_sse_event must not raise KeyError when 'data' is absent."""
        body = DEFAULT_BODY
        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            initial = mock_post.call_count
            # Simulate the proxy's parameter-less SSE event arriving directly
            # at the dispatcher (no 'data' key in event_data).
            gb._dispatch_sse_event({"type": "features-updated"})
            # The dispatcher must (a) not crash and (b) trigger a refetch.
            # Same payload → cache hit, so no extra POST is required for (b).
            # What we're really asserting is that no exception escaped.
            gb.set_attributes({"id": "u2"})  # force a payload change → POST
            assert mock_post.call_count == initial + 1
            gb.destroy()

    def test_sse_parser_dispatches_parameter_less_events(self):
        """SSEClient's parser must dispatch type-only events. Previously it
        gated on both 'type' and 'data' being present, silently dropping the
        proxy's parameter-less features-updated event."""
        from growthbook.growthbook import SSEClient

        # A captured event stream the proxy might emit in remote-eval mode:
        # one event with data (initial features push), one without (the
        # subsequent cache-invalidation signal).
        events_seen = []

        class _FakeResponse:
            class _Content:
                def __init__(self, lines):
                    self._lines = lines

                def __aiter__(self):
                    self._iter = iter(self._lines)
                    return self

                async def __anext__(self):
                    try:
                        return next(self._iter)
                    except StopIteration:
                        raise StopAsyncIteration

            def __init__(self, lines):
                self.content = self._Content(lines)

        sse = SSEClient(api_host="http://h", client_key="k", on_event=events_seen.append)
        sse.is_running = True

        # Two events: one with data, one without. Each terminated by a blank line.
        raw = [
            b"event: features\n",
            b"data: {\"features\": {}}\n",
            b"\n",
            b"event: features-updated\n",
            b"\n",
        ]
        response = _FakeResponse(raw)

        asyncio.run(sse._process_response(response))

        types_dispatched = [e.get("type") for e in events_seen]
        assert "features" in types_dispatched, f"first event lost: {types_dispatched}"
        assert "features-updated" in types_dispatched, (
            f"parameter-less features-updated was silently dropped: {types_dispatched}"
        )
        # Single-line data event must NOT have a leading newline (W3C spec:
        # data lines are joined with newlines BETWEEN them, not prepended).
        first_data = next(e for e in events_seen if e.get("type") == "features")["data"]
        assert first_data == '{"features": {}}', (
            f"data line had leading newline: {first_data!r}"
        )

    def test_sse_parser_joins_multi_data_lines_per_spec(self):
        """Multi-line data: events must be joined with `\\n` BETWEEN lines,
        not prepended to each. Per W3C EventSource."""
        from growthbook.growthbook import SSEClient

        events_seen = []

        class _FakeResponse:
            class _Content:
                def __init__(self, lines):
                    self._lines = lines

                def __aiter__(self):
                    self._iter = iter(self._lines)
                    return self

                async def __anext__(self):
                    try:
                        return next(self._iter)
                    except StopIteration:
                        raise StopAsyncIteration

            def __init__(self, lines):
                self.content = self._Content(lines)

        sse = SSEClient(api_host="http://h", client_key="k", on_event=events_seen.append)
        sse.is_running = True
        raw = [
            b"event: chunked\n",
            b"data: line-1\n",
            b"data: line-2\n",
            b"data: line-3\n",
            b"\n",
        ]
        asyncio.run(sse._process_response(_FakeResponse(raw)))
        assert events_seen[0]["data"] == "line-1\nline-2\nline-3", (
            f"data joined incorrectly: {events_seen[0]['data']!r}"
        )


# ---------------------------------------------------------------------------
# Async GrowthBookClient tests
# ---------------------------------------------------------------------------


def _reset_singletons():
    """Drop the EnhancedFeatureRepository singleton so each test gets fresh state.

    test_growthbook_client.py uses `EnhancedFeatureRepository._instances = {}`
    which creates a class attribute on EnhancedFeatureRepository that SHADOWS
    the metaclass attribute `SingletonMeta._instances`. Once that has happened,
    `SingletonMeta._instances.clear()` clears the wrong dict. To survive that
    pattern, we ASSIGN a fresh empty dict on EnhancedFeatureRepository (so it
    becomes the shadow that the metaclass's `cls._instances` lookup will find),
    and ALSO clear the metaclass attribute in case nothing's shadowed it yet."""
    EnhancedFeatureRepository._instances = {}
    SingletonMeta._instances.clear()


@pytest.fixture(autouse=True)
def _reset_async_singletons():
    _reset_singletons()
    yield
    _reset_singletons()


class TestRemoteEvalAsyncValidation:
    def test_missing_client_key_raises(self):
        with pytest.raises(ValueError, match="client_key for remote eval"):
            GrowthBookClient(Options(api_host="https://proxy.example.com", remote_eval=True))

    def test_empty_api_host_raises(self):
        """Explicit api_host="" bypasses the cloud-host check (which sees
        an empty hostname) and would otherwise fall through to the Cloud
        CDN at runtime — must fail fast at construction."""
        with pytest.raises(ValueError, match="api_host .* for remote eval"):
            GrowthBookClient(Options(
                api_host="", client_key="k", remote_eval=True, refresh_strategy=None,
            ))

    def test_none_api_host_raises(self):
        """Explicit api_host=None has the same effect — fail fast."""
        with pytest.raises(ValueError, match="api_host .* for remote eval"):
            GrowthBookClient(Options(
                api_host=None, client_key="k", remote_eval=True, refresh_strategy=None,
            ))

    def test_decryption_key_raises(self):
        with pytest.raises(ValueError, match="Encryption is not available"):
            GrowthBookClient(Options(
                api_host="https://proxy.example.com",
                client_key="k",
                decryption_key="d",
                remote_eval=True,
            ))

    def test_sticky_bucket_service_raises(self):
        with pytest.raises(ValueError, match="sticky_bucket_service is not compatible"):
            GrowthBookClient(Options(
                api_host="https://proxy.example.com",
                client_key="k",
                sticky_bucket_service=InMemoryStickyBucketService(),
                remote_eval=True,
            ))

    def test_cloud_host_raises(self):
        with pytest.raises(ValueError, match="Cloud host does not support remote eval"):
            GrowthBookClient(Options(
                api_host="https://cdn.growthbook.io",
                client_key="k",
                remote_eval=True,
                refresh_strategy=None,  # bypass the STALE_WHILE_REVALIDATE guard
            ))

    def test_stale_while_revalidate_raises(self):
        """Async parity with sync: STALE_WHILE_REVALIDATE is incompatible
        with remote_eval (HTTP polling has no per-user payload to send)."""
        from growthbook.common_types import FeatureRefreshStrategy
        with pytest.raises(ValueError, match="STALE_WHILE_REVALIDATE is not compatible"):
            GrowthBookClient(Options(
                api_host="https://proxy.example.com",
                client_key="k",
                remote_eval=True,
                # default refresh_strategy IS STALE_WHILE_REVALIDATE, so the
                # plainest possible misconfig — no explicit refresh_strategy —
                # must be rejected.
            ))
        # SERVER_SENT_EVENTS is allowed.
        GrowthBookClient(Options(
            api_host="https://proxy.example.com",
            client_key="k",
            remote_eval=True,
            refresh_strategy=FeatureRefreshStrategy.SERVER_SENT_EVENTS,
        ))


async def _make_async_client(post_handler, **opts):
    """Build a GrowthBookClient with remote_eval on and the network method mocked.

    Defensively clears the SingletonMeta cache so each test gets a fresh
    EnhancedFeatureRepository — pytest-asyncio creates a new event loop per
    test, and the previous test's instance can still hold asyncio primitives
    bound to a now-closed loop. Without this clear the conftest's async
    teardown can hang on `await stop_refresh()` and the next test inherits
    the stale `_remote_eval_cache`."""
    EnhancedFeatureRepository._instances = {}
    SingletonMeta._instances.clear()
    client = GrowthBookClient(Options(
        api_host="https://proxy.example.com",
        client_key="sdk-async",
        remote_eval=True,
        refresh_strategy=None,
        **opts,
    ))
    client._features_repository._fetch_and_decode_post_async = post_handler
    await client.initialize()
    return client


@pytest.mark.asyncio
async def test_async_cache_hit_same_user_context():
    calls = []

    async def post_handler(api_host, client_key, payload):
        calls.append(payload)
        return DEFAULT_BODY

    client = await _make_async_client(post_handler)
    uc = UserContext(attributes={"id": "u1"})

    assert await client.is_on("flag1", uc) is True
    assert len(calls) == 1
    # Second eval with same context = cache hit
    assert await client.is_on("flag1", uc) is True
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_singleton_keyed_by_api_host_and_client_key():
    """Two GrowthBookClients with different (api_host, client_key) must get
    different EnhancedFeatureRepository instances — otherwise the second
    client silently inherits the first's `_remote_eval` flag, `_api_host`,
    `_client_key`, etc., which broke SSE routing and the cache-flush path."""
    EnhancedFeatureRepository._instances = {}
    SingletonMeta._instances.clear()

    cdn = GrowthBookClient(Options(
        api_host="https://cdn-A.example.com",
        client_key="key-A",
        refresh_strategy=None,
    ))
    remote = GrowthBookClient(Options(
        api_host="https://proxy-B.example.com",
        client_key="key-B",
        remote_eval=True,
        refresh_strategy=None,
    ))

    assert cdn._features_repository is not remote._features_repository
    assert cdn._features_repository._remote_eval is False
    assert remote._features_repository._remote_eval is True
    assert cdn._features_repository._api_host == "https://cdn-A.example.com"
    assert remote._features_repository._api_host == "https://proxy-B.example.com"


@pytest.mark.asyncio
async def test_post_and_cache_releases_inflight_on_cancellation():
    """`asyncio.CancelledError` derives from BaseException — `except Exception`
    used to miss it, leaving the inflight map populated and any concurrent
    waiters hung on the never-resolved Future. After the fix, cancelling a
    leader call cleanly releases the inflight slot."""
    release = asyncio.Event()

    async def slow_post(api_host, client_key, payload):
        await release.wait()  # never set
        return DEFAULT_BODY

    client = await _make_async_client(slow_post)
    repo = client._features_repository
    uc = UserContext(attributes={"id": "u-cancel"})

    task = asyncio.create_task(client.is_on("flag1", uc))
    # Give it time to enter the inflight map.
    await asyncio.sleep(0.02)
    assert len(repo._remote_eval_inflight) == 1, "inflight slot should be populated"

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # CRITICAL: inflight map must NOT leak the cache key after cancellation.
    assert len(repo._remote_eval_inflight) == 0, (
        "cancellation leaked the inflight slot — future callers would hang forever"
    )


@pytest.mark.asyncio
async def test_async_cache_miss_different_user_context():
    calls = []

    async def post_handler(api_host, client_key, payload):
        calls.append(payload)
        return DEFAULT_BODY

    client = await _make_async_client(post_handler)
    await client.is_on("flag1", UserContext(attributes={"id": "u1"}))
    await client.is_on("flag1", UserContext(attributes={"id": "u2"}))
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_async_forced_features_flow_through_user_context():
    """UserContext.forced_features serializes as [[k, v], ...] on the wire
    and feeds into the cache key the same way other payload fields do."""
    calls = []

    async def post_handler(api_host, client_key, payload):
        calls.append(payload)
        return DEFAULT_BODY

    client = await _make_async_client(post_handler)
    uc = UserContext(attributes={"id": "u1"}, forced_features={"banner": "v2"})
    await client.is_on("flag1", uc)
    assert calls[0]["forcedFeatures"] == [["banner", "v2"]]

    # NOTE: forcedFeatures is intentionally NOT part of the cache key (matches
    # JS — the proxy doesn't filter on it). Same attrs/forced_vars/url with
    # different forced_features → cache hit, no extra POST.
    uc2 = UserContext(attributes={"id": "u1"}, forced_features={"banner": "v9"})
    await client.is_on("flag1", uc2)
    assert len(calls) == 1, "forcedFeatures should not invalidate the cache"


@pytest.mark.asyncio
async def test_async_cache_ttl_hard_expiry():
    """Past cache_ttl, the cache entry is gone — next eval re-POSTs synchronously."""
    posts = 0

    async def post_handler(api_host, client_key, payload):
        nonlocal posts
        posts += 1
        return DEFAULT_BODY

    client = await _make_async_client(post_handler, cache_ttl=0)  # expire immediately
    uc = UserContext(attributes={"id": "u1"})
    await client.is_on("flag1", uc)
    assert posts == 1
    # cache_ttl=0 means every read is "past max_age" — every eval re-POSTs.
    await client.is_on("flag1", uc)
    assert posts == 2, "cache_ttl=0 should expire entries on every read"


@pytest.mark.asyncio
async def test_async_stale_ttl_swr_serves_cached_then_refreshes():
    """In the [stale_ttl, cache_ttl) window: serve the cached value
    immediately, fire a background refresh, and the cached entry's stale_at
    bumps forward when the refresh lands."""
    posts = 0
    posted = asyncio.Event()

    async def post_handler(api_host, client_key, payload):
        nonlocal posts
        posts += 1
        posted.set()
        return DEFAULT_BODY

    # stale_ttl=0 (always stale) + cache_ttl=60 (huge window): every eval
    # serves cached + fires a background refresh.
    client = await _make_async_client(post_handler, cache_ttl=60, stale_ttl=0)
    uc = UserContext(attributes={"id": "u1"})
    await client.is_on("flag1", uc)
    assert posts == 1, "first eval is a cache miss"

    # Second eval: cache hit (returns immediately) AND schedules a bg refresh.
    posted.clear()
    await client.is_on("flag1", uc)
    await posted.wait()  # deterministic — no fragile sleep
    assert posts == 2, "stale_ttl should trigger a background SWR refetch"


@pytest.mark.asyncio
async def test_async_swr_dedupes_concurrent_background_refreshes():
    """Inside the SWR window, multiple cache hits during the same inflight
    background-refresh POST must NOT spawn duplicate refreshes."""
    started = asyncio.Event()
    release = asyncio.Event()
    posts = 0

    async def gated_post(api_host, client_key, payload):
        nonlocal posts
        posts += 1
        started.set()
        await release.wait()
        return DEFAULT_BODY

    # Initial POST: let it through immediately so the cache populates.
    release.set()
    client = await _make_async_client(gated_post, cache_ttl=60, stale_ttl=0)
    uc = UserContext(attributes={"id": "u1"})
    await client.is_on("flag1", uc)
    assert posts == 1

    # Now block subsequent POSTs so we can race the SWR triggers.
    release.clear()
    started.clear()

    # First cache hit schedules a background refresh; subsequent cache hits
    # while it's still inflight must coalesce (no duplicate POSTs).
    await client.is_on("flag1", uc)
    await started.wait()  # bg refresh has reached gated_post
    await client.is_on("flag1", uc)
    await client.is_on("flag1", uc)
    assert posts == 2, "concurrent SWR triggers must coalesce"

    # Unblock and confirm clean shutdown.
    release.set()
    await asyncio.sleep(0.02)


@pytest.mark.asyncio
async def test_async_inflight_coalescing():
    """Three concurrent evals for the same UserContext = exactly 1 POST."""
    calls = []
    started = asyncio.Event()
    finish = asyncio.Event()

    async def slow_post(api_host, client_key, payload):
        calls.append(payload)
        started.set()
        await finish.wait()
        return DEFAULT_BODY

    client = await _make_async_client(slow_post)
    uc = UserContext(attributes={"id": "u1"})

    t1 = asyncio.create_task(client.is_on("flag1", uc))
    t2 = asyncio.create_task(client.is_on("flag1", uc))
    t3 = asyncio.create_task(client.is_on("flag1", uc))
    await started.wait()
    # Give the other coroutines time to reach the inflight check
    await asyncio.sleep(0.05)
    finish.set()
    r1, r2, r3 = await asyncio.gather(t1, t2, t3)
    assert (r1, r2, r3) == (True, True, True)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_async_lru_eviction():
    """Cache size 2: three distinct users in order A,B,C then A again -> A is a miss."""
    calls = []

    async def post_handler(api_host, client_key, payload):
        calls.append(payload["attributes"]["id"])
        return DEFAULT_BODY

    client = await _make_async_client(post_handler, remote_eval_cache_size=2)

    await client.is_on("flag1", UserContext(attributes={"id": "A"}))
    await client.is_on("flag1", UserContext(attributes={"id": "B"}))
    await client.is_on("flag1", UserContext(attributes={"id": "C"}))  # evicts A
    await client.is_on("flag1", UserContext(attributes={"id": "A"}))  # cache miss
    assert calls == ["A", "B", "C", "A"]


@pytest.mark.asyncio
async def test_async_preload_warms_cache():
    calls = []

    async def post_handler(api_host, client_key, payload):
        calls.append(payload)
        return DEFAULT_BODY

    client = await _make_async_client(post_handler)
    uc = UserContext(attributes={"id": "u1"})

    await client.preload_remote_eval(uc)
    assert len(calls) == 1
    # Subsequent eval is a cache hit
    await client.is_on("flag1", uc)
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_async_sse_features_updated_flushes_cache():
    calls = []

    async def post_handler(api_host, client_key, payload):
        calls.append(payload)
        return DEFAULT_BODY

    client = await _make_async_client(post_handler)
    uc = UserContext(attributes={"id": "u1"})

    await client.is_on("flag1", uc)
    assert len(calls) == 1

    # Simulate the proxy's features-updated SSE event
    await client._features_repository._handle_sse_event({"type": "features-updated"})

    await client.is_on("flag1", uc)
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_async_sse_event_before_any_eval_does_not_call_cdn_path():
    """Regression: an SSE features-updated event arriving BEFORE any user
    has been evaluated used to fall through to the CDN GET path (because
    the repo inferred remote-eval mode from cache-non-empty, and the cache
    is empty at that point). Must now route through the remote-eval flush
    regardless of cache contents."""
    posts = 0
    cdn_gets = 0

    async def post_handler(*args, **kwargs):
        nonlocal posts
        posts += 1
        return {"features": {}, "savedGroups": {}}

    async def cdn_fetch(*args, **kwargs):
        nonlocal cdn_gets
        cdn_gets += 1
        return {"features": {}, "savedGroups": {}}

    client = await _make_async_client(post_handler)
    # Patch the CDN GET path so we can detect erroneous calls into it.
    client._features_repository._fetch_features_async = cdn_fetch

    # Sanity: no evals yet, so the remote-eval cache is empty.
    assert not client._features_repository._remote_eval_cache

    # The SSE event must NOT take the CDN branch.
    await client._features_repository._handle_sse_event({"type": "features-updated"})

    assert cdn_gets == 0, "SSE handler took the CDN path on a remote-eval client"
    assert posts == 0, "SSE handler shouldn't POST either — flush only"


@pytest.mark.asyncio
async def test_async_preload_noop_in_cdn_mode():
    """preload_remote_eval is a safe no-op when remote_eval is False."""
    # Autouse _reset_async_singletons handles singleton cleanup.
    client = GrowthBookClient(Options(
        api_host="https://proxy.example.com",
        client_key="sdk-cdn",
        remote_eval=False,
        refresh_strategy=None,
    ))
    # Don't initialize (no features to load in a CDN-only smoke). Just make
    # sure preload doesn't raise or hit the network.
    await client.preload_remote_eval(UserContext(attributes={"id": "u1"}))


class TestRemoteEvalRuleTracks:
    """The proxy attaches a `tracks: [{experiment, result}]` array to rules so
    the SDK can fire trackingCallback for experiments that were evaluated
    server-side. Mirrors `if (rule.tracks)` in JS sdk-js/src/core.ts."""

    def setup_method(self):
        _reset_repo()

    def test_sync_fires_tracking_callback_for_each_track_entry(self):
        # Server-evaluated rule: force=True with two deferred tracking events.
        body = {
            "features": {
                "ab-feature": {
                    "defaultValue": False,
                    "rules": [{
                        "force": True,
                        "tracks": [
                            {
                                "experiment": {"key": "exp-A", "variations": [0, 1]},
                                "result": {
                                    "variationId": 1,
                                    "inExperiment": True,
                                    "value": 1,
                                    "hashUsed": True,
                                    "hashAttribute": "id",
                                    "hashValue": "u1",
                                    "featureId": "ab-feature",
                                    "key": "1",
                                },
                            },
                            {
                                "experiment": {"key": "exp-B", "variations": ["c", "t"]},
                                "result": {
                                    "variationId": 0,
                                    "inExperiment": True,
                                    "value": "c",
                                    "hashUsed": True,
                                    "hashAttribute": "id",
                                    "hashValue": "u1",
                                    "featureId": "ab-feature",
                                    "key": "0",
                                },
                            },
                        ],
                    }],
                },
            },
            "savedGroups": {},
        }

        tracked = []
        def cb(experiment, result, user_context):
            tracked.append((experiment.key, result.variationId, result.value))

        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)):
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                on_experiment_viewed=cb,
                remoteEval=True,
            )
            result = gb.eval_feature("ab-feature")
            assert result.value is True
            assert result.source == "force"
            assert tracked == [("exp-A", 1, 1), ("exp-B", 0, "c")]

    def test_sync_no_tracks_no_callback_fire(self):
        """Force rule with no tracks fires nothing."""
        body = {
            "features": {
                "f": {"defaultValue": False, "rules": [{"force": True}]},
            },
            "savedGroups": {},
        }
        tracked = []
        def cb(experiment, result, user_context):
            tracked.append((experiment.key,))

        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)):
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                on_experiment_viewed=cb,
                remoteEval=True,
            )
            gb.eval_feature("f")
            assert tracked == []

    def test_sync_malformed_track_entry_doesnt_crash(self):
        """A tracks entry missing required experiment fields is skipped, not raised."""
        body = {
            "features": {
                "f": {
                    "defaultValue": False,
                    "rules": [{
                        "force": True,
                        "tracks": [
                            {"experiment": {}, "result": {}},  # missing key/variations
                            {
                                "experiment": {"key": "exp", "variations": [0, 1]},
                                "result": {"variationId": 1, "inExperiment": True, "value": 1},
                            },
                        ],
                    }],
                },
            },
            "savedGroups": {},
        }
        tracked = []
        def cb(experiment, result, user_context):
            tracked.append(experiment.key)

        with patch.object(FeatureRepository, "_post", return_value=_make_post_response(body)):
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                on_experiment_viewed=cb,
                remoteEval=True,
            )
            gb.eval_feature("f")
            assert tracked == ["exp"]  # only the valid one fired


@pytest.mark.asyncio
async def test_async_rule_tracks_fires_tracking_callback():
    """Same as sync — async client also fires rule.tracks on force-rule path."""
    body = {
        "features": {
            "ab-feature": {
                "defaultValue": False,
                "rules": [{
                    "force": True,
                    "tracks": [{
                        "experiment": {"key": "exp-A", "variations": [0, 1]},
                        "result": {
                            "variationId": 1, "inExperiment": True, "value": 1,
                            "hashUsed": True, "hashAttribute": "id", "hashValue": "u1",
                            "featureId": "ab-feature", "key": "1",
                        },
                    }],
                }],
            },
        },
        "savedGroups": {},
    }
    tracked = []

    async def post_handler(api_host, client_key, payload):
        return body

    def cb(experiment, result, *_):
        tracked.append((experiment.key, result.variationId))

    client = await _make_async_client(post_handler, on_experiment_viewed=cb)

    result = await client.eval_feature("ab-feature", UserContext(attributes={"id": "u1"}))
    assert result.value is True
    assert tracked == [("exp-A", 1)]


@pytest.mark.asyncio
async def test_async_non_remote_eval_still_works():
    """Regression: when remote_eval is False, initialize() does the CDN load
    and create_evaluation_context uses _global_context as before."""
    cdn_response = {"features": {"flag1": {"defaultValue": "cdn-value"}}, "savedGroups": {}}

    async def cdn_fetch(api_host, client_key, decryption_key=""):
        return cdn_response

    # Autouse _reset_async_singletons handles singleton cleanup.
    client = GrowthBookClient(Options(
        api_host="https://cdn.growthbook.io",
        client_key="k",
        refresh_strategy=None,
    ))
    client._features_repository._fetch_features_async = cdn_fetch
    feature_repo.clear_cache()
    client._features_repository.cache.cache.clear() if hasattr(client._features_repository.cache, "cache") else None
    assert await client.initialize() is True
    assert await client.get_feature_value("flag1", "fallback", UserContext(attributes={"id": "u1"})) == "cdn-value"
