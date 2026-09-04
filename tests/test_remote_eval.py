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

    def test_features_updated_invalidates_cache_and_fetches_fresh_payload(self):
        """Regression: SSE `features-updated` used to call load_features() which
        hit the cache and returned the stale payload — the proxy's invalidation
        signal was silently dropped. After fix, the dispatcher passes
        force_refresh=True so the cached entry is bypassed."""
        responses = iter([
            _make_post_response({"features": {"flag": {"defaultValue": False}}, "savedGroups": {}}),
            _make_post_response({"features": {"flag": {"defaultValue": True}},  "savedGroups": {}}),
        ])
        with patch.object(FeatureRepository, "_post", side_effect=lambda *a, **kw: next(responses)) as mock_post:
            gb = GrowthBook(
                api_host="https://proxy.example.com",
                client_key="sdk-test",
                attributes={"id": "u1"},
                remoteEval=True,
            )
            assert gb.is_on("flag") is False
            assert mock_post.call_count == 1

            # Simulate proxy publishing — sends a parameter-less features-updated.
            gb._dispatch_sse_event({"type": "features-updated"})

            assert mock_post.call_count == 2, (
                "features-updated didn't bypass the cache — POST count should have grown"
            )
            assert gb.is_on("flag") is True, (
                "still serving stale cached payload after SSE invalidation"
            )
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

    def test_stale_while_revalidate_explicit_also_raises(self):
        """Sanity: the guard fires when STALE_WHILE_REVALIDATE is set
        explicitly too, not just on the default."""
        from growthbook.common_types import FeatureRefreshStrategy
        with pytest.raises(ValueError, match="STALE_WHILE_REVALIDATE is not compatible"):
            GrowthBookClient(Options(
                api_host="https://proxy.example.com",
                client_key="k",
                remote_eval=True,
                refresh_strategy=FeatureRefreshStrategy.STALE_WHILE_REVALIDATE,
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
    entered = asyncio.Event()  # signals "POST has been entered"
    release = asyncio.Event()  # never set — we cancel before this

    async def slow_post(api_host, client_key, payload):
        entered.set()
        await release.wait()
        return DEFAULT_BODY

    client = await _make_async_client(slow_post)
    repo = client._features_repository
    uc = UserContext(attributes={"id": "u-cancel"})

    task = asyncio.create_task(client.is_on("flag1", uc))
    # Deterministic: wait until we KNOW the inflight POST has started, not a
    # speculative sleep. The inflight map entry is added BEFORE the await on
    # _fetch_and_decode_post_async, so `entered.wait()` returning guarantees
    # the slot is populated.
    await entered.wait()
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

    # Stronger assertion: a fresh foreground call for the same UserContext
    # after cancellation must not hang. If the future was orphaned in the
    # inflight map, this would await indefinitely.
    release.set()  # so future POSTs can complete
    # Have to swap in a different handler now since `entered` is already set.
    async def fast_post(*a, **kw):
        return DEFAULT_BODY
    repo._fetch_and_decode_post_async = fast_post
    result = await asyncio.wait_for(client.is_on("flag1", uc), timeout=1.0)
    assert result is True, "post-cancellation foreground call should complete"


@pytest.mark.asyncio
async def test_cancellation_does_not_log_future_exception_warning():
    """Cancelling a leader POST used to call `inflight.set_exception(
    CancelledError)`, which triggers asyncio's "Future exception was never
    retrieved" warning when the Future is garbage-collected with no observer.
    The fix is `inflight.cancel()` — cancelled Futures don't trigger the warning
    while still propagating CancelledError to any waiters.

    We intercept the loop's exception handler to detect the warning rather
    than capture stderr, because asyncio routes it through
    `loop.call_exception_handler(...)`."""
    import gc

    captured: list = []
    loop = asyncio.get_running_loop()
    prev_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda lp, ctx: captured.append(ctx.get("message", "")))
    try:
        release = asyncio.Event()  # never set

        async def slow_post(*a, **kw):
            await release.wait()
            return DEFAULT_BODY

        client = await _make_async_client(slow_post)
        uc = UserContext(attributes={"id": "u-cancel-quiet"})

        task = asyncio.create_task(client.is_on("flag1", uc))
        # Wait until the inflight slot is populated.
        while not client._features_repository._remote_eval_inflight:
            await asyncio.sleep(0)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # Force GC of the inflight Future and let __del__ fire its handler.
        del task
        gc.collect()
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        gc.collect()

        unretrieved = [m for m in captured if "exception was never retrieved" in m]
        assert not unretrieved, (
            f"asyncio logged unretrieved future exception(s): {unretrieved}"
        )
    finally:
        loop.set_exception_handler(prev_handler)


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

    # Unblock and confirm clean shutdown — wait for the bg task to finish
    # instead of a speculative sleep.
    release.set()
    if client._features_repository._swr_tasks:
        await asyncio.gather(
            *list(client._features_repository._swr_tasks),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_async_inflight_coalescing():
    """Three concurrent evals for the same UserContext = exactly 1 POST."""
    calls = []
    waiting = asyncio.Event()  # POST has entered the handler
    finish = asyncio.Event()   # gate to release the POST

    async def slow_post(api_host, client_key, payload):
        calls.append(payload)
        waiting.set()
        await finish.wait()
        return DEFAULT_BODY

    client = await _make_async_client(slow_post)
    uc = UserContext(attributes={"id": "u1"})

    t1 = asyncio.create_task(client.is_on("flag1", uc))
    t2 = asyncio.create_task(client.is_on("flag1", uc))
    t3 = asyncio.create_task(client.is_on("flag1", uc))
    await waiting.wait()  # leader entered POST handler

    # Yield until t2 and t3 have parked on the inflight Future. We can't
    # observe the future-awaiters directly, but each `await` suspends so
    # asyncio.sleep(0) ticks the scheduler past their await points.
    # Three yields = all three coroutines have reached their respective
    # `await existing` (or in t1's case, the long POST).
    for _ in range(3):
        await asyncio.sleep(0)
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
async def test_async_preload_snapshots_context_against_swr_cache_poisoning():
    """preload_remote_eval computes the cache key immediately, but an SWR
    background refresh serializes the POST body later. Without the same
    call-time snapshot create_evaluation_context takes, mutating the
    UserContext after preload returns would POST the NEW attributes and cache
    that response under the OLD attributes' key. Deterministic probe: the
    mocked proxy derives the flag value from the attributes it receives, so a
    poisoned cache is directly observable."""
    bodies = []
    posted = asyncio.Event()

    async def post_handler(api_host, client_key, payload):
        # Serialize NOW — this is what the wire would carry at POST time.
        attrs = json.loads(json.dumps(payload))["attributes"]
        bodies.append(attrs)
        posted.set()
        return {
            "features": {"flag1": {"defaultValue": attrs["tier"] == "pro"}},
            "savedGroups": {},
        }

    # stale_ttl=0: every cache hit is in the SWR window and schedules a
    # background refetch.
    client = await _make_async_client(post_handler, cache_ttl=60, stale_ttl=0)
    uc = UserContext(attributes={"id": "u1", "tier": "pro"})

    await client.preload_remote_eval(uc)  # miss -> foreground POST ("pro")
    posted.clear()
    await client.preload_remote_eval(uc)  # SWR hit -> schedules background POST
    uc.attributes["tier"] = "free"        # caller mutates AFTER preload returned
    await posted.wait()

    assert len(bodies) == 2
    # The background POST must describe the attributes its cache key was
    # computed from, not the mutated ones.
    assert bodies[1]["tier"] == "pro"
    # End-to-end: the cache entry for the "pro" context still answers as "pro".
    assert await client.is_on("flag1", UserContext(attributes={"id": "u1", "tier": "pro"})) is True


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
async def test_async_flush_during_inflight_post_doesnt_repopulate_stale():
    """Race: SSE flush_remote_eval_cache() fires while a foreground POST is
    in flight. After the POST lands, the stale (pre-flush) response must NOT
    be written into the cache — otherwise the proxy's invalidation signal is
    effectively reverted by the late-landing POST.

    Verified by reproducing the bug pre-fix: the cache would re-contain
    'stale-value' after the flush + late POST. After fix, the cache is empty
    and the next eval triggers a fresh POST."""
    entered = asyncio.Event()
    release = asyncio.Event()
    posts = 0

    async def slow_post(*a, **kw):
        nonlocal posts
        posts += 1
        entered.set()
        await release.wait()
        # First POST returns "stale-value"; subsequent POSTs return "fresh-value".
        return {
            "features": {"f": {"defaultValue": "stale-value" if posts == 1 else "fresh-value"}},
            "savedGroups": {},
        }

    client = await _make_async_client(slow_post)
    uc = UserContext(attributes={"id": "u1"})

    task = asyncio.create_task(client.get_feature_value("f", "fb", uc))
    await entered.wait()

    # Proxy publishes → SSE event flushes the cache.
    client._features_repository.flush_remote_eval_cache()
    # Now let the (now-stale) inflight POST complete.
    release.set()
    await task

    # The stale response must NOT have been written back into the cache.
    assert len(client._features_repository._remote_eval_cache) == 0, (
        "flush race: stale-response repopulated the cache after the proxy "
        "told us to invalidate"
    )
    # Next eval forces a fresh POST that returns the new value.
    fresh = await client.get_feature_value("f", "fb", uc)
    assert fresh == "fresh-value"
    assert posts == 2


@pytest.mark.asyncio
async def test_async_negative_cache_size_doesnt_crash():
    """Regression: `remote_eval_cache_size < 0` used to raise KeyError when
    the LRU eviction loop tried to popitem from an already-empty dict. The
    guard `while self._remote_eval_cache and ...` prevents the crash; the
    cache effectively holds nothing."""
    posts = 0
    async def post_handler(*a, **kw):
        nonlocal posts; posts += 1
        return DEFAULT_BODY

    client = await _make_async_client(post_handler, remote_eval_cache_size=-1)
    uc = UserContext(attributes={"id": "u1"})
    # Two evals — both should succeed and both POST (cache holds nothing).
    assert await client.is_on("flag1", uc) is True
    assert await client.is_on("flag1", uc) is True
    assert posts == 2
    assert len(client._features_repository._remote_eval_cache) == 0


@pytest.mark.asyncio
async def test_async_swr_tasks_cancelled_on_close():
    """Regression: SWR background tasks were created via asyncio.create_task
    but never tracked. On close()/stop_refresh() they kept running against a
    potentially-closed aiohttp session/loop, emitting "Task was destroyed"
    warnings. After fix, stop_refresh() cancels them and awaits drainage."""
    started = asyncio.Event()
    release = asyncio.Event()

    async def gated_post(*a, **kw):
        started.set()
        await release.wait()
        return DEFAULT_BODY

    client = await _make_async_client(gated_post, cache_ttl=60, stale_ttl=0)
    uc = UserContext(attributes={"id": "u1"})

    # Prime cache (let initial POST through).
    release.set()
    await client.is_on("flag1", uc)
    release.clear()
    started.clear()

    # Trigger a background SWR refresh that will be left pending.
    await client.is_on("flag1", uc)  # cache hit + schedules bg refresh
    await started.wait()
    repo = client._features_repository
    assert len(repo._swr_tasks) == 1, "SWR task should be tracked"
    swr_task = next(iter(repo._swr_tasks))

    # Close without releasing the gate. The background task must be cancelled.
    release.set()  # allow the cancelled task to unblock its await and exit
    await client.close()
    assert swr_task.cancelled() or swr_task.done(), (
        "SWR task should have been cancelled/awaited by close()"
    )
    assert len(repo._swr_tasks) == 0


@pytest.mark.asyncio
async def test_async_post_returning_none_doesnt_poison_cache():
    """Network failure / 5xx surfaces as `_fetch_and_decode_post_async`
    returning None. That None must NOT be written to the cache (otherwise
    subsequent evals would hit a cached None and silently return falsy
    fallbacks forever)."""
    responses = iter([None, DEFAULT_BODY])  # first call fails, second succeeds

    async def post_handler(*a, **kw):
        return next(responses)

    client = await _make_async_client(post_handler)
    uc = UserContext(attributes={"id": "u1"})

    # First call: POST returns None — cache must NOT be populated.
    result = await client.get_feature_value("flag1", "fallback", uc)
    assert result == "fallback", "failed POST should not produce a result"
    assert len(client._features_repository._remote_eval_cache) == 0

    # Second call: POST succeeds — cache populates normally.
    result = await client.is_on("flag1", uc)
    assert result is True


@pytest.mark.asyncio
async def test_async_preload_noop_in_cdn_mode():
    """preload_remote_eval is a safe no-op when remote_eval is False — must
    not POST and must not touch the remote-eval cache."""
    client = GrowthBookClient(Options(
        api_host="https://proxy.example.com",
        client_key="sdk-cdn",
        remote_eval=False,
        refresh_strategy=None,
    ))
    posts = 0
    async def post_handler(*args, **kwargs):
        nonlocal posts
        posts += 1
        return {"features": {}, "savedGroups": {}}
    client._features_repository._fetch_and_decode_post_async = post_handler

    await client.preload_remote_eval(UserContext(attributes={"id": "u1"}))
    assert posts == 0, "preload_remote_eval should never POST in CDN mode"
    assert len(client._features_repository._remote_eval_cache) == 0


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

    def cb(experiment, result, user_context):
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
