# Remote-eval verification scripts

End-to-end checks for the `remote_eval` / `remoteEval` SDK option, covering
both the sync `GrowthBook` class and the async `GrowthBookClient`.

## TL;DR

```bash
# Fake mode — no setup, ~0.1s, runs every scenario against an in-process proxy.
python3 tests/scripts/verify_remote_eval.py

# Real mode — runs against a live growthbook-proxy. Useful pre-merge.
GB_API_KEY=secret_... GB_CLIENT_KEY=sdk-... \
    ./tests/scripts/bootstrap_proxy.sh
eval "$(GB_API_KEY=secret_... GB_CLIENT_KEY=sdk-... ./tests/scripts/bootstrap_proxy.sh --export)"
python3 tests/scripts/verify_remote_eval.py --real

# Tear down
./tests/scripts/bootstrap_proxy.sh --stop
```

## What gets covered

| # | Sync | Async | Scenario |
|---|------|-------|----------|
| 1  | ✓ | ✓ | Initial POST + body shape (`{attributes, forcedFeatures, forcedVariations, url}` + `Content-Type: application/json`) |
| 2  | ✓ |   | Response parsed, eval returns rule-forced value *(fixture-dependent)* |
| 3  | ✓ | ✓ | Same payload → cache hit, no extra POST |
| 4  | ✓ | ✓ | Different attributes → cache miss + new POST |
| 5  | ✓ |   | `set_url` triggers blocking refetch |
| 6  | ✓ |   | `set_forced_variations` triggers blocking refetch |
| 7  | ✓ | ✓ | `cache_key_attributes` narrows the cache key |
| 8  | ✓ | ✓ | All 5 validation guards raise at construction (missing client_key, decryption_key, sticky_bucket_service, stale_while_revalidate, cloud host) |
| 9  | ✓ | ✓ | `rule.tracks` fires `trackingCallback` *(fixture-dependent)* |
| 10 | ✓ |   | Two instances same client_key → no cross-pollution *(fixture-dependent)* |
| 11 |   | ✓ | `initialize()` does NOT fetch; first eval does |
| 12 |   | ✓ | `preload_remote_eval` warms cache; subsequent eval is pure-local |
| 13 |   | ✓ | 20 concurrent evals → 1 POST (inflight coalescing) *(fixture-dependent)* |
| 14 |   | ✓ | LRU eviction at `remote_eval_cache_size` |

Scenarios marked *fixture-dependent* skip in `--real` mode — they depend on
specific feature shapes (`verify-string`, `verify-tracks`) that only the fake
proxy serves.

## Modes

**Fake (default):** spawns an `aiohttp` server in a background thread on a
random localhost port. It implements the exact wire contract the real
growthbook-proxy implements: `POST /api/eval/{client_key}` returns
`{"features": {<key>: {"defaultValue": ..., "rules": [{"force": ..., "tracks": ...}]}}, "savedGroups": {}}`.
Rules with `tracks` carry deferred experiment-tracking entries, exactly as
the real proxy emits them.

**Real (`--real`):** runs against a real growthbook-proxy reachable at
`GB_PROXY_URL` with API key `GB_CLIENT_KEY`. Use `bootstrap_proxy.sh` to
spin one up in Docker.

POST counting works in both modes — we wrap `FeatureRepository._post`
(sync) and `EnhancedFeatureRepository._fetch_and_decode_post_async` (async)
at the class level so every SDK instance is observed regardless of how it
talks to the proxy.

## Useful flags

```bash
python3 tests/scripts/verify_remote_eval.py --only cache    # run scenarios with "cache" in the name
VERBOSE=1 python3 tests/scripts/verify_remote_eval.py       # tracebacks on failures
NO_COLOR=1 python3 tests/scripts/verify_remote_eval.py      # strip ANSI for CI
```

---

## Running friction log

Things a new developer must figure out to get remote eval working. These
are the moments where docs (or the SDK itself) could remove a step.

1. **Cloud doesn't expose `/api/eval`.** First-time setup error is opaque
   (404 from `cdn.growthbook.io/api/eval/<key>`). Our SDK now raises at
   construction with a clear message, but the proxy/edge requirement still
   needs to be visible *before* the user picks a client key.

2. **Two different keys.** The proxy needs a **server-side secret** API key
   (`SECRET_API_KEY=secret_...`) to fetch features from the GrowthBook API,
   while the SDK uses a separate **SDK Connection client key** (`sdk-...`).
   It's easy to swap them. The `bootstrap_proxy.sh` probe surfaces this
   immediately with a real POST.

3. **Wire format is undocumented outside SDK source.** Until you read either
   the JS SDK or this script, you don't know the POST body must be
   `{"attributes": ..., "forcedFeatures": [[k,v], ...], "forcedVariations": ..., "url": ...}`
   and that `forcedFeatures` is a list of tuples, not a dict. This script's
   fake proxy doubles as living documentation of the wire format.

4. **`forcedFeatures` semantics in remote-eval.** Both clients now wire
   `forcedFeatures` (sync via `GrowthBook(forced_features={...})` +
   `set_forced_features()`, async via `UserContext.forced_features`). On
   the wire it ships as `[[k, v], ...]` (JS-SDK-shaped). One quirk worth
   docs: `forcedFeatures` is **deliberately excluded from the cache key**
   (matches JS — the proxy doesn't filter on it). So calling
   `set_forced_features()` alone is a cache hit, not a network round-trip;
   the new value ships on the next refetch triggered by another setter.

5. **Refresh strategy interaction.** `stale_while_revalidate=True` is
   rejected with `remote_eval=True` (raises). SSE refresh is allowed (cache
   gets flushed on `features-updated`). HTTP polling is silently a no-op in
   remote-eval mode because there's no global features payload to refresh.
   Documenting this matrix would prevent confusion.

5b. **`cache_ttl` / `stale_ttl` asymmetry between sync and async.** Both
   options live on `Options`, but in remote-eval mode they're honored
   differently:
   - **Async `GrowthBookClient`** — full JS-style cache lifecycle on the
     per-user cache. `cache_ttl` is hard expiry (max_age). `stale_ttl` (when
     set < `cache_ttl`) enables SWR: serve cached + fire-and-forget
     background refresh in `[stale_ttl, cache_ttl)`. Stale_at resets on
     every successful write, even unchanged payloads (matches JS).
   - **Sync `GrowthBook`** — `cache_ttl` only (hard expiry via the existing
     `InMemoryFeatureCache`). `stale_ttl` and `stale_while_revalidate` are
     rejected in remote-eval mode. Sync uses a threading-based background
     refresh worker that's harder to fit per-user payloads into safely;
     we deliberately don't open that door. **If you need SWR semantics
     with remote-eval, use the async client.**

6. **Tracking on the force path.** Experiments evaluated server-side return
   to the SDK as force-rules with a `tracks` array. Without firing
   `trackingCallback` for each entry (which the SDK now does), analytics
   silently break in remote-eval mode. Worth surfacing in the upgrade notes
   so customers know they get tracking parity automatically.

7. **Singleton cross-pollination in the async client.** `EnhancedFeatureRepository`
   uses `SingletonMeta`, which means two `GrowthBookClient` instances in the
   same process share one repo. We skip the global `_on_feature_update`
   callback in remote-eval mode so per-user responses don't bleed across
   instances. Worth keeping in mind for anyone debugging a "wrong features
   for this user" issue.

8. **Cache size defaults to 1000.** A high-throughput async service with many
   unique users will hit LRU eviction quickly; the next eval per evicted
   user pays the network round-trip. Tunable via
   `Options.remote_eval_cache_size`, but the default is invisible — should
   be in the docs.
