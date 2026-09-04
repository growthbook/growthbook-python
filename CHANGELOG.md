# Changelog

## Unreleased

### Features

* Contextual bandit support in both clients, at behavioral parity with the JavaScript SDK:
  * Consumes the `contextualBandits` payload section (and `encryptedContextualBandits`), evaluates `contextualBanditRef`/`contextualVariations` rules with per-leaf weight substitution, and reports `leafId`, `variationWeights`, and `banditVersion` on experiment results for exposure logging.
  * New `set_payload()` on `GrowthBook` and `GrowthBookClient` for seeding full SDK payloads; only the sections present are overwritten, and encrypted sections are decrypted with the configured `decryption_key` (JS `setPayload` semantics).
  * Payload refreshes missing a section (`savedGroups`, `contextualBandits`) preserve the previous value at every layer instead of wiping it; the synchronous client serializes payload writers and publishes one coherent evaluation snapshot per update (evaluations stay lock-free), matching the async client.
  * Malformed bandit definitions or leaves degrade to the rule's aggregate weights instead of raising during evaluation; reported `variationWeights` always match the weights bucketing actually used, and a single, total validity rule governs every weight vector: `getBucketRanges` normalizes vectors with negative, non-finite, boolean, non-numeric, or float-overflowing entries (not just wrong length/sum) to equal weights — never raising, even on arbitrary-precision integers, so bucket ranges can never be inverted, and bandit leaf/aggregate/override propensities always describe the vector actually used. Bandit identifiers get the same treatment: a leaf with a non-integer `leafId` is malformed (aggregate-weights fallback), and a non-integer `banditVersion` is omitted, so exposure metadata never carries invalid attribution ids into bandit training. A rule that pairs `contextualBanditRef` with explicit `ranges` buckets on the ranges (unchanged, matching the JS SDK) but drops the bandit metadata, since leaf propensities cannot describe a ranges-governed assignment.
  * Contextual bandit exposure metadata survives remote evaluation: `rule.tracks` results from the proxy keep `leafId`, `variationWeights`, and `banditVersion` when replayed through the tracking callback, held to the same validity rules as locally evaluated exposures (invalid identifiers or weight vectors are dropped, not forwarded).
  * Async evaluations that perform I/O (remote eval or a sticky bucket service) freeze every mutable evaluation input — attributes, groups, overrides, forced variations and features, nested containers included — before their first await, so mutating a `UserContext` mid-flight cannot change leaf routing, remote payloads, or forced assignments. Plain CDN evaluations never yield and skip the copy entirely; deferred callbacks get fire-time snapshots in both clients. `preload_remote_eval` takes the same call-time snapshot, so mutating the context after preloading can no longer cache one attribute state's response under another's key via the stale-while-revalidate background refresh.
* Deferred tracking: buffer experiment exposures for forwarding to a client SDK (which fires them via `setDeferredTrackingCalls` + `fireDeferredTrackingCalls`). Opt-in and independent of `on_experiment_viewed` — the buffer records first, the callback still fires. Entries use the JS SDK's `TrackingData` shape, deduped per unique assignment and JSON round-tripped at exposure time, so buffered payloads are always `json.dumps`-ready — an exposure carrying a non-JSON value (e.g. a `datetime` attribute) is dropped and logged, never the batch. Firing the forwarded `user` context on the receiving side requires JS SDK 1.7.0+.
  * Sync client: `GrowthBook(defer_tracking=True)` with `get_deferred_tracking_calls()` / `clear_deferred_tracking_calls()`.
  * Async client: pass a per-request `TrackingBuffer` to `eval_feature` / `run` / `is_on` / `is_off` / `get_feature_value` and read it with `buffer.get_calls()`; the caller owns the buffer, so concurrent requests never mix exposures.

### Bug Fixes

* Experiments that decide a prerequisite feature now report telemetry: previously `eval_prereqs` dropped every callback, so those exposures were silently lost. Fixed structurally by carrying `tracking_cb` / `feature_usage_cb` / `callback_subscription` on the `EvaluationContext` (like the JS SDK's global context) so nested evaluations inherit them. As a result:
  * `on_experiment_viewed` fires for prerequisite experiment assignments (including when a gate ultimately fails, matching the JS SDK).
  * `on_feature_usage` fires for every feature evaluated — prerequisites included — once per key per evaluation (previously it fired only for the top-level key, on every call, with no dedupe).
  * The sync client's `subscribe()` callbacks see prerequisite experiments, and its `get_all_results()` includes them. The async client's subscriptions still fire only from `run()` — like the JS multi-user client, it has no per-user assignment change-detection, so per-eval firing would repeat every subscriber callback on every request.
  * `core.eval_feature` / `core.run_experiment` now read callbacks from the `EvaluationContext`; the old `tracking_cb` / `callback_subscription` keyword arguments still work but are deprecated (copied onto the context with a `DeprecationWarning`).
* The tracking dedupe key is now a field tuple instead of a concatenated string, so distinct exposures cannot collide on field boundaries regardless of what the values contain.
* `Experiment.to_dict()` no longer coerces an explicit `coverage` of `0` to `1`.
* `savedGroups` from a feature refresh were applied to the evaluation context one refresh late in the synchronous client.
* The built-in tracking plugin now sends the exposure-time user context attributes with experiment events (previously async client events had none).

## [3.0.0](https://github.com/growthbook/growthbook-python/compare/v2.4.0...v3.0.0) (2026-08-26)


### Features

* Type Safety for GrowthBook Clients - Merge pull request [#126](https://github.com/growthbook/growthbook-python/issues/126) from growthbook/type-safety-hardening ([154e9c1](https://github.com/growthbook/growthbook-python/commit/154e9c19283dabccc3715e2f8597f4e250ac3d90))
* typed-client generator for strict feature-key checking ([7ffe31e](https://github.com/growthbook/growthbook-python/commit/7ffe31ebdeb5f64baa1340a2a9370633043ab2cd))

## [2.4.0](https://github.com/growthbook/growthbook-python/compare/v2.3.1...v2.4.0) (2026-08-18)


### Features

* Async sticky bucketing for `GrowthBookClient` ([#128](https://github.com/growthbook/growthbook-python/pull/128), [#129](https://github.com/growthbook/growthbook-python/pull/129)):
  * New `AbstractAsyncStickyBucketService` base class with `async` `get_assignments` / `save_assignments`; override `get_all_assignments` to batch all lookups for a user into one round trip (e.g. a single Redis `MGET`). Existing synchronous `AbstractStickyBucketService` implementations keep working with both clients — the async client offloads their blocking calls to a thread pool.
  * Sticky bucket reads no longer block the event loop. Assignments are fetched per evaluation for the supplied `UserContext`, matching the JavaScript SDK's multi-user client; concurrent evaluations for the same user share a single cancellation-safe in-flight lookup.
  * Sticky bucket writes are fire-and-forget: evaluation never waits on persistence. New assignments are immediately visible to later evaluations in the same process, and the client keeps an authoritative in-process copy of every document it has written, so a slow or stale store read can never roll back an assignment. Saves are serialized per document key so completion order cannot regress the stored document.
  * New `GrowthBookClient.flush_sticky_bucket_saves()` waits for all pending writes to persist (useful for serverless and short-lived processes); `close()` flushes automatically.
  * Optional bounded read cache for hot users via `Options(sticky_bucket_cache_ttl=..., sticky_bucket_cache_size=...)`; disabled by default.
* Async user callbacks in `GrowthBookClient` ([#128](https://github.com/growthbook/growthbook-python/pull/128), [#129](https://github.com/growthbook/growthbook-python/pull/129)): `on_experiment_viewed`, `on_feature_usage`, and `subscribe()` callbacks may now be coroutines. They are scheduled on the event loop without blocking evaluation, and a tracking callback that raises is retried on the next evaluation of the same experiment/user pair.
* Added `customFields` property to `Experiment` ([#125](https://github.com/growthbook/growthbook-python/pull/125))


### Performance Improvements

* Lock-free evaluation in `GrowthBookClient`: feature updates now swap an immutable snapshot instead of taking a per-evaluation lock ([#129](https://github.com/growthbook/growthbook-python/pull/129))
* `stop_refresh()` no longer blocks the event loop during shutdown ([#128](https://github.com/growthbook/growthbook-python/pull/128))
* In the included benchmark (`tests/scripts/benchmark_async_client.py`: 100 concurrent requests, 1 ms simulated service latency), distinct-user throughput with a network-backed sticky bucket service goes from ~350 evaluations/second with multi-second event-loop stalls on 2.3.x to ~20,000 evaluations/second with sub-2 ms loop lag.


### Bug Fixes

* Sticky bucket refresh no longer triggers a redundant feature reload per identifier attribute in the synchronous client ([#124](https://github.com/growthbook/growthbook-python/pull/124))
* The shared sticky bucket assignment-docs dict is now mutated in place instead of replaced when initially empty, preserving in-process read-your-writes ([#128](https://github.com/growthbook/growthbook-python/pull/128))
* The synchronous `GrowthBook` class now raises `ValueError` at construction if given an async sticky bucket service, instead of failing silently at runtime ([#128](https://github.com/growthbook/growthbook-python/pull/128))


### Tests and CI

* Synced the conformance corpus with the JavaScript SDK 0.8.0 cases, with documented skips for unsupported contextual-bandit cases ([#130](https://github.com/growthbook/growthbook-python/pull/130))
* Consumer-facing typing is now checked in CI via a mypy probe; public callback annotations accept both plain functions and coroutines ([#128](https://github.com/growthbook/growthbook-python/pull/128))
* Replaced sleep-based concurrency tests with deterministic event-gated tests and added a high-concurrency benchmark harness for the async client ([#128](https://github.com/growthbook/growthbook-python/pull/128))


### Compatibility notes

* Sticky bucket writes from `GrowthBookClient` are now eventual rather than synchronous with evaluation. Read-your-writes is preserved in-process; short-lived processes should `await client.flush_sticky_bucket_saves()` (or `close()`) before exit to guarantee persistence.
* `GrowthBookClient` now fetches sticky bucket assignments per evaluation instead of caching them for the lifetime of the process. This matches the JavaScript SDK and picks up cross-process assignment changes promptly, but increases service lookups; opt into bounded caching with `sticky_bucket_cache_ttl` / `sticky_bucket_cache_size` if needed.

## [2.3.1](https://github.com/growthbook/growthbook-python/compare/v2.3.0...v2.3.1) (2026-06-18)


### Fixes & Enhancements

* Align condition equality with JavaScript strict equality semantics:
  * `$eq` and direct equality no longer coerce across types, so values like `5` and `"5"` or `true` and `1` no longer match.
  * `$ne` now returns the inverse of strict equality for these cases.
  * Array and object operands follow JavaScript reference-identity semantics, so separately parsed but structurally equal arrays/objects do not match with `$eq`.
  ([7f9d2d](https://github.com/growthbook/growthbook-python/commit/7f9d2d24e6df1b863a1dc269933e4868105a1ac5), [a8ff302](https://github.com/growthbook/growthbook-python/commit/a8ff302a985f123ebea77a4d624814a376632979), [b011fee](https://github.com/growthbook/growthbook-python/commit/b011fee86fadad6f3db58bdf234c9dc6fe099285))
* Fix `NaN` comparison handling so `NaN` does not compare equal to itself and ordered comparisons involving `NaN` evaluate as false.
  ([672136a](https://github.com/growthbook/growthbook-python/commit/672136a629cb3dd191825ceb6250b890c3fab626))
* Preserve JavaScript-compatible Unicode lowercasing behavior for case-insensitive operators such as `$ini`, `$nini`, and `$alli`.
  ([5f45087](https://github.com/growthbook/growthbook-python/commit/5f45087a75a68896a0efdd43107a55da1600f85f))

### Tests and CI

* Expanded SDK conformance coverage for condition operators, prerequisite/parent-condition cases, force-rule `hashVersion: 2`, and sticky-bucket bucket-version boundaries.
  ([505c8e1](https://github.com/growthbook/growthbook-python/commit/505c8e1d44a0e9f51f9d4c8c98bc656ad0d965238), [a95ce02](https://github.com/growthbook/growthbook-python/commit/a95ce023a86d91c337f87aed415a57e05eb05ceb), [076eab8](https://github.com/growthbook/growthbook-python/commit/076eab860afc7016fa812f41f5d57db14382c7cd))
* Added a corpus freshness check against the JavaScript SDK cases corpus to catch missing or drifted cases.
  ([e428acb](https://github.com/growthbook/growthbook-python/commit/e428acba2559e8eddf3714703f03b0994b3b25220), [460f581](https://github.com/growthbook/growthbook-python/commit/460f5818c4771296d925a3a9af62d59e1435ad14), [a030b66](https://github.com/growthbook/growthbook-python/commit/a030b669a41a2cff2831f98195898dff4e00ebe3))

### Compatibility note

There are no API changes, but this release can change feature targeting results for customers whose conditions relied on Python’s previous coercive equality behavior.

## [2.3.0](https://github.com/growthbook/growthbook-python/compare/v2.2.2...v2.3.0) (2026-06-05)

### Features

* Added opt-in remote evaluation support for sync and async clients including cache invalidation, concurrent request coalescing, and cancellation handling. Merge pull request [#118](https://github.com/growthbook/growthbook-python/issues/118) from growthbook/feat/remote-eval ([f6b7c0a](https://github.com/growthbook/growthbook-python/commit/f6b7c0a224a021bd66eda9d6a939aad2064add35))
* Merge pull request [#119](https://github.com/growthbook/growthbook-python/issues/119) from vazarkevych/fix/sticky-bucket-hash-value-resolution ([2edd6a3](https://github.com/growthbook/growthbook-python/commit/2edd6a343dd15de9f96eb36c7c4407e213941336))


### Bug Fixes
* Fixed user context synchronization in evaluation/logging paths. ([75f1dcb](https://github.com/growthbook/growthbook-python/pull/118/changes/75f1dcb71e8cddbe0d88a51e2475a439e33318b5))
* Fixed `$ne`, `$notRegex`, and `$notRegexi` condition behavior for incompatible inputs. ([26d549b](https://github.com/growthbook/growthbook-python/pull/118/changes/26d549b8b0c4cee6171db955f3069a5d9a6b03d4))
* correct sticky bucket hash attribute resolution ([2edd6a3](https://github.com/growthbook/growthbook-python/commit/2edd6a343dd15de9f96eb36c7c4407e213941336))
* correct sticky bucket hash attribute resolution ([875316c](https://github.com/growthbook/growthbook-python/commit/875316c494e50d60a680767838c68b622f8fd402))


## [2.2.2](https://github.com/growthbook/growthbook-python/compare/v2.2.1...v2.2.2) (2026-05-11)


### Minor Enhancements - Performance

* optimize Feature(...) payload parsing
* `fnv1a32` processing fixes for ascii values ([4a0b5f6](https://github.com/growthbook/growthbook-python/commit/4a0b5f6d973a513452e61e165918627ffbd492ab)) 
* preserve version comparison normalization ([4d24d39](https://github.com/growthbook/growthbook-python/commit/4d24d39059b752ddeefb8fc70192605cb01d2413))

## [2.2.1](https://github.com/growthbook/growthbook-python/compare/v2.2.0...v2.2.1) (2026-04-23)


### Bug Fixes

* decrypt SSE encrypted features payload and fix cache key ([#110](https://github.com/growthbook/growthbook-python/issues/110)) ([a07004c](https://github.com/growthbook/growthbook-python/commit/a07004cc77d3a49916177f8be5ea6a230e016a78))

## [2.2.0](https://github.com/growthbook/growthbook-python/compare/v2.1.5...v2.2.0) (2026-03-23)


### Features

* add log_event and set_event_logger to GrowthBook and GrowthBookClient ([ae095b7](https://github.com/growthbook/growthbook-python/commit/ae095b79624cc375d1ed5f261fe5d8c48952936f))


### Bug Fixes

* sync _user_ctx before invoking event logger in log_event ([e2eaab6](https://github.com/growthbook/growthbook-python/commit/e2eaab634be9bf1baea5e0e028d8109d240d97e6))

## [2.1.5](https://github.com/growthbook/growthbook-python/compare/v2.1.4...v2.1.5) (2026-03-06)


### Bug Fixes

* Add optional timeout for PoolManager ([#91](https://github.com/growthbook/growthbook-python/issues/91)) ([2fe21f6](https://github.com/growthbook/growthbook-python/commit/2fe21f692189d7a37712d445b4545571ff2d3039))

## [2.1.4](https://github.com/growthbook/growthbook-python/compare/v2.1.3...v2.1.4) (2026-02-23)


### Bug Fixes

* Fixes for process hanging and shutdown errors - Merge pull request [#103](https://github.com/growthbook/growthbook-python/issues/103) from growthbook/pr102 ([c89a385](https://github.com/growthbook/growthbook-python/commit/c89a385b0cd0b0c4a776b7b81fc1ab3d27e40738))
* parsing data for SSE in GrowthbookClient ([d390223](https://github.com/growthbook/growthbook-python/commit/d390223c0035d65d91a930391d4731321f4c2f15))
* prevent SSE thread from blocking process exit and suppressing shutdown errors ([bddfb82](https://github.com/growthbook/growthbook-python/commit/bddfb82fce6284d4edf48b6135b51b362f82eab9))

## [2.1.3](https://github.com/growthbook/growthbook-python/compare/v2.1.2...v2.1.3) (2026-02-05)


### Features

* Supporting Dict Subclasses in Evaluation - Merge pull request [#99](https://github.com/growthbook/growthbook-python/issues/99) from growthbook/feat/isInstanceTypeCheck ([8ed4d4e](https://github.com/growthbook/growthbook-python/commit/8ed4d4e1aaf5b79408d60b16f856d66146600f91))
* Replaced all type(x) is T checks with isinstance(x, T).
* Updated getType, getPath, compare, and operator functions to use these new checks.

## [2.1.2](https://github.com/growthbook/growthbook-python/compare/v2.1.1...v2.1.2) (2026-01-29)


### Bug Fixes

* Disabled features not being removed from cache ([#93](https://github.com/growthbook/growthbook-python/issues/93)) ([eac9717](https://github.com/growthbook/growthbook-python/commit/eac971782f7776ff4261cc4ef9b7894b5735eb9d))

## [2.1.1](https://github.com/growthbook/growthbook-python/compare/v2.1.0...v2.1.1) (2026-01-27)


### Features

* Add support for case-insensitive membership operators: `$ini`, `$nini`, `$alli`
  - `$ini`: Case-insensitive version of `$in` operator
  - `$nini`: Case-insensitive version of `$nin` operator
  - `$alli`: Case-insensitive version of `$all` operator ([0e26f7d](https://github.com/growthbook/growthbook-python/commit/0e26f7d55e2b4b5908a9e3dd0921c1ea1fa49f97))

## [2.1.0](https://github.com/growthbook/growthbook-python/compare/v2.0.0...v2.1.0) (2026-01-22)


### Features

* Adds support for `regexi` and `$notRegexi` - Case insensitive regex ([b9fce8a](https://github.com/growthbook/growthbook-python/commit/b9fce8ab2e7c91e38a0f2cb7b1d2446d564e650b))
- Adds support for `$notRegex`

## [2.0.0](https://github.com/growthbook/growthbook-python/compare/v1.4.10...v2.0.0) (2026-01-14)


### ⚠ BREAKING CHANGES

* Fixes for Async wrapper execution and other enhancements

### Bug Fixes

* Fixes for Async wrapper execution and other enhancements ([e6a0eaf](https://github.com/growthbook/growthbook-python/commit/e6a0eaff7dcc391819ad92eeb94a4fd3aac7bdda))

## [1.4.10](https://github.com/growthbook/growthbook-python/compare/v1.4.9...v1.4.10) (2025-12-19)


### Minor Enhancements

* Add user agent suffix optional prop - Merge pull request [#87](https://github.com/growthbook/growthbook-python/issues/87) from growthbook/fix/fetch-metadata ([e786dd8](https://github.com/growthbook/growthbook-python/commit/e786dd815f9a5608cbfb0681eba55b4cf0e94298))

## [1.4.9](https://github.com/growthbook/growthbook-python/compare/v1.4.8...v1.4.9) (2025-12-06)


### Enhancements

* Add gzip encoding header to features call - Merge pull request [#83](https://github.com/growthbook/growthbook-python/issues/83) from growthbook/feat/etag-cache ([82bdee2](https://github.com/growthbook/growthbook-python/commit/82bdee29663b07e2733a841536634ca680e9b276))

## [1.4.8](https://github.com/growthbook/growthbook-python/compare/v1.4.7...v1.4.8) (2025-12-03)


### Features

* Handle ETags natively for both sync & async clients - Merge pull request [#81](https://github.com/growthbook/growthbook-python/issues/81) from growthbook/feat/etag-cache ([95b06c3](https://github.com/growthbook/growthbook-python/commit/95b06c312428d7af1b81b2cbb29ddf6421f05ef7))

## [1.4.7](https://github.com/growthbook/growthbook-python/compare/v1.4.6...v1.4.7) (2025-11-12)


### Bug Fixes

* Type checks & Other enhancements - Merge pull request [#77](https://github.com/growthbook/growthbook-python/issues/77) from growthbook/feat/enhancements ([ea1567a](https://github.com/growthbook/growthbook-python/commit/ea1567a97284d61587493dd1a15f3105d06a67bb))

## [1.4.6](https://github.com/growthbook/growthbook-python/compare/v1.4.5...v1.4.6) (2025-10-27)


### Bug Fixes

* bug fixes and tracking enhancements ([6d3638b](https://github.com/growthbook/growthbook-python/commit/6d3638bf5e6c57c8882e4de35bc5e7ab5eafa5a5))
* Bug fixes and tracking enhancements ([a9b34a6](https://github.com/growthbook/growthbook-python/commit/a9b34a655d818758d6d622fea83945abd4e079f7))

## [1.4.5](https://github.com/growthbook/growthbook-python/compare/v1.4.4...v1.4.5) (2025-10-08)


### Bug Fixes

* Add FeatureUsageCallback ([83359a0](https://github.com/growthbook/growthbook-python/commit/83359a02cd00d645ba7c12d8ffea2a3ce6077411))

## [1.4.4](https://github.com/growthbook/growthbook-python/compare/v1.4.3...v1.4.4) (2025-09-30)


### Bug Fixes

* Background Refresh task for Features ([bda8050](https://github.com/growthbook/growthbook-python/commit/bda8050c7c1b72cf4589fd64e9bca884dcbb629c))

## [1.4.3](https://github.com/growthbook/growthbook-python/compare/v1.4.2...v1.4.3) (2025-09-19)


### Bug Fixes

* Fixes for graceful shutdown ([ab158ad](https://github.com/growthbook/growthbook-python/commit/ab158ad7a748bd7380c9ad0fda46cc91acc3b473))

## [1.4.2](https://github.com/growthbook/growthbook-python/compare/v1.4.1...v1.4.2) (2025-09-12)


### Bug Fixes

* Keep the Socket open with a configurable connection timeout ([f4783fc](https://github.com/growthbook/growthbook-python/commit/f4783fc451fdf7544b764239e71d89895ba8096c))

## [1.4.1](https://github.com/growthbook/growthbook-python/compare/v1.4.0...v1.4.1) (2025-09-12)


### Bug Fixes

* add timeout to SSE client ([ff6c2a7](https://github.com/growthbook/growthbook-python/commit/ff6c2a77269d691da984ef2e6b88405cf465caec))

## [1.4.0](https://github.com/growthbook/growthbook-python/compare/v1.3.1...v1.4.0) (2025-09-09)


### Features

* Tracking Plugins Compatibility with Async & Sync Clients ([a54c06d](https://github.com/growthbook/growthbook-python/commit/a54c06d22726a3702bacbf895165ef5bff02061b))

## [1.3.1](https://github.com/growthbook/growthbook-python/compare/v1.3.0...v1.3.1) (2025-06-13)


### Bug Fixes

* Tracking linked experiments ([becee2c](https://github.com/growthbook/growthbook-python/commit/becee2c7b306fd0e0f450c3a5676de77e39c9410))

## [1.3.0](https://github.com/growthbook/growthbook-python/compare/v1.2.0...v1.3.0) (2025-06-11)


### Features

* tracking plugins ([fde4d42](https://github.com/growthbook/growthbook-python/commit/fde4d4283343758ca1ec034052b8bdb2c0639b22))


### Bug Fixes

* tracking plugins and Caching ([ef6529a](https://github.com/growthbook/growthbook-python/commit/ef6529a113f5c1b074a9b700232d2e3343a6b152))

## [1.2.1](https://github.com/growthbook/growthbook-python/compare/v1.1.0...v) (2024-XX-XX)

### Bug Fixes

* PEP-561 compatibility
* Fix zero value evaluation for _getOrigHashValue
* Saved groups, $inGroup $notInGroup operators, versionCompare
* Added SSE client
* Support for multi-context and Enhanced GrowthBookClient with Async processing
* Update to test spec 0.7.1

### Features

* Enhanced async client with better error handling and retry logic
* Added comprehensive test coverage for async functionality

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

## **1.1.0** - Apr 11, 2024

- Support for prerequisite feature flags
- Optional Sticky Bucketing for experiment variation assignments
- SemVer targeting support
- Fixed multiple bugs and edge cases when comparing different data types
- Fixed bugs with the $in and $nin operators
- Now, we ignore unknown fields in feature definitions instead of throwing Exceptions
- Support for feature rule ids (for easier debugging)

## **1.0.0** - Apr 23, 2023

- Update to the official 0.4.1 GrowthBook SDK spec version
- Built-in fetching and caching of feature flags from the GrowthBook API
- Added detailed logging for easier debugging
- Support for new feature/experiment properties that enable holdout groups, meta info, better hashing algorithms, and more

## **0.3.1** - Aug 1, 2022

- Bug fix - skip experiment when the hashAttribute's value is `None`

## **0.3.0** - May 24, 2022

- Bug fix - don't skip feature rules when experiment variation is forced

## **0.2.0** - Feb 13, 2022

- Support for Feature Flags

## **0.1.1** - Jun 15, 2021

- Initial release (inline experiments only)
