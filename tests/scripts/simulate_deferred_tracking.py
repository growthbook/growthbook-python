"""End-to-end simulation of deferred tracking and prerequisite telemetry.

Two realistic server scenarios, verified against the REAL JavaScript SDK as
the receiving side (tests/scripts/js_receiver.js runs it in Node):

1. Sync SSR: one GrowthBook instance per request (the documented pattern)
   renders a page for each user, buffers exposures, embeds them as JSON, and
   a simulated browser fires them through the JS SDK's
   setDeferredTrackingCalls + fireDeferredTrackingCalls.

2. Async API server: one long-lived GrowthBookClient serves concurrent
   requests; each handler owns a per-request TrackingBuffer whose contents go
   back in the API response. A server-side on_experiment_viewed callback runs
   at the same time (buffering is independent of callbacks).

Nothing is hard-coded to pass: 50/50 experiments are decided by real hashing,
expected exposures are derived from the SDK's own evaluation results, and the
fired events must match them user-by-user. Attributes are mutated after each
request to prove buffered snapshots are immune to caller aliasing.

Run:  python tests/scripts/simulate_deferred_tracking.py
Requires node + a built JS SDK (env GB_JS_SDK overrides the default path).
"""
import asyncio
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from growthbook import GrowthBook, TrackingBuffer  # noqa: E402
from growthbook.common_types import Options, UserContext  # noqa: E402
from growthbook.growthbook_client import GrowthBookClient  # noqa: E402

# Realistic payload: a prerequisite that is itself an experiment, a gated
# feature with its own experiment, a ramp with a passthrough holdback, and a
# plain flag. All splits decided by real hashing.
FEATURES = {
    "pricing-engine": {
        "defaultValue": "legacy",
        "rules": [{
            "key": "pricing-engine-rollout", "coverage": 1,
            "variations": ["legacy", "v2"], "weights": [0.5, 0.5],
            "meta": [{"key": "control"}, {"key": "v2"}],
        }],
    },
    "checkout-redesign": {
        "defaultValue": "classic",
        "rules": [{
            "parentConditions": [{"id": "pricing-engine", "condition": {"value": "v2"}}],
            "key": "checkout-redesign-exp", "coverage": 1,
            "variations": ["classic", "one-page"], "weights": [0.5, 0.5],
        }],
    },
    "onboarding-tour": {
        "defaultValue": "none",
        "rules": [
            {
                "key": "onboarding-tour-ramp", "coverage": 1,
                "variations": ["candidate", "holdback"], "weights": [0.2, 0.8],
                "meta": [{"key": "candidate"}, {"key": "holdback", "passthrough": True}],
            },
            {"force": "checklist"},
        ],
    },
    "cta-copy": {"defaultValue": "Start free trial"},
}

USERS = [{"id": f"user-{n}", "country": "US" if n % 3 else "CA"} for n in range(1, 31)]

PAGE_FEATURES = ["checkout-redesign", "onboarding-tour", "cta-copy"]


def expected_exposures(attributes):
    """Derive the exposures a page render must produce, from the SDK's own
    assignments (no hard-coded variations)."""
    gb = GrowthBook(attributes=dict(attributes), features=FEATURES)
    pricing = gb.eval_feature("pricing-engine")
    checkout = gb.eval_feature("checkout-redesign")
    tour = gb.eval_feature("onboarding-tour")
    gb.destroy()

    expected = {("pricing-engine-rollout", pricing.experimentResult.variationId)}
    if pricing.value == "v2":
        expected.add(("checkout-redesign-exp", checkout.experimentResult.variationId))
    # The ramp tracks either way: candidate directly, holdback via passthrough.
    tour_variation = 0 if tour.value == "candidate" else 1
    expected.add(("onboarding-tour-ramp", tour_variation))
    return expected


def render_page_sync(attributes):
    """One SSR request: per-request instance, several evals (one repeated,
    as templates do), then harvest the buffer."""
    gb = GrowthBook(attributes=attributes, features=FEATURES, defer_tracking=True)
    flags = {key: gb.get_feature_value(key, None) for key in PAGE_FEATURES}
    gb.is_on("checkout-redesign")  # template re-checks the same flag
    calls = gb.get_deferred_tracking_calls()
    gb.destroy()
    return flags, calls


async def handle_api_request(client, attributes):
    """One async API request: caller-owned buffer, several evals, JSON body."""
    buf = TrackingBuffer()
    ctx = UserContext(attributes=attributes)
    flags = {}
    for key in PAGE_FEATURES:
        flags[key] = (await client.eval_feature(key, ctx, tracking_buffer=buf)).value
    await client.is_on("checkout-redesign", ctx, tracking_buffer=buf)  # duplicate eval
    return {"flags": flags, "trackingCalls": buf.get_calls()}


def fire_through_js_sdk(batches):
    """Feed [{label, calls}] through the real JS SDK in Node; return
    {label: fired events}."""
    receiver = os.path.join(os.path.dirname(__file__), "js_receiver.js")
    proc = subprocess.run(
        ["node", receiver], input=json.dumps(batches).encode(),
        capture_output=True, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"js_receiver failed: {proc.stderr.decode()}")
    fired = json.loads(proc.stdout)
    print(f"  (receiver: JS SDK from {fired.pop('_receiver')})")
    return fired


def check(condition, message):
    if not condition:
        raise AssertionError(message)
    return 1


def verify_fired(label, fired, expected, attributes):
    checks = 0
    got = {(f["experiment"], f["variationId"]) for f in fired}
    checks += check(got == expected, f"{label}: fired {got} != expected {expected}")
    for f in fired:
        checks += check(f["hashValue"] == attributes["id"],
                        f"{label}: exposure carries wrong hashValue {f['hashValue']}")
        checks += check(f["userAttributes"] == attributes,
                        f"{label}: exposure user {f['userAttributes']} != {attributes}")
    return checks


def simulate_sync():
    checks, gated_out, batches, expectations = 0, 0, [], {}
    for user in USERS:
        attrs = dict(user)
        expected = expected_exposures(attrs)
        flags, calls = render_page_sync(attrs)

        # Request teardown mutates the attributes dict the SDK saw (destroy()
        # already cleared it in place — buffered snapshots must survive both).
        attrs["id"] = "recycled"
        attrs["country"] = "XX"

        page_json = json.dumps({"flags": flags, "trackingCalls": calls})  # the rendered page
        batches.append({"label": user["id"], "calls": json.loads(page_json)["trackingCalls"]})
        expectations[user["id"]] = (expected, dict(user))

        if flags["checkout-redesign"] == "classic" and ("checkout-redesign-exp", 0) not in expected:
            gated_out += 1  # gated by prereq: pricing exposure must still exist
            checks += check(("pricing-engine-rollout", 0) in expected,
                            f"{user['id']}: gated user lost the prerequisite exposure")

    fired_by_user = fire_through_js_sdk(batches)
    for label, (expected, attrs) in expectations.items():
        checks += verify_fired(label, fired_by_user[label], expected, attrs)

    exposures = sum(len(v) for v in fired_by_user.values())
    print(f"sync SSR: {len(USERS)} requests, {exposures} exposures fired by the JS SDK, "
          f"{gated_out} users gated out of checkout (prereq exposure still fired), "
          f"{checks} assertions passed")


async def simulate_async():
    server_side = []
    client = GrowthBookClient(Options(
        api_host="https://localhost.growthbook.io", client_key="sim",
        on_experiment_viewed=lambda experiment, result, user_context: server_side.append(
            (user_context.attributes["id"], experiment.key)
        ),
    ))
    await client.set_features(FEATURES)
    try:
        responses = await asyncio.gather(
            *(handle_api_request(client, dict(u)) for u in USERS)
        )
    finally:
        await client.close()

    checks, batches, expectations = 0, [], {}
    for user, response in zip(USERS, responses):
        expected = expected_exposures(user)  # sync reference: same assignments
        calls = response["trackingCalls"]
        checks += check(all(c["user"]["attributes"] == user for c in calls),
                        f"{user['id']}: buffer mixed in another user's exposure")
        got = {(c["experiment"]["key"], c["result"]["variationId"]) for c in calls}
        checks += check(len(calls) == len(got), f"{user['id']}: duplicate eval not deduped")
        checks += check(got == expected, f"{user['id']}: async {got} != sync reference {expected}")
        batches.append({"label": user["id"], "calls": calls})
        expectations[user["id"]] = (expected, dict(user))

    fired_by_user = fire_through_js_sdk(batches)
    for label, (expected, attrs) in expectations.items():
        checks += verify_fired(label, fired_by_user[label], expected, attrs)

    # Buffering was independent of the server-side callback, which deduped
    # per (user, experiment) exactly once.
    for user in USERS:
        expected_keys = {key for key, _ in expectations[user["id"]][0]}
        got_keys = [k for uid, k in server_side if uid == user["id"]]
        checks += check(sorted(got_keys) == sorted(expected_keys),
                        f"{user['id']}: server-side callback fired {got_keys}, expected {expected_keys}")

    exposures = sum(len(v) for v in fired_by_user.values())
    print(f"async API: {len(USERS)} concurrent requests, {exposures} exposures fired by the "
          f"JS SDK, {len(server_side)} server-side callback events alongside, "
          f"{checks} assertions passed")


if __name__ == "__main__":
    simulate_sync()
    asyncio.run(simulate_async())
    print("OK: every buffered exposure matched the SDK's own assignments and was "
          "fired by the real JS SDK with the exposure-time user context.")
