// Receiving half of the deferred-tracking simulation: the REAL GrowthBook JS
// SDK hydrates the Python-produced payloads and fires them, exactly as a
// browser would after an SSR render.
//
// stdin:  [{"label": str, "calls": TrackingData[]}, ...]
// stdout: {label: [{experiment, variationId, hashAttribute, hashValue,
//                   userAttributes}, ...]}
//
// SDK resolution: $GB_JS_SDK or an installed @growthbook/growthbook package,
// >= 1.7.0 (older SDKs fire deferred calls without the user argument).
// Deliberately NO fallback to a repo checkout's dist/ — a stale local build
// artifact must fail loudly, not silently verify the wrong SDK.
function loadSdk() {
  const candidates = [process.env.GB_JS_SDK, "@growthbook/growthbook"].filter(Boolean);
  for (const candidate of candidates) {
    try {
      return { sdk: require(candidate), source: candidate };
    } catch (e) {
      /* try the next candidate */
    }
  }
  throw new Error(
    "No GrowthBook JS SDK found. `npm install @growthbook/growthbook` (>=1.7.0) " +
      "or point GB_JS_SDK at one."
  );
}
const { sdk, source } = loadSdk();
const { GrowthBook } = sdk;

let input = "";
process.stdin.on("data", (d) => (input += d));
process.stdin.on("end", async () => {
  const payloads = JSON.parse(input);
  const out = {};
  for (const { label, calls } of payloads) {
    const fired = [];
    const gb = new GrowthBook({
      trackingCallback: (experiment, result, user) => {
        fired.push({
          experiment: experiment.key,
          variationId: result.variationId,
          hashAttribute: result.hashAttribute,
          hashValue: result.hashValue,
          userAttributes: user ? user.attributes : null,
        });
      },
    });
    gb.setDeferredTrackingCalls(calls);
    await gb.fireDeferredTrackingCalls();
    out[label] = fired;
    gb.destroy();
  }
  out._receiver = source;
  process.stdout.write(JSON.stringify(out));
});
