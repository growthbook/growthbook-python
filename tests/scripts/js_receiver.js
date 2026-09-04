// Receiving half of the deferred-tracking simulation: the REAL GrowthBook JS
// SDK hydrates the Python-produced payloads and fires them, exactly as a
// browser would after an SSR render.
//
// stdin:  [{"label": str, "calls": TrackingData[]}, ...]
// stdout: {label: [{experiment, variationId, hashAttribute, hashValue,
//                   userAttributes}, ...]}
//
// SDK resolution (needs @growthbook/growthbook >= 1.7.0 for the user arg on
// tracking callbacks): $GB_JS_SDK, an installed @growthbook/growthbook, or
// the growthbook monorepo checkout next to this repo.
const path = require("path");
function loadSdk() {
  const candidates = [
    process.env.GB_JS_SDK,
    "@growthbook/growthbook",
    path.resolve(__dirname, "../../../growthbook/packages/sdk-js/dist/cjs/index.js"),
  ].filter(Boolean);
  for (const candidate of candidates) {
    try {
      return { sdk: require(candidate), source: candidate };
    } catch (e) {
      /* try the next candidate */
    }
  }
  throw new Error("No GrowthBook JS SDK found; set GB_JS_SDK");
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
