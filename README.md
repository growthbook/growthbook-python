# GrowthBook Python SDK

![Build Status](https://github.com/growthbook/growthbook-python/workflows/Build/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/growthbook.svg?maxAge=2592000)](https://pypi.org/project/growthbook/)
[![PyPI](https://img.shields.io/pypi/pyversions/growthbook.svg)](https://pypi.org/project/growthbook/)

## Overview

Powerful Feature flagging and A/B testing for Python apps.

- **Lightweight and fast**
- **Local evaluation**, no network requests required
- Flexible **targeting**
- **Use your existing event tracking** (GA, Segment, Mixpanel, custom)
- **Remote configuration** to change feature flags without deploying new code
- **Async support** with real-time feature updates
- Python 3.9+
- 100% test coverage

## Installation

`pip install growthbook`

## Type Checking Support

The GrowthBook Python SDK is fully typed and includes inline type hints for all public APIs. This enables:

- **Better IDE support** with autocomplete and inline documentation
- **Type safety** - catch bugs at development time with mypy or other type checkers
- **Better code documentation** - types serve as inline documentation
- **Safer refactoring** - type checkers will catch breaking changes

To use type checking with mypy:

```bash
pip install mypy
mypy your_code.py
```

The SDK includes a `py.typed` marker file and is compliant with [PEP 561](https://www.python.org/dev/peps/pep-0561/).

### Built-in type inference

Feature evaluation infers types from the fallback value you provide, and
experiments infer their result type from the variations — no extra setup:

```python
# Inferred as `str` from the fallback ("blue")
color = gb.get_feature_value("button-color", "blue")

# result.value is inferred as `str` from the variations
result = gb.run(Experiment(key="my-test", variations=["blue", "green"]))
```

### Strict typing (generated feature keys)

For compile-time checking of feature *keys* and per-key value types — the
equivalent of the JS SDK's `GrowthBook<AppFeatures>` — generate a typed client
from your features JSON (the SDK endpoint payload, e.g.
`https://cdn.growthbook.io/api/features/<client_key>`):

```bash
python -m growthbook.codegen --input features.json --output growthbook_features.py
```

If your endpoint has encryption enabled (the payload carries `encryptedFeatures`),
pass `--decryption-key <key>` to decrypt it before generating.

Then use the generated subclasses anywhere you'd use the plain clients. They
add zero runtime behavior — all checking happens in mypy/pyright/your IDE:

```python
from growthbook_features import TypedGrowthBook

gb = TypedGrowthBook(api_host="...", client_key="...")

gb.is_on("dark_mode")                        # OK
gb.is_on("dark_mod")                         # checker error: typo in feature key
gb.get_feature_value("max_items", "10")      # checker error: fallback must be a number
gb.get_feature_value("banner_text", None)    # Optional[str]: None if the flag isn't set
gb.eval_feature("dark_mode").value           # Optional[bool] instead of Any
```

The async client gets the same treatment:

```python
from growthbook_features import TypedGrowthBookClient

client = TypedGrowthBookClient(options)
color = await client.get_feature_value("banner_text", "hello", user)  # str
```

For dynamic keys (e.g. looping over feature names), the generated `FeatureKey`
Literal is the escape hatch:

```python
from typing import cast
from growthbook_features import FeatureKey

for k in my_keys:
    gb.is_on(cast(FeatureKey, k))
```

Regenerate the file whenever your feature list changes (e.g. as a CI step).
Features whose type can't be inferred (no `defaultValue` and no typed rules)
are listed in a warning at generation time and fall back to `Any`.

## Quick Usage

```python
from growthbook import GrowthBook

# User attributes for targeting and experimentation
attributes = {
  "id": "123",
  "customUserAttribute": "foo"
}

# The parameter names matter: the SDK invokes this callback
# with keyword arguments (experiment=, result=, user_context=)
def on_experiment_viewed(experiment, result, user_context):
  # Use whatever event tracking system you want
  analytics.track(attributes["id"], "Experiment Viewed", {
    'experimentId': experiment.key,
    'variationId': result.variationId
  })

# Create a GrowthBook instance
gb = GrowthBook(
  attributes = attributes,
  on_experiment_viewed = on_experiment_viewed,
  api_host = "https://cdn.growthbook.io",
  client_key = "sdk-abc123"
)

# Load features from the GrowthBook API with caching
gb.load_features()

# Simple on/off feature gating
if gb.is_on("my-feature"):
  print("My feature is on!")

# Get the value of a feature with a fallback
color = gb.get_feature_value("button-color-feature", "blue")
```

### Web Frameworks (Django, Flask, etc.)

For web frameworks, you should create a new `GrowthBook` instance for every incoming request and call `destroy()` at the end of the request to clean up resources.

In Django, for example, this is best done with a simple middleware:

```python
from growthbook import GrowthBook

def growthbook_middleware(get_response):
    def middleware(request):
        request.gb = GrowthBook(
          # ...
        )
        request.gb.load_features()

        response = get_response(request)

        request.gb.destroy() # Cleanup

        return response
    return middleware
```

Then, you can easily use GrowthBook in any of your views:

```python
def index(request):
    feature_enabled = request.gb.is_on("my-feature")
    # ...
```

## Quick Usage - Async Client

```python
from growthbook import GrowthBookClient, Options, UserContext, FeatureRefreshStrategy
import asyncio

async def main():
    # Create client options
    options = Options(
        api_host="https://cdn.growthbook.io",
        client_key="sdk-abc123",
        # Optional: Enable real-time feature updates
        refresh_strategy=FeatureRefreshStrategy.SERVER_SENT_EVENTS
    )

    # Create and initialize client
    client = GrowthBookClient(options)
    try:
        # Initialize the client before using it
        success = await client.initialize()
        if not success:
            print("Failed to initialize GrowthBook client")
            return

        # Create user context for targeting
        user = UserContext(
            attributes={
                "id": "123",
                "country": "US",
                "premium": True
            }
        )

        # Simple feature evaluation
        if await client.is_on("new-homepage", user):
            print("New homepage is enabled!")

        # Get feature value with fallback
        color = await client.get_feature_value("button-color", "blue", user)
        print(f"Button color is {color}")

        # Run an experiment
        result = await client.run(
            Experiment(
                key="my-test",
                variations=["A", "B"]
            ),
            user
        )
        print(f"User got variation: {result.value}")
    finally:
        # Always close the client when done
        await client.close()

# Run the async code
asyncio.run(main())
```

### Async Web Framework Integration

The async client works great with async web frameworks like FastAPI:

```python
from fastapi import FastAPI, Depends
from growthbook import GrowthBookClient, Options, UserContext

app = FastAPI()

# Create a single client instance
gb_client = GrowthBookClient(
    Options(
        api_host="https://cdn.growthbook.io",
        client_key="sdk-abc123"
    )
)

@app.on_event("startup")
async def startup():
    # Initialize the client when the app starts
    await gb_client.initialize()

@app.on_event("shutdown")
async def shutdown():
    # Clean up when the app shuts down
    await gb_client.close()

@app.get("/")
async def root(user_id: str):
    # Create user context for the request
    user = UserContext(attributes={"id": user_id})
    
    # Use features
    show_new_ui = await gb_client.is_on("new-ui", user)
    return {"new_ui": show_new_ui}
```

### Real-time Feature Updates

The async client supports real-time feature updates using Server-Sent Events:

```python
from growthbook import GrowthBookClient, Options, FeatureRefreshStrategy

client = GrowthBookClient(
    Options(
        api_host="https://cdn.growthbook.io",
        client_key="sdk-abc123",
        # Enable SSE for real-time updates
        refresh_strategy=FeatureRefreshStrategy.SERVER_SENT_EVENTS
    )
)
```

### Concurrency and Thread Safety

The async client is designed to be thread-safe and handle concurrent requests efficiently. You can safely use a single client instance across multiple coroutines. For web applications, you can create a single client instance at startup and share it across requests. Here's an example:

```python
from fastapi import FastAPI
from growthbook import GrowthBookClient, Options, UserContext
import asyncio

app = FastAPI()

# Single client instance shared across all requests
gb_client = GrowthBookClient(Options(
    api_host="https://cdn.growthbook.io",
    client_key="sdk-abc123"
))

@app.on_event("startup")
async def startup():
    await gb_client.initialize()

@app.on_event("shutdown")
async def shutdown():
    await gb_client.close()

@app.get("/batch")
async def batch_process(user_ids: list[str]):
    # Safely process multiple users concurrently
    tasks = []
    for user_id in user_ids:
        user = UserContext(attributes={"id": user_id})
        tasks.append(gb_client.eval_feature("new-feature", user))
    
    results = await asyncio.gather(*tasks)
    return {"results": results}
```

Note: While the client is thread-safe, you should not share a single `UserContext` instance across different requests. Create a new `UserContext` for each request to maintain proper isolation.

## Loading Features

There are two ways to load feature flags into the GrowthBook SDK. You can either use the built-in fetching/caching logic or implement your own custom solution.

### Built-in Fetching and Caching

To use the built-in fetching and caching logic, in the `GrowthBook` constructor, pass in your GrowthBook `api_host` and `client_key`. If you have encryption enabled for your GrowthBook endpoint, you also need to pass the `decryption_key` into the constructor.

Then, call the `load_features()` method to initiate the HTTP request with a cache layer.

Here's a full example:

```python
gb = GrowthBook(
  api_host = "https://cdn.growthbook.io",
  client_key = "sdk-abc123",
  # How long to cache features in seconds (Optional, default 60s)
  cache_ttl = 60,
)
gb.load_features()
```

#### Caching

GrowthBook comes with a custom in-memory cache. If you run Python in a multi-process mode, the different processes cannot share memory, so you likely want to switch to a distributed cache system like Redis instead.

Here is an example of using Redis:

```python
from redis import Redis
import json
from growthbook import GrowthBook, AbstractFeatureCache, feature_repo

class RedisFeatureCache(AbstractFeatureCache):
  def __init__(self):
    self.r = Redis(host='localhost', port=6379)
    self.prefix = "gb:"

  def get(self, key: str):
    data = self.r.get(self.prefix + key)
    # Data stored as a JSON string, parse into dict before returning
    return None if data is None else json.loads(data)

  def set(self, key: str, value: dict, ttl: int) -> None:
    self.r.set(self.prefix + key, json.dumps(value))
    self.r.expire(self.prefix + key, ttl)

# Configure GrowthBook to use your custom cache class
feature_repo.set_cache(RedisFeatureCache())
```

### Custom Implementation

If you prefer to handle the entire fetching/caching logic yourself, you can just pass in a `dict` of features from the GrowthBook API directly into the constructor:

```python
# From the GrowthBook API
features = {'my-feature':{'defaultValue':False}}

gb = GrowthBook(
  features = features
)
```

Note: When doing this, you do not need to specify your `api_host` or `client_key` and you don't need to call `gb.load_features()`.

## GrowthBook class

The GrowthBook constructor has the following parameters:

- **enabled** (`bool`) - Flag to globally disable all experiments. Default true.
- **attributes** (`dict`) - Dictionary of user attributes that are used for targeting and to assign variations
- **url** (`str`) - The URL of the current request (if applicable)
- **qa_mode** (`boolean`) - If true, random assignment is disabled and only explicitly forced variations are used.
- **on_experiment_viewed** (`callable`) - A function invoked with keyword arguments `experiment`, `result`, and `user_context` when a user is assigned to an experiment. Implementations must use these exact parameter names.
- **api_host** (`str`) - The GrowthBook API host to fetch feature flags from. Defaults to `https://cdn.growthbook.io`
- **client_key** (`str`) - The client key that will be passed to the API Host to fetch feature flags
- **decryption_key** (`str`) - If the GrowthBook API endpoint has encryption enabled, specify the decryption key here
- **cache_ttl** (`int`) - How long to cache features in-memory from the GrowthBook API (seconds, default `60`)
- **features** (`dict`) - Feature definitions from the GrowthBook API (only required if `client_key` is not specified)
- **forced_variations** (`dict`) - Dictionary of forced experiment variations (used for QA)

There are also getter and setter methods for features and attributes if you need to update them later in the request:

```python
gb.set_features(gb.get_features())
gb.set_attributes(gb.get_attributes())
```

### Attributes

You can specify attributes about the current user and request. These are used for two things:

1. Feature targeting (e.g. paid users get one value, free users get another)
2. Assigning persistent variations in A/B tests (e.g. user id "123" always gets variation B)

Attributes can be any JSON data type - boolean, integer, float, string, list, or dict.

```python
attributes = {
  'id': "123",
  'loggedIn': True,
  'age': 21.5,
  'tags': ["tag1", "tag2"],
  'account': {
    'age': 90
  }
}

# Pass into constructor
gb = GrowthBook(attributes = attributes)

# Or set later
gb.set_attributes(attributes)
```

### Tracking Experiments

Any time an experiment is run to determine the value of a feature, you want to track that event in your analytics system.

You can use the `on_experiment_viewed` option to do this:

```python
from typing import Any, Optional
from growthbook import GrowthBook, Experiment, Result, UserContext

# The parameter names matter: the SDK invokes this callback
# with keyword arguments (experiment=, result=, user_context=)
def on_experiment_viewed(
    experiment: Experiment[Any], result: Result[Any], user_context: UserContext
) -> None:
  # Use whatever event tracking system you want
  analytics.track(attributes["id"], "Experiment Viewed", {
    'experimentId': experiment.key,
    'variationId': result.variationId
  })

# Pass into constructor
gb = GrowthBook(
  on_experiment_viewed = on_experiment_viewed
)
```

The callback fires once for each experiment a user is assigned to during an evaluation. That includes experiments that decide a prerequisite feature and passthrough variations (the control arm of a holdout or ramp), not just the experiment that produced the final value. Likewise, `on_feature_usage` fires for every feature evaluated along the way, prerequisites included, once per evaluation.

#### Deferred Tracking

Sometimes the process evaluating features isn't the right place to send analytics from — for example a server that evaluates on behalf of browser or mobile clients, which then report their own exposures. For that, turn on `defer_tracking`: each exposure is buffered as a `{"experiment", "result", "user"}` dict — the same shape the JavaScript SDK uses — so you can hand the list straight to a client's `setDeferredTrackingCalls`. Buffering is independent of `on_experiment_viewed`; if you configure both, exposures are reported through both channels, so pick one.

```python
gb = GrowthBook(attributes={"id": user_id}, defer_tracking=True)
gb.eval_feature("my-feature")
exposures = gb.get_deferred_tracking_calls()  # forward these to the client
```

The buffer keeps one entry per unique assignment and lives as long as the instance does, so use one instance per request. With the async `GrowthBookClient`, the buffer lives on each `UserContext` instead: create it with `UserContext(attributes=..., defer_tracking=True)` and read `user_context.get_deferred_tracking_calls()` after evaluating, so one request's exposures never mix with another's.

On the receiving side, a `GrowthBook` with `on_experiment_viewed` set can load forwarded exposures with `set_deferred_tracking_calls(calls)` and send them through its callback with `fire_deferred_tracking_calls()`; each exposure is delivered with the user it was recorded for.

#### Built-in Tracking Plugin

For easier setup, you can use the built-in tracking plugin that automatically sends experiment and feature events to GrowthBook's data warehouse:

```python
from growthbook import GrowthBook
from growthbook.plugins import growthbook_tracking_plugin, request_context_plugin

gb = GrowthBook(
  attributes={"id": "user-123"},
  plugins=[
    request_context_plugin(),  # Extracts request data
    growthbook_tracking_plugin(
      ingestor_host="https://gb-ingest.growthbook.io",
      # Optional: Add custom tracking callback
      additional_callback=my_custom_tracker
    )
  ]
)

# Events are now automatically tracked for experiments and features
result = gb.run(experiment)  # -> Tracked automatically
is_enabled = gb.is_on("my-feature")  # -> Tracked automatically
```

The tracking plugin provides batching, error handling, and works alongside your existing tracking callbacks. See the [plugin documentation](https://docs.growthbook.io/lib/python#tracking-plugins) for more details.

## Using Features

There are 3 main methods for interacting with features.

- `gb.is_on("feature-key")` returns true if the feature is on
- `gb.is_off("feature-key")` returns false if the feature is on
- `gb.get_feature_value("feature-key", "default")` returns the value of the feature with a fallback

In addition, you can use `gb.evalFeature("feature-key")` to get back a `FeatureResult` object with the following properties:

- **value** - The JSON-decoded value of the feature (or `None` if not defined)
- **on** and **off** - The JSON-decoded value cast to booleans
- **source** - Why the value was assigned to the user. One of `unknownFeature`, `defaultValue`, `force`, or `experiment`
- **experiment** - Information about the experiment (if any) which was used to assign the value to the user
- **experimentResult** - The result of the experiment (if any) which was used to assign the value to the user

## Sticky Bucketing

By default GrowthBook does not persist assigned experiment variations for a user. We rely on deterministic hashing to ensure that the same user attributes always map to the same experiment variation. However, there are cases where this isn't good enough. For example, if you change targeting conditions in the middle of an experiment, users may stop being shown a variation even if they were previously bucketed into it.

Sticky Bucketing is a solution to these issues. You can provide a Sticky Bucket Service to the GrowthBook instance to persist previously seen variations and ensure that the user experience remains consistent for your users.

A sample `InMemoryStickyBucketService` implementation is provided for reference, but in production you will definitely want to implement your own version using a database, cookies, or similar for persistence.

Sticky Bucket documents contain three fields

- `attributeName` - The name of the attribute used to identify the user (e.g. `id`, `cookie_id`, etc.)
- `attributeValue` - The value of the attribute (e.g. `123`)
- `assignments` - A dictionary of persisted experiment assignments. For example: `{"exp1__0":"control"}`

The attributeName/attributeValue combo is the primary key.

Here's an example implementation using a theoretical `db` object:

```python
from growthbook import AbstractStickyBucketService, GrowthBook

class MyStickyBucketService(AbstractStickyBucketService):
    # Lookup a sticky bucket document
    def get_assignments(self, attributeName: str, attributeValue: str) -> Optional[Dict]:
        return db.find({
          "attributeName": attributeName,
          "attributeValue": attributeValue
        })

    def save_assignments(self, doc: Dict) -> None:
        # Insert new record if not exists, otherwise update
        db.upsert({
            "attributeName": doc["attributeName"],
            "attributeValue": doc["attributeValue"]
        }, {
          "$set": {
            "assignments": doc["assignments"]
          }
        })

# Pass in an instance of this service to your GrowthBook constructor
gb = GrowthBook(
  sticky_bucket_service = MyStickyBucketService()
)
```

### Async Sticky Bucketing (GrowthBookClient)

The async `GrowthBookClient` supports network-backed sticky bucket services (Redis, DynamoDB, etc.) without blocking the event loop. There are two options:

- **Async services** — subclass `AbstractAsyncStickyBucketService` and implement `async` versions of `get_assignments` / `save_assignments`. Optionally override `get_all_assignments` to batch lookups into one round trip (e.g. a single Redis `MGET`).
- **Existing sync services** — any `AbstractStickyBucketService` subclass also works with `GrowthBookClient` unchanged; its blocking calls are offloaded to a thread pool.

```python
import json
from growthbook import AbstractAsyncStickyBucketService, GrowthBookClient, Options

class RedisStickyBucketService(AbstractAsyncStickyBucketService):
    def __init__(self, redis):  # e.g. redis.asyncio.Redis
        self.redis = redis

    async def get_assignments(self, attributeName: str, attributeValue: str):
        raw = await self.redis.get(self.get_key(attributeName, attributeValue))
        return json.loads(raw) if raw else None

    async def save_assignments(self, doc: dict) -> None:
        key = self.get_key(doc["attributeName"], doc["attributeValue"])
        await self.redis.set(key, json.dumps(doc))

    # Optional: batch all lookups for a user into a single MGET
    async def get_all_assignments(self, attributes: dict):
        keys = [self.get_key(k, v) for k, v in attributes.items()]
        docs = {}
        for raw in await self.redis.mget(keys):
            if raw:
                doc = json.loads(raw)
                docs[self.get_key(doc["attributeName"], doc["attributeValue"])] = doc
        return docs

client = GrowthBookClient(Options(
    api_host="https://cdn.growthbook.io",
    client_key="sdk-abc123",
    sticky_bucket_service=RedisStickyBucketService(redis),
))
```

Behaviors to be aware of:

- **Reads are per evaluation.** Assignments are fetched for the supplied `UserContext` on each evaluation, matching the JavaScript SDK's multi-user client. Concurrent evaluations for the same user share one in-flight lookup. To trade staleness for fewer lookups on hot users, opt into a bounded cache with `Options(sticky_bucket_cache_ttl=30, sticky_bucket_cache_size=1000)` (seconds / max users; disabled by default).
- **Writes are fire-and-forget.** Evaluation never waits on persistence. New assignments are immediately visible to later evaluations in the same process (the client keeps an authoritative in-process copy of every document it has written, so a slow or stale store read can never roll back an assignment). Writes from *other* processes become visible on the next fetch.
- **Flushing.** `await client.flush_sticky_bucket_saves()` waits for all pending writes to persist — useful in serverless environments and tests. `await client.close()` flushes automatically.
- The synchronous `GrowthBook` class only accepts synchronous services; passing an async service raises `ValueError` at construction.

### Async Callbacks (GrowthBookClient)

With `GrowthBookClient`, the `on_experiment_viewed` and `on_feature_usage` options — and callbacks registered via `client.subscribe()` — may be either regular functions or coroutines. Coroutine callbacks are scheduled on the event loop without blocking evaluation, and a tracking callback that raises is retried on the next evaluation of the same experiment/user pair.

## Inline Experiments

Instead of declaring all features up-front and referencing them by ids in your code, you can also just run an experiment directly. This is done with the `run` method:

```python
from growthbook import Experiment

exp = Experiment(
  key = "my-experiment",
  variations = ["red", "blue", "green"]
)

# Either "red", "blue", or "green"
print(gb.run(exp).value)
```

As you can see, there are 2 required parameters for experiments, a string key, and an array of variations. Variations can be any data type, not just strings.

There are a number of additional settings to control the experiment behavior:

- **key** (`str`) - The globally unique tracking key for the experiment
- **variations** (`any[]`) - The different variations to choose between
- **seed** (`str`) - Added to the user id when hashing to determine a variation. Defaults to the experiment `key`
- **weights** (`float[]`) - How to weight traffic between variations. Must add to 1.
- **coverage** (`float`) - What percent of users should be included in the experiment (between 0 and 1, inclusive)
- **condition** (`dict`) - Targeting conditions
- **force** (`int`) - All users included in the experiment will be forced into the specified variation index
- **hashAttribute** (`string`) - What user attribute should be used to assign variations (defaults to "id")
- **hashVersion** (`int`) - What version of our hashing algorithm to use. We recommend using the latest version `2`.
- **namespace** (`tuple[str,float,float]`) - Used to run mutually exclusive experiments.

Here's an example that uses all of them:

```python
exp = Experiment(
  key="my-test",
  # Variations can be a list of any data type
  variations=[0, 1],
  # If this changes, it will re-randomize all users in the experiment
  seed="abcdef123456",
  # Run a 40/60 experiment instead of the default even split (50/50)
  weights=[0.4, 0.6],
  # Only include 20% of users in the experiment
  coverage=0.2,
  # Targeting condition using a MongoDB-like syntax
  condition={
    'country': 'US',
    'browser': {
      '$in': ['chrome', 'firefox']
    }
  },
  # Use an alternate attribute for assigning variations (default is 'id')
  hashAttribute="sessionId",
  # Use the latest hashing algorithm
  hashVersion=2,
  # Includes the first 50% of users in the "pricing" namespace
  # Another experiment with a non-overlapping range will be mutually exclusive (e.g. [0.5, 1])
  namespace=("pricing", 0, 0.5),
)
```

### Inline Experiment Return Value

A call to `run` returns a `Result` object with a few useful properties:

```python
result = gb.run(exp)

# If user is part of the experiment
print(result.inExperiment) # True or False

# The index of the assigned variation
print(result.variationId) # e.g. 0 or 1

# The value of the assigned variation
print(result.value) # e.g. "A" or "B"

# If the variation was randomly assigned by hashing user attributes
print(result.hashUsed) # True or False

# The user attribute used to assign a variation
print(result.hashAttribute) # "id"

# The value of that attribute
print(result.hashValue) # e.g. "123"
```

The `inExperiment` flag will be false if the user was excluded from being part of the experiment for any reason (e.g. failed targeting conditions).

The `hashUsed` flag will only be true if the user was randomly assigned a variation. If the user was forced into a specific variation instead, this flag will be false.

### Example Experiments

3-way experiment with uneven variation weights:

```python
gb.run(Experiment(
  key = "3-way-uneven",
  variations = ["A","B","C"],
  weights = [0.5, 0.25, 0.25]
))
```

Slow rollout (10% of users who match the targeting condition):

```python
# User is marked as being in "qa" and "beta"
gb = GrowthBook(
  attributes = {
    "id": "123",
    "beta": True,
    "qa": True,
  },
)

gb.run(Experiment(
  key = "slow-rollout",
  variations = ["A", "B"],
  coverage = 0.1,
  condition = {
    'beta': True
  }
))
```

Complex variations

```python
result = gb.run(Experiment(
  key = "complex-variations",
  variations = [
    ("blue", "large"),
    ("green", "small")
  ],
))

# Either "blue,large" OR "green,small"
print(result.value[0] + "," + result.value[1])
```

Assign variations based on something other than user id

```python
gb = GrowthBook(
  attributes = {
    "id": "123",
    "company": "growthbook"
  }
)

# Users in the same company will always get the same variation
gb.run(Experiment(
  key = "by-company-id",
  variations = ["A", "B"],
  hashAttribute = "company"
)) 
```

## Logging

The GrowthBook SDK uses a Python logger with the name `growthbook` and includes helpful info for debugging as well as warnings/errors if something is misconfigured.

Here's an example of logging to the console

```python
import logging

logger = logging.getLogger('growthbook')
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
```
