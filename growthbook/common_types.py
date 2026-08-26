#!/usr/bin/env python

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    NoReturn,
    Optional,
    Protocol,
    TypedDict,
    TypeVar,
    Union,
    Set,
    Tuple,
)
from enum import Enum
from abc import ABC, abstractmethod
from urllib.parse import urlparse as _urlparse

from typing_extensions import Required

# Runtime import (not TYPE_CHECKING): Options is a dataclass with a
# `tracking_plugins: Optional[List["PluginLike"]]` field, so the name must be
# resolvable at runtime for typing.get_type_hints(Options) and any dataclass
# introspection (pydantic, dacite, ...). plugins.base only imports stdlib at
# runtime, so this is cycle-free.
from .plugins.base import PluginLike

# Generic feature/experiment value type. Deliberately unbounded: a JSONValue
# bound would reject TypedDict/dataclass-shaped fallbacks (see JS SDK issue #1729,
# where the equivalent bound was shipped and then reverted).
T = TypeVar("T")

# The shape of a JSON-serializable value, mirroring the JS SDK's JSONValue.
# Used to annotate payload data; NOT used as a TypeVar bound (see note on T).
# bool precedes int because bool is an int subclass; int and float together
# cover the JS `number`.
JSONValue = Union[None, bool, int, float, str, List["JSONValue"], Dict[str, "JSONValue"]]

# Wire shapes: real payloads routinely omit keys, hence total=False.
class VariationMeta(TypedDict, total=False):
    key: str
    name: str
    passthrough: bool


class Filter(TypedDict, total=False):
    seed: str
    # A filter without ranges is meaningless; the eval loop indexes it directly.
    ranges: Required[List[Tuple[float, float]]]
    hashVersion: int
    attribute: str

class Experiment(Generic[T]):
    def __init__(
        self,
        key: str,
        variations: List[T],
        weights: Optional[List[float]] = None,
        active: bool = True,
        status: str = "running",
        coverage: Optional[float] = None,
        condition: Optional[Dict[str, Any]] = None,
        namespace: Optional[Tuple[str, float, float]] = None,
        url: str = "",
        include: Optional[Any] = None,
        groups: Optional[List[Any]] = None,
        force: Optional[int] = None,
        hashAttribute: str = "id",
        fallbackAttribute: Optional[str] = None,
        hashVersion: Optional[int] = None,
        ranges: Optional[List[Tuple[float, float]]] = None,
        meta: Optional[List[VariationMeta]] = None,
        filters: Optional[List[Filter]] = None,
        seed: Optional[str] = None,
        name: Optional[str] = None,
        phase: Optional[str] = None,
        disableStickyBucketing: bool = False,
        bucketVersion: Optional[int] = None,
        minBucketVersion: Optional[int] = None,
        parentConditions: Optional[List[Dict[str, Any]]] = None,
        customFields: Optional[Dict[str, Any]] = None,
        # NoReturn makes literal unknown kwargs a checker error (like TS excess
        # property checks) while **dict payload splats (typed Any) still pass;
        # at runtime unknown payload keys are swallowed as before.
        **_ignored: NoReturn,
    ) -> None:
        self.key = key
        self.variations = variations
        self.weights = weights
        self.active = active
        self.coverage = coverage
        self.condition = condition
        self.namespace = namespace
        self.force = force
        self.hashAttribute = hashAttribute
        self.hashVersion = hashVersion or 1
        self.ranges = ranges
        self.meta = meta
        self.filters = filters
        self.seed = seed
        self.name = name
        self.phase = phase
        self.disableStickyBucketing = disableStickyBucketing
        self.bucketVersion = bucketVersion or 0
        self.minBucketVersion = minBucketVersion or 0
        self.parentConditions = parentConditions
        # Custom Fields defined for the experiment in the GrowthBook UI.
        # Arrives from the API as a flat dict (e.g. {"cfl_abc123": "value"}).
        self.customFields = customFields or {}

        self.fallbackAttribute = None
        if not self.disableStickyBucketing:
            self.fallbackAttribute = fallbackAttribute

        # Deprecated properties
        self.status = status
        self.url = url
        self.include = include
        self.groups = groups

    def to_dict(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "key": self.key,
            "variations": self.variations,
            "weights": self.weights,
            "active": self.active,
            "coverage": self.coverage or 1,
            "condition": self.condition,
            "namespace": self.namespace,
            "force": self.force,
            "hashAttribute": self.hashAttribute,
            "hashVersion": self.hashVersion,
            "ranges": self.ranges,
            "meta": self.meta,
            "filters": self.filters,
            "seed": self.seed,
            "name": self.name,
            "phase": self.phase,
        }

        if self.fallbackAttribute:
            obj["fallbackAttribute"] = self.fallbackAttribute
        if self.disableStickyBucketing:
            obj["disableStickyBucketing"] = True
        if self.bucketVersion:
            obj["bucketVersion"] = self.bucketVersion
        if self.minBucketVersion:
            obj["minBucketVersion"] = self.minBucketVersion
        if self.parentConditions:
            obj["parentConditions"] = self.parentConditions
        if self.customFields:
            obj["customFields"] = self.customFields

        return obj

    def update(self, data: Dict[str, Any]) -> None:
        weights = data.get("weights", None)
        status = data.get("status", None)
        coverage = data.get("coverage", None)
        url = data.get("url", None)
        groups = data.get("groups", None)
        force = data.get("force", None)

        if weights is not None:
            self.weights = weights
        if status is not None:
            self.status = status
        if coverage is not None:
            self.coverage = coverage
        if url is not None:
            self.url = url
        if groups is not None:
            self.groups = groups
        if force is not None:
            self.force = force


class Result(Generic[T]):
    def __init__(
        self,
        variationId: int,
        inExperiment: bool,
        value: T,
        hashUsed: bool,
        hashAttribute: str,
        hashValue: str,
        featureId: Optional[str],
        meta: Optional[VariationMeta] = None,
        bucket: Optional[float] = None,
        stickyBucketUsed: bool = False,
    ) -> None:
        self.variationId = variationId
        self.inExperiment = inExperiment
        self.value = value
        self.hashUsed = hashUsed
        self.hashAttribute = hashAttribute
        self.hashValue = hashValue
        self.featureId = featureId or None
        self.bucket = bucket
        self.stickyBucketUsed = stickyBucketUsed

        self.key = str(variationId)
        self.name = ""
        self.passthrough = False

        if meta:
            if "name" in meta:
                self.name = meta["name"]
            if "key" in meta:
                self.key = meta["key"]
            if "passthrough" in meta:
                self.passthrough = meta["passthrough"]

    def to_dict(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "featureId": self.featureId,
            "variationId": self.variationId,
            "inExperiment": self.inExperiment,
            "value": self.value,
            "hashUsed": self.hashUsed,
            "hashAttribute": self.hashAttribute,
            "hashValue": self.hashValue,
            "key": self.key,
            "stickyBucketUsed": self.stickyBucketUsed,
        }

        if self.bucket is not None:
            obj["bucket"] = self.bucket
        if self.name:
            obj["name"] = self.name
        if self.passthrough:
            obj["passthrough"] = True

        return obj

class FeatureResult(Generic[T]):
    def __init__(
        self,
        value: Optional[T],
        source: str,
        experiment: Optional[Experiment[T]] = None,
        experimentResult: Optional[Result[T]] = None,
        ruleId: Optional[str] = None,
    ) -> None:
        self.value = value
        self.source = source
        self.ruleId = ruleId
        self.experiment = experiment
        self.experimentResult = experimentResult
        self.on = bool(value)
        self.off = not bool(value)

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "value": self.value,
            "source": self.source,
            "on": self.on,
            "off": self.off,
            "ruleId": self.ruleId or "",
        }
        if self.experiment:
            data["experiment"] = self.experiment.to_dict()
        if self.experimentResult:
            data["experimentResult"] = self.experimentResult.to_dict()

        return data

class Feature(object):
    def __init__(self, defaultValue: Any = None, rules: Optional[List[Union["FeatureRule", Dict[str, Any]]]] = None) -> None:
        if rules is None:
            rules = []
        self.defaultValue = defaultValue
        self.rules: List[FeatureRule] = [
            r if isinstance(r, FeatureRule) else FeatureRule(**r) for r in rules
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "defaultValue": self.defaultValue,
            "rules": [rule.to_dict() for rule in self.rules],
        }

class FeatureRule(object):
    def __init__(
        self,
        id: Optional[str] = None,
        key: str = "",
        variations: Optional[List[Any]] = None,
        weights: Optional[List[float]] = None,
        coverage: Optional[float] = None,
        condition: Optional[Dict[str, Any]] = None,
        namespace: Optional[Tuple[str, float, float]] = None,
        force: Optional[Any] = None,
        hashAttribute: str = "id",
        fallbackAttribute: Optional[str] = None,
        hashVersion: Optional[int] = None,
        range: Optional[Tuple[float, float]] = None,
        ranges: Optional[List[Tuple[float, float]]] = None,
        meta: Optional[List[VariationMeta]] = None,
        filters: Optional[List[Filter]] = None,
        seed: Optional[str] = None,
        name: Optional[str] = None,
        phase: Optional[str] = None,
        disableStickyBucketing: bool = False,
        bucketVersion: Optional[int] = None,
        minBucketVersion: Optional[int] = None,
        parentConditions: Optional[List[Dict[str, Any]]] = None,
        tracks: Optional[List[Dict[str, Any]]] = None,
        # See Experiment.__init__: checker-strict, runtime-permissive.
        **_ignored: NoReturn,
    ) -> None:

        if disableStickyBucketing:
            fallbackAttribute = None

        self.id = id
        self.key = key
        self.variations = variations
        self.weights = weights
        self.coverage = coverage
        self.condition = condition
        self.namespace = namespace
        self.force = force
        self.hashAttribute = hashAttribute
        self.fallbackAttribute = fallbackAttribute
        self.hashVersion = hashVersion or 1
        self.range = range
        self.ranges = ranges
        self.meta = meta
        self.filters = filters
        self.seed = seed
        self.name = name
        self.phase = phase
        self.disableStickyBucketing = disableStickyBucketing
        self.bucketVersion = bucketVersion or 0
        self.minBucketVersion = minBucketVersion or 0
        self.parentConditions = parentConditions
        # Remote-eval rules carry pre-evaluated experiment tracking events on
        # the force branch; see _fireRuleTracks in core.py.
        self.tracks = tracks

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if self.id:
            data["id"] = self.id
        if self.key:
            data["key"] = self.key
        if self.variations is not None:
            data["variations"] = self.variations
        if self.weights is not None:
            data["weights"] = self.weights
        if self.coverage and self.coverage != 1:
            data["coverage"] = self.coverage
        if self.condition is not None:
            data["condition"] = self.condition
        if self.namespace is not None:
            data["namespace"] = self.namespace
        if self.force is not None:
            data["force"] = self.force
        if self.hashAttribute != "id":
            data["hashAttribute"] = self.hashAttribute
        if self.hashVersion:
            data["hashVersion"] = self.hashVersion
        if self.range is not None:
            data["range"] = self.range
        if self.ranges is not None:
            data["ranges"] = self.ranges
        if self.meta is not None:
            data["meta"] = self.meta
        if self.filters is not None:
            data["filters"] = self.filters
        if self.seed is not None:
            data["seed"] = self.seed
        if self.name is not None:
            data["name"] = self.name
        if self.phase is not None:
            data["phase"] = self.phase
        if self.fallbackAttribute:
            data["fallbackAttribute"] = self.fallbackAttribute
        if self.disableStickyBucketing:
            data["disableStickyBucketing"] = True
        if self.bucketVersion:
            data["bucketVersion"] = self.bucketVersion
        if self.minBucketVersion:
            data["minBucketVersion"] = self.minBucketVersion
        if self.parentConditions:
            data["parentConditions"] = self.parentConditions
        if self.tracks:
            data["tracks"] = self.tracks

        return data

class AbstractStickyBucketService(ABC):
    # Assignment docs are Dict[str, Any] with a fixed shape:
    #   {"attributeName": str, "attributeValue": str, "assignments": Dict[str, str]}
    # Kept as plain dicts (not TypedDict) so third-party implementations
    # annotated with Dict stay compatible under type checking.
    @abstractmethod
    def get_assignments(self, attributeName: str, attributeValue: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def save_assignments(self, doc: Dict[str, Any]) -> None:
        pass

    def get_key(self, attributeName: str, attributeValue: str) -> str:
        return f"{attributeName}||{attributeValue}"

    # By default, just loop through all attributes and call get_assignments
    # Override this method in subclasses to perform a multi-query instead
    def get_all_assignments(self, attributes: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        docs: Dict[str, Dict[str, Any]] = {}
        for attributeName, attributeValue in attributes.items():
            doc = self.get_assignments(attributeName, attributeValue)
            if doc:
                docs[self.get_key(attributeName, attributeValue)] = doc
        return docs


class AbstractAsyncStickyBucketService(ABC):
    """Async twin of AbstractStickyBucketService for network-backed stores
    (Redis, DynamoDB, ...). Only usable with the async GrowthBookClient;
    the sync GrowthBook class rejects it at construction."""

    @abstractmethod
    async def get_assignments(self, attributeName: str, attributeValue: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def save_assignments(self, doc: Dict[str, Any]) -> None:
        pass

    def get_key(self, attributeName: str, attributeValue: str) -> str:
        return f"{attributeName}||{attributeValue}"

    # By default, just loop through all attributes and call get_assignments
    # Override this method in subclasses to perform a multi-query instead
    async def get_all_assignments(self, attributes: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        docs = {}
        for attributeName, attributeValue in attributes.items():
            doc = await self.get_assignments(attributeName, attributeValue)
            if doc:
                docs[self.get_key(attributeName, attributeValue)] = doc
        return docs

@dataclass
class StackContext:
    id: Optional[str] = None
    evaluated_features: Set[str] = field(default_factory=set)

class FeatureRefreshStrategy(Enum):
    STALE_WHILE_REVALIDATE = 'HTTP_REFRESH'
    SERVER_SENT_EVENTS = 'SSE'
@dataclass
class UserContext:
    # user_id: Optional[str] = None
    url: str = ""
    attributes: Dict[str, Any] = field(default_factory=dict)
    groups: Dict[str, str] = field(default_factory=dict)
    forced_variations: Dict[str, Any] = field(default_factory=dict)
    # Caller-supplied forced feature values. Sent to the proxy in remote-eval
    # mode (wire format: list of [key, value] tuples, matches JS SDK).
    forced_features: Dict[str, Any] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)
    sticky_bucket_assignment_docs: Dict[str, Any] = field(default_factory=dict)
    skip_all_experiments: bool = False


class TrackingCallback(Protocol):
    """Callback invoked when a user is assigned to an experiment variation.

    The parameter names are part of the contract: both the sync and async
    clients invoke this callback with keyword arguments (experiment=...,
    result=..., user_context=...), so implementations must use these exact
    names. Positional-only parameters cannot satisfy this contract.
    """

    def __call__(
        self,
        *,
        experiment: Experiment[Any],
        result: Result[Any],
        user_context: UserContext,
    ) -> None: ...


# Invoked positionally: (feature_key, feature_result, user_context).
FeatureUsageCallback = Callable[[str, "FeatureResult[Any]", UserContext], None]

# Invoked positionally: (event_name, properties, user_context). The sync
# GrowthBook client neither awaits nor schedules a returned value, so its
# event logger must be synchronous (an async def would produce a coroutine
# that is silently dropped).
EventLogger = Callable[[str, Dict[str, Any], UserContext], None]

# Async-client callback contracts (Options is consumed by GrowthBookClient):
# the async client schedules returned awaitables on the running loop, so
# implementations may be sync (return None) or async (return a coroutine).
# The sync GrowthBook client accepts the sync-only contracts above.
class AsyncTrackingCallback(Protocol):
    """Tracking callback for the async client. Same keyword-invocation
    contract as TrackingCallback (parameter names are part of the contract);
    additionally may be async — a returned awaitable is scheduled on the
    running loop, fire-and-forget."""

    def __call__(
        self,
        *,
        experiment: Experiment[Any],
        result: Result[Any],
        user_context: UserContext,
    ) -> Union[None, Awaitable[None]]: ...


AsyncFeatureUsageCallback = Callable[
    [str, "FeatureResult[Any]", UserContext], Union[None, Awaitable[None]]
]

# Async-client event logger: the async client awaits a returned coroutine,
# so implementations may be sync or async.
AsyncEventLogger = Callable[
    [str, Dict[str, Any], UserContext], Union[None, Awaitable[None]]
]


@dataclass
class Options:
    url: Optional[str] = None
    api_host: Optional[str] = "https://cdn.growthbook.io"
    client_key: Optional[str] = None
    decryption_key: Optional[str] = None
    cache_ttl: int = 60  # max_age: hard expiry for cached payloads (seconds).
    # Soft-expiry threshold (seconds). When set < cache_ttl AND remote_eval is
    # on, the async GrowthBookClient serves stale cached payloads inside
    # [stale_ttl, cache_ttl) and fires a fire-and-forget background refetch.
    # None = no SWR window (hard expiry at cache_ttl). Sync GrowthBook remote_eval
    # uses cache_ttl-only and ignores this field.
    stale_ttl: Optional[int] = None
    enabled: bool = True
    qa_mode: bool = False
    enable_dev_mode: bool = False
    # forced_variations: Dict[str, Any] = field(default_factory=dict)
    refresh_strategy: Optional[FeatureRefreshStrategy] = FeatureRefreshStrategy.STALE_WHILE_REVALIDATE
    sticky_bucket_service: Optional[Union[AbstractStickyBucketService, AbstractAsyncStickyBucketService]] = None
    sticky_bucket_identifier_attributes: Optional[List[str]] = None
    on_experiment_viewed: Optional[AsyncTrackingCallback] = None
    on_feature_usage: Optional[AsyncFeatureUsageCallback] = None
    tracking_plugins: Optional[List["PluginLike"]] = None
    http_connect_timeout: Optional[int] = None
    http_read_timeout: Optional[int] = None
    event_logger: Optional[AsyncEventLogger] = None
    remote_eval: bool = False
    cache_key_attributes: Optional[List[str]] = None
    remote_eval_cache_size: int = 1000
    # Opt-in sticky bucket prefetch cache for the async client. 0 (default)
    # disables caching: assignments are fetched per evaluation context,
    # matching the JS SDK's server-side GrowthBookClient. When > 0, fetched
    # assignments are reused for this many seconds per attributes dict
    # (bounded staleness across workers), LRU-bounded by
    # sticky_bucket_cache_size. Non-positive values disable caching.
    sticky_bucket_cache_ttl: float = 0
    sticky_bucket_cache_size: int = 1000


@dataclass
class GlobalContext:
    options: Options
    features: Dict[str, "Feature"] = field(default_factory=dict)
    saved_groups: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationContext:
    user: UserContext
    global_ctx: GlobalContext
    stack: StackContext
    # When set, core calls this instead of sticky_bucket_service.save_assignments
    # directly, letting the async client schedule persistence off the event loop.
    # None (the default) preserves the sync client's direct-call behavior.
    save_sticky_bucket_doc: Optional[Callable[[Dict[str, Any]], None]] = None


# ---------------------------------------------------------------------------
# Shared helpers used by both the sync GrowthBook class and the async
# GrowthBookClient. Living here (instead of one of the client modules) keeps
# the dependency direction one-way: clients depend on common_types, not on
# each other.
# ---------------------------------------------------------------------------


def features_from_dict(features_data: Optional[Dict[str, Any]]) -> Dict[str, "Feature"]:
    """Materialize a {key: feature_dict_or_Feature} mapping into Feature
    objects. Pass-through if a value is already a Feature."""
    out: Dict[str, "Feature"] = {}
    for key, feature in (features_data or {}).items():
        if isinstance(feature, Feature):
            out[key] = feature
        else:
            out[key] = Feature(
                rules=feature.get("rules", []),
                defaultValue=feature.get("defaultValue", None),
            )
    return out


def build_remote_eval_payload(
    attributes: Optional[Dict[str, Any]],
    forced_variations: Optional[Dict[str, Any]],
    url: Optional[str],
    forced_features: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct the POST body for /api/eval/{client_key}. Single source of
    truth for the wire shape — both sync and async clients route through here.

    `forced_features` is the caller's natural dict shape; on the wire it
    becomes a list of `[key, value]` pairs (matches the JS SDK's
    `Array.from(map.entries())` — JS-arrays serialize identically to either
    Python tuples or lists, but we emit lists for in-memory parity)."""
    return {
        "attributes": attributes or {},
        "forcedFeatures": [[k, v] for k, v in (forced_features or {}).items()],
        "forcedVariations": forced_variations or {},
        "url": url or "",
    }


def is_cloud_host(api_host: Optional[str]) -> bool:
    """True if api_host points at GrowthBook Cloud (which doesn't expose
    /api/eval). Handles schemeless inputs like "cdn.growthbook.io" — naive
    urlparse on those returns no hostname."""
    raw = (api_host or "").strip()
    if not raw:
        return False
    parsed = _urlparse(raw if "://" in raw else "https://" + raw)
    host = parsed.hostname or ""
    return host == "growthbook.io" or host.endswith(".growthbook.io")


def validate_remote_eval_options(
    client_key: Optional[str],
    decryption_key: Optional[str],
    sticky_bucket_service: Any,
    api_host: Optional[str],
) -> None:
    """Raise ValueError on any combination of options that's incompatible with
    remote-eval mode. Caller is responsible for any class-specific extras
    (e.g., the sync class's `stale_while_revalidate` flag, the async class's
    `refresh_strategy=STALE_WHILE_REVALIDATE`)."""
    if not client_key:
        raise ValueError("Must specify client_key for remote eval")
    if not api_host:
        # Without this guard, `_get_remote_eval_url(api_host, ...)` would fall
        # back to `https://cdn.growthbook.io` and POST `/api/eval/{key}` to
        # Cloud, which doesn't expose that endpoint — surfacing as an opaque
        # 404 (or SSL/connectivity error) instead of a clear config error.
        # The sync class's `api_host` defaults to "" and the async client's
        # Options accepts "" or None — both hit this without an explicit guard.
        raise ValueError(
            "Must specify api_host (pointing at a self-hosted proxy/edge) for remote eval"
        )
    if decryption_key:
        raise ValueError("Encryption is not available for remote eval")
    if sticky_bucket_service is not None:
        raise ValueError(
            "sticky_bucket_service is not compatible with remote_eval; "
            "the proxy handles sticky bucketing server-side"
        )
    if is_cloud_host(api_host):
        raise ValueError(
            "Cloud host does not support remote eval; use a self-hosted proxy/edge"
        )
