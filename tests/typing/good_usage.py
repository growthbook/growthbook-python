"""Correct public-API usage — must type-check with ZERO errors under mypy AND pyright.

This file is checked by tests/test_typing.py (and the pyright CI step). If a
change to the SDK makes any line here error, the public typing contract broke.
"""
from typing import Any, Dict, List, Optional

from growthbook import (
    ContextualBanditAssignment,
    ContextualBanditDefinition,
    Experiment,
    FeatureResult,
    FeatureRule,
    GrowthBook,
    GrowthBookClient,
    JSONValue,
    Options,
    Result,
    UserContext,
)

gb = GrowthBook(attributes={"id": "1"}, qa_mode=True)

# Fallback-driven inference: return type comes from the fallback.
color: str = gb.get_feature_value("banner", "blue")
count: int = gb.get_feature_value("max-items", 5)
shouted = gb.get_feature_value("banner", "blue").upper()
flag: bool = gb.is_on("dark-mode")

# Contextual bandit payload fields are first-class named kwargs, and the
# assignment/definition shapes are exported TypedDicts.
cb_definition: ContextualBanditDefinition = {
    "banditVersion": 7,
    "contexts": [{"leafId": 1, "condition": {}, "weights": [1.0, 0.0]}],
}
cb_rule = FeatureRule(contextualBanditRef="cb_1", contextualVariations=["a", "b"])
cb_exp = Experiment(
    key="cb",
    variations=["a", "b"],
    contextualBandit={"leafId": 1, "variationWeights": [1.0, 0.0]},
)
cb_assignment: Optional[ContextualBanditAssignment] = cb_exp.contextualBandit

# run() infers Result[T] from the experiment's variations.
res: Result[int] = gb.run(Experiment(key="t", variations=[1, 2]))
doubled: int = res.value * 2
# Contextual bandit exposure metadata on Result (None for non-CB results).
res_leaf: Optional[int] = res.leafId
res_weights: Optional[List[float]] = res.variationWeights
res_bandit_version: Optional[int] = res.banditVersion
str_res = gb.run(Experiment(key="t2", variations=["a", "b"]))
upper: str = str_res.value.upper()


# Tracking callbacks use these exact parameter names (both clients call by keyword).
# The clients always pass a real UserContext, so the natural (non-Optional)
# annotation must be accepted.
def on_experiment_viewed(
    experiment: Experiment[Any], result: Result[Any], user_context: UserContext
) -> None: ...


gb2 = GrowthBook(on_experiment_viewed=on_experiment_viewed)


# Keyword-only implementations are valid too — the clients only ever call by keyword.
def kw_only_cb(
    *, experiment: Experiment[Any], result: Result[Any], user_context: UserContext
) -> None: ...


gb3 = GrowthBook(on_experiment_viewed=kw_only_cb)


# Widening user_context to Optional stays valid (param contravariance).
def optional_ctx_cb(
    experiment: Experiment[Any], result: Result[Any], user_context: Optional[UserContext]
) -> None: ...


gb4 = GrowthBook(on_experiment_viewed=optional_ctx_cb)

# The async client (Options) accepts sync AND async callbacks — awaitables
# are scheduled on the running loop. Regression guard for the widened types.
async def async_viewed(
    experiment: Experiment[Any], result: Result[Any], user_context: UserContext
) -> None: ...


def sync_usage(key: str, result: FeatureResult[Any], user_context: UserContext) -> None: ...
async def async_usage(key: str, result: FeatureResult[Any], user_context: UserContext) -> None: ...
def sync_sub(experiment: Experiment[Any], result: Result[Any]) -> None: ...
async def async_sub(experiment: Experiment[Any], result: Result[Any]) -> None: ...


def sync_logger(event_name: str, properties: Dict[str, Any], user_context: UserContext) -> None: ...
async def async_logger(event_name: str, properties: Dict[str, Any], user_context: UserContext) -> None: ...


sync_opts = Options(on_experiment_viewed=on_experiment_viewed, on_feature_usage=sync_usage)
async_opts = Options(on_experiment_viewed=async_viewed, on_feature_usage=async_usage)
client = GrowthBookClient(async_opts)
client.subscribe(sync_sub)
client.subscribe(async_sub)
client.set_event_logger(sync_logger)
client.set_event_logger(async_logger)
gb.set_event_logger(sync_logger)

# The async client's inference mirrors the sync client end-to-end.
# (Statically checked only — never executed.)
async def _async_usage() -> None:
    ctx = UserContext(attributes={"id": "1"})
    a_color: str = await client.get_feature_value("banner", "blue", ctx)
    a_count: int = await client.get_feature_value("max-items", 5, ctx)
    a_res: Result[int] = await client.run(Experiment(key="t", variations=[1, 2]), ctx)
    a_doubled: int = a_res.value * 2
    a_fr: FeatureResult[Any] = await client.eval_feature("banner", ctx)
    a_flag: bool = await client.is_on("dark-mode", ctx)


# Payload dict splats with unknown server keys stay valid.
payload: Dict[str, Any] = {"key": "t", "variations": [1, 2], "unknownServerField": True}
exp = Experiment(**payload)

# JSONValue annotates JSON-shaped payload data.
value: JSONValue = {"nested": [1, "two", None, {"deep": True}]}

# Options and setters.
opts = Options(client_key="sdk-abc123", cache_ttl=60)
gb.set_attributes({"id": "2", "beta": True})
attrs: Dict[str, Any] = gb.get_attributes()
unsubscribe = gb.subscribe(lambda experiment, result: None)
unsubscribe()
