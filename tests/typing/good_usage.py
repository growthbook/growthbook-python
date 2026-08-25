"""Correct public-API usage — must type-check with ZERO errors under mypy AND pyright.

This file is checked by tests/test_typing.py (and the pyright CI step). If a
change to the SDK makes any line here error, the public typing contract broke.
"""
from typing import Any, Dict, List, Optional

from growthbook import (
    Experiment,
    FeatureResult,
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

# run() infers Result[T] from the experiment's variations.
res: Result[int] = gb.run(Experiment(key="t", variations=[1, 2]))
doubled: int = res.value * 2
str_res = gb.run(Experiment(key="t2", variations=["a", "b"]))
upper: str = str_res.value.upper()


# Tracking callbacks use these exact parameter names (both clients call by keyword).
def on_experiment_viewed(
    experiment: Experiment[Any], result: Result[Any], user_context: Optional[UserContext]
) -> None: ...


gb2 = GrowthBook(on_experiment_viewed=on_experiment_viewed)


# Keyword-only implementations are valid too — the clients only ever call by keyword.
def kw_only_cb(
    *, experiment: Experiment[Any], result: Result[Any], user_context: Optional[UserContext]
) -> None: ...


gb3 = GrowthBook(on_experiment_viewed=kw_only_cb)

# The async client (Options) accepts sync AND async callbacks — awaitables
# are scheduled on the running loop. Regression guard for the widened types.
async def async_viewed(
    experiment: Experiment[Any], result: Result[Any], user_context: Optional[UserContext]
) -> None: ...


def sync_usage(key: str, result: FeatureResult[Any], user_context: UserContext) -> None: ...
async def async_usage(key: str, result: FeatureResult[Any], user_context: UserContext) -> None: ...
def sync_sub(experiment: Experiment[Any], result: Result[Any]) -> None: ...
async def async_sub(experiment: Experiment[Any], result: Result[Any]) -> None: ...


def sync_logger(event_name: str, properties: Dict[str, Any], user_context: Optional[UserContext]) -> None: ...
async def async_logger(event_name: str, properties: Dict[str, Any], user_context: Optional[UserContext]) -> None: ...


sync_opts = Options(on_experiment_viewed=on_experiment_viewed, on_feature_usage=sync_usage)
async_opts = Options(on_experiment_viewed=async_viewed, on_feature_usage=async_usage)
client = GrowthBookClient(async_opts)
client.subscribe(sync_sub)
client.subscribe(async_sub)
client.set_event_logger(sync_logger)
client.set_event_logger(async_logger)
gb.set_event_logger(sync_logger)

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
