"""Wrong public-API usage — every line carrying an expect-error tag MUST
produce a checker error, and untagged lines must stay clean.

Checked by tests/test_typing.py (and the pyright CI step). If a tagged line
stops erroring, a type-safety guarantee regressed.
"""
from typing import Any, Dict

from growthbook import Experiment, FeatureRule, GrowthBook, Result, UserContext

gb = GrowthBook()

# Fallback-driven inference catches wrong usage of the returned value.
gb.get_feature_value("banner", "blue") + 1  # expect-error
bad_int: int = gb.get_feature_value("banner", "blue")  # expect-error

# run() result value is typed from the variations.
res = gb.run(Experiment(key="t", variations=[1, 2]))
bad_str: str = res.value  # expect-error


# Callback with wrong parameter names: crashes at runtime in the sync path.
def bad_names_cb(exp: Experiment[Any], result: Result[Any], ctx: UserContext) -> None: ...


GrowthBook(on_experiment_viewed=bad_names_cb)  # expect-error


# Callback with wrong arity.
def bad_arity_cb(experiment: Experiment[Any], result: Result[Any]) -> None: ...


GrowthBook(on_experiment_viewed=bad_arity_cb)  # expect-error


# Positional-only parameters cannot accept the keyword invocation.
def bad_pos_only_cb(
    experiment: Experiment[Any], result: Result[Any], user_context: UserContext, /
) -> None: ...


GrowthBook(on_experiment_viewed=bad_pos_only_cb)  # expect-error

# Async callbacks are only supported by the async GrowthBookClient; the sync
# client never awaits them (the coroutine would be silently dropped).
async def async_cb(
    experiment: Experiment[Any], result: Result[Any], user_context: UserContext
) -> None: ...


GrowthBook(on_experiment_viewed=async_cb)  # expect-error


# The sync client's event logger must also be synchronous: log_event neither
# awaits nor schedules a returned coroutine, so it would be silently dropped.
async def async_logger(
    event_name: str, properties: Dict[str, Any], user_context: UserContext
) -> None: ...


gb.set_event_logger(async_logger)  # expect-error

# Typo'd keyword arguments are no longer silently swallowed by checkers.
Experiment(key="t", variations=[1, 2], weigths=[0.5, 0.5])  # expect-error
FeatureRule(force="on", coverege=0.5)  # expect-error
FeatureRule(contextualVariatons=["a", "b"])  # expect-error

# Wrong argument types to public methods.
GrowthBook(attributes="not-a-dict")  # expect-error
gb.set_attributes(["not", "a", "dict"])  # expect-error

# Internal/third-party names no longer leak from the package root.
from growthbook import PoolManager  # expect-error
