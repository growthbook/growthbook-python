"""Static-typing probe for the PUBLIC callback surface.

Not a pytest module (never collected: no test_ prefix). Checked by the CI
mypy step so a consumer assigning valid sync OR async callbacks typechecks.
Regression guard: async callbacks were runtime-supported but rejected by the
declared annotations.
"""
from typing import Optional

from growthbook.common_types import (
    Experiment,
    FeatureResult,
    Options,
    Result,
    UserContext,
)
from growthbook.growthbook_client import GrowthBookClient


def sync_viewed(experiment: Experiment, result: Result, user_context: Optional[UserContext]) -> None: ...
def sync_usage(key: str, result: FeatureResult, user_context: UserContext) -> None: ...
def sync_sub(experiment: Experiment, result: Result) -> None: ...


async def async_viewed(experiment: Experiment, result: Result, user_context: Optional[UserContext]) -> None: ...
async def async_usage(key: str, result: FeatureResult, user_context: UserContext) -> None: ...
async def async_sub(experiment: Experiment, result: Result) -> None: ...


sync_opts = Options(on_experiment_viewed=sync_viewed, on_feature_usage=sync_usage)
async_opts = Options(on_experiment_viewed=async_viewed, on_feature_usage=async_usage)

client = GrowthBookClient(async_opts)
client.subscribe(sync_sub)
client.subscribe(async_sub)
