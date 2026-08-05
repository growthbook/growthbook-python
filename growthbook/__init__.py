from .common_types import (
    AbstractStickyBucketService,
    Experiment,
    Feature,
    FeatureResult,
    FeatureRule,
    FeatureRefreshStrategy,
    Options,
    Result,
    UserContext,
)

from .growthbook import (
    AbstractFeatureCache,
    FeatureRepository,
    GrowthBook,
    InMemoryFeatureCache,
    InMemoryStickyBucketService,
    SSEClient,
    decrypt,
    feature_repo,
)

from .common_types import AbstractAsyncStickyBucketService

from .growthbook_client import (
    GrowthBookClient,
    EnhancedFeatureRepository,
    FeatureCache,
    BackoffStrategy
)

# Plugin support
from .plugins import (
    GrowthBookTrackingPlugin,
    growthbook_tracking_plugin,
    RequestContextPlugin,
    ClientSideAttributes,
    request_context_plugin,
    client_side_attributes
)

__all__ = [
    # Core clients
    "GrowthBook",
    "GrowthBookClient",
    # Configuration / context
    "Options",
    "UserContext",
    "FeatureRefreshStrategy",
    # Data model
    "Experiment",
    "Result",
    "Feature",
    "FeatureResult",
    "FeatureRule",
    # Feature loading / caching
    "FeatureRepository",
    "EnhancedFeatureRepository",
    "feature_repo",
    "AbstractFeatureCache",
    "InMemoryFeatureCache",
    "FeatureCache",
    "BackoffStrategy",
    "SSEClient",
    # Sticky bucketing
    "AbstractStickyBucketService",
    "InMemoryStickyBucketService",
    # Crypto
    "decrypt",
    # Plugins
    "GrowthBookTrackingPlugin",
    "growthbook_tracking_plugin",
    "RequestContextPlugin",
    "ClientSideAttributes",
    "request_context_plugin",
    "client_side_attributes",
    "__version__",
]

# x-release-please-start-version
__version__ = "2.4.0"
# x-release-please-end
