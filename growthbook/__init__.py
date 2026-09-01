from .common_types import (
    AbstractStickyBucketService,
    AsyncEventLogger,
    AsyncFeatureUsageCallback,
    AsyncTrackingCallback,
    EventLogger,
    Experiment,
    Feature,
    FeatureResult,
    FeatureRule,
    FeatureRefreshStrategy,
    FeatureUsageCallback,
    JSONValue,
    Options,
    Result,
    TrackingCallback,
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

from .common_types import (
    AbstractAsyncStickyBucketService,
    CBContext,
    ContextualBanditContext,
    ContextualBanditDefinition,
)

from .growthbook_client import (
    GrowthBookClient,
    EnhancedFeatureRepository,
    FeatureCache,
    BackoffStrategy
)

# Plugin support
from .plugins import (
    GrowthBookPlugin,
    PluginLike,
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
    "CBContext",
    "ContextualBanditContext",
    "ContextualBanditDefinition",
    # Typing helpers
    "JSONValue",
    "TrackingCallback",
    "FeatureUsageCallback",
    "EventLogger",
    "AsyncTrackingCallback",
    "AsyncFeatureUsageCallback",
    "AsyncEventLogger",
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
    "AbstractAsyncStickyBucketService",
    "InMemoryStickyBucketService",
    # Crypto
    "decrypt",
    # Plugins
    "GrowthBookPlugin",
    "PluginLike",
    "GrowthBookTrackingPlugin",
    "growthbook_tracking_plugin",
    "RequestContextPlugin",
    "ClientSideAttributes",
    "request_context_plugin",
    "client_side_attributes",
    "__version__",
]

# x-release-please-start-version
__version__ = "3.0.0"
# x-release-please-end
