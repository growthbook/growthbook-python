import json
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Callable, TYPE_CHECKING
from .base import GrowthBookPlugin

if TYPE_CHECKING:
    import requests
else:
    try:
        import requests  # type: ignore
    except ImportError:
        requests = None

logger = logging.getLogger("growthbook.plugins.growthbook_tracking")

# Attribute keys promoted to top-level fields in the ingestor payload.
# All remaining attributes are stored in context_json.
_TOP_LEVEL_ATTR_KEYS = {
    "user_id",
    "device_id",
    "anonymous_id",
    "id",
    "page_id",
    "session_id",
    "utmCampaign",
    "utmContent",
    "utmMedium",
    "utmSource",
    "utmTerm",
    "pageTitle",
}


def _parse_string(value) -> Optional[str]:
    return value if isinstance(value, str) else None


def _build_event_payload(
    event_name: str,
    properties: Dict[str, Any],
    attributes: Dict[str, Any],
    url: str,
    sdk_version: str,
) -> Dict[str, Any]:
    """Build an ingestor payload that matches the JS SDK EventPayload shape."""
    nested = {k: v for k, v in attributes.items() if k not in _TOP_LEVEL_ATTR_KEYS}

    payload: Dict[str, Any] = {
        "event_name": event_name,
        "properties_json": properties or {},
        "sdk_language": "python",
        "sdk_version": sdk_version,
        "url": url or "",
        "context_json": nested,
        "user_id": _parse_string(attributes.get("user_id")),
        "device_id": _parse_string(
            attributes.get("device_id")
            or attributes.get("anonymous_id")
            or attributes.get("id")
        ),
        "page_id": _parse_string(attributes.get("page_id")),
        "session_id": _parse_string(attributes.get("session_id")),
    }

    # Optional UTM / page_title fields — omit rather than null
    for src_key, dest_key in [
        ("utmCampaign", "utm_campaign"),
        ("utmContent", "utm_content"),
        ("utmMedium", "utm_medium"),
        ("utmSource", "utm_source"),
        ("utmTerm", "utm_term"),
        ("pageTitle", "page_title"),
    ]:
        value = _parse_string(attributes.get(src_key))
        if value is not None:
            payload[dest_key] = value

    return payload


class GrowthBookTrackingPlugin(GrowthBookPlugin):
    """
    GrowthBook tracking plugin for Built-in Warehouse.

    Automatically tracks "Experiment Viewed" and "Feature Evaluated" events
    and enables custom event logging via ``gb.log_event()``, all sent to
    GrowthBook's ingestor pipeline.
    """

    def __init__(
        self,
        ingestor_host: str = "https://us1.gb-ingest.com",
        track_experiment_viewed: bool = True,
        track_feature_evaluated: bool = True,
        batch_size: int = 10,
        batch_timeout: float = 10.0,
        additional_callback: Optional[Callable] = None,
        **options,
    ):
        """
        Initialize GrowthBook tracking plugin.

        Args:
            ingestor_host: Base URL of the GrowthBook ingestor (default: ``https://us1.gb-ingest.com``).
            track_experiment_viewed: Whether to auto-track experiment viewed events.
            track_feature_evaluated: Whether to auto-track feature evaluated events.
            batch_size: Number of events to accumulate before flushing to the ingestor.
            batch_timeout: Max seconds to wait before flushing a partial batch.
            additional_callback: Optional extra callback invoked on experiment viewed events.
        """
        super().__init__(**options)

        if not requests:
            raise ImportError(
                "requests library is required for GrowthBookTrackingPlugin. "
                "Install with: pip install requests"
            )

        self.ingestor_host = ingestor_host.rstrip("/")
        self.track_experiment_viewed = track_experiment_viewed
        self.track_feature_evaluated = track_feature_evaluated
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.additional_callback = additional_callback

        # Batching state
        self._event_batch: List[Dict[str, Any]] = []
        self._batch_lock = threading.Lock()
        self._flush_timer: Optional[threading.Timer] = None
        self._client_key: Optional[str] = None

    # ------------------------------------------------------------------
    # Plugin lifecycle
    # ------------------------------------------------------------------

    def initialize(self, gb_instance) -> None:
        """Initialize plugin with a GrowthBook instance."""
        try:
            self._client_key = getattr(gb_instance, "_client_key", "") or ""

            if self.track_experiment_viewed:
                self._setup_experiment_tracking(gb_instance)

            if self.track_feature_evaluated:
                self._setup_feature_tracking(gb_instance)

            # Wire log_event so custom events flow through the same pipeline
            if hasattr(gb_instance, "set_event_logger"):
                gb_instance.set_event_logger(self._handle_log_event)

            self._set_initialized(gb_instance)
            self.logger.info("Tracking enabled → %s", self.ingestor_host)

        except Exception as e:
            self.logger.error("Failed to initialize tracking plugin: %s", e)

    def cleanup(self) -> None:
        """Flush any pending events and release resources."""
        self._flush_events()
        with self._batch_lock:
            if self._flush_timer:
                self._flush_timer.cancel()
                self._flush_timer = None
        super().cleanup()

    # ------------------------------------------------------------------
    # Experiment / feature auto-tracking (legacy hooks)
    # ------------------------------------------------------------------

    def _setup_experiment_tracking(self, gb_instance) -> None:
        def tracking_wrapper(experiment, result, user_context=None):
            self._track_experiment_viewed(experiment, result)
            if self.additional_callback:
                self._safe_execute(self.additional_callback, experiment, result, user_context)

        if hasattr(gb_instance, "_trackingCallback"):
            original = gb_instance._trackingCallback

            def legacy_wrapper(experiment, result, user_context=None):
                tracking_wrapper(experiment, result, user_context)
                if original:
                    self._safe_execute(original, experiment, result, user_context)

            gb_instance._trackingCallback = legacy_wrapper

        elif hasattr(gb_instance, "options") and hasattr(
            gb_instance.options, "on_experiment_viewed"
        ):
            original = gb_instance.options.on_experiment_viewed

            def async_wrapper(experiment, result, user_context):
                tracking_wrapper(experiment, result, user_context)
                if original:
                    self._safe_execute(original, experiment, result, user_context)

            gb_instance.options.on_experiment_viewed = async_wrapper

        else:
            self.logger.warning(
                "_trackingCallback or on_experiment_viewed not found — "
                "experiment tracking may not work"
            )

    def _setup_feature_tracking(self, gb_instance) -> None:
        original_eval_feature = gb_instance.eval_feature

        def eval_feature_wrapper(key: str, *args, **kwargs):
            result = original_eval_feature(key, *args, **kwargs)
            self._track_feature_evaluated(key, result, gb_instance)
            return result

        gb_instance.eval_feature = eval_feature_wrapper

    def _track_experiment_viewed(self, experiment, result) -> None:
        try:
            attrs: Dict[str, Any] = {}
            if self._gb_instance is not None:
                attrs = getattr(self._gb_instance, "_attributes", {}) or {}

            payload = _build_event_payload(
                event_name="$$experiment_viewed",
                properties={
                    "experiment_id": experiment.key,
                    "variation_id": result.variationId,
                    "variation_key": getattr(result, "key", str(result.variationId)),
                    "in_experiment": result.inExperiment,
                    "hash_used": result.hashUsed,
                    "hash_attribute": result.hashAttribute,
                    "hash_value": result.hashValue,
                },
                attributes=attrs,
                url=getattr(self._gb_instance, "_url", "") if self._gb_instance else "",
                sdk_version=self._get_sdk_version(),
            )
            self._add_event_to_batch(payload)
        except Exception as e:
            self.logger.error("Error tracking experiment: %s", e)

    def _track_feature_evaluated(self, feature_key: str, result, gb_instance) -> None:
        try:
            attrs: Dict[str, Any] = getattr(gb_instance, "_attributes", {}) or {}
            props: Dict[str, Any] = {
                "feature_key": feature_key,
                "feature_value": result.value,
                "source": result.source,
            }
            if hasattr(result, "ruleId") and result.ruleId:
                props["rule_id"] = result.ruleId
            if hasattr(result, "experiment") and result.experiment:
                props["experiment_id"] = result.experiment.key
                if hasattr(result, "experimentResult") and result.experimentResult:
                    props["variation_id"] = result.experimentResult.variationId
                    props["in_experiment"] = result.experimentResult.inExperiment

            payload = _build_event_payload(
                event_name="$$feature_evaluated",
                properties=props,
                attributes=attrs,
                url=getattr(gb_instance, "_url", ""),
                sdk_version=self._get_sdk_version(),
            )
            self._add_event_to_batch(payload)
        except Exception as e:
            self.logger.error("Error tracking feature: %s", e)

    # ------------------------------------------------------------------
    # log_event handler (custom events)
    # ------------------------------------------------------------------

    def _handle_log_event(
        self, event_name: str, properties: Dict[str, Any], user_context
    ) -> None:
        """Called by GrowthBook.log_event / GrowthBookClient.log_event."""
        try:
            attrs: Dict[str, Any] = getattr(user_context, "attributes", {}) or {}
            url: str = getattr(user_context, "url", "") or ""
            payload = _build_event_payload(
                event_name=event_name,
                properties=properties or {},
                attributes=attrs,
                url=url,
                sdk_version=self._get_sdk_version(),
            )
            self._add_event_to_batch(payload)
        except Exception as e:
            self.logger.error("Error in event logger: %s", e)

    # ------------------------------------------------------------------
    # Batching helpers
    # ------------------------------------------------------------------

    def _add_event_to_batch(self, payload: Dict[str, Any]) -> None:
        with self._batch_lock:
            self._event_batch.append(payload)
            if len(self._event_batch) >= self.batch_size:
                self._flush_batch_locked()
            elif len(self._event_batch) == 1:
                self._start_flush_timer()

    def _start_flush_timer(self) -> None:
        if self._flush_timer:
            self._flush_timer.cancel()
        self._flush_timer = threading.Timer(self.batch_timeout, self._flush_events)
        self._flush_timer.daemon = True
        self._flush_timer.start()

    def _flush_events(self) -> None:
        with self._batch_lock:
            self._flush_batch_locked()

    def _flush_batch_locked(self) -> None:
        """Flush current batch — must be called while holding _batch_lock."""
        if not self._event_batch:
            return

        events_to_send = self._event_batch.copy()
        self._event_batch.clear()

        if self._flush_timer:
            self._flush_timer.cancel()
            self._flush_timer = None

        threading.Thread(
            target=self._send_events, args=(events_to_send,), daemon=True
        ).start()

    def _send_events(self, events: List[Dict[str, Any]]) -> None:
        """POST a batch of events to the ingestor (runs in a daemon thread)."""
        if not events or not self._client_key:
            return

        url = f"{self.ingestor_host}/track?client_key={self._client_key}"
        body = json.dumps(events)

        try:
            response = requests.post(
                url,
                data=body,
                headers={
                    "Content-Type": "text/plain",
                    "Accept": "application/json",
                    "User-Agent": f"growthbook-python-sdk/{self._get_sdk_version()}",
                },
                timeout=30,
            )
            if response.status_code == 200:
                self.logger.debug("Sent %d event(s) to ingestor", len(events))
            else:
                self.logger.warning(
                    "Ingestor returned HTTP %s for %d event(s)",
                    response.status_code,
                    len(events),
                )
        except Exception as e:
            self.logger.error("Failed to send events: %s", e)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _get_sdk_version(self) -> str:
        try:
            import growthbook

            return getattr(growthbook, "__version__", "unknown")
        except Exception:
            return "unknown"


def growthbook_tracking_plugin(**options) -> GrowthBookTrackingPlugin:
    """Factory helper — returns a configured :class:`GrowthBookTrackingPlugin`."""
    return GrowthBookTrackingPlugin(**options)
