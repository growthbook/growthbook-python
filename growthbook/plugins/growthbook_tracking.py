import json
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Callable, TYPE_CHECKING
from .base import GrowthBookPlugin

if TYPE_CHECKING:
    import requests  # type: ignore
else:
    try:
        import requests  # type: ignore
    except ImportError:
        requests = None

logger = logging.getLogger("growthbook.plugins.growthbook_tracking")


class GrowthBookTrackingPlugin(GrowthBookPlugin):
    """
    GrowthBook tracking plugin for Built-in Warehouse.
    
    This plugin automatically tracks "Experiment Viewed" and "Feature Evaluated" 
    events to GrowthBook's built-in data warehouse.
    """
    
    def __init__(
        self,
        ingestor_host: str,
        track_experiment_viewed: bool = True,
        track_feature_evaluated: bool = True,
        batch_size: int = 10,
        batch_timeout: float = 10.0,
        additional_callback: Optional[Callable] = None,
        **options
    ):
        """
        Initialize GrowthBook tracking plugin.
        
        Args:
            ingestor_host: The GrowthBook ingestor endpoint
            track_experiment_viewed: Whether to track experiment viewed events
            track_feature_evaluated: Whether to track feature evaluated events
            batch_size: Number of events to batch before sending
            batch_timeout: Maximum time (seconds) to wait before sending a batch
            additional_callback: Optional additional tracking callback
        """
        super().__init__(**options)
        
        if not requests:
            raise ImportError("requests library is required for GrowthBookTrackingPlugin. Install with: pip install requests")
        
        self.ingestor_host = ingestor_host.rstrip('/')
        self.track_experiment_viewed = track_experiment_viewed
        self.track_feature_evaluated = track_feature_evaluated
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.additional_callback = additional_callback
        
        # batching
        self._event_batch: List[Dict[str, Any]] = []
        self._batch_lock = threading.Lock()
        self._flush_timer: Optional[threading.Timer] = None
        self._client_key: Optional[str] = None
        
    def initialize(self, gb_instance) -> None:
        """Initialize plugin with GrowthBook instance."""
        try:
            self._client_key = getattr(gb_instance, '_client_key', '')
            
            # Hook into experiment tracking
            if self.track_experiment_viewed:
                self._setup_experiment_tracking(gb_instance)
            
            # Hook into feature evaluation
            if self.track_feature_evaluated:
                self._setup_feature_tracking(gb_instance)
            
            self._set_initialized(gb_instance)
            self.logger.info(f"Tracking enabled for {self.ingestor_host}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tracking plugin: {e}")
    
    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self._flush_events()
        if self._flush_timer:
            self._flush_timer.cancel()
        super().cleanup()
    
    def _setup_experiment_tracking(self, gb_instance) -> None:
        """Setup experiment tracking for both legacy and async clients."""
        
        def tracking_wrapper(experiment, result, user_context=None):
            # Track to ingestor
            self._track_experiment_viewed(experiment, result)
            
            # Call additional callback
            if self.additional_callback:
                self._safe_execute(self.additional_callback, experiment, result, user_context)
        
        # Check if it's the legacy GrowthBook client (has _trackingCallback)
        if hasattr(gb_instance, '_trackingCallback'):
            # Legacy GrowthBook client
            original_callback = getattr(gb_instance, '_trackingCallback', None)
            
            def legacy_wrapper(experiment, result, user_context=None):
                tracking_wrapper(experiment, result, user_context)
                # Call original callback
                if original_callback:
                    self._safe_execute(original_callback, experiment, result, user_context)
            
            gb_instance._trackingCallback = legacy_wrapper
            
        elif hasattr(gb_instance, 'options') and hasattr(gb_instance.options, 'on_experiment_viewed'):
            # New GrowthBookClient (async)
            original_callback = gb_instance.options.on_experiment_viewed
            
            def async_wrapper(experiment, result, user_context):
                tracking_wrapper(experiment, result, user_context)
                # Call original callback
                if original_callback:
                    self._safe_execute(original_callback, experiment, result, user_context)
            
            gb_instance.options.on_experiment_viewed = async_wrapper
        
        else:
            self.logger.warning("_trackingCallback or on_experiment_viewed properties not found - tracking may not work properly")
    
    def _setup_feature_tracking(self, gb_instance):
        """Setup feature evaluation tracking."""
        original_eval_feature = gb_instance.eval_feature
        
        def eval_feature_wrapper(key: str):
            result = original_eval_feature(key)
            self._track_feature_evaluated(key, result, gb_instance)
            return result
        
        gb_instance.eval_feature = eval_feature_wrapper
    
    def _track_experiment_viewed(self, experiment, result) -> None:
        """Track experiment viewed event."""
        try:
            # Build event data with all metadata
            event_data = {
                'event_type': 'experiment_viewed',
                'timestamp': int(time.time() * 1000),
                'client_key': self._client_key,
                'sdk_language': 'python',
                'sdk_version': self._get_sdk_version(),
                # Core experiment data
                'experiment_id': experiment.key,
                'variation_id': result.variationId,
                'variation_key': getattr(result, 'key', str(result.variationId)),
                'variation_value': result.value,
                'in_experiment': result.inExperiment,
                'hash_used': result.hashUsed,
                'hash_attribute': result.hashAttribute,
                'hash_value': result.hashValue,
            }
            
            # Add optional metadata if available
            if hasattr(experiment, 'name') and experiment.name:
                event_data['experiment_name'] = experiment.name
            if hasattr(result, 'featureId') and result.featureId:
                event_data['feature_id'] = result.featureId
            
            self._add_event_to_batch(event_data)
            
        except Exception as e:
            self.logger.error(f"Error tracking experiment: {e}")
    
    def _track_feature_evaluated(self, feature_key: str, result, gb_instance) -> None:
        """Track feature evaluated event."""
        try:
            # Build event data with all metadata
            event_data = {
                'event_type': 'feature_evaluated',
                'timestamp': int(time.time() * 1000),
                'client_key': self._client_key,
                'sdk_language': 'python',
                'sdk_version': self._get_sdk_version(),
                # Core feature data
                'feature_key': feature_key,
                'feature_value': result.value,
                'source': result.source,
                'on': getattr(result, 'on', bool(result.value)),
                'off': getattr(result, 'off', not bool(result.value)),
            }
            
            # Add optional metadata if available
            if hasattr(result, 'ruleId') and result.ruleId:
                event_data['rule_id'] = result.ruleId
            
            # Add experiment info if feature came from experiment
            if hasattr(result, 'experiment') and result.experiment:
                event_data['experiment_id'] = result.experiment.key
                if hasattr(result, 'experimentResult') and result.experimentResult:
                    event_data['variation_id'] = result.experimentResult.variationId
                    event_data['in_experiment'] = result.experimentResult.inExperiment
            
            self._add_event_to_batch(event_data)
            
        except Exception as e:
            self.logger.error(f"Error tracking feature: {e}")
    
    def _add_event_to_batch(self, event_data: Dict[str, Any]) -> None:
        with self._batch_lock:
            self._event_batch.append(event_data)
            
            # Flush if batch is full
            if len(self._event_batch) >= self.batch_size:
                self._flush_batch_locked()
            elif len(self._event_batch) == 1:
                # Start timer for first event
                self._start_flush_timer()
    
    def _start_flush_timer(self) -> None:
        """Start flush timer."""
        if self._flush_timer:
            self._flush_timer.cancel()
        
        self._flush_timer = threading.Timer(self.batch_timeout, self._flush_events)
        self._flush_timer.start()
    
    def _flush_events(self) -> None:
        """Flush events with lock."""
        with self._batch_lock:
            self._flush_batch_locked()
    
    def _flush_batch_locked(self) -> None:
        """Flush current batch (called while holding lock)."""
        if not self._event_batch:
            return
        
        events_to_send = self._event_batch.copy()
        self._event_batch.clear()
        
        if self._flush_timer:
            self._flush_timer.cancel()
            self._flush_timer = None
        
        # Send in background thread
        threading.Thread(target=self._send_events, args=(events_to_send,), daemon=True).start()
    
    def _send_events(self, events: List[Dict[str, Any]]) -> None:
        """Send events using requests library."""
        if not events:
            return
        
        try:
            payload = {
                'events': events,
                'client_key': self._client_key
            }
            
            url = f"{self.ingestor_host}/events"
            response = requests.post(
                url,
                json=payload,
                headers={'User-Agent': f'growthbook-python-sdk/{self._get_sdk_version()}'},
                timeout=30
            )
            
            if response.status_code == 200:
                self.logger.debug(f"Successfully sent {len(events)} events")
            else:
                self.logger.warning(f"Ingestor returned status {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Failed to send events: {e}")
    
    def _get_sdk_version(self) -> str:
        """Get SDK version."""
        try:
            import growthbook
            return getattr(growthbook, '__version__', 'unknown')
        except:
            return 'unknown'


def growthbook_tracking_plugin(**options) -> GrowthBookTrackingPlugin:
    """Create a GrowthBook tracking plugin."""
    return GrowthBookTrackingPlugin(**options) 