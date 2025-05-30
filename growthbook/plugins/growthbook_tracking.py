import json
import logging
import threading
import time
from typing import Dict, Any, Optional, List, Callable
import urllib.request
import urllib.parse
from .base import GrowthBookPlugin

logger = logging.getLogger("growthbook.plugins.growthbook_tracking")


class GrowthBookTrackingPlugin(GrowthBookPlugin):
    """
    GrowthBook tracking plugin for Built-in Warehouse.
    
    This plugin automatically tracks "Experiment Viewed" and "Feature Evaluated" 
    events to GrowthBook's built-in data warehouse for organizations that use 
    this feature.
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
            ingestor_host: The GrowthBook ingestor endpoint (e.g., "https://us1.gb-ingest.com")
            track_experiment_viewed: Whether to track experiment viewed events
            track_feature_evaluated: Whether to track feature evaluated events
            batch_size: Number of events to batch before sending
            batch_timeout: Maximum time (seconds) to wait before sending a batch
            additional_callback: Optional additional tracking callback
        """
        super().__init__(**options)
        self.ingestor_host = ingestor_host.rstrip('/')
        self.track_experiment_viewed = track_experiment_viewed
        self.track_feature_evaluated = track_feature_evaluated
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.additional_callback = additional_callback
        
        # Event batching
        self._event_batch: List[Dict[str, Any]] = []
        self._batch_lock = threading.Lock()
        self._last_batch_time = time.time()
        self._flush_timer: Optional[threading.Timer] = None
        
        # Client key for authentication
        self._client_key: Optional[str] = None
        
    def initialize(self, gb_instance) -> None:
        """Initialize plugin with GrowthBook instance."""
        try:
            # Store client key for authentication
            self._client_key = getattr(gb_instance, '_client_key', None)
            
            if not self._client_key:
                self.logger.warning("No client_key found. Events will not be authenticated.")
            
            # Hook into experiment tracking if enabled
            if self.track_experiment_viewed:
                self._setup_experiment_tracking(gb_instance)
            
            # Hook into feature evaluation if enabled  
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
        """Setup experiment tracking by wrapping the tracking callback."""
        original_callback = getattr(gb_instance, '_trackingCallback', None)
        
        def enhanced_tracking_wrapper(experiment, result):
            # Track to GrowthBook ingestor
            self._track_experiment_viewed(experiment, result)
            
            # Call additional callback if provided
            if self.additional_callback:
                self._safe_execute(self.additional_callback, experiment, result)
            
            # Call original callback if it exists
            if original_callback:
                self._safe_execute(original_callback, experiment, result)
        
        gb_instance._trackingCallback = enhanced_tracking_wrapper
    
    def _setup_feature_tracking(self, gb_instance):
        """Setup feature evaluation tracking by wrapping eval_feature."""
        original_eval_feature = gb_instance.eval_feature
        
        def eval_feature_wrapper(key: str):
            result = original_eval_feature(key)
            
            # Track feature evaluation
            self._track_feature_evaluated(key, result, gb_instance)
            
            return result
        
        # Replace the method
        gb_instance.eval_feature = eval_feature_wrapper
    
    def _track_experiment_viewed(self, experiment, result) -> None:
        """Track experiment viewed event."""
        try:
            event_data = {
                'event_type': 'experiment_viewed',
                'timestamp': int(time.time() * 1000),  # milliseconds
                'experiment_id': experiment.key,
                'variation_id': result.variationId,
                'variation_key': getattr(result, 'key', str(result.variationId)),
                'variation_value': result.value,
                'in_experiment': result.inExperiment,
                'hash_used': result.hashUsed,
                'hash_attribute': result.hashAttribute,
                'hash_value': result.hashValue,
            }
            
            # Add experiment metadata if available
            if hasattr(experiment, 'name') and experiment.name:
                event_data['experiment_name'] = experiment.name
            
            if hasattr(result, 'featureId') and result.featureId:
                event_data['feature_id'] = result.featureId
            
            self._add_event_to_batch(event_data)
            
        except Exception as e:
            self.logger.error(f"Error tracking experiment viewed event: {e}")
    
    def _track_feature_evaluated(self, feature_key: str, result, gb_instance) -> None:
        """Track feature evaluated event."""
        try:
            event_data = {
                'event_type': 'feature_evaluated',
                'timestamp': int(time.time() * 1000),  # milliseconds
                'feature_key': feature_key,
                'feature_value': result.value,
                'source': result.source,
                'on': result.on if hasattr(result, 'on') else bool(result.value),
                'off': result.off if hasattr(result, 'off') else not bool(result.value),
            }
            
            # Add rule information if available
            if hasattr(result, 'ruleId') and result.ruleId:
                event_data['rule_id'] = result.ruleId
            
            # Add experiment information if the feature evaluation came from an experiment
            if hasattr(result, 'experiment') and result.experiment:
                event_data['experiment_id'] = result.experiment.key
                if hasattr(result, 'experimentResult') and result.experimentResult:
                    event_data['variation_id'] = result.experimentResult.variationId
                    event_data['in_experiment'] = result.experimentResult.inExperiment
            
            self._add_event_to_batch(event_data)
            
        except Exception as e:
            self.logger.error(f"Error tracking feature evaluated event: {e}")
    
    def _add_event_to_batch(self, event_data: Dict[str, Any]) -> None:
        """Add event to batch and handle flushing."""
        with self._batch_lock:
            # Add common attributes
            event_data.update(self._get_common_event_data())
            
            self._event_batch.append(event_data)
            
            # Check if we should flush the batch
            should_flush = len(self._event_batch) >= self.batch_size
            
            if should_flush:
                self._flush_events_locked()
            elif len(self._event_batch) == 1:
                # Start timer for first event in batch
                self._schedule_flush()
    
    def _get_common_event_data(self) -> Dict[str, Any]:
        """Get common data to add to all events."""
        return {
            'client_key': self._client_key,
            'sdk_version': self._get_sdk_version(),
            'sdk_language': 'python',
        }
    
    def _get_sdk_version(self) -> str:
        """Get SDK version."""
        try:
            # Try to get from growthbook package
            import growthbook
            return getattr(growthbook, '__version__', 'unknown')
        except:
            return 'unknown'
    
    def _schedule_flush(self) -> None:
        """Schedule a flush after timeout."""
        if self._flush_timer:
            self._flush_timer.cancel()
        
        self._flush_timer = threading.Timer(self.batch_timeout, self._flush_events)
        self._flush_timer.start()
    
    def _flush_events(self) -> None:
        """Flush events to the ingestor."""
        with self._batch_lock:
            self._flush_events_locked()
    
    def _flush_events_locked(self) -> None:
        """Flush events (called while holding lock)."""
        if not self._event_batch:
            return
        
        events_to_send = self._event_batch.copy()
        self._event_batch.clear()
        
        if self._flush_timer:
            self._flush_timer.cancel()
            self._flush_timer = None
        
        # Send events in background thread
        threading.Thread(
            target=self._send_events_to_ingestor,
            args=(events_to_send,),
            daemon=True
        ).start()
    
    def _send_events_to_ingestor(self, events: List[Dict[str, Any]]) -> None:
        """Send events to GrowthBook ingestor."""
        if not events:
            return
        
        try:
            # Prepare request
            url = f"{self.ingestor_host}/events"
            data = {
                'events': events,
                'client_key': self._client_key
            }
            
            # Convert to JSON
            json_data = json.dumps(data).encode('utf-8')
            
            # Create request
            request = urllib.request.Request(
                url,
                data=json_data,
                headers={
                    'Content-Type': 'application/json',
                    'User-Agent': f'GrowthBook-Python-SDK/{self._get_sdk_version()}'
                }
            )
            
            # Send request
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status == 200:
                    self.logger.debug(f"Successfully sent {len(events)} events to ingestor")
                else:
                    self.logger.warning(f"Ingestor returned status {response.status}")
                    
        except Exception as e:
            self.logger.error(f"Failed to send events to ingestor: {e}")
            # In a production implementation, you might want to retry or queue events


# Convenience function for easy usage
def growthbook_tracking_plugin(**options) -> GrowthBookTrackingPlugin:
    """
    Create a GrowthBook tracking plugin with options.
    
    Usage:
        gb = GrowthBook(
            plugins=[
                growthbook_tracking_plugin(
                    ingestor_host="https://us1.gb-ingest.com"
                )
            ]
        )
    """
    return GrowthBookTrackingPlugin(**options) 