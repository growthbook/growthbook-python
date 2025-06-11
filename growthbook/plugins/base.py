from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class GrowthBookPlugin(ABC):
    """
    Base class for all GrowthBook plugins.
    
    Plugins extend GrowthBook functionality by adding auto-attributes,
    tracking capabilities, or other enhancements.
    
    Lifecycle:
    1. Plugin is instantiated with configuration options
    2. initialize(gb_instance) is called when GrowthBook is created
    3. Plugin enhances GrowthBook functionality
    4. cleanup() is called when GrowthBook.destroy() is called
    """
    
    def __init__(self, **options):
        """Initialize plugin with configuration options."""
        self.options = options
        self._initialized = False
        self._gb_instance = None
        self.logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
    
    @abstractmethod
    def initialize(self, gb_instance) -> None:
        """
        Initialize the plugin with a GrowthBook instance.
        
        This method is called automatically when the GrowthBook instance
        is created. Use this to set up the plugin functionality.
        
        Args:
            gb_instance: The GrowthBook instance to enhance
        """
        pass
    
    def cleanup(self) -> None:
        """
        Cleanup plugin resources when GrowthBook instance is destroyed.
        
        Override this method if your plugin needs to:
        - Close network connections
        - Cancel timers/threads
        - Flush pending data
        - Release resources
        
        Default implementation does nothing.
        """
        self.logger.debug(f"Cleaning up plugin {self.__class__.__name__}")
        self._gb_instance = None
    
    def is_initialized(self) -> bool:
        """Check if plugin has been initialized."""
        return self._initialized
    
    def _set_initialized(self, gb_instance) -> None:
        """Mark plugin as initialized and store GrowthBook reference."""
        self._initialized = True
        self._gb_instance = gb_instance
        self.logger.debug(f"Plugin {self.__class__.__name__} initialized successfully")
    
    def _get_option(self, key: str, default: Any = None) -> Any:
        """Get a configuration option with optional default."""
        return self.options.get(key, default)
    
    def _merge_attributes(self, new_attributes: Dict[str, Any]) -> None:
        """
        Helper method to merge new attributes with existing ones.
        
        Args:
            new_attributes: Dictionary of attributes to add/update
        """
        if not self._gb_instance:
            self.logger.warning("Cannot merge attributes - plugin not initialized")
            return
        
        current_attributes = self._gb_instance.get_attributes()
        merged_attributes = {**new_attributes, **current_attributes}  # Existing attrs take precedence
        self._gb_instance.set_attributes(merged_attributes)
        
        self.logger.debug(f"Merged {len(new_attributes)} attributes: {list(new_attributes.keys())}")
    
    def _safe_execute(self, func, *args, **kwargs):
        """
        Safely execute a function, logging any exceptions.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result or None if exception occurred
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {func.__name__}: {e}")
            return None 