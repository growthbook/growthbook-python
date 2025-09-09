"""
Request Context Plugin for GrowthBook Python SDK

This plugin extracts attributes from HTTP request context using a framework-agnostic approach.
"""

import logging
import uuid
import time
from typing import Dict, Any, Optional, Callable, Union
from urllib.parse import urlparse
from .base import GrowthBookPlugin

logger = logging.getLogger("growthbook.plugins.request_context")

# Global context variable for storing current request
_current_request_context: Dict[str, Any] = {}


class ClientSideAttributes:
    """
    Client-side attributes that can't be detected server-side.
    """
    
    def __init__(self, **attributes: Any):
        """
        Initialize with any client-side attributes.
        
        Common attributes:
            pageTitle: Current page title
            deviceType: "mobile" | "desktop" | "tablet"  
            browser: "chrome" | "firefox" | "safari" | "edge"
            timezone: User's timezone (e.g., "America/New_York")
            language: User's language (e.g., "en-US")
        
        Args:
            **attributes: Any client-side attributes as key-value pairs
        """
        for key, value in attributes.items():
            if value is not None:
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items() if v is not None}


class RequestContextPlugin(GrowthBookPlugin):
    """
    Framework-agnostic request context plugin.
    
    This plugin uses:
    1. Manual request object passing via set_request_context()
    2. Context variables set by middleware
    3. Direct attribute extraction from provided data
    """
    
    def __init__(
        self,
        request_extractor: Optional[Callable] = None,
        client_side_attributes: Optional[ClientSideAttributes] = None,
        extract_utm: bool = True,
        extract_user_agent: bool = True,
        **options
    ):
        """
        Initialize request context plugin.
        
        Args:
            request_extractor: Optional function to extract request object from context
            client_side_attributes: Manual client-side attributes
            extract_utm: Whether to extract UTM parameters
            extract_user_agent: Whether to parse User-Agent header
        """
        super().__init__(**options)
        self.request_extractor = request_extractor
        self.client_side_attributes = client_side_attributes
        self.extract_utm = extract_utm
        self.extract_user_agent = extract_user_agent
        self._extracted_attributes: Dict[str, Any] = {}
        
    def initialize(self, gb_instance) -> None:
        """Initialize plugin - extract attributes from request context."""
        try:
            self._set_initialized(gb_instance)
            
            # Get request data from various sources
            request_attributes = self._extract_all_attributes()
            
            if request_attributes:
                # Check client type and merge attributes accordingly
                if hasattr(gb_instance, 'get_attributes') and hasattr(gb_instance, 'set_attributes'):
                    # Legacy GrowthBook client
                    current_attributes = gb_instance.get_attributes()
                    merged_attributes = {**request_attributes, **current_attributes}
                    gb_instance.set_attributes(merged_attributes)
                    self.logger.info(f"Extracted {len(request_attributes)} request attributes for legacy client")
                    
                elif hasattr(gb_instance, 'options'):
                    # New GrowthBookClient - store attributes for future use
                    # Note: GrowthBookClient doesn't have get/set_attributes, but we can store
                    # the extracted attributes for potential future use or logging
                    self._extracted_attributes = request_attributes
                    self.logger.info(f"Extracted {len(request_attributes)} request attributes for async client (stored for reference)")
                    
                else:
                    self.logger.warning("Unknown client type - cannot set attributes")
            else:
                self.logger.debug("No request context available")
                
        except Exception as e:
            self.logger.error(f"Failed to extract request attributes: {e}")
    
    def get_extracted_attributes(self) -> Dict[str, Any]:
        """Get the attributes extracted from request context."""
        return self._extracted_attributes.copy()
    
    def _extract_all_attributes(self) -> Dict[str, Any]:
        """Extract all available attributes from request context."""
        attributes = {}
        
        # Get request object/data
        request_data = self._get_request_data()
        
        if request_data:
            # Extract core request info
            attributes.update(self._extract_basic_info(request_data))
            
            # Extract UTM parameters
            if self.extract_utm:
                attributes.update(self._extract_utm_params(request_data))
            
            # Extract User-Agent info
            if self.extract_user_agent:
                attributes.update(self._extract_user_agent(request_data))
        
        # Add client-side attributes (these override auto-detected)
        if self.client_side_attributes:
            attributes.update(self.client_side_attributes.to_dict())
        
        # Add server context
        attributes.update({
            'server_timestamp': int(time.time()),
            'request_id': str(uuid.uuid4())[:8],
            'sdk_context': 'server'
        })
        
        return attributes
    
    def _get_request_data(self) -> Optional[Dict[str, Any]]:
        """Get request data from various sources."""
        # 1. Try custom extractor
        if self.request_extractor:
            try:
                request_obj = self.request_extractor()
                if request_obj:
                    return self._normalize_request_object(request_obj)
            except Exception as e:
                self.logger.debug(f"Custom extractor failed: {e}")
        
        # 2. Try global context
        if _current_request_context:
            return _current_request_context.copy()
        
        # 3. Try thread-local storage
        import threading
        thread_local = getattr(threading.current_thread(), 'gb_request_context', None)
        if thread_local:
            return thread_local
        
        return None
    
    def _normalize_request_object(self, request_obj) -> Dict[str, Any]:
        """Convert various request objects to normalized dict."""
        normalized = {}
        
        try:
            # Extract URL
            url = None
            if hasattr(request_obj, 'build_absolute_uri'):  # Django
                url = request_obj.build_absolute_uri()
            elif hasattr(request_obj, 'url'):  # Flask/FastAPI
                url = str(request_obj.url)
            
            if url:
                normalized['url'] = url
                parsed = urlparse(url)
                normalized['path'] = parsed.path
                normalized['host'] = parsed.netloc
                normalized['query_string'] = parsed.query
            
            # Extract query parameters
            query_params = {}
            if hasattr(request_obj, 'GET'):  # Django
                query_params = dict(request_obj.GET)
            elif hasattr(request_obj, 'args'):  # Flask
                query_params = dict(request_obj.args)
            elif hasattr(request_obj, 'query_params'):  # FastAPI
                query_params = dict(request_obj.query_params)
            
            normalized['query_params'] = query_params
            
            # Extract User-Agent
            user_agent = None
            if hasattr(request_obj, 'META'):  # Django
                user_agent = request_obj.META.get('HTTP_USER_AGENT')
            elif hasattr(request_obj, 'headers'):  # Flask/FastAPI
                user_agent = request_obj.headers.get('user-agent') or request_obj.headers.get('User-Agent')
            
            if user_agent:
                normalized['user_agent'] = user_agent
            
            # Extract user info (if available)
            if hasattr(request_obj, 'user') and hasattr(request_obj.user, 'id'):
                if getattr(request_obj.user, 'is_authenticated', True):
                    normalized['user_id'] = str(request_obj.user.id)
            
        except Exception as e:
            self.logger.debug(f"Error normalizing request object: {e}")
        
        return normalized
    
    def _extract_basic_info(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic request information."""
        info = {}
        
        if 'url' in request_data:
            info['url'] = request_data['url']
        if 'path' in request_data:
            info['path'] = request_data['path'] 
        if 'host' in request_data:
            info['host'] = request_data['host']
        if 'user_id' in request_data:
            info['id'] = request_data['user_id']
            
        return info
    
    def _extract_utm_params(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract UTM parameters from query string."""
        utm_params = {}
        query_params = request_data.get('query_params', {})
        
        utm_mappings = {
            'utm_source': 'utmSource',
            'utm_medium': 'utmMedium', 
            'utm_campaign': 'utmCampaign',
            'utm_term': 'utmTerm',
            'utm_content': 'utmContent'
        }
        
        for param, attr_name in utm_mappings.items():
            value = query_params.get(param)
            if value:
                utm_params[attr_name] = value
                
        return utm_params
    
    def _extract_user_agent(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract browser and device info from User-Agent."""
        user_agent = request_data.get('user_agent')
        if not user_agent:
            return {}
        
        ua_lower = user_agent.lower()
        info = {'userAgent': user_agent}
        
        # Simple browser detection
        if 'edge' in ua_lower or 'edg' in ua_lower:
            info['browser'] = 'edge'
        elif 'chrome' in ua_lower:
            info['browser'] = 'chrome'
        elif 'firefox' in ua_lower:
            info['browser'] = 'firefox'
        elif 'safari' in ua_lower:
            info['browser'] = 'safari'
        else:
            info['browser'] = 'unknown'
        
        # Simple device detection
        mobile_indicators = ['mobile', 'android', 'iphone', 'ipad']
        if any(indicator in ua_lower for indicator in mobile_indicators):
            info['deviceType'] = 'mobile'
        else:
            info['deviceType'] = 'desktop'
        
        return info


# Framework-agnostic helper functions
def set_request_context(request_data: Union[Dict[str, Any], Any]) -> None:
    """
    Set request context globally for the current thread.
    
    Args:
        request_data: Either a dict of request data or a request object to normalize
    """
    global _current_request_context
    
    if isinstance(request_data, dict):
        _current_request_context = request_data
    else:
        # Try to normalize request object
        plugin = RequestContextPlugin()
        _current_request_context = plugin._normalize_request_object(request_data)


def clear_request_context() -> None:
    """Clear the global request context."""
    global _current_request_context
    _current_request_context = {}


# Convenience functions
def request_context_plugin(**options) -> RequestContextPlugin:
    """
    Create a request context plugin.
    
    Usage examples:
    
    # 1. With middleware setting global context:
    set_request_context(request)
    gb = GrowthBook(plugins=[request_context_plugin()])
    
    # 2. With custom extractor:
    def get_current_request():
        return my_framework.get_current_request()
    
    gb = GrowthBook(plugins=[
        request_context_plugin(request_extractor=get_current_request)
    ])
    
    # 3. With client-side attributes:
    gb = GrowthBook(plugins=[
        request_context_plugin(
            client_side_attributes=client_side_attributes(
                pageTitle="Dashboard",
                deviceType="mobile"
            )
        )
    ])
    """
    return RequestContextPlugin(**options)


def client_side_attributes(**kwargs) -> ClientSideAttributes:
    """
    Create client-side attributes.
    
    Usage:
        attrs = client_side_attributes(
            pageTitle="My Page",
            deviceType="mobile", 
            browser="chrome",
            customField="value"
        )
    """
    return ClientSideAttributes(**kwargs)