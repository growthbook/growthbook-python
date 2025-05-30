"""
Request Context Plugin for GrowthBook Python SDK

This plugin extracts attributes from HTTP request context that's actually 
available server-side, rather than trying to mimic browser-side behavior.
"""

import os
import threading
import logging
import uuid
import time
from typing import Dict, Any, Optional, Callable
from urllib.parse import urlparse, parse_qs
from .base import GrowthBookPlugin

# Optional web framework imports (type checking only)
try:
    from flask import session, g, request, has_request_context
except ImportError:
    session = None  # type: ignore
    g = None  # type: ignore
    request = None  # type: ignore
    has_request_context = None  # type: ignore

logger = logging.getLogger("growthbook.plugins.request_context")


class ClientSideAttributes:
    """
    Structure for manual client-side attributes that can't be detected server-side.
    
    This allows users to provide client-side data that the server can't access,
    such as page titles, device info, or override auto-detected values.
    """
    
    def __init__(
        self,
        pageTitle: Optional[str] = None,
        deviceType: Optional[str] = None,  # "mobile" | "desktop" | "tablet"
        browser: Optional[str] = None,      # "chrome" | "firefox" | "safari" | "edge" | "unknown"
        browserVersion: Optional[str] = None,
        os: Optional[str] = None,           # "windows" | "macos" | "linux" | "ios" | "android"
        timezone: Optional[str] = None,     # "America/New_York"
        language: Optional[str] = None,     # "en-US"
        cookiesEnabled: Optional[bool] = None,
        **custom_attributes: Any
    ):
        """
        Initialize client-side attributes with optional data from browser/app.
        
        Args:
            pageTitle: Current page title from document.title
            deviceType: Device type override ("mobile", "desktop", "tablet")
            browser: Browser name override
            browserVersion: Browser version (e.g., "91.0.4472.124")
            os: Operating system ("windows", "macos", "linux", "ios", "android")
            timezone: User's timezone (e.g., "America/New_York")
            language: User's language preference (e.g., "en-US")
            cookiesEnabled: Whether cookies are enabled
            **custom_attributes: Any additional custom attributes
        """
        self.pageTitle = pageTitle
        self.deviceType = deviceType
        self.browser = browser
        self.browserVersion = browserVersion
        self.os = os
        self.timezone = timezone
        self.language = language
        self.cookiesEnabled = cookiesEnabled
        
        # Store custom attributes
        for key, value in custom_attributes.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result


class RequestContextPlugin(GrowthBookPlugin):
    """
    Request context plugin that extracts available server-side attributes.
    
    Unlike browser-side auto-attributes, this focuses on HTTP request data
    that's actually accessible on the server:
    - URL components from web framework request
    - User-Agent parsing for device/browser detection
    - UTM parameters from query string
    - Server-side session/user management
    - Custom request headers
    - Optional manual client-side attributes for browser/app data
    
    Requires web framework integration (middleware) to work properly.
    """
    
    def __init__(
        self, 
        id_attribute: str = "id",
        user_id_extractor: Optional[Callable] = None,
        session_id_extractor: Optional[Callable] = None,
        include_request_info: bool = True,
        include_utm_params: bool = True,
        include_user_agent: bool = True,
        custom_extractors: Optional[Dict[str, Callable]] = None,
        client_side_attributes: Optional[ClientSideAttributes] = None,
        **options
    ):
        """
        Initialize request context plugin.
        
        Args:
            id_attribute: Name for the user ID attribute
            user_id_extractor: Function to extract user ID from request context
            session_id_extractor: Function to extract session ID  
            include_request_info: Include URL, path, host, etc.
            include_utm_params: Include UTM parameters from query string
            include_user_agent: Parse User-Agent for browser/device info
            custom_extractors: Dict of attribute_name -> extractor_function
            client_side_attributes: Manual client-side attributes (pageTitle, device info, etc.)
        """
        super().__init__(**options)
        self.id_attribute = id_attribute
        self.user_id_extractor = user_id_extractor
        self.session_id_extractor = session_id_extractor
        self.include_request_info = include_request_info
        self.include_utm_params = include_utm_params
        self.include_user_agent = include_user_agent
        self.custom_extractors = custom_extractors or {}
        self.client_side_attributes = client_side_attributes
        
    def initialize(self, gb_instance) -> None:
        """Initialize plugin - extract attributes from current request context."""
        try:
            self._set_initialized(gb_instance)
            
            # Extract what's available from request context
            request_attributes = self._extract_request_attributes()
            
            # Add manual client-side attributes
            if self.client_side_attributes:
                manual_attributes = self.client_side_attributes.to_dict()
                request_attributes.update(manual_attributes)
                self.logger.debug(f"Added {len(manual_attributes)} manual client-side attributes")
            
            if request_attributes:
                # Merge with existing attributes (existing take precedence)
                current_attributes = gb_instance.get_attributes()
                merged_attributes = {**request_attributes, **current_attributes}
                gb_instance.set_attributes(merged_attributes)
                
                self.logger.info(f"Extracted {len(request_attributes)} total attributes: {list(request_attributes.keys())}")
            else:
                self.logger.debug("No request context available - plugin will be inactive")
                
        except Exception as e:
            self.logger.error(f"Failed to extract request attributes: {e}")
    
    def _extract_request_attributes(self) -> Dict[str, Any]:
        """Extract attributes from current HTTP request context."""
        attributes: Dict[str, Any] = {}
        
        # Get request object from web framework
        request = self._get_request_object()
        if not request:
            self.logger.debug("No request context available")
            return attributes
        
        # User identification
        user_id = self._extract_user_id(request)
        if user_id:
            attributes[self.id_attribute] = user_id
        
        # Session identification
        session_id = self._extract_session_id(request)
        if session_id:
            attributes['session_id'] = session_id
        
        # Request information  
        if self.include_request_info:
            attributes.update(self._extract_request_info(request))
        
        # UTM parameters
        if self.include_utm_params:
            attributes.update(self._extract_utm_params(request))
        
        # User-Agent parsing (can be overridden by client_side_attributes)
        if self.include_user_agent:
            ua_attributes = self._extract_user_agent_info(request)
            # Only add if not provided in client_side_attributes
            for key, value in ua_attributes.items():
                if not self.client_side_attributes or not hasattr(self.client_side_attributes, key) or getattr(self.client_side_attributes, key) is None:
                    attributes[key] = value
        
        # Custom extractors
        for attr_name, extractor in self.custom_extractors.items():
            try:
                value = extractor(request)
                if value is not None:
                    attributes[attr_name] = value
            except Exception as e:
                self.logger.debug(f"Custom extractor {attr_name} failed: {e}")
        
        # Server-side context
        attributes.update(self._extract_server_context())
        
        return attributes
    
    def _extract_user_id(self, request) -> Optional[str]:
        """Extract user ID using custom extractor or defaults."""
        try:
            if self.user_id_extractor:
                return self.user_id_extractor(request)
            
            # Default: try common patterns
            # Django: request.user.id if authenticated
            if hasattr(request, 'user') and hasattr(request.user, 'id'):
                if request.user.is_authenticated:
                    return str(request.user.id)
            
            # Flask: session or g.user
            if hasattr(request, 'environ'):  # Flask
                try:
                    if hasattr(g, 'user') and g.user:
                        return str(getattr(g.user, 'id', g.user))
                    return session.get('user_id')
                except:
                    pass
            
            # Fallback: generate temporary ID
            return str(uuid.uuid4())
            
        except Exception as e:
            self.logger.debug(f"Could not extract user ID: {e}")
            return str(uuid.uuid4())
    
    def _extract_session_id(self, request) -> Optional[str]:
        """Extract session ID using custom extractor or defaults."""
        try:
            if self.session_id_extractor:
                return self.session_id_extractor(request)
            
            # Django
            if hasattr(request, 'session') and hasattr(request.session, 'session_key'):
                return request.session.session_key
            
            # Flask
            try:
                return session.get('_id') or session.get('session_id')
            except:
                pass
            
            return None
            
        except Exception as e:
            self.logger.debug(f"Could not extract session ID: {e}")
            return None
    
    def _extract_request_info(self, request) -> Dict[str, Any]:
        """Extract URL and request information."""
        info = {}
        
        try:
            # URL components
            url = self._get_full_url(request)
            if url:
                parsed = urlparse(url)
                info['url'] = url
                info['path'] = parsed.path
                info['host'] = parsed.netloc
                if parsed.query:
                    info['query'] = parsed.query
            
            # HTTP method
            method = getattr(request, 'method', None)
            if method:
                info['http_method'] = method
            
            # Remote IP
            ip = self._get_client_ip(request)
            if ip:
                info['client_ip'] = ip
            
            # Referrer
            referrer = self._get_referrer(request)
            if referrer:
                info['referrer'] = referrer
                
        except Exception as e:
            self.logger.debug(f"Could not extract request info: {e}")
        
        return info
    
    def _extract_utm_params(self, request) -> Dict[str, Any]:
        """Extract UTM parameters from query string."""
        utm_params = {}
        
        try:
            query_params = self._get_query_params(request)
            
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
                    
        except Exception as e:
            self.logger.debug(f"Could not extract UTM params: {e}")
        
        return utm_params
    
    def _extract_user_agent_info(self, request) -> Dict[str, Any]:
        """Extract browser and device info from User-Agent."""
        info: Dict[str, Any] = {}
        
        try:
            user_agent = self._get_user_agent(request)
            if not user_agent:
                return info
            
            ua_lower = user_agent.lower()
            
            # Browser detection
            if 'edg' in ua_lower:
                info['browser'] = 'edge'
            elif 'chrome' in ua_lower and 'edg' not in ua_lower:
                info['browser'] = 'chrome'
            elif 'firefox' in ua_lower:
                info['browser'] = 'firefox'
            elif 'safari' in ua_lower and 'chrome' not in ua_lower:
                info['browser'] = 'safari'
            else:
                info['browser'] = 'unknown'
            
            # Device type
            mobile_indicators = ['mobile', 'android', 'iphone', 'ipad', 'ipod', 'blackberry', 'windows phone']
            if any(indicator in ua_lower for indicator in mobile_indicators):
                info['deviceType'] = 'mobile'
            else:
                info['deviceType'] = 'desktop'
            
            # Store full user agent if needed
            info['userAgent'] = user_agent
            
        except Exception as e:
            self.logger.debug(f"Could not extract user agent info: {e}")
        
        return info
    
    def _extract_server_context(self) -> Dict[str, Any]:
        """Extract server-side context information."""
        return {
            'server_timestamp': int(time.time()),
            'request_id': str(uuid.uuid4())[:8],
            'sdk_context': 'server'
        }
    
    def _get_request_object(self):
        """Get request object from various web frameworks."""
        # Try Django
        try:
            if hasattr(threading.current_thread(), 'request'):
                return threading.current_thread().request
        except:
            pass
        
        # Try Flask
        try:
            if has_request_context():
                return request
        except:
            pass
        
        # Try FastAPI (contextvars)
        try:
            import contextvars
            request_var = contextvars.ContextVar('request', default=None)
            return request_var.get()
        except:
            pass
        
        return None
    
    def _get_full_url(self, request) -> Optional[str]:
        """Extract full URL from request."""
        try:
            # Django
            if hasattr(request, 'build_absolute_uri'):
                return request.build_absolute_uri()
            
            # Flask
            if hasattr(request, 'url'):
                return request.url
            
            # FastAPI
            if hasattr(request, 'url'):
                return str(request.url)
                
        except Exception:
            pass
        
        return None
    
    def _get_query_params(self, request) -> Dict[str, str]:
        """Extract query parameters."""
        try:
            # Django
            if hasattr(request, 'GET'):
                return dict(request.GET)
            
            # Flask
            if hasattr(request, 'args'):
                return dict(request.args)
            
            # FastAPI
            if hasattr(request, 'query_params'):
                return dict(request.query_params)
                
        except Exception:
            pass
        
        return {}
    
    def _get_user_agent(self, request) -> Optional[str]:
        """Extract User-Agent header."""
        try:
            # Django
            if hasattr(request, 'META'):
                return request.META.get('HTTP_USER_AGENT')
            
            # Flask
            if hasattr(request, 'user_agent'):
                return str(request.user_agent)
            
            # FastAPI
            if hasattr(request, 'headers'):
                return request.headers.get('user-agent')
                
        except Exception:
            pass
        
        return None
    
    def _get_client_ip(self, request) -> Optional[str]:
        """Extract client IP address."""
        try:
            # Django
            if hasattr(request, 'META'):
                x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
                if x_forwarded_for:
                    return x_forwarded_for.split(',')[0]
                return request.META.get('REMOTE_ADDR')
            
            # Flask/FastAPI
            if hasattr(request, 'remote_addr'):
                return request.remote_addr
                
        except Exception:
            pass
        
        return None
    
    def _get_referrer(self, request) -> Optional[str]:
        """Extract HTTP Referrer."""
        try:
            # Django
            if hasattr(request, 'META'):
                return request.META.get('HTTP_REFERER')
            
            # Flask
            if hasattr(request, 'referrer'):
                return request.referrer
            
            # FastAPI
            if hasattr(request, 'headers'):
                return request.headers.get('referer')
                
        except Exception:
            pass
        
        return None


# Convenience functions
def request_context_plugin(**options) -> RequestContextPlugin:
    """
    Create a request context plugin for server-side attribute extraction.
    
    Usage:
        gb = GrowthBook(
            plugins=[
                request_context_plugin(
                    user_id_extractor=lambda req: req.user.id if req.user.is_authenticated else None,
                    client_side_attributes=client_side_attributes(
                        pageTitle="Dashboard",
                        deviceType="desktop"
                    )
                )
            ]
        )
    """
    return RequestContextPlugin(**options)


def client_side_attributes(**kwargs) -> ClientSideAttributes:
    """
    Create client-side attributes for manual browser/app data.
    
    Usage:
        attrs = client_side_attributes(
            pageTitle="My Dashboard",
            deviceType="mobile",
            timezone="America/New_York",
            customAttribute="value"
        )
    """
    return ClientSideAttributes(**kwargs) 