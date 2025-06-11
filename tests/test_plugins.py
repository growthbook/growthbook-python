#!/usr/bin/env python3
"""
Plugin Tests for GrowthBook Python SDK

- RequestContextPlugin with mocked HTTP request context
- GrowthBookTrackingPlugin functionality
"""

import unittest
from unittest.mock import patch, MagicMock
from growthbook import (
    GrowthBook, 
    Experiment,
    request_context_plugin, 
    client_side_attributes,
    growthbook_tracking_plugin,
    GrowthBookTrackingPlugin
)
from growthbook.plugins.request_context import set_request_context, clear_request_context


class MockDjangoRequest:
    """Mock Django-style request object."""
    
    def __init__(self):
        self.META = {
            # Proper Safari on iOS User-Agent that contains "Safari"
            'HTTP_USER_AGENT': 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1',
            'HTTP_REFERER': 'https://google.com',
            'REMOTE_ADDR': '192.168.1.100'
        }
        self.GET = {
            'utm_source': 'google',
            'utm_campaign': 'summer2023'
        }
        self.user = MagicMock()
        self.user.id = 12345
        self.user.is_authenticated = True

    def build_absolute_uri(self):
        return 'https://example.com/dashboard?utm_source=google&utm_campaign=summer2023'


class TestRequestContextPlugin(unittest.TestCase):
    """Test RequestContextPlugin functionality."""
    
    def setUp(self):
        """Clear request context before each test."""
        clear_request_context()
    
    def tearDown(self):
        """Clear request context after each test."""
        clear_request_context()
    
    def test_request_context_extraction_with_mock(self):
        """Test actual request context extraction with mocked HTTP request."""
        # Create a mock request object
        mock_request = MockDjangoRequest()
        
        # Set request context using the middleware pattern
        set_request_context(mock_request)
        
        plugin = request_context_plugin(
            extract_utm=True,
            extract_user_agent=True
        )
        
        gb = GrowthBook(plugins=[plugin])
        
        # Verify extracted attributes
        attrs = gb.get_attributes()
        
        # Should extract UTM parameters
        self.assertEqual(attrs.get('utmSource'), 'google')
        self.assertEqual(attrs.get('utmCampaign'), 'summer2023')
        
        # Should detect mobile device from User-Agent
        self.assertEqual(attrs.get('deviceType'), 'mobile')
        self.assertEqual(attrs.get('browser'), 'safari')
        
        # Should include server context
        self.assertIn('server_timestamp', attrs)
        self.assertEqual(attrs.get('sdk_context'), 'server')
        
        # Should extract user ID
        self.assertEqual(attrs.get('id'), '12345')
        
        gb.destroy()
    
    def test_client_side_attributes_override(self):
        """Test that client-side attributes override auto-detected values."""
        mock_request = MockDjangoRequest()
        
        # Manual client-side attributes that override detection
        client_attrs = client_side_attributes(
            deviceType="desktop",  # Override mobile detection
            pageTitle="Dashboard",
            customData="test_value"
        )
        
        # Set request context using middleware pattern
        set_request_context(mock_request)
        
        plugin = request_context_plugin(
            client_side_attributes=client_attrs,
            extract_user_agent=True
        )
        
        gb = GrowthBook(plugins=[plugin])
        attrs = gb.get_attributes()
        
        # Client-side attributes should take precedence
        self.assertEqual(attrs.get('deviceType'), 'desktop')  # Override
        self.assertEqual(attrs.get('pageTitle'), 'Dashboard')  # Manual only
        self.assertEqual(attrs.get('customData'), 'test_value')
        
        # But other UA info should still be detected
        self.assertEqual(attrs.get('browser'), 'safari')
        
        gb.destroy()
    
    def test_no_request_context_fallback(self):
        """Test plugin behavior when no request context is available."""
        # Plugin should still work with manual client-side attributes
        client_attrs = client_side_attributes(
            pageTitle="Offline Test",
            deviceType="mobile"
        )
        
        # No request context set - plugin should detect no request context
        plugin = request_context_plugin(client_side_attributes=client_attrs)
        gb = GrowthBook(plugins=[plugin])
        
        attrs = gb.get_attributes()
        
        # Should have manual attributes
        self.assertEqual(attrs.get('pageTitle'), 'Offline Test')
        self.assertEqual(attrs.get('deviceType'), 'mobile')
        
        # Should not have request-extracted attributes
        self.assertNotIn('utmSource', attrs)
        self.assertNotIn('userAgent', attrs)
        
        # Should still have server context
        self.assertIn('server_timestamp', attrs)
        
        gb.destroy()

    def test_middleware_pattern_usage(self):
        """Test the recommended middleware pattern usage."""
        # Simulate middleware setting request context
        request_data = {
            'url': 'https://example.com/page?utm_source=facebook',
            'query_params': {'utm_source': 'facebook', 'utm_medium': 'social'},
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/91.0',
            'user_id': 'user-789'
        }
        
        set_request_context(request_data)
        
        gb = GrowthBook(plugins=[request_context_plugin()])
        attrs = gb.get_attributes()
        
        # Should extract data from middleware-set context
        self.assertEqual(attrs.get('utmSource'), 'facebook')
        self.assertEqual(attrs.get('utmMedium'), 'social')
        self.assertEqual(attrs.get('browser'), 'chrome')
        self.assertEqual(attrs.get('deviceType'), 'desktop')
        self.assertEqual(attrs.get('id'), 'user-789')
        
        gb.destroy()


class TestTrackingPlugin(unittest.TestCase):
    """Test GrowthBookTrackingPlugin functionality."""
    
    def test_experiment_tracking_with_callback(self):
        """Test that tracking plugin calls callback when experiments run."""
        tracked_experiments = []
        
        def track_callback(experiment, result):
            tracked_experiments.append({
                'key': experiment.key,
                'value': result.value,
                'inExperiment': result.inExperiment
            })
        
        gb = GrowthBook(
            attributes={"id": "test-user"},
            plugins=[
                growthbook_tracking_plugin(
                    ingestor_host="https://test.growthbook.io",
                    additional_callback=track_callback
                )
            ]
        )
        
        # Run experiment
        result = gb.run(Experiment(
            key="simple-test",
            variations=["control", "treatment"],
            coverage=1.0,
            weights=[0.5, 0.5]
        ))
        
        # Verify experiment worked and was tracked
        self.assertTrue(result.inExperiment)
        self.assertIn(result.value, ["control", "treatment"])
        self.assertEqual(len(tracked_experiments), 1)
        self.assertEqual(tracked_experiments[0]['key'], 'simple-test')
        
        gb.destroy()
    
    def test_feature_tracking_with_callback(self):
        """Test that tracking plugin calls callback for feature evaluations."""
        tracked_features = []
        
        # Custom tracking callback to capture feature evaluations
        original_track_feature = GrowthBookTrackingPlugin._track_feature_evaluated
        
        def mock_track_feature(self, feature_key, result, gb_instance):
            tracked_features.append({
                'key': feature_key,
                'value': result.value,
                'source': result.source
            })
            # Call original if needed
            original_track_feature(self, feature_key, result, gb_instance)
        
        with patch.object(GrowthBookTrackingPlugin, '_track_feature_evaluated', mock_track_feature):
            gb = GrowthBook(
                plugins=[growthbook_tracking_plugin(ingestor_host="https://test.growthbook.io")]
            )
            
            # Set up and evaluate features
            gb.set_features({
                "test-flag": {"defaultValue": True},
                "test-string": {"defaultValue": "hello"}
            })
            
            result1 = gb.is_on("test-flag")
            result2 = gb.get_feature_value("test-string", "default")
            
            # Verify results
            self.assertTrue(result1)
            self.assertEqual(result2, "hello")
            
            # Verify tracking was called
            self.assertEqual(len(tracked_features), 2)
            
            flag_track = next(f for f in tracked_features if f['key'] == 'test-flag')
            string_track = next(f for f in tracked_features if f['key'] == 'test-string')
            
            self.assertEqual(flag_track['value'], True)
            self.assertEqual(string_track['value'], 'hello')
            
            gb.destroy()