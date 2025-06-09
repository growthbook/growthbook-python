#!/usr/bin/env python3
"""
Plugin Tests for GrowthBook Python SDK

Simplified and organized tests for:
- RequestContextPlugin with mocked HTTP request context
- GrowthBookTrackingPlugin with mocked ingestor
"""

import json
import time
import unittest
from unittest.mock import patch, MagicMock
from growthbook import (
    GrowthBook, 
    Experiment,
    request_context_plugin, 
    client_side_attributes,
    growthbook_tracking_plugin,
    ClientSideAttributes,
    RequestContextPlugin,
    GrowthBookTrackingPlugin
)


class MockIngestor:
    """Simple mock ingestor for capturing events."""
    
    def __init__(self):
        self.events = []
        self.requests = []
    
    def mock_urlopen(self, request, timeout=None):
        """Capture request data."""
        if hasattr(request, 'data') and request.data:
            try:
                data = json.loads(request.data.decode('utf-8'))
                self.events.extend(data.get('events', []))
                self.requests.append({
                    'url': request.full_url,
                    'client_key': data.get('client_key'),
                    'event_count': len(data.get('events', []))
                })
            except Exception:
                pass
        
        # Return success response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.__enter__ = lambda x: mock_response
        mock_response.__exit__ = lambda x, y, z, w: None
        return mock_response


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


class TestRequestContextPlugin(unittest.TestCase):
    """Test RequestContextPlugin functionality."""
    
    def test_request_context_extraction_with_mock(self):
        """Test actual request context extraction with mocked HTTP request."""
        # Create a mock request object
        mock_request = MockDjangoRequest()
        
        # Patch the plugin's request detection to return our mock
        with patch.object(RequestContextPlugin, '_get_request_object', return_value=mock_request):
            plugin = request_context_plugin(
                include_request_info=True,
                include_utm_params=True,
                include_user_agent=True
            )
            
            gb = GrowthBook(plugins=[plugin])
            
            # Verify extracted attributes
            attrs = gb.get_attributes()
            
            # Should extract UTM parameters
            self.assertEqual(attrs.get('utmSource'), 'google')
            self.assertEqual(attrs.get('utmCampaign'), 'summer2023')
            
            # Should detect mobile device from User-Agent
            self.assertEqual(attrs.get('deviceType'), 'mobile')
            self.assertEqual(attrs.get('browser'), 'safari')  # WebKit -> Safari on iOS
            
            # Should include server context
            self.assertIn('server_timestamp', attrs)
            self.assertEqual(attrs.get('sdk_context'), 'server')
            
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
        
        with patch.object(RequestContextPlugin, '_get_request_object', return_value=mock_request):
            plugin = request_context_plugin(
                client_side_attributes=client_attrs,
                include_user_agent=True
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
        
        # No mocking - plugin should detect no request context
        plugin = request_context_plugin(client_side_attributes=client_attrs)
        gb = GrowthBook(plugins=[plugin])
        
        attrs = gb.get_attributes()
        
        # Should have manual attributes
        self.assertEqual(attrs.get('pageTitle'), 'Offline Test')
        self.assertEqual(attrs.get('deviceType'), 'mobile')
        
        # Should not have request-extracted attributes
        self.assertNotIn('utmSource', attrs)
        self.assertNotIn('userAgent', attrs)
        
        gb.destroy()


class TestTrackingPlugin(unittest.TestCase):
    """Test GrowthBookTrackingPlugin functionality."""
    
    def setUp(self):
        """Set up mock ingestor for each test."""
        self.mock_ingestor = MockIngestor()
        self.patch = patch('urllib.request.urlopen', side_effect=self.mock_ingestor.mock_urlopen)
        self.patch.start()
    
    def tearDown(self):
        """Clean up patches."""
        self.patch.stop()
    
    def assert_request_data(self, expected_url, expected_event_count, expected_client_key=''):
        """Helper to assert HTTP request data."""
        self.assertGreater(len(self.mock_ingestor.requests), 0, "No HTTP requests captured")
        
        request = self.mock_ingestor.requests[0]
        self.assertEqual(request['url'], expected_url)
        self.assertEqual(request['event_count'], expected_event_count)
        if expected_client_key is not None:
            self.assertEqual(request['client_key'], expected_client_key)
    
    def assert_common_event_fields(self, event):
        """Helper to assert common event fields (timestamp, SDK info)."""
        self.assertIn('timestamp', event)
        self.assertIsInstance(event['timestamp'], (int, float))
        self.assertEqual(event['sdk_language'], 'python')
        self.assertIn('sdk_version', event)
    
    def assert_experiment_event(self, event, experiment_id, expected_variations, user_id=None):
        """Helper to assert experiment event structure."""
        self.assertEqual(event['event_type'], 'experiment_viewed')
        self.assertEqual(event['experiment_id'], experiment_id)
        self.assertIn(event['variation_value'], expected_variations)
        self.assertTrue(event['in_experiment'])
        
        if user_id:
            self.assertEqual(event['hash_attribute'], 'id')
            self.assertEqual(event['hash_value'], user_id)
        
        self.assert_common_event_fields(event)
    
    def assert_feature_event(self, event, feature_key, expected_value, expected_source='defaultValue'):
        """Helper to assert feature evaluation event structure."""
        self.assertEqual(event['event_type'], 'feature_evaluated')
        self.assertEqual(event['feature_key'], feature_key)
        self.assertEqual(event['feature_value'], expected_value)
        self.assertEqual(event['source'], expected_source)
        
        # Boolean flags
        self.assertEqual(event['on'], bool(expected_value))
        self.assertEqual(event['off'], not bool(expected_value))
        
        self.assert_common_event_fields(event)
    
    def find_events_by_type(self, event_type, additional_filter=None):
        """Helper to find events by type with optional additional filtering."""
        events = [e for e in self.mock_ingestor.events if e.get('event_type') == event_type]
        
        if additional_filter:
            events = [e for e in events if additional_filter(e)]
        
        return events
    
    def test_feature_tracking(self):
        """Test feature evaluation tracking."""
        plugin = growthbook_tracking_plugin(
            ingestor_host="https://test.growthbook.io",
            batch_size=2,
            batch_timeout=0.5
        )
        
        gb = GrowthBook(plugins=[plugin])
        
        # Set up test features
        gb.set_features({
            "test-feature": {"defaultValue": True},
            "another-feature": {"defaultValue": "test_value"}
        })
        
        # Trigger feature evaluations
        result1 = gb.is_on("test-feature")
        result2 = gb.get_feature_value("another-feature", "default")
        
        self.assertTrue(result1)
        self.assertEqual(result2, "test_value")
        
        # Wait for events and cleanup
        time.sleep(1)
        gb.destroy()
        time.sleep(0.5)
        
        # Assert HTTP request
        self.assert_request_data("https://test.growthbook.io/events", 2)
        
        # Find and assert feature events
        feature_events = self.find_events_by_type('feature_evaluated')
        self.assertEqual(len(feature_events), 2)
        
        # Assert individual feature events
        test_feature_event = next(e for e in feature_events if e['feature_key'] == 'test-feature')
        another_feature_event = next(e for e in feature_events if e['feature_key'] == 'another-feature')
        
        self.assert_feature_event(test_feature_event, 'test-feature', True)
        self.assert_feature_event(another_feature_event, 'another-feature', 'test_value')
    
    def test_tracking_plugin_captures_experiment_events(self):
        """
        Test tracking plugin to capture experiment events.
        """
        # Track experiment events with a custom callback
        tracked_events = []
        
        def custom_tracking_callback(experiment, result):
            tracked_events.append({
                'experiment_key': experiment.key,
                'variation_value': result.value,
                'in_experiment': result.inExperiment,
                'user_id': result.hashValue
            })
        
        gb = GrowthBook(
            attributes={"id": "manual-user-123"},
            plugins=[
                growthbook_tracking_plugin(
                    ingestor_host="https://test.growthbook.io",
                    batch_size=1,
                    additional_callback=custom_tracking_callback  # Custom tracking
                )
            ]
        )
        
        result = gb.run(Experiment(
            key="tracking-test", 
            variations=["a", "b"],
            coverage=1.0,  # Ensure experiment runs
            weights=[0.5, 0.5]  # Equal weights
        ))
        
        # Verify experiment ran
        self.assertTrue(result.inExperiment)
        self.assertIn(result.value, ["a", "b"])
        
        # Verify our custom callback was called
        self.assertEqual(len(tracked_events), 1)
        event = tracked_events[0]
        self.assertEqual(event['experiment_key'], 'tracking-test')
        self.assertIn(event['variation_value'], ["a", "b"])
        self.assertTrue(event['in_experiment'])
        self.assertEqual(event['user_id'], 'manual-user-123')
        
        gb.destroy()