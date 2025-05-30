#!/usr/bin/env python3
"""
Plugin Tests for GrowthBook Python SDK

Simplified and organized tests for:
- ClientSideAttributes functionality
- RequestContextPlugin 
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


class TestClientSideAttributes(unittest.TestCase):
    """Test ClientSideAttributes functionality."""
    
    def test_basic_attributes(self):
        """Test basic client-side attribute creation."""
        attrs = client_side_attributes(
            pageTitle="Test Page",
            deviceType="mobile",
            browser="chrome",
            timezone="America/New_York",
            language="en-US"
        )
        
        self.assertIsInstance(attrs, ClientSideAttributes)
        
        attrs_dict = attrs.to_dict()
        self.assertEqual(attrs_dict['pageTitle'], "Test Page")
        self.assertEqual(attrs_dict['deviceType'], "mobile")
        self.assertEqual(attrs_dict['browser'], "chrome")
        self.assertEqual(len(attrs_dict), 5)
    
    def test_custom_attributes(self):
        """Test custom attributes support."""
        attrs = client_side_attributes(
            pageTitle="Dashboard",
            customField="custom_value",
            businessContext="important_data"
        )
        
        attrs_dict = attrs.to_dict()
        self.assertEqual(attrs_dict['customField'], "custom_value")
        self.assertEqual(attrs_dict['businessContext'], "important_data")
    
    def test_none_values_excluded(self):
        """Test that None values are excluded from output."""
        attrs = client_side_attributes(
            pageTitle="Test",
            deviceType=None,  # Should be excluded
            browser="firefox"
        )
        
        attrs_dict = attrs.to_dict()
        self.assertNotIn('deviceType', attrs_dict)
        self.assertIn('pageTitle', attrs_dict)
        self.assertIn('browser', attrs_dict)


class TestRequestContextPlugin(unittest.TestCase):
    """Test RequestContextPlugin functionality."""
    
    def test_plugin_creation(self):
        """Test plugin creation with client-side attributes."""
        client_attrs = client_side_attributes(
            pageTitle="Plugin Test",
            deviceType="desktop"
        )
        
        plugin = request_context_plugin(
            id_attribute="user_id",
            client_side_attributes=client_attrs,
            custom_extractors={"test_attr": lambda req: "test_value"}
        )
        
        self.assertIsInstance(plugin, RequestContextPlugin)
        self.assertEqual(plugin.id_attribute, "user_id")
        self.assertEqual(plugin.client_side_attributes, client_attrs)
    
    def test_growthbook_integration(self):
        """Test plugin integration with GrowthBook."""
        client_attrs = client_side_attributes(
            pageTitle="Integration Test",
            deviceType="mobile",
            customAttr="test_value"
        )
        
        plugin = request_context_plugin(client_side_attributes=client_attrs)
        
        # Don't provide client_key to avoid API calls
        gb = GrowthBook(plugins=[plugin])
        
        # Verify client-side attributes were merged
        final_attrs = gb.get_attributes()
        self.assertIn('pageTitle', final_attrs)
        self.assertIn('deviceType', final_attrs)
        self.assertIn('customAttr', final_attrs)
        
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
    
    def test_plugin_creation(self):
        """Test tracking plugin creation."""
        plugin = growthbook_tracking_plugin(
            ingestor_host="https://test.growthbook.io",
            track_experiment_viewed=True,
            track_feature_evaluated=True,
            batch_size=5
        )
        
        self.assertIsInstance(plugin, GrowthBookTrackingPlugin)
        self.assertEqual(plugin.ingestor_host, "https://test.growthbook.io")
        self.assertTrue(plugin.track_experiment_viewed)
        self.assertTrue(plugin.track_feature_evaluated)
        self.assertEqual(plugin.batch_size, 5)
    
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
    
    def test_experiment_tracking(self):
        """Test experiment tracking."""
        plugin = growthbook_tracking_plugin(
            ingestor_host="https://test.growthbook.io",
            batch_size=1,
            batch_timeout=0.5
        )
        
        gb = GrowthBook(
            attributes={"id": "test-user-123"},
            plugins=[plugin]
        )
        
        # Run experiment
        result = gb.run(Experiment(
            key="test-experiment",
            variations=["control", "treatment"],
            coverage=1.0,
            weights=[0.5, 0.5]
        ))
        
        self.assertIn(result.value, ["control", "treatment"])
        self.assertTrue(result.inExperiment)
        
        # Wait for events and cleanup
        time.sleep(1)
        gb.destroy()
        time.sleep(0.5)
        
        # Assert HTTP request
        self.assert_request_data("https://test.growthbook.io/events", 1)
        
        # Find and assert experiment events
        experiment_events = self.find_events_by_type(
            'experiment_viewed',
            lambda e: e.get('experiment_id') == 'test-experiment'
        )
        self.assertEqual(len(experiment_events), 1)
        
        # Assert experiment event
        self.assert_experiment_event(
            experiment_events[0], 
            'test-experiment', 
            ["control", "treatment"],
            user_id="test-user-123"
        )


class TestPluginIntegration(unittest.TestCase):
    """Test complete plugin integration scenarios."""
    
    def setUp(self):
        """Set up mock ingestor."""
        self.mock_ingestor = MockIngestor()
        self.patch = patch('urllib.request.urlopen', side_effect=self.mock_ingestor.mock_urlopen)
        self.patch.start()
    
    def tearDown(self):
        """Clean up patches."""
        self.patch.stop()
    
    def test_combined_plugins(self):
        """Test request context and tracking plugins working together."""
        # Create client-side attributes
        client_attrs = client_side_attributes(
            pageTitle="Integration Test Page",
            deviceType="mobile",
            browser="chrome",
            pageType="checkout",
            cartValue=199.99
        )
        
        # Create GrowthBook with both plugins (no client_key to avoid API calls)
        gb = GrowthBook(
            plugins=[
                request_context_plugin(
                    client_side_attributes=client_attrs,
                    custom_extractors={
                        "user_tier": lambda req: "premium"
                    }
                ),
                growthbook_tracking_plugin(
                    ingestor_host="https://integration.growthbook.io",
                    batch_size=3,
                    batch_timeout=1.0
                )
            ]
        )
        
        # Verify attributes are merged
        final_attrs = gb.get_attributes()
        self.assertIn('pageTitle', final_attrs)
        self.assertIn('deviceType', final_attrs)
        self.assertIn('cartValue', final_attrs)
        
        # Set up and test features manually
        gb.set_features({
            "feature-a": {"defaultValue": True},
            "feature-b": {"defaultValue": False}
        })
        
        result_a = gb.is_on("feature-a")
        result_b = gb.is_on("feature-b")
        
        self.assertTrue(result_a)
        self.assertFalse(result_b)
        
        # Test experiment
        exp_result = gb.run(Experiment(
            key="integration-experiment",
            variations=["variant-a", "variant-b"]
        ))
        
        self.assertIn(exp_result.value, ["variant-a", "variant-b"])
        
        # Wait for events
        time.sleep(2)
        gb.destroy()
        time.sleep(0.5)
        
        # Verify tracking worked
        self.assertGreater(len(self.mock_ingestor.events), 0)
        self.assertGreater(len(self.mock_ingestor.requests), 0)


def run_interactive_demo():
    """Optional demo showing plugin functionality with visual output."""
    print("🚀 GROWTHBOOK PLUGIN DEMO")
    print("=" * 40)
    
    # mock_ingestor = MockIngestor()
    
    # with patch('urllib.request.urlopen', side_effect=mock_ingestor.mock_urlopen):
        
    #     # Create client-side attributes
    #     client_attrs = client_side_attributes(
    #         pageTitle="Demo Store - Product Page",
    #         deviceType="mobile",
    #         browser="safari",
    #         os="ios",
            
    #         # Business context
    #         productId="iphone-15",
    #         category="electronics",
    #         price=999.99
    #     )
        
    #     # Create GrowthBook with plugins (no client_key to avoid API calls)
    #     gb = GrowthBook(
    #         plugins=[
    #             request_context_plugin(client_side_attributes=client_attrs),
    #             growthbook_tracking_plugin(
    #                 ingestor_host="https://demo.growthbook.io",
    #                 batch_size=2,
    #                 batch_timeout=1.0
    #             )
    #         ]
    #     )
        
    #     print(f"\n📱 User Context: {len(gb.get_attributes())} attributes")
    #     for key, value in gb.get_attributes().items():
    #         print(f"   {key}: {value}")
        
    #     # Test features - set manually to avoid API calls
    #     gb.set_features({
    #         "express-checkout": {"defaultValue": True},
    #         "price-alerts": {"defaultValue": False}
    #     })
        
    #     print(f"\n🎯 Feature Tests:")
    #     express = gb.is_on("express-checkout")
    #     alerts = gb.is_on("price-alerts")
    #     print(f"   Express checkout: {express}")
    #     print(f"   Price alerts: {alerts}")
        
    #     # Test experiment
    #     print(f"\n🧪 A/B Test:")
    #     ui_test = gb.run(Experiment(
    #         key="ui-variant",
    #         variations=["classic", "modern"]
    #     ))
    #     print(f"   UI variant: {ui_test.value}")
        
    #     # Wait and cleanup
    #     time.sleep(2)
    #     gb.destroy()
    #     time.sleep(0.5)
        
    #     print(f"\n📊 Results:")
    #     print(f"   Events captured: {len(mock_ingestor.events)}")
    #     print(f"   HTTP requests: {len(mock_ingestor.requests)}")
    #     print(f"   ✅ Demo completed successfully!")


if __name__ == "__main__":
    import sys
    
    # Check if demo is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_interactive_demo()
    else:
        # Run unit tests
        unittest.main(verbosity=2) 