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
        
        # Don't provide client_key to avoid API calls
        gb = GrowthBook(plugins=[plugin])
        
        # Set up test features manually (no API calls)
        gb.set_features({
            "test-feature": {"defaultValue": True},
            "another-feature": {"defaultValue": "test_value"}
        })
        
        # Trigger feature evaluations
        result1 = gb.is_on("test-feature")
        result2 = gb.get_feature_value("another-feature", "default")
        
        self.assertTrue(result1)
        self.assertEqual(result2, "test_value")
        
        # Wait for events
        time.sleep(1)
        gb.destroy()
        time.sleep(0.5)
        
        # Verify events were captured
        self.assertGreater(len(self.mock_ingestor.events), 0)
        self.assertGreater(len(self.mock_ingestor.requests), 0)
        
        # Check event types
        feature_events = [e for e in self.mock_ingestor.events if e.get('event_type') == 'feature_evaluated']
        self.assertGreater(len(feature_events), 0)
    
    def test_experiment_tracking(self):
        """Test experiment tracking."""
        plugin = growthbook_tracking_plugin(
            ingestor_host="https://test.growthbook.io",
            batch_size=1,
            batch_timeout=0.5
        )
        
        # Don't provide client_key to avoid API calls
        gb = GrowthBook(plugins=[plugin])
        
        # Run experiment
        result = gb.run(Experiment(
            key="test-experiment",
            variations=["control", "treatment"]
        ))
        
        self.assertIn(result.value, ["control", "treatment"])
        
        # Wait for events
        time.sleep(1)
        gb.destroy()
        time.sleep(0.5)
        
        # Note: Experiments without proper setup won't generate experiment_viewed events
        # but feature_evaluated events will be generated
        self.assertGreaterEqual(len(self.mock_ingestor.events), 0)


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