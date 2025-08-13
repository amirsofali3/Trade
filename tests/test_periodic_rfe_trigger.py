#!/usr/bin/env python3
"""
Test periodic RFE trigger functionality.

Simulates performance drop triggering re-run and validates time-based triggers.
"""

import unittest
import time
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.gating import FeatureGatingModule


class TestPeriodicRFETrigger(unittest.TestCase):
    """Test periodic RFE triggering logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.feature_groups = {
            'indicator': 20,
            'ohlcv': 5
        }
        self.gating = FeatureGatingModule(
            feature_groups=self.feature_groups,
            rfe_enabled=True,
            rfe_n_features=10
        )
        
        # Set initial RFE time
        self.gating.last_rfe_time = time.time()
    
    def test_periodic_rfe_disabled_by_default(self):
        """Test that periodic RFE is disabled by default"""
        config = {
            'rfe': {
                'periodic': {
                    'enabled': False
                }
            }
        }
        
        # Should not trigger when disabled
        should_trigger = self.gating.should_trigger_periodic_rfe(config)
        self.assertFalse(should_trigger)
    
    def test_time_based_trigger(self):
        """Test time-based RFE triggering"""
        config = {
            'rfe': {
                'periodic': {
                    'enabled': True,
                    'interval_minutes': 30
                }
            }
        }
        
        # Set last RFE time to more than 30 minutes ago
        self.gating.last_rfe_time = time.time() - (31 * 60)  # 31 minutes ago
        
        should_trigger = self.gating.should_trigger_periodic_rfe(config)
        self.assertTrue(should_trigger)
    
    def test_time_based_no_trigger_within_interval(self):
        """Test that RFE doesn't trigger within time interval"""
        config = {
            'rfe': {
                'periodic': {
                    'enabled': True,
                    'interval_minutes': 30
                }
            }
        }
        
        # Set last RFE time to within interval
        self.gating.last_rfe_time = time.time() - (15 * 60)  # 15 minutes ago
        
        should_trigger = self.gating.should_trigger_periodic_rfe(config)
        self.assertFalse(should_trigger)
    
    def test_performance_based_trigger(self):
        """Test performance drop triggering RFE"""
        config = {
            'rfe': {
                'periodic': {
                    'enabled': True,
                    'interval_minutes': 30
                },
                'trigger': {
                    'performance_drop_pct': 15
                }
            }
        }
        
        # Set recent RFE time (within interval)
        self.gating.last_rfe_time = time.time() - (10 * 60)  # 10 minutes ago
        
        # Performance dropped by more than 15%
        current_performance = 0.70  # 70% success rate
        min_threshold = 1.0 - (15 / 100.0)  # 85% threshold
        
        should_trigger = self.gating.should_trigger_periodic_rfe(config, current_performance)
        self.assertTrue(should_trigger)
    
    def test_performance_no_trigger_within_threshold(self):
        """Test that good performance doesn't trigger RFE"""
        config = {
            'rfe': {
                'periodic': {
                    'enabled': True,
                    'interval_minutes': 30
                },
                'trigger': {
                    'performance_drop_pct': 15
                }
            }
        }
        
        # Set recent RFE time (within interval)
        self.gating.last_rfe_time = time.time() - (10 * 60)
        
        # Good performance (above threshold)
        current_performance = 0.90  # 90% success rate
        
        should_trigger = self.gating.should_trigger_periodic_rfe(config, current_performance)
        self.assertFalse(should_trigger)
    
    def test_first_run_trigger(self):
        """Test that RFE triggers on first run (no previous RFE time)"""
        config = {
            'rfe': {
                'periodic': {
                    'enabled': True,
                    'interval_minutes': 30
                }
            }
        }
        
        # No previous RFE time
        self.gating.last_rfe_time = None
        
        should_trigger = self.gating.should_trigger_periodic_rfe(config)
        self.assertTrue(should_trigger)
    
    def test_performance_tracking_update(self):
        """Test performance tracking updates"""
        initial_length = len(self.gating.recent_success_rates)
        
        # Add performance data
        for i in range(25):  # Add more than the 20 limit
            self.gating.update_performance_tracking(0.8 + (i * 0.01))
        
        # Should keep only last 20 values
        self.assertEqual(len(self.gating.recent_success_rates), 20)
        
        # Should have the most recent values
        self.assertAlmostEqual(self.gating.recent_success_rates[-1], 0.8 + (24 * 0.01))
    
    def test_trigger_with_custom_thresholds(self):
        """Test triggering with custom performance thresholds"""
        config = {
            'rfe': {
                'periodic': {
                    'enabled': True,
                    'interval_minutes': 60
                },
                'trigger': {
                    'performance_drop_pct': 10  # More sensitive threshold
                }
            }
        }
        
        # Set recent RFE time
        self.gating.last_rfe_time = time.time() - (30 * 60)  # 30 minutes ago
        
        # Performance dropped by more than 10% (threshold = 90%)
        current_performance = 0.85  # 85% success rate
        
        should_trigger = self.gating.should_trigger_periodic_rfe(config, current_performance)
        self.assertTrue(should_trigger)
    
    def test_trigger_with_missing_config(self):
        """Test graceful handling of missing configuration"""
        config = {}  # Empty config
        
        should_trigger = self.gating.should_trigger_periodic_rfe(config)
        self.assertFalse(should_trigger)
    
    def test_trigger_with_partial_config(self):
        """Test handling of partial configuration"""
        config = {
            'rfe': {
                'periodic': {
                    'enabled': True
                    # Missing interval_minutes - should use default
                }
            }
        }
        
        # Set last RFE time to more than default interval (30 min)
        self.gating.last_rfe_time = time.time() - (35 * 60)
        
        should_trigger = self.gating.should_trigger_periodic_rfe(config)
        self.assertTrue(should_trigger)
    
    def test_performance_tracking_initialization(self):
        """Test that performance tracking initializes correctly"""
        self.assertIsInstance(self.gating.recent_success_rates, list)
        self.assertEqual(len(self.gating.recent_success_rates), 0)
        self.assertIsNotNone(self.gating.last_performance_check)
    
    def test_multiple_trigger_conditions(self):
        """Test that either time or performance can trigger RFE"""
        config = {
            'rfe': {
                'periodic': {
                    'enabled': True,
                    'interval_minutes': 60
                },
                'trigger': {
                    'performance_drop_pct': 15
                }
            }
        }
        
        # Test time trigger (performance good, but time exceeded)
        self.gating.last_rfe_time = time.time() - (65 * 60)  # 65 minutes ago
        current_performance = 0.95  # Great performance
        
        should_trigger = self.gating.should_trigger_periodic_rfe(config, current_performance)
        self.assertTrue(should_trigger)
        
        # Test performance trigger (time recent, but performance poor)
        self.gating.last_rfe_time = time.time() - (10 * 60)  # 10 minutes ago
        current_performance = 0.70  # Poor performance
        
        should_trigger = self.gating.should_trigger_periodic_rfe(config, current_performance)
        self.assertTrue(should_trigger)


if __name__ == '__main__':
    unittest.main()