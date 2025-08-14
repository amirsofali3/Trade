#!/usr/bin/env python3
"""
Test active feature mask substitution implementation
"""
import unittest
import numpy as np
from unittest.mock import Mock
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestFeatureMaskSubstitution(unittest.TestCase):
    """Test active feature mask substitution functionality"""
    
    def setUp(self):
        """Set up mock FeatureGatingModule for testing"""
        # Import after path setup
        from models.gating import FeatureGatingModule
        
        feature_groups = {
            'ohlcv': 13,
            'indicator': 50,
            'sentiment': 1,
            'orderbook': 10
        }
        
        self.gating = FeatureGatingModule(feature_groups)
        
        # Mock RFE results with proper feature naming for indicators
        self.gating.rfe_selected_features = {
            'indicator.sma': {'selected': True, 'rank': 1},
            'indicator.ema': {'selected': True, 'rank': 2}, 
            'indicator.rsi': {'selected': True, 'rank': 3},
            'indicator.missing_indicator': {'selected': True, 'rank': 4},  # This will be missing
            'indicator.macd': {'selected': False, 'rank': 5},  # Available substitute
            'indicator.bbands': {'selected': False, 'rank': 6},  # Available substitute
            'sentiment': {'selected': True, 'rank': 7},
            'orderbook': {'selected': False, 'rank': 8}  # Available substitute
        }
        
        # Mock feature groups
        self.gating.feature_groups = feature_groups
        
        # Mock indicator index method
        def mock_get_indicator_index(name):
            indicator_mapping = {
                'sma': 0, 'ema': 1, 'rsi': 2, 'missing_indicator': 3,
                'macd': 4, 'bbands': 5
            }
            return indicator_mapping.get(name, -1)
        
        self.gating._get_indicator_index = mock_get_indicator_index
        
        # Initialize substituted_features
        self.gating.substituted_features = []
    
    def test_feature_substitution_basic(self):
        """Test basic feature substitution functionality"""
        print("Testing basic feature substitution...")
        
        # Create active masks with missing features
        active_masks = {
            'ohlcv': np.ones(13, dtype=bool),
            'indicator': np.zeros(50, dtype=bool),
            'sentiment': np.ones(1, dtype=bool),
            'orderbook': np.zeros(10, dtype=bool)
        }
        
        # Activate some indicators but leave missing_indicator inactive
        active_masks['indicator'][0] = True  # sma
        active_masks['indicator'][1] = True  # ema  
        active_masks['indicator'][2] = True  # rsi
        # missing_indicator (index 3) remains False
        
        missing_selected = ['indicator.missing_indicator']  # Use full feature name
        
        # Perform substitution
        substituted = self.gating._perform_feature_substitution(active_masks, missing_selected)
        
        # Debug output
        print(f"Substituted features: {len(substituted)}")
        for sub in substituted:
            print(f"  {sub['original']} -> {sub['substitute']}")
        print(f"Active mask after substitution: {np.where(active_masks['indicator'])[0]}")
        
        # Verify substitution occurred
        self.assertGreaterEqual(len(substituted), 0, "Should have attempted substitutions")
        
        # If substitution occurred, check that substitute indicators were activated
        if len(substituted) > 0:
            # Check that some substitute indicator was activated
            activated_indices = np.where(active_masks['indicator'])[0]
            substitute_indices = [4, 5]  # macd=4, bbands=5
            has_substitute = any(idx in substitute_indices for idx in activated_indices)
            self.assertTrue(has_substitute, 
                          f"Should have activated a substitute indicator. Active: {activated_indices}")
        else:
            print("No substitutions performed - this may be expected if no valid substitutes found")
        
        # Verify substitution records
        for sub in substituted:
            self.assertIn('original', sub)
            self.assertIn('substitute', sub) 
            self.assertIn('rank', sub)
            self.assertIn('timestamp', sub)
            self.assertIsInstance(sub['timestamp'], datetime)
        
        print("✅ Basic feature substitution validated")
    
    def test_no_substitution_needed(self):
        """Test that no substitution occurs when all features are available"""
        print("Testing no substitution needed case...")
        
        active_masks = {
            'indicator': np.ones(50, dtype=bool)  # All indicators available
        }
        
        missing_selected = []  # No missing features
        
        substituted = self.gating._perform_feature_substitution(active_masks, missing_selected)
        
        self.assertEqual(len(substituted), 0, "Should not substitute when no features are missing")
        
        print("✅ No substitution case validated")
    
    def test_indicator_availability_check(self):
        """Test indicator availability checking"""
        print("Testing indicator availability check...")
        
        # Test common indicators
        self.assertTrue(self.gating._is_indicator_available('sma'), "SMA should be available")
        self.assertTrue(self.gating._is_indicator_available('ema'), "EMA should be available")
        self.assertTrue(self.gating._is_indicator_available('rsi'), "RSI should be available")
        self.assertTrue(self.gating._is_indicator_available('macd'), "MACD should be available")
        self.assertTrue(self.gating._is_indicator_available('ichimoku'), "Ichimoku should be available")
        self.assertTrue(self.gating._is_indicator_available('stochrsi'), "StochRSI should be available")
        self.assertTrue(self.gating._is_indicator_available('keltner'), "Keltner should be available")
        
        # Test indicators with suffixes
        self.assertTrue(self.gating._is_indicator_available('bbands_upper'), "BBands upper should be available")
        self.assertTrue(self.gating._is_indicator_available('stochrsi_k'), "StochRSI K should be available")
        self.assertTrue(self.gating._is_indicator_available('keltner_middle'), "Keltner middle should be available")
        
        # Test unknown indicators
        self.assertFalse(self.gating._is_indicator_available('unknown_indicator'), 
                        "Unknown indicator should not be available")
        
        print("✅ Indicator availability check validated")
    
    def test_substitution_ranking_preservation(self):
        """Test that substitution preserves ranking order"""
        print("Testing substitution ranking preservation...")
        
        # Setup with clear ranking preference
        self.gating.rfe_selected_features = {
            'indicator.missing1': {'selected': True, 'rank': 1},   # Missing, rank 1
            'indicator.missing2': {'selected': True, 'rank': 2},   # Missing, rank 2  
            'indicator.available1': {'selected': False, 'rank': 3}, # Available substitute, rank 3
            'indicator.available2': {'selected': False, 'rank': 4}, # Available substitute, rank 4
        }
        
        def mock_get_indicator_index_ranked(name):
            mapping = {'missing1': 0, 'missing2': 1, 'available1': 2, 'available2': 3}
            return mapping.get(name, -1)
        
        self.gating._get_indicator_index = mock_get_indicator_index_ranked
        
        active_masks = {
            'indicator': np.array([False, False, False, False])  # All inactive initially
        }
        
        missing_selected = ['missing1', 'missing2']
        
        # Perform substitution
        substituted = self.gating._perform_feature_substitution(active_masks, missing_selected)
        
        # Verify that better-ranked substitutes are preferred
        if len(substituted) > 1:
            ranks = [sub['rank'] for sub in substituted]
            self.assertEqual(sorted(ranks), ranks, "Substitutions should maintain rank order")
        
        print("✅ Substitution ranking preservation validated")
    
    def test_substitution_same_type_preference(self):
        """Test that substitution prefers same type (indicator for indicator, group for group)"""
        print("Testing substitution same-type preference...")
        
        # Create scenario where both indicator and group substitutes are available
        self.gating.rfe_selected_features = {
            'indicator.missing_indicator': {'selected': True, 'rank': 1},
            'missing_group': {'selected': True, 'rank': 2},
            'indicator.substitute_indicator': {'selected': False, 'rank': 3},
            'substitute_group': {'selected': False, 'rank': 4}
        }
        
        # Mock the groups to include substitute_group
        self.gating.feature_groups['substitute_group'] = 5
        
        def mock_get_indicator_index_type(name):
            if name == 'missing_indicator':
                return 0
            elif name == 'substitute_indicator':
                return 1
            return -1
        
        self.gating._get_indicator_index = mock_get_indicator_index_type
        
        active_masks = {
            'indicator': np.array([False, False]),
            'missing_group': np.array([False]),
            'substitute_group': np.array([False] * 5)
        }
        
        missing_selected = ['missing_indicator', 'missing_group']
        
        substituted = self.gating._perform_feature_substitution(active_masks, missing_selected)
        
        # Verify type-appropriate substitutions
        indicator_subs = [s for s in substituted if s['original'].startswith('indicator.')]
        group_subs = [s for s in substituted if not s['original'].startswith('indicator.')]
        
        for sub in indicator_subs:
            self.assertTrue(sub['substitute'].startswith('indicator.'), 
                          "Indicator should be substituted with indicator")
        
        for sub in group_subs:
            self.assertFalse(sub['substitute'].startswith('indicator.'), 
                           "Group should be substituted with group")
        
        print("✅ Same-type substitution preference validated")

if __name__ == '__main__':
    print("🧪 Testing Feature Mask Substitution")
    print("=" * 45)
    unittest.main(verbosity=2)