#!/usr/bin/env python3
"""
Test active feature mask functionality.

Validates that mask length aligns with encoder output and only active features are passed.
"""

import unittest
import numpy as np
import torch
from unittest.mock import Mock, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.gating import FeatureGatingModule


class TestActiveFeatureMask(unittest.TestCase):
    """Test active feature mask creation and alignment"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.feature_groups = {
            'ohlcv': 5,
            'indicator': 20,
            'sentiment': 1,
            'orderbook': 10
        }
        self.gating = FeatureGatingModule(
            feature_groups=self.feature_groups,
            rfe_enabled=True,
            rfe_n_features=15
        )
    
    def test_active_mask_creation_without_rfe(self):
        """Test that active masks can't be created without RFE"""
        masks = self.gating.build_active_feature_masks()
        self.assertEqual(masks, {})
    
    def test_active_mask_creation_with_rfe(self):
        """Test active mask creation after RFE"""
        # Mock RFE results
        self._setup_mock_rfe_results()
        
        # Build active masks
        masks = self.gating.build_active_feature_masks()
        
        # Check that masks are created for each group
        self.assertIn('ohlcv', masks)
        self.assertIn('indicator', masks)
        self.assertIn('sentiment', masks)
        self.assertIn('orderbook', masks)
        
        # Check mask lengths match feature group sizes
        self.assertEqual(len(masks['ohlcv']), self.feature_groups['ohlcv'])
        # For indicators, the mask length should match the available indicators count  
        available_indicators = len(self.gating._get_indicator_names())
        self.assertEqual(len(masks['indicator']), available_indicators)
        self.assertEqual(len(masks['sentiment']), self.feature_groups['sentiment'])
        self.assertEqual(len(masks['orderbook']), self.feature_groups['orderbook'])
    
    def test_mask_data_types(self):
        """Test that masks are boolean arrays"""
        # Mock RFE results
        self._setup_mock_rfe_results()
        
        # Build active masks
        masks = self.gating.build_active_feature_masks()
        
        for group_name, mask in masks.items():
            self.assertIsInstance(mask, np.ndarray)
            self.assertEqual(mask.dtype, bool)
    
    def test_minimum_active_features_fallback(self):
        """Test fallback mechanism for minimum active features"""
        # Mock RFE results with very few selected features
        self._setup_mock_rfe_results_minimal()
        
        # Build active masks with higher minimum
        min_features = 8
        masks = self.gating.build_active_feature_masks(min_features)
        
        # Count total active features
        total_active = sum(np.sum(mask) for mask in masks.values())
        
        # Should meet minimum threshold
        self.assertGreaterEqual(total_active, min_features)
    
    def test_mask_alignment_with_selected_features(self):
        """Test that masks align with RFE selected features"""
        # Mock RFE results
        self._setup_mock_rfe_results()
        
        # Build active masks
        masks = self.gating.build_active_feature_masks()
        
        # For indicator group, check alignment with selected indicators
        selected_indicators = []
        for feature_name, info in self.gating.rfe_selected_features.items():
            if feature_name.startswith('indicator.') and info['selected']:
                indicator_name = feature_name.split('.', 1)[1]
                selected_indicators.append(indicator_name)
        
        # Count active indicators in mask
        active_indicator_count = np.sum(masks['indicator'])
        
        # Should match number of selected indicators
        self.assertEqual(active_indicator_count, len(selected_indicators))
    
    def test_mask_persistence(self):
        """Test that masks are stored persistently in gating module"""
        # Mock RFE results
        self._setup_mock_rfe_results()
        
        # Build active masks
        masks = self.gating.build_active_feature_masks()
        
        # Check that masks are stored in gating module
        self.assertIsNotNone(self.gating.active_feature_masks)
        self.assertEqual(len(self.gating.active_feature_masks), len(self.feature_groups))
        
        # Check that stored masks match returned masks
        for group_name in self.feature_groups.keys():
            np.testing.assert_array_equal(
                self.gating.active_feature_masks[group_name],
                masks[group_name]
            )
    
    def test_group_level_selection(self):
        """Test group-level feature selection (non-indicator groups)"""
        # Mock RFE with group-level selections
        self.gating.rfe_performed = True
        self.gating.rfe_selected_features = {
            'ohlcv': {'selected': True, 'rank': 1},
            'sentiment': {'selected': False, 'rank': 25},
            'orderbook': {'selected': True, 'rank': 5}
        }
        
        # Build active masks
        masks = self.gating.build_active_feature_masks()
        
        # Check group-level selections
        self.assertTrue(np.all(masks['ohlcv']))  # All features active
        self.assertFalse(np.any(masks['sentiment']))  # No features active
        self.assertTrue(np.all(masks['orderbook']))  # All features active
    
    def _setup_mock_rfe_results(self):
        """Set up mock RFE results for testing"""
        self.gating.rfe_performed = True
        self.gating.rfe_selected_features = {}
        
        # Get actual indicator names and create mock selected indicators
        indicator_names = self.gating._get_indicator_names()
        num_to_select = min(10, len(indicator_names))
        
        for i, name in enumerate(indicator_names):
            selected = i < num_to_select  # First 10 selected
            self.gating.rfe_selected_features[f'indicator.{name}'] = {
                'selected': selected,
                'rank': i + 1
            }
        
        # Add other groups
        self.gating.rfe_selected_features.update({
            'ohlcv': {'selected': True, 'rank': 1},
            'sentiment': {'selected': True, 'rank': 8},
            'orderbook': {'selected': True, 'rank': 12}
        })
        
        # Set selected features list
        self.gating.selected_features = [
            name for name, info in self.gating.rfe_selected_features.items()
            if info['selected']
        ]
    
    def _setup_mock_rfe_results_minimal(self):
        """Set up mock RFE results with minimal selections"""
        self.gating.rfe_performed = True
        self.gating.rfe_selected_features = {}
        
        # Only select 2 indicators
        indicator_names = self.gating._get_indicator_names()
        num_to_select = min(2, len(indicator_names))
        
        for i, name in enumerate(indicator_names):
            selected = i < num_to_select  # Only first 2 selected
            self.gating.rfe_selected_features[f'indicator.{name}'] = {
                'selected': selected,
                'rank': i + 1
            }
        
        # Only select OHLCV group
        self.gating.rfe_selected_features.update({
            'ohlcv': {'selected': True, 'rank': 1},
            'sentiment': {'selected': False, 'rank': 25},
            'orderbook': {'selected': False, 'rank': 30}
        })
        
        # Set selected features list
        self.gating.selected_features = [
            name for name, info in self.gating.rfe_selected_features.items()
            if info['selected']
        ]


if __name__ == '__main__':
    unittest.main()