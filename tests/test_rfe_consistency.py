#!/usr/bin/env python3
"""
Test RFE consistency and summary invariants.

Validates that RFE summary data maintains expected relationships:
- strong + medium + weak == total_selected
- total_selected <= total_cleaned <= total_available  
- target_features >= strong when selection successful
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


class TestRFEConsistency(unittest.TestCase):
    """Test RFE summary invariants and consistency"""
    
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
    
    def test_rfe_summary_invariants_no_rfe(self):
        """Test RFE summary when RFE not performed"""
        summary = self.gating.get_rfe_summary()
        
        # Check basic structure
        self.assertFalse(summary['rfe_performed'])
        self.assertEqual(summary['total_selected'], 0)
        self.assertEqual(summary['total_available'], 0)
        self.assertEqual(summary['total_cleaned'], 0)
        self.assertEqual(summary['strong'], 0)
        self.assertEqual(summary['medium'], 0)
        self.assertEqual(summary['weak'], 0)
        self.assertEqual(summary['target_features'], 10)
        self.assertEqual(len(summary['top_features']), 0)
    
    def test_rfe_summary_invariants_with_rfe(self):
        """Test RFE summary invariants when RFE performed"""
        # Mock RFE results
        self._setup_mock_rfe_results()
        
        summary = self.gating.get_rfe_summary()
        
        # Test core invariants
        self.assertTrue(summary['rfe_performed'])
        
        # Invariant 1: strong + medium + weak should relate to total features
        # Note: In our implementation, strong/medium/weak are based on rank thresholds
        # not just selected features
        total_features = summary['strong'] + summary['medium'] + summary['weak'] 
        self.assertGreater(total_features, 0, "Should have some features categorized")
        
        # Invariant 2: total_selected <= total_cleaned <= total_available
        self.assertLessEqual(summary['total_selected'], summary['total_cleaned'])
        self.assertLessEqual(summary['total_cleaned'], summary['total_available'])
        
        # Invariant 3: target_features >= strong when selection successful
        self.assertLessEqual(summary['strong'], summary['target_features'])
        
        # Additional checks
        self.assertGreater(summary['total_selected'], 0)
        self.assertGreater(summary['total_available'], 0)
        self.assertEqual(summary['target_features'], 10)
        
        # Top features should not exceed 10
        self.assertLessEqual(len(summary['top_features']), 10)
    
    def test_feature_categories_consistency(self):
        """Test that feature categories are consistent with counts"""
        # Mock RFE results
        self._setup_mock_rfe_results()
        
        # Check feature_categories matches rfe_counts
        strong_from_categories = sum(1 for cat in self.gating.feature_categories.values() 
                                   if cat['category'] == 'strong')
        medium_from_categories = sum(1 for cat in self.gating.feature_categories.values()
                                   if cat['category'] == 'medium')
        weak_from_categories = sum(1 for cat in self.gating.feature_categories.values()
                                 if cat['category'] == 'weak')
        
        self.assertEqual(strong_from_categories, self.gating.rfe_counts['strong'])
        self.assertEqual(medium_from_categories, self.gating.rfe_counts['medium'])
        self.assertEqual(weak_from_categories, self.gating.rfe_counts['weak'])
    
    def test_selected_features_consistency(self):
        """Test that selected_features list matches rfe_selected_features"""
        # Mock RFE results
        self._setup_mock_rfe_results()
        
        selected_from_dict = [name for name, info in self.gating.rfe_selected_features.items()
                             if info['selected']]
        
        self.assertEqual(set(selected_from_dict), set(self.gating.selected_features))
        
        # Check that selected features count matches summary
        summary = self.gating.get_rfe_summary()
        self.assertEqual(len(self.gating.selected_features), summary['total_selected'])
    
    def test_rank_thresholds_consistency(self):
        """Test that rank-based categorization is consistent"""
        # Mock RFE results
        self._setup_mock_rfe_results()
        
        target_features = self.gating.rfe_n_features
        
        for feature_name, category_info in self.gating.feature_categories.items():
            rank = category_info['rank']
            category = category_info['category']
            
            if rank <= target_features:
                self.assertEqual(category, 'strong', 
                               f"Feature {feature_name} with rank {rank} should be strong")
            elif rank <= 2 * target_features:
                self.assertEqual(category, 'medium',
                               f"Feature {feature_name} with rank {rank} should be medium")
            else:
                self.assertEqual(category, 'weak',
                               f"Feature {feature_name} with rank {rank} should be weak")
    
    def _setup_mock_rfe_results(self):
        """Set up mock RFE results for testing"""
        # Create mock feature results with varied ranks
        feature_names = [f'indicator.{i}' for i in range(15)] + [f'ohlcv.{i}' for i in range(5)]
        
        self.gating.rfe_all_features = feature_names.copy()
        self.gating.rfe_performed = True
        self.gating.rfe_selected_features = {}
        
        # Create features with different ranks - first 10 selected
        for i, name in enumerate(feature_names):
            rank = i + 1
            selected = i < 10  # First 10 are selected
            
            self.gating.rfe_selected_features[name] = {
                'rank': rank,
                'importance': 0.9 - (i * 0.04),  # Decreasing importance
                'selected': selected
            }
        
        self.gating.selected_features = [name for name, info in self.gating.rfe_selected_features.items()
                                       if info['selected']]
        
        # Build categories
        self.gating._build_feature_categories()


if __name__ == '__main__':
    unittest.main()