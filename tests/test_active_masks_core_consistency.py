import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from models.gating import FeatureGatingModule
from models.feature_catalog import reset_feature_catalog


class TestActiveMasksCoreConsistency(unittest.TestCase):
    """Test that active feature masks maintain consistency with core OHLCV requirements"""
    
    def setUp(self):
        """Set up test fixtures"""
        reset_feature_catalog()
        
        # Define expected OHLCV core columns
        self.expected_ohlcv_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        
        self.feature_groups = {
            'ohlcv': len(self.expected_ohlcv_columns),
            'indicator': 15,
            'sentiment': 1,
            'orderbook': 10
        }
        
        self.gating = FeatureGatingModule(
            feature_groups=self.feature_groups,
            rfe_n_features=8
        )
    
    def tearDown(self):
        """Clean up after tests"""
        reset_feature_catalog()
    
    def test_ohlcv_group_mask_length_consistency(self):
        """Test that OHLCV group mask length matches expected core columns"""
        # Create comprehensive training data with all expected OHLCV columns
        ohlcv_data = pd.DataFrame({
            col: np.random.randn(100) + (50000 if col in ['open', 'high', 'low', 'close'] 
                                        else 1000 if col == 'volume' 
                                        else np.arange(100))
            for col in self.expected_ohlcv_columns
        })
        
        training_data = {
            'ohlcv': ohlcv_data,
            'indicator': pd.DataFrame({
                f'tech_indicator_{i}': np.random.randn(100) for i in range(15)
            }),
            'sentiment': pd.DataFrame({
                'sentiment_score': np.random.randn(100)
            })
        }
        
        training_labels = np.random.choice([0, 1, 2], size=100)
        
        # Mock catalog to make OHLCV must-keep
        with patch.object(self.gating.feature_catalog, 'is_must_keep') as mock_is_must_keep, \
             patch.object(self.gating.feature_catalog, 'is_rfe_eligible') as mock_is_rfe_eligible:
            
            def mock_must_keep_side_effect(feature_name):
                return any(col in feature_name.lower() for col in self.expected_ohlcv_columns)
            
            def mock_rfe_eligible_side_effect(feature_name):
                return 'tech_indicator_' in feature_name or 'sentiment' in feature_name
            
            mock_is_must_keep.side_effect = mock_must_keep_side_effect
            mock_is_rfe_eligible.side_effect = mock_rfe_eligible_side_effect
            
            # Perform RFE
            self.gating.perform_rfe_selection(training_data, training_labels)
            
            # Build active masks
            active_masks = self.gating.build_active_feature_masks()
            
            # Verify OHLCV mask length matches expected core columns
            self.assertIn('ohlcv', active_masks, "OHLCV group should be in active masks")
            ohlcv_mask = active_masks['ohlcv']
            
            expected_length = len(self.expected_ohlcv_columns)
            actual_length = len(ohlcv_mask)
            
            self.assertEqual(actual_length, expected_length,
                           f"OHLCV mask length should be {expected_length} (core columns), got {actual_length}")
            
            # Verify all OHLCV features are active (since they're core/must-keep)
            self.assertTrue(np.all(ohlcv_mask), 
                          "All OHLCV core features should be active in the mask")
    
    def test_core_ohlcv_always_active_after_rfe(self):
        """Test that core OHLCV features are always active regardless of RFE results"""
        training_data = {
            'ohlcv': pd.DataFrame({
                'open': np.random.randn(50) + 50000,
                'high': np.random.randn(50) + 50100,
                'low': np.random.randn(50) + 49900,
                'close': np.random.randn(50) + 50000,
                'volume': np.random.randn(50) + 1000
            }),
            'indicator': pd.DataFrame({
                'weak_indicator_1': np.random.randn(50) * 0.001,  # Very weak signal
                'weak_indicator_2': np.random.randn(50) * 0.001   # Very weak signal
            })
        }
        
        training_labels = np.random.choice([0, 1, 2], size=50)
        
        with patch.object(self.gating.feature_catalog, 'is_must_keep') as mock_is_must_keep, \
             patch.object(self.gating.feature_catalog, 'is_rfe_eligible') as mock_is_rfe_eligible:
            
            # Core OHLCV is must-keep, weak indicators are RFE-eligible
            mock_is_must_keep.side_effect = lambda name: any(col in name.lower() 
                                                           for col in ['open', 'high', 'low', 'close', 'volume'])
            mock_is_rfe_eligible.side_effect = lambda name: 'weak_indicator_' in name
            
            # Run RFE
            self.gating.perform_rfe_selection(training_data, training_labels)
            
            # Build active masks
            active_masks = self.gating.build_active_feature_masks()
            
            # Verify OHLCV group is fully active
            ohlcv_mask = active_masks.get('ohlcv', np.array([]))
            self.assertGreater(len(ohlcv_mask), 0, "OHLCV mask should not be empty")
            
            # All OHLCV features should be active (core requirement)
            active_ohlcv_count = np.sum(ohlcv_mask)
            total_ohlcv_features = len(ohlcv_mask)
            
            self.assertEqual(active_ohlcv_count, total_ohlcv_features,
                           f"All {total_ohlcv_features} OHLCV features should be active, "
                           f"but only {active_ohlcv_count} are active")
    
    def test_prerequisite_features_preserved(self):
        """Test that prerequisite features are preserved in active masks"""
        training_data = {
            'ohlcv': pd.DataFrame({
                'close': np.random.randn(80) + 50000,
                'high': np.random.randn(80) + 50100,
                'low': np.random.randn(80) + 49900
            }),
            'indicator': pd.DataFrame({
                'prev_close': np.random.randn(80) + 50000,  # Prerequisite for many indicators
                'true_range': np.random.randn(80) + 100,    # Prerequisite for ATR
                'typical_price': np.random.randn(80) + 50000,  # Prerequisite for VWAP
                'regular_indicator': np.random.randn(80)     # Regular indicator
            })
        }
        
        training_labels = np.random.choice([0, 1, 2], size=80)
        
        with patch.object(self.gating.feature_catalog, 'is_must_keep') as mock_is_must_keep, \
             patch.object(self.gating.feature_catalog, 'is_rfe_eligible') as mock_is_rfe_eligible, \
             patch.object(self.gating.feature_catalog, 'is_prereq') as mock_is_prereq:
            
            # Core OHLCV is must-keep
            mock_is_must_keep.side_effect = lambda name: any(col in name.lower() 
                                                           for col in ['close', 'high', 'low'])
            
            # Regular indicators are RFE-eligible
            mock_is_rfe_eligible.side_effect = lambda name: 'regular_indicator' in name
            
            # Prerequisites are marked as such
            mock_is_prereq.side_effect = lambda name: any(prereq in name.lower() 
                                                        for prereq in ['prev_close', 'true_range', 'typical_price'])
            
            # Run RFE
            self.gating.perform_rfe_selection(training_data, training_labels)
            
            # Build active masks 
            active_masks = self.gating.build_active_feature_masks()
            
            # Check that prerequisite features are considered in mask generation
            # (They should be active through the _is_core_or_prereq logic)
            indicator_mask = active_masks.get('indicator', np.array([]))
            self.assertGreater(len(indicator_mask), 0, "Indicator mask should not be empty")
            
            # At least some indicators should be active (prerequisites + any RFE selected)
            active_indicators = np.sum(indicator_mask)
            self.assertGreater(active_indicators, 0, 
                             "At least some indicators (prerequisites) should be active")
    
    def test_mask_consistency_with_feature_group_sizes(self):
        """Test that mask sizes are consistent with declared feature group sizes"""
        training_data = {
            'ohlcv': pd.DataFrame({
                col: np.random.randn(60) for col in ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            }),
            'indicator': pd.DataFrame({
                f'indicator_{i}': np.random.randn(60) for i in range(15)  # Should match feature_groups['indicator'] = 15
            }),
            'sentiment': pd.DataFrame({
                'sentiment': np.random.randn(60)  # Should match feature_groups['sentiment'] = 1
            })
        }
        
        training_labels = np.random.choice([0, 1, 2], size=60)
        
        with patch.object(self.gating.feature_catalog, 'is_must_keep') as mock_is_must_keep, \
             patch.object(self.gating.feature_catalog, 'is_rfe_eligible') as mock_is_rfe_eligible:
            
            mock_is_must_keep.side_effect = lambda name: 'ohlcv.' in name or name in ['ohlcv']
            mock_is_rfe_eligible.side_effect = lambda name: 'indicator_' in name or name in ['sentiment']
            
            # Run RFE
            self.gating.perform_rfe_selection(training_data, training_labels)
            
            # Build active masks
            active_masks = self.gating.build_active_feature_masks()
            
            # Verify mask sizes match declared feature group sizes
            for group_name, expected_size in self.feature_groups.items():
                if group_name in active_masks:
                    actual_size = len(active_masks[group_name])
                    self.assertEqual(actual_size, expected_size,
                                   f"Mask size for {group_name} should be {expected_size}, got {actual_size}")
    
    def test_minimum_features_fallback_preserves_core(self):
        """Test that minimum features fallback still preserves core OHLCV"""
        # Create minimal training data that might trigger fallback
        training_data = {
            'ohlcv': pd.DataFrame({
                'close': np.random.randn(30) + 50000
            }),
            'indicator': pd.DataFrame({
                'weak_ind': np.random.randn(30) * 0.001  # Very weak, likely to be unselected
            })
        }
        
        training_labels = np.random.choice([0, 1, 2], size=30)
        
        with patch.object(self.gating.feature_catalog, 'is_must_keep') as mock_is_must_keep, \
             patch.object(self.gating.feature_catalog, 'is_rfe_eligible') as mock_is_rfe_eligible:
            
            mock_is_must_keep.side_effect = lambda name: 'close' in name.lower()
            mock_is_rfe_eligible.side_effect = lambda name: 'weak_ind' in name
            
            # Run RFE
            self.gating.perform_rfe_selection(training_data, training_labels)
            
            # Build active masks with high minimum requirement to trigger fallback
            active_masks = self.gating.build_active_feature_masks(min_active_features=10)
            
            # Even with fallback, core OHLCV should remain active
            ohlcv_mask = active_masks.get('ohlcv', np.array([]))
            
            # The core OHLCV features should still be preserved
            if len(ohlcv_mask) > 0:
                # At least the must-keep 'close' should be active
                self.assertTrue(np.any(ohlcv_mask), 
                              "Core OHLCV should remain active even with minimum features fallback")
    
    def test_mask_persistence_in_gating_module(self):
        """Test that active masks are properly stored in the gating module"""
        training_data = {
            'ohlcv': pd.DataFrame({
                'close': np.random.randn(40) + 50000,
                'volume': np.random.randn(40) + 1000
            }),
            'indicator': pd.DataFrame({
                'rsi': np.random.randn(40)
            })
        }
        
        training_labels = np.random.choice([0, 1, 2], size=40)
        
        with patch.object(self.gating.feature_catalog, 'is_must_keep') as mock_is_must_keep, \
             patch.object(self.gating.feature_catalog, 'is_rfe_eligible') as mock_is_rfe_eligible:
            
            mock_is_must_keep.side_effect = lambda name: any(col in name.lower() for col in ['close', 'volume'])
            mock_is_rfe_eligible.side_effect = lambda name: 'rsi' in name
            
            # Run RFE
            self.gating.perform_rfe_selection(training_data, training_labels)
            
            # Build active masks
            returned_masks = self.gating.build_active_feature_masks()
            
            # Verify masks are stored in the gating module
            self.assertIsNotNone(self.gating.active_feature_masks, 
                               "Active feature masks should be stored in gating module")
            
            # Verify stored masks match returned masks
            for group_name in returned_masks:
                self.assertIn(group_name, self.gating.active_feature_masks,
                            f"Group {group_name} should be in stored active masks")
                
                np.testing.assert_array_equal(
                    self.gating.active_feature_masks[group_name],
                    returned_masks[group_name],
                    f"Stored mask for {group_name} should match returned mask"
                )


if __name__ == '__main__':
    unittest.main()