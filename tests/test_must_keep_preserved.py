import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from models.gating import FeatureGatingModule
from models.feature_catalog import reset_feature_catalog


class TestMustKeepPreserved(unittest.TestCase):
    """Test that must-keep features are preserved even when RFE selects zero indicators"""
    
    def setUp(self):
        """Set up test fixtures"""
        reset_feature_catalog()
        
        self.feature_groups = {
            'ohlcv': 5,
            'indicator': 8,
            'sentiment': 1
        }
        
        # Force very small RFE selection to test edge cases
        self.gating = FeatureGatingModule(
            feature_groups=self.feature_groups,
            rfe_n_features=0  # Force RFE to select 0 indicators
        )
    
    def tearDown(self):
        """Clean up after tests"""
        reset_feature_catalog()
    
    def test_must_keep_preserved_with_zero_rfe_selection(self):
        """Test that must-keep features remain active when RFE selects zero indicators"""
        # Create training data
        training_data = {
            'ohlcv': pd.DataFrame({
                'open': np.random.randn(50) + 50000,
                'high': np.random.randn(50) + 50100, 
                'low': np.random.randn(50) + 49900,
                'close': np.random.randn(50) + 50000,
                'volume': np.random.randn(50) + 1000
            }),
            'indicator': pd.DataFrame({
                f'tech_ind_{i}': np.random.randn(50) for i in range(8)
            })
        }
        
        training_labels = np.random.choice([0, 1, 2], size=50)
        
        # Mock catalog to ensure OHLCV is must-keep and indicators are RFE-eligible
        with patch.object(self.gating.feature_catalog, 'is_must_keep') as mock_is_must_keep, \
             patch.object(self.gating.feature_catalog, 'is_rfe_eligible') as mock_is_rfe_eligible:
            
            def mock_must_keep_side_effect(feature_name):
                core_features = ['open', 'high', 'low', 'close', 'volume']
                return any(core in feature_name.lower() for core in core_features)
            
            def mock_rfe_eligible_side_effect(feature_name):
                return 'tech_ind_' in feature_name
            
            mock_is_must_keep.side_effect = mock_must_keep_side_effect
            mock_is_rfe_eligible.side_effect = mock_rfe_eligible_side_effect
            
            # Run RFE with n_features=0 (should select no indicators)
            results = self.gating.perform_rfe_selection(training_data, training_labels)
            
            # Verify that RFE was performed successfully
            self.assertTrue(self.gating.rfe_performed, "RFE should have been performed")
            self.assertGreater(len(results), 0, "Should have results despite 0 RFE selection")
            
            # Verify all must-keep OHLCV features are selected
            must_keep_features = [name for name, info in results.items() 
                                if info['selected'] and info['rank'] == 0]
            
            self.assertGreater(len(must_keep_features), 0, 
                             "Should have must-keep features even with 0 RFE selection")
            
            # Check that all OHLCV features are in must-keep
            ohlcv_features = [f for f in results.keys() if f.startswith('ohlcv.')]
            for ohlcv_feature in ohlcv_features:
                self.assertTrue(results[ohlcv_feature]['selected'],
                              f"{ohlcv_feature} should be selected as must-keep")
                self.assertEqual(results[ohlcv_feature]['rank'], 0,
                               f"{ohlcv_feature} should have rank 0 (must-keep)")
                self.assertEqual(results[ohlcv_feature]['importance'], 1.0,
                               f"{ohlcv_feature} should have high importance")
            
            # Verify no indicators were selected (since rfe_n_features=0)
            indicator_selected = [name for name, info in results.items() 
                                if name.startswith('indicator.') and info['selected'] and info['rank'] > 0]
            self.assertEqual(len(indicator_selected), 0, 
                           "No indicators should be selected when rfe_n_features=0")
    
    def test_must_keep_preserved_in_active_masks(self):
        """Test that must-keep features appear in active masks even if not RFE-selected"""
        training_data = {
            'ohlcv': pd.DataFrame({
                'close': np.random.randn(30) + 50000,
                'volume': np.random.randn(30) + 1000
            }),
            'indicator': pd.DataFrame({
                'rsi': np.random.randn(30),
                'macd': np.random.randn(30)
            })
        }
        
        training_labels = np.random.choice([0, 1, 2], size=30)
        
        with patch.object(self.gating.feature_catalog, 'is_must_keep') as mock_is_must_keep, \
             patch.object(self.gating.feature_catalog, 'is_rfe_eligible') as mock_is_rfe_eligible:
            
            # OHLCV features are must-keep
            mock_is_must_keep.side_effect = lambda name: any(core in name.lower() 
                                                           for core in ['close', 'volume'])
            # Indicators are RFE-eligible
            mock_is_rfe_eligible.side_effect = lambda name: name in ['rsi', 'macd']
            
            # Run RFE first
            self.gating.perform_rfe_selection(training_data, training_labels)
            
            # Build active masks
            active_masks = self.gating.build_active_feature_masks()
            
            # Verify OHLCV group is fully active (core group should always be active)
            self.assertIn('ohlcv', active_masks, "OHLCV group should be in active masks")
            ohlcv_mask = active_masks['ohlcv']
            
            # At least some OHLCV features should be active (must-keep guarantee)
            self.assertTrue(np.any(ohlcv_mask), 
                          "At least some OHLCV features should be active (must-keep)")
    
    def test_mixed_must_keep_and_rfe_selection(self):
        """Test scenario with both must-keep features and some RFE selection"""
        # Use higher RFE selection to allow some indicators
        self.gating.rfe_n_features = 2
        
        training_data = {
            'ohlcv': pd.DataFrame({
                'close': np.random.randn(100) + 50000
            }),
            'indicator': pd.DataFrame({
                'must_keep_indicator': np.random.randn(100),  # This should be must-keep
                'rfe_eligible_1': np.random.randn(100),       # RFE eligible
                'rfe_eligible_2': np.random.randn(100),       # RFE eligible
                'rfe_eligible_3': np.random.randn(100)        # RFE eligible
            })
        }
        
        training_labels = np.random.choice([0, 1, 2], size=100)
        
        with patch.object(self.gating.feature_catalog, 'is_must_keep') as mock_is_must_keep, \
             patch.object(self.gating.feature_catalog, 'is_rfe_eligible') as mock_is_rfe_eligible:
            
            def mock_must_keep_side_effect(feature_name):
                return 'close' in feature_name or 'must_keep_indicator' in feature_name
            
            def mock_rfe_eligible_side_effect(feature_name):
                return 'rfe_eligible_' in feature_name
            
            mock_is_must_keep.side_effect = mock_must_keep_side_effect
            mock_is_rfe_eligible.side_effect = mock_rfe_eligible_side_effect
            
            # Run RFE
            results = self.gating.perform_rfe_selection(training_data, training_labels)
            
            # Verify must-keep features are selected with rank 0
            must_keep_close = 'ohlcv.close'
            must_keep_indicator = 'indicator.must_keep_indicator'
            
            self.assertIn(must_keep_close, results)
            self.assertTrue(results[must_keep_close]['selected'])
            self.assertEqual(results[must_keep_close]['rank'], 0)
            
            self.assertIn(must_keep_indicator, results)
            self.assertTrue(results[must_keep_indicator]['selected'])
            self.assertEqual(results[must_keep_indicator]['rank'], 0)
            
            # Verify some RFE-eligible indicators were processed
            rfe_eligible_features = [name for name in results.keys() 
                                   if 'rfe_eligible_' in name]
            self.assertGreater(len(rfe_eligible_features), 0, 
                             "Should have processed RFE-eligible features")
            
            # Count actually selected RFE features (non-must-keep)
            rfe_selected = [name for name, info in results.items() 
                          if info['selected'] and info['rank'] > 0]
            
            # Should have selected up to rfe_n_features from RFE candidates
            self.assertLessEqual(len(rfe_selected), self.gating.rfe_n_features,
                               f"Should not select more than {self.gating.rfe_n_features} RFE features")
    
    def test_must_keep_count_tracking(self):
        """Test that must_keep_count is properly tracked"""
        training_data = {
            'ohlcv': pd.DataFrame({
                'open': np.random.randn(50),
                'close': np.random.randn(50)
            }),
            'indicator': pd.DataFrame({
                'indicator_1': np.random.randn(50)
            })
        }
        
        training_labels = np.random.choice([0, 1, 2], size=50)
        
        with patch.object(self.gating.feature_catalog, 'is_must_keep') as mock_is_must_keep, \
             patch.object(self.gating.feature_catalog, 'is_rfe_eligible') as mock_is_rfe_eligible:
            
            # Only 'open' and 'close' are must-keep
            mock_is_must_keep.side_effect = lambda name: name.lower() in ['open', 'close']
            mock_is_rfe_eligible.side_effect = lambda name: 'indicator_' in name
            
            # Run RFE
            self.gating.perform_rfe_selection(training_data, training_labels)
            
            # Check tracking attributes
            self.assertEqual(self.gating.must_keep_count, 2, 
                           "Should track 2 must-keep features (open, close)")
            self.assertGreaterEqual(self.gating.pool_candidates, 0, 
                                  "Should track RFE candidate count")
            
            # Verify in RFE summary
            summary = self.gating.get_rfe_summary()
            self.assertEqual(summary['must_keep_count'], 2, 
                           "RFE summary should include must_keep_count")
            self.assertIn('pool_candidates', summary,
                         "RFE summary should include pool_candidates")


if __name__ == '__main__':
    unittest.main()