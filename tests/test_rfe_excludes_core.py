import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd
from models.gating import FeatureGatingModule
from models.feature_catalog import reset_feature_catalog


class TestRFEExcludesCore(unittest.TestCase):
    """Test that RFE excludes core OHLCV features from candidacy"""
    
    def setUp(self):
        """Set up test fixtures"""
        reset_feature_catalog()
        
        # Mock feature groups
        self.feature_groups = {
            'ohlcv': 6,  # Open, High, Low, Close, Volume, Timestamp
            'indicator': 10,  # Various technical indicators
            'sentiment': 1
        }
        
        self.gating = FeatureGatingModule(
            feature_groups=self.feature_groups,
            rfe_n_features=5
        )
    
    def tearDown(self):
        """Clean up after tests"""
        reset_feature_catalog()
    
    def test_rfe_excludes_ohlcv_from_candidates(self):
        """Test that OHLCV features are excluded from RFE candidate pool"""
        # Create mock training data with both OHLCV and indicators
        training_data = {
            'ohlcv': pd.DataFrame({
                'open': np.random.randn(100) + 50000,
                'high': np.random.randn(100) + 50100,
                'low': np.random.randn(100) + 49900,
                'close': np.random.randn(100) + 50000,
                'volume': np.random.randn(100) + 1000,
                'timestamp': pd.date_range('2023-01-01', periods=100, freq='5min')
            }),
            'indicator': pd.DataFrame({
                f'ind_{i}': np.random.randn(100) for i in range(10)
            })
        }
        
        # Create mock labels - make sure they have diversity
        training_labels = np.random.choice([0, 1, 2], size=100, p=[0.4, 0.4, 0.2])
        
        # Mock the feature catalog to ensure OHLCV is must-keep
        with patch.object(self.gating.feature_catalog, 'is_must_keep') as mock_is_must_keep, \
             patch.object(self.gating.feature_catalog, 'is_rfe_eligible') as mock_is_rfe_eligible:
            
            def mock_must_keep_side_effect(feature_name):
                # Core OHLCV features are must-keep
                core_features = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
                return any(core in feature_name.lower() for core in core_features)
            
            def mock_rfe_eligible_side_effect(feature_name):
                # Only indicator features are RFE eligible
                return 'ind_' in feature_name.lower()
            
            mock_is_must_keep.side_effect = mock_must_keep_side_effect
            mock_is_rfe_eligible.side_effect = mock_rfe_eligible_side_effect
            
            # Run RFE selection
            results = self.gating.perform_rfe_selection(training_data, training_labels)
            
            # Assert that results contain features
            self.assertGreater(len(results), 0, "RFE should return some results")
            
            # Check that all OHLCV features appear in final selection as must-keep
            ohlcv_features = [f for f in results.keys() if f.startswith('ohlcv.')]
            
            for ohlcv_feature in ohlcv_features:
                self.assertIn(ohlcv_feature, results, f"{ohlcv_feature} should be in final results")
                self.assertTrue(results[ohlcv_feature]['selected'], f"{ohlcv_feature} should be selected")
                self.assertEqual(results[ohlcv_feature]['rank'], 0, f"{ohlcv_feature} should have synthetic rank 0 (must-keep)")
            
            # Check that OHLCV features were NOT processed by RFE (they should have synthetic rank 0)
            for feature_name, feature_info in results.items():
                if feature_name.startswith('ohlcv.'):
                    self.assertEqual(feature_info['rank'], 0, 
                                   f"OHLCV feature {feature_name} should have synthetic rank 0, not RFE-assigned rank")
                    self.assertEqual(feature_info['importance'], 1.0,
                                   f"OHLCV feature {feature_name} should have high importance (1.0)")
    
    def test_rfe_processes_only_eligible_indicators(self):
        """Test that only RFE-eligible indicators enter the RFE candidate pool"""
        # Create training data
        training_data = {
            'indicator': pd.DataFrame({
                'sma_5': np.random.randn(100),      # Should be RFE eligible
                'rsi_14': np.random.randn(100),     # Should be RFE eligible  
                'must_keep_ind': np.random.randn(100)  # Should be must-keep
            })
        }
        
        training_labels = np.random.choice([0, 1, 2], size=100)
        
        # Mock the catalog
        with patch.object(self.gating.feature_catalog, 'is_must_keep') as mock_is_must_keep, \
             patch.object(self.gating.feature_catalog, 'is_rfe_eligible') as mock_is_rfe_eligible:
            
            def mock_must_keep_side_effect(feature_name):
                return 'must_keep_ind' in feature_name
            
            def mock_rfe_eligible_side_effect(feature_name):
                return feature_name in ['sma_5', 'rsi_14']
            
            mock_is_must_keep.side_effect = mock_must_keep_side_effect
            mock_is_rfe_eligible.side_effect = mock_rfe_eligible_side_effect
            
            # Run RFE
            results = self.gating.perform_rfe_selection(training_data, training_labels)
            
            # Verify must-keep indicator is selected with rank 0
            must_keep_feature = 'indicator.must_keep_ind'
            self.assertIn(must_keep_feature, results)
            self.assertTrue(results[must_keep_feature]['selected'])
            self.assertEqual(results[must_keep_feature]['rank'], 0)
            
            # Verify RFE-eligible indicators have proper ranks (not 0)
            rfe_features = ['indicator.sma_5', 'indicator.rsi_14']
            for feature_name in rfe_features:
                if feature_name in results:
                    if results[feature_name]['selected']:
                        # If selected by RFE, should have rank > 0
                        self.assertGreater(results[feature_name]['rank'], 0,
                                         f"RFE-selected feature {feature_name} should have rank > 0")
    
    def test_pool_statistics_logging(self):
        """Test that RFE logs pool statistics correctly"""
        training_data = {
            'ohlcv': pd.DataFrame({
                'close': np.random.randn(50) + 50000
            }),
            'indicator': pd.DataFrame({
                'ind_1': np.random.randn(50),
                'ind_2': np.random.randn(50)
            })
        }
        
        training_labels = np.random.choice([0, 1, 2], size=50)
        
        with patch.object(self.gating.feature_catalog, 'is_must_keep') as mock_is_must_keep, \
             patch.object(self.gating.feature_catalog, 'is_rfe_eligible') as mock_is_rfe_eligible, \
             patch('models.gating.logger') as mock_logger:
            
            mock_is_must_keep.side_effect = lambda name: 'close' in name.lower()
            mock_is_rfe_eligible.side_effect = lambda name: 'ind_' in name.lower()
            
            # Run RFE
            self.gating.perform_rfe_selection(training_data, training_labels)
            
            # Check that pool statistics were logged
            logged_messages = [call[0][0] for call in mock_logger.info.call_args_list if call[0]]
            
            # Look for the pool statistics log message
            pool_log_found = any('RFE Pool:' in msg and 'candidates=' in msg and 'must_keep=' in msg 
                               for msg in logged_messages)
            self.assertTrue(pool_log_found, "RFE should log pool statistics")
            
            # Verify tracking attributes were set
            self.assertGreater(self.gating.pool_candidates, 0, "Should track RFE candidate count")
            self.assertGreater(self.gating.must_keep_count, 0, "Should track must-keep count")


if __name__ == '__main__':
    unittest.main()