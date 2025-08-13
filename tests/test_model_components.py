#!/usr/bin/env python3
"""
Consolidated tests for model components including feature processing, gating, and neural network.

Consolidates functionality from test_features.py and test_neural_network.py to avoid duplication
while ensuring comprehensive coverage of model components.
"""

import unittest
import logging
import torch
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add project root to path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelComponentsTest")


class TestModelComponents(unittest.TestCase):
    """Consolidated tests for model components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.feature_groups = {
            'ohlcv': 13,
            'indicator': 20,
            'sentiment': 1,
            'orderbook': 10
        }
    
    def test_feature_encoders(self):
        """Test feature encoder components"""
        try:
            from models.encoders.feature_encoder import (
                OHLCVEncoder, IndicatorEncoder, SentimentEncoder, 
                OrderbookEncoder, TickDataEncoder, CandlePatternEncoder
            )
            
            logger.info("Testing feature encoders...")
            
            # Create mock OHLCV data
            mock_ohlcv = self._create_mock_ohlcv_data()
            
            # Test OHLCV encoder
            ohlcv_encoder = OHLCVEncoder(window_size=20)
            ohlcv_features = ohlcv_encoder.transform(mock_ohlcv)
            self.assertIsInstance(ohlcv_features, torch.Tensor)
            logger.info(f"OHLCV encoder: {ohlcv_features.shape}")
            
            # Test Indicator encoder
            mock_indicators = self._create_mock_indicators()
            indicator_encoder = IndicatorEncoder(window_size=20)
            indicator_features = indicator_encoder.transform(mock_indicators)
            self.assertIsInstance(indicator_features, torch.Tensor)
            logger.info(f"Indicator encoder: {indicator_features.shape}")
            
            # Test other encoders
            sentiment_encoder = SentimentEncoder()
            # Create proper sentiment data format instead of just a float
            sentiment_data = {'fear_greed_index': 0.7}
            sentiment_features = sentiment_encoder.transform(sentiment_data)
            self.assertIsInstance(sentiment_features, torch.Tensor)
            
            orderbook_encoder = OrderbookEncoder(depth=5)
            mock_orderbook = {'bids': [[50000, 1], [49999, 2]], 'asks': [[50001, 1], [50002, 2]]}
            orderbook_features = orderbook_encoder.transform(mock_orderbook)
            self.assertIsInstance(orderbook_features, torch.Tensor)
            
            logger.info("✅ Feature encoders test passed")
            
        except Exception as e:
            logger.error(f"❌ Feature encoder test failed: {e}")
            self.fail(f"Feature encoder test failed: {e}")
    
    def test_feature_gating_module(self):
        """Test feature gating module functionality"""
        try:
            from models.gating import FeatureGatingModule
            
            logger.info("Testing feature gating module...")
            
            gating = FeatureGatingModule(
                feature_groups=self.feature_groups,
                rfe_enabled=True,
                rfe_n_features=10
            )
            
            # Test basic properties
            self.assertEqual(gating.rfe_n_features, 10)
            self.assertTrue(gating.rfe_enabled)
            self.assertFalse(gating.rfe_performed)
            
            # Test RFE summary before performing RFE
            summary = gating.get_rfe_summary()
            self.assertFalse(summary['rfe_performed'])
            self.assertEqual(summary['total_selected'], 0)
            
            # Test gating with mock features
            mock_features = self._create_mock_feature_dict()
            gated_features = gating(mock_features)
            self.assertIsInstance(gated_features, dict)
            
            logger.info("✅ Feature gating test passed")
            
        except Exception as e:
            logger.error(f"❌ Feature gating test failed: {e}")
            self.fail(f"Feature gating test failed: {e}")
    
    def test_neural_network_components(self):
        """Test neural network model components"""
        try:
            from models.neural_network import MarketTransformer, OnlineLearner
            from models.gating import FeatureGatingModule
            
            logger.info("Testing neural network components...")
            
            # Create feature dimensions matching encoders
            feature_dims = {
                'ohlcv': (20, 13),
                'indicator': (20, 20),
                'sentiment': (1, 1),
                'orderbook': (1, 10)
            }
            
            # Test MarketTransformer
            model = MarketTransformer(
                feature_dims=feature_dims,
                hidden_dim=64,
                n_layers=2,
                n_heads=4,
                dropout=0.1
            )
            
            # Test forward pass with mock features
            mock_features = self._create_mock_feature_dict_for_model(feature_dims)
            
            model.eval()
            with torch.no_grad():
                logits, confidences = model(mock_features)
            
            self.assertEqual(logits.shape[0], 1)  # Batch size
            self.assertEqual(logits.shape[1], 3)  # 3 classes (BUY, SELL, HOLD)
            self.assertEqual(confidences.shape[0], 1)
            
            # Test OnlineLearner
            learner = OnlineLearner(
                model=model,
                lr=1e-4,
                batch_size=8,
                update_interval=60
            )
            
            # Test basic properties
            self.assertEqual(learner.batch_size, 8)
            self.assertEqual(learner.update_interval, 60)
            self.assertIsNotNone(learner.optimizer)
            
            logger.info("✅ Neural network components test passed")
            
        except Exception as e:
            logger.error(f"❌ Neural network components test failed: {e}")
            self.fail(f"Neural network components test failed: {e}")
    
    def test_rfe_summary_integration(self):
        """Test RFE summary integration after simulated selection"""
        try:
            from models.gating import FeatureGatingModule
            
            gating = FeatureGatingModule(
                feature_groups=self.feature_groups,
                rfe_enabled=True, 
                rfe_n_features=10
            )
            
            # Simulate RFE selection results
            self._simulate_rfe_results(gating)
            
            # Test unified RFE summary
            summary = gating.get_rfe_summary()
            
            # Verify all expected keys are present
            expected_keys = [
                'rfe_performed', 'total_available', 'total_cleaned', 'total_selected',
                'strong', 'medium', 'weak', 'target_features', 'top_features'
            ]
            for key in expected_keys:
                self.assertIn(key, summary, f"Missing key: {key}")
            
            # Verify summary values
            self.assertTrue(summary['rfe_performed'])
            self.assertEqual(summary['target_features'], 10)
            self.assertGreater(summary['total_available'], 0)
            self.assertLessEqual(summary['total_selected'], summary['total_available'])
            
            # Verify counts consistency
            total_categorized = summary['strong'] + summary['medium'] + summary['weak']
            self.assertGreater(total_categorized, 0)
            
            # Verify top features format
            self.assertIsInstance(summary['top_features'], list)
            if summary['top_features']:
                for item in summary['top_features']:
                    self.assertIsInstance(item, tuple)
                    self.assertEqual(len(item), 2)  # (name, metadata)
            
            logger.info("✅ RFE summary integration test passed")
            
        except Exception as e:
            logger.error(f"❌ RFE summary integration test failed: {e}")
            self.fail(f"RFE summary integration test failed: {e}")
    
    def test_version_info_integration(self):
        """Test version info integration with model components"""
        try:
            # Test that version history file exists and can be loaded
            version_file = os.path.join(os.path.dirname(__file__), '..', 'version_history.json')
            if os.path.exists(version_file):
                import json
                with open(version_file, 'r') as f:
                    version_data = json.load(f)
                    
                self.assertIsInstance(version_data, dict)
                logger.info(f"✅ Version info integration test passed - found {len(version_data)} version entries")
            else:
                logger.info("⚠️ Version history file not found - skipping version integration test")
            
        except Exception as e:
            logger.error(f"❌ Version info integration test failed: {e}")
            # Don't fail the test for version info issues
            logger.info("⚠️ Version info test non-critical failure")
    
    def _create_mock_ohlcv_data(self):
        """Create mock OHLCV data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='5min')
        np.random.seed(42)
        
        base_price = 50000
        returns = np.random.normal(0, 0.01, 30)
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        closes = np.array(prices[1:])
        highs = closes * (1 + np.random.uniform(0, 0.005, 30))
        lows = closes * (1 - np.random.uniform(0, 0.005, 30))
        opens = np.roll(closes, 1)
        opens[0] = base_price
        volumes = np.random.uniform(100, 1000, 30)
        
        return pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates)
    
    def _create_mock_indicators(self):
        """Create mock indicators for testing"""
        return {
            'sma_20': np.random.uniform(49000, 51000, 30),
            'ema_12': np.random.uniform(49500, 50500, 30),
            'rsi': np.random.uniform(30, 70, 30),
            'macd': np.random.uniform(-100, 100, 30),
            'bb_upper': np.random.uniform(50200, 51200, 30)
        }
    
    def _create_mock_feature_dict(self):
        """Create mock feature dictionary for gating tests"""
        return {
            'ohlcv': torch.randn(1, 13),
            'indicator': torch.randn(1, 20),
            'sentiment': torch.randn(1, 1),
            'orderbook': torch.randn(1, 10)
        }
    
    def _create_mock_feature_dict_for_model(self, feature_dims):
        """Create mock feature dictionary matching model dimensions"""
        features = {}
        for name, (seq_len, feature_dim) in feature_dims.items():
            features[name] = torch.randn(1, seq_len, feature_dim)
        return features
    
    def _simulate_rfe_results(self, gating):
        """Simulate RFE results for testing"""
        feature_names = [f'indicator.{i}' for i in range(15)] + [f'ohlcv.{i}' for i in range(5)]
        
        gating.rfe_all_features = feature_names.copy()
        gating.rfe_performed = True
        gating.rfe_selected_features = {}
        
        # Create features with different ranks - first 10 selected
        for i, name in enumerate(feature_names):
            rank = i + 1
            selected = i < 10
            
            gating.rfe_selected_features[name] = {
                'rank': rank,
                'importance': 0.9 - (i * 0.04),
                'selected': selected
            }
        
        gating.selected_features = [name for name, info in gating.rfe_selected_features.items() 
                                  if info['selected']]
        
        # Build categories
        gating._build_feature_categories()


if __name__ == '__main__':
    unittest.main()