#!/usr/bin/env python3
"""
Comprehensive test script to validate all the fixes implemented
"""
import os
import sys
import json
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
import logging

# Add the project root to the path
sys.path.append('/home/runner/work/Trade/Trade')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestFixes")

def test_rfe_pipeline():
    """Test RFE pipeline with length alignment and fallback"""
    logger.info("🧪 Testing RFE Pipeline Fixes...")
    
    try:
        from models.gating import FeatureGatingModule
        
        # Create test feature groups
        feature_groups = {
            'ohlcv': 13,
            'indicator': 100,
            'sentiment': 1,
            'orderbook': 42
        }
        
        # Initialize gating module with RFE
        gating = FeatureGatingModule(
            feature_groups, 
            rfe_enabled=True,
            rfe_n_features=15,
            min_weight=0.01
        )
        
        # Create test data with different lengths (simulating the original problem)
        training_data = {
            'ohlcv': np.random.randn(20, 13),     # Length 20
            'indicator': np.random.randn(500, 100), # Length 500 (original issue)
            'sentiment': np.array([0.5] * 50),    # Length 50
        }
        
        # Create diverse labels (fixing single-class issue)
        training_labels = [0] * 10 + [1] * 8 + [2] * 12  # Mixed BUY/SELL/HOLD
        
        logger.info(f"Test data lengths: {[(k, v.shape[0] if hasattr(v, 'shape') else len(v)) for k, v in training_data.items()]}")
        logger.info(f"Labels: {len(training_labels)}")
        
        # Perform RFE (should handle length alignment)
        results = gating.perform_rfe_selection(training_data, training_labels)
        
        if results:
            logger.info("✅ RFE Pipeline: SUCCESS - Length alignment and feature selection completed")
            
            # Check if external file was created
            if os.path.exists('rfe_results.json'):
                with open('rfe_results.json', 'r') as f:
                    saved_results = json.load(f)
                logger.info("✅ RFE External Persistence: SUCCESS - JSON file created")
                logger.info(f"   Method: {saved_results.get('method')}")
                logger.info(f"   Selected: {saved_results.get('weights_mapping', {})}")
            else:
                logger.warning("⚠️ RFE External Persistence: JSON file not found")
            
            # Test weight mapping
            rfe_weights = gating.get_rfe_weights()
            if rfe_weights:
                logger.info("✅ RFE Weight Mapping: SUCCESS")
                for group, weights in rfe_weights.items():
                    if isinstance(weights, torch.Tensor):
                        strong = torch.sum(weights >= 0.7).item()
                        medium = torch.sum((weights >= 0.3) & (weights < 0.7)).item()
                        weak = torch.sum(weights <= 0.01).item()
                        logger.info(f"   {group}: strong={strong}, medium={medium}, weak={weak}")
            else:
                logger.warning("⚠️ RFE Weight Mapping: No weights generated")
        else:
            logger.error("❌ RFE Pipeline: FAILED - No results returned")
            
        return bool(results)
        
    except Exception as e:
        logger.error(f"❌ RFE Pipeline: FAILED - {str(e)}")
        return False

def test_neural_network_projection():
    """Test neural network tensor dimension fixes"""
    logger.info("🧪 Testing Neural Network Projection Fixes...")
    
    try:
        from models.neural_network import MarketTransformer
        
        # Create feature dimensions with different input sizes (simulating the 128 vs 100 issue)
        feature_dims = {
            'ohlcv': (20, 13),
            'indicator': (20, 100),  # This would cause the dimension mismatch
            'sentiment': (1, 1),
            'orderbook': (1, 42)
        }
        
        # Create model with projections
        model = MarketTransformer(
            feature_dims=feature_dims,
            hidden_dim=128,
            n_layers=2,
            n_heads=4
        )
        
        # Create test features with different dimensions
        test_features = {
            'ohlcv': torch.randn(1, 20, 13),
            'indicator': torch.randn(1, 20, 100),
            'sentiment': torch.randn(1, 1, 1),
            'orderbook': torch.randn(1, 1, 42)
        }
        
        logger.info("Input tensor shapes:")
        for name, tensor in test_features.items():
            logger.info(f"   {name}: {tensor.shape}")
        
        # Test forward pass (should not fail with dimension errors)
        model.eval()
        with torch.no_grad():
            probs, confidence = model(test_features)
        
        logger.info("✅ Neural Network Projection: SUCCESS")
        logger.info(f"   Output shapes: probs={probs.shape}, confidence={confidence.shape}")
        logger.info(f"   Sample output: {probs[0].tolist()[:3]}...")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Neural Network Projection: FAILED - {str(e)}")
        return False

def test_encoder_fixes():
    """Test encoder fitting and deprecated fillna fixes"""
    logger.info("🧪 Testing Encoder Fixes...")
    
    try:
        from models.encoders.feature_encoder import OHLCVEncoder, IndicatorEncoder
        
        # Test OHLCV encoder auto-fitting
        ohlcv_encoder = OHLCVEncoder(window_size=20)
        
        # Create test OHLCV data
        test_ohlcv = pd.DataFrame({
            'open': np.random.randn(30) * 100 + 50000,
            'high': np.random.randn(30) * 100 + 50100,
            'low': np.random.randn(30) * 100 + 49900,
            'close': np.random.randn(30) * 100 + 50000,
            'volume': np.random.randn(30) * 1000 + 5000,
        })
        
        # Transform without fitting first (should auto-fit)
        ohlcv_features = ohlcv_encoder.transform(test_ohlcv)
        
        logger.info("✅ OHLCV Encoder Auto-fitting: SUCCESS")
        logger.info(f"   Output shape: {ohlcv_features.shape}")
        logger.info(f"   Encoder fitted: {ohlcv_encoder.is_fitted}")
        
        # Test Indicator encoder with new fillna approach
        indicator_encoder = IndicatorEncoder(window_size=20)
        
        # Create test indicator data with NaN values
        test_indicators = {
            'rsi': [50.0] * 15 + [np.nan] * 5 + [60.0] * 10,
            'macd': [0.0] * 10 + [np.nan] * 10 + [1.0] * 10,
            'sma': np.random.randn(30).tolist()
        }
        
        # Convert to pandas Series (simulating real indicator data)
        test_indicators_series = {
            name: pd.Series(values) for name, values in test_indicators.items()
        }
        
        # Transform (should handle NaN with new fillna method)
        indicator_features = indicator_encoder.transform(test_indicators_series)
        
        logger.info("✅ Indicator Encoder fillna Fix: SUCCESS")
        logger.info(f"   Output shape: {indicator_features.shape}")
        logger.info(f"   No NaN values: {not torch.isnan(indicator_features).any()}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Encoder Fixes: FAILED - {str(e)}")
        return False

def test_warmup_training():
    """Test warmup training functionality"""
    logger.info("🧪 Testing Warmup Training...")
    
    try:
        from models.neural_network import MarketTransformer, OnlineLearner
        
        # Create simple model
        feature_dims = {'ohlcv': (20, 13), 'indicator': (20, 10)}
        model = MarketTransformer(feature_dims=feature_dims, hidden_dim=64)
        
        # Create learner
        learner = OnlineLearner(
            model=model,
            lr=1e-3,
            batch_size=8,
            update_interval=60
        )
        
        # Create warmup data
        warmup_samples = []
        for i in range(20):
            features = {
                'ohlcv': torch.randn(1, 20, 13),
                'indicator': torch.randn(1, 20, 10)
            }
            label = i % 3  # Mix of 0, 1, 2
            warmup_samples.append((features, label))
        
        # Perform warmup training
        results = learner.perform_warmup_training(warmup_samples, max_batches=5)
        
        if results['batches_completed'] > 0:
            logger.info("✅ Warmup Training: SUCCESS")
            logger.info(f"   Batches completed: {results['batches_completed']}")
            logger.info(f"   Loss improvement: {results.get('loss_improvement', 0):.4f}")
        else:
            logger.warning("⚠️ Warmup Training: No batches completed")
        
        # Test prediction diversity check
        test_predictions = []
        model.eval()
        for _ in range(10):
            features = warmup_samples[0][0]  # Use first sample features
            with torch.no_grad():
                probs, confidence = model(features)
                test_predictions.append((probs, confidence))
        
        diversity = learner.check_prediction_diversity(test_predictions)
        logger.info(f"✅ Prediction Diversity Check: {diversity['diverse']}")
        logger.info(f"   Confidence std: {diversity.get('confidence_std', 0):.6f}")
        logger.info(f"   Logits std: {diversity.get('logits_std', 0):.6f}")
        
        return results['batches_completed'] > 0
        
    except Exception as e:
        logger.error(f"❌ Warmup Training: FAILED - {str(e)}")
        return False

def test_api_endpoints():
    """Test API endpoint functionality"""
    logger.info("🧪 Testing API Endpoints...")
    
    try:
        # Test that we can import app without errors
        from app import app as flask_app
        
        # Create test client
        with flask_app.test_client() as client:
            
            # Test model-stats endpoint
            response = client.get('/api/model-stats')
            if response.status_code == 200:
                data = response.get_json()
                logger.info("✅ /api/model-stats: SUCCESS")
                logger.info(f"   Response keys: {list(data.keys())}")
            else:
                logger.error(f"❌ /api/model-stats: FAILED - Status {response.status_code}")
                return False
            
            # Test feature-selection endpoint
            response = client.get('/api/feature-selection')
            if response.status_code == 200:
                data = response.get_json()
                logger.info("✅ /api/feature-selection: SUCCESS")
                logger.info(f"   RFE performed: {data.get('rfe_performed', False)}")
            else:
                logger.error(f"❌ /api/feature-selection: FAILED - Status {response.status_code}")
                return False
            
            # Test version-history endpoint
            response = client.get('/api/version-history')
            if response.status_code == 200:
                data = response.get_json()
                logger.info("✅ /api/version-history: SUCCESS")
                logger.info(f"   Current version: {data.get('current_version', 'unknown')}")
            else:
                logger.error(f"❌ /api/version-history: FAILED - Status {response.status_code}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API Endpoints: FAILED - {str(e)}")
        return False

def test_version_history_persistence():
    """Test version history file persistence"""
    logger.info("🧪 Testing Version History Persistence...")
    
    try:
        from models.neural_network import OnlineLearner, MarketTransformer
        
        # Create simple components
        feature_dims = {'test': (1, 1)}
        model = MarketTransformer(feature_dims=feature_dims, hidden_dim=32)
        learner = OnlineLearner(model=model, initial_version="1.0.0")
        
        # Test version increment
        learner._increment_version('patch')
        
        # Check if file was created
        if os.path.exists('version_history.json'):
            with open('version_history.json', 'r') as f:
                version_data = json.load(f)
            logger.info("✅ Version History Persistence: SUCCESS")
            logger.info(f"   Current version: {version_data.get('current_version')}")
            logger.info(f"   History entries: {version_data.get('total_entries')}")
        else:
            logger.warning("⚠️ Version History Persistence: File not created")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Version History Persistence: FAILED - {str(e)}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting Comprehensive Fix Testing...")
    logger.info("=" * 60)
    
    tests = [
        ("RFE Pipeline", test_rfe_pipeline),
        ("Neural Network Projection", test_neural_network_projection),
        ("Encoder Fixes", test_encoder_fixes),
        ("Warmup Training", test_warmup_training),
        ("API Endpoints", test_api_endpoints),
        ("Version History Persistence", test_version_history_persistence),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        logger.info("-" * 60)
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                failed += 1
                logger.error(f"❌ {test_name}: FAILED")
        except Exception as e:
            failed += 1
            logger.error(f"❌ {test_name}: FAILED - {str(e)}")
    
    logger.info("=" * 60)
    logger.info(f"🏁 Testing Complete: {passed} passed, {failed} failed")
    
    if passed == len(tests):
        logger.info("🎉 All tests passed! The fixes are working correctly.")
    else:
        logger.warning(f"⚠️ {failed} tests failed. Some fixes need more work.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)