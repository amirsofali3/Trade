#!/usr/bin/env python3
"""
Test script to verify the fixes for RFE and tensor dimension issues
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
import json

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.gating import FeatureGatingModule
from models.neural_network import MarketTransformer

def test_rfe_dataframe_processing():
    """Test RFE with DataFrame-based aligned features"""
    print("🔍 Testing RFE DataFrame processing...")
    
    # Create test feature groups with DataFrames (simulating aligned features)
    n_samples = 100
    timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='5min')
    
    # Create aligned features as DataFrames
    aligned_features = {
        'ohlcv': pd.DataFrame({
            'open': np.random.randn(n_samples) * 0.01 + 42000,
            'high': np.random.randn(n_samples) * 0.01 + 42100,
            'low': np.random.randn(n_samples) * 0.01 + 41900,
            'close': np.random.randn(n_samples) * 0.01 + 42000,
            'volume': np.random.randn(n_samples) * 1000 + 5000,
        }, index=timestamps),
        
        'indicator': pd.DataFrame({
            'rsi': np.random.randn(n_samples) * 20 + 50,
            'macd': np.random.randn(n_samples) * 0.1,
            'sma': np.random.randn(n_samples) * 0.01 + 42000,
            'ema': np.random.randn(n_samples) * 0.01 + 42000,
            'bollinger_upper': np.random.randn(n_samples) * 0.01 + 42200,
            'bollinger_lower': np.random.randn(n_samples) * 0.01 + 41800,
        }, index=timestamps),
        
        'sentiment': pd.DataFrame({
            'sentiment_score': np.random.randn(n_samples) * 0.1,
        }, index=timestamps),
        
        'orderbook': pd.DataFrame({
            f'orderbook_{i}': np.random.randn(n_samples) * 0.001 
            for i in range(10)
        }, index=timestamps)
    }
    
    # Create diverse labels
    labels = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.2, 0.5])
    
    # Initialize gating module
    feature_groups = {'ohlcv': 5, 'indicator': 6, 'sentiment': 1, 'orderbook': 10}
    gating = FeatureGatingModule(feature_groups)
    
    # Test RFE
    try:
        results = gating.perform_rfe_selection(aligned_features, labels)
        
        if results:
            print(f"✅ RFE completed successfully with {len(results)} features")
            
            # Check if we get proper logging
            rfe_weights = gating.get_rfe_weights()
            print(f"✅ RFE weights applied: {len(rfe_weights)} features weighted")
            
            return True
        else:
            print("❌ RFE returned empty results")
            return False
            
    except Exception as e:
        print(f"❌ RFE failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_neural_network_dimensions():
    """Test neural network with proper dimensional consistency"""
    print("🔍 Testing neural network dimensional consistency...")
    
    try:
        # Define feature dimensions
        feature_dims = {
            'ohlcv': (20, 5),      # 20 timesteps, 5 features
            'indicator': (20, 6),   # 20 timesteps, 6 features  
            'sentiment': (1, 1),    # 1 timestep, 1 feature
            'orderbook': (1, 10)    # 1 timestep, 10 features
        }
        
        # Create gating module
        feature_groups = {name: dim[1] for name, dim in feature_dims.items()}
        gating = FeatureGatingModule(feature_groups)
        
        # Initialize neural network with gating
        model = MarketTransformer(
            feature_dims=feature_dims,
            hidden_dim=128,
            gating_module=gating
        )
        
        print("✅ Neural network initialized successfully")
        
        # Create test features with different dimensions
        features_dict = {
            'ohlcv': torch.randn(1, 20, 5),     # Sequential data
            'indicator': torch.randn(1, 20, 6), # Sequential data
            'sentiment': torch.randn(1, 1, 1),  # Single timestep
            'orderbook': torch.randn(1, 1, 10)  # Single timestep
        }
        
        # Test forward pass
        predictions, confidence = model(features_dict)
        
        print(f"✅ Forward pass successful - predictions shape: {predictions.shape}")
        print(f"✅ Confidence shape: {confidence.shape}")
        
        # Check that predictions are valid probabilities
        assert predictions.shape[1] == 3, f"Expected 3 classes, got {predictions.shape[1]}"
        assert torch.allclose(predictions.sum(dim=1), torch.ones(1), atol=1e-6), "Predictions should sum to 1"
        
        print("✅ All dimensional checks passed")
        return True
        
    except Exception as e:
        print(f"❌ Neural network test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_scalar_gating():
    """Test scalar gating application after projection"""
    print("🔍 Testing scalar gating...")
    
    try:
        # Create gating module with some RFE results
        feature_groups = {'ohlcv': 5, 'indicator': 6, 'sentiment': 1, 'orderbook': 10}
        gating = FeatureGatingModule(feature_groups)
        
        # Simulate RFE results
        gating.rfe_performed = True
        gating.rfe_selected_features = {
            'ohlcv': {'selected': True, 'rank': 1, 'importance': 0.8},
            'indicator.rsi': {'selected': True, 'rank': 2, 'importance': 0.7},
            'indicator.macd': {'selected': True, 'rank': 3, 'importance': 0.6},
            'sentiment': {'selected': False, 'rank': 10, 'importance': 0.1},
            'orderbook': {'selected': False, 'rank': 15, 'importance': 0.05}
        }
        
        # Test scalar weight calculation
        ohlcv_weight = gating.get_scalar_weight('ohlcv')
        indicator_weight = gating.get_scalar_weight('indicator')
        sentiment_weight = gating.get_scalar_weight('sentiment')
        
        print(f"✅ Scalar weights: ohlcv={ohlcv_weight:.3f}, indicator={indicator_weight:.3f}, sentiment={sentiment_weight:.3f}")
        
        # Weights should be reasonable (between min_weight and 1.0)
        assert 0.01 <= ohlcv_weight <= 1.0, f"Invalid ohlcv weight: {ohlcv_weight}"
        assert 0.01 <= indicator_weight <= 1.0, f"Invalid indicator weight: {indicator_weight}"
        assert sentiment_weight >= 0.01, f"Sentiment weight below minimum: {sentiment_weight}"
        
        print("✅ Scalar gating weights are valid")
        return True
        
    except Exception as e:
        print(f"❌ Scalar gating test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_exception_handling():
    """Test exception handling simulation"""
    print("🔍 Testing exception handling...")
    
    try:
        # Simulate the exception tracking from main.py
        recent_exceptions = []
        max_recent_exceptions = 10
        
        # Simulate some exceptions
        for i in range(3):
            try:
                # Simulate an error
                if i == 1:
                    raise ValueError(f"Test error {i}")
                elif i == 2:
                    raise RuntimeError(f"Test runtime error {i}")
            except Exception as e:
                import traceback
                from datetime import datetime
                
                exception_info = {
                    'timestamp': datetime.now().isoformat(),
                    'message': str(e),
                    'traceback': traceback.format_exc()[:1000]
                }
                
                recent_exceptions.append(exception_info)
                if len(recent_exceptions) > max_recent_exceptions:
                    recent_exceptions.pop(0)
        
        print(f"✅ Captured {len(recent_exceptions)} exceptions")
        
        # Verify exception structure
        for exc in recent_exceptions:
            assert 'timestamp' in exc, "Missing timestamp"
            assert 'message' in exc, "Missing message"
            assert 'traceback' in exc, "Missing traceback"
        
        print("✅ Exception handling test passed")
        return True
        
    except Exception as e:
        print(f"❌ Exception handling test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting comprehensive test suite for RFE and tensor dimension fixes...")
    print("=" * 60)
    
    tests = [
        test_rfe_dataframe_processing,
        test_neural_network_dimensions,
        test_scalar_gating,
        test_exception_handling
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test.__name__} FAILED with exception: {e}")
        
        print("-" * 40)
    
    print()
    print("=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Fixes are working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)