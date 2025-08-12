#!/usr/bin/env python3
"""
Test RFE fallback mechanism when insufficient data is available
"""

import sys
import numpy as np
import torch
import logging
from models.gating import FeatureGatingModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestRFEFallback")

def test_rfe_fallback_correlation():
    """Test that RFE falls back to correlation-based selection when data is limited"""
    
    print("🧪 Testing RFE fallback mechanism...")
    
    # Setup feature groups
    feature_groups = {
        'ohlcv': 5,
        'indicator': 10,
        'sentiment': 1
    }
    
    # Initialize gating module with RFE enabled
    gating = FeatureGatingModule(
        feature_groups=feature_groups,
        rfe_enabled=True,
        rfe_n_features=5
    )
    
    # Create limited training data (less than 10 samples to trigger fallback)
    n_samples = 8
    training_data = {
        'indicator': np.random.randn(n_samples, 10),
        'ohlcv': np.random.randn(n_samples, 5),
        'sentiment': np.random.randn(n_samples, 1)
    }
    
    # Create training labels
    training_labels = np.random.randint(0, 3, n_samples)
    
    print(f"📊 Training data: {n_samples} samples (triggering fallback threshold)")
    
    # Perform RFE - should trigger fallback
    results = gating.perform_rfe_selection(training_data, training_labels)
    
    # Verify fallback was used
    assert gating.rfe_performed, "RFE should be marked as performed even with fallback"
    assert len(results) > 0, "Should have results from fallback method"
    
    # Check that some features were selected
    selected_count = sum(1 for f in results.values() if f['selected'])
    print(f"✅ Fallback selected {selected_count} features")
    
    # Verify logging shows fallback was used
    # (This would need to be tested by capturing log output in a real implementation)
    
    return True

def test_rfe_weight_mapping():
    """Test that RFE results are properly mapped to weight categories"""
    
    print("🧪 Testing RFE weight mapping...")
    
    feature_groups = {
        'indicator': 20
    }
    
    gating = FeatureGatingModule(
        feature_groups=feature_groups,
        rfe_enabled=True,
        rfe_n_features=10
    )
    
    # Manually set up some RFE results to test weight mapping
    gating.rfe_performed = True
    gating.rfe_selected_features = {}
    
    # Create mock features with different importance levels using real indicator names
    indicator_names = [
        'sma', 'ema', 'rsi', 'macd', 'macd_signal', 'macd_hist',
        'stoch_k', 'stoch_d', 'bbands_upper', 'bbands_middle', 'bbands_lower',
        'adx', 'atr', 'supertrend', 'willr', 'mfi', 'obv', 'ad', 'vwap', 'engulfing'
    ]
    
    for i in range(20):
        indicator_name = indicator_names[i] if i < len(indicator_names) else f'indicator_{i}'
        importance = 0.9 if i < 5 else (0.5 if i < 10 else 0.01)  # Strong, medium, weak
        selected = i < 10  # First 10 are selected
        
        gating.rfe_selected_features[f'indicator.{indicator_name}'] = {
            'rank': i + 1,
            'importance': importance,
            'selected': selected
        }
    
    # Get RFE weights
    weights = gating.get_rfe_weights()
    
    assert 'indicator' in weights, "Should have indicator weights"
    
    indicator_weights = weights['indicator']
    
    # Check strong features (first 5)
    strong_weights = indicator_weights[:5]
    assert torch.all(strong_weights >= 0.7), f"Strong features should have weights >= 0.7, got {strong_weights}"
    
    # Check unselected features (last 10) should have min weight
    weak_weights = indicator_weights[10:]
    min_weight = gating.min_weight
    assert torch.all(torch.abs(weak_weights - min_weight) < 0.01), f"Weak features should have min weight {min_weight}, got {weak_weights}"
    
    print("✅ Weight mapping works correctly")
    return True

def test_rfe_persistence():
    """Test that RFE results are saved to JSON file"""
    
    print("🧪 Testing RFE persistence...")
    
    import os
    import json
    
    feature_groups = {'indicator': 5}
    gating = FeatureGatingModule(feature_groups, rfe_enabled=True)
    
    # Mock RFE results
    gating.rfe_performed = True
    gating.rfe_selected_features = {
        'indicator.rsi': {'rank': 1, 'importance': 0.8, 'selected': True},
        'indicator.macd': {'rank': 2, 'importance': 0.6, 'selected': True},
        'indicator.sma': {'rank': 3, 'importance': 0.2, 'selected': False}
    }
    gating.rfe_feature_rankings = {
        'indicator.rsi': 1,
        'indicator.macd': 2, 
        'indicator.sma': 3
    }
    
    # Save to file
    gating._save_rfe_results_to_file()
    
    # Check file exists
    assert os.path.exists('rfe_results.json'), "RFE results file should be created"
    
    # Check file content
    with open('rfe_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert 'timestamp' in data, "Should have timestamp"
    assert 'method' in data, "Should have method"
    assert 'selected_features' in data, "Should have selected features"
    assert data['n_features_selected'] == 2, "Should show 2 selected features"
    
    # Clean up
    os.remove('rfe_results.json')
    
    print("✅ RFE persistence works correctly")
    return True

def run_all_tests():
    """Run all RFE fallback tests"""
    
    print("🚀 Starting RFE fallback tests...\n")
    
    tests = [
        test_rfe_fallback_correlation,
        test_rfe_weight_mapping,
        test_rfe_persistence
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_func.__name__} PASSED\n")
            else:
                print(f"❌ {test_func.__name__} FAILED\n")
        except Exception as e:
            print(f"❌ {test_func.__name__} FAILED: {str(e)}\n")
    
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)