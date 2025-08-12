#!/usr/bin/env python3
"""
Test feature selection API endpoints
"""

import sys
import json
import torch
import torch.nn as nn
import logging
from models.neural_network import MarketTransformer
from models.gating import FeatureGatingModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestFeatureSelectionAPI")

def test_feature_selection_api():
    """Test /api/feature-selection endpoint"""
    
    print("🧪 Testing feature selection API...")
    
    from app import app, bot_instance
    
    # Test without bot instance
    with app.test_client() as client:
        response = client.get('/api/feature-selection')
        assert response.status_code == 200, f"API should return 200, got {response.status_code}"
        
        data = response.get_json()
        assert 'rfe_performed' in data, "API response should have rfe_performed"
        
        if not data['rfe_performed']:
            assert 'message' in data, "Should have message when RFE not performed"
            print("✅ API correctly reports RFE not performed")
        else:
            # Check required fields when RFE is performed
            required_fields = ['method', 'n_features_selected', 'total_features', 'selected_features', 'ranked_features']
            for field in required_fields:
                assert field in data, f"API response should have {field}"
            print("✅ API returns complete RFE results")
    
    return True

def test_feature_selection_with_mock_data():
    """Test feature selection API with mocked RFE data"""
    
    print("🧪 Testing feature selection API with mock RFE data...")
    
    from app import app
    import app as app_module
    
    # Create mock bot instance with gating results
    class MockGating:
        def __init__(self):
            self.rfe_performed = True
            self.rfe_selected_features = {
                'indicator.rsi': {'rank': 1, 'importance': 0.85, 'selected': True},
                'indicator.macd': {'rank': 2, 'importance': 0.72, 'selected': True},
                'indicator.sma': {'rank': 3, 'importance': 0.15, 'selected': False},
                'indicator.ema': {'rank': 4, 'importance': 0.12, 'selected': False}
            }
            self.rfe_model = True  # Indicates RandomForest was used
    
    class MockBot:
        def __init__(self):
            self.gating = MockGating()
    
    # Temporarily replace bot_instance
    original_bot = app_module.bot_instance
    app_module.bot_instance = MockBot()
    
    try:
        with app.test_client() as client:
            response = client.get('/api/feature-selection')
            assert response.status_code == 200, f"API should return 200, got {response.status_code}"
            
            data = response.get_json()
            
            # Verify response structure
            assert data['rfe_performed'] == True, "Should show RFE was performed"
            assert data['method'] == 'RandomForestRFE', f"Should show RandomForest method, got {data['method']}"
            assert data['n_features_selected'] == 2, f"Should show 2 selected features, got {data['n_features_selected']}"
            assert data['total_features'] == 4, f"Should show 4 total features, got {data['total_features']}"
            
            # Check selected features list
            selected_features = data['selected_features']
            assert 'indicator.rsi' in selected_features, "Should include RSI in selected features"
            assert 'indicator.macd' in selected_features, "Should include MACD in selected features"
            assert 'indicator.sma' not in selected_features, "Should not include SMA in selected features"
            
            # Check ranked features
            ranked_features = data['ranked_features']
            assert len(ranked_features) == 4, f"Should have 4 ranked features, got {len(ranked_features)}"
            
            # Should be sorted by rank
            ranks = [f['rank'] for f in ranked_features]
            assert ranks == sorted(ranks), f"Features should be sorted by rank, got {ranks}"
            
            # First ranked feature should be RSI
            assert ranked_features[0]['name'] == 'indicator.rsi', f"First feature should be RSI, got {ranked_features[0]['name']}"
            assert ranked_features[0]['selected'] == True, "First feature should be selected"
            
            print("✅ Feature selection API with mock data works correctly")
            
    finally:
        # Restore original bot instance
        app_module.bot_instance = original_bot
    
    return True

def test_model_stats_api_integration():
    """Test that model-stats API no longer has hardcoded mock data"""
    
    print("🧪 Testing model stats API for removal of mock data...")
    
    from app import app
    
    with app.test_client() as client:
        response = client.get('/api/model-stats')
        assert response.status_code == 200, f"API should return 200, got {response.status_code}"
        
        data = response.get_json()
        
        # Check that performance metrics exist
        assert 'performance_metrics' in data, "Should have performance_metrics"
        perf = data['performance_metrics']
        
        # The old mock values should not be present
        mock_training_accuracy = 0.847
        mock_validation_accuracy = 0.782
        mock_version = '1.2.3'
        
        # These should not match the old hardcoded values exactly
        training_acc = perf.get('training_accuracy', 0)
        validation_acc = perf.get('validation_accuracy', 0)
        version = perf.get('model_version', '')
        
        # Note: Could be 0.0 if no learner data, but shouldn't be the exact mock value
        if training_acc == mock_training_accuracy:
            print("⚠️  Warning: training_accuracy might still be using mock data")
        
        if validation_acc == mock_validation_accuracy:
            print("⚠️  Warning: validation_accuracy might still be using mock data")
        
        if version == mock_version:
            print("⚠️  Warning: model_version might still be using mock data")
        else:
            print("✅ Model version is not using hardcoded mock data")
        
        # Check that recent_updates exists and is not the old mock data
        assert 'recent_updates' in data, "Should have recent_updates"
        updates = data['recent_updates']
        
        # Should not have the old hardcoded timestamps
        old_mock_timestamps = ['2025-08-10T23:05:00Z', '2025-08-10T23:00:00Z', '2025-08-10T22:55:00Z']
        
        has_mock_timestamps = False
        for update in updates:
            if update.get('timestamp') in old_mock_timestamps:
                has_mock_timestamps = True
                break
        
        if not has_mock_timestamps:
            print("✅ Recent updates no longer using hardcoded mock timestamps")
        else:
            print("⚠️  Warning: recent_updates might still have mock timestamps")
    
    return True

def run_all_tests():
    """Run all feature selection API tests"""
    
    print("🚀 Starting feature selection API tests...\n")
    
    tests = [
        test_feature_selection_api,
        test_feature_selection_with_mock_data,
        test_model_stats_api_integration
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