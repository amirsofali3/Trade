#!/usr/bin/env python3
"""
Test version history persistence functionality
"""

import sys
import os
import json
import torch
import torch.nn as nn
import logging
from datetime import datetime
from models.neural_network import OnlineLearner
from models.neural_network import MarketTransformer
from models.gating import FeatureGatingModule

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestVersionHistory")

def test_version_history_persistence():
    """Test that version history is saved to external JSON file"""
    
    print("🧪 Testing version history persistence...")
    
    # Clean up any existing file
    version_file = 'version_history.json'
    if os.path.exists(version_file):
        os.remove(version_file)
    
    # Create a simple mock model and gating
    feature_dims = {'test': (20, 5)}  # (seq_len, feature_dim)
    mock_model = MarketTransformer(
        feature_dims=feature_dims,
        hidden_dim=64,
        n_layers=2,
        n_heads=4
    )
    
    gating = FeatureGatingModule({'test': 5})  # Just feature_dim for gating
    
    # Create learner
    learner = OnlineLearner(
        model=mock_model,
        gating_module=gating,
        lr=0.001,
        save_dir='test_models'
    )
    
    # Test initial version
    assert learner.model_version == '1.0.0', f"Initial version should be 1.0.0, got {learner.model_version}"
    
    # Trigger version increments
    learner._increment_version('patch')
    assert learner.model_version == '1.0.1', f"After patch increment should be 1.0.1, got {learner.model_version}"
    
    learner._increment_version('minor')
    assert learner.model_version == '1.1.0', f"After minor increment should be 1.1.0, got {learner.model_version}"
    
    learner._increment_version('major')
    assert learner.model_version == '2.0.0', f"After major increment should be 2.0.0, got {learner.model_version}"
    
    # Check that file was created
    assert os.path.exists(version_file), "Version history file should be created"
    
    # Check file content
    with open(version_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    assert 'current_version' in data, "Should have current_version"
    assert 'history' in data, "Should have history"
    assert data['current_version'] == '2.0.0', f"Current version should be 2.0.0, got {data['current_version']}"
    
    # Check history entries
    history = data['history']
    assert len(history) >= 3, f"Should have at least 3 history entries, got {len(history)}"
    
    # Check that entries have required fields
    for entry in history:
        assert 'from_version' in entry, "History entry should have from_version"
        assert 'to_version' in entry, "History entry should have to_version"
        assert 'timestamp' in entry, "History entry should have timestamp"
        assert 'update_type' in entry, "History entry should have update_type"
    
    # Test version increments are correct
    last_entry = history[-1]
    assert last_entry['update_type'] == 'major', f"Last update should be major, got {last_entry['update_type']}"
    assert last_entry['to_version'] == '2.0.0', f"Last update to_version should be 2.0.0, got {last_entry['to_version']}"
    
    # Clean up
    os.remove(version_file)
    if os.path.exists('test_models'):
        import shutil
        shutil.rmtree('test_models')
    
    print("✅ Version history persistence works correctly")
    return True

def test_version_history_merge():
    """Test that version history merges with existing file"""
    
    print("🧪 Testing version history file merging...")
    
    version_file = 'version_history.json'
    
    # Create initial file with some history
    initial_data = {
        'current_version': '1.2.0',
        'history': [
            {
                'from_version': '1.0.0',
                'to_version': '1.1.0', 
                'timestamp': '2024-01-01T10:00:00',
                'update_type': 'minor',
                'updates_count': 100
            },
            {
                'from_version': '1.1.0',
                'to_version': '1.2.0',
                'timestamp': '2024-01-02T10:00:00', 
                'update_type': 'minor',
                'updates_count': 200
            }
        ]
    }
    
    with open(version_file, 'w', encoding='utf-8') as f:
        json.dump(initial_data, f, indent=2)
    
    # Create learner and add more version history
    feature_dims = {'test': (20, 3)}  # (seq_len, feature_dim)
    mock_model = MarketTransformer(
        feature_dims=feature_dims,
        hidden_dim=32,
        n_layers=1,
        n_heads=2
    )
    
    gating = FeatureGatingModule({'test': 3})  # Just feature_dim for gating
    learner = OnlineLearner(mock_model, gating, save_dir='test_models')
    
    # Set current version higher
    learner.model_version = '1.2.0'
    
    # Add new version increment
    learner._increment_version('patch')
    
    # Read merged file
    with open(version_file, 'r', encoding='utf-8') as f:
        merged_data = json.load(f)
    
    # Should have original entries plus new one
    assert len(merged_data['history']) >= 3, f"Should have at least 3 entries after merge, got {len(merged_data['history'])}"
    assert merged_data['current_version'] == '1.2.1', f"Current version should be 1.2.1, got {merged_data['current_version']}"
    
    # Clean up
    os.remove(version_file)
    if os.path.exists('test_models'):
        import shutil
        shutil.rmtree('test_models')
    
    print("✅ Version history merging works correctly")
    return True

def test_version_history_api():
    """Test version history API endpoint"""
    
    print("🧪 Testing version history API...")
    
    from app import app, bot_instance
    
    # Test without bot instance
    with app.test_client() as client:
        response = client.get('/api/version-history')
        assert response.status_code == 200, f"API should return 200, got {response.status_code}"
        
        data = response.get_json()
        assert 'current_version' in data, "API response should have current_version"
        assert 'history' in data, "API response should have history"
    
    print("✅ Version history API works correctly")
    return True

def run_all_tests():
    """Run all version history tests"""
    
    print("🚀 Starting version history tests...\n")
    
    tests = [
        test_version_history_persistence,
        test_version_history_merge,
        test_version_history_api
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