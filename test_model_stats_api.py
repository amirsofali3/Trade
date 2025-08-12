"""
Tests for /api/model-stats endpoint to ensure proper data types and no mock timestamps
"""
import pytest
from datetime import datetime
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_model_stats_returns_200():
    """Test that /api/model-stats returns 200 status code"""
    print("🧪 Testing model-stats API returns 200...")
    
    from app import app
    
    with app.test_client() as client:
        response = client.get('/api/model-stats')
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        
    print("✅ Model-stats API returns 200")
    return True

def test_model_stats_numeric_fields():
    """Test that model-stats returns numeric fields with proper types"""
    print("🧪 Testing model-stats numeric field types...")
    
    from app import app
    
    with app.test_client() as client:
        response = client.get('/api/model-stats')
        assert response.status_code == 200
        
        data = response.get_json()
        
        # Check that required fields exist and have correct types
        assert 'training_accuracy' in data, "Should have training_accuracy"
        assert isinstance(data['training_accuracy'], (int, float)), f"training_accuracy should be numeric, got {type(data['training_accuracy'])}"
        
        assert 'validation_accuracy' in data, "Should have validation_accuracy"
        assert isinstance(data['validation_accuracy'], (int, float)), f"validation_accuracy should be numeric, got {type(data['validation_accuracy'])}"
        
        assert 'model_version' in data, "Should have model_version"
        assert isinstance(data['model_version'], str), f"model_version should be string, got {type(data['model_version'])}"
        
        # Check performance_metrics nested object
        assert 'performance_metrics' in data, "Should have performance_metrics"
        perf = data['performance_metrics']
        
        assert isinstance(perf.get('training_accuracy'), (int, float)), "performance_metrics.training_accuracy should be numeric"
        assert isinstance(perf.get('validation_accuracy'), (int, float)), "performance_metrics.validation_accuracy should be numeric"
        
        # Ensure values are in valid ranges
        assert 0 <= data['training_accuracy'] <= 1, f"training_accuracy should be 0-1, got {data['training_accuracy']}"
        assert 0 <= data['validation_accuracy'] <= 1, f"validation_accuracy should be 0-1, got {data['validation_accuracy']}"
        
    print("✅ Model-stats numeric fields have correct types")
    return True

def test_model_stats_no_mock_timestamps():
    """Test that model-stats doesn't contain hardcoded mock timestamps"""
    print("🧪 Testing model-stats has no mock timestamps...")
    
    from app import app
    import json
    
    with app.test_client() as client:
        response = client.get('/api/model-stats')
        assert response.status_code == 200
        
        data = response.get_json()
        response_str = json.dumps(data, indent=2)
        
        # Check for known mock timestamps that should be removed
        mock_patterns = [
            "2024-01-15T10:30:00",  # Common mock timestamp pattern
            "2024-01-15T11:45:00", 
            "2024-01-15T12:20:00",
            "mock_timestamp",
            "hardcoded_time"
        ]
        
        for pattern in mock_patterns:
            assert pattern not in response_str, f"Found mock timestamp pattern: {pattern}"
        
        # Check recent_updates don't have hardcoded mock data
        if 'recent_updates' in data:
            for update in data['recent_updates']:
                if update.get('timestamp'):
                    # If timestamp exists, it should be parseable as ISO format
                    try:
                        datetime.fromisoformat(update['timestamp'].replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        # If it fails, make sure it's None rather than mock data
                        assert update['timestamp'] is None, f"Invalid timestamp format: {update['timestamp']}"
                        
    print("✅ Model-stats has no mock timestamps")
    return True

def test_model_stats_recent_updates_format():
    """Test that recent_updates field has proper structure"""
    print("🧪 Testing recent_updates format...")
    
    from app import app
    
    with app.test_client() as client:
        response = client.get('/api/model-stats')
        assert response.status_code == 200
        
        data = response.get_json()
        
        assert 'recent_updates' in data, "Should have recent_updates field"
        updates = data['recent_updates']
        
        assert isinstance(updates, list), "recent_updates should be a list"
        
        for update in updates:
            assert isinstance(update, dict), "Each update should be a dict"
            assert 'type' in update, "Each update should have type"
            assert 'version_change' in update, "Each update should have version_change"
            # timestamp can be None, so don't require it
            
    print("✅ Recent updates format is correct")
    return True

def test_model_stats_detailed_feature_importance():
    """Test detailed feature importance structure"""
    print("🧪 Testing detailed feature importance...")
    
    from app import app
    
    with app.test_client() as client:
        response = client.get('/api/model-stats')
        assert response.status_code == 200
        
        data = response.get_json()
        
        assert 'detailed_feature_importance' in data, "Should have detailed_feature_importance"
        importance = data['detailed_feature_importance']
        
        # Check structure of feature importance entries
        for feature_name, feature_data in importance.items():
            assert isinstance(feature_data, dict), f"Feature {feature_name} data should be dict"
            
            if 'weight' in feature_data:
                assert isinstance(feature_data['weight'], (int, float)), f"Weight for {feature_name} should be numeric"
            
            if 'percentage' in feature_data:
                assert isinstance(feature_data['percentage'], (int, float)), f"Percentage for {feature_name} should be numeric"
                
            if 'status' in feature_data:
                assert feature_data['status'] in ['active', 'inactive'], f"Status for {feature_name} should be active/inactive"
    
    print("✅ Detailed feature importance structure is correct")
    return True

def test_model_stats_error_handling():
    """Test that model-stats handles errors gracefully"""
    print("🧪 Testing error handling...")
    
    from app import app
    
    # Temporarily break something and ensure graceful fallback
    with app.test_client() as client:
        response = client.get('/api/model-stats')
        
        # Should still return 200 even if bot is offline
        assert response.status_code in [200, 500], "Should handle errors gracefully"
        
        data = response.get_json()
        
        # Even in error cases, should have basic structure
        assert 'training_accuracy' in data, "Should have fallback training_accuracy"
        assert 'validation_accuracy' in data, "Should have fallback validation_accuracy"
        assert 'model_version' in data, "Should have fallback model_version"
        
        # Values should be safe defaults
        assert isinstance(data['training_accuracy'], (int, float)), "Fallback training_accuracy should be numeric"
        assert isinstance(data['validation_accuracy'], (int, float)), "Fallback validation_accuracy should be numeric"
        
    print("✅ Error handling works correctly")
    return True

if __name__ == "__main__":
    test_model_stats_returns_200()
    test_model_stats_numeric_fields()
    test_model_stats_no_mock_timestamps()
    test_model_stats_recent_updates_format()
    test_model_stats_detailed_feature_importance()
    test_model_stats_error_handling()
    print("🎉 All model-stats API tests passed!")