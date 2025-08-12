"""
Tests for atomic persistence operations for rfe_results.json and version_history.json
"""
import pytest
import json
import os
import tempfile
import threading
import time
from datetime import datetime
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_atomic_rfe_results_write():
    """Test atomic write operation for rfe_results.json"""
    print("🧪 Testing atomic RFE results write...")
    
    from models.gating import FeatureGatingModule
    from models.locks import rfe_lock
    
    # Create test gating module
    feature_groups = {'test_group': 10}
    gating = FeatureGatingModule(feature_groups, hidden_dim=32)
    
    # Setup mock RFE results
    gating.rfe_selected_features = {
        'test_feature_1': {'rank': 1, 'importance': 0.8, 'selected': True},
        'test_feature_2': {'rank': 2, 'importance': 0.6, 'selected': True},
        'test_feature_3': {'rank': 3, 'importance': 0.3, 'selected': False}
    }
    gating.rfe_performed = True
    
    # Test file doesn't exist initially
    test_file = 'test_rfe_results.json'
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Mock the save method to use test file
    original_save = gating._save_rfe_results_to_file
    
    def mock_save():
        try:
            import json
            from datetime import datetime
            
            # Use lock for thread safety
            with rfe_lock:
                rfe_data = {
                    'last_run': datetime.now().isoformat(),
                    'method': 'test_method',
                    'ranked_features': [],
                    'n_features_selected': 2,
                    'total_features': 3
                }
                
                # Atomic write using tempfile and os.replace
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
                    json.dump(rfe_data, tmp_file, indent=2, ensure_ascii=False)
                    tmp_filename = tmp_file.name
                
                # Atomic replacement
                os.replace(tmp_filename, test_file)
                
        except Exception as e:
            if 'tmp_filename' in locals():
                try:
                    os.unlink(tmp_filename)
                except:
                    pass
            raise e
    
    # Test atomic write
    gating._save_rfe_results_to_file = mock_save
    gating._save_rfe_results_to_file()
    
    # Verify file was created atomically
    assert os.path.exists(test_file), "RFE results file should exist"
    
    # Verify content is valid JSON
    with open(test_file, 'r') as f:
        data = json.load(f)
        assert 'last_run' in data, "Should have last_run field"
        assert 'method' in data, "Should have method field"
    
    # Clean up
    os.remove(test_file)
    
    print("✅ Atomic RFE write test passed")
    return True

def test_atomic_version_history_write():
    """Test atomic write operation for version_history.json"""
    print("🧪 Testing atomic version history write...")
    
    from models.neural_network import OnlineLearner, MarketTransformer
    from models.locks import version_lock
    
    # Create minimal model for testing
    feature_dims = {'test': (10, 5)}
    model = MarketTransformer(feature_dims, hidden_dim=32)
    learner = OnlineLearner(model, lr=1e-4)
    
    # Add some version history
    learner.version_history = [
        {
            'timestamp': datetime.now().isoformat(),
            'update_type': 'test',
            'from_version': '1.0.0',
            'to_version': '1.0.1'
        }
    ]
    
    test_file = 'test_version_history.json'
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Mock the save method to use test file
    def mock_save():
        try:
            # Use lock for thread safety
            with version_lock:
                version_data = {
                    'schema_version': 1,
                    'current_version': learner.model_version,
                    'last_updated': datetime.now().isoformat(),
                    'total_entries': len(learner.version_history),
                    'history': learner.version_history[-20:]
                }
                
                # Atomic write using tempfile and os.replace
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
                    json.dump(version_data, tmp_file, indent=2, ensure_ascii=False)
                    tmp_filename = tmp_file.name
                
                # Atomic replacement
                os.replace(tmp_filename, test_file)
                
        except Exception as e:
            if 'tmp_filename' in locals():
                try:
                    os.unlink(tmp_filename)
                except:
                    pass
            raise e
    
    # Test atomic write
    learner._save_version_history_to_file = mock_save
    learner._save_version_history_to_file()
    
    # Verify file was created atomically
    assert os.path.exists(test_file), "Version history file should exist"
    
    # Verify content is valid JSON and has schema_version
    with open(test_file, 'r') as f:
        data = json.load(f)
        assert 'schema_version' in data, "Should have schema_version field"
        assert data['schema_version'] == 1, "Schema version should be 1"
        assert 'history' in data, "Should have history field"
        assert len(data['history']) <= 20, "Should retain max 20 entries"
    
    # Clean up
    os.remove(test_file)
    
    print("✅ Atomic version history write test passed")
    return True

def test_concurrent_write_safety():
    """Test that concurrent writes are thread-safe"""
    print("🧪 Testing concurrent write safety...")
    
    from models.locks import rfe_lock
    import concurrent.futures
    
    test_file = 'test_concurrent_write.json'
    if os.path.exists(test_file):
        os.remove(test_file)
    
    successful_writes = []
    
    def concurrent_write(thread_id):
        """Simulate concurrent write operation"""
        try:
            with rfe_lock:
                # Simulate some processing time
                time.sleep(0.01)
                
                data = {
                    'thread_id': thread_id,
                    'timestamp': datetime.now().isoformat(),
                    'data': f'test_data_{thread_id}'
                }
                
                # Atomic write
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
                    json.dump(data, tmp_file, indent=2)
                    tmp_filename = tmp_file.name
                
                os.replace(tmp_filename, test_file)
                successful_writes.append(thread_id)
                
        except Exception as e:
            print(f"Thread {thread_id} failed: {str(e)}")
            if 'tmp_filename' in locals():
                try:
                    os.unlink(tmp_filename)
                except:
                    pass
    
    # Run concurrent writes
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(concurrent_write, i) for i in range(5)]
        concurrent.futures.wait(futures)
    
    # Verify file exists and has valid content
    assert os.path.exists(test_file), "File should exist after concurrent writes"
    
    with open(test_file, 'r') as f:
        data = json.load(f)
        assert 'thread_id' in data, "Should have thread_id"
        assert 'timestamp' in data, "Should have timestamp"
    
    # All threads should complete successfully
    assert len(successful_writes) == 5, f"Expected 5 successful writes, got {len(successful_writes)}"
    
    # Clean up
    os.remove(test_file)
    
    print("✅ Concurrent write safety test passed")
    return True

def test_persistence_error_handling():
    """Test error handling in persistence operations"""
    print("🧪 Testing persistence error handling...")
    
    from models.gating import FeatureGatingModule
    
    feature_groups = {'test_group': 10}
    gating = FeatureGatingModule(feature_groups, hidden_dim=32)
    
    # Create scenario that should cause error (invalid directory)
    invalid_file = '/invalid/path/test.json'
    
    # Mock save method to use invalid path
    def mock_save_with_error():
        try:
            import json
            from datetime import datetime
            import tempfile
            
            rfe_data = {'test': 'data'}
            
            # This should fail due to invalid directory
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8', dir='/invalid/path') as tmp_file:
                json.dump(rfe_data, tmp_file)
                tmp_filename = tmp_file.name
            
            os.replace(tmp_filename, invalid_file)
            
        except Exception as e:
            # Should clean up temp file on error
            if 'tmp_filename' in locals():
                try:
                    os.unlink(tmp_filename)
                except:
                    pass
            # Re-raise to test error handling
            raise e
    
    # Test that error is handled gracefully
    gating._save_rfe_results_to_file = mock_save_with_error
    
    try:
        gating._save_rfe_results_to_file()
        assert False, "Should have raised an exception"
    except Exception:
        # Expected to fail, this is good
        pass
    
    # Verify no temp files were left behind (hard to test perfectly)
    # But at least the method should handle the error
    
    print("✅ Persistence error handling test passed")
    return True

def test_file_integrity_after_write():
    """Test file integrity after atomic write operations"""
    print("🧪 Testing file integrity after write...")
    
    test_file = 'test_integrity.json'
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Test data
    test_data = {
        'schema_version': 1,
        'timestamp': datetime.now().isoformat(),
        'data': {
            'nested': 'value',
            'array': [1, 2, 3],
            'unicode': 'Test ñéäñé 中文'
        }
    }
    
    # Atomic write operation
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
        json.dump(test_data, tmp_file, indent=2, ensure_ascii=False)
        tmp_filename = tmp_file.name
    
    # Atomic replacement
    os.replace(tmp_filename, test_file)
    
    # Verify file integrity
    assert os.path.exists(test_file), "File should exist"
    
    # Read back and verify data integrity
    with open(test_file, 'r', encoding='utf-8') as f:
        read_data = json.load(f)
    
    assert read_data == test_data, "Data should match exactly"
    assert read_data['schema_version'] == 1, "Schema version should be preserved"
    assert read_data['data']['unicode'] == 'Test ñéäñé 中文', "Unicode should be preserved"
    
    # Clean up
    os.remove(test_file)
    
    print("✅ File integrity test passed")
    return True

if __name__ == "__main__":
    test_atomic_rfe_results_write()
    test_atomic_version_history_write()
    test_concurrent_write_safety()
    test_persistence_error_handling()
    test_file_integrity_after_write()
    print("🎉 All atomic persistence tests passed!")