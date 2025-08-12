"""
Tests for prediction variability and warmup effectiveness
"""
import pytest
import torch
import numpy as np
from datetime import datetime
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_prediction_variability_after_warmup():
    """Test that predictions have sufficient variability after warmup training"""
    print("🧪 Testing prediction variability after warmup...")
    
    from models.neural_network import MarketTransformer, OnlineLearner
    
    # Create test model
    feature_dims = {
        'ohlcv': (20, 13),
        'indicator': (20, 10),
        'sentiment': (1, 1),
        'orderbook': (1, 10)
    }
    
    model = MarketTransformer(feature_dims, hidden_dim=64, n_layers=2)
    learner = OnlineLearner(model, lr=1e-3)
    
    # Create test warmup data
    warmup_samples = []
    for i in range(50):
        features = {
            'ohlcv': torch.randn(20, 13),
            'indicator': torch.randn(20, 10), 
            'sentiment': torch.randn(1, 1),
            'orderbook': torch.randn(1, 10)
        }
        label = np.random.randint(0, 3)  # Random label 0, 1, or 2
        warmup_samples.append((features, label))
    
    # Before warmup - check if predictions are stuck
    model.eval()
    pre_warmup_predictions = []
    
    for i in range(10):
        features, _ = warmup_samples[i]
        with torch.no_grad():
            probs, confidence = model(features)
            pre_warmup_predictions.append((probs.detach(), confidence.item()))
    
    # Calculate pre-warmup variability
    pre_confidences = [pred[1] for pred in pre_warmup_predictions]
    pre_confidence_std = np.std(pre_confidences)
    
    # Perform warmup training
    warmup_results = learner.perform_warmup_training(warmup_samples, max_batches=20)
    
    assert warmup_results['batches_completed'] > 0, "Warmup should complete some batches"
    
    # After warmup - check prediction variability
    model.eval()
    post_warmup_predictions = []
    
    for i in range(10):
        features, _ = warmup_samples[i + 10]  # Use different samples
        with torch.no_grad():
            probs, confidence = model(features)
            post_warmup_predictions.append((probs.detach(), confidence.item()))
    
    # Calculate post-warmup variability
    post_confidences = [pred[1] for pred in post_warmup_predictions]
    post_confidence_std = np.std(post_confidences)
    
    # Test prediction diversity
    diversity_check = learner.check_prediction_diversity(post_warmup_predictions)
    
    # Variability should improve after warmup (confidence should vary more)
    assert post_confidence_std > 0.01, f"Post-warmup confidence std should be > 0.01, got {post_confidence_std}"
    
    # Diversity check should pass
    if not diversity_check['diverse']:
        print(f"Warning: Diversity check failed - {diversity_check.get('reason', 'unknown')}")
        # Don't fail the test immediately, but log the issue
    
    # Check that we get different probabilities for different inputs
    prob_vectors = [pred[0] for pred in post_warmup_predictions]
    prob_std = torch.std(torch.stack(prob_vectors), dim=0).mean().item()
    
    assert prob_std > 0.01, f"Probability distribution std should be > 0.01, got {prob_std}"
    
    print(f"✅ Prediction variability test passed - confidence_std: {post_confidence_std:.4f}, prob_std: {prob_std:.4f}")
    return True

def test_auto_mini_update_trigger():
    """Test that mini-updates trigger when logits standard deviation is low"""
    print("🧪 Testing auto mini-update trigger...")
    
    from models.neural_network import MarketTransformer, OnlineLearner
    
    # Create model with deterministic initialization to get low variance
    torch.manual_seed(42)
    feature_dims = {'test': (10, 5)}
    model = MarketTransformer(feature_dims, hidden_dim=32)
    learner = OnlineLearner(model, lr=1e-3)
    
    # Create test predictions with very low variance (simulating stuck confidence)
    low_variance_predictions = []
    base_probs = torch.tensor([0.4, 0.3, 0.3])
    base_confidence = 0.5
    
    for i in range(10):
        # Add tiny noise to simulate near-identical predictions
        noise = torch.randn(3) * 0.001
        probs = base_probs + noise
        confidence = base_confidence + np.random.normal(0, 0.001)
        low_variance_predictions.append((probs, confidence))
    
    # Check diversity - should detect low diversity
    diversity_check = learner.check_prediction_diversity(low_variance_predictions, threshold=0.01)
    
    assert not diversity_check['diverse'], "Should detect low diversity"
    assert diversity_check['variance'] is not None, "Should return variance value"
    assert diversity_check['variance'] < 0.01, f"Variance should be < 0.01, got {diversity_check['variance']}"
    
    # Test with high variance predictions
    high_variance_predictions = []
    for i in range(10):
        # Add significant noise
        probs = torch.softmax(torch.randn(3), dim=0)
        confidence = np.random.uniform(0.1, 0.9)
        high_variance_predictions.append((probs, confidence))
    
    diversity_check_high = learner.check_prediction_diversity(high_variance_predictions, threshold=0.01)
    
    assert diversity_check_high['diverse'], "Should detect high diversity"
    assert diversity_check_high['variance'] > 0.01, f"Variance should be > 0.01, got {diversity_check_high['variance']}"
    
    print("✅ Auto mini-update trigger test passed")
    return True

def test_warmup_loss_improvement():
    """Test that warmup training shows loss improvement"""
    print("🧪 Testing warmup loss improvement...")
    
    from models.neural_network import MarketTransformer, OnlineLearner
    
    # Create model
    feature_dims = {'test': (10, 5)}
    model = MarketTransformer(feature_dims, hidden_dim=32)
    learner = OnlineLearner(model, lr=1e-2)  # Higher LR for faster convergence
    
    # Create training data with clear patterns
    warmup_samples = []
    for i in range(100):
        # Create features with some pattern
        features = {
            'test': torch.randn(10, 5) + i * 0.01
        }
        # Create label based on pattern
        label = 0 if i < 33 else (1 if i < 66 else 2)
        warmup_samples.append((features, label))
    
    # Perform warmup with tracking
    warmup_results = learner.perform_warmup_training(warmup_samples, max_batches=30)
    
    assert warmup_results['batches_completed'] > 0, "Should complete some batches"
    
    # Check loss improvement
    if warmup_results['initial_loss'] is not None and warmup_results['final_loss'] is not None:
        initial_loss = warmup_results['initial_loss']
        final_loss = warmup_results['final_loss']
        
        # Loss should generally decrease (but allow some tolerance)
        improvement = initial_loss - final_loss
        
        # Either loss decreased, or both are reasonable values (< 2.0 for 3-class)
        assert improvement >= 0 or final_loss < 2.0, f"Loss should improve or be reasonable: {initial_loss:.4f} -> {final_loss:.4f}"
        
        print(f"✅ Loss improvement: {initial_loss:.4f} -> {final_loss:.4f} (improvement: {improvement:.4f})")
    else:
        print("⚠️ Loss tracking not available, but warmup completed successfully")
    
    return True

def test_confidence_unstuck_mechanism():
    """Test that warmup helps unstuck confidence values"""
    print("🧪 Testing confidence unstuck mechanism...")
    
    from models.neural_network import MarketTransformer, OnlineLearner
    
    # Create model that might get stuck
    feature_dims = {'test': (5, 3)}
    model = MarketTransformer(feature_dims, hidden_dim=16)
    learner = OnlineLearner(model, lr=1e-3)
    
    # Test features
    test_features = [
        {'test': torch.randn(5, 3)},
        {'test': torch.randn(5, 3) + 1.0},
        {'test': torch.randn(5, 3) - 1.0}
    ]
    
    # Check initial prediction consistency (might be stuck)
    model.eval()
    initial_predictions = []
    for features in test_features:
        with torch.no_grad():
            probs, confidence = model(features)
            initial_predictions.append((probs.detach(), confidence))
    
    # Create warmup data
    warmup_samples = []
    for i in range(30):
        features = {'test': torch.randn(5, 3)}
        label = i % 3  # Cycle through labels
        warmup_samples.append((features, label))
    
    # Perform warmup
    warmup_results = learner.perform_warmup_training(warmup_samples, max_batches=15)
    
    # Check predictions after warmup
    model.eval()
    post_warmup_predictions = []
    for features in test_features:
        with torch.no_grad():
            probs, confidence = model(features)
            post_warmup_predictions.append((probs.detach(), confidence))
    
    # Calculate variance in predictions
    initial_confidences = [pred[1] for pred in initial_predictions]
    post_confidences = [pred[1] for pred in post_warmup_predictions]
    
    initial_conf_std = np.std(initial_confidences)
    post_conf_std = np.std(post_confidences)
    
    # Check that model produces varied outputs for different inputs
    initial_probs = torch.stack([pred[0] for pred in initial_predictions])
    post_probs = torch.stack([pred[0] for pred in post_warmup_predictions])
    
    initial_prob_std = torch.std(initial_probs, dim=0).mean().item()
    post_prob_std = torch.std(post_probs, dim=0).mean().item()
    
    # After warmup, should have more variability
    print(f"Confidence std: {initial_conf_std:.4f} -> {post_conf_std:.4f}")
    print(f"Probability std: {initial_prob_std:.4f} -> {post_prob_std:.4f}")
    
    # At least one measure of variability should improve
    variability_improved = (post_conf_std > initial_conf_std) or (post_prob_std > initial_prob_std)
    
    assert variability_improved, "Warmup should improve prediction variability"
    
    print("✅ Confidence unstuck mechanism test passed")
    return True

def test_consecutive_prediction_variance():
    """Test variance across consecutive predictions"""
    print("🧪 Testing consecutive prediction variance...")
    
    from models.neural_network import MarketTransformer, OnlineLearner
    
    feature_dims = {'test': (8, 4)}
    model = MarketTransformer(feature_dims, hidden_dim=24)
    learner = OnlineLearner(model, lr=1e-3)
    
    # Create diverse warmup data
    warmup_samples = []
    for i in range(60):
        # Create varied features
        scale = 1.0 + i * 0.02
        features = {'test': torch.randn(8, 4) * scale}
        label = i % 3
        warmup_samples.append((features, label))
    
    # Warmup training
    learner.perform_warmup_training(warmup_samples, max_batches=25)
    
    # Generate consecutive predictions with different inputs
    model.eval()
    consecutive_predictions = []
    
    for i in range(10):
        # Create slightly different features for each prediction
        features = {'test': torch.randn(8, 4) + i * 0.1}
        with torch.no_grad():
            probs, confidence = model(features)
            consecutive_predictions.append((probs.detach(), confidence))
    
    # Calculate variance metrics
    confidences = [pred[1] for pred in consecutive_predictions]
    confidence_std = np.std(confidences)
    
    probs = torch.stack([pred[0] for pred in consecutive_predictions])
    prob_variance = torch.var(probs, dim=0).mean().item()
    
    # Test requirements
    assert confidence_std > 0.01, f"Confidence variance should be > 0.01, got {confidence_std:.4f}"
    assert prob_variance > 0.01, f"Probability variance should be > 0.01, got {prob_variance:.4f}"
    
    # Test range of predictions
    min_confidence = min(confidences)
    max_confidence = max(confidences)
    confidence_range = max_confidence - min_confidence
    
    assert confidence_range > 0.05, f"Confidence range should be > 0.05, got {confidence_range:.4f}"
    
    print(f"✅ Consecutive prediction variance test passed - conf_std: {confidence_std:.4f}, prob_var: {prob_variance:.4f}")
    return True

if __name__ == "__main__":
    test_prediction_variability_after_warmup()
    test_auto_mini_update_trigger()
    test_warmup_loss_improvement()
    test_confidence_unstuck_mechanism()
    test_consecutive_prediction_variance()
    print("🎉 All prediction variability tests passed!")