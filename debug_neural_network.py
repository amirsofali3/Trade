#!/usr/bin/env python3
"""
Debug neural network tensor dimension issue
"""
import torch
import sys
sys.path.append('/home/runner/work/Trade/Trade')

from models.neural_network import MarketTransformer

# Create simple test
feature_dims = {
    'test_seq': (5, 3),    # Sequential
    'test_single': (1, 2)  # Single timestep
}

model = MarketTransformer(feature_dims=feature_dims, hidden_dim=16)

# Create test data
test_features = {
    'test_seq': torch.randn(1, 5, 3),
    'test_single': torch.randn(1, 1, 2)
}

print("Input shapes:")
for name, tensor in test_features.items():
    print(f"  {name}: {tensor.shape}")

# Try forward pass
try:
    model.eval()
    with torch.no_grad():
        probs, confidence = model(test_features)
    print("✅ Forward pass successful!")
    print(f"Output shapes: probs={probs.shape}, confidence={confidence.shape}")
except Exception as e:
    print(f"❌ Forward pass failed: {str(e)}")
    import traceback
    traceback.print_exc()