#!/usr/bin/env python3
"""
Debug the actual forward method issue
"""
import torch
import sys
sys.path.append('/home/runner/work/Trade/Trade')

# Replicate the forward method logic step by step
feature_dims = {
    'test_seq': (5, 3),    # Sequential
    'test_single': (1, 2)  # Single timestep
}
hidden_dim = 16

# Create projections
projections = torch.nn.ModuleDict()
for name, (seq_len, dim) in feature_dims.items():
    projections[name] = torch.nn.Linear(dim, hidden_dim)

# Create test data
test_features = {
    'test_seq': torch.randn(1, 5, 3),
    'test_single': torch.randn(1, 1, 2)
}

print("Processing features...")
embedded_features = []

for name in test_features:
    x = test_features[name]
    expected_seq_len, expected_feature_dim = feature_dims[name]
    
    print(f"\nProcessing {name}: input shape {x.shape}")
    
    if name in ['test_seq']:  # Sequential
        batch_size, seq_len, feature_dim = x.shape
        x_flat = x.view(-1, feature_dim)
        projected_flat = projections[name](x_flat)
        embedded = projected_flat.view(batch_size, seq_len, hidden_dim)
    else:  # Non-sequential
        projected = projections[name](x)
        embedded = projected.unsqueeze(1)
    
    print(f"After processing {name}: shape {embedded.shape}")
    embedded_features.append(embedded)

# Now try the expansion logic
print(f"\nEmbedded features shapes: {[f.shape for f in embedded_features]}")

max_seq_len = max(feat.shape[1] for feat in embedded_features)
print(f"Max sequence length: {max_seq_len}")

# Expand all features to max sequence length
expanded_features = []
for i, feat in enumerate(embedded_features):
    print(f"\nExpanding feature {i}: {feat.shape}")
    if feat.shape[1] < max_seq_len:
        seq_diff = max_seq_len - feat.shape[1]
        last_timestep = feat[:, -1:, :]  # (batch, 1, hidden_dim)
        print(f"  Last timestep shape: {last_timestep.shape}")
        repeated = last_timestep.expand(-1, seq_diff, -1)  # Expand along sequence dim
        print(f"  Repeated shape: {repeated.shape}")
        feat_expanded = torch.cat([feat, repeated], dim=1)
        print(f"  Expanded shape: {feat_expanded.shape}")
    else:
        feat_expanded = feat
        print(f"  No expansion needed: {feat_expanded.shape}")
    expanded_features.append(feat_expanded)

print(f"\nExpanded features shapes: {[f.shape for f in expanded_features]}")

# Test stacking
try:
    stacked = torch.stack(expanded_features, dim=0)
    print(f"Stacked shape: {stacked.shape}")
    
    summed = stacked.sum(dim=0)
    print(f"Summed shape: {summed.shape}")
    print("✅ Success!")
except Exception as e:
    print(f"❌ Failed: {str(e)}")
    import traceback
    traceback.print_exc()