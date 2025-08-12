#!/usr/bin/env python3
"""
Step by step debug neural network tensor
"""
import torch
import sys
sys.path.append('/home/runner/work/Trade/Trade')

# Test the individual components
feature_dims = {'test': (3, 4)}  # seq_len=3, feature_dim=4
hidden_dim = 8

# Create projection layer
projection = torch.nn.Linear(4, hidden_dim)

# Test data
x = torch.randn(1, 3, 4)  # (batch, seq, features)
print(f"Input shape: {x.shape}")

# Test projection
batch_size, seq_len, feature_dim = x.shape
x_flat = x.view(-1, feature_dim)
print(f"Flattened shape: {x_flat.shape}")

projected_flat = projection(x_flat)
print(f"Projected flat shape: {projected_flat.shape}")

embedded = projected_flat.view(batch_size, seq_len, hidden_dim)
print(f"Final embedded shape: {embedded.shape}")

# Test position encoding
max_seq_len = 5
pe = torch.zeros(1, max_seq_len, hidden_dim)
print(f"Position encoding shape: {pe.shape}")

# Test adding position encoding
seq_len_actual = min(embedded.shape[1], pe.shape[1])
embedded_with_pe = embedded.clone()
embedded_with_pe[:, :seq_len_actual, :] += pe[:, :seq_len_actual, :]
print(f"After position encoding shape: {embedded_with_pe.shape}")

# Test stacking different sequence lengths
embedded1 = torch.randn(1, 5, hidden_dim)
embedded2 = torch.randn(1, 1, hidden_dim)
print(f"Embedded1 shape: {embedded1.shape}")
print(f"Embedded2 shape: {embedded2.shape}")

# Expand embedded2 to match embedded1
expanded2 = embedded2.expand(-1, 5, -1)
print(f"Expanded2 shape: {expanded2.shape}")

# Test stacking after expansion
try:
    stacked = torch.stack([embedded1, expanded2], dim=0)
    print(f"Stacked shape: {stacked.shape}")
    summed = stacked.sum(dim=0)
    print(f"Summed shape: {summed.shape}")
    print("✅ Stack and sum successful!")
except Exception as e:
    print(f"❌ Stack failed: {str(e)}")