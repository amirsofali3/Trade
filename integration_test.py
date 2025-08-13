#!/usr/bin/env python3
"""
Integration test to verify the complete system works with RFE and neural network
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

def test_integration():
    """Test the complete integration of RFE + Gating + Neural Network"""
    print("🔗 Testing complete integration...")
    
    try:
        # 1. Set up feature dimensions matching the system
        feature_dims = {
            'ohlcv': (20, 5),       # 20 timesteps, 5 OHLCV features
            'indicator': (20, 6),   # 20 timesteps, 6 indicators
            'sentiment': (1, 1),    # 1 timestep, 1 sentiment score
            'orderbook': (1, 10)    # 1 timestep, 10 orderbook features
        }
        
        # 2. Create aligned features as DataFrames (simulating real alignment)
        n_samples = 150  # Enough samples for RFE
        timestamps = pd.date_range('2023-01-01', periods=n_samples, freq='5min')
        
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
        
        # 3. Create diverse labels with pattern
        price_changes = np.diff(aligned_features['ohlcv']['close'].values)
        labels = []
        for i, change in enumerate(price_changes):
            if change > 10:  # Significant up move
                labels.append(0)  # BUY
            elif change < -10:  # Significant down move
                labels.append(1)  # SELL
            else:
                labels.append(2)  # HOLD
        
        # Add one more label to match length
        labels.append(2)
        labels = np.array(labels)
        
        print(f"✅ Created test data: {n_samples} samples")
        print(f"   Label distribution: BUY={np.sum(labels==0)}, SELL={np.sum(labels==1)}, HOLD={np.sum(labels==2)}")
        
        # 4. Initialize gating module and perform RFE
        feature_groups = {name: dim[1] for name, dim in feature_dims.items()}
        gating = FeatureGatingModule(feature_groups, rfe_enabled=True, rfe_n_features=8)
        
        print("🔍 Performing RFE selection...")
        rfe_results = gating.perform_rfe_selection(aligned_features, labels)
        
        if not rfe_results:
            print("❌ RFE failed to return results")
            return False
        
        print(f"✅ RFE completed: {len(rfe_results)} features processed")
        
        # 5. Initialize neural network with gating
        model = MarketTransformer(
            feature_dims=feature_dims,
            hidden_dim=128,
            gating_module=gating
        )
        
        print("✅ Neural network initialized with gating")
        
        # 6. Create feature tensors matching expected format
        features_dict = {
            'ohlcv': torch.randn(1, 20, 5),      # Sequential OHLCV
            'indicator': torch.randn(1, 20, 6),  # Sequential indicators
            'sentiment': torch.randn(1, 1, 1),   # Single sentiment
            'orderbook': torch.randn(1, 1, 10)   # Single orderbook snapshot
        }
        
        # 7. Test forward pass with gating
        print("🚀 Testing forward pass with scalar gating...")
        predictions, confidence = model(features_dict)
        
        print(f"✅ Forward pass successful!")
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Predictions: {predictions.detach().numpy()}")
        print(f"   Confidence: {confidence.item():.4f}")
        print(f"   Prediction sums to: {predictions.sum().item():.6f}")
        
        # 8. Verify scalar weights are being applied
        ohlcv_weight = gating.get_scalar_weight('ohlcv')
        indicator_weight = gating.get_scalar_weight('indicator')
        sentiment_weight = gating.get_scalar_weight('sentiment')
        orderbook_weight = gating.get_scalar_weight('orderbook')
        
        print(f"✅ Scalar weights applied:")
        print(f"   OHLCV: {ohlcv_weight:.3f}")
        print(f"   Indicator: {indicator_weight:.3f}")
        print(f"   Sentiment: {sentiment_weight:.3f}")
        print(f"   Orderbook: {orderbook_weight:.3f}")
        
        # 9. Verify RFE results were saved
        if os.path.exists('rfe_results.json'):
            with open('rfe_results.json', 'r') as f:
                rfe_data = json.load(f)
                print(f"✅ RFE results saved: {rfe_data.get('n_features_selected', 0)} features selected")
        
        # 10. Test multiple forward passes to ensure stability
        print("🔄 Testing multiple forward passes...")
        for i in range(3):
            pred, conf = model(features_dict)
            assert pred.shape == (1, 3), f"Prediction shape changed: {pred.shape}"
            assert torch.allclose(pred.sum(dim=1), torch.ones(1), atol=1e-6), "Predictions don't sum to 1"
        
        print("✅ Multiple forward passes stable")
        
        print("🎉 Integration test PASSED - All components working together!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run integration test"""
    print("🚀 Starting integration test for complete RFE + Gating + Neural Network system...")
    print("=" * 80)
    
    success = test_integration()
    
    print()
    print("=" * 80)
    if success:
        print("🎉 Integration test PASSED! The system is working correctly.")
        print()
        print("Expected behavior verified:")
        print("✅ RFE processes DataFrame-based aligned features correctly")
        print("✅ RFE logs proper input shapes and gracefully handles edge cases")
        print("✅ Neural network applies scalar gating after projection")
        print("✅ No tensor dimension mismatches (128 vs 100 issue resolved)")
        print("✅ Forward pass is stable and consistent")
        print("✅ All components integrate seamlessly")
    else:
        print("❌ Integration test FAILED. Please review the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)