#!/usr/bin/env python3
"""
Integration test to demonstrate the complete RFE filtering functionality with feature catalog.

This script shows how the feature catalog integration works end-to-end, including:
1. Loading the feature catalog from CSV
2. Filtering RFE candidates vs must-keep features  
3. Generating active feature masks
4. Logging pool statistics
"""

import numpy as np
import pandas as pd
from models.gating import FeatureGatingModule
from models.feature_catalog import get_feature_catalog, reset_feature_catalog
import logging

# Set up logging to see the integration in action
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

def main():
    print("🚀 Feature Catalog Integration Demo")
    print("=" * 50)
    
    # Reset catalog for clean demo
    reset_feature_catalog()
    
    # Create realistic feature groups
    feature_groups = {
        'ohlcv': 6,      # Open, High, Low, Close, Volume, Timestamp
        'indicator': 20,  # Various technical indicators  
        'sentiment': 1,   # Market sentiment
        'orderbook': 10   # Order book features
    }
    
    print(f"📊 Feature Groups: {feature_groups}")
    
    # Initialize gating module (this will load the feature catalog)
    gating = FeatureGatingModule(
        feature_groups=feature_groups,
        rfe_n_features=8  # Select 8 features via RFE
    )
    
    print(f"📂 Feature Catalog Summary: {gating.feature_catalog.get_summary()}")
    
    # Create realistic training data
    print("\n🔄 Creating training data...")
    np.random.seed(42)  # For reproducible results
    
    training_data = {
        'ohlcv': pd.DataFrame({
            'open': np.random.randn(200) + 50000,
            'high': np.random.randn(200) + 50100,
            'low': np.random.randn(200) + 49900,
            'close': np.random.randn(200) + 50000,
            'volume': np.random.randn(200) + 1000,
            'timestamp': pd.date_range('2023-01-01', periods=200, freq='5min')
        }),
        'indicator': pd.DataFrame({
            'SMA_5': np.random.randn(200),
            'SMA_20': np.random.randn(200),
            'RSI_14': np.random.randn(200),
            'EMA_10': np.random.randn(200),
            'MACD_12_26_9': np.random.randn(200),
            'ATR_14': np.random.randn(200),
            'Bollinger_20_x2.0': np.random.randn(200),
            'Volume_SMA_20': np.random.randn(200),
            'MFI_14': np.random.randn(200),
            'Stoch_%K_14': np.random.randn(200),
            **{f'custom_ind_{i}': np.random.randn(200) for i in range(10)}
        }),
        'sentiment': pd.DataFrame({
            'fear_greed_index': np.random.randn(200)
        }),
        'orderbook': pd.DataFrame({
            f'orderbook_feature_{i}': np.random.randn(200) for i in range(10)
        })
    }
    
    # Create diverse training labels
    training_labels = np.random.choice([0, 1, 2], size=200, p=[0.4, 0.3, 0.3])
    
    print(f"📈 Training data shapes: {[(k, v.shape) for k, v in training_data.items()]}")
    print(f"🎯 Label distribution: BUY={sum(training_labels==0)}, SELL={sum(training_labels==1)}, HOLD={sum(training_labels==2)}")
    
    # Perform RFE selection with catalog filtering
    print("\n🔍 Performing RFE with feature catalog filtering...")
    results = gating.perform_rfe_selection(training_data, training_labels)
    
    if results:
        print(f"✅ RFE completed successfully!")
        print(f"📊 Results summary:")
        print(f"   - Total features processed: {len(results)}")
        print(f"   - Selected features: {len([f for f in results.values() if f['selected']])}")
        print(f"   - Must-keep features: {gating.must_keep_count}")
        print(f"   - RFE candidate pool: {gating.pool_candidates}")
        
        # Show must-keep vs RFE-selected breakdown
        must_keep_selected = [name for name, info in results.items() 
                            if info['selected'] and info['rank'] == 0]
        rfe_selected = [name for name, info in results.items() 
                       if info['selected'] and info['rank'] > 0]
        
        print(f"\n🔒 Must-keep features ({len(must_keep_selected)}):")
        for feature in must_keep_selected:
            print(f"   - {feature}")
        
        print(f"\n⚡ RFE-selected features ({len(rfe_selected)}):")
        for feature in rfe_selected:
            rank = results[feature]['rank']
            importance = results[feature]['importance']
            print(f"   - {feature} (rank: {rank}, importance: {importance:.3f})")
    
    # Generate active feature masks
    print("\n🎭 Generating active feature masks...")
    active_masks = gating.build_active_feature_masks()
    
    if active_masks:
        print("✅ Active masks generated successfully!")
        for group_name, mask in active_masks.items():
            active_count = np.sum(mask)
            total_count = len(mask)
            print(f"   - {group_name}: {active_count}/{total_count} features active")
    
    # Get RFE summary with new metadata
    print("\n📋 RFE Summary with metadata:")
    summary = gating.get_rfe_summary()
    for key, value in summary.items():
        if key not in ['top_features']:  # Skip detailed top features for brevity
            print(f"   - {key}: {value}")
    
    print(f"\n🎉 Integration demo completed successfully!")
    print("=" * 50)
    

if __name__ == "__main__":
    main()