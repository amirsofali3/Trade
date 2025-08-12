"""
Tests for timestamp alignment and adaptive label generation
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import the main module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_timestamp_alignment():
    """Test timestamp alignment across feature groups and labels"""
    print("🧪 Testing timestamp alignment...")
    
    from main import TradingBot
    import json
    
    # Create a minimal config
    config = {
        'mysql': {'host': 'localhost', 'user': 'test', 'password': '', 'database': 'test'},
        'trading': {'symbols': ['BTCUSDT'], 'timeframes': ['5m']},
        'model': {'hidden_dim': 64}, 
        'rfe': {'enabled': True, 'lookahead': 3}
    }
    
    # Create bot instance (without full initialization)
    bot = TradingBot(config)
    
    # Create test data with timestamps
    timestamps = pd.date_range('2024-01-01', periods=50, freq='5min')
    
    # Create features by group with different timestamp coverage
    features_by_group = {
        'ohlcv': pd.DataFrame({
            'feature1': np.random.randn(45),
            'feature2': np.random.randn(45)
        }, index=timestamps[:45]),  # Missing last 5 timestamps
        
        'indicator': pd.DataFrame({
            'rsi': np.random.randn(48),
            'macd': np.random.randn(48)
        }, index=timestamps[2:]),  # Missing first 2 timestamps
    }
    
    # Create labels with timestamps
    label_series = [0, 1, 2] * 15  # 45 labels
    label_timestamps = timestamps[5:]  # Missing first 5 timestamps
    
    # Test alignment
    aligned_features, aligned_labels, common_timestamps = bot._align_on_timestamps(
        features_by_group, label_series, label_timestamps
    )
    
    # Verify alignment worked
    assert len(aligned_features) == 2, "Should have both feature groups"
    assert len(aligned_labels) > 0, "Should have aligned labels"
    assert len(common_timestamps) > 0, "Should have common timestamps"
    
    # Check that aligned data has consistent timestamps
    for group_data in aligned_features.values():
        if hasattr(group_data, 'index'):
            assert len(group_data) == len(common_timestamps), "Feature group should match common timestamps"
    
    assert len(aligned_labels) == len(common_timestamps), "Labels should match common timestamps"
    
    print(f"✅ Alignment test passed: {len(common_timestamps)} common timestamps")
    return True


def test_adaptive_labels():
    """Test adaptive label generation with different thresholds"""
    print("🧪 Testing adaptive label generation...")
    
    from main import TradingBot
    import json
    
    # Create a minimal config
    config = {
        'mysql': {'host': 'localhost', 'user': 'test', 'password': '', 'database': 'test'},
        'trading': {'symbols': ['BTCUSDT'], 'timeframes': ['5m']},
        'model': {'hidden_dim': 64},
        'rfe': {'enabled': True, 'lookahead': 3, 'initial_threshold': 0.002}
    }
    
    bot = TradingBot(config)
    
    # Create test OHLCV data with varying price movements
    timestamps = pd.date_range('2024-01-01', periods=100, freq='5min')
    prices = [100.0]
    
    # Create price series with different volatility patterns
    for i in range(99):
        # Create some strong moves and some sideways action
        if i % 20 == 0:
            change = np.random.choice([-0.005, 0.005])  # Strong move (±0.5%)
        elif i % 10 == 0:
            change = np.random.choice([-0.003, 0.003])  # Medium move (±0.3%)
        else:
            change = np.random.uniform(-0.001, 0.001)   # Small move (±0.1%)
        
        prices.append(prices[-1] * (1 + change))
    
    ohlcv_data = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * 1.001 for p in prices],
        'low': [p * 0.999 for p in prices], 
        'close': prices,
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Test adaptive labeling
    labels, label_timestamps, label_info = bot._create_adaptive_labels(
        ohlcv_data, lookahead=3, initial_threshold=0.002
    )
    
    # Verify label generation
    assert len(labels) > 0, "Should generate labels"
    assert len(labels) == len(label_timestamps), "Labels and timestamps should match"
    assert label_info['unique_classes'] > 1, "Should have multiple classes"
    
    # Check for proper class distribution
    label_counts = label_info['label_counts']
    total_labels = sum(label_counts.values())
    
    assert total_labels == len(labels), "Label counts should sum to total labels"
    
    # Test that lookahead prevents leakage (should have fewer labels than input data)
    assert len(labels) == len(ohlcv_data) - 3, "Should have lookahead samples fewer labels"
    
    # Test with more restrictive thresholds (should still create diverse labels)
    labels2, _, label_info2 = bot._create_adaptive_labels(
        ohlcv_data, lookahead=5, initial_threshold=0.001
    )
    
    assert label_info2['unique_classes'] > 1, "Should handle smaller thresholds"
    assert len(labels2) == len(ohlcv_data) - 5, "Should respect different lookahead"
    
    print(f"✅ Adaptive labels test passed: {label_info['unique_classes']} classes, {len(labels)} labels")
    return True


def test_label_diversity():
    """Test that adaptive thresholds guarantee label diversity"""
    print("🧪 Testing label diversity with extreme cases...")
    
    from main import TradingBot
    
    config = {
        'mysql': {'host': 'localhost', 'user': 'test', 'password': '', 'database': 'test'},
        'trading': {'symbols': ['BTCUSDT'], 'timeframes': ['5m']},
        'model': {'hidden_dim': 64},
        'rfe': {'enabled': True, 'lookahead': 3}
    }
    
    bot = TradingBot(config)
    
    # Test case 1: Flat prices (should trigger adaptive thresholds)
    timestamps = pd.date_range('2024-01-01', periods=50, freq='5min')
    flat_prices = [100.0] * 50  # No price movement
    
    flat_ohlcv = pd.DataFrame({
        'timestamp': timestamps,
        'open': flat_prices,
        'high': flat_prices,
        'low': flat_prices,
        'close': flat_prices,
        'volume': np.random.randint(1000, 5000, 50)
    })
    
    labels, _, label_info = bot._create_adaptive_labels(flat_ohlcv, lookahead=3)
    
    # Should still create multiple classes through adaptive thresholds
    assert label_info['unique_classes'] >= 2, "Should create multiple classes even with flat prices"
    
    # Test case 2: Only trending up (should trigger quantile-based approach)
    trend_prices = [100 + i * 0.1 for i in range(50)]  # Steady uptrend
    
    trend_ohlcv = pd.DataFrame({
        'timestamp': timestamps,
        'open': trend_prices,
        'high': [p * 1.001 for p in trend_prices],
        'low': [p * 0.999 for p in trend_prices],
        'close': trend_prices, 
        'volume': np.random.randint(1000, 5000, 50)
    })
    
    labels2, _, label_info2 = bot._create_adaptive_labels(trend_ohlcv, lookahead=3)
    
    # Should use quantile-based thresholds to create diversity
    assert label_info2['unique_classes'] >= 2, "Should handle trending data with quantile thresholds"
    
    print(f"✅ Label diversity test passed: flat={label_info['unique_classes']} classes, trend={label_info2['unique_classes']} classes")
    return True


if __name__ == "__main__":
    test_timestamp_alignment()
    test_adaptive_labels() 
    test_label_diversity()
    print("🎉 All alignment and labeling tests passed!")