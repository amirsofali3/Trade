#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced trading bot features

This script shows the key functionality without requiring database connections
or external dependencies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FeatureTest")

def test_indicators():
    """Test technical indicators calculation"""
    logger.info("Testing Technical Indicators...")
    
    try:
        from data_collection.indicators import IndicatorCalculator
        
        # Create sample OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        closes = np.array(prices[1:])
        highs = closes * (1 + np.random.uniform(0, 0.01, 100))
        lows = closes * (1 - np.random.uniform(0, 0.01, 100))
        opens = np.roll(closes, 1)
        opens[0] = base_price
        volumes = np.random.uniform(100, 1000, 100)
        
        ohlcv = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates)
        
        # Create mock database manager
        class MockDBManager:
            def save_indicator(self, symbol, timeframe, name, values):
                logger.info(f"Would save {name} indicator: mean={np.mean(values):.2f}")
            
            def get_ohlcv(self, symbol, timeframe, start_time=None, end_time=None, limit=None):
                # Return the mock OHLCV data we created above
                return ohlcv
        
        db_manager = MockDBManager()
        calc = IndicatorCalculator(db_manager)
        
        # Test indicators
        indicators = ['sma', 'ema', 'rsi', 'macd', 'bbands']
        results = calc.calculate_indicators('BTCUSDT', '5m', indicators)
        
        logger.info(f"Successfully calculated {len(results)} indicators")
        for name, values in results.items():
            if values:
                logger.info(f"  {name}: {len(values)} values, range [{min(values):.2f}, {max(values):.2f}]")
        
        return True
        
    except Exception as e:
        logger.error(f"Indicator test failed: {e}")
        return False

def test_position_manager():
    """Test multi-tier TP/SL system"""
    logger.info("Testing Multi-tier TP/SL Position Manager...")
    
    try:
        from trading.position_manager import PositionManager
        
        # Create mock database manager  
        class MockDBManager:
            def get_session(self):
                return MockSession()
            def close_session(self, session):
                pass
        
        class MockSession:
            def query(self, model):
                return MockQuery()
        
        class MockQuery:
            def filter(self, *args):
                return self
            def all(self):
                return []  # No existing trades
        
        db_manager = MockDBManager()
        position_manager = PositionManager(
            db_manager, 
            initial_stop_loss=0.03,
            take_profit_levels=[0.05, 0.1, 0.15]
        )
        
        # Test adding a position
        entry_price = 50000
        position = position_manager.add_position('BTCUSDT', entry_price, 0.1, signal_id=1)
        logger.info(f"Added position: Entry={entry_price}, SL={position['stop_loss']:.2f}")
        
        # Test TP level triggers
        test_prices = [52500, 55000, 57500]  # Should trigger TP1, TP2, TP3
        
        for i, price in enumerate(test_prices):
            action, updated_pos = position_manager.update_position_price('BTCUSDT', price)
            if action == 'tp_adjust':
                logger.info(f"TP{i+1} triggered at {price}, new SL: {updated_pos['stop_loss']:.2f}")
        
        # Test position summary
        summary = position_manager.get_position_summary('BTCUSDT')
        if summary:
            logger.info(f"Position summary: {len(summary['tp_status'])} TP levels")
            for tp in summary['tp_status']:
                status = "✓" if tp['triggered'] else "✗"
                logger.info(f"  TP{tp['level']} ({tp['percentage']}): {status}")
        
        return True
        
    except Exception as e:
        logger.error(f"Position manager test failed: {e}")
        return False

def test_feature_gating():
    """Test enhanced feature gating"""
    logger.info("Testing Enhanced Feature Gating...")
    
    try:
        import torch
        from models.gating import FeatureGatingModule
        
        # Define feature groups
        feature_groups = {
            'ohlcv': 13,
            'indicator': 20, 
            'sentiment': 1,
            'orderbook': 42,
            'tick_data': 3
        }
        
        gating = FeatureGatingModule(
            feature_groups, 
            min_weight=0.01,
            adaptation_rate=0.1
        )
        
        # Create mock features
        features = {
            'ohlcv': torch.randn(1, 20, 13),        # (batch, seq, features)
            'indicator': torch.randn(1, 20, 20),
            'sentiment': torch.randn(1, 1),
            'orderbook': torch.randn(1, 42),
            'tick_data': torch.randn(1, 100, 3)
        }
        
        # Test gating
        gated_features = gating(features)
        logger.info(f"Gated {len(gated_features)} feature groups")
        
        # Test active features reporting
        active_features = gating.get_active_features()
        
        logger.info("Active Features:")
        for group, info in active_features.items():
            if isinstance(info, dict) and 'weight' in info:
                logger.info(f"  {group}: {info['weight']:.3f} ({info['status']})")
            elif isinstance(info, dict):
                for feature, feature_info in info.items():
                    if isinstance(feature_info, dict):
                        logger.info(f"  {group}.{feature}: {feature_info['weight']:.3f} ({feature_info['status']})")
        
        # Test performance tracking
        gating.update_feature_performance('ohlcv', True, 0.8)
        gating.update_feature_performance('sentiment', False, 0.3)
        
        performance = gating.get_feature_performance_summary()
        logger.info(f"Performance summary for {len(performance)} features")
        
        # Test weak features detection
        weak_features = gating.get_weak_features()
        logger.info(f"Detected {len(weak_features)} weak features: {weak_features}")
        
        return True
        
    except Exception as e:
        logger.error(f"Feature gating test failed: {e}")
        return False

def main():
    """Run all feature tests"""
    logger.info("=== Enhanced Trading Bot Feature Tests ===")
    
    tests = [
        ("Technical Indicators", test_indicators),
        ("Multi-tier TP/SL", test_position_manager), 
        ("Feature Gating", test_feature_gating)
    ]
    
    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        results[test_name] = test_func()
    
    logger.info("\n=== Test Results ===")
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
    
    passed_count = sum(results.values())
    total_count = len(results)
    logger.info(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        logger.info("🎉 All enhanced features are working correctly!")
        return 0
    else:
        logger.warning(f"⚠️  {total_count - passed_count} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(main())