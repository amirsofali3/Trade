#!/usr/bin/env python3
"""
Test indicator implementations for missing indicators as specified in requirements:
- Ichimoku chikou span visibility
- StochRSI K/D outputs  
- Keltner Channels functionality
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection.indicators import IndicatorCalculator
from database.db_manager import DatabaseManager

class TestIndicatorImplementations(unittest.TestCase):
    """Test specific indicator implementations"""
    
    def setUp(self):
        """Set up test environment with mock database manager"""
        class MockDBManager:
            pass
        
        self.indicator_calc = IndicatorCalculator(MockDBManager())
        
        # Create test OHLCV data
        np.random.seed(42)  # For reproducible results
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        
        prices = []
        base_price = 50000.0
        for i in range(100):
            change = np.random.normal(0, 0.01)  # 1% volatility
            base_price *= (1 + change)
            prices.append(base_price)
        
        self.test_ohlcv = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
    
    def test_ichimoku_chikou_span_visibility(self):
        """Test that Ichimoku includes chikou span and it's properly visible"""
        print("Testing Ichimoku chikou span implementation...")
        
        # Calculate Ichimoku
        ichimoku_result = self.indicator_calc._calculate_ichimoku(self.test_ohlcv)
        
        # Verify all expected components are present
        expected_keys = ['tenkan_sen', 'kijun_sen', 'senkou_a', 'senkou_b', 'chikou']
        for key in expected_keys:
            self.assertIn(key, ichimoku_result, f"Missing {key} in Ichimoku result")
        
        # Specifically test chikou span
        chikou = ichimoku_result['chikou']
        self.assertIsNotNone(chikou, "Chikou span should not be None")
        self.assertIsInstance(chikou, pd.Series, "Chikou should be a pandas Series")
        
        # Test that chikou is shifted properly (lagging span)
        # Chikou should be close prices shifted backward by kijun period (26)
        expected_chikou = self.test_ohlcv['close'].shift(-26)
        pd.testing.assert_series_equal(chikou, expected_chikou, check_names=False)
        
        print("✅ Ichimoku chikou span implementation validated")
    
    def test_stochrsi_k_d_outputs(self):
        """Test that StochRSI returns both K and D values"""
        print("Testing StochRSI K/D outputs...")
        
        # Calculate StochRSI
        stochrsi_result = self.indicator_calc._calculate_stochrsi(self.test_ohlcv)
        
        # Verify both K and D are present
        expected_keys = ['stochrsi_k', 'stochrsi_d']
        for key in expected_keys:
            self.assertIn(key, stochrsi_result, f"Missing {key} in StochRSI result")
        
        # Test K and D are valid series
        stochrsi_k = stochrsi_result['stochrsi_k']
        stochrsi_d = stochrsi_result['stochrsi_d']
        
        self.assertIsInstance(stochrsi_k, pd.Series, "StochRSI K should be pandas Series")
        self.assertIsInstance(stochrsi_d, pd.Series, "StochRSI D should be pandas Series")
        
        # Test that values are in reasonable range [0, 1]
        valid_k = stochrsi_k.dropna()
        valid_d = stochrsi_d.dropna()
        
        if len(valid_k) > 0:
            self.assertTrue((valid_k >= 0).all() and (valid_k <= 1).all(), 
                          "StochRSI K values should be between 0 and 1")
        
        if len(valid_d) > 0:
            self.assertTrue((valid_d >= 0).all() and (valid_d <= 1).all(), 
                          "StochRSI D values should be between 0 and 1")
        
        # Test that D is smoother than K (D is moving average of K)
        if len(valid_k) > 10 and len(valid_d) > 10:
            k_volatility = valid_k.std()
            d_volatility = valid_d.std()
            self.assertLessEqual(d_volatility, k_volatility, 
                               "StochRSI D should be smoother (less volatile) than K")
        
        print("✅ StochRSI K/D implementation validated")
    
    def test_keltner_channels_functionality(self):
        """Test that Keltner Channels function correctly"""
        print("Testing Keltner Channels functionality...")
        
        # Calculate Keltner Channels
        keltner_result = self.indicator_calc._calculate_keltner_channel(self.test_ohlcv)
        
        # Verify all components are present
        expected_keys = ['keltner_upper', 'keltner_middle', 'keltner_lower']
        for key in expected_keys:
            self.assertIn(key, keltner_result, f"Missing {key} in Keltner result")
        
        # Extract components
        upper = keltner_result['keltner_upper']
        middle = keltner_result['keltner_middle'] 
        lower = keltner_result['keltner_lower']
        
        # Test that all are pandas Series
        for name, series in [('upper', upper), ('middle', middle), ('lower', lower)]:
            self.assertIsInstance(series, pd.Series, f"Keltner {name} should be pandas Series")
        
        # Test channel relationships - upper > middle > lower
        valid_indices = ~(upper.isna() | middle.isna() | lower.isna())
        if valid_indices.sum() > 0:
            valid_upper = upper[valid_indices]
            valid_middle = middle[valid_indices]
            valid_lower = lower[valid_indices]
            
            self.assertTrue((valid_upper >= valid_middle).all(), 
                          "Keltner upper should be >= middle")
            self.assertTrue((valid_middle >= valid_lower).all(), 
                          "Keltner middle should be >= lower")
        
        # Test that middle line is EMA of close prices
        expected_middle = self.test_ohlcv['close'].ewm(span=20).mean()
        pd.testing.assert_series_equal(middle, expected_middle, check_names=False)
        
        # Test that channel width is based on ATR
        atr = self.indicator_calc._calculate_atr(self.test_ohlcv, 20)
        expected_upper = expected_middle + (2 * atr)
        expected_lower = expected_middle - (2 * atr)
        
        pd.testing.assert_series_equal(upper, expected_upper, check_names=False)
        pd.testing.assert_series_equal(lower, expected_lower, check_names=False)
        
        print("✅ Keltner Channels implementation validated")
    
    def test_indicator_cache_functionality(self):
        """Test indicator caching and full refresh cycle functionality"""
        print("Testing indicator cache functionality...")
        
        # Reset cache first
        self.indicator_calc.reset_cache()
        initial_stats = self.indicator_calc.get_computation_stats()
        
        # Test cache key generation and storage directly
        cache_key = 'BTCUSDT_5m'
        
        # Manually populate cache to test functionality
        self.indicator_calc.indicator_cache[cache_key] = {'sma': [1, 2, 3]}
        self.indicator_calc.indicator_timestamps[cache_key] = '2024-01-01T12:00:00'
        
        # Test cache exists
        self.assertIn(cache_key, self.indicator_calc.indicator_cache,
                     "Cache key should exist after manual population")
        
        # Test reset cache functionality  
        self.indicator_calc.reset_cache()
        self.assertEqual(len(self.indicator_calc.indicator_cache), 0,
                        "Cache should be empty after reset")
        self.assertEqual(len(self.indicator_calc.indicator_timestamps), 0,
                        "Timestamps should be empty after reset")
        
        # Test timestamp comparison logic directly
        should_skip = self.indicator_calc._should_skip_computation('BTCUSDT', '5m', '2024-01-01T12:00:00')
        self.assertFalse(should_skip, "Should not skip when no previous timestamp exists")
        
        # Add timestamp and test again
        self.indicator_calc.indicator_timestamps['BTCUSDT_5m'] = '2024-01-01T12:00:00'
        should_skip_same = self.indicator_calc._should_skip_computation('BTCUSDT', '5m', '2024-01-01T12:00:00')
        should_skip_new = self.indicator_calc._should_skip_computation('BTCUSDT', '5m', '2024-01-01T12:05:00')
        
        self.assertTrue(should_skip_same, "Should skip when timestamp is unchanged")
        self.assertFalse(should_skip_new, "Should not skip when timestamp is new")
        
        print("✅ Indicator cache functionality validated")
    
    def test_full_refresh_cycle_configuration(self):
        """Test configurable full refresh cycles"""
        print("Testing full refresh cycle configuration...")
        
        # Mock config with full_refresh_every_n_cycles = 3
        config = {'indicators': {'full_refresh_every_n_cycles': 3}}
        
        # Test should force full refresh logic
        should_refresh_1 = self.indicator_calc._should_force_full_refresh('BTCUSDT', '5m', config)
        should_refresh_2 = self.indicator_calc._should_force_full_refresh('BTCUSDT', '5m', config)
        should_refresh_3 = self.indicator_calc._should_force_full_refresh('BTCUSDT', '5m', config)
        
        # First and second calls should not force refresh, third should
        self.assertFalse(should_refresh_1, "First call should not force refresh")
        self.assertFalse(should_refresh_2, "Second call should not force refresh") 
        self.assertTrue(should_refresh_3, "Third call should force refresh (cycle=3)")
        
        # Test disabled refresh (value = 0)
        config_disabled = {'indicators': {'full_refresh_every_n_cycles': 0}}
        should_refresh_disabled = self.indicator_calc._should_force_full_refresh('ETHUSDT', '1h', config_disabled)
        self.assertFalse(should_refresh_disabled, "Should not refresh when disabled (value=0)")
        
        print("✅ Full refresh cycle configuration validated")

if __name__ == '__main__':
    print("🧪 Testing Indicator Implementations")
    print("=" * 50)
    unittest.main(verbosity=2)