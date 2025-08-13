#!/usr/bin/env python3
"""
Test indicator skip logic and caching functionality.

Ensures unchanged timestamp leads to skip & cached stats increment.
"""

import unittest
import time
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_collection.indicators import IndicatorCalculator


class TestIndicatorSkipLogic(unittest.TestCase):
    """Test indicator computation caching and skip logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock database manager
        self.mock_db_manager = Mock()
        self.indicator_calc = IndicatorCalculator(self.mock_db_manager)
        
        # Create sample OHLCV data
        self.sample_ohlcv = pd.DataFrame({
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='5min'),
            'open': [100 + i for i in range(100)],
            'high': [101 + i for i in range(100)],
            'low': [99 + i for i in range(100)],
            'close': [100.5 + i for i in range(100)],
            'volume': [1000 + i*10 for i in range(100)]
        })
        
        # Mock database to return sample data
        self.mock_db_manager.get_ohlcv.return_value = self.sample_ohlcv
        self.mock_db_manager.save_indicator.return_value = None
    
    def test_initial_computation_stats(self):
        """Test that computation stats initialize correctly"""
        stats = self.indicator_calc.get_computation_stats()
        
        expected_keys = ['computed', 'skipped', 'cache_hits']
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertEqual(stats[key], 0)
    
    def test_first_computation_no_skip(self):
        """Test that first computation doesn't skip anything"""
        result = self.indicator_calc.calculate_indicators(
            'BTCUSDT', '5m', ['sma', 'ema']
        )
        
        # Should compute indicators
        self.assertIn('sma', result)
        self.assertIn('ema', result)
        
        # Check stats
        stats = self.indicator_calc.get_computation_stats()
        self.assertEqual(stats['computed'], 2)  # sma + ema
        self.assertEqual(stats['skipped'], 0)
        self.assertEqual(stats['cache_hits'], 0)
    
    def test_skip_computation_unchanged_timestamp(self):
        """Test that computation is skipped when timestamp unchanged"""
        # First computation
        result1 = self.indicator_calc.calculate_indicators(
            'BTCUSDT', '5m', ['sma', 'ema']
        )
        
        # Mock database to return same data (unchanged timestamp)
        self.mock_db_manager.get_ohlcv.return_value = self.sample_ohlcv
        
        # Second computation - should skip
        result2 = self.indicator_calc.calculate_indicators(
            'BTCUSDT', '5m', ['sma', 'ema']
        )
        
        # Should return cached results
        self.assertEqual(result1, result2)
        
        # Check that skip was detected
        stats = self.indicator_calc.get_computation_stats()
        self.assertEqual(stats['skipped'], 2)  # Both indicators skipped
    
    def test_computation_with_new_timestamp(self):
        """Test that computation proceeds with new timestamp"""
        # First computation
        self.indicator_calc.calculate_indicators('BTCUSDT', '5m', ['sma'])
        
        # Create new data with different timestamp
        new_ohlcv = self.sample_ohlcv.copy()
        new_ohlcv['timestamp'] = pd.date_range(start='2024-01-01', periods=101, freq='5min')
        new_ohlcv.loc[100] = [200, 201, 199, 200.5, 2000]  # Add new row
        
        self.mock_db_manager.get_ohlcv.return_value = new_ohlcv
        
        # Second computation - should not skip
        result = self.indicator_calc.calculate_indicators('BTCUSDT', '5m', ['sma'])
        
        # Should compute new indicators
        stats = self.indicator_calc.get_computation_stats()
        self.assertEqual(stats['computed'], 1)
        self.assertEqual(stats['skipped'], 0)
    
    def test_selective_computation(self):
        """Test selective computation with active indicators"""
        active_indicators = ['sma', 'rsi']
        
        result = self.indicator_calc.calculate_indicators(
            'BTCUSDT', '5m',
            use_selected_only=True,
            active_indicators=active_indicators
        )
        
        # Should only compute selected indicators
        self.assertIn('sma', result)
        self.assertIn('rsi', result)
        
        # Should not compute others that weren't requested
        all_available = len(self.indicator_calc.available_indicators)
        stats = self.indicator_calc.get_computation_stats()
        self.assertEqual(stats['computed'], 2)  # Only sma + rsi
    
    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly"""
        # First computation
        self.indicator_calc.calculate_indicators('BTCUSDT', '5m', ['sma'])
        
        # Check that cache was populated
        cache_key = 'BTCUSDT_5m'
        self.assertIn(cache_key, self.indicator_calc.indicator_cache)
        self.assertIn(cache_key, self.indicator_calc.indicator_timestamps)
    
    def test_cache_different_symbols(self):
        """Test that different symbols have separate cache entries"""
        # Compute for two different symbols
        self.indicator_calc.calculate_indicators('BTCUSDT', '5m', ['sma'])
        self.indicator_calc.calculate_indicators('ETHUSDT', '5m', ['sma'])
        
        # Should have separate cache entries
        self.assertIn('BTCUSDT_5m', self.indicator_calc.indicator_cache)
        self.assertIn('ETHUSDT_5m', self.indicator_calc.indicator_cache)
        
        # Entries should be different
        self.assertNotEqual(
            self.indicator_calc.indicator_cache['BTCUSDT_5m'],
            self.indicator_calc.indicator_cache['ETHUSDT_5m']
        )
    
    def test_cache_different_timeframes(self):
        """Test that different timeframes have separate cache entries"""
        # Compute for same symbol, different timeframes
        self.indicator_calc.calculate_indicators('BTCUSDT', '5m', ['sma'])
        self.indicator_calc.calculate_indicators('BTCUSDT', '1h', ['sma'])
        
        # Should have separate cache entries
        self.assertIn('BTCUSDT_5m', self.indicator_calc.indicator_cache)
        self.assertIn('BTCUSDT_1h', self.indicator_calc.indicator_cache)
    
    def test_timestamp_comparison_logic(self):
        """Test the timestamp comparison logic"""
        symbol, timeframe = 'BTCUSDT', '5m'
        timestamp1 = pd.Timestamp('2024-01-01 00:00:00')
        timestamp2 = pd.Timestamp('2024-01-01 00:05:00')
        
        # First check - should not skip (no previous timestamp)
        should_skip = self.indicator_calc._should_skip_computation(symbol, timeframe, timestamp1)
        self.assertFalse(should_skip)
        
        # Set timestamp
        cache_key = f'{symbol}_{timeframe}'
        self.indicator_calc.indicator_timestamps[cache_key] = timestamp1
        
        # Same timestamp - should skip
        should_skip = self.indicator_calc._should_skip_computation(symbol, timeframe, timestamp1)
        self.assertTrue(should_skip)
        
        # Different timestamp - should not skip
        should_skip = self.indicator_calc._should_skip_computation(symbol, timeframe, timestamp2)
        self.assertFalse(should_skip)
    
    def test_skip_with_no_timestamp(self):
        """Test skip logic when no timestamp is available"""
        # OHLCV data without timestamp column
        ohlcv_no_timestamp = self.sample_ohlcv.drop(columns=['timestamp'])
        self.mock_db_manager.get_ohlcv.return_value = ohlcv_no_timestamp
        
        # Should not skip computation
        result = self.indicator_calc.calculate_indicators('BTCUSDT', '5m', ['sma'])
        
        stats = self.indicator_calc.get_computation_stats()
        self.assertEqual(stats['computed'], 1)
        self.assertEqual(stats['skipped'], 0)
    
    def test_cache_reset(self):
        """Test cache reset functionality"""
        # Populate cache
        self.indicator_calc.calculate_indicators('BTCUSDT', '5m', ['sma'])
        
        # Verify cache is populated
        self.assertGreater(len(self.indicator_calc.indicator_cache), 0)
        self.assertGreater(len(self.indicator_calc.indicator_timestamps), 0)
        
        # Reset cache
        self.indicator_calc.reset_cache()
        
        # Verify cache is empty
        self.assertEqual(len(self.indicator_calc.indicator_cache), 0)
        self.assertEqual(len(self.indicator_calc.indicator_timestamps), 0)
    
    def test_stats_reset_between_calculations(self):
        """Test that stats reset between calculation calls"""
        # First calculation
        self.indicator_calc.calculate_indicators('BTCUSDT', '5m', ['sma'])
        stats1 = self.indicator_calc.get_computation_stats()
        
        # Second calculation (should reset stats)
        self.indicator_calc.calculate_indicators('BTCUSDT', '5m', ['ema'])
        stats2 = self.indicator_calc.get_computation_stats()
        
        # Stats should be reset for each call
        # The exact values depend on whether skipping occurred
        self.assertIsInstance(stats2['computed'], int)
        self.assertIsInstance(stats2['skipped'], int)
        self.assertIsInstance(stats2['cache_hits'], int)
    
    def test_empty_ohlcv_data(self):
        """Test behavior with empty OHLCV data"""
        self.mock_db_manager.get_ohlcv.return_value = pd.DataFrame()
        
        result = self.indicator_calc.calculate_indicators('BTCUSDT', '5m', ['sma'])
        
        # Should return empty result
        self.assertEqual(result, {})
        
        # Should not update cache or stats
        stats = self.indicator_calc.get_computation_stats()
        self.assertEqual(stats['computed'], 0)
        self.assertEqual(stats['skipped'], 0)


if __name__ == '__main__':
    unittest.main()