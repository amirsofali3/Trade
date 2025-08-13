#!/usr/bin/env python3
"""
Test indicator profiling functionality.

Validates that indicator calculation profiling works correctly:
- Times are recorded for each indicator
- Profiling results are stored in last_indicator_profile
- Slowest indicators are properly identified and logged
"""

import unittest
import time
from unittest.mock import Mock, MagicMock, patch
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data_collection.indicators import IndicatorCalculator


class TestIndicatorProfiling(unittest.TestCase):
    """Test indicator profiling functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock database manager
        self.mock_db = Mock()
        self.mock_db.get_ohlcv.return_value = self._create_mock_ohlcv_data()
        self.mock_db.save_indicator.return_value = None
        
        self.calc = IndicatorCalculator(self.mock_db)
        
        # Add a few mock indicators with controllable timing
        self.calc.available_indicators = {
            'fast_indicator': self._mock_fast_indicator,
            'slow_indicator': self._mock_slow_indicator,
            'medium_indicator': self._mock_medium_indicator
        }
        
        # Mock default params to avoid issues
        self.calc.default_params = {
            'fast_indicator': {},
            'slow_indicator': {},
            'medium_indicator': {}
        }
    
    def test_profiling_enabled(self):
        """Test profiling when enabled (default)"""
        indicators = ['fast_indicator', 'slow_indicator', 'medium_indicator']
        
        result = self.calc.calculate_indicators('BTCUSDT', '5m', indicators, profile=True)
        
        # Check that indicators were calculated
        self.assertEqual(len(result), 3)
        
        # Check that profiling data was stored
        self.assertIsInstance(self.calc.last_indicator_profile, dict)
        
        profile = self.calc.last_indicator_profile
        self.assertIn('total', profile)
        self.assertIn('total_time', profile)
        self.assertIn('avg_time', profile)
        self.assertIn('slowest', profile)
        
        # Verify profile content
        self.assertEqual(profile['total'], 3)
        self.assertGreater(profile['total_time'], 0)
        self.assertGreater(profile['avg_time'], 0)
        
        # Check slowest indicators list
        slowest = profile['slowest']
        self.assertIsInstance(slowest, list)
        self.assertGreater(len(slowest), 0)
        
        # Should have tuples of (name, time)
        for item in slowest:
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            self.assertIn(item[0], indicators)
            self.assertIsInstance(item[1], (int, float))
            self.assertGreater(item[1], 0)
    
    def test_profiling_disabled(self):
        """Test profiling when disabled"""
        indicators = ['fast_indicator', 'slow_indicator']
        
        # Mock _log_indicator_profile to verify it's not called
        with patch.object(self.calc, '_log_indicator_profile') as mock_log:
            result = self.calc.calculate_indicators('BTCUSDT', '5m', indicators, profile=False)
            
            # Profiling should not have been called
            mock_log.assert_not_called()
        
        # Check that indicators were still calculated
        self.assertEqual(len(result), 2)
    
    def test_slowest_indicators_ordering(self):
        """Test that slowest indicators are properly ordered"""
        indicators = ['fast_indicator', 'slow_indicator', 'medium_indicator']
        
        result = self.calc.calculate_indicators('BTCUSDT', '5m', indicators, profile=True)
        
        slowest = self.calc.last_indicator_profile['slowest']
        
        # Should be ordered by time descending (slowest first)
        times = [item[1] for item in slowest]
        self.assertEqual(times, sorted(times, reverse=True))
        
        # Slow indicator should be first in slowest list
        slowest_names = [item[0] for item in slowest]
        self.assertEqual(slowest_names[0], 'slow_indicator')
    
    def test_profile_logging_format(self):
        """Test the profile logging format"""
        indicators = ['fast_indicator', 'slow_indicator']
        
        # Capture log output
        with patch('data_collection.indicators.logger') as mock_logger:
            result = self.calc.calculate_indicators('BTCUSDT', '5m', indicators, profile=True)
            
            # Verify info log was called with expected format
            mock_logger.info.assert_called()
            log_call = mock_logger.info.call_args[0][0]
            
            # Check log format contains expected elements
            self.assertIn('Indicator profile:', log_call)
            self.assertIn('total=', log_call)
            self.assertIn('total_time=', log_call)
            self.assertIn('avg_time=', log_call)
            self.assertIn('slowest=', log_call)
    
    def test_profile_with_empty_indicators(self):
        """Test profiling with empty indicator list"""
        # Override available indicators to avoid default processing
        original_indicators = self.calc.available_indicators.copy()
        self.calc.available_indicators = {}  # Empty indicators
        
        try:
            result = self.calc.calculate_indicators('BTCUSDT', '5m', [], profile=True)
            
            # Should return empty result
            self.assertEqual(result, {})
            
            # Profile should not be set for empty indicator list
            self.assertEqual(self.calc.last_indicator_profile, {})
        finally:
            # Restore original indicators
            self.calc.available_indicators = original_indicators
    
    def test_profile_with_failing_indicator(self):
        """Test profiling when an indicator fails"""
        # Add a failing indicator
        self.calc.available_indicators['failing_indicator'] = self._mock_failing_indicator
        self.calc.default_params['failing_indicator'] = {}
        
        indicators = ['fast_indicator', 'failing_indicator', 'slow_indicator']
        
        result = self.calc.calculate_indicators('BTCUSDT', '5m', indicators, profile=True)
        
        # Should only have results from successful indicators
        self.assertEqual(len(result), 2)  # fast and slow
        
        # Profile should only include successful indicators
        profile = self.calc.last_indicator_profile
        self.assertEqual(profile['total'], 2)
        
        slowest_names = [item[0] for item in profile['slowest']]
        self.assertNotIn('failing_indicator', slowest_names)
    
    def _create_mock_ohlcv_data(self):
        """Create mock OHLCV data for testing"""
        dates = pd.date_range('2024-01-01', periods=100, freq='5min')
        
        data = pd.DataFrame({
            'open': np.random.uniform(50000, 51000, 100),
            'high': np.random.uniform(50500, 51500, 100),
            'low': np.random.uniform(49500, 50500, 100),
            'close': np.random.uniform(50000, 51000, 100),
            'volume': np.random.uniform(10, 100, 100)
        }, index=dates)
        
        return data
    
    def _mock_fast_indicator(self, ohlcv, **params):
        """Mock fast indicator (minimal delay)"""
        time.sleep(0.01)  # 10ms delay
        return pd.Series(np.random.random(len(ohlcv)), name='fast_indicator')
    
    def _mock_slow_indicator(self, ohlcv, **params):
        """Mock slow indicator (longer delay)"""
        time.sleep(0.05)  # 50ms delay
        return pd.Series(np.random.random(len(ohlcv)), name='slow_indicator')
    
    def _mock_medium_indicator(self, ohlcv, **params):
        """Mock medium speed indicator"""
        time.sleep(0.03)  # 30ms delay
        return pd.Series(np.random.random(len(ohlcv)), name='medium_indicator')
    
    def _mock_failing_indicator(self, ohlcv, **params):
        """Mock failing indicator"""
        raise Exception("Test indicator failure")


if __name__ == '__main__':
    unittest.main()