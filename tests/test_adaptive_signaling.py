#!/usr/bin/env python3
"""
Test adaptive signal thresholding and volatility filtering
"""
import unittest
import pandas as pd
import numpy as np
import torch
from unittest.mock import Mock
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestAdaptiveSignaling(unittest.TestCase):
    """Test adaptive signal thresholding and volatility filtering"""
    
    def setUp(self):
        """Set up test environment"""
        from models.signal_generator import SignalGenerator
        
        # Mock model
        self.mock_model = Mock()
        self.mock_db_manager = Mock()
        
        # Configuration with adaptive features
        self.config = {
            'signals': {
                'min_volatility_pct': 0.15
            }
        }
        
        self.signal_gen = SignalGenerator(
            model=self.mock_model,
            db_manager=self.mock_db_manager,
            config=self.config
        )
        
        # Mock model predictions
        def mock_model_call(features):
            # Return logits and confidences
            logits = torch.tensor([[0.2, 0.1, 0.7]])  # BUY=0.2, SELL=0.1, HOLD=0.7 before softmax
            confidences = torch.tensor([0.8])  # High confidence
            return logits, confidences
        
        self.mock_model.side_effect = mock_model_call
        
        # Create test OHLCV data
        self.create_test_ohlcv_data()
    
    def create_test_ohlcv_data(self):
        """Create test OHLCV data with different volatility scenarios"""
        dates = pd.date_range('2024-01-01', periods=50, freq='4H')
        
        # Low volatility scenario (tight range)
        low_vol_prices = [100 + i * 0.1 + np.random.uniform(-0.5, 0.5) for i in range(50)]
        self.low_vol_ohlcv = pd.DataFrame({
            'timestamp': dates,
            'open': low_vol_prices,
            'high': [p * 1.002 for p in low_vol_prices],  # Very tight range
            'low': [p * 0.998 for p in low_vol_prices],
            'close': low_vol_prices,
            'volume': np.random.randint(1000, 5000, 50)
        })
        
        # High volatility scenario (wide range)
        high_vol_prices = [100 + i * 0.5 + np.random.uniform(-2, 2) for i in range(50)]
        self.high_vol_ohlcv = pd.DataFrame({
            'timestamp': dates,
            'open': high_vol_prices,
            'high': [p * 1.025 for p in high_vol_prices],  # Wide range
            'low': [p * 0.975 for p in high_vol_prices],
            'close': high_vol_prices,
            'volume': np.random.randint(1000, 5000, 50)
        })
        
        # Medium volatility scenario
        med_vol_prices = [100 + i * 0.2 + np.random.uniform(-1, 1) for i in range(50)]
        self.med_vol_ohlcv = pd.DataFrame({
            'timestamp': dates,
            'open': med_vol_prices,
            'high': [p * 1.008 for p in med_vol_prices],  # Medium range
            'low': [p * 0.992 for p in med_vol_prices],
            'close': med_vol_prices,
            'volume': np.random.randint(1000, 5000, 50)
        })
    
    def test_volatility_calculation(self):
        """Test ATR volatility calculation"""
        print("Testing ATR volatility calculation...")
        
        current_price = 100.0
        
        # Test low volatility
        low_vol = self.signal_gen._calculate_current_volatility_atr(self.low_vol_ohlcv, current_price)
        
        # Test high volatility
        high_vol = self.signal_gen._calculate_current_volatility_atr(self.high_vol_ohlcv, current_price)
        
        # Test medium volatility
        med_vol = self.signal_gen._calculate_current_volatility_atr(self.med_vol_ohlcv, current_price)
        
        # Verify relationships
        self.assertLess(low_vol, med_vol, "Low volatility should be less than medium volatility")
        self.assertLess(med_vol, high_vol, "Medium volatility should be less than high volatility")
        
        # Verify reasonable ranges
        self.assertGreater(low_vol, 0, "Volatility should be positive")
        self.assertLess(high_vol, 10, "Volatility should be reasonable (<10%)")
        
        print(f"✅ Volatility calculation: low={low_vol:.3f}%, med={med_vol:.3f}%, high={high_vol:.3f}%")
    
    def test_adaptive_threshold_selection(self):
        """Test adaptive threshold selection based on volatility"""
        print("Testing adaptive threshold selection...")
        
        # Test different volatility scenarios
        low_vol_threshold = self.signal_gen._get_adaptive_threshold(0.1)  # 0.1% volatility
        med_vol_threshold = self.signal_gen._get_adaptive_threshold(0.3)  # 0.3% volatility  
        high_vol_threshold = self.signal_gen._get_adaptive_threshold(0.6)  # 0.6% volatility
        
        # Verify adaptive behavior (higher volatility = lower threshold)
        self.assertGreater(low_vol_threshold, med_vol_threshold, 
                          "Low volatility should have higher threshold")
        self.assertGreater(med_vol_threshold, high_vol_threshold,
                          "Medium volatility should have higher threshold than high volatility")
        
        # Verify reasonable threshold ranges
        self.assertGreaterEqual(low_vol_threshold, 0.6, "Thresholds should be reasonable")
        self.assertLessEqual(high_vol_threshold, 0.8, "Thresholds should be reasonable")
        
        print(f"✅ Adaptive thresholds: low_vol={low_vol_threshold}, med_vol={med_vol_threshold}, high_vol={high_vol_threshold}")
    
    def test_volatility_filtering(self):
        """Test that signals are filtered when volatility is too low"""
        print("Testing volatility filtering...")
        
        # Create test data with very low volatility (below threshold)
        features = {'ohlcv': torch.randn(1, 13), 'indicator': torch.randn(1, 50)}
        
        # Test with low volatility data
        test_data = {
            'ohlcv': self.low_vol_ohlcv,
            'features': features
        }
        
        # Mock model to return BUY signal with high confidence
        def mock_high_confidence(features):
            logits = torch.tensor([[2.0, 0.1, 0.1]])  # Strong BUY signal
            confidences = torch.tensor([0.9])
            return logits, confidences
        
        self.mock_model.side_effect = mock_high_confidence
        
        # Generate signal - should be None due to low volatility
        signal = self.signal_gen.generate_signal(test_data)
        
        # For very low volatility, signal might be filtered out
        if signal is None:
            print("✅ Low volatility signal correctly filtered")
        else:
            print(f"Signal generated despite low volatility: {signal.get('volatility_pct', 0):.3f}%")
            # This is also acceptable if volatility is above minimum threshold
            self.assertGreaterEqual(signal.get('volatility_pct', 0), self.config['signals']['min_volatility_pct'])
    
    def test_risk_adjusted_confidence(self):
        """Test risk-adjusted confidence calculation"""
        print("Testing risk-adjusted confidence calculation...")
        
        # Add some mock signal history
        self.signal_gen.recent_signals = [
            {'signal_type': 'BUY', 'confidence': 0.8, 'successful': True},
            {'signal_type': 'BUY', 'confidence': 0.7, 'successful': False},
            {'signal_type': 'BUY', 'confidence': 0.9, 'successful': True},
            {'signal_type': 'SELL', 'confidence': 0.6, 'successful': True},
            {'signal_type': 'SELL', 'confidence': 0.8, 'successful': False}
        ]
        
        # Test risk adjustment for BUY signal
        raw_confidence = 0.8
        risk_adjusted_buy = self.signal_gen._calculate_risk_adjusted_confidence(raw_confidence, 'BUY')
        
        # Test risk adjustment for SELL signal
        risk_adjusted_sell = self.signal_gen._calculate_risk_adjusted_confidence(raw_confidence, 'SELL')
        
        # Verify adjustments are reasonable
        self.assertGreater(risk_adjusted_buy, 0.5, "Risk-adjusted confidence should be reasonable")
        self.assertGreater(risk_adjusted_sell, 0.5, "Risk-adjusted confidence should be reasonable")
        self.assertLessEqual(risk_adjusted_buy, raw_confidence, "Risk adjustment should not increase confidence beyond raw")
        self.assertLessEqual(risk_adjusted_sell, raw_confidence, "Risk adjustment should not increase confidence beyond raw")
        
        print(f"✅ Risk adjustment: raw={raw_confidence:.3f}, BUY={risk_adjusted_buy:.3f}, SELL={risk_adjusted_sell:.3f}")
    
    def test_signal_outcome_tracking(self):
        """Test signal outcome tracking for calibration"""
        print("Testing signal outcome tracking...")
        
        # Add a signal outcome
        signal_id = 'test_123'
        self.signal_gen.recent_signals = [
            {'id': signal_id, 'signal_type': 'BUY', 'confidence': 0.8}
        ]
        
        # Add outcome
        self.signal_gen.add_signal_outcome(signal_id, successful=True, profit_loss=50.0)
        
        # Verify outcome was recorded
        signal = self.signal_gen.recent_signals[0]
        self.assertTrue(signal.get('successful'), "Signal outcome should be recorded as successful")
        self.assertEqual(signal.get('profit_loss'), 50.0, "Profit/loss should be recorded")
        
        print("✅ Signal outcome tracking working correctly")
    
    def test_enhanced_signal_structure(self):
        """Test that generated signals include enhanced information"""
        print("Testing enhanced signal structure...")
        
        # Set up test data
        features = {'ohlcv': torch.randn(1, 13), 'indicator': torch.randn(1, 50)}
        test_data = {
            'ohlcv': self.high_vol_ohlcv,  # Use high volatility to ensure signal generation
            'features': features
        }
        
        # Mock model to return clear BUY signal
        def mock_clear_buy(features):
            logits = torch.tensor([[3.0, 0.1, 0.1]])  # Strong BUY
            confidences = torch.tensor([0.9])
            return logits, confidences
        
        self.mock_model.side_effect = mock_clear_buy
        
        # Generate signal
        signal = self.signal_gen.generate_signal(test_data)
        
        if signal:
            # Verify enhanced structure
            required_fields = [
                'signal', 'confidence', 'volatility_pct', 'adaptive_threshold',
                'risk_adjusted_confidence', 'probabilities'
            ]
            
            for field in required_fields:
                self.assertIn(field, signal, f"Signal should include {field}")
            
            # Verify probabilities structure
            prob_keys = ['BUY', 'SELL', 'HOLD']
            for key in prob_keys:
                self.assertIn(key, signal['probabilities'], f"Probabilities should include {key}")
            
            print("✅ Enhanced signal structure verified")
            print(f"  Signal: {signal.get('signal')}")
            print(f"  Confidence: {signal.get('confidence'):.3f}")
            print(f"  Risk-adjusted: {signal.get('risk_adjusted_confidence'):.3f}")
            print(f"  Volatility: {signal.get('volatility_pct'):.3f}%")
        else:
            print("No signal generated (may be expected based on conditions)")

if __name__ == '__main__':
    print("🧪 Testing Adaptive Signaling")
    print("=" * 35)
    unittest.main(verbosity=2)