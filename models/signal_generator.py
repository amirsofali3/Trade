import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
import json
from database.db_manager import DatabaseManager

logger = logging.getLogger("SignalGenerator")

class SignalGenerator:
    """
    Generates trading signals based on model predictions
    """
    
    def __init__(self, model, db_manager, confidence_threshold=0.7,
                cooldown_period=6, symbol='BTCUSDT'):
        """
        Initialize signal generator
        
        Args:
            model: Neural network model
            db_manager: Database manager instance
            confidence_threshold: Minimum confidence threshold for generating signals (0-1)
            cooldown_period: Minimum number of candles between signals
            symbol: Trading symbol
        """
        self.model = model
        self.db_manager = db_manager
        self.confidence_threshold = confidence_threshold
        self.cooldown_period = cooldown_period
        self.symbol = symbol
        
        # Keep track of last executed signal (not just generated)
        self.last_executed_signal_time = None
        self.last_executed_signal_type = None
        
        logger.info(f"Initialized SignalGenerator with confidence threshold {confidence_threshold}")
    
    def generate_signal(self, features, active_features, current_price, timestamp=None):
        """
        Generate trading signal based on features
        
        Args:
            features: Dictionary with feature tensors
            active_features: Dictionary with active features and weights
            current_price: Current market price
            timestamp: Signal timestamp (default: current time)
            
        Returns:
            Dictionary with signal information or None
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Check cooldown period (only for executed signals)
            if self.last_executed_signal_time is not None:
                time_since_last = (timestamp - self.last_executed_signal_time).total_seconds()
                cooldown_seconds = self.cooldown_period * 5 * 60  # Convert candles to seconds for 5m timeframe
                
                if time_since_last < cooldown_seconds:
                    logger.debug(f"Signal cooldown in effect, {cooldown_seconds - time_since_last:.0f} seconds remaining")
                    return None
            
            # Evaluate model
            self.model.eval()
            with torch.no_grad():
                probs, confidence = self.model(features)
            
            # Get prediction
            probs_np = probs.cpu().numpy()[0]
            confidence_val = confidence.cpu().numpy()[0]
            
            # Determine signal type
            # 0 = buy, 1 = sell, 2 = hold
            signal_idx = np.argmax(probs_np)
            signal_types = ['BUY', 'SELL', 'HOLD']
            signal_type = signal_types[signal_idx]
            
            logger.info(f"Model prediction: {signal_type} with confidence {confidence_val:.4f} (probs: {probs_np})")
            
            # Check confidence threshold - different logic for BUY vs SELL
            # BUY signals always need high confidence
            # SELL signals can be generated with low confidence but need high confidence to execute
            if signal_type == 'BUY' and confidence_val < self.confidence_threshold:
                logger.info(f"BUY signal confidence {confidence_val:.4f} below threshold {self.confidence_threshold}, no signal")
                return None
            elif signal_type == 'SELL' and confidence_val < 0.3:  # Minimum threshold for SELL signal generation
                logger.info(f"SELL signal confidence {confidence_val:.4f} too low for generation (minimum 0.3), no signal")
                return None
            
            # No signal if HOLD
            if signal_type == 'HOLD':
                logger.info("Hold signal, no action needed")
                return None
            
            # Create signal
            signal = {
                'symbol': self.symbol,
                'timestamp': timestamp,
                'signal_type': signal_type,
                'price': current_price,
                'confidence': float(confidence_val),
                'executed': False,
                'features_used': json.dumps(active_features)
            }
            
            # Save to database
            signal_id = self.db_manager.save_signal(signal)
            
            if signal_id:
                signal['id'] = signal_id
                
                # Note: We don't update cooldown timer here anymore
                # It will be updated when the signal is actually executed
                
                logger.info(f"Generated {signal_type} signal with confidence {confidence_val:.4f}")
                return signal
            else:
                logger.warning("Failed to save signal to database")
                return None
        
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None
    
    def mark_signal_executed(self, signal_type, timestamp=None):
        """
        Mark a signal as executed and start cooldown timer
        
        Args:
            signal_type: Type of signal that was executed ('BUY' or 'SELL')
            timestamp: Execution timestamp (default: current time)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.last_executed_signal_time = timestamp
        self.last_executed_signal_type = signal_type
        
        logger.info(f"Marked {signal_type} signal as executed, cooldown period started")
    
    def set_confidence_threshold(self, threshold):
        """Update confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Updated confidence threshold to {self.confidence_threshold}")
    
    def set_cooldown_period(self, period):
        """Update cooldown period in number of candles"""
        self.cooldown_period = max(1, period)
        logger.info(f"Updated cooldown period to {self.cooldown_period} candles")