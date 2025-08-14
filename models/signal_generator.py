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
                cooldown_period=6, symbol='BTCUSDT', config=None):
        """
        Initialize signal generator
        
        Args:
            model: Neural network model
            db_manager: Database manager instance
            confidence_threshold: Minimum confidence threshold for generating signals (0-1)
            cooldown_period: Minimum number of candles between signals
            symbol: Trading symbol
            config: Configuration dictionary for adaptive features
        """
        self.model = model
        self.db_manager = db_manager
        self.confidence_threshold = confidence_threshold
        self.cooldown_period = cooldown_period
        self.symbol = symbol
        self.config = config or {}
        
        # Adaptive signal thresholding & volatility filter (Phase 2)
        self.min_volatility_pct = self.config.get('signals', {}).get('min_volatility_pct', 0.15)
        self.adaptive_thresholds = {
            'high_volatility': 0.65,  # Lower threshold in high volatility
            'medium_volatility': 0.70,  # Standard threshold
            'low_volatility': 0.75  # Higher threshold in low volatility
        }
        
        # Rolling calibration for risk-adjusted confidence (Phase 2)
        self.recent_signals = []  # Store recent signals for Brier score calculation
        self.brier_window = 50  # Number of recent signals to consider
        
        # Keep track of last executed signal (not just generated)
        self.last_executed_signal_time = None
        self.last_executed_signal_type = None
        
        logger.info(f"Initialized SignalGenerator with adaptive thresholding (min_volatility={self.min_volatility_pct}%)")
        logger.info(f"Adaptive thresholds: {self.adaptive_thresholds}")
    
    def _calculate_current_volatility_atr(self, ohlcv_data, current_price, period=14):
        """
        Calculate current market volatility using ATR% 
        
        Args:
            ohlcv_data: OHLCV DataFrame or dict
            current_price: Current price for percentage calculation
            period: ATR calculation period
            
        Returns:
            float: Volatility as percentage of current price
        """
        try:
            # Convert to DataFrame if needed
            if isinstance(ohlcv_data, dict):
                ohlcv_df = pd.DataFrame([ohlcv_data])
            elif hasattr(ohlcv_data, 'iloc'):
                ohlcv_df = ohlcv_data.tail(period + 1)
            else:
                return self.min_volatility_pct  # Fallback
            
            if len(ohlcv_df) < 2:
                return self.min_volatility_pct  # Fallback
            
            # Calculate True Range
            high = ohlcv_df['high']
            low = ohlcv_df['low']
            close_prev = ohlcv_df['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Calculate ATR (simple moving average of True Range)
            atr = true_range.rolling(window=min(period, len(true_range))).mean().iloc[-1]
            
            # Convert to percentage of current price
            atr_pct = (atr / current_price) * 100
            
            return float(atr_pct)
            
        except Exception as e:
            logger.debug(f"Error calculating ATR volatility: {e}")
            return self.min_volatility_pct  # Fallback to minimum
    
    def _get_adaptive_threshold(self, volatility_pct):
        """
        Get adaptive confidence threshold based on market volatility
        
        Args:
            volatility_pct: Current volatility percentage
            
        Returns:
            float: Adaptive confidence threshold
        """
        if volatility_pct > 0.5:  # High volatility (> 0.5%)
            return self.adaptive_thresholds['high_volatility']
        elif volatility_pct > 0.25:  # Medium volatility (0.25% - 0.5%)
            return self.adaptive_thresholds['medium_volatility']
        else:  # Low volatility (< 0.25%)
            return self.adaptive_thresholds['low_volatility']
    
    def _calculate_risk_adjusted_confidence(self, confidence, signal_type):
        """
        Calculate risk-adjusted confidence using Brier score calibration
        
        Args:
            confidence: Raw model confidence
            signal_type: Signal type ('BUY' or 'SELL')
            
        Returns:
            float: Risk-adjusted confidence
        """
        try:
            if len(self.recent_signals) < 10:
                return confidence  # Not enough history, return raw confidence
            
            # Calculate Brier score for recent signals of same type
            same_type_signals = [s for s in self.recent_signals if s.get('signal_type') == signal_type]
            
            if len(same_type_signals) < 5:
                return confidence  # Not enough same-type signals
            
            # Calculate average actual success rate vs predicted confidence  
            total_brier_score = 0.0
            for signal in same_type_signals[-20:]:  # Last 20 signals of same type
                predicted_prob = signal.get('confidence', 0.5)
                actual_outcome = signal.get('successful', False)
                actual_prob = 1.0 if actual_outcome else 0.0
                
                # Brier score component
                total_brier_score += (predicted_prob - actual_prob) ** 2
            
            avg_brier_score = total_brier_score / len(same_type_signals[-20:])
            
            # Adjust confidence based on calibration
            # Lower Brier score means better calibration, higher means overconfidence
            calibration_factor = max(0.5, 1.0 - avg_brier_score)  # Don't reduce confidence below 50%
            
            risk_adjusted = confidence * calibration_factor
            
            logger.debug(f"Risk adjustment: {confidence:.3f} -> {risk_adjusted:.3f} (Brier: {avg_brier_score:.3f})")
            
            return risk_adjusted
            
        except Exception as e:
            logger.debug(f"Error calculating risk-adjusted confidence: {e}")
            return confidence  # Fallback to raw confidence
    
    def add_signal_outcome(self, signal_id, successful, profit_loss=None):
        """
        Add outcome for a signal to improve calibration
        
        Args:
            signal_id: Signal identifier
            successful: Whether the signal was successful
            profit_loss: Profit/loss amount (optional)
        """
        try:
            # Find the signal in recent signals
            for signal in self.recent_signals:
                if signal.get('id') == signal_id:
                    signal['successful'] = successful
                    if profit_loss is not None:
                        signal['profit_loss'] = profit_loss
                    break
            
            # Keep only recent signals
            if len(self.recent_signals) > self.brier_window:
                self.recent_signals = self.recent_signals[-self.brier_window:]
                
        except Exception as e:
            logger.debug(f"Error adding signal outcome: {e}")
    
    def generate_signal(self, data, active_features=None, timestamp=None):
        """
        Generate trading signal with adaptive thresholding and volatility filtering
        
        Args:
            data: Dictionary with OHLCV data and features, or legacy features dict
            active_features: Dictionary with active features and weights (optional)
            timestamp: Signal timestamp (default: current time)
            
        Returns:
            Dictionary with signal information or None
        """
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            # Handle both new data structure and legacy features-only calls
            if 'ohlcv' in data and 'features' in data:
                # New structure with OHLCV and features
                ohlcv_data = data['ohlcv']
                features = data['features']
                current_price = float(ohlcv_data['close'].iloc[-1]) if hasattr(ohlcv_data['close'], 'iloc') else float(ohlcv_data['close'])
                
                # Calculate volatility and apply filter
                volatility_pct = self._calculate_current_volatility_atr(ohlcv_data, current_price)
                
                if volatility_pct < self.min_volatility_pct:
                    logger.info(f"Market volatility {volatility_pct:.3f}% below minimum {self.min_volatility_pct}%, skipping signal")
                    return None
                
                # Get adaptive threshold
                adaptive_threshold = self._get_adaptive_threshold(volatility_pct)
                
            else:
                # Legacy structure - features only
                features = data
                current_price = active_features.get('price', 0.0) if active_features else 0.0
                volatility_pct = self.min_volatility_pct  # Use minimum as fallback
                adaptive_threshold = self.confidence_threshold  # Use standard threshold
            
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
                logits, confidences = self.model(features)
            
            # Get prediction  
            probs = torch.softmax(logits, dim=-1)
            probs_np = probs.cpu().numpy().flatten()
            confidence_val = float(probs_np[np.argmax(probs_np)])
            
            # Determine signal type
            signal_idx = np.argmax(probs_np)
            signal_types = ['BUY', 'SELL', 'HOLD']
            signal_type = signal_types[signal_idx]
            
            logger.info(f"Model prediction: {signal_type} with confidence {confidence_val:.4f} (volatility: {volatility_pct:.3f}%)")
            
            # Apply adaptive confidence thresholding
            if signal_type == 'BUY' and confidence_val < adaptive_threshold:
                logger.info(f"BUY signal confidence {confidence_val:.4f} below adaptive threshold {adaptive_threshold:.3f}")
                return None
            elif signal_type == 'SELL' and confidence_val < max(0.3, adaptive_threshold - 0.1):
                logger.info(f"SELL signal confidence {confidence_val:.4f} below threshold")
                return None
            
            # No signal if HOLD
            if signal_type == 'HOLD':
                logger.info("Hold signal, no action needed")
                return None
            
            # Calculate risk-adjusted confidence
            risk_adjusted_confidence = self._calculate_risk_adjusted_confidence(confidence_val, signal_type)
            
            # Create enhanced signal
            signal = {
                'symbol': self.symbol,
                'timestamp': timestamp,
                'signal': signal_type,  # Standardized key for inference scheduler
                'signal_type': signal_type,  # Keep for backward compatibility
                'price': current_price,
                'confidence': float(confidence_val),
                'risk_adjusted_confidence': risk_adjusted_confidence,
                'volatility_pct': volatility_pct,
                'adaptive_threshold': adaptive_threshold,
                'probabilities': {
                    'BUY': float(probs_np[0]),
                    'SELL': float(probs_np[1]),
                    'HOLD': float(probs_np[2])
                },
                'executed': False,
                'features_used': json.dumps(active_features) if active_features else '{}'
            }
            
            # Store signal for calibration tracking
            signal_copy = signal.copy()
            signal_copy['id'] = None  # Will be set when saved to DB
            self.recent_signals.append(signal_copy)
            
            # Keep only recent signals
            if len(self.recent_signals) > self.brier_window:
                self.recent_signals = self.recent_signals[-self.brier_window:]
            
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