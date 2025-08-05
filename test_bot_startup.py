#!/usr/bin/env python3
"""
Test script to verify the main bot can start without tensor errors
"""

import logging
import time
import threading
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BotTest")

def test_bot_startup():
    """Test that the main bot can start without errors"""
    try:
        from main import TradingBot
        
        # Create minimal config for testing
        config = {
            'mysql': {
                'host': 'localhost',
                'user': 'root',
                'password': '',
                'database': 'trade_test'
            },
            'trading': {
                'symbols': ['BTCUSDT'],
                'timeframes': ['5m'],
                'exchange': 'binance',  # Use binance since we have collectors for it
                'demo_mode': True,
                'initial_balance': 100,
                'risk_per_trade': 0.02,
                'max_open_trades': 1,
                'initial_stop_loss': 0.03,
                'take_profit_levels': [0.05, 0.1, 0.15],
                'confidence_threshold': 0.7,
                'cooldown_period': 6,
                'max_open_risk': 0.06,
                'max_daily_loss': 0.05
            },
            'web': {
                'host': '0.0.0.0',
                'port': 5000
            },
            'model': {
                'hidden_dim': 64,  # Smaller for testing
                'n_layers': 1,     # Single layer for testing
                'n_heads': 2,      # Fewer heads for testing
                'dropout': 0.1,
                'learning_rate': 1e-4,
                'update_interval': 3600
            }
        }
        
        logger.info("Creating trading bot instance...")
        bot = TradingBot(config)
        
        logger.info("Bot created successfully! Testing signal generation with mock data...")
        
        # Test feature processing and signal generation
        import torch
        import pandas as pd
        import numpy as np
        
        # Create mock data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='5min')
        np.random.seed(42)
        
        # Generate realistic OHLCV data
        base_price = 50000
        returns = np.random.normal(0, 0.01, 30)
        prices = [base_price]
        
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))
        
        closes = np.array(prices[1:])
        highs = closes * (1 + np.random.uniform(0, 0.005, 30))
        lows = closes * (1 - np.random.uniform(0, 0.005, 30))
        opens = np.roll(closes, 1)
        opens[0] = base_price
        volumes = np.random.uniform(100, 1000, 30)
        
        mock_ohlcv = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=dates)
        
        # Process data through the encoders
        logger.info("Testing OHLCV encoder...")
        ohlcv_features = bot.encoders['ohlcv'].fit_transform(mock_ohlcv)
        logger.info(f"OHLCV features shape: {ohlcv_features.shape}")
        
        # Create mock indicators
        logger.info("Testing indicator encoder...")
        mock_indicators = {
            'sma': pd.Series(np.random.normal(50000, 1000, 30), index=dates),
            'ema': pd.Series(np.random.normal(50000, 1000, 30), index=dates),
            'rsi': pd.Series(np.random.uniform(20, 80, 30), index=dates),
            'macd': pd.Series(np.random.normal(0, 100, 30), index=dates),
            'bbands_upper': pd.Series(np.random.normal(51000, 1000, 30), index=dates)
        }
        indicator_features = bot.encoders['indicator'].fit_transform(mock_indicators)
        logger.info(f"Indicator features shape: {indicator_features.shape}")
        
        # Test sentiment encoder
        logger.info("Testing sentiment encoder...")
        mock_sentiment = {'fear_greed_index': 65}
        sentiment_features = bot.encoders['sentiment'].transform(mock_sentiment)
        logger.info(f"Sentiment features shape: {sentiment_features.shape}")
        
        # Test orderbook encoder
        logger.info("Testing orderbook encoder...")
        mock_orderbook = {
            'bids': [[49950, 1.5], [49940, 2.0], [49930, 1.8]],
            'asks': [[50050, 1.2], [50060, 1.7], [50070, 2.1]]
        }
        orderbook_features = bot.encoders['orderbook'].transform(mock_orderbook)
        logger.info(f"Orderbook features shape: {orderbook_features.shape}")
        
        # Test tick encoder
        logger.info("Testing tick encoder...")
        mock_tick_data = pd.DataFrame({
            'price': np.random.normal(50000, 100, 100),
            'volume': np.random.uniform(0.1, 2.0, 100),
            'is_buyer': np.random.choice([True, False], 100),
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1s')
        }).set_index('timestamp')
        tick_features = bot.encoders['tick_data'].fit_transform(mock_tick_data)
        logger.info(f"Tick features shape: {tick_features.shape}")
        
        # Create features dictionary
        features = {
            'ohlcv': ohlcv_features.unsqueeze(0),  # Add batch dimension
            'indicator': indicator_features.unsqueeze(0),
            'sentiment': sentiment_features,
            'orderbook': orderbook_features.unsqueeze(0),
            'tick_data': tick_features.unsqueeze(0)
        }
        
        logger.info("Testing feature gating...")
        gated_features = bot.gating(features)
        logger.info(f"Gated features groups: {list(gated_features.keys())}")
        
        for group, tensor in gated_features.items():
            logger.info(f"  {group}: {tensor.shape}")
        
        logger.info("Testing neural network model...")
        probs, confidence = bot.model(gated_features)
        logger.info(f"Model output - Probs: {probs.shape}, Confidence: {confidence.shape}")
        logger.info(f"Prediction probabilities: {probs[0].detach().numpy()}")
        logger.info(f"Confidence: {confidence[0].item():.4f}")
        
        # Test signal generation
        logger.info("Testing signal generator...")
        active_features = bot.gating.get_active_features()
        current_price = 50000
        
        signal = bot.signal_generator.generate_signal(
            gated_features, 
            active_features,
            current_price
        )
        
        if signal:
            logger.info(f"Generated signal: {signal['signal_type']} with confidence {signal['confidence']:.4f}")
        else:
            logger.info("No signal generated (confidence too low or cooldown)")
        
        logger.info("✅ All bot components working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Bot test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("=== Trading Bot Startup Test ===")
    success = test_bot_startup()
    
    if success:
        logger.info("🎉 Bot startup test passed!")
        exit(0)
    else:
        logger.error("❌ Bot startup test failed!")
        exit(1)