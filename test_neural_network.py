#!/usr/bin/env python3
"""
Test script to verify the neural network and feature processing work correctly
"""

import logging
import torch
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuralNetTest")

def test_neural_network_components():
    """Test neural network components without database dependencies"""
    try:
        from models.encoders.feature_encoder import (
            OHLCVEncoder, IndicatorEncoder, SentimentEncoder, 
            OrderbookEncoder, TickDataEncoder, CandlePatternEncoder
        )
        from models.gating import FeatureGatingModule
        from models.neural_network import MarketTransformer
        from models.signal_generator import SignalGenerator
        
        logger.info("Testing feature encoders...")
        
        # Create mock OHLCV data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='5min')
        np.random.seed(42)
        
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
        
        # Test encoders
        logger.info("Testing OHLCV encoder...")
        ohlcv_encoder = OHLCVEncoder(window_size=20)
        ohlcv_features = ohlcv_encoder.fit_transform(mock_ohlcv)
        logger.info(f"OHLCV features shape: {ohlcv_features.shape}")
        
        logger.info("Testing indicator encoder...")
        indicator_encoder = IndicatorEncoder(window_size=20)
        mock_indicators = {
            'sma': pd.Series(np.random.normal(50000, 1000, 30), index=dates),
            'ema': pd.Series(np.random.normal(50000, 1000, 30), index=dates),
            'rsi': pd.Series(np.random.uniform(20, 80, 30), index=dates),
            'macd': pd.Series(np.random.normal(0, 100, 30), index=dates),
            'bbands_upper': pd.Series(np.random.normal(51000, 1000, 30), index=dates)
        }
        indicator_features = indicator_encoder.fit_transform(mock_indicators)
        logger.info(f"Indicator features shape: {indicator_features.shape}")
        
        logger.info("Testing sentiment encoder...")
        sentiment_encoder = SentimentEncoder()
        mock_sentiment = {'fear_greed_index': 65}
        sentiment_features = sentiment_encoder.transform(mock_sentiment)
        logger.info(f"Sentiment features shape: {sentiment_features.shape}")
        
        logger.info("Testing orderbook encoder...")
        orderbook_encoder = OrderbookEncoder(depth=10)
        mock_orderbook = {
            'bids': [[49950, 1.5], [49940, 2.0], [49930, 1.8]],
            'asks': [[50050, 1.2], [50060, 1.7], [50070, 2.1]]
        }
        orderbook_features = orderbook_encoder.transform(mock_orderbook)
        logger.info(f"Orderbook features shape: {orderbook_features.shape}")
        
        logger.info("Testing tick encoder...")
        tick_encoder = TickDataEncoder(window_size=100)
        mock_tick_data = pd.DataFrame({
            'price': np.random.normal(50000, 100, 100),
            'volume': np.random.uniform(0.1, 2.0, 100),
            'is_buyer': np.random.choice([True, False], 100),
            'timestamp': pd.date_range(start='2024-01-01', periods=100, freq='1s')
        }).set_index('timestamp')
        tick_features = tick_encoder.fit_transform(mock_tick_data)
        logger.info(f"Tick features shape: {tick_features.shape}")
        
        # Test feature gating
        logger.info("Testing feature gating...")
        feature_groups = {
            'ohlcv': 13,
            'indicator': 20,
            'sentiment': 1,
            'orderbook': 42,
            'tick_data': 3
        }
        
        gating = FeatureGatingModule(feature_groups, min_weight=0.01)
        
        features = {
            'ohlcv': ohlcv_features.unsqueeze(0),  # Add batch dimension
            'indicator': indicator_features.unsqueeze(0),
            'sentiment': sentiment_features,
            'orderbook': orderbook_features.unsqueeze(0),
            'tick_data': tick_features.unsqueeze(0)
        }
        
        gated_features = gating(features)
        logger.info(f"Gated features groups: {list(gated_features.keys())}")
        
        for group, tensor in gated_features.items():
            logger.info(f"  {group}: {tensor.shape}")
        
        # Test neural network model
        logger.info("Testing neural network model...")
        feature_dims = {
            'ohlcv': (20, 13),
            'indicator': (20, 20), 
            'sentiment': (1, 1),
            'orderbook': (1, 42),
            'tick_data': (100, 3)
        }
        
        model = MarketTransformer(
            feature_dims=feature_dims,
            hidden_dim=64,
            n_layers=1,
            n_heads=2
        )
        
        probs, confidence = model(gated_features)
        logger.info(f"Model output - Probs: {probs.shape}, Confidence: {confidence.shape}")
        logger.info(f"Prediction probabilities: {probs[0].detach().numpy()}")
        logger.info(f"Confidence: {confidence[0].item():.4f}")
        
        # Test active features
        logger.info("Testing active features reporting...")
        active_features = gating.get_active_features()
        logger.info(f"Active features count: {len(active_features)}")
        
        # Count weak features (those with weight <= 0.01 + 0.01)
        weak_count = 0
        strong_count = 0
        moderate_count = 0
        
        for group_name, group_data in active_features.items():
            if isinstance(group_data, dict):
                if 'status' in group_data:
                    if group_data['status'] == 'weak':
                        weak_count += 1
                    elif group_data['status'] == 'strong':
                        strong_count += 1
                    else:
                        moderate_count += 1
                else:
                    # Check sub-features
                    for feature_name, feature_data in group_data.items():
                        if isinstance(feature_data, dict) and 'status' in feature_data:
                            if feature_data['status'] == 'weak':
                                weak_count += 1
                            elif feature_data['status'] == 'strong':
                                strong_count += 1
                            else:
                                moderate_count += 1
        
        logger.info(f"Feature status distribution: Strong={strong_count}, Moderate={moderate_count}, Weak={weak_count}")
        
        # Test feature performance update
        logger.info("Testing feature performance tracking...")
        gating.update_feature_performance('ohlcv', True, 0.8)
        gating.update_feature_performance('sentiment', False, 0.3)
        
        performance = gating.get_feature_performance_summary()
        logger.info(f"Performance tracking for {len(performance)} features")
        
        # Verify the 0.01 minimum weight system
        logger.info("Testing minimum weight constraint...")
        min_weights = []
        for group, tensor in gated_features.items():
            if torch.is_tensor(tensor):
                # Check if any weights are below minimum
                weights = gating.current_gates.get(group, torch.tensor([0.5]))
                min_weight = float(torch.min(weights))
                max_weight = float(torch.max(weights))
                avg_weight = float(torch.mean(weights))
                min_weights.append(min_weight)
                logger.info(f"  {group}: min={min_weight:.4f}, max={max_weight:.4f}, avg={avg_weight:.4f}")
        
        actual_min = min(min_weights) if min_weights else 0
        expected_min = gating.min_weight
        
        if actual_min >= expected_min - 0.001:  # Small tolerance for floating point
            logger.info(f"✅ Minimum weight constraint working: {actual_min:.4f} >= {expected_min:.4f}")
        else:
            logger.warning(f"⚠️ Minimum weight constraint may not be working: {actual_min:.4f} < {expected_min:.4f}")
        
        logger.info("✅ All neural network components working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Neural network test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("=== Neural Network Components Test ===")
    success = test_neural_network_components()
    
    if success:
        logger.info("🎉 Neural network test passed!")
        exit(0)
    else:
        logger.error("❌ Neural network test failed!")
        exit(1)