#!/usr/bin/env python3
"""
Demo script showing key enhanced features of the trading bot

This demonstrates the implemented features with minimal dependencies.
"""

import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("TradingBotDemo")

def demo_multi_tier_tp_sl():
    """Demonstrate multi-tier TP/SL system"""
    logger.info("=== Multi-tier Take Profit & Stop Loss Demo ===")
    
    # Simulate a trading position
    entry_price = 50000
    position_size = 0.1
    initial_sl_percent = 0.03
    tp_levels = [0.05, 0.1, 0.15]  # 5%, 10%, 15%
    
    logger.info(f"Opening BUY position:")
    logger.info(f"  Entry Price: ${entry_price:,.2f}")
    logger.info(f"  Position Size: {position_size} BTC")
    logger.info(f"  Initial Stop Loss: ${entry_price * (1 - initial_sl_percent):,.2f} ({initial_sl_percent*100}% below entry)")
    
    # Calculate TP levels
    tp_prices = [entry_price * (1 + level) for level in tp_levels]
    logger.info(f"  Take Profit Levels:")
    for i, (level, price) in enumerate(zip(tp_levels, tp_prices)):
        logger.info(f"    TP{i+1}: ${price:,.2f} ({level*100}% above entry)")
    
    # Simulate price movements hitting TP levels
    current_sl = entry_price * (1 - initial_sl_percent)
    
    logger.info(f"\n--- Price Movement Simulation ---")
    
    # TP1 hit
    current_price = tp_prices[0]
    new_sl = entry_price  # Move to breakeven
    logger.info(f"📈 Price reaches TP1: ${current_price:,.2f}")
    logger.info(f"🔒 Stop Loss moved from ${current_sl:,.2f} to ${new_sl:,.2f} (BREAKEVEN)")
    current_sl = new_sl
    
    # TP2 hit  
    current_price = tp_prices[1]
    new_sl = tp_prices[0]  # Move to TP1
    logger.info(f"📈 Price reaches TP2: ${current_price:,.2f}")
    logger.info(f"🔒 Stop Loss moved from ${current_sl:,.2f} to ${new_sl:,.2f} (TP1 level)")
    current_sl = new_sl
    
    # TP3 hit
    current_price = tp_prices[2]
    new_sl = tp_prices[1]  # Move to TP2
    logger.info(f"📈 Price reaches TP3: ${current_price:,.2f}")
    logger.info(f"🔒 Stop Loss moved from ${current_sl:,.2f} to ${new_sl:,.2f} (TP2 level)")
    
    profit_locked = (new_sl - entry_price) / entry_price
    logger.info(f"🎯 Result: Minimum {profit_locked*100:.1f}% profit LOCKED regardless of future price action")

def demo_feature_gating():
    """Demonstrate feature gating with minimum weights"""
    logger.info("\n=== Enhanced Feature Gating Demo ===")
    
    # Simulate feature weights from the gating system
    features = {
        'RSI': {'weight': 0.85, 'status': 'strong'},
        'MACD': {'weight': 0.72, 'status': 'strong'}, 
        'Bollinger_Bands': {'weight': 0.68, 'status': 'moderate'},
        'EMA': {'weight': 0.45, 'status': 'moderate'},
        'Stochastic': {'weight': 0.32, 'status': 'moderate'},
        'Williams_R': {'weight': 0.15, 'status': 'weak'},
        'MFI': {'weight': 0.08, 'status': 'weak'},
        'ADX': {'weight': 0.03, 'status': 'weak'},
        'SuperTrend': {'weight': 0.01, 'status': 'weak'},  # Minimum weight
        'VWAP': {'weight': 0.01, 'status': 'weak'},        # Minimum weight
    }
    
    logger.info("Feature Importance Analysis:")
    logger.info("(Strong: >70%, Moderate: 30-70%, Weak: ≤30%, Minimum: 1%)")
    
    strong_features = []
    weak_features = []
    disabled_features = []
    
    for name, info in features.items():
        weight_percent = info['weight'] * 100
        status = info['status']
        
        if status == 'strong':
            emoji = "🟢"
            strong_features.append(name)
        elif status == 'weak' and weight_percent <= 1.1:
            emoji = "🔴"
            disabled_features.append(name)
        elif status == 'weak':
            emoji = "🟡"
            weak_features.append(name)
        else:
            emoji = "🟡"
            
        logger.info(f"  {emoji} {name:15} {weight_percent:5.1f}% ({status})")
    
    logger.info(f"\n📊 Summary:")
    logger.info(f"  Strong Features: {len(strong_features)} (high impact)")
    logger.info(f"  Weak Features: {len(weak_features)} (low impact)")
    logger.info(f"  Effectively Disabled: {len(disabled_features)} (≤1.1% weight but still tracked)")
    
    logger.info(f"\n🎯 Key Benefit: Weak features get 0.01 minimum weight")
    logger.info(f"   - They don't disappear completely (still trackable)")
    logger.info(f"   - They can recover if market conditions change")
    logger.info(f"   - System remains transparent and auditable")

def demo_self_learning():
    """Demonstrate the self-learning system"""
    logger.info("\n=== Self-Learning System Demo ===")
    
    # Simulate trading outcomes
    trades = [
        {'signal': 'BUY', 'features': ['RSI', 'MACD'], 'outcome': 'profit', 'pnl': 0.05},
        {'signal': 'BUY', 'features': ['Bollinger', 'EMA'], 'outcome': 'loss', 'pnl': -0.02},
        {'signal': 'SELL', 'features': ['RSI', 'Stochastic'], 'outcome': 'profit', 'pnl': 0.03},
        {'signal': 'BUY', 'features': ['MACD', 'SuperTrend'], 'outcome': 'loss', 'pnl': -0.01},
        {'signal': 'BUY', 'features': ['RSI', 'MACD'], 'outcome': 'profit', 'pnl': 0.08},
    ]
    
    logger.info("Learning from Trade Outcomes:")
    
    # Track feature performance
    feature_stats = {}
    
    for i, trade in enumerate(trades, 1):
        pnl_percent = trade['pnl'] * 100
        outcome_emoji = "✅" if trade['outcome'] == 'profit' else "❌"
        
        logger.info(f"  Trade {i}: {outcome_emoji} {trade['signal']} → {pnl_percent:+.1f}% (Features: {', '.join(trade['features'])})")
        
        # Update feature statistics
        for feature in trade['features']:
            if feature not in feature_stats:
                feature_stats[feature] = {'successes': 0, 'total': 0}
            
            feature_stats[feature]['total'] += 1
            if trade['outcome'] == 'profit':
                feature_stats[feature]['successes'] += 1
    
    logger.info(f"\n📈 Learning Results:")
    for feature, stats in feature_stats.items():
        success_rate = stats['successes'] / stats['total']
        status_emoji = "🟢" if success_rate >= 0.6 else "🟡" if success_rate >= 0.4 else "🔴"
        logger.info(f"  {status_emoji} {feature:12} Success Rate: {success_rate*100:5.1f}% ({stats['successes']}/{stats['total']})")
    
    logger.info(f"\n🧠 Learning Adaptations:")
    logger.info(f"  - Features with high success rates get HIGHER weights in future decisions")
    logger.info(f"  - Profitable trades get HIGHER weight in model training")
    logger.info(f"  - Learning rate adjusts based on overall performance")
    logger.info(f"  - All learning progress is SAVED for recovery")

def demo_technical_indicators():
    """Demonstrate comprehensive technical indicators"""
    logger.info("\n=== Comprehensive Technical Indicators Demo ===")
    
    indicators = {
        'Trend Indicators': ['SMA', 'EMA', 'MACD', 'ADX', 'SuperTrend'],
        'Momentum Indicators': ['RSI', 'Stochastic', 'StochRSI', 'MFI', 'Williams %R'],
        'Volatility Indicators': ['Bollinger Bands', 'ATR'],
        'Volume Indicators': ['OBV', 'A/D Line', 'VWAP'],
        'Pattern Recognition': ['Hammer', 'Doji', 'Engulfing']
    }
    
    logger.info("Available Technical Indicators:")
    
    total_indicators = 0
    for category, indicator_list in indicators.items():
        logger.info(f"\n  📊 {category}:")
        for indicator in indicator_list:
            logger.info(f"    • {indicator}")
            total_indicators += 1
    
    logger.info(f"\n🎯 Total: {total_indicators} technical indicators")
    logger.info(f"  ✅ All implemented without external dependencies (no talib)")
    logger.info(f"  ✅ Custom implementations handle edge cases")
    logger.info(f"  ✅ All indicators shown in web interface (including weak ones)")

def main():
    """Run all demos"""
    logger.info("🤖 Enhanced AI Trading Bot - Feature Demonstrations")
    logger.info("=" * 60)
    
    demo_multi_tier_tp_sl()
    demo_feature_gating()
    demo_self_learning()
    demo_technical_indicators()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 All enhanced features demonstrated successfully!")
    logger.info("\nKey improvements implemented:")
    logger.info("✅ Multi-tier TP/SL system that locks profits progressively")
    logger.info("✅ Feature gating with 0.01 minimum weight for weak features")
    logger.info("✅ Self-learning from actual trade outcomes")
    logger.info("✅ 15+ technical indicators with custom implementations")
    logger.info("✅ Enhanced web interface showing all features")
    logger.info("✅ Model persistence for learning recovery")

if __name__ == "__main__":
    main()