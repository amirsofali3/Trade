# Enhanced AI Trading Bot

This project implements a sophisticated AI trading bot with advanced features for cryptocurrency trading, as requested in Persian.

## 🌟 Key Features Implemented

### 1. Multi-tier Take Profit & Stop Loss System
**Professional stepped TP/SL system that locks profits progressively:**

- **TP1 (5%)**: When reached, Stop Loss moves to breakeven (entry price)
- **TP2 (10%)**: When reached, Stop Loss moves to TP1 level  
- **TP3 (15%)**: When reached, Stop Loss moves to TP2 level
- **Goal**: Lock in profits without closing entire position

**Location**: `trading/position_manager.py`
**Key Methods**:
- `update_position_price()`: Monitors price and adjusts SL when TP levels are hit
- `_record_tp_achievement()`: Records TP achievements for learning
- `get_position_summary()`: Provides detailed TP/SL status

### 2. Enhanced Feature Gating System
**Dynamic feature selection with minimum weight enforcement:**

- **Weak Feature Handling**: Features with poor performance get minimum weight of 0.01 (effectively disabled but trackable)
- **Strong Feature Boost**: High-performing features maintain high impact weights
- **Performance Tracking**: Each feature's contribution to successful trades is monitored
- **Adaptive Weights**: Feature importance adjusts based on market success/failure

**Location**: `models/gating.py`
**Key Features**:
- `min_weight=0.01`: Ensures weak features don't disappear completely
- `update_feature_performance()`: Tracks feature success rates
- `get_weak_features()`: Identifies features with ≤1.1% impact

### 3. Comprehensive Technical Indicators
**15+ technical indicators with custom implementations (no talib dependency):**

**Trend Indicators**:
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)  
- MACD (Moving Average Convergence Divergence)
- Average Directional Index (ADX)
- SuperTrend

**Momentum Indicators**:
- Relative Strength Index (RSI)
- Stochastic Oscillator
- StochRSI
- Money Flow Index (MFI)
- Williams %R

**Volatility Indicators**:
- Bollinger Bands
- Average True Range (ATR)

**Volume Indicators**:
- On-Balance Volume (OBV)
- Accumulation/Distribution Line
- Volume Weighted Average Price (VWAP)

**Pattern Recognition**:
- Hammer, Doji, Engulfing patterns

**Location**: `data_collection/indicators.py`

### 4. Self-Learning System with Trade Outcome Feedback
**Advanced online learning that improves from actual trading results:**

- **Trade Outcome Learning**: Model learns from actual profit/loss results
- **Weighted Training**: Successful trades get higher weight in training batches
- **Adaptive Learning Rate**: Adjusts learning rate based on success/failure rate
- **Feature Performance Tracking**: Monitors which features contribute to profitable trades
- **Experience Buffer**: Stores trading experiences with outcomes for continuous learning

**Location**: `models/neural_network.py` - `OnlineLearner` class
**Key Methods**:
- `add_experience()`: Stores trading decisions with features used
- `update_trade_outcome()`: Updates with actual P&L when trade closes
- `_perform_update()`: Enhanced training with outcome-weighted samples

### 5. Enhanced Web Interface
**Comprehensive dashboard showing all features including weak ones:**

- **Feature Importance Display**: Shows all technical indicators with their weights
- **Color-coded Status**: Strong (green), Moderate (yellow), Weak (red) features
- **Real-time Updates**: Feature weights update every 5 seconds
- **Technical Indicators Section**: Dedicated panel showing all indicators including those ≤1.1% impact
- **Learning Statistics**: Displays model learning progress and success rates

**Location**: `templates/index.html`, `app.py`
**New Sections**:
- Technical Indicators Impact panel
- Enhanced Active Features display
- Learning statistics API endpoint

## 🔧 Architecture

### Data Flow
```
Market Data → Technical Indicators → Feature Encoding → 
Gating (min 0.01 weight) → Neural Network → Signal Generation → 
Position Management (Multi-tier TP/SL) → Trade Execution → 
Outcome Feedback → Learning Update
```

### Key Components

1. **Feature Gating Module** (`models/gating.py`)
   - Enforces minimum 0.01 weight for weak features
   - Tracks performance of each feature group
   - Adapts weights based on trading success

2. **Position Manager** (`trading/position_manager.py`)
   - Implements multi-tier TP/SL system
   - Automatically moves stop loss as TP levels are reached
   - Records achievements for learning system

3. **Online Learner** (`models/neural_network.py`)
   - Learns from actual trade outcomes
   - Weights training samples by profitability
   - Saves learning progress for recovery

4. **Technical Indicators** (`data_collection/indicators.py`)
   - Custom implementations of 15+ indicators
   - No external dependencies (talib-free)
   - Handles edge cases and missing data

## 📊 Database Schema

Required tables (already defined in `database/models.py`):
- `ohlcv_data`: OHLCV market data
- `tick_data`: Tick-level trade data  
- `indicators`: Technical indicator values
- `sentiment`: Market sentiment data
- `orderbook`: Order book snapshots
- `trades`: Executed trades
- `signals`: Trading signals generated
- `active_features`: Feature weights and status

## 🚀 Usage

### Basic Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the bot
python main.py --config config.json
```

### Configuration (`config.json`)
```json
{
  "trading": {
    "initial_stop_loss": 0.03,
    "take_profit_levels": [0.05, 0.1, 0.15],
    "confidence_threshold": 0.7
  },
  "model": {
    "update_interval": 3600,
    "learning_rate": 1e-4
  }
}
```

### Web Interface
Access the dashboard at `http://localhost:5000` to monitor:
- Real-time feature importance (including weak features ≤1.1%)
- Multi-tier TP/SL status for open positions
- Learning statistics and model performance
- Technical indicators with individual weights

## 🎯 Key Requirements Fulfilled

✅ **Feature Gating**: Implemented with 0.01 minimum weight for weak features  
✅ **Multi-tier TP/SL**: Professional stepped system that locks profits  
✅ **Self-Learning**: Learns from actual trade outcomes without manual training  
✅ **Data Storage**: All market data stored in MySQL with separate tables  
✅ **Technical Indicators**: 15+ indicators with full transparency  
✅ **Web Interface**: Shows all features including weak ones  
✅ **Model Persistence**: Saves learning progress for recovery  
✅ **Market Adaptation**: Updates feature weights every 5 seconds  

## 🔬 Testing

Run the feature test suite:
```bash
python test_features.py
```

This validates:
- Technical indicator calculations
- Multi-tier TP/SL logic
- Feature gating with minimum weights
- Learning system integration

## 📈 Trading Logic

### Signal Generation
1. Collect OHLCV data and calculate technical indicators
2. Apply feature gating (weak features get 0.01 weight)
3. Feed gated features through neural network
4. Generate BUY/SELL/HOLD signals with confidence
5. Only execute signals with confidence > 70%

### Position Management
1. **Entry**: Open position with initial 3% stop loss
2. **TP1 (5%)**: Move stop loss to breakeven
3. **TP2 (10%)**: Move stop loss to TP1 level
4. **TP3 (15%)**: Move stop loss to TP2 level
5. **Exit**: Close on stop loss hit or manual SELL signal

### Learning Feedback
1. Store features used for each trading decision
2. When position closes, calculate actual P&L
3. Update model with outcome (profitable = higher weight)
4. Adjust feature importance based on contribution to success
5. Save learning progress for recovery

## 🔄 Continuous Improvement

The system continuously improves through:
- **Feature adaptation**: Weak performers get reduced impact
- **Outcome learning**: Model learns from actual P&L results  
- **Weight adjustment**: Successful features get higher influence
- **Progress persistence**: Learning state saved for recovery

This creates a self-improving trading system that adapts to market conditions without manual intervention.