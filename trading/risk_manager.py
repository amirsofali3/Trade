import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger("RiskManager")

class RiskManager:
    """
    Manages trading risk including position sizing, max drawdown protection,
    and market volatility assessment
    """
    
    def __init__(self, db_manager, max_risk_per_trade=0.02, max_open_risk=0.06, 
                max_daily_loss=0.05, initial_capital=100):
        """
        Initialize risk manager
        
        Args:
            db_manager: Database manager instance
            max_risk_per_trade: Maximum risk per trade as percentage of capital (0-1)
            max_open_risk: Maximum total risk for all open positions (0-1)
            max_daily_loss: Maximum allowed daily loss (0-1)
            initial_capital: Initial capital amount
        """
        self.db_manager = db_manager
        self.max_risk_per_trade = max_risk_per_trade
        self.max_open_risk = max_open_risk
        self.max_daily_loss = max_daily_loss
        self.initial_capital = initial_capital
        
        # Track daily losses
        self.daily_loss = 0
        self.daily_loss_reset_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        
        logger.info(f"Initialized RiskManager with max_risk_per_trade={max_risk_per_trade}, max_daily_loss={max_daily_loss}")
    
    def calculate_position_size(self, symbol, entry_price, stop_loss, current_capital=None):
        """
        Calculate appropriate position size based on risk parameters
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            current_capital: Current capital (uses initial_capital if None)
            
        Returns:
            Position size amount
        """
        if current_capital is None:
            current_capital = self.initial_capital
        
        # Calculate risk per trade in currency units
        risk_amount = current_capital * self.max_risk_per_trade
        
        # Calculate position size based on stop loss distance
        if entry_price <= stop_loss:
            logger.warning(f"Invalid stop loss: entry={entry_price}, stop={stop_loss}")
            return 0
        
        # Risk per unit
        risk_per_unit = entry_price - stop_loss
        
        # Position size
        position_size = risk_amount / risk_per_unit
        
        logger.info(f"Calculated position size for {symbol}: {position_size} (risk: {risk_amount}, stop distance: {risk_per_unit})")
        
        return position_size
    
    def check_max_open_risk(self, positions, new_position=None):
        """
        Check if adding a new position would exceed maximum open risk
        
        Args:
            positions: List of current open positions
            new_position: New position to add (or None to just check current positions)
            
        Returns:
            (is_allowed, current_risk_percentage)
        """
        # Calculate current open risk
        total_risk = 0
        
        for position in positions:
            entry = position.get('entry_price', 0)
            stop = position.get('stop_loss', 0)
            amount = position.get('amount', 0)
            
            # Calculate risk for this position
            if entry > stop:
                position_risk = (entry - stop) * amount
                total_risk += position_risk
        
        # Add new position if provided
        if new_position:
            entry = new_position.get('entry_price', 0)
            stop = new_position.get('stop_loss', 0)
            amount = new_position.get('amount', 0)
            
            if entry > stop:
                position_risk = (entry - stop) * amount
                total_risk += position_risk
        
        # Calculate risk percentage
        risk_percentage = total_risk / self.initial_capital
        
        # Check against max open risk
        is_allowed = risk_percentage <= self.max_open_risk
        
        if not is_allowed:
            logger.warning(f"Maximum open risk exceeded: {risk_percentage:.2%} > {self.max_open_risk:.2%}")
        
        return is_allowed, risk_percentage
    
    def update_daily_loss(self, pnl):
        """
        Update daily loss tracker
        
        Args:
            pnl: Profit/Loss amount (negative for loss)
            
        Returns:
            (can_trade, current_daily_loss_percentage)
        """
        # Check if we need to reset daily loss (new day)
        now = datetime.now()
        if now >= self.daily_loss_reset_time:
            self.daily_loss = 0
            self.daily_loss_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            logger.info("Reset daily loss tracker for new day")
        
        # Update if loss
        if pnl < 0:
            self.daily_loss += abs(pnl)
        
        # Calculate percentage of initial capital
        loss_percentage = self.daily_loss / self.initial_capital
        
        # Check against max daily loss
        can_trade = loss_percentage < self.max_daily_loss
        
        if not can_trade:
            logger.warning(f"Maximum daily loss reached: {loss_percentage:.2%} >= {self.max_daily_loss:.2%}")
        
        return can_trade, loss_percentage
    
    def calculate_market_volatility(self, symbol, timeframe='5m', lookback=20):
        """
        Calculate current market volatility
        
        Args:
            symbol: Trading symbol
            timeframe: Candle timeframe
            lookback: Number of candles to consider
            
        Returns:
            Volatility percentage
        """
        try:
            # Get OHLCV data
            ohlcv = self.db_manager.get_ohlcv(symbol, timeframe, limit=lookback)
            
            if ohlcv.empty:
                logger.warning(f"No OHLCV data for volatility calculation ({symbol} {timeframe})")
                return None
            
            # Calculate returns
            returns = ohlcv['close'].pct_change().dropna()
            
            # Calculate volatility (standard deviation of returns)
            volatility = returns.std() * np.sqrt(288)  # Annualized (288 5-minute candles in a day)
            
            return float(volatility)
            
        except Exception as e:
            logger.error(f"Error calculating market volatility: {str(e)}")
            return None
    
    def adjust_risk_for_volatility(self, base_risk, volatility):
        """
        Adjust risk percentage based on market volatility
        
        Args:
            base_risk: Base risk percentage (0-1)
            volatility: Market volatility
            
        Returns:
            Adjusted risk percentage
        """
        if volatility is None:
            return base_risk
        
        # Base volatility assumption
        normal_volatility = 0.02  # 2% daily volatility is considered normal
        
        # Adjust risk - lower risk in high volatility, higher risk in low volatility
        volatility_ratio = normal_volatility / max(volatility, 0.001)  # Prevent division by zero
        
        # Cap the adjustment
        volatility_ratio = max(0.5, min(1.5, volatility_ratio))
        
        adjusted_risk = base_risk * volatility_ratio
        
        # Ensure within limits
        adjusted_risk = max(base_risk * 0.5, min(base_risk * 1.5, adjusted_risk))
        
        logger.debug(f"Risk adjusted for volatility: {base_risk:.2%} -> {adjusted_risk:.2%} (volatility: {volatility:.2%})")
        
        return adjusted_risk
    
    def can_open_position(self, symbol, entry_price, stop_loss, position_size, positions):
        """
        Check if a new position can be opened considering all risk factors
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            position_size: Position size amount
            positions: Current open positions
            
        Returns:
            (can_open, reason) tuple
        """
        # Check if stop loss is valid
        if entry_price <= stop_loss:
            return False, "Invalid stop loss"
        
        # Create position object for risk check
        new_position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'amount': position_size
        }
        
        # Check maximum open risk
        max_risk_allowed, current_risk = self.check_max_open_risk(positions, new_position)
        if not max_risk_allowed:
            return False, f"Maximum open risk exceeded ({current_risk:.2%})"
        
        # All checks passed
        return True, "Position allowed"