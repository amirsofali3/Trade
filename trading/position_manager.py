import logging
import pandas as pd
from datetime import datetime
from database.db_manager import DatabaseManager

logger = logging.getLogger("PositionManager")

class PositionManager:
    """
    Manages trading positions and implements multi-tier take profit and stop loss system
    
    Features:
    - Multi-tier TP/SL: When price reaches TP1, move SL to entry price
    - At TP2, move SL to TP1, and so on
    - Goal: Lock in profits without closing entire position
    """
    
    def __init__(self, db_manager, initial_stop_loss=0.03, take_profit_levels=[0.05, 0.1, 0.15]):
        """
        Initialize position manager
        
        Args:
            db_manager: Database manager instance
            initial_stop_loss: Initial stop loss percentage
            take_profit_levels: List of take profit levels (percentages)
        """
        self.db_manager = db_manager
        self.initial_stop_loss = initial_stop_loss
        self.take_profit_levels = take_profit_levels
        
        # Track open positions
        self.positions = {}  # symbol -> position_info
        
        # Load positions from database
        self._load_positions()
        
        logger.info(f"Initialized PositionManager with SL={initial_stop_loss}, TP={take_profit_levels}")
        logger.info("Multi-tier TP/SL system: SL moves to previous TP level when new TP is reached")
    
    def _load_positions(self):
        """Load active positions from database"""
        try:
            session = self.db_manager.get_session()
            
            from database.models import Trade
            
            # Get recent BUY trades without matching SELL trades
            buy_trades = session.query(Trade).filter(Trade.trade_type == 'BUY').all()
            sell_trades = session.query(Trade).filter(Trade.trade_type == 'SELL').all()
            
            # Track sold positions by signal_id
            sold_signals = set()
            for sell in sell_trades:
                if sell.signal_id:
                    sold_signals.add(sell.signal_id)
            
            # Process active positions
            for trade in buy_trades:
                if trade.signal_id and trade.signal_id not in sold_signals:
                    # This is an active position
                    symbol = trade.symbol
                    entry_price = trade.price
                    amount = trade.amount
                    timestamp = trade.timestamp
                    
                    # Create position
                    self.positions[symbol] = {
                        'symbol': symbol,
                        'entry_price': entry_price,
                        'amount': amount,
                        'timestamp': timestamp,
                        'signal_id': trade.signal_id,
                        'stop_loss': entry_price * (1 - self.initial_stop_loss),
                        'take_profit_levels': [entry_price * (1 + level) for level in self.take_profit_levels],
                        'take_profit_triggered': [False] * len(self.take_profit_levels)
                    }
            
            logger.info(f"Loaded {len(self.positions)} active positions from database")
        
        except Exception as e:
            logger.error(f"Error loading positions: {str(e)}")
        
        finally:
            self.db_manager.close_session(session)
    
    def add_position(self, symbol, entry_price, amount, signal_id=None):
        """
        Add a new position
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            amount: Position size
            signal_id: Signal ID that generated this position
            
        Returns:
            Position info dictionary
        """
        # Create position
        position = {
            'symbol': symbol,
            'entry_price': entry_price,
            'amount': amount,
            'timestamp': datetime.now(),
            'signal_id': signal_id,
            'stop_loss': entry_price * (1 - self.initial_stop_loss),
            'take_profit_levels': [entry_price * (1 + level) for level in self.take_profit_levels],
            'take_profit_triggered': [False] * len(self.take_profit_levels)
        }
        
        # Store position
        self.positions[symbol] = position
        
        logger.info(f"Added position for {symbol}: {amount} @ {entry_price} (SL: {position['stop_loss']})")
        
        return position
    
    def remove_position(self, symbol):
        """
        Remove a position
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position info or None if not found
        """
        if symbol in self.positions:
            position = self.positions.pop(symbol)
            logger.info(f"Removed position for {symbol}")
            return position
        else:
            logger.warning(f"No position found for {symbol}")
            return None
    
    def update_position_price(self, symbol, current_price):
        """
        Update position with current price and check for TP/SL triggers
        Multi-tier system: SL moves to previous TP when new TP is reached
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Tuple of (action, position) where action is None, 'tp_adjust', 'tp_partial', or 'stop_loss'
        """
        if symbol not in self.positions:
            return None, None
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        
        # Check stop loss first
        if current_price <= position['stop_loss']:
            logger.info(f"Stop loss triggered for {symbol} at {current_price} (SL: {position['stop_loss']})")
            return 'stop_loss', position
        
        # Check take profit levels (multi-tier system)
        action = None
        for i, (tp_level, triggered) in enumerate(zip(position['take_profit_levels'], position['take_profit_triggered'])):
            if not triggered and current_price >= tp_level:
                # This take profit level is reached
                position['take_profit_triggered'][i] = True
                
                # Calculate percentage reached
                tp_percent = self.take_profit_levels[i] * 100
                
                # Multi-tier stop loss adjustment
                if i == 0:
                    # First TP reached - move SL to entry price (breakeven)
                    new_sl = entry_price
                    position['stop_loss'] = new_sl
                    logger.info(f"TP1 ({tp_percent:.1f}%) reached for {symbol}. Stop loss moved to breakeven: {new_sl}")
                    action = 'tp_adjust'
                    
                elif i == 1:
                    # Second TP reached - move SL to first TP level
                    new_sl = position['take_profit_levels'][0]
                    position['stop_loss'] = new_sl
                    logger.info(f"TP2 ({tp_percent:.1f}%) reached for {symbol}. Stop loss moved to TP1: {new_sl}")
                    action = 'tp_adjust'
                    
                elif i >= 2:
                    # Third+ TP reached - move SL to previous TP level
                    new_sl = position['take_profit_levels'][i-1]
                    position['stop_loss'] = new_sl
                    logger.info(f"TP{i+1} ({tp_percent:.1f}%) reached for {symbol}. Stop loss moved to TP{i}: {new_sl}")
                    action = 'tp_adjust'
                
                # Log detailed information
                logger.info(f"Position {symbol} TP progression: " + 
                          ", ".join([f"TP{j+1}({'✓' if trig else '✗'})" for j, trig in enumerate(position['take_profit_triggered'])]))
                
                # Record TP achievement in database for learning
                self._record_tp_achievement(symbol, i+1, current_price, tp_level)
                
                # Return after first TP trigger (process one at a time)
                return action, position
        
        # No triggers, just return updated position
        return None, position
    
    def _record_tp_achievement(self, symbol, tp_level, current_price, tp_price):
        """
        Record take profit achievement for learning purposes
        
        Args:
            symbol: Trading symbol
            tp_level: TP level number (1, 2, 3, etc.)
            current_price: Price when TP was achieved
            tp_price: Target TP price
        """
        try:
            # This could be expanded to save TP achievements to database
            # for model learning and performance analysis
            logger.info(f"TP{tp_level} achieved for {symbol}: target={tp_price:.4f}, actual={current_price:.4f}")
            
            # Future enhancement: Save to database for learning
            # tp_data = {
            #     'symbol': symbol,
            #     'tp_level': tp_level,
            #     'target_price': tp_price,
            #     'actual_price': current_price,
            #     'timestamp': datetime.now()
            # }
            # self.db_manager.save_tp_achievement(tp_data)
            
        except Exception as e:
            logger.error(f"Error recording TP achievement: {str(e)}")
    
    def get_position_summary(self, symbol):
        """
        Get detailed position summary including TP/SL status
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with position details
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        # Calculate unrealized P&L would need current price
        # This is a placeholder for current_price parameter
        summary = {
            'symbol': symbol,
            'entry_price': position['entry_price'],
            'amount': position['amount'],
            'timestamp': position['timestamp'],
            'stop_loss': position['stop_loss'],
            'take_profit_levels': position['take_profit_levels'],
            'take_profit_triggered': position['take_profit_triggered'],
            'tp_status': [
                {
                    'level': i+1,
                    'target': tp,
                    'triggered': triggered,
                    'percentage': f"{self.take_profit_levels[i]*100:.1f}%"
                }
                for i, (tp, triggered) in enumerate(zip(position['take_profit_levels'], position['take_profit_triggered']))
            ]
        }
        
        return summary
    
    def get_position(self, symbol):
        """Get position by symbol"""
        return self.positions.get(symbol)
    
    def get_all_positions(self):
        """Get all positions"""
        return list(self.positions.values())
    
    def has_position(self, symbol):
        """Check if position exists for symbol"""
        return symbol in self.positions
    
    def get_position_count(self):
        """Get number of open positions"""
        return len(self.positions)