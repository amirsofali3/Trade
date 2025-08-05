import logging
import pandas as pd
from datetime import datetime
from database.db_manager import DatabaseManager

logger = logging.getLogger("PositionManager")

class PositionManager:
    """
    Manages trading positions and implements step-by-step take profit and stop loss
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
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            Tuple of (action, position) where action is None, 'tp_adjust', or 'stop_loss'
        """
        if symbol not in self.positions:
            return None, None
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        
        # Check stop loss
        if current_price <= position['stop_loss']:
            logger.info(f"Stop loss triggered for {symbol} at {current_price}")
            return 'stop_loss', position
        
        # Check take profit levels
        for i, (tp_level, triggered) in enumerate(zip(position['take_profit_levels'], position['take_profit_triggered'])):
            if not triggered and current_price >= tp_level:
                # This take profit level is reached
                position['take_profit_triggered'][i] = True
                
                # Move stop loss to previous take profit level or entry price
                if i > 0:
                    position['stop_loss'] = position['take_profit_levels'][i-1]
                else:
                    position['stop_loss'] = entry_price
                
                logger.info(f"Take profit {i+1} triggered for {symbol}. Stop loss moved to {position['stop_loss']}")
                return 'tp_adjust', position
        
        # No triggers, just return updated position
        return None, position
    
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