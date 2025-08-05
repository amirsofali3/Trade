import logging
import time
import json
import hmac
import hashlib
import requests
from datetime import datetime
import threading
import ccxt
import pandas as pd
from database.db_manager import DatabaseManager

logger = logging.getLogger("TradeExecutor")

class TradeExecutor:
    """
    Executes trades based on generated signals
    """
    
    def __init__(self, db_manager, exchange_name='coinex', demo_mode=True, 
                api_key=None, api_secret=None, initial_balance=100, 
                risk_per_trade=0.02, max_open_trades=1):
        """
        Initialize trade executor
        
        Args:
            db_manager: Database manager instance
            exchange_name: Name of exchange to use
            demo_mode: Whether to run in demo mode
            api_key: API key for exchange
            api_secret: API secret for exchange
            initial_balance: Initial balance for demo mode
            risk_per_trade: Percentage of balance to risk per trade (0-1)
            max_open_trades: Maximum number of open trades
        """
        self.db_manager = db_manager
        self.exchange_name = exchange_name.lower()
        self.demo_mode = demo_mode
        self.api_key = api_key
        self.api_secret = api_secret
        self.risk_per_trade = risk_per_trade
        self.max_open_trades = max_open_trades
        
        # Demo account state
        self.demo_balance = initial_balance
        self.demo_positions = []
        
        # Exchange connection
        self.exchange = None
        if not demo_mode:
            self._init_exchange()
        
        # Running state
        self.running = False
        self.thread = None
        
        logger.info(f"Initialized TradeExecutor with exchange={exchange_name}, demo_mode={demo_mode}")
    
    def _init_exchange(self):
        """Initialize exchange connection"""
        try:
            if self.exchange_name == 'coinex':
                self.exchange = ccxt.coinex({
                    'apiKey': self.api_key,
                    'secret': self.api_secret,
                    'enableRateLimit': True,
                })
            else:
                raise ValueError(f"Unsupported exchange: {self.exchange_name}")
            
            # Test connection
            self.exchange.load_markets()
            logger.info(f"Connected to {self.exchange_name} exchange")
        
        except Exception as e:
            logger.error(f"Error connecting to exchange: {str(e)}")
            raise
    
    def start(self):
        """Start trade executor"""
        if self.running:
            logger.warning("Trade executor is already running")
            return
        
        self.running = True
        
        # Start execution thread
        self.thread = threading.Thread(target=self._execution_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info("Trade executor started")
    
    def stop(self):
        """Stop trade executor"""
        self.running = False
        logger.info("Trade executor stopped")
    
    def _execution_loop(self):
        """Main execution loop"""
        while self.running:
            try:
                # Check for unexecuted signals
                session = self.db_manager.get_session()
                
                from database.models import Signal
                unexecuted = session.query(Signal).filter_by(executed=False).all()
                
                for signal in unexecuted:
                    # Check if signal is still valid (not too old)
                    age = (datetime.now() - signal.timestamp).total_seconds()
                    
                    # Skip signals older than 10 minutes
                    if age > 600:
                        logger.warning(f"Skipping old signal: {signal.id} from {signal.timestamp}")
                        signal.executed = True
                        continue
                    
                    # Execute signal
                    self._execute_signal(signal)
                    
                    # Mark as executed
                    signal.executed = True
                
                session.commit()
                self.db_manager.close_session(session)
                
                # Sleep to prevent high CPU usage
                time.sleep(5)
            
            except Exception as e:
                logger.error(f"Error in execution loop: {str(e)}")
                time.sleep(10)
    
    def _execute_signal(self, signal):
        """
        Execute a trading signal
        
        Args:
            signal: Signal object from database
        """
        try:
            if self.demo_mode:
                self._execute_demo(signal)
            else:
                self._execute_live(signal)
        
        except Exception as e:
            logger.error(f"Error executing signal {signal.id}: {str(e)}")
    
    def _execute_demo(self, signal):
        """Execute signal in demo mode"""
        symbol = signal.symbol
        signal_type = signal.signal_type
        price = signal.price
        timestamp = signal.timestamp
        
        logger.info(f"Demo executing {signal_type} for {symbol} at {price}")
        
        if signal_type == 'BUY':
            # Check if we already have a position
            if len(self.demo_positions) >= self.max_open_trades:
                logger.warning(f"Maximum open trades reached ({self.max_open_trades}), skipping buy signal")
                return
            
            # Calculate amount to buy
            risk_amount = self.demo_balance * self.risk_per_trade
            amount = risk_amount / price
            cost = amount * price
            fee = cost * 0.001  # Assume 0.1% fee
            
            # Update balance
            self.demo_balance -= cost + fee
            
            # Add position
            position = {
                'symbol': symbol,
                'entry_price': price,
                'amount': amount,
                'timestamp': timestamp,
                'signal_id': signal.id,
            }
            
            self.demo_positions.append(position)
            
            # Save trade to database
            trade_data = {
                'symbol': symbol,
                'timestamp': timestamp,
                'trade_type': 'BUY',
                'price': price,
                'amount': amount,
                'cost': cost,
                'fee': fee,
                'realized_pnl': 0,
                'signal_id': signal.id
            }
            
            self.db_manager.save_trade(trade_data)
            
            logger.info(f"Demo bought {amount} {symbol} at {price} (balance: {self.demo_balance})")
        
        elif signal_type == 'SELL':
            # Check if we have a position to sell
            if not self.demo_positions:
                logger.warning("No positions to sell, skipping sell signal")
                return
            
            # Find position for this symbol
            position_idx = None
            for i, pos in enumerate(self.demo_positions):
                if pos['symbol'] == symbol:
                    position_idx = i
                    break
            
            if position_idx is None:
                logger.warning(f"No position for {symbol}, skipping sell signal")
                return
            
            # Get position details
            position = self.demo_positions[position_idx]
            amount = position['amount']
            entry_price = position['entry_price']
            
            # Calculate proceeds
            proceeds = amount * price
            fee = proceeds * 0.001  # Assume 0.1% fee
            net_proceeds = proceeds - fee
            
            # Calculate PnL
            cost = amount * entry_price
            pnl = net_proceeds - cost
            pnl_percent = (price / entry_price - 1) * 100
            
            # Update balance
            self.demo_balance += net_proceeds
            
            # Remove position
            self.demo_positions.pop(position_idx)
            
            # Save trade to database
            trade_data = {
                'symbol': symbol,
                'timestamp': timestamp,
                'trade_type': 'SELL',
                'price': price,
                'amount': amount,
                'cost': proceeds,
                'fee': fee,
                'realized_pnl': pnl,
                'signal_id': signal.id
            }
            
            self.db_manager.save_trade(trade_data)
            
            logger.info(f"Demo sold {amount} {symbol} at {price} (PnL: {pnl_percent:.2f}%, balance: {self.demo_balance})")
    
    def _execute_live(self, signal):
        """Execute signal in live mode"""
        try:
            if not self.exchange:
                logger.error("Exchange not initialized")
                return
            
            symbol = signal.symbol
            signal_type = signal.signal_type
            price = signal.price
            
            # Format symbol for exchange
            exchange_symbol = f"{symbol[:3]}/{symbol[3:]}"
            
            # Get current balance
            balance = self.exchange.fetch_balance()
            
            if signal_type == 'BUY':
                # Check if we have enough positions
                positions = self._get_open_positions()
                if len(positions) >= self.max_open_trades:
                    logger.warning(f"Maximum open trades reached ({self.max_open_trades}), skipping buy signal")
                    return
                
                # Calculate amount to buy
                quote_currency = symbol[3:]
                quote_balance = balance.get(quote_currency, {}).get('free', 0)
                
                risk_amount = quote_balance * self.risk_per_trade
                
                # Get market price
                ticker = self.exchange.fetch_ticker(exchange_symbol)
                market_price = ticker['last']
                
                # Calculate amount
                amount = risk_amount / market_price
                
                # Create order
                logger.info(f"Creating buy order for {amount} {exchange_symbol} at market price")
                
                order = self.exchange.create_market_buy_order(
                    exchange_symbol,
                    amount
                )
                
                # Log order details
                logger.info(f"Buy order executed: {order}")
                
                # Save trade to database
                trade_data = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'trade_type': 'BUY',
                    'price': market_price,
                    'amount': amount,
                    'cost': amount * market_price,
                    'fee': order.get('fee', {}).get('cost', 0),
                    'realized_pnl': 0,
                    'signal_id': signal.id
                }
                
                self.db_manager.save_trade(trade_data)
            
            elif signal_type == 'SELL':
                # Find if we have a position for this symbol
                base_currency = symbol[:3]
                base_balance = balance.get(base_currency, {}).get('free', 0)
                
                if base_balance <= 0:
                    logger.warning(f"No {base_currency} balance to sell, skipping sell signal")
                    return
                
                # Get market price
                ticker = self.exchange.fetch_ticker(exchange_symbol)
                market_price = ticker['last']
                
                # Create order
                logger.info(f"Creating sell order for {base_balance} {exchange_symbol} at market price")
                
                order = self.exchange.create_market_sell_order(
                    exchange_symbol,
                    base_balance
                )
                
                # Log order details
                logger.info(f"Sell order executed: {order}")
                
                # Save trade to database
                trade_data = {
                    'symbol': symbol,
                    'timestamp': datetime.now(),
                    'trade_type': 'SELL',
                    'price': market_price,
                    'amount': base_balance,
                    'cost': base_balance * market_price,
                    'fee': order.get('fee', {}).get('cost', 0),
                    'realized_pnl': 0,  # Cannot calculate without knowing entry price
                    'signal_id': signal.id
                }
                
                self.db_manager.save_trade(trade_data)
        
        except Exception as e:
            logger.error(f"Error executing live trade: {str(e)}")
    
    def _get_open_positions(self):
        """Get currently open positions"""
        if self.demo_mode:
            return self.demo_positions
        else:
            try:
                # For spot trading, we use balance
                balance = self.exchange.fetch_balance()
                positions = []
                
                # Only count non-zero crypto balances (exclude fiat)
                for currency, data in balance.items():
                    if currency not in ['USDT', 'USD', 'BUSD', 'DAI', 'USDC']:
                        free = data.get('free', 0)
                        if free > 0:
                            positions.append({
                                'symbol': currency,
                                'amount': free,
                                'current_price': 0  # Would need to fetch current price
                            })
                
                return positions
            
            except Exception as e:
                logger.error(f"Error getting open positions: {str(e)}")
                return []
    
    def get_account_summary(self):
        """Get account summary (balance, positions, etc.)"""
        if self.demo_mode:
            # Calculate equity (balance + position values)
            equity = self.demo_balance
            
            # Add value of open positions
            for position in self.demo_positions:
                # For demo, we just use the entry price
                position_value = position['amount'] * position['entry_price']
                equity += position_value
            
            # Format positions for display
            formatted_positions = []
            for position in self.demo_positions:
                formatted_positions.append({
                    'symbol': position['symbol'],
                    'amount': position['amount'],
                    'entry_price': position['entry_price'],
                    'current_price': position['entry_price'],  # In demo mode, we don't track current price
                    'unrealized_pnl': 0  # In demo mode, we don't track unrealized PnL
                })
            
            return {
                'balance': self.demo_balance,
                'equity': equity,
                'positions': formatted_positions
            }
        else:
            try:
                if not self.exchange:
                    logger.error("Exchange not initialized")
                    return {'balance': 0, 'equity': 0, 'positions': []}
                
                # Get balance from exchange
                balance_data = self.exchange.fetch_balance()
                
                # Extract total balance (convert to USD)
                total_balance = balance_data.get('total', {}).get('USD', 0)
                if total_balance == 0:
                    # Try USDT if USD not available
                    total_balance = balance_data.get('total', {}).get('USDT', 0)
                
                # Get positions
                positions = self._get_open_positions()
                
                # Calculate equity (balance + position values)
                equity = total_balance
                
                # Format positions for display
                formatted_positions = []
                for position in positions:
                    formatted_positions.append({
                        'symbol': position['symbol'],
                        'amount': position['amount'],
                        'entry_price': 0,  # We don't know entry price from exchange API
                        'current_price': position['current_price'],
                        'unrealized_pnl': 0  # We don't know PnL without entry price
                    })
                
                return {
                    'balance': total_balance,
                    'equity': equity,
                    'positions': formatted_positions
                }
            
            except Exception as e:
                logger.error(f"Error getting account summary: {str(e)}")
                return {'balance': 0, 'equity': 0, 'positions': []}
    
    def set_risk_per_trade(self, risk):
        """Update risk per trade setting"""
        self.risk_per_trade = max(0.01, min(risk, 0.1))
        logger.info(f"Updated risk per trade to {self.risk_per_trade}")
    
    def set_max_open_trades(self, max_trades):
        """Update maximum open trades setting"""
        self.max_open_trades = max(1, max_trades)
        logger.info(f"Updated max open trades to {self.max_open_trades}")