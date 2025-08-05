import logging
import os
import sys
import time
import torch
import argparse
from datetime import datetime
import threading
import json
import pandas as pd
import numpy as np

# تنظیم لاگر
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trade.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("Main")

# وارد کردن ماژول‌های پروژه
from database.db_manager import DatabaseManager
from data_collection.ohlcv_collector import OHLCVCollector
from data_collection.tick_collector import TickCollector
from data_collection.indicators import IndicatorCalculator
from data_collection.sentiment_collector import SentimentCollector
from data_collection.orderbook_collector import OrderbookCollector
from models.encoders.feature_encoder import OHLCVEncoder, IndicatorEncoder, SentimentEncoder, OrderbookEncoder, TickDataEncoder, CandlePatternEncoder
from models.gating import FeatureGatingModule
from models.neural_network import MarketTransformer, OnlineLearner
from models.signal_generator import SignalGenerator
from trading.executor import TradeExecutor
from trading.position_manager import PositionManager
from trading.risk_manager import RiskManager
import app  # Flask web interface

class TradingBot:
    """
    Main trading bot class that orchestrates all components
    """
    
    def __init__(self, config):
        """
        Initialize trading bot
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.is_running = False
        
        # MySQL connection string
        mysql_config = config.get('mysql', {})
        connection_string = f"mysql+pymysql://{mysql_config.get('user')}:{mysql_config.get('password')}@{mysql_config.get('host')}/{mysql_config.get('database')}"
        
        # Initialize components
        logger.info("Initializing components...")
        
        # Database
        self.db_manager = DatabaseManager(connection_string)
        
        # Data collectors
        self.collectors = {}
        self._init_data_collectors()
        
        # Model components
        self.encoders = {}
        self.model = None
        self.gating = None
        self.learner = None
        self.signal_generator = None
        self._init_model_components()
        
        # Trading components
        self.executor = None
        self.position_manager = None
        self.risk_manager = None
        self._init_trading_components()
        
        # Communication queue with web interface
        self.message_queue = app.bot_message_queue
        
        logger.info("Trading bot initialization complete")
    
    def _init_data_collectors(self):
        """Initialize data collection components"""
        symbols = self.config.get('trading', {}).get('symbols', ['BTCUSDT'])
        timeframes = self.config.get('trading', {}).get('timeframes', ['5m'])
        
        # OHLCV collector
        self.collectors['ohlcv'] = OHLCVCollector(
            self.db_manager,
            symbols=symbols,
            timeframes=timeframes
        )
        
        # Indicator calculator
        self.collectors['indicators'] = IndicatorCalculator(self.db_manager)
        
        # Tick collector
        self.collectors['tick'] = TickCollector(self.db_manager, symbols=symbols)
        
        # Sentiment collector
        self.collectors['sentiment'] = SentimentCollector(self.db_manager)
        
        # Orderbook collector
        self.collectors['orderbook'] = OrderbookCollector(self.db_manager, symbols=symbols)
    
    def _init_model_components(self):
        """Initialize model components"""
        # Feature encoders
        self.encoders['ohlcv'] = OHLCVEncoder(window_size=20)
        self.encoders['indicator'] = IndicatorEncoder(window_size=20)
        self.encoders['sentiment'] = SentimentEncoder()
        self.encoders['orderbook'] = OrderbookEncoder(depth=10)
        self.encoders['tick_data'] = TickDataEncoder(window_size=100)
        self.encoders['candle_pattern'] = CandlePatternEncoder(num_patterns=100)
        
        # Feature dimensions
        feature_dims = {
            'ohlcv': (20, 13),  # (seq_length, feature_dim)
            'indicator': (20, 20),
            'sentiment': (1, 1),
            'orderbook': (1, 42),
            'tick_data': (100, 3),
            'candle_pattern': (1, 100)
        }
        
        # Gating mechanism
        feature_groups = {
            'ohlcv': 13,
            'indicator': 20,
            'sentiment': 1,
            'orderbook': 42,
            'tick_data': 3,
            'candle_pattern': 100
        }
        
        self.gating = FeatureGatingModule(feature_groups)
        
        # Neural network model
        model_config = self.config.get('model', {})
        self.model = MarketTransformer(
            feature_dims=feature_dims,
            hidden_dim=model_config.get('hidden_dim', 128),
            n_layers=model_config.get('n_layers', 2),
            n_heads=model_config.get('n_heads', 4),
            dropout=model_config.get('dropout', 0.1)
        )
        
        # Online learner
        self.learner = OnlineLearner(
            model=self.model,
            gating_module=self.gating,  # Pass gating module for feedback
            lr=model_config.get('learning_rate', 1e-4),
            buffer_size=1000,
            batch_size=32,
            update_interval=model_config.get('update_interval', 3600),  # Update every hour
            save_dir='saved_models'
        )
        
        # Signal generator
        trading_config = self.config.get('trading', {})
        self.signal_generator = SignalGenerator(
            model=self.model,
            db_manager=self.db_manager,
            confidence_threshold=trading_config.get('confidence_threshold', 0.7),
            cooldown_period=trading_config.get('cooldown_period', 6),  # 6 candles cooldown
            symbol=trading_config.get('symbols', ['BTCUSDT'])[0]
        )
        
        # Try to load saved model if available
        self._load_latest_model()
    
    def _init_trading_components(self):
        """Initialize trading components"""
        trading_config = self.config.get('trading', {})
        
        # Trading executor
        self.executor = TradeExecutor(
            db_manager=self.db_manager,
            exchange_name=trading_config.get('exchange', 'coinex'),
            demo_mode=trading_config.get('demo_mode', True),
            api_key=trading_config.get('api_key'),
            api_secret=trading_config.get('api_secret'),
            initial_balance=trading_config.get('initial_balance', 100),
            risk_per_trade=trading_config.get('risk_per_trade', 0.02),
            max_open_trades=trading_config.get('max_open_trades', 1)
        )
        
        # Position manager
        self.position_manager = PositionManager(
            db_manager=self.db_manager,
            initial_stop_loss=trading_config.get('initial_stop_loss', 0.03),
            take_profit_levels=trading_config.get('take_profit_levels', [0.05, 0.1, 0.15])
        )
        
        # Risk manager
        self.risk_manager = RiskManager(
            db_manager=self.db_manager,
            max_risk_per_trade=trading_config.get('risk_per_trade', 0.02),
            max_open_risk=trading_config.get('max_open_risk', 0.06),
            max_daily_loss=trading_config.get('max_daily_loss', 0.05),
            initial_capital=trading_config.get('initial_balance', 100)
        )
    
    def _load_latest_model(self):
        """Try to load the latest saved model"""
        save_dir = 'saved_models'
        if not os.path.exists(save_dir):
            logger.info("No saved models found")
            return
        
        # Find the latest model checkpoint
        model_files = [f for f in os.listdir(save_dir) if f.startswith('model_checkpoint_')]
        if not model_files:
            logger.info("No model checkpoints found")
            return
        
        # Sort by update counter
        model_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]), reverse=True)
        latest_model = os.path.join(save_dir, model_files[0])
        
        # Load the model
        if self.learner.load_model(latest_model):
            logger.info(f"Loaded latest model: {latest_model}")
        else:
            logger.warning(f"Failed to load model: {latest_model}")
    
    def start(self):
        """Start the trading bot"""
        if self.is_running:
            logger.warning("Trading bot is already running")
            return
        
        logger.info("Starting trading bot...")
        self.is_running = True
        
        # Start data collectors
        for name, collector in self.collectors.items():
            if hasattr(collector, 'start'):
                collector.start()
                logger.info(f"Started {name} collector")
        
        # Start online learner
        self.learner.start()
        logger.info("Started online learner")
        
        # Start trade executor
        self.executor.start()
        logger.info("Started trade executor")
        
        # Start main loop in a separate thread
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.daemon = True
        self.main_thread.start()
        logger.info("Started main loop")
        
        # Start web interface in a separate thread
        self.web_thread = threading.Thread(target=self._start_web_interface)
        self.web_thread.daemon = True
        self.web_thread.start()
        logger.info("Started web interface")
    
    def stop(self):
        """Stop the trading bot"""
        if not self.is_running:
            logger.warning("Trading bot is not running")
            return
        
        logger.info("Stopping trading bot...")
        self.is_running = False
        
        # Stop data collectors
        for name, collector in self.collectors.items():
            if hasattr(collector, 'stop'):
                collector.stop()
                logger.info(f"Stopped {name} collector")
        
        # Stop online learner
        self.learner.stop()
        logger.info("Stopped online learner")
        
        # Stop trade executor
        self.executor.stop()
        logger.info("Stopped trade executor")
        
        logger.info("Trading bot stopped")
    
    def _main_loop(self):
        """Main bot loop"""
        while self.is_running:
            try:
                # Process any messages from web interface
                self._process_messages()
                
                # Collect fresh data
                data = self._collect_data()
                
                # Process data
                features = self._process_data(data)
                
                # Apply gating mechanism
                gated_features = self.gating(features)
                
                # Generate trading signal
                active_features = self.gating.get_active_features()
                current_price = data.get('ohlcv', pd.DataFrame()).iloc[-1].close if not data.get('ohlcv', pd.DataFrame()).empty else 0
                
                signal = self.signal_generator.generate_signal(
                    gated_features, 
                    active_features,
                    current_price
                )
                
                # Process signal with position and risk management
                if signal:
                    self._process_signal(signal, current_price, data, gated_features, active_features)
                
                # Check for position management (stop loss, take profit adjustments)
                closed_positions = self._manage_positions(current_price)
                
                # Update learning system with trade outcomes
                if closed_positions:
                    self._update_learning_from_outcomes(closed_positions)
                
                # Save active features to database
                self.db_manager.update_active_features(active_features)
                
                # Update web interface with data
                self._update_web_interface(data, features, active_features, signal)
                
                # Sleep to prevent high CPU usage
                time.sleep(5)  # Check every 5 seconds
            
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                time.sleep(10)  # Wait longer on error
    
    def _process_signal(self, signal, current_price, data, gated_features=None, active_features=None):
        """
        Process a trading signal with position and risk management
        
        Args:
            signal: Signal dictionary
            current_price: Current market price
            data: Data dictionary
            gated_features: Gated features used for this signal
            active_features: Active features information
        """
        try:
            symbol = signal['symbol']
            signal_type = signal['signal_type']
            confidence = signal.get('confidence', 0.5)
            
            # Store signal for learning (regardless of execution)
            if gated_features and active_features:
                # Convert signal type to label
                label = {'BUY': 0, 'SELL': 1, 'HOLD': 2}.get(signal_type, 2)
                
                # Calculate feature contributions
                feature_contributions = {}
                for group_name, group_data in active_features.items():
                    if isinstance(group_data, dict) and 'weight' in group_data:
                        feature_contributions[group_name] = group_data['weight']
                    elif isinstance(group_data, dict):
                        for feature_name, feature_info in group_data.items():
                            if isinstance(feature_info, dict) and 'weight' in feature_info:
                                feature_contributions[f"{group_name}.{feature_name}"] = feature_info['weight']
                
                # Add to learning buffer
                self.learner.add_experience(
                    features=gated_features,
                    label=label,
                    confidence=confidence,
                    feature_contributions=feature_contributions
                )
            
            if signal_type == 'BUY':
                # Check if we already have a position
                if self.position_manager.has_position(symbol):
                    logger.info(f"Already have position for {symbol}, ignoring buy signal")
                    return
                
                # Calculate position parameters
                entry_price = current_price
                stop_loss = entry_price * (1 - self.config.get('trading', {}).get('initial_stop_loss', 0.03))
                
                # Calculate position size based on risk
                position_size = self.risk_manager.calculate_position_size(
                    symbol, 
                    entry_price, 
                    stop_loss, 
                    current_capital=self.executor.demo_balance if hasattr(self.executor, 'demo_balance') else None
                )
                
                # Check if position is allowed by risk manager
                can_open, reason = self.risk_manager.can_open_position(
                    symbol, 
                    entry_price, 
                    stop_loss, 
                    position_size, 
                    self.position_manager.get_all_positions()
                )
                
                if not can_open:
                    logger.warning(f"Risk manager rejected position: {reason}")
                    return
                
                # Add position to position manager
                self.position_manager.add_position(
                    symbol,
                    entry_price,
                    position_size,
                    signal.get('id')
                )
                
                logger.info(f"Processed BUY signal for {symbol} at {entry_price} (confidence: {confidence:.2f})")
                
            elif signal_type == 'SELL':
                # Check if we have a position to sell
                position = self.position_manager.get_position(symbol)
                
                if not position:
                    logger.info(f"No position for {symbol}, ignoring sell signal")
                    return
                
                # Remove position from position manager
                closed_position = self.position_manager.remove_position(symbol)
                
                # Calculate P&L for learning
                if closed_position:
                    entry_price = closed_position['entry_price']
                    pnl_percent = (current_price - entry_price) / entry_price
                    
                    # This will be used by learning system
                    closed_position['exit_price'] = current_price
                    closed_position['pnl_percent'] = pnl_percent
                    closed_position['exit_reason'] = 'signal'
                
                logger.info(f"Processed SELL signal for {symbol} at {current_price} (confidence: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
    
    def _manage_positions(self, current_price):
        """
        Manage existing positions (check for stop loss/take profit)
        
        Args:
            current_price: Current market price
            
        Returns:
            List of closed positions for learning feedback
        """
        closed_positions = []
        
        try:
            # Get all positions
            positions = self.position_manager.get_all_positions()
            
            for position in positions:
                symbol = position['symbol']
                
                # Update position with current price
                action, updated_position = self.position_manager.update_position_price(symbol, current_price)
                
                if action == 'stop_loss':
                    # Close position due to stop loss
                    closed_position = self.position_manager.remove_position(symbol)
                    
                    if closed_position:
                        entry_price = closed_position['entry_price']
                        pnl_percent = (current_price - entry_price) / entry_price
                        
                        closed_position['exit_price'] = current_price
                        closed_position['pnl_percent'] = pnl_percent
                        closed_position['exit_reason'] = 'stop_loss'
                        closed_positions.append(closed_position)
                    
                    # Log the event
                    logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                    
                    # In a real system, you would execute the sell order here
                    
                elif action == 'tp_adjust':
                    # Stop loss was adjusted due to take profit level
                    logger.info(f"Take profit level reached for {symbol}, stop loss moved to {updated_position['stop_loss']}")
        
        except Exception as e:
            logger.error(f"Error managing positions: {str(e)}")
        
        return closed_positions
    
    def _update_learning_from_outcomes(self, closed_positions):
        """
        Update learning system with trade outcomes
        
        Args:
            closed_positions: List of closed position dictionaries
        """
        try:
            for position in closed_positions:
                pnl_percent = position.get('pnl_percent', 0)
                exit_reason = position.get('exit_reason', 'unknown')
                
                # Define success criteria
                was_successful = pnl_percent > 0.01  # More than 1% profit
                
                # Find the corresponding experience in learning buffer
                # This is a simplified approach - in a real system you'd want better tracking
                signal_id = position.get('signal_id')
                if signal_id and len(self.learner.experience_buffer['labels']) > 0:
                    # Update the most recent BUY experience (simplified)
                    # In a more sophisticated system, you'd match by signal_id
                    for i in range(len(self.learner.experience_buffer['labels']) - 1, -1, -1):
                        if (self.learner.experience_buffer['labels'][i] == 0 and  # BUY signal
                            self.learner.experience_buffer['outcomes'][i] is None):  # No outcome yet
                            
                            self.learner.update_trade_outcome(i, pnl_percent, was_successful)
                            break
                
                logger.info(f"Learning update: Position closed with {pnl_percent*100:.2f}% P&L, "
                           f"Success: {'Yes' if was_successful else 'No'}, Reason: {exit_reason}")
        
        except Exception as e:
            logger.error(f"Error updating learning from outcomes: {str(e)}")
    
    def _start_web_interface(self):
        """Start Flask web interface"""
        app.start_web_server(
            host=self.config.get('web', {}).get('host', '0.0.0.0'),
            port=self.config.get('web', {}).get('port', 5000)
        )
    
    def _process_messages(self):
        """Process messages from web interface"""
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get_nowait()
                
                if message.get('type') == 'command':
                    command = message.get('command')
                    
                    if command == 'start':
                        if not self.is_running:
                            self.start()
                    elif command == 'stop':
                        if self.is_running:
                            self.stop()
                    elif command == 'restart':
                        self.stop()
                        time.sleep(1)
                        self.start()
                
                elif message.get('type') == 'settings_update':
                    settings = message.get('settings', {})
                    
                    # Update executor settings
                    if 'risk_per_trade' in settings:
                        risk = float(settings['risk_per_trade'])
                        self.executor.set_risk_per_trade(risk)
                        self.risk_manager.max_risk_per_trade = risk
                    
                    if 'max_open_trades' in settings:
                        self.executor.set_max_open_trades(int(settings['max_open_trades']))
                    
                    # Update signal generator settings
                    if 'confidence_threshold' in settings:
                        self.signal_generator.set_confidence_threshold(float(settings['confidence_threshold']))
            
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
    
    def _collect_data(self):
        """Collect data from database"""
        symbol = self.config.get('trading', {}).get('symbols', ['BTCUSDT'])[0]
        timeframe = self.config.get('trading', {}).get('timeframes', ['5m'])[0]
        
        # Collect OHLCV data
        ohlcv_data = self.db_manager.get_ohlcv(symbol, timeframe, limit=100)
        
        # Calculate indicators
        indicators = {}
        if not ohlcv_data.empty:
            indicators_result = self.collectors['indicators'].calculate_indicators(symbol, timeframe)
            indicators = {name: values[-20:] if len(values) > 20 else values for name, values in indicators_result.items()}
        
        # Get latest sentiment
        sentiment = self.db_manager.get_latest_sentiment()
        
        # Get active features
        active_features = self.db_manager.get_active_features()
        
        return {
            'ohlcv': ohlcv_data,
            'indicators': indicators,
            'sentiment': sentiment,
            'active_features': active_features
        }
    
    def _process_data(self, data):
        """
        Process and encode data for the model
        
        Args:
            data: Dictionary with collected data
            
        Returns:
            Dictionary with encoded features
        """
        features = {}
        
        # Encode OHLCV data
        if 'ohlcv' in data and not data['ohlcv'].empty:
            features['ohlcv'] = self.encoders['ohlcv'].transform(data['ohlcv'])
        
        # Encode indicators
        if 'indicators' in data and data['indicators']:
            features['indicator'] = self.encoders['indicator'].transform(data['indicators'])
        
        # Encode sentiment
        if 'sentiment' in data and data['sentiment']:
            features['sentiment'] = self.encoders['sentiment'].transform(data['sentiment'])
        
        # We'll have orderbook and tick data when those collectors are running
        # For now, we'll proceed with what we have
        
        return features
    
    def _update_web_interface(self, data, features, active_features, signal):
        """Update web interface with current data"""
        # Get account summary
        account = self.executor.get_account_summary()
        
        # Create signals list
        session = self.db_manager.get_session()
        from database.models import Signal
        signals = session.query(Signal).order_by(Signal.timestamp.desc()).limit(10).all()
        signals_data = []
        
        for s in signals:
            signals_data.append({
                'timestamp': s.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'signal': s.signal_type,
                'price': s.price,
                'confidence': s.confidence
            })
        
        self.db_manager.close_session(session)
        
        # Calculate performance metrics
        performance = self._calculate_performance()
        
        # Prepare model stats
        model_stats = {
            'feature_importance': {},
            'update_counts': {'model': self.learner.updates_counter},
            'market_regime': self._detect_market_regime(data.get('ohlcv', pd.DataFrame()))
        }
        
        # Extract feature importance from active features
        if active_features:
            for group, features in active_features.items():
                if isinstance(features, dict) and 'weight' in features:
                    model_stats['feature_importance'][group] = features['weight']
        
        # Create system stats
        import psutil
        system_stats = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'uptime': 0,  # We don't track this yet
            'errors': []  # We don't track this yet
        }
        
        # Update app.data_store
        app.data_store['bot_status'] = 'running' if self.is_running else 'stopped'
        app.data_store['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        app.data_store['trading_data']['portfolio'] = account
        app.data_store['trading_data']['signals'] = signals_data
        app.data_store['trading_data']['performance'] = performance
        
        if 'ohlcv' in data:
            app.data_store['trading_data']['ohlcv'] = data['ohlcv']
        
        if 'indicators' in data and data['indicators']:
            app.data_store['trading_data']['indicators'] = data['indicators']
        
        app.data_store['system_stats'] = system_stats
        app.data_store['model_stats'] = model_stats
        app.data_store['active_features'] = active_features
        
        # Add learning statistics
        learning_summary = self.learner.get_learning_summary()
        app.data_store['learning_stats'] = learning_summary
    
    def _calculate_performance(self):
        """Calculate performance metrics"""
        session = self.db_manager.get_session()
        from database.models import Trade
        import pandas as pd
        
        try:
            # Get all trades
            trades = session.query(Trade).order_by(Trade.timestamp.asc()).all()
            
            if not trades:
                return {'daily': 0, 'weekly': 0, 'monthly': 0, 'all_time': 0}
            
            # Calculate PnL
            df = pd.DataFrame([{
                'timestamp': t.timestamp,
                'pnl': t.realized_pnl if t.realized_pnl else 0
            } for t in trades])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate performance metrics
            now = datetime.now()
            daily_cutoff = now - pd.Timedelta(days=1)
            weekly_cutoff = now - pd.Timedelta(days=7)
            monthly_cutoff = now - pd.Timedelta(days=30)
            
            daily_pnl = df[df.index >= daily_cutoff]['pnl'].sum()
            weekly_pnl = df[df.index >= weekly_cutoff]['pnl'].sum()
            monthly_pnl = df[df.index >= monthly_cutoff]['pnl'].sum()
            all_time_pnl = df['pnl'].sum()
            
            # Convert to percentage
            initial_balance = 100  # Assume we started with 100 USD
            daily_pct = (daily_pnl / initial_balance) * 100
            weekly_pct = (weekly_pnl / initial_balance) * 100
            monthly_pct = (monthly_pnl / initial_balance) * 100
            all_time_pct = (all_time_pnl / initial_balance) * 100
            
            return {
                'daily': daily_pct,
                'weekly': weekly_pct,
                'monthly': monthly_pct,
                'all_time': all_time_pct
            }
        
        except Exception as e:
            logger.error(f"Error calculating performance: {str(e)}")
            return {'daily': 0, 'weekly': 0, 'monthly': 0, 'all_time': 0}
        
        finally:
            self.db_manager.close_session(session)
    
    def _detect_market_regime(self, ohlcv):
        """Detect current market regime"""
        if ohlcv.empty:
            return 'neutral'
        
        try:
            # Simple detection based on recent price movement
            closes = ohlcv['close'].values
            if len(closes) < 20:
                return 'neutral'
            
            # Calculate returns
            returns = np.diff(closes) / closes[:-1]
            
            # Calculate volatility
            volatility = np.std(returns[-20:]) * 100
            
            # Calculate trend
            trend = (closes[-1] / closes[-20] - 1) * 100
            
            # Determine regime
            if volatility > 3:  # High volatility
                if trend > 5:
                    return 'bullish_volatile'
                elif trend < -5:
                    return 'bearish_volatile'
                else:
                    return 'choppy'
            else:  # Low volatility
                if trend > 3:
                    return 'bullish'
                elif trend < -3:
                    return 'bearish'
                else:
                    return 'neutral'
        
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return 'neutral'


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        config = {
            'mysql': {
                'host': 'localhost',
                'user': 'root',
                'password': '',
                'database': 'trade'
            },
            'trading': {
                'symbols': ['BTCUSDT'],
                'timeframes': ['5m'],
                'exchange': 'coinex',
                'demo_mode': True,
                'initial_balance': 100,
                'risk_per_trade': 0.02,
                'max_open_trades': 1,
                'initial_stop_loss': 0.03,
                'take_profit_levels': [0.05, 0.1, 0.15],
                'max_open_risk': 0.06,
                'max_daily_loss': 0.05
            },
            'web': {
                'host': '0.0.0.0',
                'port': 5000
            },
            'model': {
                'hidden_dim': 128,
                'n_layers': 2,
                'n_heads': 4,
                'dropout': 0.1,
                'learning_rate': 1e-4,
                'update_interval': 3600
            }
        }
    
    # Create and start bot
    bot = TradingBot(config)
    
    try:
        bot.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Stopping bot due to keyboard interrupt")
        bot.stop()
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        bot.stop()