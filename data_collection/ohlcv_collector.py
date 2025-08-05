import logging
import pandas as pd
import time
from datetime import datetime, timedelta
import ccxt
from database.db_manager import DatabaseManager

logger = logging.getLogger("OHLCVCollector")

class OHLCVCollector:
    """Collects OHLCV data from exchange"""
    
    def __init__(self, db_manager, exchange='binance', symbols=None, timeframes=None):
        """
        Initialize OHLCV collector
        
        Args:
            db_manager: Database manager instance
            exchange: Exchange to collect data from
            symbols: List of symbols to collect data for
            timeframes: List of timeframes to collect data for
        """
        self.db_manager = db_manager
        
        # Default to BTC/USDT if no symbols provided
        self.symbols = symbols or ['BTC/USDT']
        
        # Default to 5m if no timeframes provided
        self.timeframes = timeframes or ['5m']
        
        # Initialize exchange
        try:
            if exchange.lower() == 'binance':
                self.exchange = ccxt.binance({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot'
                    }
                })
            else:
                raise ValueError(f"Unsupported exchange: {exchange}")
            
            logger.info(f"Initialized OHLCV collector for {exchange}")
        except Exception as e:
            logger.error(f"Error initializing exchange: {str(e)}")
            raise
    
    def collect_historical_data(self, symbol, timeframe, since=None, limit=1000):
        """
        Collect historical OHLCV data
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g. '1m', '5m', '1h')
            since: Start time in milliseconds
            limit: Maximum number of records to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Format symbol for ccxt
            formatted_symbol = symbol.replace('/', '')
            
            if not since:
                # Default to 1000 candles back from now
                since = int((datetime.utcnow() - timedelta(minutes=limit * self._timeframe_to_minutes(timeframe))).timestamp() * 1000)
            
            logger.info(f"Fetching historical data for {symbol} {timeframe} since {datetime.fromtimestamp(since/1000)}")
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Save to database
            self.db_manager.save_ohlcv(formatted_symbol, timeframe, df)
            
            logger.info(f"Collected {len(df)} historical records for {symbol} {timeframe}")
            return df
        
        except Exception as e:
            logger.error(f"Error collecting historical data for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    def collect_recent_data(self, symbol, timeframe, limit=100):
        """
        Collect most recent OHLCV data
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g. '1m', '5m', '1h')
            limit: Maximum number of records to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Format symbol for ccxt
            formatted_symbol = symbol.replace('/', '')
            
            logger.info(f"Fetching recent data for {symbol} {timeframe}")
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Save to database
            self.db_manager.save_ohlcv(formatted_symbol, timeframe, df)
            
            logger.info(f"Collected {len(df)} recent records for {symbol} {timeframe}")
            return df
        
        except Exception as e:
            logger.error(f"Error collecting recent data for {symbol} {timeframe}: {str(e)}")
            return pd.DataFrame()
    
    def start_collection_loop(self, interval=60, historical_first=True):
        """
        Start continuous data collection
        
        Args:
            interval: Interval between collections in seconds
            historical_first: Whether to collect historical data first
        """
        logger.info("Starting OHLCV collection loop")
        
        try:
            # Collect historical data first if requested
            if historical_first:
                for symbol in self.symbols:
                    for timeframe in self.timeframes:
                        self.collect_historical_data(symbol, timeframe)
            
            # Start continuous collection
            while True:
                for symbol in self.symbols:
                    for timeframe in self.timeframes:
                        self.collect_recent_data(symbol, timeframe)
                
                logger.debug(f"Sleeping for {interval} seconds")
                time.sleep(interval)
        
        except KeyboardInterrupt:
            logger.info("OHLCV collection loop stopped by user")
        except Exception as e:
            logger.error(f"Error in OHLCV collection loop: {str(e)}")
    
    def _timeframe_to_minutes(self, timeframe):
        """Convert timeframe string to minutes"""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 1440
        else:
            return 60  # default to 60 minutes
        