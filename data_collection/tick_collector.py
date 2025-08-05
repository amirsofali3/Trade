import logging
import json
import pandas as pd
from datetime import datetime
import websocket
import threading
import time
from database.db_manager import DatabaseManager

logger = logging.getLogger("TickCollector")

class TickCollector:
    """Collects tick data from exchange websockets"""
    
    def __init__(self, db_manager, symbols=None):
        """
        Initialize tick collector
        
        Args:
            db_manager: Database manager instance
            symbols: List of symbols to collect data for
        """
        self.db_manager = db_manager
        
        # Default to BTC/USDT if no symbols provided
        self.symbols = symbols or ['BTCUSDT']
        
        # Buffer for storing ticks before batch insert
        self.tick_buffer = {symbol: [] for symbol in self.symbols}
        self.buffer_size = 100  # Number of ticks to store before saving to DB
        self.running = False
        self.ws = None
        
        logger.info(f"Initialized Tick collector for {', '.join(self.symbols)}")
    
    def start(self):
        """Start the tick collector websocket"""
        if self.running:
            logger.warning("Tick collector is already running")
            return
        
        self.running = True
        
        # Start websocket in a separate thread
        thread = threading.Thread(target=self._start_websocket)
        thread.daemon = True
        thread.start()
        
        # Start buffer saver thread
        buffer_thread = threading.Thread(target=self._save_buffer_periodically)
        buffer_thread.daemon = True
        buffer_thread.start()
        
        logger.info("Tick collector started")
    
    def stop(self):
        """Stop the tick collector websocket"""
        self.running = False
        if self.ws:
            self.ws.close()
            self.ws = None
        
        # Save any remaining ticks in buffer
        for symbol in self.symbols:
            self._save_ticks(symbol)
        
        logger.info("Tick collector stopped")
    
    def _start_websocket(self):
        """Start the websocket connection to Binance"""
        try:
            # Format streams for all symbols
            streams = [f"{symbol.lower()}@trade" for symbol in self.symbols]
            stream_path = "/".join(streams)
            socket_url = f"wss://stream.binance.com:9443/ws/{stream_path}"
            
            logger.info(f"Connecting to websocket: {socket_url}")
            
            self.ws = websocket.WebSocketApp(
                socket_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            self.ws.run_forever()
        except Exception as e:
            logger.error(f"Error in tick websocket: {str(e)}")
            self.running = False
    
    def _on_message(self, ws, message):
        """Handle incoming websocket message"""
        try:
            data = json.loads(message)
            
            # Extract symbol and trade data
            symbol = data.get('s')
            
            if symbol and symbol in self.symbols:
                timestamp = datetime.fromtimestamp(data.get('T') / 1000.0)
                price = float(data.get('p'))
                volume = float(data.get('q'))
                is_buyer = data.get('m') == False  # m = is_buyer_maker, we want is_buyer
                
                # Add to buffer
                self.tick_buffer[symbol].append({
                    'timestamp': timestamp,
                    'price': price,
                    'volume': volume,
                    'is_buyer': is_buyer
                })
                
                # Save buffer if it reaches threshold
                if len(self.tick_buffer[symbol]) >= self.buffer_size:
                    self._save_ticks(symbol)
        
        except Exception as e:
            logger.error(f"Error processing tick message: {str(e)}")
    
    def _on_error(self, ws, error):
        """Handle websocket error"""
        logger.error(f"Websocket error: {str(error)}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle websocket close"""
        logger.info(f"Websocket closed: {close_msg} (code: {close_status_code})")
        
        # Try to reconnect if still running
        if self.running:
            logger.info("Attempting to reconnect in 5 seconds...")
            time.sleep(5)
            thread = threading.Thread(target=self._start_websocket)
            thread.daemon = True
            thread.start()
    
    def _on_open(self, ws):
        """Handle websocket open"""
        logger.info("Websocket connection established")
    
    def _save_ticks(self, symbol):
        """Save buffered ticks to database"""
        if not self.tick_buffer[symbol]:
            return
        
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame(self.tick_buffer[symbol])
            
            # Save to database
            self.db_manager.save_tick(symbol, df)
            
            # Clear buffer
            self.tick_buffer[symbol] = []
            
            logger.debug(f"Saved {len(df)} ticks for {symbol}")
        except Exception as e:
            logger.error(f"Error saving ticks to database: {str(e)}")
    
    def _save_buffer_periodically(self, interval=30):
        """Save buffer periodically even if not full"""
        while self.running:
            time.sleep(interval)
            
            for symbol in self.symbols:
                if self.tick_buffer[symbol]:
                    self._save_ticks(symbol)