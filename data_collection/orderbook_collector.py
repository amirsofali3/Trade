import logging
import json
import threading
import time
import websocket
from datetime import datetime
from database.db_manager import DatabaseManager

logger = logging.getLogger("OrderbookCollector")

class OrderbookCollector:
    """Collects orderbook data from exchange"""
    
    def __init__(self, db_manager, symbols=None, depth=20):
        """
        Initialize orderbook collector
        
        Args:
            db_manager: Database manager instance
            symbols: List of symbols to collect data for
            depth: Depth of orderbook to collect
        """
        self.db_manager = db_manager
        
        # Default to BTC/USDT if no symbols provided
        self.symbols = symbols or ['BTCUSDT']
        
        self.depth = depth
        self.running = False
        self.ws = {}  # Dictionary of websockets by symbol
        
        logger.info(f"Initialized Orderbook collector for {', '.join(self.symbols)} with depth {depth}")
    
    def start(self):
        """Start the orderbook collector"""
        if self.running:
            logger.warning("Orderbook collector is already running")
            return
        
        self.running = True
        
        # Start websocket for each symbol
        for symbol in self.symbols:
            thread = threading.Thread(target=self._start_websocket, args=(symbol,))
            thread.daemon = True
            thread.start()
        
        logger.info("Orderbook collector started")
    
    def stop(self):
        """Stop the orderbook collector"""
        self.running = False
        
        for symbol, ws in self.ws.items():
            if ws:
                ws.close()
                self.ws[symbol] = None
        
        logger.info("Orderbook collector stopped")
    
    def _start_websocket(self, symbol):
        """Start websocket for a single symbol"""
        try:
            socket_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@depth{self.depth}@100ms"
            
            logger.info(f"Connecting to orderbook websocket for {symbol}: {socket_url}")
            
            self.ws[symbol] = websocket.WebSocketApp(
                socket_url,
                on_message=lambda ws, msg: self._on_message(symbol, msg),
                on_error=lambda ws, err: self._on_error(symbol, err),
                on_close=lambda ws, close_status_code, close_msg: self._on_close(symbol, close_status_code, close_msg),
                on_open=lambda ws: self._on_open(symbol)
            )
            
            self.ws[symbol].run_forever()
        
        except Exception as e:
            logger.error(f"Error in orderbook websocket for {symbol}: {str(e)}")
            
            # Try to reconnect if still running
            if self.running:
                logger.info(f"Attempting to reconnect {symbol} in 5 seconds...")
                time.sleep(5)
                thread = threading.Thread(target=self._start_websocket, args=(symbol,))
                thread.daemon = True
                thread.start()
    
    def _on_message(self, symbol, message):
        """Handle incoming orderbook message"""
        try:
            data = json.loads(message)
            
            # Create orderbook data structure
            orderbook = {
                'timestamp': datetime.utcnow(),
                'symbol': symbol,
                'bids': data.get('bids', []),  # Format: [price, quantity]
                'asks': data.get('asks', [])   # Format: [price, quantity]
            }
            
            # Convert strings to floats
            orderbook['bids'] = [[float(price), float(qty)] for price, qty in orderbook['bids']]
            orderbook['asks'] = [[float(price), float(qty)] for price, qty in orderbook['asks']]
            
            # Save to database
            self.db_manager.save_orderbook(symbol, orderbook)
            
            logger.debug(f"Processed orderbook for {symbol} with {len(orderbook['bids'])} bids and {len(orderbook['asks'])} asks")
        
        except Exception as e:
            logger.error(f"Error processing orderbook message for {symbol}: {str(e)}")
    
    def _on_error(self, symbol, error):
        """Handle websocket error"""
        logger.error(f"Orderbook websocket error for {symbol}: {str(error)}")
    
    def _on_close(self, symbol, close_status_code, close_msg):
        """Handle websocket close"""
        logger.info(f"Orderbook websocket closed for {symbol}: {close_msg} (code: {close_status_code})")
        
        # Try to reconnect if still running
        if self.running:
            logger.info(f"Attempting to reconnect {symbol} in 5 seconds...")
            time.sleep(5)
            thread = threading.Thread(target=self._start_websocket, args=(symbol,))
            thread.daemon = True
            thread.start()
    
    def _on_open(self, symbol):
        """Handle websocket open"""
        logger.info(f"Orderbook websocket connection established for {symbol}")