import logging
import json
import requests
from datetime import datetime
import time
import threading
from database.db_manager import DatabaseManager

logger = logging.getLogger("SentimentCollector")

class SentimentCollector:
    """Collects market sentiment data"""
    
    def __init__(self, db_manager):
        """
        Initialize sentiment collector
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        self.running = False
        self.sources = {
            'alternative': {
                'url': 'https://api.alternative.me/fng/',
                'parser': self._parse_alternative_me
            }
        }
        
        logger.info("Sentiment collector initialized")
    
    def start(self, interval=3600):
        """
        Start collecting sentiment data
        
        Args:
            interval: Interval between collections in seconds
        """
        if self.running:
            logger.warning("Sentiment collector is already running")
            return
        
        self.running = True
        
        # Start collection thread
        thread = threading.Thread(target=self._collection_loop, args=(interval,))
        thread.daemon = True
        thread.start()
        
        logger.info("Sentiment collector started")
    
    def stop(self):
        """Stop collecting sentiment data"""
        self.running = False
        logger.info("Sentiment collector stopped")
    
    def collect_now(self):
        """Collect sentiment data immediately"""
        for source_name, source_info in self.sources.items():
            try:
                sentiment_data = self._collect_from_source(source_name, source_info)
                if sentiment_data:
                    self.db_manager.save_sentiment(sentiment_data)
            except Exception as e:
                logger.error(f"Error collecting sentiment from {source_name}: {str(e)}")
    
    def _collection_loop(self, interval):
        """Collection loop for periodically fetching sentiment data"""
        while self.running:
            try:
                self.collect_now()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in sentiment collection loop: {str(e)}")
                time.sleep(60)  # Wait a bit on error before retry
    
    def _collect_from_source(self, source_name, source_info):
        """Collect data from a specific source"""
        try:
            url = source_info['url']
            parser = source_info['parser']
            
            logger.info(f"Collecting sentiment data from {source_name}")
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            parsed_data = parser(data)
            
            if parsed_data:
                parsed_data['source'] = source_name
                parsed_data['timestamp'] = datetime.utcnow()
                parsed_data['raw_data'] = json.dumps(data)
                
                logger.info(f"Collected sentiment data from {source_name}: {parsed_data['market_sentiment']} (index: {parsed_data['fear_greed_index']})")
                return parsed_data
            else:
                logger.warning(f"Failed to parse sentiment data from {source_name}")
                return None
        
        except Exception as e:
            logger.error(f"Error collecting sentiment from {source_name}: {str(e)}")
            return None
    
    # Parser methods for different sources
    def _parse_alternative_me(self, data):
        """Parse Fear & Greed Index from alternative.me"""
        try:
            if 'data' not in data or not data['data']:
                return None
            
            latest = data['data'][0]
            value = int(latest['value'])
            
            # Determine sentiment based on value
            if value <= 20:
                sentiment = 'extreme_fear'
            elif value <= 40:
                sentiment = 'fear'
            elif value <= 60:
                sentiment = 'neutral'
            elif value <= 80:
                sentiment = 'greed'
            else:
                sentiment = 'extreme_greed'
            
            return {
                'fear_greed_index': value,
                'market_sentiment': sentiment
            }
        
        except Exception as e:
            logger.error(f"Error parsing alternative.me data: {str(e)}")
            return None