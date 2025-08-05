import logging
import pandas as pd
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, scoped_session
from .models import Base, OHLCV, Tick, Indicator, Sentiment, OrderBook, Trade, Signal, ActiveFeature
from datetime import datetime, timedelta

logger = logging.getLogger("DBManager")

class DatabaseManager:
    """Handles database operations for the trading bot"""
    
    def __init__(self, connection_string):
        """
        Initialize the database manager
        
        Args:
            connection_string: SQLAlchemy connection string
        """
        self.engine = create_engine(connection_string)
        
        # Create tables if they don't exist
        try:
            Base.metadata.create_all(self.engine)
            logger.info("Database tables created or already exist")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise
        
        # Create session factory
        self.session_factory = sessionmaker(bind=self.engine)
        self.Session = scoped_session(self.session_factory)
        
        logger.info("DatabaseManager initialized")
    
    def get_session(self):
        """Get a new session"""
        return self.Session()
    
    def close_session(self, session):
        """Close a session"""
        session.close()
    
    def save_ohlcv(self, symbol, timeframe, data):
        """
        Save OHLCV data to database
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g. '1m', '5m', '1h')
            data: DataFrame with OHLCV data
        """
        try:
            session = self.get_session()
            
            for timestamp, row in data.iterrows():
                # Check if data already exists
                existing = session.query(OHLCV).filter(
                    OHLCV.symbol == symbol,
                    OHLCV.timestamp == timestamp,
                    OHLCV.timeframe == timeframe
                ).first()
                
                if existing:
                    # Update existing record
                    existing.open = row['open']
                    existing.high = row['high']
                    existing.low = row['low']
                    existing.close = row['close']
                    existing.volume = row['volume']
                else:
                    # Create new record
                    new_ohlcv = OHLCV(
                        symbol=symbol,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        open=row['open'],
                        high=row['high'],
                        low=row['low'],
                        close=row['close'],
                        volume=row['volume']
                    )
                    session.add(new_ohlcv)
            
            session.commit()
            logger.debug(f"Saved {len(data)} OHLCV records for {symbol} {timeframe}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving OHLCV data: {str(e)}")
        finally:
            self.close_session(session)
    
    def save_tick(self, symbol, tick_data):
        """
        Save tick data to database
        
        Args:
            symbol: Trading pair symbol
            tick_data: DataFrame with tick data
        """
        try:
            session = self.get_session()
            
            for _, row in tick_data.iterrows():
                new_tick = Tick(
                    symbol=symbol,
                    timestamp=row['timestamp'],
                    price=row['price'],
                    volume=row['volume'],
                    is_buyer=row['is_buyer']
                )
                session.add(new_tick)
            
            session.commit()
            logger.debug(f"Saved {len(tick_data)} tick records for {symbol}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving tick data: {str(e)}")
        finally:
            self.close_session(session)
    
    def save_indicator(self, symbol, timeframe, indicator_name, data):
        """
        Save indicator data to database
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g. '1m', '5m', '1h')
            indicator_name: Name of the indicator
            data: DataFrame with indicator values
        """
        try:
            session = self.get_session()
            
            for timestamp, value in data.items():
                if pd.isna(value):
                    continue
                
                # Check if indicator already exists
                existing = session.query(Indicator).filter(
                    Indicator.symbol == symbol,
                    Indicator.timestamp == timestamp,
                    Indicator.timeframe == timeframe,
                    Indicator.indicator_name == indicator_name
                ).first()
                
                if existing:
                    # Update existing record
                    existing.value = value
                else:
                    # Create new record
                    new_indicator = Indicator(
                        symbol=symbol,
                        timestamp=timestamp,
                        timeframe=timeframe,
                        indicator_name=indicator_name,
                        value=value
                    )
                    session.add(new_indicator)
            
            session.commit()
            logger.debug(f"Saved {len(data)} {indicator_name} records for {symbol} {timeframe}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving indicator data: {str(e)}")
        finally:
            self.close_session(session)
    
    def save_sentiment(self, sentiment_data):
        """
        Save market sentiment data to database
        
        Args:
            sentiment_data: Dictionary with sentiment data
        """
        try:
            session = self.get_session()
            
            timestamp = sentiment_data.get('timestamp')
            
            # Check if sentiment already exists
            existing = session.query(Sentiment).filter(
                Sentiment.timestamp == timestamp,
                Sentiment.source == sentiment_data.get('source')
            ).first()
            
            if existing:
                # Update existing record
                existing.fear_greed_index = sentiment_data.get('fear_greed_index')
                existing.market_sentiment = sentiment_data.get('market_sentiment')
                existing.raw_data = sentiment_data.get('raw_data')
            else:
                # Create new record
                new_sentiment = Sentiment(
                    timestamp=timestamp,
                    fear_greed_index=sentiment_data.get('fear_greed_index'),
                    market_sentiment=sentiment_data.get('market_sentiment'),
                    source=sentiment_data.get('source'),
                    raw_data=sentiment_data.get('raw_data')
                )
                session.add(new_sentiment)
            
            session.commit()
            logger.debug(f"Saved sentiment data from {sentiment_data.get('source')}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving sentiment data: {str(e)}")
        finally:
            self.close_session(session)
    
    def save_orderbook(self, symbol, orderbook_data):
        """
        Save orderbook data to database
        
        Args:
            symbol: Trading pair symbol
            orderbook_data: Dictionary with orderbook data
        """
        try:
            session = self.get_session()
            timestamp = orderbook_data.get('timestamp')
            
            # Remove existing orderbook data for this timestamp
            session.query(OrderBook).filter(
                OrderBook.symbol == symbol,
                OrderBook.timestamp == timestamp
            ).delete()
            
            # Add bids
            for price, quantity in orderbook_data.get('bids', []):
                new_bid = OrderBook(
                    symbol=symbol,
                    timestamp=timestamp,
                    price_level=price,
                    quantity=quantity,
                    is_bid=True
                )
                session.add(new_bid)
            
            # Add asks
            for price, quantity in orderbook_data.get('asks', []):
                new_ask = OrderBook(
                    symbol=symbol,
                    timestamp=timestamp,
                    price_level=price,
                    quantity=quantity,
                    is_bid=False
                )
                session.add(new_ask)
            
            session.commit()
            logger.debug(f"Saved orderbook data for {symbol} at {timestamp}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving orderbook data: {str(e)}")
        finally:
            self.close_session(session)
    
    def save_signal(self, signal_data):
        """
        Save trading signal to database
        
        Args:
            signal_data: Dictionary with signal data
        """
        try:
            session = self.get_session()
            
            new_signal = Signal(
                symbol=signal_data.get('symbol'),
                timestamp=signal_data.get('timestamp'),
                signal_type=signal_data.get('signal_type'),
                price=signal_data.get('price'),
                confidence=signal_data.get('confidence'),
                executed=signal_data.get('executed', False),
                features_used=signal_data.get('features_used')
            )
            session.add(new_signal)
            session.commit()
            
            signal_id = new_signal.id
            logger.info(f"Saved signal {signal_data.get('signal_type')} for {signal_data.get('symbol')} with confidence {signal_data.get('confidence')}")
            
            return signal_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving signal data: {str(e)}")
            return None
        finally:
            self.close_session(session)
    
    def save_trade(self, trade_data):
        """
        Save trade execution to database
        
        Args:
            trade_data: Dictionary with trade data
        """
        try:
            session = self.get_session()
            
            new_trade = Trade(
                symbol=trade_data.get('symbol'),
                timestamp=trade_data.get('timestamp'),
                trade_type=trade_data.get('trade_type'),
                price=trade_data.get('price'),
                amount=trade_data.get('amount'),
                cost=trade_data.get('cost'),
                fee=trade_data.get('fee'),
                realized_pnl=trade_data.get('realized_pnl'),
                signal_id=trade_data.get('signal_id')
            )
            session.add(new_trade)
            
            # Update signal as executed if signal_id is provided
            if trade_data.get('signal_id'):
                signal = session.query(Signal).filter(
                    Signal.id == trade_data.get('signal_id')
                ).first()
                if signal:
                    signal.executed = True
            
            session.commit()
            logger.info(f"Saved trade {trade_data.get('trade_type')} for {trade_data.get('symbol')} at {trade_data.get('price')}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving trade data: {str(e)}")
        finally:
            self.close_session(session)
    
    def update_active_features(self, feature_updates):
        """
        Update active features and their weights
        
        Args:
            feature_updates: Dictionary with feature updates
        """
        try:
            session = self.get_session()
            timestamp = datetime.utcnow()
            
            for group, features in feature_updates.items():
                if isinstance(features, dict) and 'active' in features and 'weight' in features:
                    # This is a top-level feature
                    self._update_single_feature(session, timestamp, group, group, features['active'], features['weight'])
                elif isinstance(features, dict):
                    # This is a group with sub-features
                    for feature_name, feature_data in features.items():
                        self._update_single_feature(session, timestamp, group, feature_name, 
                                                 feature_data['active'], feature_data['weight'])
            
            session.commit()
            logger.debug(f"Updated active features")
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating active features: {str(e)}")
        finally:
            self.close_session(session)
    
    def _update_single_feature(self, session, timestamp, group, name, is_active, weight):
        """Helper method to update a single feature"""
        # Check if feature already exists
        existing = session.query(ActiveFeature).filter(
            ActiveFeature.feature_group == group,
            ActiveFeature.feature_name == name
        ).order_by(ActiveFeature.timestamp.desc()).first()
        
        # Only add a new record if status changed
        if not existing or existing.is_active != is_active or abs(existing.weight - weight) > 0.01:
            new_feature = ActiveFeature(
                timestamp=timestamp,
                feature_group=group,
                feature_name=name,
                is_active=is_active,
                weight=weight
            )
            session.add(new_feature)
    
    def get_ohlcv(self, symbol, timeframe, start_time=None, end_time=None, limit=None):
        """
        Get OHLCV data from database
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g. '1m', '5m', '1h')
            start_time: Start time for data query
            end_time: End time for data query
            limit: Maximum number of records to return
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            session = self.get_session()
            query = session.query(OHLCV).filter(
                OHLCV.symbol == symbol,
                OHLCV.timeframe == timeframe
            )
            
            if start_time:
                query = query.filter(OHLCV.timestamp >= start_time)
            
            if end_time:
                query = query.filter(OHLCV.timestamp <= end_time)
            
            query = query.order_by(OHLCV.timestamp)
            
            if limit:
                query = query.limit(limit)
            
            results = query.all()
            
            # Convert to DataFrame
            data = pd.DataFrame([{
                'timestamp': r.timestamp,
                'open': r.open,
                'high': r.high,
                'low': r.low,
                'close': r.close,
                'volume': r.volume
            } for r in results])
            
            if not data.empty:
                data.set_index('timestamp', inplace=True)
            
            return data
        except Exception as e:
            logger.error(f"Error retrieving OHLCV data: {str(e)}")
            return pd.DataFrame()
        finally:
            self.close_session(session)
    
    def get_active_features(self):
        """
        Get current active features and their weights
        
        Returns:
            Dictionary with active features
        """
        try:
            session = self.get_session()
            
            # Get the latest record for each feature
            subquery = session.query(
                ActiveFeature.feature_group,
                ActiveFeature.feature_name,
                func.max(ActiveFeature.timestamp).label('max_timestamp')
            ).group_by(ActiveFeature.feature_group, ActiveFeature.feature_name).subquery('latest')
            
            results = session.query(ActiveFeature).join(
                subquery,
                (ActiveFeature.feature_group == subquery.c.feature_group) &
                (ActiveFeature.feature_name == subquery.c.feature_name) &
                (ActiveFeature.timestamp == subquery.c.max_timestamp)
            ).all()
            
            # Organize results
            features = {}
            
            for r in results:
                if r.feature_group not in features:
                    if r.feature_group == r.feature_name:
                        # This is a top-level feature
                        features[r.feature_group] = {
                            'active': r.is_active,
                            'weight': r.weight
                        }
                    else:
                        # This is a group with sub-features
                        features[r.feature_group] = {}
                
                if r.feature_group != r.feature_name:
                    # Add sub-feature
                    features[r.feature_group][r.feature_name] = {
                        'active': r.is_active,
                        'weight': r.weight
                    }
            
            return features
        except Exception as e:
            logger.error(f"Error retrieving active features: {str(e)}")
            return {}
        finally:
            self.close_session(session)
    
    def get_latest_sentiment(self):
        """
        Get the most recent sentiment data
        
        Returns:
            Dictionary with sentiment data
        """
        try:
            session = self.get_session()
            
            latest = session.query(Sentiment).order_by(Sentiment.timestamp.desc()).first()
            
            if latest:
                return {
                    'timestamp': latest.timestamp,
                    'fear_greed_index': latest.fear_greed_index,
                    'market_sentiment': latest.market_sentiment,
                    'source': latest.source
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Error retrieving sentiment data: {str(e)}")
            return None
        finally:
            self.close_session(session)