from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class OHLCV(Base):
    __tablename__ = 'ohlcv_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    def __repr__(self):
        return f"<OHLCV(symbol='{self.symbol}', timestamp='{self.timestamp}', close={self.close})>"

class Tick(Base):
    __tablename__ = 'tick_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    is_buyer = Column(Boolean, nullable=False)
    
    def __repr__(self):
        return f"<Tick(symbol='{self.symbol}', timestamp='{self.timestamp}', price={self.price})>"

class Indicator(Base):
    __tablename__ = 'indicators'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    indicator_name = Column(String(50), nullable=False, index=True)
    value = Column(Float, nullable=False)
    
    def __repr__(self):
        return f"<Indicator(symbol='{self.symbol}', name='{self.indicator_name}', timestamp='{self.timestamp}')>"

class Sentiment(Base):
    __tablename__ = 'sentiment'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    fear_greed_index = Column(Integer, nullable=True)
    market_sentiment = Column(String(20), nullable=True)
    source = Column(String(50), nullable=False)
    raw_data = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<Sentiment(timestamp='{self.timestamp}', fear_greed_index={self.fear_greed_index})>"

class OrderBook(Base):
    __tablename__ = 'orderbook'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    price_level = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    is_bid = Column(Boolean, nullable=False)
    
    def __repr__(self):
        return f"<OrderBook(symbol='{self.symbol}', timestamp='{self.timestamp}', price={self.price_level})>"

class Trade(Base):
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    trade_type = Column(String(10), nullable=False)  # 'BUY' or 'SELL'
    price = Column(Float, nullable=False)
    amount = Column(Float, nullable=False)
    cost = Column(Float, nullable=False)
    fee = Column(Float, nullable=False)
    realized_pnl = Column(Float, nullable=True)
    signal_id = Column(Integer, ForeignKey('signals.id'), nullable=True)
    
    signal = relationship("Signal", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade(symbol='{self.symbol}', type='{self.trade_type}', price={self.price})>"

class Signal(Base):
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(20), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    signal_type = Column(String(10), nullable=False)  # 'BUY', 'SELL', 'HOLD'
    price = Column(Float, nullable=True)
    confidence = Column(Float, nullable=False)
    executed = Column(Boolean, default=False)
    features_used = Column(Text, nullable=True)  # JSON string of features used
    
    trades = relationship("Trade", back_populates="signal")
    
    def __repr__(self):
        return f"<Signal(symbol='{self.symbol}', type='{self.signal_type}', confidence={self.confidence})>"

class ActiveFeature(Base):
    __tablename__ = 'active_features'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    feature_group = Column(String(50), nullable=False)  # 'ohlcv', 'indicator', 'sentiment', 'orderbook', etc.
    feature_name = Column(String(50), nullable=False)
    is_active = Column(Boolean, nullable=False)
    weight = Column(Float, nullable=False)
    
    def __repr__(self):
        return f"<ActiveFeature(name='{self.feature_name}', active={self.is_active}, weight={self.weight})>"