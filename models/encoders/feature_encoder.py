import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import torch.nn as nn

logger = logging.getLogger("FeatureEncoder")

class FeatureEncoder:
    """Base class for feature encoders"""
    
    def __init__(self, name):
        """
        Initialize feature encoder
        
        Args:
            name: Name of the encoder
        """
        self.name = name
        self.is_fitted = False
        logger.debug(f"Initialized {name} encoder")
    
    def fit(self, data):
        """
        Fit encoder to data
        
        Args:
            data: Data to fit encoder to
        """
        raise NotImplementedError("Encoder must implement fit method")
    
    def transform(self, data):
        """
        Transform data using encoder
        
        Args:
            data: Data to transform
            
        Returns:
            Transformed data
        """
        raise NotImplementedError("Encoder must implement transform method")
    
    def fit_transform(self, data):
        """
        Fit encoder to data and transform
        
        Args:
            data: Data to fit and transform
            
        Returns:
            Transformed data
        """
        self.fit(data)
        return self.transform(data)

class OHLCVEncoder(FeatureEncoder):
    """Encoder for OHLCV data"""
    
    def __init__(self, window_size=20):
        """
        Initialize OHLCV encoder
        
        Args:
            window_size: Number of candles to use
        """
        super().__init__(name="OHLCV")
        self.window_size = window_size
        self.scaler = StandardScaler()
        
        # Features to extract
        self.features = [
            'open', 'high', 'low', 'close', 'volume',  # Basic OHLCV
            'body_size', 'shadow_size', 'upper_shadow', 'lower_shadow',  # Candle features
            'range', 'close_to_high', 'close_to_low',  # Price action
            'volume_delta'  # Volume feature
        ]
        
        self.output_dim = len(self.features)
        logger.info(f"Initialized OHLCV encoder with window size {window_size} and {self.output_dim} features")
    
    def fit(self, data):
        """
        Fit scaler to OHLCV data
        
        Args:
            data: DataFrame with OHLCV data
        """
        if len(data) < self.window_size:
            logger.warning(f"Data length {len(data)} is less than window size {self.window_size}")
            return
        
        # Extract features
        features = self._extract_features(data)
        
        # Fit scaler
        self.scaler.fit(features)
        self.is_fitted = True
        
        logger.debug(f"Fitted OHLCV encoder on {len(features)} samples")
    
    def transform(self, data):
        """
        Transform OHLCV data
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tensor with encoded features
        """
        if not self.is_fitted:
            logger.warning("OHLCV encoder not fitted, using fit_transform instead")
            return self.fit_transform(data)
        
        if len(data) < self.window_size:
            logger.warning(f"Data length {len(data)} is less than window size {self.window_size}")
            # Pad with zeros if needed
            pad_length = self.window_size - len(data)
            padded_features = np.zeros((self.window_size, self.output_dim))
            
            # Extract features from available data
            available_features = self._extract_features(data)
            available_features = self.scaler.transform(available_features)
            
            # Fill in the available features
            padded_features[pad_length:, :] = available_features
            
            return torch.tensor(padded_features, dtype=torch.float32)
        
        # Extract features
        features = self._extract_features(data)
        
        # Transform with scaler
        scaled_features = self.scaler.transform(features)
        
        # Convert to tensor
        return torch.tensor(scaled_features, dtype=torch.float32)
    
    def _extract_features(self, data):
        """Extract features from OHLCV data"""
        df = data.copy().iloc[-self.window_size:]
        
        # Calculate additional features
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        df['shadow_size'] = (df['high'] - df['low']) / df['open']
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        df['range'] = (df['high'] - df['low']) / df['open']
        df['close_to_high'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['close_to_low'] = (df['high'] - df['close']) / (df['high'] - df['low'])
        df['volume_delta'] = df['volume'].pct_change().fillna(0)
        
        # Normalize volume
        max_volume = df['volume'].max()
        if max_volume > 0:
            df['volume'] = df['volume'] / max_volume
        
        # Normalize OHLC using previous close
        for i in range(1, len(df)):
            prev_close = df['close'].iloc[i-1]
            if prev_close > 0:
                df.loc[df.index[i], 'open'] = df['open'].iloc[i] / prev_close - 1
                df.loc[df.index[i], 'high'] = df['high'].iloc[i] / prev_close - 1
                df.loc[df.index[i], 'low'] = df['low'].iloc[i] / prev_close - 1
                df.loc[df.index[i], 'close'] = df['close'].iloc[i] / prev_close - 1
        
        # First row can't be normalized with previous close
        df.loc[df.index[0], ['open', 'high', 'low', 'close']] = 0
        
        return df[self.features].values

class IndicatorEncoder(FeatureEncoder):
    """Encoder for technical indicators"""
    
    def __init__(self, window_size=20):
        """
        Initialize indicator encoder
        
        Args:
            window_size: Number of candles to use
        """
        super().__init__(name="Indicator")
        self.window_size = window_size
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Track available indicators
        self.available_indicators = []
        self.output_dim = 0
        
        logger.info(f"Initialized Indicator encoder with window size {window_size}")
    
    def fit(self, data):
        """
        Fit scaler to indicator data
        
        Args:
            data: Dictionary with indicator series
        """
        if not data:
            logger.warning("No indicator data provided for fitting")
            return
        
        # Keep track of available indicators
        self.available_indicators = list(data.keys())
        self.output_dim = len(self.available_indicators)
        
        # Combine indicators into a single DataFrame
        combined = pd.DataFrame({name: series for name, series in data.items()})
        
        # Fill NaN values with forward fill then backward fill
        combined = combined.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Take last window_size rows
        combined = combined.iloc[-self.window_size:]
        
        # Fit scaler
        self.scaler.fit(combined.values)
        self.is_fitted = True
        
        logger.debug(f"Fitted Indicator encoder on {len(combined)} samples with {len(self.available_indicators)} indicators")
    
    def transform(self, data):
        """
        Transform indicator data
        
        Args:
            data: Dictionary with indicator series
            
        Returns:
            Tensor with encoded indicators
        """
        if not self.is_fitted:
            logger.warning("Indicator encoder not fitted, using fit_transform instead")
            return self.fit_transform(data)
        
        if not data or not self.available_indicators:
            logger.warning("No indicator data or available indicators")
            # Return zeros with correct shape
            zeros = np.zeros((self.window_size, self.output_dim if self.output_dim > 0 else 1))
            return torch.tensor(zeros, dtype=torch.float32)
        
        # Create DataFrame with same order of indicators as during fitting
        df_dict = {}
        for ind in self.available_indicators:
            if ind in data:
                df_dict[ind] = data[ind]
            else:
                # If indicator not available, use zeros
                logger.warning(f"Indicator {ind} not found in data, using zeros")
                df_dict[ind] = pd.Series(0, index=data[list(data.keys())[0]].index)
        
        combined = pd.DataFrame(df_dict)
        
        # Fill NaN values
        combined = combined.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Handle case when data is shorter than window_size
        if len(combined) < self.window_size:
            # Pad with zeros
            pad_length = self.window_size - len(combined)
            padded_data = np.zeros((self.window_size, combined.shape[1]))
            padded_data[pad_length:] = combined.values
            scaled_data = self.scaler.transform(padded_data)
        else:
            # Take last window_size rows and scale
            combined = combined.iloc[-self.window_size:]
            scaled_data = self.scaler.transform(combined.values)
        
        return torch.tensor(scaled_data, dtype=torch.float32)

class SentimentEncoder(FeatureEncoder):
    """Encoder for market sentiment data"""
    
    def __init__(self):
        """Initialize sentiment encoder"""
        super().__init__(name="Sentiment")
        self.output_dim = 1  # Just one feature: fear & greed index
        self.is_fitted = True  # No fitting required
        
        logger.info(f"Initialized Sentiment encoder with output dim {self.output_dim}")
    
    def fit(self, data):
        """No fitting required"""
        return
    
    def transform(self, data):
        """
        Transform sentiment data
        
        Args:
            data: Dictionary with sentiment data
            
        Returns:
            Tensor with encoded sentiment
        """
        if not data or 'fear_greed_index' not in data:
            logger.warning("No sentiment data available")
            return torch.tensor([[0.0]], dtype=torch.float32)
        
        # Normalize fear & greed index to [-1, 1]
        # 0 = extreme fear, 100 = extreme greed, 50 = neutral
        normalized = (data['fear_greed_index'] - 50) / 50
        
        return torch.tensor([[normalized]], dtype=torch.float32)

class OrderbookEncoder(FeatureEncoder):
    """Encoder for orderbook data"""
    
    def __init__(self, depth=10):
        """
        Initialize orderbook encoder
        
        Args:
            depth: Number of price levels to use
        """
        super().__init__(name="Orderbook")
        self.depth = depth
        self.scaler = StandardScaler()
        
        # Features to extract:
        # - Price levels (normalized)
        # - Quantities at each level (normalized)
        # - Bid-ask spread
        # - Bid-ask imbalance
        # - Cumulative volume at each level
        
        self.output_dim = depth * 4 + 2  # price & quantity for bids/asks + spread & imbalance
        self.is_fitted = True  # No fitting required for orderbook data
        
        logger.info(f"Initialized Orderbook encoder with depth {depth} and output dim {self.output_dim}")
    
    def fit(self, data):
        """No fitting required"""
        return
    
    def transform(self, data):
        """
        Transform orderbook data
        
        Args:
            data: Dictionary with orderbook data
            
        Returns:
            Tensor with encoded orderbook
        """
        if not data or 'bids' not in data or 'asks' not in data:
            logger.warning("No orderbook data available")
            return torch.zeros((self.output_dim,), dtype=torch.float32)
        
        bids = data.get('bids', [])[:self.depth]
        asks = data.get('asks', [])[:self.depth]
        
        # Pad if necessary
        if len(bids) < self.depth:
            bids = bids + [[0, 0]] * (self.depth - len(bids))
        if len(asks) < self.depth:
            asks = asks + [[0, 0]] * (self.depth - len(asks))
        
        # Calculate bid-ask spread
        if bids[0][0] > 0 and asks[0][0] > 0:
            spread = (asks[0][0] - bids[0][0]) / bids[0][0]
        else:
            spread = 0
        
        # Calculate bid-ask imbalance
        bid_volume = sum(qty for _, qty in bids)
        ask_volume = sum(qty for _, qty in asks)
        
        if bid_volume + ask_volume > 0:
            imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        else:
            imbalance = 0
        
        # Normalize price levels relative to mid price
        if bids[0][0] > 0 and asks[0][0] > 0:
            mid_price = (bids[0][0] + asks[0][0]) / 2
        else:
            mid_price = 1  # Default if no valid prices
        
        # Calculate features
        features = []
        
        # Normalized bid prices
        for price, _ in bids:
            features.append(price / mid_price - 1 if mid_price > 0 else 0)
        
        # Normalized ask prices
        for price, _ in asks:
            features.append(price / mid_price - 1 if mid_price > 0 else 0)
        
        # Normalized bid quantities
        total_qty = max(sum(qty for _, qty in bids + asks), 1)
        for _, qty in bids:
            features.append(qty / total_qty if total_qty > 0 else 0)
        
        # Normalized ask quantities
        for _, qty in asks:
            features.append(qty / total_qty if total_qty > 0 else 0)
        
        # Add spread and imbalance
        features.append(spread)
        features.append(imbalance)
        
        return torch.tensor(features, dtype=torch.float32)

class TickDataEncoder(FeatureEncoder):
    """Encoder for tick data"""
    
    def __init__(self, window_size=100):
        """
        Initialize tick data encoder
        
        Args:
            window_size: Number of ticks to use
        """
        super().__init__(name="TickData")
        self.window_size = window_size
        self.scaler = StandardScaler()
        
        # Features: price change, volume, buy/sell flag
        self.output_dim = 3
        
        logger.info(f"Initialized TickData encoder with window size {window_size} and output dim {self.output_dim}")
    
    def fit(self, data):
        """
        Fit scaler to tick data
        
        Args:
            data: DataFrame with tick data
        """
        if len(data) < 2:
            logger.warning("Not enough tick data for fitting")
            return
        
        # Extract features
        features = self._extract_features(data)
        
        # Fit scaler
        self.scaler.fit(features)
        self.is_fitted = True
        
        logger.debug(f"Fitted TickData encoder on {len(features)} samples")
    
    def transform(self, data):
        """
        Transform tick data
        
        Args:
            data: DataFrame with tick data
            
        Returns:
            Tensor with encoded tick data
        """
        if not self.is_fitted:
            logger.warning("TickData encoder not fitted, using fit_transform instead")
            return self.fit_transform(data)
        
        if len(data) < 2:
            logger.warning("Not enough tick data for transform")
            # Return zeros with correct shape
            zeros = np.zeros((self.window_size, self.output_dim))
            return torch.tensor(zeros, dtype=torch.float32)
        
        # Extract features
        features = self._extract_features(data)
        
        # Handle case when data is shorter than window_size
        if len(features) < self.window_size:
            # Pad with zeros
            pad_length = self.window_size - len(features)
            padded_features = np.zeros((self.window_size, features.shape[1]))
            padded_features[pad_length:] = features
            scaled_features = self.scaler.transform(padded_features)
        else:
            # Take last window_size rows and scale
            features = features[-self.window_size:]
            scaled_features = self.scaler.transform(features)
        
        return torch.tensor(scaled_features, dtype=torch.float32)
    
    def _extract_features(self, data):
        """Extract features from tick data"""
        df = data.copy()
        
        # Sort by timestamp if needed
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        
        # Calculate price change
        df['price_change'] = df['price'].pct_change().fillna(0)
        
        # Normalize volume
        max_volume = df['volume'].max()
        if max_volume > 0:
            df['volume_norm'] = df['volume'] / max_volume
        else:
            df['volume_norm'] = 0
        
        # Convert is_buyer to -1 (sell) or 1 (buy)
        df['trade_direction'] = df['is_buyer'].apply(lambda x: 1 if x else -1)
        
        # Extract features
        features = df[['price_change', 'volume_norm', 'trade_direction']].values
        
        return features

class CandlePatternEncoder(FeatureEncoder):
    """Encoder for candlestick patterns"""
    
    def __init__(self, num_patterns=100):
        """
        Initialize candle pattern encoder
        
        Args:
            num_patterns: Maximum number of patterns to detect
        """
        super().__init__(name="CandlePattern")
        self.num_patterns = num_patterns
        self.output_dim = num_patterns
        self.is_fitted = True  # No fitting required
        
        # List of pattern detection functions from talib
        self.pattern_funcs = [
            # Commonly used patterns (example)
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE', 
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 
            'CDLABANDONEDBABY', 'CDLADVANCEBLOCK', 'CDLBELTHOLD', 
            'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU', 'CDLCONCEALBABYSWALL', 
            'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI', 
            'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 
            'CDLEVENINGDOJISTAR', 'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 
            'CDLGRAVESTONEDOJI', 'CDLHAMMER', 'CDLHANGINGMAN', 
            'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE', 
            'CDLHIKKAKE', 'CDLHIKKAKEMOD', 'CDLHOMINGPIGEON', 
            'CDLIDENTICAL3CROWS', 'CDLINNECK', 'CDLINVERTEDHAMMER', 
            'CDLKICKING', 'CDLKICKINGBYLENGTH', 'CDLLADDERBOTTOM', 
            'CDLLONGLEGGEDDOJI', 'CDLLONGLINE', 'CDLMARUBOZU', 
            'CDLMATCHINGLOW', 'CDLMATHOLD', 'CDLMORNINGDOJISTAR', 
            'CDLMORNINGSTAR', 'CDLONNECK', 'CDLPIERCING', 
            'CDLRICKSHAWMAN', 'CDLRISEFALL3METHODS', 'CDLSEPARATINGLINES', 
            'CDLSHOOTINGSTAR', 'CDLSHORTLINE', 'CDLSPINNINGTOP', 
            'CDLSTALLEDPATTERN', 'CDLSTICKSANDWICH', 'CDLTAKURI', 
            'CDLTASUKIGAP', 'CDLTHRUSTING', 'CDLTRISTAR', 
            'CDLUNIQUE3RIVER', 'CDLUPSIDEGAP2CROWS', 'CDLXSIDEGAP3METHODS'
        ][:num_patterns]  # Limit to num_patterns
        
        logger.info(f"Initialized CandlePattern encoder with {len(self.pattern_funcs)} patterns")
    
    def fit(self, data):
        """No fitting required"""
        return
    
    def transform(self, data):
        """
        Transform OHLCV data to detect patterns
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tensor with detected patterns
        """
        if len(data) < 10:  # Need at least 10 candles for pattern detection
            logger.warning("Not enough candles for pattern detection")
            return torch.zeros((self.num_patterns,), dtype=torch.float32)
        
        try:
            import talib
        except ImportError:
            logger.error("TA-Lib not installed, cannot detect candle patterns")
            return torch.zeros((self.num_patterns,), dtype=torch.float32)
        
        # Dictionary to hold pattern results
        patterns = {}
        
        # Get OHLC arrays
        open_price = data['open'].values
        high = data['high'].values
        low = data['low'].values
        close = data['close'].values
        
        # Calculate patterns
        for i, pattern_name in enumerate(self.pattern_funcs):
            if i >= self.num_patterns:
                break
                
            try:
                pattern_func = getattr(talib, pattern_name)
                result = pattern_func(open_price, high, low, close)
                
                # Normalize to range [0, 1] where:
                # 0 = bearish pattern
                # 0.5 = no pattern
                # 1 = bullish pattern
                normalized = (result / 100) / 2 + 0.5
                
                # Use the last value (most recent candle)
                patterns[pattern_name] = normalized[-1]
            except Exception as e:
                logger.warning(f"Error calculating pattern {pattern_name}: {str(e)}")
                patterns[pattern_name] = 0.5
        
        # Create feature vector
        feature_vector = np.array([patterns.get(name, 0.5) for name in self.pattern_funcs])
        
        # Pad if necessary
        if len(feature_vector) < self.num_patterns:
            feature_vector = np.pad(feature_vector, 
                                  (0, self.num_patterns - len(feature_vector)), 
                                  'constant', 
                                  constant_values=0.5)
        
        return torch.tensor(feature_vector, dtype=torch.float32)