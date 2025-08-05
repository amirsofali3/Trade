import logging
import pandas as pd
import numpy as np
from database.db_manager import DatabaseManager

logger = logging.getLogger("IndicatorCalculator")

class IndicatorCalculator:
    """Calculates technical indicators from OHLCV data"""
    
    def __init__(self, db_manager):
        """
        Initialize indicator calculator
        
        Args:
            db_manager: Database manager instance
        """
        self.db_manager = db_manager
        
        # Define available indicators
        self.available_indicators = {
            # Trend indicators
            'sma': self._calculate_sma,
            'ema': self._calculate_ema,
            'macd': self._calculate_macd,
            'adx': self._calculate_adx,
            'supertrend': self._calculate_supertrend,
            
            # Momentum indicators
            'rsi': self._calculate_rsi,
            'stoch': self._calculate_stoch,
            'stochrsi': self._calculate_stochrsi,
            'mfi': self._calculate_mfi,
            'willr': self._calculate_willr,
            
            # Volatility indicators
            'bbands': self._calculate_bbands,
            'atr': self._calculate_atr,
            
            # Volume indicators
            'obv': self._calculate_obv,
            'ad': self._calculate_ad,
            'vwap': self._calculate_vwap,
            
            # Pattern recognition
            'engulfing': self._calculate_engulfing,
            'hammer': self._calculate_hammer,
            'doji': self._calculate_doji
        }
        
        # Define indicator parameters
        self.default_params = {
            'sma': {'timeperiod': 20},
            'ema': {'timeperiod': 20},
            'macd': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
            'rsi': {'timeperiod': 14},
            'stoch': {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3},
            'stochrsi': {'timeperiod': 14, 'fastk_period': 5, 'fastd_period': 3},
            'bbands': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
            'adx': {'timeperiod': 14},
            'atr': {'timeperiod': 14},
            'supertrend': {'period': 10, 'multiplier': 3},
            'willr': {'timeperiod': 14},
            'mfi': {'timeperiod': 14},
            'obv': {},
            'ad': {},
            'vwap': {'timeperiod': 14},
            'engulfing': {},
            'hammer': {},
            'doji': {}
        }
        
        logger.info("Indicator calculator initialized")
    
    def calculate_indicators(self, symbol, timeframe, indicators=None, params=None):
        """
        Calculate specified indicators and save to database
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g. '1m', '5m', '1h')
            indicators: List of indicators to calculate or None for all
            params: Dictionary of parameter overrides for indicators
            
        Returns:
            Dictionary with calculated indicators
        """
        try:
            # Get OHLCV data
            ohlcv_data = self.db_manager.get_ohlcv(symbol, timeframe)
            
            if ohlcv_data.empty:
                logger.warning(f"No OHLCV data available for {symbol} {timeframe}")
                return {}
            
            # Use all available indicators if none specified
            if not indicators:
                indicators = list(self.available_indicators.keys())
            
            # Merge default parameters with overrides
            combined_params = {}
            if params:
                for ind in indicators:
                    combined_params[ind] = {**self.default_params.get(ind, {}), **(params.get(ind, {}))}
            else:
                combined_params = self.default_params
            
            results = {}
            
            for ind in indicators:
                if ind not in self.available_indicators:
                    logger.warning(f"Indicator {ind} not available, skipping")
                    continue
                
                try:
                    # Calculate indicator
                    calculator = self.available_indicators[ind]
                    ind_params = combined_params.get(ind, {})
                    
                    ind_result = calculator(ohlcv_data, **ind_params)
                    
                    if not isinstance(ind_result, dict):
                        ind_result = {ind: ind_result}
                    
                    # Save each indicator to database
                    for name, values in ind_result.items():
                        if isinstance(values, pd.Series):
                            self.db_manager.save_indicator(symbol, timeframe, name, values)
                            results[name] = values.tolist()
                        else:
                            logger.warning(f"Indicator {name} result is not a Series")
                
                except Exception as e:
                    logger.error(f"Error calculating indicator {ind}: {str(e)}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error in calculate_indicators: {str(e)}")
            return {}
    
    # Indicator calculation methods
    def _calculate_sma(self, ohlcv, timeperiod=20):
        """Calculate Simple Moving Average"""
        return ohlcv['close'].rolling(window=timeperiod).mean()
    
    def _calculate_ema(self, ohlcv, timeperiod=20):
        """Calculate Exponential Moving Average"""
        return ohlcv['close'].ewm(span=timeperiod).mean()
    
    def _calculate_macd(self, ohlcv, fastperiod=12, slowperiod=26, signalperiod=9):
        """Calculate MACD"""
        ema_fast = ohlcv['close'].ewm(span=fastperiod).mean()
        ema_slow = ohlcv['close'].ewm(span=slowperiod).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signalperiod).mean()
        hist = macd - signal
        return {'macd': macd, 'macd_signal': signal, 'macd_hist': hist}
    
    def _calculate_rsi(self, ohlcv, timeperiod=14):
        """Calculate Relative Strength Index"""
        delta = ohlcv['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_stoch(self, ohlcv, fastk_period=14, slowk_period=3, slowd_period=3):
        """Calculate Stochastic Oscillator"""
        low_min = ohlcv['low'].rolling(window=fastk_period).min()
        high_max = ohlcv['high'].rolling(window=fastk_period).max()
        k_percent = 100 * (ohlcv['close'] - low_min) / (high_max - low_min)
        slowk = k_percent.rolling(window=slowk_period).mean()
        slowd = slowk.rolling(window=slowd_period).mean()
        return {'stoch_k': slowk, 'stoch_d': slowd}
    
    def _calculate_stochrsi(self, ohlcv, timeperiod=14, fastk_period=5, fastd_period=3):
        """Calculate StochRSI"""
        rsi = self._calculate_rsi(ohlcv, timeperiod)
        rsi_min = rsi.rolling(window=timeperiod).min()
        rsi_max = rsi.rolling(window=timeperiod).max()
        stochrsi = (rsi - rsi_min) / (rsi_max - rsi_min)
        fastk = stochrsi.rolling(window=fastk_period).mean()
        fastd = fastk.rolling(window=fastd_period).mean()
        return {'stochrsi_k': fastk, 'stochrsi_d': fastd}
    
    def _calculate_bbands(self, ohlcv, timeperiod=20, nbdevup=2, nbdevdn=2):
        """Calculate Bollinger Bands"""
        sma = ohlcv['close'].rolling(window=timeperiod).mean()
        std = ohlcv['close'].rolling(window=timeperiod).std()
        upper = sma + (std * nbdevup)
        lower = sma - (std * nbdevdn)
        return {'bbands_upper': upper, 'bbands_middle': sma, 'bbands_lower': lower}
    
    def _calculate_adx(self, ohlcv, timeperiod=14):
        """Calculate Average Directional Index"""
        high_diff = ohlcv['high'].diff()
        low_diff = ohlcv['low'].diff()
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        tr1 = ohlcv['high'] - ohlcv['low']
        tr2 = abs(ohlcv['high'] - ohlcv['close'].shift())
        tr3 = abs(ohlcv['low'] - ohlcv['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=timeperiod).mean()
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=timeperiod).mean()
        tr_smooth = tr.rolling(window=timeperiod).mean()
        
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=timeperiod).mean()
        
        return adx
    
    def _calculate_atr(self, ohlcv, timeperiod=14):
        """Calculate Average True Range"""
        tr1 = ohlcv['high'] - ohlcv['low']
        tr2 = abs(ohlcv['high'] - ohlcv['close'].shift())
        tr3 = abs(ohlcv['low'] - ohlcv['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=timeperiod).mean()
        return atr
    
    def _calculate_supertrend(self, ohlcv, period=10, multiplier=3):
        """Calculate SuperTrend indicator"""
        atr = self._calculate_atr(ohlcv, timeperiod=period)
        
        # Calculate basic upper and lower bands
        hl2 = (ohlcv['high'] + ohlcv['low']) / 2
        basic_upper = hl2 + (multiplier * atr)
        basic_lower = hl2 - (multiplier * atr)
        
        # Initialize supertrend
        supertrend = pd.Series(0.0, index=ohlcv.index)
        trend = pd.Series(1, index=ohlcv.index)  # 1 for uptrend, -1 for downtrend
        
        # Calculate supertrend
        for i in range(1, len(ohlcv)):
            if ohlcv['close'].iloc[i] > basic_upper.iloc[i-1]:
                trend.iloc[i] = 1
            elif ohlcv['close'].iloc[i] < basic_lower.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]
                
                if trend.iloc[i] == 1 and basic_lower.iloc[i] < basic_lower.iloc[i-1]:
                    basic_lower.iloc[i] = basic_lower.iloc[i-1]
                
                if trend.iloc[i] == -1 and basic_upper.iloc[i] > basic_upper.iloc[i-1]:
                    basic_upper.iloc[i] = basic_upper.iloc[i-1]
            
            if trend.iloc[i] == 1:
                supertrend.iloc[i] = basic_lower.iloc[i]
            else:
                supertrend.iloc[i] = basic_upper.iloc[i]
        
        return {'supertrend': supertrend, 'supertrend_trend': trend}
    
    def _calculate_willr(self, ohlcv, timeperiod=14):
        """Calculate Williams' %R"""
        high_max = ohlcv['high'].rolling(window=timeperiod).max()
        low_min = ohlcv['low'].rolling(window=timeperiod).min()
        willr = -100 * (high_max - ohlcv['close']) / (high_max - low_min)
        return willr
    
    def _calculate_mfi(self, ohlcv, timeperiod=14):
        """Calculate Money Flow Index"""
        typical_price = (ohlcv['high'] + ohlcv['low'] + ohlcv['close']) / 3
        money_flow = typical_price * ohlcv['volume']
        
        # Positive and negative money flow
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        # Rolling sums
        positive_mf = positive_flow.rolling(window=timeperiod).sum()
        negative_mf = negative_flow.rolling(window=timeperiod).sum()
        
        # Money flow ratio
        mf_ratio = positive_mf / negative_mf
        mfi = 100 - (100 / (1 + mf_ratio))
        
        return mfi
    
    def _calculate_obv(self, ohlcv):
        """Calculate On-Balance Volume"""
        obv = pd.Series(index=ohlcv.index, dtype='float64')
        obv.iloc[0] = ohlcv['volume'].iloc[0]
        
        for i in range(1, len(ohlcv)):
            if ohlcv['close'].iloc[i] > ohlcv['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + ohlcv['volume'].iloc[i]
            elif ohlcv['close'].iloc[i] < ohlcv['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - ohlcv['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_ad(self, ohlcv):
        """Calculate Accumulation/Distribution Line"""
        clv = ((ohlcv['close'] - ohlcv['low']) - (ohlcv['high'] - ohlcv['close'])) / (ohlcv['high'] - ohlcv['low'])
        ad_volume = clv * ohlcv['volume']
        ad = ad_volume.cumsum()
        return ad
    
    def _calculate_vwap(self, ohlcv, timeperiod=14):
        """Calculate Volume Weighted Average Price"""
        df = ohlcv.copy()
        
        # Check if index is datetime
        if isinstance(df.index, pd.DatetimeIndex):
            # Group by day
            df['date'] = df.index.date
            vwap = pd.Series(index=df.index)
            
            for date, group in df.groupby('date'):
                # Calculate VWAP for each day
                vwap.loc[group.index] = (group['volume'] * (group['high'] + group['low'] + group['close']) / 3).cumsum() / group['volume'].cumsum()
            
            return vwap
        else:
            # If no datetime index, use rolling window
            typical_price = (ohlcv['high'] + ohlcv['low'] + ohlcv['close']) / 3
            vwap = (typical_price * ohlcv['volume']).rolling(window=timeperiod).sum() / ohlcv['volume'].rolling(window=timeperiod).sum()
            return vwap
    
    def _calculate_engulfing(self, ohlcv):
        """Calculate Engulfing pattern"""
        # Bullish engulfing: previous candle is bearish, current is bullish and engulfs previous
        prev_bearish = ohlcv['close'].shift() < ohlcv['open'].shift()
        curr_bullish = ohlcv['close'] > ohlcv['open']
        curr_high_higher = ohlcv['high'] > ohlcv['high'].shift()
        curr_low_lower = ohlcv['low'] < ohlcv['low'].shift()
        
        bullish_engulfing = prev_bearish & curr_bullish & curr_high_higher & curr_low_lower
        
        # Bearish engulfing: previous candle is bullish, current is bearish and engulfs previous
        prev_bullish = ohlcv['close'].shift() > ohlcv['open'].shift()
        curr_bearish = ohlcv['close'] < ohlcv['open']
        
        bearish_engulfing = prev_bullish & curr_bearish & curr_high_higher & curr_low_lower
        
        # Return as numeric values: 1 for bullish, -1 for bearish, 0 for none
        pattern = pd.Series(0, index=ohlcv.index)
        pattern[bullish_engulfing] = 1
        pattern[bearish_engulfing] = -1
        
        return pattern
    
    def _calculate_hammer(self, ohlcv):
        """Calculate Hammer pattern"""
        body_size = abs(ohlcv['close'] - ohlcv['open'])
        upper_shadow = ohlcv['high'] - ohlcv[['close', 'open']].max(axis=1)
        lower_shadow = ohlcv[['close', 'open']].min(axis=1) - ohlcv['low']
        
        # Hammer conditions: small body, long lower shadow, short upper shadow
        small_body = body_size < (ohlcv['high'] - ohlcv['low']) * 0.3
        long_lower_shadow = lower_shadow > body_size * 2
        short_upper_shadow = upper_shadow < body_size * 0.5
        
        hammer = small_body & long_lower_shadow & short_upper_shadow
        
        return hammer.astype(int)
    
    def _calculate_doji(self, ohlcv):
        """Calculate Doji pattern"""
        body_size = abs(ohlcv['close'] - ohlcv['open'])
        total_range = ohlcv['high'] - ohlcv['low']
        
        # Doji condition: very small body relative to total range
        doji = body_size < total_range * 0.1
        
        return doji.astype(int)