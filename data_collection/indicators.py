import logging
import pandas as pd
import numpy as np
import talib
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
        return talib.SMA(ohlcv['close'], timeperiod=timeperiod)
    
    def _calculate_ema(self, ohlcv, timeperiod=20):
        """Calculate Exponential Moving Average"""
        return talib.EMA(ohlcv['close'], timeperiod=timeperiod)
    
    def _calculate_macd(self, ohlcv, fastperiod=12, slowperiod=26, signalperiod=9):
        """Calculate MACD"""
        macd, signal, hist = talib.MACD(
            ohlcv['close'], 
            fastperiod=fastperiod, 
            slowperiod=slowperiod, 
            signalperiod=signalperiod
        )
        return {'macd': macd, 'macd_signal': signal, 'macd_hist': hist}
    
    def _calculate_rsi(self, ohlcv, timeperiod=14):
        """Calculate Relative Strength Index"""
        return talib.RSI(ohlcv['close'], timeperiod=timeperiod)
    
    def _calculate_stoch(self, ohlcv, fastk_period=14, slowk_period=3, slowd_period=3):
        """Calculate Stochastic Oscillator"""
        slowk, slowd = talib.STOCH(
            ohlcv['high'], 
            ohlcv['low'], 
            ohlcv['close'], 
            fastk_period=fastk_period, 
            slowk_period=slowk_period, 
            slowd_period=slowd_period
        )
        return {'stoch_k': slowk, 'stoch_d': slowd}
    
    def _calculate_stochrsi(self, ohlcv, timeperiod=14, fastk_period=5, fastd_period=3):
        """Calculate StochRSI"""
        fastk, fastd = talib.STOCHRSI(
            ohlcv['close'], 
            timeperiod=timeperiod, 
            fastk_period=fastk_period, 
            fastd_period=fastd_period
        )
        return {'stochrsi_k': fastk, 'stochrsi_d': fastd}
    
    def _calculate_bbands(self, ohlcv, timeperiod=20, nbdevup=2, nbdevdn=2):
        """Calculate Bollinger Bands"""
        upper, middle, lower = talib.BBANDS(
            ohlcv['close'], 
            timeperiod=timeperiod, 
            nbdevup=nbdevup, 
            nbdevdn=nbdevdn
        )
        return {'bbands_upper': upper, 'bbands_middle': middle, 'bbands_lower': lower}
    
    def _calculate_adx(self, ohlcv, timeperiod=14):
        """Calculate Average Directional Index"""
        return talib.ADX(ohlcv['high'], ohlcv['low'], ohlcv['close'], timeperiod=timeperiod)
    
    def _calculate_atr(self, ohlcv, timeperiod=14):
        """Calculate Average True Range"""
        return talib.ATR(ohlcv['high'], ohlcv['low'], ohlcv['close'], timeperiod=timeperiod)
    
    def _calculate_supertrend(self, ohlcv, period=10, multiplier=3):
        """Calculate SuperTrend indicator"""
        atr = talib.ATR(ohlcv['high'], ohlcv['low'], ohlcv['close'], timeperiod=period)
        
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
        return talib.WILLR(ohlcv['high'], ohlcv['low'], ohlcv['close'], timeperiod=timeperiod)
    
    def _calculate_mfi(self, ohlcv, timeperiod=14):
        """Calculate Money Flow Index"""
        return talib.MFI(ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'], timeperiod=timeperiod)
    
    def _calculate_obv(self, ohlcv):
        """Calculate On-Balance Volume"""
        return talib.OBV(ohlcv['close'], ohlcv['volume'])
    
    def _calculate_ad(self, ohlcv):
        """Calculate Accumulation/Distribution Line"""
        return talib.AD(ohlcv['high'], ohlcv['low'], ohlcv['close'], ohlcv['volume'])
    
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
        bullish = talib.CDLENGULFING(ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'])
        # Convert to 0-1 where 1 is bullish engulfing and -1 is bearish engulfing
        normalized = bullish / 100
        return normalized
    
    def _calculate_hammer(self, ohlcv):
        """Calculate Hammer pattern"""
        hammer = talib.CDLHAMMER(ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'])
        # Convert to 0-1
        normalized = hammer / 100
        return normalized
    
    def _calculate_doji(self, ohlcv):
        """Calculate Doji pattern"""
        doji = talib.CDLDOJI(ohlcv['open'], ohlcv['high'], ohlcv['low'], ohlcv['close'])
        # Convert to 0-1
        normalized = doji / 100
        return normalized