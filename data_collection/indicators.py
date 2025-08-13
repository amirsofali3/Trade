import logging
import pandas as pd
import numpy as np
import time
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
        
        # Profiling attributes for indicator timing
        self.last_indicator_profile = {}  # Store last profiling results
        
        # Phase 2: Optimization attributes
        self.indicator_cache = {}  # Cache computed indicator results  
        self.indicator_timestamps = {}  # Track last computation timestamp per indicator
        self.computation_stats = {
            'computed': 0,
            'skipped': 0, 
            'cache_hits': 0
        }
        
        # Define available indicators - Extended to ~100 technical indicators
        self.available_indicators = {
            # Trend indicators (15 total)
            'sma': self._calculate_sma,
            'ema': self._calculate_ema,
            'macd': self._calculate_macd,
            'adx': self._calculate_adx,
            'supertrend': self._calculate_supertrend,
            'ppo': self._calculate_ppo,
            'psar': self._calculate_psar,
            'trix': self._calculate_trix,
            'dmi': self._calculate_dmi,
            'aroon': self._calculate_aroon,
            'cci': self._calculate_cci,
            'dpo': self._calculate_dpo,
            'kst': self._calculate_kst,
            'ichimoku': self._calculate_ichimoku,
            'tema': self._calculate_tema,
            
            # Momentum indicators (25 total)
            'rsi': self._calculate_rsi,
            'stoch': self._calculate_stoch,
            'stochrsi': self._calculate_stochrsi,
            'mfi': self._calculate_mfi,
            'willr': self._calculate_willr,
            'roc': self._calculate_roc,
            'momentum': self._calculate_momentum,
            'bop': self._calculate_bop,
            'apo': self._calculate_apo,
            'cmo': self._calculate_cmo,
            'rsi_2': self._calculate_rsi_2period,
            'rsi_14': self._calculate_rsi_14period,
            'stoch_fast': self._calculate_stoch_fast,
            'stoch_slow': self._calculate_stoch_slow,
            'ultimate_osc': self._calculate_ultimate_oscillator,
            'kama': self._calculate_kama,
            'fisher': self._calculate_fisher_transform,
            'awesome_osc': self._calculate_awesome_oscillator,
            'bias': self._calculate_bias,
            'dmi_adx': self._calculate_dmi_adx,
            'tsi': self._calculate_tsi,
            'elder_ray': self._calculate_elder_ray,
            'schaff_trend': self._calculate_schaff_trend_cycle,
            'chaikin_osc': self._calculate_chaikin_oscillator,
            'mass_index': self._calculate_mass_index,
            
            # Volatility indicators (15 total)
            'bbands': self._calculate_bbands,
            'atr': self._calculate_atr,
            'keltner': self._calculate_keltner_channel,
            'donchian': self._calculate_donchian_channel,
            'volatility': self._calculate_volatility,
            'chaikin_vol': self._calculate_chaikin_volatility,
            'std_dev': self._calculate_standard_deviation,
            'rvi': self._calculate_relative_volatility_index,
            'true_range': self._calculate_true_range,
            'avg_range': self._calculate_average_range,
            'natr': self._calculate_normalized_atr,
            'pvt': self._calculate_price_volume_trend,
            'envelope': self._calculate_envelope,
            'price_channel': self._calculate_price_channel,
            'volatility_system': self._calculate_volatility_system,
            
            # Volume indicators (20 total)
            'obv': self._calculate_obv,
            'ad': self._calculate_ad,
            'vwap': self._calculate_vwap,
            'cmf': self._calculate_cmf,
            'emv': self._calculate_emv,
            'fi': self._calculate_fi,
            'nvi': self._calculate_nvi,
            'pvi': self._calculate_pvi,
            'vol_osc': self._calculate_volume_oscillator,
            'vol_rate': self._calculate_volume_rate_change,
            'klinger': self._calculate_klinger_oscillator,
            'vol_sma': self._calculate_volume_sma,
            'vol_ema': self._calculate_volume_ema,
            'mfv': self._calculate_money_flow_volume,
            'ad_line': self._calculate_ad_line,
            'obv_ma': self._calculate_obv_ma,
            'vol_price_confirm': self._calculate_volume_price_confirmation,
            'vol_weighted_macd': self._calculate_volume_weighted_macd,
            'ease_of_movement': self._calculate_ease_of_movement,
            'vol_accumulation': self._calculate_volume_accumulation,
            
            # Pattern recognition (25 total)
            'engulfing': self._calculate_engulfing,
            'hammer': self._calculate_hammer,
            'doji': self._calculate_doji,
            'shooting_star': self._calculate_shooting_star,
            'hanging_man': self._calculate_hanging_man,
            'morning_star': self._calculate_morning_star,
            'evening_star': self._calculate_evening_star,
            'three_white_soldiers': self._calculate_three_white_soldiers,
            'three_black_crows': self._calculate_three_black_crows,
            'harami': self._calculate_harami,
            'piercing': self._calculate_piercing_pattern,
            'dark_cloud': self._calculate_dark_cloud_cover,
            'spinning_top': self._calculate_spinning_top,
            'marubozu': self._calculate_marubozu,
            'gravestone_doji': self._calculate_gravestone_doji,
            'dragonfly_doji': self._calculate_dragonfly_doji,
            'tweezer': self._calculate_tweezer_patterns,
            'inside_bar': self._calculate_inside_bar,
            'outside_bar': self._calculate_outside_bar,
            'pin_bar': self._calculate_pin_bar,
            'gap_up': self._calculate_gap_up,
            'gap_down': self._calculate_gap_down,
            'long_legged_doji': self._calculate_long_legged_doji,
            'rickshaw_man': self._calculate_rickshaw_man,
            'belt_hold': self._calculate_belt_hold
        }
        
        # Define indicator parameters for all ~100 indicators
        self.default_params = {
            # Trend indicators
            'sma': {'timeperiod': 20},
            'ema': {'timeperiod': 20},
            'macd': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
            'adx': {'timeperiod': 14},
            'supertrend': {'period': 10, 'multiplier': 3},
            'ppo': {'fastperiod': 12, 'slowperiod': 26, 'matype': 0},
            'psar': {'af': 0.02, 'max_af': 0.2},
            'trix': {'timeperiod': 14},
            'dmi': {'timeperiod': 14},
            'aroon': {'timeperiod': 14},
            'cci': {'timeperiod': 20},
            'dpo': {'timeperiod': 20},
            'kst': {'roc1': 10, 'roc2': 15, 'roc3': 20, 'roc4': 30},
            'ichimoku': {'tenkan': 9, 'kijun': 26, 'senkou': 52},
            'tema': {'timeperiod': 30},
            
            # Momentum indicators
            'rsi': {'timeperiod': 14},
            'stoch': {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3},
            'stochrsi': {'timeperiod': 14, 'fastk_period': 5, 'fastd_period': 3},
            'mfi': {'timeperiod': 14},
            'willr': {'timeperiod': 14},
            'roc': {'timeperiod': 10},
            'momentum': {'timeperiod': 10},
            'bop': {},
            'apo': {'fastperiod': 12, 'slowperiod': 26, 'matype': 0},
            'cmo': {'timeperiod': 14},
            'rsi_2': {'timeperiod': 2},
            'rsi_14': {'timeperiod': 14},
            'stoch_fast': {'fastk_period': 5, 'fastd_period': 3},
            'stoch_slow': {'fastk_period': 14, 'slowk_period': 3, 'slowd_period': 3},
            'ultimate_osc': {'timeperiod1': 7, 'timeperiod2': 14, 'timeperiod3': 28},
            'kama': {'timeperiod': 30},
            'fisher': {'timeperiod': 10},
            'awesome_osc': {'fast': 5, 'slow': 34},
            'bias': {'timeperiod': 20},
            'dmi_adx': {'timeperiod': 14},
            'tsi': {'slow': 25, 'fast': 13},
            'elder_ray': {'timeperiod': 13},
            'schaff_trend': {'fast': 23, 'slow': 50, 'factor': 0.5},
            'chaikin_osc': {'fast': 3, 'slow': 10},
            'mass_index': {'fast': 9, 'slow': 25},
            
            # Volatility indicators
            'bbands': {'timeperiod': 20, 'nbdevup': 2, 'nbdevdn': 2},
            'atr': {'timeperiod': 14},
            'keltner': {'timeperiod': 20, 'multiplier': 2},
            'donchian': {'timeperiod': 20},
            'volatility': {'timeperiod': 10},
            'chaikin_vol': {'timeperiod': 10},
            'std_dev': {'timeperiod': 20},
            'rvi': {'timeperiod': 10},
            'true_range': {},
            'avg_range': {'timeperiod': 14},
            'natr': {'timeperiod': 14},
            'pvt': {},
            'envelope': {'timeperiod': 20, 'pct': 2.5},
            'price_channel': {'timeperiod': 20},
            'volatility_system': {'timeperiod': 20, 'factor': 2},
            
            # Volume indicators
            'obv': {},
            'ad': {},
            'vwap': {'timeperiod': 14},
            'cmf': {'timeperiod': 20},
            'emv': {'scale': 1000000},
            'fi': {'timeperiod': 13},
            'nvi': {},
            'pvi': {},
            'vol_osc': {'fast': 5, 'slow': 10},
            'vol_rate': {'timeperiod': 10},
            'klinger': {'fast': 34, 'slow': 55, 'signal': 13},
            'vol_sma': {'timeperiod': 20},
            'vol_ema': {'timeperiod': 20},
            'mfv': {'timeperiod': 14},
            'ad_line': {},
            'obv_ma': {'timeperiod': 10},
            'vol_price_confirm': {'timeperiod': 10},
            'vol_weighted_macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'ease_of_movement': {'timeperiod': 14},
            'vol_accumulation': {'timeperiod': 10},
            
            # Pattern recognition (no parameters needed)
            'engulfing': {},
            'hammer': {},
            'doji': {},
            'shooting_star': {},
            'hanging_man': {},
            'morning_star': {},
            'evening_star': {},
            'three_white_soldiers': {},
            'three_black_crows': {},
            'harami': {},
            'piercing': {},
            'dark_cloud': {},
            'spinning_top': {},
            'marubozu': {},
            'gravestone_doji': {},
            'dragonfly_doji': {},
            'tweezer': {},
            'inside_bar': {},
            'outside_bar': {},
            'pin_bar': {},
            'gap_up': {},
            'gap_down': {},
            'long_legged_doji': {},
            'rickshaw_man': {},
            'belt_hold': {}
        }
        
        logger.info("Indicator calculator initialized")
    
    def calculate_indicators(self, symbol, timeframe, indicators=None, params=None, profile=True, 
                           use_selected_only=False, active_indicators=None):
        """
        Calculate specified indicators and save to database with optional profiling.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe (e.g. '1m', '5m', '1h')
            indicators: List of indicators to calculate or None for all
            params: Dictionary of parameter overrides for indicators
            profile: Whether to profile indicator computation times (default True)
            use_selected_only: Whether to compute only selected indicators from RFE (Phase 2)
            active_indicators: List of active indicator names from RFE selection (Phase 2)
            
        Returns:
            Dictionary with calculated indicators
        """
        try:
            # Reset computation stats for this cycle
            self.computation_stats = {'computed': 0, 'skipped': 0, 'cache_hits': 0}
            
            # Get OHLCV data
            ohlcv_data = self.db_manager.get_ohlcv(symbol, timeframe)
            
            if ohlcv_data.empty:
                logger.warning(f"No OHLCV data available for {symbol} {timeframe}")
                return {}
            
            # Phase 2: Use only selected indicators if enabled
            if use_selected_only and active_indicators:
                indicators = active_indicators
                logger.info(f"Using selected indicators only: {len(indicators)} of {len(self.available_indicators)} total")
            elif not indicators:
                indicators = list(self.available_indicators.keys())
            
            # Check if we can skip computation based on timestamp (Phase 2 optimization)
            last_close_timestamp = ohlcv_data['timestamp'].iloc[-1] if 'timestamp' in ohlcv_data.columns else None
            if self._should_skip_computation(symbol, timeframe, last_close_timestamp):
                logger.info("Skipping indicator computation - no new data since last calculation")
                self.computation_stats['skipped'] = len(indicators)
                return self.indicator_cache.get(f"{symbol}_{timeframe}", {})
            
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
            indicator_times = []  # Track per-indicator timing
            profile_start = time.time()
            
            for ind in indicators:
                if ind not in self.available_indicators:
                    logger.warning(f"Indicator {ind} not available, skipping")
                    continue
                
                try:
                    # Time individual indicator computation
                    ind_start = time.time()
                    
                    # Calculate indicator
                    calculator = self.available_indicators[ind]
                    ind_params = combined_params.get(ind, {})
                    
                    ind_result = calculator(ohlcv_data, **ind_params)
                    self.computation_stats['computed'] += 1
                    
                    # Record timing
                    ind_elapsed = time.time() - ind_start
                    indicator_times.append((ind, ind_elapsed))
                    
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
            
            # Profile logging and storage
            if profile and indicator_times:
                self._log_indicator_profile(indicator_times, profile_start)
            
            # Phase 2: Cache results and update timestamps
            cache_key = f"{symbol}_{timeframe}"
            self.indicator_cache[cache_key] = results.copy()
            if last_close_timestamp:
                self.indicator_timestamps[cache_key] = last_close_timestamp
            
            # Log computation statistics
            stats = self.computation_stats
            logger.info(f"Indicator stats: computed={stats['computed']} skipped={stats['skipped']} cache_hits={stats['cache_hits']}")
            
            return results
        
        except Exception as e:
            logger.error(f"Error in calculate_indicators: {str(e)}")
            return {}
    
    def _log_indicator_profile(self, indicator_times, start_time):
        """
        Log indicator profiling results and store for diagnostics.
        
        Args:
            indicator_times: List of (indicator_name, elapsed_seconds) tuples
            start_time: Start time of profiling
        """
        total_time = time.time() - start_time
        total_indicators = len(indicator_times)
        avg_time = total_time / total_indicators if total_indicators > 0 else 0
        
        # Sort by elapsed time descending to get slowest
        slowest_indicators = sorted(indicator_times, key=lambda x: x[1], reverse=True)
        slowest_n = slowest_indicators[:5]  # Top 5 slowest
        
        # Store last profile for diagnostics
        self.last_indicator_profile = {
            'total': total_indicators,
            'total_time': total_time,
            'avg_time': avg_time,
            'slowest': slowest_n
        }
        
        # Log profile with deterministic ordering
        slowest_str = ', '.join([f"('{name}', {time:.3f})" for name, time in slowest_n])
        logger.info(f"Indicator profile: total={total_indicators} total_time={total_time:.3f}s avg_time={avg_time:.4f}s slowest=[{slowest_str}]")
    
    # Phase 2: Optimization methods
    
    def _should_skip_computation(self, symbol, timeframe, current_timestamp):
        """
        Check if indicator computation should be skipped based on timestamp.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            current_timestamp: Current latest timestamp
            
        Returns:
            bool: True if computation should be skipped
        """
        if not current_timestamp:
            return False
        
        cache_key = f"{symbol}_{timeframe}"
        last_timestamp = self.indicator_timestamps.get(cache_key)
        
        if not last_timestamp:
            return False
        
        # Skip if timestamp hasn't changed (no new data)
        return current_timestamp == last_timestamp
    
    def get_computation_stats(self):
        """
        Get current computation statistics.
        
        Returns:
            dict: Statistics with computed, skipped, cache_hits counts
        """
        return self.computation_stats.copy()
    
    def reset_cache(self):
        """Reset indicator cache and timestamps."""
        self.indicator_cache.clear()
        self.indicator_timestamps.clear()
        logger.info("Indicator cache reset")
    
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
    
    # Additional Trend Indicators (80 new indicators)
    def _calculate_ppo(self, ohlcv, fastperiod=12, slowperiod=26, matype=0):
        """Calculate Percentage Price Oscillator"""
        ema_fast = ohlcv['close'].ewm(span=fastperiod).mean()
        ema_slow = ohlcv['close'].ewm(span=slowperiod).mean()
        ppo = ((ema_fast - ema_slow) / ema_slow) * 100
        return ppo
    
    def _calculate_psar(self, ohlcv, af=0.02, max_af=0.2):
        """Calculate Parabolic SAR"""
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']
        
        psar = pd.Series(index=ohlcv.index, dtype='float64')
        trend = pd.Series(index=ohlcv.index, dtype='int64')
        ep = pd.Series(index=ohlcv.index, dtype='float64')
        af_series = pd.Series(index=ohlcv.index, dtype='float64')
        
        # Initialize
        psar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1  # 1 for uptrend, -1 for downtrend
        ep.iloc[0] = high.iloc[0]
        af_series.iloc[0] = af
        
        for i in range(1, len(ohlcv)):
            if trend.iloc[i-1] == 1:  # Uptrend
                psar.iloc[i] = psar.iloc[i-1] + af_series.iloc[i-1] * (ep.iloc[i-1] - psar.iloc[i-1])
                
                if low.iloc[i] <= psar.iloc[i]:  # Trend reversal
                    trend.iloc[i] = -1
                    psar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = low.iloc[i]
                    af_series.iloc[i] = af
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high.iloc[i]
                        af_series.iloc[i] = min(max_af, af_series.iloc[i-1] + af)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af_series.iloc[i] = af_series.iloc[i-1]
            else:  # Downtrend
                psar.iloc[i] = psar.iloc[i-1] - af_series.iloc[i-1] * (psar.iloc[i-1] - ep.iloc[i-1])
                
                if high.iloc[i] >= psar.iloc[i]:  # Trend reversal
                    trend.iloc[i] = 1
                    psar.iloc[i] = ep.iloc[i-1]
                    ep.iloc[i] = high.iloc[i]
                    af_series.iloc[i] = af
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low.iloc[i]
                        af_series.iloc[i] = min(max_af, af_series.iloc[i-1] + af)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af_series.iloc[i] = af_series.iloc[i-1]
        
        return {'psar': psar, 'psar_trend': trend}
    
    def _calculate_trix(self, ohlcv, timeperiod=14):
        """Calculate TRIX"""
        ema1 = ohlcv['close'].ewm(span=timeperiod).mean()
        ema2 = ema1.ewm(span=timeperiod).mean()
        ema3 = ema2.ewm(span=timeperiod).mean()
        trix = ema3.pct_change() * 10000
        return trix
    
    def _calculate_dmi(self, ohlcv, timeperiod=14):
        """Calculate Directional Movement Index"""
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
        
        return {'plus_di': plus_di, 'minus_di': minus_di}
    
    def _calculate_aroon(self, ohlcv, timeperiod=14):
        """Calculate Aroon Oscillator"""
        high_idx = ohlcv['high'].rolling(window=timeperiod).apply(lambda x: np.argmax(x), raw=False)
        low_idx = ohlcv['low'].rolling(window=timeperiod).apply(lambda x: np.argmin(x), raw=False)
        
        aroon_up = ((timeperiod - high_idx) / timeperiod) * 100
        aroon_down = ((timeperiod - low_idx) / timeperiod) * 100
        aroon_osc = aroon_up - aroon_down
        
        return {'aroon_up': aroon_up, 'aroon_down': aroon_down, 'aroon_osc': aroon_osc}
    
    def _calculate_cci(self, ohlcv, timeperiod=20):
        """Calculate Commodity Channel Index"""
        typical_price = (ohlcv['high'] + ohlcv['low'] + ohlcv['close']) / 3
        sma_tp = typical_price.rolling(window=timeperiod).mean()
        mad = typical_price.rolling(window=timeperiod).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci
    
    def _calculate_dpo(self, ohlcv, timeperiod=20):
        """Calculate Detrended Price Oscillator"""
        sma = ohlcv['close'].rolling(window=timeperiod).mean()
        shift_periods = timeperiod // 2 + 1
        dpo = ohlcv['close'] - sma.shift(shift_periods)
        return dpo
    
    def _calculate_kst(self, ohlcv, roc1=10, roc2=15, roc3=20, roc4=30):
        """Calculate Know Sure Thing"""
        roc1_val = ohlcv['close'].pct_change(roc1) * 100
        roc2_val = ohlcv['close'].pct_change(roc2) * 100
        roc3_val = ohlcv['close'].pct_change(roc3) * 100
        roc4_val = ohlcv['close'].pct_change(roc4) * 100
        
        rcma1 = roc1_val.rolling(window=10).mean()
        rcma2 = roc2_val.rolling(window=10).mean()
        rcma3 = roc3_val.rolling(window=10).mean()
        rcma4 = roc4_val.rolling(window=15).mean()
        
        kst = (rcma1 * 1) + (rcma2 * 2) + (rcma3 * 3) + (rcma4 * 4)
        return kst
    
    def _calculate_ichimoku(self, ohlcv, tenkan=9, kijun=26, senkou=52):
        """Calculate Ichimoku Kinko Hyo"""
        # Tenkan-sen (Conversion Line)
        tenkan_high = ohlcv['high'].rolling(window=tenkan).max()
        tenkan_low = ohlcv['low'].rolling(window=tenkan).min()
        tenkan_sen = (tenkan_high + tenkan_low) / 2
        
        # Kijun-sen (Base Line)
        kijun_high = ohlcv['high'].rolling(window=kijun).max()
        kijun_low = ohlcv['low'].rolling(window=kijun).min()
        kijun_sen = (kijun_high + kijun_low) / 2
        
        # Senkou Span A (Leading Span A)
        senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
        
        # Senkou Span B (Leading Span B)
        senkou_high = ohlcv['high'].rolling(window=senkou).max()
        senkou_low = ohlcv['low'].rolling(window=senkou).min()
        senkou_b = ((senkou_high + senkou_low) / 2).shift(kijun)
        
        # Chikou Span (Lagging Span)
        chikou = ohlcv['close'].shift(-kijun)
        
        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_a': senkou_a,
            'senkou_b': senkou_b,
            'chikou': chikou
        }
    
    def _calculate_tema(self, ohlcv, timeperiod=30):
        """Calculate Triple Exponential Moving Average"""
        ema1 = ohlcv['close'].ewm(span=timeperiod).mean()
        ema2 = ema1.ewm(span=timeperiod).mean()
        ema3 = ema2.ewm(span=timeperiod).mean()
        tema = 3 * (ema1 - ema2) + ema3
        return tema
    
    # Additional Momentum Indicators
    def _calculate_roc(self, ohlcv, timeperiod=10):
        """Calculate Rate of Change"""
        roc = ((ohlcv['close'] - ohlcv['close'].shift(timeperiod)) / ohlcv['close'].shift(timeperiod)) * 100
        return roc
    
    def _calculate_momentum(self, ohlcv, timeperiod=10):
        """Calculate Momentum"""
        momentum = ohlcv['close'] - ohlcv['close'].shift(timeperiod)
        return momentum
    
    def _calculate_bop(self, ohlcv):
        """Calculate Balance of Power"""
        bop = (ohlcv['close'] - ohlcv['open']) / (ohlcv['high'] - ohlcv['low'])
        return bop.fillna(0)
    
    def _calculate_apo(self, ohlcv, fastperiod=12, slowperiod=26, matype=0):
        """Calculate Absolute Price Oscillator"""
        ema_fast = ohlcv['close'].ewm(span=fastperiod).mean()
        ema_slow = ohlcv['close'].ewm(span=slowperiod).mean()
        apo = ema_fast - ema_slow
        return apo
    
    def _calculate_cmo(self, ohlcv, timeperiod=14):
        """Calculate Chande Momentum Oscillator"""
        delta = ohlcv['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=timeperiod).sum()
        loss = -delta.where(delta < 0, 0).rolling(window=timeperiod).sum()
        cmo = 100 * ((gain - loss) / (gain + loss))
        return cmo.fillna(0)
    
    def _calculate_rsi_2period(self, ohlcv, timeperiod=2):
        """Calculate 2-period RSI"""
        return self._calculate_rsi(ohlcv, timeperiod=timeperiod)
    
    def _calculate_rsi_14period(self, ohlcv, timeperiod=14):
        """Calculate 14-period RSI"""
        return self._calculate_rsi(ohlcv, timeperiod=timeperiod)
    
    def _calculate_stoch_fast(self, ohlcv, fastk_period=5, fastd_period=3):
        """Calculate Fast Stochastic"""
        low_min = ohlcv['low'].rolling(window=fastk_period).min()
        high_max = ohlcv['high'].rolling(window=fastk_period).max()
        k_percent = 100 * (ohlcv['close'] - low_min) / (high_max - low_min)
        fastd = k_percent.rolling(window=fastd_period).mean()
        return {'fastk': k_percent, 'fastd': fastd}
    
    def _calculate_stoch_slow(self, ohlcv, fastk_period=14, slowk_period=3, slowd_period=3):
        """Calculate Slow Stochastic"""
        return self._calculate_stoch(ohlcv, fastk_period, slowk_period, slowd_period)
    
    def _calculate_ultimate_oscillator(self, ohlcv, timeperiod1=7, timeperiod2=14, timeperiod3=28):
        """Calculate Ultimate Oscillator"""
        min_low_close = ohlcv[['low', 'close']].min(axis=1).shift()
        max_high_close = ohlcv[['high', 'close']].max(axis=1).shift()
        
        bp = ohlcv['close'] - min_low_close
        tr = max_high_close - min_low_close
        
        avg1 = bp.rolling(window=timeperiod1).sum() / tr.rolling(window=timeperiod1).sum()
        avg2 = bp.rolling(window=timeperiod2).sum() / tr.rolling(window=timeperiod2).sum()
        avg3 = bp.rolling(window=timeperiod3).sum() / tr.rolling(window=timeperiod3).sum()
        
        uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
        return uo
    
    def _calculate_kama(self, ohlcv, timeperiod=30):
        """Calculate Kaufman Adaptive Moving Average"""
        close = ohlcv['close']
        change = abs(close - close.shift(timeperiod))
        volatility = abs(close.diff()).rolling(window=timeperiod).sum()
        
        er = change / volatility  # Efficiency Ratio
        er = er.fillna(0)
        
        # Smoothing constants
        sc = (er * (2.0 / (2 + 1) - 2.0 / (30 + 1)) + 2.0 / (30 + 1)) ** 2
        
        kama = pd.Series(index=close.index, dtype='float64')
        kama.iloc[0] = close.iloc[0]
        
        for i in range(1, len(close)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])
        
        return kama
    
    def _calculate_fisher_transform(self, ohlcv, timeperiod=10):
        """Calculate Fisher Transform"""
        hl2 = (ohlcv['high'] + ohlcv['low']) / 2
        
        max_high = hl2.rolling(window=timeperiod).max()
        min_low = hl2.rolling(window=timeperiod).min()
        
        normalized = 2 * ((hl2 - min_low) / (max_high - min_low)) - 1
        normalized = np.clip(normalized, -0.99, 0.99)
        
        fisher = pd.Series(index=ohlcv.index, dtype='float64')
        fisher.iloc[0] = 0
        
        for i in range(1, len(normalized)):
            if not pd.isna(normalized.iloc[i]):
                fisher.iloc[i] = 0.5 * np.log((1 + normalized.iloc[i]) / (1 - normalized.iloc[i]))
            else:
                fisher.iloc[i] = fisher.iloc[i-1]
        
        return fisher
    
    def _calculate_awesome_oscillator(self, ohlcv, fast=5, slow=34):
        """Calculate Awesome Oscillator"""
        median_price = (ohlcv['high'] + ohlcv['low']) / 2
        ao = median_price.rolling(window=fast).mean() - median_price.rolling(window=slow).mean()
        return ao
    
    def _calculate_bias(self, ohlcv, timeperiod=20):
        """Calculate Bias Indicator"""
        sma = ohlcv['close'].rolling(window=timeperiod).mean()
        bias = ((ohlcv['close'] - sma) / sma) * 100
        return bias
    
    def _calculate_dmi_adx(self, ohlcv, timeperiod=14):
        """Calculate DMI with ADX"""
        dmi_result = self._calculate_dmi(ohlcv, timeperiod)
        adx = self._calculate_adx(ohlcv, timeperiod)
        
        return {
            'plus_di': dmi_result['plus_di'],
            'minus_di': dmi_result['minus_di'],
            'adx': adx
        }
    
    def _calculate_tsi(self, ohlcv, slow=25, fast=13):
        """Calculate True Strength Index"""
        momentum = ohlcv['close'].diff()
        
        # Double smoothed momentum
        sm1 = momentum.ewm(span=slow).mean()
        sm2 = sm1.ewm(span=fast).mean()
        
        # Double smoothed absolute momentum
        abs_sm1 = abs(momentum).ewm(span=slow).mean()
        abs_sm2 = abs_sm1.ewm(span=fast).mean()
        
        tsi = 100 * (sm2 / abs_sm2)
        return tsi.fillna(0)
    
    def _calculate_elder_ray(self, ohlcv, timeperiod=13):
        """Calculate Elder Ray Index"""
        ema = ohlcv['close'].ewm(span=timeperiod).mean()
        bull_power = ohlcv['high'] - ema
        bear_power = ohlcv['low'] - ema
        
        return {'bull_power': bull_power, 'bear_power': bear_power}
    
    def _calculate_schaff_trend_cycle(self, ohlcv, fast=23, slow=50, factor=0.5):
        """Calculate Schaff Trend Cycle"""
        # Calculate MACD
        ema_fast = ohlcv['close'].ewm(span=fast).mean()
        ema_slow = ohlcv['close'].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        
        # Stochastic of MACD
        stoch_length = 10
        macd_min = macd.rolling(window=stoch_length).min()
        macd_max = macd.rolling(window=stoch_length).max()
        
        pf = pd.Series(index=ohlcv.index, dtype='float64')
        pf.fillna(0, inplace=True)
        
        for i in range(len(macd)):
            if i > 0:
                if macd_max.iloc[i] != macd_min.iloc[i]:
                    v1 = (macd.iloc[i] - macd_min.iloc[i]) / (macd_max.iloc[i] - macd_min.iloc[i]) * 100
                    pf.iloc[i] = pf.iloc[i-1] + factor * (v1 - pf.iloc[i-1])
                else:
                    pf.iloc[i] = pf.iloc[i-1]
        
        return pf
    
    def _calculate_chaikin_oscillator(self, ohlcv, fast=3, slow=10):
        """Calculate Chaikin Oscillator"""
        ad = self._calculate_ad(ohlcv)
        chaikin_osc = ad.ewm(span=fast).mean() - ad.ewm(span=slow).mean()
        return chaikin_osc
    
    def _calculate_mass_index(self, ohlcv, fast=9, slow=25):
        """Calculate Mass Index"""
        high_low = ohlcv['high'] - ohlcv['low']
        ema1 = high_low.ewm(span=fast).mean()
        ema2 = ema1.ewm(span=fast).mean()
        
        mass_index = (ema1 / ema2).rolling(window=slow).sum()
        return mass_index
    
    # Additional Volatility Indicators
    def _calculate_keltner_channel(self, ohlcv, timeperiod=20, multiplier=2):
        """Calculate Keltner Channel"""
        ema = ohlcv['close'].ewm(span=timeperiod).mean()
        atr = self._calculate_atr(ohlcv, timeperiod)
        
        upper = ema + (multiplier * atr)
        lower = ema - (multiplier * atr)
        
        return {'keltner_upper': upper, 'keltner_middle': ema, 'keltner_lower': lower}
    
    def _calculate_donchian_channel(self, ohlcv, timeperiod=20):
        """Calculate Donchian Channel"""
        upper = ohlcv['high'].rolling(window=timeperiod).max()
        lower = ohlcv['low'].rolling(window=timeperiod).min()
        middle = (upper + lower) / 2
        
        return {'donchian_upper': upper, 'donchian_middle': middle, 'donchian_lower': lower}
    
    def _calculate_volatility(self, ohlcv, timeperiod=10):
        """Calculate Price Volatility"""
        returns = ohlcv['close'].pct_change()
        volatility = returns.rolling(window=timeperiod).std() * np.sqrt(252)  # Annualized
        return volatility
    
    def _calculate_chaikin_volatility(self, ohlcv, timeperiod=10):
        """Calculate Chaikin Volatility"""
        hl_spread = ohlcv['high'] - ohlcv['low']
        ema_spread = hl_spread.ewm(span=timeperiod).mean()
        chaikin_vol = (ema_spread - ema_spread.shift(timeperiod)) / ema_spread.shift(timeperiod) * 100
        return chaikin_vol
    
    def _calculate_standard_deviation(self, ohlcv, timeperiod=20):
        """Calculate Standard Deviation"""
        std_dev = ohlcv['close'].rolling(window=timeperiod).std()
        return std_dev
    
    def _calculate_relative_volatility_index(self, ohlcv, timeperiod=10):
        """Calculate Relative Volatility Index"""
        close = ohlcv['close']
        std_dev = close.rolling(window=timeperiod).std()
        
        # Calculate RSI on standard deviation
        delta = std_dev.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
        
        rs = gain / loss
        rvi = 100 - (100 / (1 + rs))
        return rvi
    
    def _calculate_true_range(self, ohlcv):
        """Calculate True Range"""
        tr1 = ohlcv['high'] - ohlcv['low']
        tr2 = abs(ohlcv['high'] - ohlcv['close'].shift())
        tr3 = abs(ohlcv['low'] - ohlcv['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr
    
    def _calculate_average_range(self, ohlcv, timeperiod=14):
        """Calculate Average Range"""
        high_low = ohlcv['high'] - ohlcv['low']
        avg_range = high_low.rolling(window=timeperiod).mean()
        return avg_range
    
    def _calculate_normalized_atr(self, ohlcv, timeperiod=14):
        """Calculate Normalized ATR"""
        atr = self._calculate_atr(ohlcv, timeperiod)
        natr = (atr / ohlcv['close']) * 100
        return natr
    
    def _calculate_price_volume_trend(self, ohlcv):
        """Calculate Price Volume Trend"""
        price_change_pct = ohlcv['close'].pct_change()
        pvt = (price_change_pct * ohlcv['volume']).cumsum()
        return pvt
    
    def _calculate_envelope(self, ohlcv, timeperiod=20, pct=2.5):
        """Calculate Envelope"""
        sma = ohlcv['close'].rolling(window=timeperiod).mean()
        upper = sma * (1 + pct/100)
        lower = sma * (1 - pct/100)
        
        return {'envelope_upper': upper, 'envelope_middle': sma, 'envelope_lower': lower}
    
    def _calculate_price_channel(self, ohlcv, timeperiod=20):
        """Calculate Price Channel"""
        highest = ohlcv['close'].rolling(window=timeperiod).max()
        lowest = ohlcv['close'].rolling(window=timeperiod).min()
        middle = (highest + lowest) / 2
        
        return {'price_channel_upper': highest, 'price_channel_middle': middle, 'price_channel_lower': lowest}
    
    def _calculate_volatility_system(self, ohlcv, timeperiod=20, factor=2):
        """Calculate Volatility System"""
        sma = ohlcv['close'].rolling(window=timeperiod).mean()
        std_dev = ohlcv['close'].rolling(window=timeperiod).std()
        
        upper = sma + (factor * std_dev)
        lower = sma - (factor * std_dev)
        
        return {'vol_sys_upper': upper, 'vol_sys_middle': sma, 'vol_sys_lower': lower}
    
    # Additional Volume Indicators
    def _calculate_cmf(self, ohlcv, timeperiod=20):
        """Calculate Chaikin Money Flow"""
        clv = ((ohlcv['close'] - ohlcv['low']) - (ohlcv['high'] - ohlcv['close'])) / (ohlcv['high'] - ohlcv['low'])
        clv = clv.fillna(0)
        cmf = (clv * ohlcv['volume']).rolling(window=timeperiod).sum() / ohlcv['volume'].rolling(window=timeperiod).sum()
        return cmf
    
    def _calculate_emv(self, ohlcv, scale=1000000):
        """Calculate Ease of Movement"""
        distance_moved = ((ohlcv['high'] + ohlcv['low']) / 2) - ((ohlcv['high'].shift() + ohlcv['low'].shift()) / 2)
        box_height = ohlcv['high'] - ohlcv['low']
        
        # Avoid division by zero
        box_height = box_height.replace(0, np.nan)
        
        emv = (distance_moved / box_height) / (ohlcv['volume'] / scale)
        return emv.fillna(0)
    
    def _calculate_fi(self, ohlcv, timeperiod=13):
        """Calculate Force Index"""
        price_change = ohlcv['close'] - ohlcv['close'].shift()
        fi = (price_change * ohlcv['volume']).ewm(span=timeperiod).mean()
        return fi
    
    def _calculate_nvi(self, ohlcv):
        """Calculate Negative Volume Index"""
        nvi = pd.Series(index=ohlcv.index, dtype='float64')
        nvi.iloc[0] = 1000
        
        for i in range(1, len(ohlcv)):
            if ohlcv['volume'].iloc[i] < ohlcv['volume'].iloc[i-1]:
                pct_change = (ohlcv['close'].iloc[i] - ohlcv['close'].iloc[i-1]) / ohlcv['close'].iloc[i-1]
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + pct_change)
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        
        return nvi
    
    def _calculate_pvi(self, ohlcv):
        """Calculate Positive Volume Index"""
        pvi = pd.Series(index=ohlcv.index, dtype='float64')
        pvi.iloc[0] = 1000
        
        for i in range(1, len(ohlcv)):
            if ohlcv['volume'].iloc[i] > ohlcv['volume'].iloc[i-1]:
                pct_change = (ohlcv['close'].iloc[i] - ohlcv['close'].iloc[i-1]) / ohlcv['close'].iloc[i-1]
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + pct_change)
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        
        return pvi
    
    def _calculate_volume_oscillator(self, ohlcv, fast=5, slow=10):
        """Calculate Volume Oscillator"""
        fast_vol = ohlcv['volume'].rolling(window=fast).mean()
        slow_vol = ohlcv['volume'].rolling(window=slow).mean()
        vol_osc = ((fast_vol - slow_vol) / slow_vol) * 100
        return vol_osc
    
    def _calculate_volume_rate_change(self, ohlcv, timeperiod=10):
        """Calculate Volume Rate of Change"""
        vol_roc = ((ohlcv['volume'] - ohlcv['volume'].shift(timeperiod)) / ohlcv['volume'].shift(timeperiod)) * 100
        return vol_roc
    
    def _calculate_klinger_oscillator(self, ohlcv, fast=34, slow=55, signal=13):
        """Calculate Klinger Oscillator"""
        # Calculate trend
        hlc3 = (ohlcv['high'] + ohlcv['low'] + ohlcv['close']) / 3
        trend = pd.Series(index=ohlcv.index, dtype='int64')
        
        for i in range(1, len(hlc3)):
            if hlc3.iloc[i] > hlc3.iloc[i-1]:
                trend.iloc[i] = 1
            elif hlc3.iloc[i] < hlc3.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1] if i > 0 else 0
        
        # Calculate volume force
        vf = ohlcv['volume'] * trend * 100
        
        # Calculate Klinger
        klinger = vf.ewm(span=fast).mean() - vf.ewm(span=slow).mean()
        klinger_signal = klinger.ewm(span=signal).mean()
        
        return {'klinger': klinger, 'klinger_signal': klinger_signal}
    
    def _calculate_volume_sma(self, ohlcv, timeperiod=20):
        """Calculate Volume Simple Moving Average"""
        return ohlcv['volume'].rolling(window=timeperiod).mean()
    
    def _calculate_volume_ema(self, ohlcv, timeperiod=20):
        """Calculate Volume Exponential Moving Average"""
        return ohlcv['volume'].ewm(span=timeperiod).mean()
    
    def _calculate_money_flow_volume(self, ohlcv, timeperiod=14):
        """Calculate Money Flow Volume"""
        typical_price = (ohlcv['high'] + ohlcv['low'] + ohlcv['close']) / 3
        money_flow = typical_price * ohlcv['volume']
        
        # Positive and negative money flow
        pos_flow = money_flow.where(typical_price > typical_price.shift(), 0)
        neg_flow = money_flow.where(typical_price < typical_price.shift(), 0)
        
        mfv = pos_flow.rolling(window=timeperiod).sum() - neg_flow.rolling(window=timeperiod).sum()
        return mfv
    
    def _calculate_ad_line(self, ohlcv):
        """Calculate Accumulation/Distribution Line (same as AD)"""
        return self._calculate_ad(ohlcv)
    
    def _calculate_obv_ma(self, ohlcv, timeperiod=10):
        """Calculate OBV Moving Average"""
        obv = self._calculate_obv(ohlcv)
        obv_ma = obv.rolling(window=timeperiod).mean()
        return obv_ma
    
    def _calculate_volume_price_confirmation(self, ohlcv, timeperiod=10):
        """Calculate Volume Price Confirmation"""
        price_change = ohlcv['close'].diff()
        volume_change = ohlcv['volume'].diff()
        
        # Confirmation when price and volume move in same direction
        confirmation = ((price_change > 0) & (volume_change > 0)) | ((price_change < 0) & (volume_change > 0))
        vpc = confirmation.rolling(window=timeperiod).sum() / timeperiod
        
        return vpc.astype(float)
    
    def _calculate_volume_weighted_macd(self, ohlcv, fast=12, slow=26, signal=9):
        """Calculate Volume Weighted MACD"""
        vwap_fast = ((ohlcv['close'] * ohlcv['volume']).ewm(span=fast).mean() / 
                     ohlcv['volume'].ewm(span=fast).mean())
        vwap_slow = ((ohlcv['close'] * ohlcv['volume']).ewm(span=slow).mean() / 
                     ohlcv['volume'].ewm(span=slow).mean())
        
        vw_macd = vwap_fast - vwap_slow
        vw_signal = vw_macd.ewm(span=signal).mean()
        vw_hist = vw_macd - vw_signal
        
        return {'vw_macd': vw_macd, 'vw_signal': vw_signal, 'vw_hist': vw_hist}
    
    def _calculate_ease_of_movement(self, ohlcv, timeperiod=14):
        """Calculate Ease of Movement (same as EMV but with smoothing)"""
        emv = self._calculate_emv(ohlcv)
        eom = emv.rolling(window=timeperiod).mean()
        return eom
    
    def _calculate_volume_accumulation(self, ohlcv, timeperiod=10):
        """Calculate Volume Accumulation"""
        # Up volume vs down volume accumulation
        up_vol = ohlcv['volume'].where(ohlcv['close'] > ohlcv['close'].shift(), 0)
        down_vol = ohlcv['volume'].where(ohlcv['close'] < ohlcv['close'].shift(), 0)
        
        vol_acc = (up_vol.rolling(window=timeperiod).sum() - 
                   down_vol.rolling(window=timeperiod).sum())
        return vol_acc
    
    # Additional Pattern Recognition Indicators (22 new patterns)
    def _calculate_shooting_star(self, ohlcv):
        """Calculate Shooting Star pattern"""
        body_size = abs(ohlcv['close'] - ohlcv['open'])
        upper_shadow = ohlcv['high'] - ohlcv[['close', 'open']].max(axis=1)
        lower_shadow = ohlcv[['close', 'open']].min(axis=1) - ohlcv['low']
        
        # Shooting star conditions: small body, long upper shadow, short lower shadow
        small_body = body_size < (ohlcv['high'] - ohlcv['low']) * 0.3
        long_upper_shadow = upper_shadow > body_size * 2
        short_lower_shadow = lower_shadow < body_size * 0.5
        
        shooting_star = small_body & long_upper_shadow & short_lower_shadow
        return shooting_star.astype(int)
    
    def _calculate_hanging_man(self, ohlcv):
        """Calculate Hanging Man pattern (same as hammer but in uptrend)"""
        return self._calculate_hammer(ohlcv)
    
    def _calculate_morning_star(self, ohlcv):
        """Calculate Morning Star pattern"""
        # Three candle pattern: bearish, small body (star), bullish
        bearish_first = ohlcv['close'] < ohlcv['open']
        bullish_third = ohlcv['close'].shift(-2) > ohlcv['open'].shift(-2)
        
        # Middle candle (star) has small body
        star_body = abs(ohlcv['close'].shift(-1) - ohlcv['open'].shift(-1))
        star_range = ohlcv['high'].shift(-1) - ohlcv['low'].shift(-1)
        small_star = star_body < star_range * 0.3
        
        # Gap conditions
        gap_down = ohlcv['high'].shift(-1) < ohlcv['low']
        gap_up = ohlcv['low'].shift(-2) > ohlcv['high'].shift(-1)
        
        morning_star = bearish_first & small_star & bullish_third & gap_down & gap_up
        return morning_star.astype(int)
    
    def _calculate_evening_star(self, ohlcv):
        """Calculate Evening Star pattern"""
        # Three candle pattern: bullish, small body (star), bearish
        bullish_first = ohlcv['close'] > ohlcv['open']
        bearish_third = ohlcv['close'].shift(-2) < ohlcv['open'].shift(-2)
        
        # Middle candle (star) has small body
        star_body = abs(ohlcv['close'].shift(-1) - ohlcv['open'].shift(-1))
        star_range = ohlcv['high'].shift(-1) - ohlcv['low'].shift(-1)
        small_star = star_body < star_range * 0.3
        
        # Gap conditions
        gap_up = ohlcv['low'].shift(-1) > ohlcv['high']
        gap_down = ohlcv['high'].shift(-2) < ohlcv['low'].shift(-1)
        
        evening_star = bullish_first & small_star & bearish_third & gap_up & gap_down
        return evening_star.astype(int)
    
    def _calculate_three_white_soldiers(self, ohlcv):
        """Calculate Three White Soldiers pattern"""
        # Three consecutive bullish candles with higher closes
        bullish_1 = ohlcv['close'] > ohlcv['open']
        bullish_2 = ohlcv['close'].shift(-1) > ohlcv['open'].shift(-1)
        bullish_3 = ohlcv['close'].shift(-2) > ohlcv['open'].shift(-2)
        
        higher_closes = ((ohlcv['close'].shift(-1) > ohlcv['close']) & 
                        (ohlcv['close'].shift(-2) > ohlcv['close'].shift(-1)))
        
        three_white = bullish_1 & bullish_2 & bullish_3 & higher_closes
        return three_white.astype(int)
    
    def _calculate_three_black_crows(self, ohlcv):
        """Calculate Three Black Crows pattern"""
        # Three consecutive bearish candles with lower closes
        bearish_1 = ohlcv['close'] < ohlcv['open']
        bearish_2 = ohlcv['close'].shift(-1) < ohlcv['open'].shift(-1)
        bearish_3 = ohlcv['close'].shift(-2) < ohlcv['open'].shift(-2)
        
        lower_closes = ((ohlcv['close'].shift(-1) < ohlcv['close']) & 
                       (ohlcv['close'].shift(-2) < ohlcv['close'].shift(-1)))
        
        three_black = bearish_1 & bearish_2 & bearish_3 & lower_closes
        return three_black.astype(int)
    
    def _calculate_harami(self, ohlcv):
        """Calculate Harami pattern"""
        # Second candle's body is within first candle's body
        first_high = ohlcv[['close', 'open']].max(axis=1)
        first_low = ohlcv[['close', 'open']].min(axis=1)
        second_high = ohlcv[['close', 'open']].shift(-1).max(axis=1)
        second_low = ohlcv[['close', 'open']].shift(-1).min(axis=1)
        
        within_body = (second_high < first_high) & (second_low > first_low)
        
        # Opposite colors
        first_bullish = ohlcv['close'] > ohlcv['open']
        second_bearish = ohlcv['close'].shift(-1) < ohlcv['open'].shift(-1)
        first_bearish = ohlcv['close'] < ohlcv['open']
        second_bullish = ohlcv['close'].shift(-1) > ohlcv['open'].shift(-1)
        
        harami = within_body & ((first_bullish & second_bearish) | (first_bearish & second_bullish))
        return harami.astype(int)
    
    def _calculate_piercing_pattern(self, ohlcv):
        """Calculate Piercing Pattern"""
        # First candle bearish, second bullish, opens below first's low, closes above first's midpoint
        first_bearish = ohlcv['close'] < ohlcv['open']
        second_bullish = ohlcv['close'].shift(-1) > ohlcv['open'].shift(-1)
        
        opens_below = ohlcv['open'].shift(-1) < ohlcv['low']
        first_midpoint = (ohlcv['open'] + ohlcv['close']) / 2
        closes_above_mid = ohlcv['close'].shift(-1) > first_midpoint
        
        piercing = first_bearish & second_bullish & opens_below & closes_above_mid
        return piercing.astype(int)
    
    def _calculate_dark_cloud_cover(self, ohlcv):
        """Calculate Dark Cloud Cover"""
        # First candle bullish, second bearish, opens above first's high, closes below first's midpoint
        first_bullish = ohlcv['close'] > ohlcv['open']
        second_bearish = ohlcv['close'].shift(-1) < ohlcv['open'].shift(-1)
        
        opens_above = ohlcv['open'].shift(-1) > ohlcv['high']
        first_midpoint = (ohlcv['open'] + ohlcv['close']) / 2
        closes_below_mid = ohlcv['close'].shift(-1) < first_midpoint
        
        dark_cloud = first_bullish & second_bearish & opens_above & closes_below_mid
        return dark_cloud.astype(int)
    
    def _calculate_spinning_top(self, ohlcv):
        """Calculate Spinning Top pattern"""
        body_size = abs(ohlcv['close'] - ohlcv['open'])
        upper_shadow = ohlcv['high'] - ohlcv[['close', 'open']].max(axis=1)
        lower_shadow = ohlcv[['close', 'open']].min(axis=1) - ohlcv['low']
        total_range = ohlcv['high'] - ohlcv['low']
        
        # Small body, long shadows on both sides
        small_body = body_size < total_range * 0.3
        long_shadows = (upper_shadow > body_size) & (lower_shadow > body_size)
        
        spinning_top = small_body & long_shadows
        return spinning_top.astype(int)
    
    def _calculate_marubozu(self, ohlcv):
        """Calculate Marubozu pattern"""
        body_size = abs(ohlcv['close'] - ohlcv['open'])
        upper_shadow = ohlcv['high'] - ohlcv[['close', 'open']].max(axis=1)
        lower_shadow = ohlcv[['close', 'open']].min(axis=1) - ohlcv['low']
        total_range = ohlcv['high'] - ohlcv['low']
        
        # Large body, minimal shadows
        large_body = body_size > total_range * 0.9
        small_shadows = (upper_shadow < total_range * 0.05) & (lower_shadow < total_range * 0.05)
        
        marubozu = large_body & small_shadows
        return marubozu.astype(int)
    
    def _calculate_gravestone_doji(self, ohlcv):
        """Calculate Gravestone Doji pattern"""
        body_size = abs(ohlcv['close'] - ohlcv['open'])
        upper_shadow = ohlcv['high'] - ohlcv[['close', 'open']].max(axis=1)
        lower_shadow = ohlcv[['close', 'open']].min(axis=1) - ohlcv['low']
        total_range = ohlcv['high'] - ohlcv['low']
        
        # Very small body, long upper shadow, minimal lower shadow
        very_small_body = body_size < total_range * 0.1
        long_upper = upper_shadow > total_range * 0.7
        small_lower = lower_shadow < total_range * 0.1
        
        gravestone = very_small_body & long_upper & small_lower
        return gravestone.astype(int)
    
    def _calculate_dragonfly_doji(self, ohlcv):
        """Calculate Dragonfly Doji pattern"""
        body_size = abs(ohlcv['close'] - ohlcv['open'])
        upper_shadow = ohlcv['high'] - ohlcv[['close', 'open']].max(axis=1)
        lower_shadow = ohlcv[['close', 'open']].min(axis=1) - ohlcv['low']
        total_range = ohlcv['high'] - ohlcv['low']
        
        # Very small body, long lower shadow, minimal upper shadow
        very_small_body = body_size < total_range * 0.1
        long_lower = lower_shadow > total_range * 0.7
        small_upper = upper_shadow < total_range * 0.1
        
        dragonfly = very_small_body & long_lower & small_upper
        return dragonfly.astype(int)
    
    def _calculate_tweezer_patterns(self, ohlcv):
        """Calculate Tweezer Top/Bottom patterns"""
        # Tweezer tops: similar highs
        similar_highs = abs(ohlcv['high'] - ohlcv['high'].shift(-1)) < (ohlcv['high'] * 0.002)
        tweezer_top = similar_highs
        
        # Tweezer bottoms: similar lows
        similar_lows = abs(ohlcv['low'] - ohlcv['low'].shift(-1)) < (ohlcv['low'] * 0.002)
        tweezer_bottom = similar_lows
        
        # Return combined pattern (1 for top, -1 for bottom, 0 for none)
        pattern = pd.Series(0, index=ohlcv.index)
        pattern[tweezer_top] = 1
        pattern[tweezer_bottom] = -1
        
        return pattern
    
    def _calculate_inside_bar(self, ohlcv):
        """Calculate Inside Bar pattern"""
        # Current bar's high/low is within previous bar's high/low
        inside = ((ohlcv['high'] < ohlcv['high'].shift()) & 
                 (ohlcv['low'] > ohlcv['low'].shift()))
        return inside.astype(int)
    
    def _calculate_outside_bar(self, ohlcv):
        """Calculate Outside Bar pattern"""
        # Current bar's high/low engulfs previous bar's high/low
        outside = ((ohlcv['high'] > ohlcv['high'].shift()) & 
                  (ohlcv['low'] < ohlcv['low'].shift()))
        return outside.astype(int)
    
    def _calculate_pin_bar(self, ohlcv):
        """Calculate Pin Bar pattern"""
        body_size = abs(ohlcv['close'] - ohlcv['open'])
        upper_shadow = ohlcv['high'] - ohlcv[['close', 'open']].max(axis=1)
        lower_shadow = ohlcv[['close', 'open']].min(axis=1) - ohlcv['low']
        total_range = ohlcv['high'] - ohlcv['low']
        
        # Small body with one long shadow (either upper or lower)
        small_body = body_size < total_range * 0.3
        long_shadow = (upper_shadow > total_range * 0.6) | (lower_shadow > total_range * 0.6)
        
        pin_bar = small_body & long_shadow
        return pin_bar.astype(int)
    
    def _calculate_gap_up(self, ohlcv):
        """Calculate Gap Up pattern"""
        gap_up = ohlcv['low'] > ohlcv['high'].shift()
        return gap_up.astype(int)
    
    def _calculate_gap_down(self, ohlcv):
        """Calculate Gap Down pattern"""
        gap_down = ohlcv['high'] < ohlcv['low'].shift()
        return gap_down.astype(int)
    
    def _calculate_long_legged_doji(self, ohlcv):
        """Calculate Long Legged Doji pattern"""
        body_size = abs(ohlcv['close'] - ohlcv['open'])
        upper_shadow = ohlcv['high'] - ohlcv[['close', 'open']].max(axis=1)
        lower_shadow = ohlcv[['close', 'open']].min(axis=1) - ohlcv['low']
        total_range = ohlcv['high'] - ohlcv['low']
        
        # Very small body, long shadows on both sides
        very_small_body = body_size < total_range * 0.1
        long_both_shadows = (upper_shadow > total_range * 0.3) & (lower_shadow > total_range * 0.3)
        
        long_legged = very_small_body & long_both_shadows
        return long_legged.astype(int)
    
    def _calculate_rickshaw_man(self, ohlcv):
        """Calculate Rickshaw Man pattern (same as long legged doji)"""
        return self._calculate_long_legged_doji(ohlcv)
    
    def _calculate_belt_hold(self, ohlcv):
        """Calculate Belt Hold pattern"""
        # Bullish belt hold: opens at low, closes near high
        bullish_belt = ((ohlcv['open'] - ohlcv['low']) < (ohlcv['high'] - ohlcv['low']) * 0.1) & \
                      ((ohlcv['high'] - ohlcv['close']) < (ohlcv['high'] - ohlcv['low']) * 0.1) & \
                      (ohlcv['close'] > ohlcv['open'])
        
        # Bearish belt hold: opens at high, closes near low  
        bearish_belt = ((ohlcv['high'] - ohlcv['open']) < (ohlcv['high'] - ohlcv['low']) * 0.1) & \
                      ((ohlcv['close'] - ohlcv['low']) < (ohlcv['high'] - ohlcv['low']) * 0.1) & \
                      (ohlcv['close'] < ohlcv['open'])
        
        # Return combined pattern (1 for bullish, -1 for bearish, 0 for none)
        pattern = pd.Series(0, index=ohlcv.index)
        pattern[bullish_belt] = 1
        pattern[bearish_belt] = -1
        
        return pattern