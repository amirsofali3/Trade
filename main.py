import logging
import os
import sys
import time
import torch
import argparse
from datetime import datetime
import threading
import json
import pandas as pd
import numpy as np

# تنظیم لاگر با پشتیبانی UTF-8 - fixed encoding issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trade.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

# Fix encoding for stdout handler - ensure UTF-8 support
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
        try:
            handler.stream = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
        except Exception:
            # Fallback: set environment variable
            import os
            os.environ['PYTHONIOENCODING'] = 'utf-8'

logger = logging.getLogger("Main")

# وارد کردن ماژول‌های پروژه
from database.db_manager import DatabaseManager
from data_collection.ohlcv_collector import OHLCVCollector
from data_collection.tick_collector import TickCollector
from data_collection.indicators import IndicatorCalculator
from data_collection.sentiment_collector import SentimentCollector
from data_collection.orderbook_collector import OrderbookCollector
from models.encoders.feature_encoder import OHLCVEncoder, IndicatorEncoder, SentimentEncoder, OrderbookEncoder, TickDataEncoder, CandlePatternEncoder
from models.gating import FeatureGatingModule
from models.neural_network import MarketTransformer, OnlineLearner
from models.signal_generator import SignalGenerator
from trading.executor import TradeExecutor
from trading.position_manager import PositionManager
from trading.risk_manager import RiskManager
import app  # Flask web interface

class TradingBot:
    """
    Main trading bot class that orchestrates all components
    """
    
    def __init__(self, config):
        """
        Initialize trading bot
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.is_running = False
        
        # Exception tracking for diagnostics
        self.recent_exceptions = []
        self.max_recent_exceptions = 10
        
        # MySQL connection string
        mysql_config = config.get('mysql', {})
        connection_string = f"mysql+pymysql://{mysql_config.get('user')}:{mysql_config.get('password')}@{mysql_config.get('host')}/{mysql_config.get('database')}"
        
        # Initialize components
        logger.info("Initializing components...")
        
        # Database
        self.db_manager = DatabaseManager(connection_string)
        
        # Data collectors
        self.collectors = {}
        self._init_data_collectors()
        
        # Model components
        self.encoders = {}
        self.model = None
        self.gating = None
        self.learner = None
        self.signal_generator = None
        self._init_model_components()
        
        # Trading components
        self.executor = None
        self.position_manager = None
        self.risk_manager = None
        self._init_trading_components()
        
        # Communication queue with web interface
        self.message_queue = app.bot_message_queue
        
        # Set bot instance for web interface access to RFE and features
        app.bot_instance = self
        
        logger.info("Trading bot initialization complete")
    
    def _create_adaptive_labels(self, ohlcv_data, lookahead=3, initial_threshold=0.002):
        """
        Create adaptive, leak-safe labels with configurable lookahead
        
        Args:
            ohlcv_data: DataFrame with OHLCV data including timestamps
            lookahead: Number of candles to look ahead for labeling (default 3)
            initial_threshold: Initial threshold for price change (default 0.2%)
            
        Returns:
            tuple: (labels array, timestamps array, label_info dict)
        """
        try:
            logger.info(f"Creating adaptive labels with lookahead={lookahead}, threshold={initial_threshold}")
            
            if len(ohlcv_data) < lookahead + 1:
                logger.warning(f"Insufficient data for lookahead labeling: need {lookahead + 1}, have {len(ohlcv_data)}")
                return [], [], {}
            
            labels = []
            timestamps = []
            
            # Use only data where we can look ahead without leakage
            valid_data = ohlcv_data.iloc[:-lookahead]  # Remove last 'lookahead' samples to prevent leakage
            
            for i in range(len(valid_data)):
                current_price = valid_data['close'].iloc[i]
                future_price = ohlcv_data['close'].iloc[i + lookahead]  # Look ahead by 'lookahead' candles
                timestamp = valid_data['timestamp'].iloc[i] if 'timestamp' in valid_data.columns else valid_data.index[i]
                
                if current_price > 0:
                    price_change = (future_price - current_price) / current_price
                    
                    if price_change > initial_threshold:
                        labels.append(0)  # BUY
                    elif price_change < -initial_threshold:
                        labels.append(1)  # SELL  
                    else:
                        labels.append(2)  # HOLD
                else:
                    labels.append(2)  # HOLD for invalid price
                
                timestamps.append(timestamp)
            
            # Check label diversity and apply adaptive thresholds
            label_counts = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
            unique_classes = sum(1 for count in label_counts.values() if count > 0)
            
            logger.info(f"Initial labels: BUY={label_counts[0]}, SELL={label_counts[1]}, HOLD={label_counts[2]}")
            
            # If we have single-class labels, try smaller threshold
            if unique_classes < 2:
                logger.info("Single-class detected, trying smaller threshold (0.001)")
                labels = []
                smaller_threshold = 0.001
                
                for i in range(len(valid_data)):
                    current_price = valid_data['close'].iloc[i]
                    future_price = ohlcv_data['close'].iloc[i + lookahead]
                    
                    if current_price > 0:
                        price_change = (future_price - current_price) / current_price
                        
                        if price_change > smaller_threshold:
                            labels.append(0)  # BUY
                        elif price_change < -smaller_threshold:
                            labels.append(1)  # SELL
                        else:
                            labels.append(2)  # HOLD
                    else:
                        labels.append(2)  # HOLD
                
                label_counts = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
                unique_classes = sum(1 for count in label_counts.values() if count > 0)
                logger.info(f"Smaller threshold labels: BUY={label_counts[0]}, SELL={label_counts[1]}, HOLD={label_counts[2]}")
            
            # If still single-class, use quantile-based method
            if unique_classes < 2:
                logger.info("Still single-class, using quantile-based adaptive thresholds")
                labels = []
                
                # Calculate price changes
                price_changes = []
                for i in range(len(valid_data)):
                    current_price = valid_data['close'].iloc[i]
                    future_price = ohlcv_data['close'].iloc[i + lookahead]
                    
                    if current_price > 0:
                        price_change = (future_price - current_price) / current_price
                        price_changes.append(price_change)
                    else:
                        price_changes.append(0.0)
                
                # Use 30th and 70th percentiles as thresholds
                if price_changes:
                    buy_threshold = np.percentile(price_changes, 70)
                    sell_threshold = np.percentile(price_changes, 30)
                    logger.info(f"Quantile thresholds: BUY>{buy_threshold:.4f}, SELL<{sell_threshold:.4f}")
                    
                    for price_change in price_changes:
                        if price_change > buy_threshold:
                            labels.append(0)  # BUY
                        elif price_change < sell_threshold:
                            labels.append(1)  # SELL
                        else:
                            labels.append(2)  # HOLD
                    
                    label_counts = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
                    unique_classes = sum(1 for count in label_counts.values() if count > 0)
                    logger.info(f"Quantile labels: BUY={label_counts[0]}, SELL={label_counts[1]}, HOLD={label_counts[2]}")
            
            # Final check - if still single class, convert to binary
            if unique_classes < 2:
                logger.warning("Still single-class after adaptive thresholds, converting to binary classification")
                # Convert all non-HOLD to the dominant class, keep some HOLD
                labels = [2 if i % 3 == 0 else (0 if labels.count(0) > labels.count(1) else 1) for i in range(len(labels))]
                label_counts = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
                logger.info(f"Binary conversion: BUY={label_counts[0]}, SELL={label_counts[1]}, HOLD={label_counts[2]}")
            
            label_info = {
                'label_counts': label_counts,
                'unique_classes': unique_classes,
                'total_samples': len(labels),
                'lookahead': lookahead,
                'threshold_used': initial_threshold
            }
            
            return labels, timestamps, label_info
            
        except Exception as e:
            logger.error(f"Error creating adaptive labels: {str(e)}")
            return [], [], {}
    
    def _align_on_timestamps(self, features_by_group, label_series, label_timestamps):
        """
        Align features and labels on timestamps using inner join
        
        Args:
            features_by_group: Dict of feature groups with timestamps
            label_series: Array of labels
            label_timestamps: Array of label timestamps
            
        Returns:
            tuple: (aligned_features_by_group, aligned_labels, common_timestamps)
        """
        try:
            logger.info("Performing timestamp alignment across feature groups and labels...")
            
            if not features_by_group or len(label_series) == 0:
                logger.warning("No features or labels provided for alignment")
                return {}, [], []
            
            # Convert label timestamps to set for fast lookup
            label_timestamp_set = set(label_timestamps)
            
            # Find common timestamps across all groups
            common_timestamps = None
            group_sizes = []
            
            for group_name, group_data in features_by_group.items():
                if hasattr(group_data, 'index'):
                    # DataFrame with timestamp index
                    group_timestamps = set(group_data.index)
                elif isinstance(group_data, dict) and 'timestamps' in group_data:
                    # Dict with explicit timestamps
                    group_timestamps = set(group_data['timestamps'])
                else:
                    logger.warning(f"Group {group_name} has no timestamp information, skipping alignment")
                    continue
                
                group_sizes.append(len(group_timestamps))
                
                if common_timestamps is None:
                    common_timestamps = group_timestamps
                else:
                    common_timestamps = common_timestamps.intersection(group_timestamps)
            
            # Intersect with label timestamps
            if common_timestamps:
                common_timestamps = common_timestamps.intersection(label_timestamp_set)
                common_timestamps = sorted(list(common_timestamps))
            else:
                common_timestamps = []
            
            logger.info(f"Length alignment: arrays={group_sizes}, common_len={len(common_timestamps)}")
            
            if len(common_timestamps) == 0:
                logger.warning("No common timestamps found for alignment")
                return {}, [], []
            
            # Align features
            aligned_features = {}
            for group_name, group_data in features_by_group.items():
                try:
                    if hasattr(group_data, 'index'):
                        # DataFrame - select rows with common timestamps
                        aligned_data = group_data.loc[group_data.index.isin(common_timestamps)].sort_index()
                        aligned_features[group_name] = aligned_data
                    elif isinstance(group_data, dict) and 'timestamps' in group_data:
                        # Dict with timestamps - filter by common timestamps
                        timestamps = group_data['timestamps']
                        data = group_data['data']
                        
                        # Create mask for common timestamps
                        mask = [ts in common_timestamps for ts in timestamps]
                        aligned_timestamps = [ts for i, ts in enumerate(timestamps) if mask[i]]
                        aligned_data = [data[i] for i, keep in enumerate(mask) if keep]
                        
                        aligned_features[group_name] = {
                            'timestamps': aligned_timestamps,
                            'data': aligned_data
                        }
                    else:
                        logger.warning(f"Cannot align group {group_name} - unsupported format")
                except Exception as e:
                    logger.warning(f"Error aligning group {group_name}: {str(e)}")
            
            # Align labels
            aligned_labels = []
            for i, timestamp in enumerate(label_timestamps):
                if timestamp in common_timestamps:
                    aligned_labels.append(label_series[i])
            
            logger.info(f"Aligned features/labels on timestamps: {len(aligned_labels)} samples; groups={list(aligned_features.keys())}")
            
            # Log training label distribution for clarity
            labels_array = np.array(aligned_labels)
            buy_count = np.sum(labels_array == 0)
            sell_count = np.sum(labels_array == 1) 
            hold_count = np.sum(labels_array == 2)
            logger.info(f"Training labels: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}")
            
            return aligned_features, aligned_labels, common_timestamps
            
        except Exception as e:
            logger.error(f"Error in timestamp alignment: {str(e)}")
            return {}, [], []
    
    def _perform_rfe_feature_selection(self):
        """
        Perform RFE feature selection with timestamp alignment and adaptive labeling
        """
        try:
            logger.info("🔍 Starting enhanced RFE feature selection process...")
            
            # Get RFE configuration
            rfe_config = self.config.get('rfe', {})
            if not rfe_config.get('enabled', True):
                logger.info("RFE disabled in configuration")
                return False
            
            lookahead = rfe_config.get('lookahead', 3)
            initial_threshold = rfe_config.get('initial_threshold', 0.002)
            min_samples = rfe_config.get('min_samples', 300)  # Increased from 30 to ensure sufficient samples
            max_candles = self.config.get('data', {}).get('max_candles_for_rfe', 1000)
            
            # Collect training data for RFE
            symbol = self.config.get('trading', {}).get('symbols', ['BTCUSDT'])[0]
            timeframe = self.config.get('trading', {}).get('timeframes', ['5m'])[0]
            
            # Get OHLCV data with more historical data
            logger.info(f"Fetching up to {max_candles} candles for RFE training...")
            ohlcv_data = self.db_manager.get_ohlcv(symbol, timeframe, limit=max_candles)
            
            # If we don't have enough data, try to collect fresh data
            if ohlcv_data.empty or len(ohlcv_data) < min_samples:
                logger.info("Insufficient historical data, collecting fresh data...")
                
                # Try to collect more data in chunks if possible
                fresh_data = self.collectors['ohlcv'].collect_historical_data(symbol, timeframe, limit=max_candles)
                if fresh_data is not None and not fresh_data.empty:
                    ohlcv_data = fresh_data
                    logger.info(f"Collected {len(ohlcv_data)} fresh OHLCV samples")
            
            if ohlcv_data.empty or len(ohlcv_data) < lookahead + 1:
                logger.warning(f"Insufficient data for RFE. Need at least {lookahead + 1} samples.")
                return False
            
            # Ensure timestamp column exists
            if 'timestamp' not in ohlcv_data.columns:
                if ohlcv_data.index.name == 'timestamp' or hasattr(ohlcv_data.index, 'to_pydatetime'):
                    ohlcv_data['timestamp'] = ohlcv_data.index
                else:
                    # Create synthetic timestamps
                    ohlcv_data['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(ohlcv_data), freq='5min')
            
            # Create adaptive labels with no leakage
            logger.info("📊 Creating adaptive, leak-safe labels...")
            labels, label_timestamps, label_info = self._create_adaptive_labels(
                ohlcv_data, lookahead=lookahead, initial_threshold=initial_threshold
            )
            
            if len(labels) == 0:
                logger.warning("Failed to create labels")
                return False
            
            logger.info(f"Training labels: BUY={label_info['label_counts'][0]}, SELL={label_info['label_counts'][1]}, HOLD={label_info['label_counts'][2]} (post-alignment and cleaning)")
            
            # Prepare feature groups with timestamps
            features_by_group = {}
            
            # OHLCV features with timestamps
            valid_ohlcv = ohlcv_data.iloc[:-lookahead]  # Remove last samples to match labels
            features_by_group['ohlcv'] = valid_ohlcv.set_index('timestamp') if 'timestamp' in valid_ohlcv.columns else valid_ohlcv
            
            # Calculate indicators (Phase 2: Support selective computation)
            logger.info("📈 Calculating technical indicators...")
            profile_enabled = self.config.get('indicators', {}).get('profile', True)
            use_selected_only = self.config.get('indicators', {}).get('use_selected_only', False)
            
            # Get active indicators if RFE has been performed and selective mode is enabled
            active_indicators = None
            if use_selected_only and self.gating and self.gating.rfe_performed:
                # Extract indicator names from RFE selected features
                active_indicators = []
                for feature_name, info in self.gating.rfe_selected_features.items():
                    if feature_name.startswith('indicator.') and info['selected']:
                        indicator_name = feature_name.split('.', 1)[1]
                        active_indicators.append(indicator_name)
                logger.info(f"Using {len(active_indicators)} selected indicators from RFE")
            
            indicators_result = self.collectors['indicators'].calculate_indicators(
                symbol, timeframe, 
                profile=profile_enabled,
                use_selected_only=use_selected_only,
                active_indicators=active_indicators
            )
            
            if indicators_result:
                # Convert indicators to timestamped DataFrame - fix fragmentation
                indicator_data = {}
                for name, values in indicators_result.items():
                    if isinstance(values, list) and len(values) > 0:
                        # Align indicators with OHLCV timestamps
                        if len(values) == len(ohlcv_data):
                            indicator_data[name] = values
                        elif len(values) > len(ohlcv_data):
                            indicator_data[name] = values[-len(ohlcv_data):]
                        else:
                            # Pad with forward fill
                            padded = [values[0]] * (len(ohlcv_data) - len(values)) + values
                            indicator_data[name] = padded
                
                # Create DataFrame from dict to avoid fragmentation warning
                indicator_df = pd.DataFrame(indicator_data) if indicator_data else pd.DataFrame()
                
                if not indicator_df.empty:
                    indicator_df.index = ohlcv_data['timestamp'].values
                    # Remove last samples to match labels
                    features_by_group['indicator'] = indicator_df.iloc[:-lookahead]
                    logger.info(f"📈 Prepared {len(indicator_df.columns)} indicators for RFE")
            
            # Add other feature groups (sentiment, orderbook) if available
            try:
                # Placeholder for sentiment (single value, timestamped)
                sentiment_df = pd.DataFrame({'sentiment_score': [0.0] * len(valid_ohlcv)}, 
                                         index=valid_ohlcv['timestamp'].values if 'timestamp' in valid_ohlcv.columns else valid_ohlcv.index)
                features_by_group['sentiment'] = sentiment_df
                
                # Placeholder for orderbook (multiple features, timestamped) 
                orderbook_cols = {f'orderbook_{i}': [0.0] * len(valid_ohlcv) for i in range(42)}
                orderbook_df = pd.DataFrame(orderbook_cols,
                                          index=valid_ohlcv['timestamp'].values if 'timestamp' in valid_ohlcv.columns else valid_ohlcv.index)
                features_by_group['orderbook'] = orderbook_df
            except Exception as e:
                logger.debug(f"Error adding placeholders: {str(e)}")
            
            # Perform timestamp alignment
            logger.info("🔗 Performing timestamp alignment...")
            aligned_features, aligned_labels, common_timestamps = self._align_on_timestamps(
                features_by_group, labels, label_timestamps
            )
            
            if len(aligned_labels) == 0:
                logger.warning("No valid samples after cleaning")
                return False
            
            if len(aligned_labels) < min_samples:
                logger.info(f"RFE samples={len(aligned_labels)} (<{min_samples} threshold) → using fallback ranking")
                # Use fallback method in gating module
                fallback_results = self.gating._perform_fallback_selection(aligned_features, aligned_labels, len(aligned_labels))
                if fallback_results:
                    logger.info(f"Fallback ranking applied: {len(fallback_results)} features ranked")
                    return True
                else:
                    logger.warning("Fallback ranking failed - no features available")
                    return False
            
            # Perform RFE with aligned data
            logger.info(f"🎯 Running RFE on {len(aligned_labels)} aligned samples...")
            logger.info(f"RFE samples={len(aligned_labels)} (≥{min_samples} threshold) → performing full RFE")
            rfe_results = self.gating.perform_rfe_selection(aligned_features, aligned_labels)
            
            if rfe_results:
                # Apply RFE weights and get status
                weights_applied = self.gating.apply_rfe_weights()
                if not weights_applied:
                    logger.warning("Failed to apply RFE weights despite successful selection")
                    return False
                
                # Get and log results
                rfe_summary = self.gating.get_rfe_summary()
                rfe_weights = self.gating.get_rfe_weights()
                
                # Use unified RFE summary instead of duplicate counting
                rfe_summary = self.gating.get_rfe_summary()
                
                # Phase 2: Build active feature masks after successful RFE
                min_active_features = self.config.get('indicators', {}).get('min_active_features', 10)
                active_masks = self.gating.build_active_feature_masks(min_active_features)
                
                total_active = sum(np.sum(mask) for mask in active_masks.values()) if active_masks else 0
                logger.info(f"Built active feature masks: {total_active} total active features")
                
                # Save RFE summary with versioning
                self.gating.save_rfe_summary_with_version()
                
                # Consolidated RFE logging (Phase 2 requirement)
                total_cleaned = rfe_summary.get('total_available', 0)  
                target_features = rfe_summary.get('target_features', self.gating.rfe_n_features)
                selected = rfe_summary['total_selected']
                strong = rfe_summary['strong']
                medium = rfe_summary['medium']
                weak = rfe_summary['weak']
                
                logger.info(f"RFE Summary: selected={selected}/{total_cleaned} (target={target_features}) strong={strong} medium={medium} weak={weak}")
                logger.info("🚀 RFE feature selection completed!")
                
                # Enable use_selected_only for future indicator calculations
                if not self.config.get('indicators', {}).get('use_selected_only', False):
                    self.config['indicators']['use_selected_only'] = True
                    logger.info("Enabled selective indicator computation for future cycles")
                
                return True
            else:
                logger.warning("RFE feature selection failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in RFE feature selection: {str(e)}")
            return False
    
    def _perform_warmup_training(self):
        """
        Perform initial warmup training to avoid stuck confidence values.
        Phase 2: Includes gating freeze and proper progress tracking.
        """
        try:
            # Check if warmup is enabled
            if not self.config.get('training', {}).get('warmup_enabled', True):
                logger.info("Warmup training disabled in configuration")
                return
            
            logger.info("🔥 Starting warmup training to initialize model weights...")
            
            # Phase 2: Initialize warmup status tracking
            self.warmup_status = {
                'status': 'RUNNING',
                'progress': 0.0,
                'batches_completed': 0,
                'total_batches': 50,
                'elapsed_time': 0.0,
                'estimated_remaining': 0.0,
                'last_loss': 0.0,
                'start_time': time.time()
            }
            
            # Phase 2: Freeze gating during warmup
            if self.gating:
                self.gating.freeze_adaptation = True
                logger.info("Freezing gating adaptation during warmup")
            
            # Collect recent data for warmup
            symbol = self.config.get('trading', {}).get('symbols', ['BTCUSDT'])[0]
            timeframe = self.config.get('trading', {}).get('timeframes', ['5m'])[0]
            
            # Get historical data
            ohlcv_data = self.db_manager.get_ohlcv(symbol, timeframe, limit=100)
            if ohlcv_data.empty or len(ohlcv_data) < 20:
                logger.info("Insufficient data for warmup training")
                self.warmup_status['status'] = 'ABORTED'
                return
            
            # Prepare warmup data samples
            warmup_samples = []
            
            # Collect several training samples
            total_batches = min(50, len(ohlcv_data) - 3)
            self.warmup_status['total_batches'] = total_batches
            
            log_every = 10  # Log progress every 10 batches
            
            for i in range(total_batches):
                try:
                    # Use historical data slice
                    sample_data = ohlcv_data.iloc[i:i+20]
                    if len(sample_data) < 20:
                        continue
                    
                    # Calculate indicators for this sample (use selective computation if enabled)
                    profile_enabled = self.config.get('indicators', {}).get('profile', True)
                    use_selected_only = self.config.get('indicators', {}).get('use_selected_only', False)
                    
                    # Get active indicators if selective mode enabled
                    active_indicators = None
                    if use_selected_only and self.gating and self.gating.rfe_performed:
                        active_indicators = []
                        for feature_name, info in self.gating.rfe_selected_features.items():
                            if feature_name.startswith('indicator.') and info['selected']:
                                indicator_name = feature_name.split('.', 1)[1]
                                active_indicators.append(indicator_name)
                    
                    indicators_result = self.collectors['indicators'].calculate_indicators(
                        symbol, timeframe,
                        profile=profile_enabled,
                        use_selected_only=use_selected_only,
                        active_indicators=active_indicators
                    )
                    if not indicators_result:
                        continue
                    
                    # Transform features
                    features = {}
                    
                    # OHLCV features
                    try:
                        ohlcv_features = self.encoders['ohlcv'].transform(sample_data)
                        features['ohlcv'] = ohlcv_features
                    except Exception as e:
                        logger.debug(f"Error transforming OHLCV for warmup sample {i}: {str(e)}")
                        continue
                    
                    # Indicator features (sample from available)
                    try:
                        sample_indicators = {}
                        for name, values in indicators_result.items():
                            if isinstance(values, list) and len(values) > i:
                                # Take a window around this sample
                                start_idx = max(0, i - 10)
                                end_idx = min(len(values), i + 10)
                                sample_indicators[name] = values[start_idx:end_idx]
                        
                        if sample_indicators:
                            indicator_features = self.encoders['indicator'].transform(sample_indicators)
                            features['indicator'] = indicator_features
                    except Exception as e:
                        logger.debug(f"Error transforming indicators for warmup sample {i}: {str(e)}")
                    
                    # Add other features with default values
                    features['sentiment'] = torch.tensor([[0.0]], dtype=torch.float32)
                    features['orderbook'] = torch.zeros((42,), dtype=torch.float32)
                    
                    # Generate label using improved lookahead method
                    if i + 3 < len(ohlcv_data):
                        current_price = ohlcv_data['close'].iloc[i]
                        future_price = ohlcv_data['close'].iloc[i + 3]  # 3 candles ahead
                        
                        if current_price > 0:
                            price_change = (future_price - current_price) / current_price
                            
                            if price_change > 0.002:  # +0.2%
                                label = 0  # BUY
                            elif price_change < -0.002:  # -0.2%
                                label = 1  # SELL
                            else:
                                label = 2  # HOLD
                        else:
                            label = 2  # HOLD for invalid price
                    else:
                        label = 2  # HOLD if not enough future data
                    
                    warmup_samples.append((features, label))
                    
                    # Phase 2: Update progress tracking
                    self.warmup_status['batches_completed'] = i + 1
                    self.warmup_status['progress'] = ((i + 1) / total_batches) * 100.0
                    
                    elapsed = time.time() - self.warmup_status['start_time']
                    self.warmup_status['elapsed_time'] = elapsed
                    
                    if i > 0:
                        estimated_total = elapsed * (total_batches / (i + 1))
                        self.warmup_status['estimated_remaining'] = max(0, estimated_total - elapsed)
                    
                    # Log progress every log_every batches
                    if (i + 1) % log_every == 0:
                        logger.info(f"Warmup training: batches={i + 1}/{total_batches} progress={self.warmup_status['progress']:.1f}%")
                    
                except Exception as e:
                    logger.debug(f"Error creating warmup sample {i}: {str(e)}")
                    continue
            
            if len(warmup_samples) < 5:
                logger.warning("⚠️ Insufficient warmup samples created")
                self.warmup_status['status'] = 'ABORTED'
                return
            
            # Check label diversity
            labels = [sample[1] for sample in warmup_samples]
            unique_labels = set(labels)
            label_counts = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
            
            logger.info(f"Warmup labels: BUY={label_counts[0]}, SELL={label_counts[1]}, HOLD={label_counts[2]}")
            
            if len(unique_labels) < 2:
                logger.warning("⚠️ Single-class labels in warmup data - model may still have stuck confidence")
            
            # Perform warmup training
            max_warmup_seconds = self.config.get('training', {}).get('max_warmup_seconds', 120)
            warmup_results = self.learner.perform_warmup_training(warmup_samples, max_batches=30, max_seconds=max_warmup_seconds)
            
            if warmup_results['batches_completed'] > 0:
                initial_loss = warmup_results.get('initial_loss', 'N/A')
                final_loss = warmup_results.get('final_loss', 'N/A')
                batches = warmup_results['batches_completed']
                
                logger.info(f"Warmup training: batches={batches} loss: {initial_loss} -> {final_loss}")
                logger.info(f"🚀 Warmup training completed: {batches} batches processed")
                
                # Phase 2: Update warmup status
                self.warmup_status['last_loss'] = final_loss if isinstance(final_loss, (int, float)) else 0.0
                self.warmup_status['status'] = 'COMPLETE'
                self.warmup_status['progress'] = 100.0
                
                # Test prediction diversity after warmup
                test_predictions = []
                self.model.eval()
                for _ in range(5):
                    # Test with a few sample features
                    if warmup_samples:
                        features, _ = warmup_samples[0]
                        with torch.no_grad():
                            probs, confidence = self.model(features)
                            test_predictions.append((probs, confidence))
                
                diversity_check = self.learner.check_prediction_diversity(test_predictions)
                if diversity_check['diverse']:
                    logger.info(f"✅ Prediction diversity check passed: confidence_std={diversity_check.get('confidence_std', 0):.4f}")
                else:
                    logger.warning(f"⚠️ Prediction diversity still low after warmup: {diversity_check.get('reason', 'unknown')}")
                    
                # Phase 2: Unfreeze gating after warmup completion
                if self.gating:
                    self.gating.freeze_adaptation = False
                    logger.info("Unfrozen gating adaptation after warmup completion")
                    
            else:
                logger.warning("⚠️ No warmup batches completed - model may have stuck confidence")
                self.warmup_status['status'] = 'ABORTED'
                
        except Exception as e:
            logger.error(f"Error in warmup training: {str(e)}")
            # Phase 2: Mark warmup as aborted on error
            if hasattr(self, 'warmup_status'):
                self.warmup_status['status'] = 'ABORTED'
            # Unfreeze gating even on error
            if self.gating:
                self.gating.freeze_adaptation = False
    
    def _init_data_collectors(self):
        """Initialize data collection components"""
        symbols = self.config.get('trading', {}).get('symbols', ['BTCUSDT'])
        timeframes = self.config.get('trading', {}).get('timeframes', ['5m'])
        
        # OHLCV collector
        self.collectors['ohlcv'] = OHLCVCollector(
            self.db_manager,
            symbols=symbols,
            timeframes=timeframes
        )
        
        # Indicator calculator
        self.collectors['indicators'] = IndicatorCalculator(self.db_manager)
        
        # Tick collector
        self.collectors['tick'] = TickCollector(self.db_manager, symbols=symbols)
        
        # Sentiment collector
        self.collectors['sentiment'] = SentimentCollector(self.db_manager)
        
        # Orderbook collector
        self.collectors['orderbook'] = OrderbookCollector(self.db_manager, symbols=symbols)
    
    def _init_model_components(self):
        """Initialize model components"""
        # Feature encoders
        self.encoders['ohlcv'] = OHLCVEncoder(window_size=20)
        self.encoders['indicator'] = IndicatorEncoder(window_size=20)
        self.encoders['sentiment'] = SentimentEncoder()
        self.encoders['orderbook'] = OrderbookEncoder(depth=10)
        self.encoders['tick_data'] = TickDataEncoder(window_size=100)
        self.encoders['candle_pattern'] = CandlePatternEncoder(num_patterns=100)
        
        # Feature dimensions - Updated for 100 technical indicators
        feature_dims = {
            'ohlcv': (20, 13),  # (seq_length, feature_dim)
            'indicator': (20, 100),  # 100 technical indicators
            'sentiment': (1, 1),
            'orderbook': (1, 42),
            'tick_data': (100, 3),
            'candle_pattern': (1, 100)  # Patterns now included in indicators
        }
        
        # Gating mechanism with RFE support
        feature_groups = {
            'ohlcv': 13,
            'indicator': 100,  # 100 technical indicators
            'sentiment': 1,
            'orderbook': 42,
            'tick_data': 3,
            'candle_pattern': 100  # Keep for compatibility but patterns are in indicators now
        }
        
        self.gating = FeatureGatingModule(
            feature_groups, 
            rfe_enabled=True,  # Enable RFE
            rfe_n_features=15  # Select best 15 features
        )
        
        # Neural network model
        model_config = self.config.get('model', {})
        self.model = MarketTransformer(
            feature_dims=feature_dims,
            hidden_dim=model_config.get('hidden_dim', 128),
            n_layers=model_config.get('n_layers', 2),
            n_heads=model_config.get('n_heads', 4),
            dropout=model_config.get('dropout', 0.1),
            gating_module=self.gating  # Pass gating module for scalar weighting
        )
        
        # Online learner - use config.online.update_interval instead of model.update_interval
        online_config = self.config.get('online', {})
        update_interval_raw = online_config.get('update_interval', 60)
        
        # Safely cast to int to avoid TypeError
        try:
            update_interval = int(update_interval_raw)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid update_interval value '{update_interval_raw}', using default 60s: {str(e)}")
            update_interval = 60
            
        self.learner = OnlineLearner(
            model=self.model,
            gating_module=self.gating,  # Pass gating module for feedback
            lr=model_config.get('learning_rate', 1e-4),
            buffer_size=1000,
            batch_size=32,
            update_interval=update_interval,  # Use safely cast online config
            save_dir='saved_models'
        )
        
        # Signal generator
        trading_config = self.config.get('trading', {})
        self.signal_generator = SignalGenerator(
            model=self.model,
            db_manager=self.db_manager,
            confidence_threshold=trading_config.get('confidence_threshold', 0.7),
            cooldown_period=trading_config.get('cooldown_period', 6),  # 6 candles cooldown
            symbol=trading_config.get('symbols', ['BTCUSDT'])[0]
        )
        
        # Try to load saved model if available
        self._load_latest_model()
    
    def _init_trading_components(self):
        """Initialize trading components"""
        trading_config = self.config.get('trading', {})
        
        # Trading executor
        self.executor = TradeExecutor(
            db_manager=self.db_manager,
            exchange_name=trading_config.get('exchange', 'coinex'),
            demo_mode=trading_config.get('demo_mode', True),
            api_key=trading_config.get('api_key'),
            api_secret=trading_config.get('api_secret'),
            initial_balance=trading_config.get('initial_balance', 100),
            risk_per_trade=trading_config.get('risk_per_trade', 0.02),
            max_open_trades=trading_config.get('max_open_trades', 1)
        )
        
        # Position manager
        self.position_manager = PositionManager(
            db_manager=self.db_manager,
            initial_stop_loss=trading_config.get('initial_stop_loss', 0.03),
            take_profit_levels=trading_config.get('take_profit_levels', [0.05, 0.1, 0.15])
        )
        
        # Risk manager
        self.risk_manager = RiskManager(
            db_manager=self.db_manager,
            max_risk_per_trade=trading_config.get('risk_per_trade', 0.02),
            max_open_risk=trading_config.get('max_open_risk', 0.06),
            max_daily_loss=trading_config.get('max_daily_loss', 0.05),
            initial_capital=trading_config.get('initial_balance', 100)
        )
    
    def _load_latest_model(self):
        """Try to load the latest saved model"""
        save_dir = 'saved_models'
        if not os.path.exists(save_dir):
            logger.info("No saved models found")
            return
        
        # Find the latest model checkpoint
        model_files = [f for f in os.listdir(save_dir) if f.startswith('model_checkpoint_')]
        if not model_files:
            logger.info("No model checkpoints found")
            return
        
        # Sort by update counter
        model_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]), reverse=True)
        latest_model = os.path.join(save_dir, model_files[0])
        
        # Load the model
        if self.learner.load_model(latest_model):
            logger.info(f"Loaded latest model: {latest_model}")
        else:
            logger.warning(f"Failed to load model: {latest_model}")
    
    def start(self):
        """Start the trading bot"""
        if self.is_running:
            logger.warning("Trading bot is already running")
            return
        
        logger.info("Starting trading bot...")
        self.is_running = True
        
        # Start data collectors first
        for name, collector in self.collectors.items():
            if hasattr(collector, 'start'):
                collector.start()
                logger.info(f"Started {name} collector")
        
        # Wait a moment for initial data collection
        time.sleep(5)
        
        # Perform RFE feature selection before training
        logger.info("🧠 Performing RFE feature selection before training...")
        rfe_success = self._perform_rfe_feature_selection()
        
        if rfe_success:
            logger.info("✅ RFE completed - model will use optimally selected features")
            # Perform warmup training to avoid stuck confidence
            self._perform_warmup_training()
        else:
            logger.warning("⚠️ RFE failed - model will use default feature gating")
        
        # Start online learner
        self.learner.start()
        logger.info("Started online learner")
        
        # Start trade executor
        self.executor.start()
        logger.info("Started trade executor")
        
        # Start main loop in a separate thread
        self.main_thread = threading.Thread(target=self._main_loop)
        self.main_thread.daemon = True
        self.main_thread.start()
        logger.info("Started main loop")
        
        # Start web interface in a separate thread
        self.web_thread = threading.Thread(target=self._start_web_interface)
        self.web_thread.daemon = True
        self.web_thread.start()
        logger.info("Started web interface")
    
    def stop(self):
        """Stop the trading bot"""
        if not self.is_running:
            logger.warning("Trading bot is not running")
            return
        
        logger.info("Stopping trading bot...")
        self.is_running = False
        
        # Stop data collectors
        for name, collector in self.collectors.items():
            if hasattr(collector, 'stop'):
                collector.stop()
                logger.info(f"Stopped {name} collector")
        
        # Stop online learner
        self.learner.stop()
        logger.info("Stopped online learner")
        
        # Stop trade executor
        self.executor.stop()
        logger.info("Stopped trade executor")
        
        logger.info("Trading bot stopped")
    
    def _main_loop(self):
        """Main bot loop"""
        while self.is_running:
            try:
                # Process any messages from web interface
                self._process_messages()
                
                # Phase 2: Check for periodic RFE trigger
                if self.gating and self.gating.rfe_performed:
                    # Calculate recent success rate for performance-based triggering
                    current_success_rate = self._calculate_recent_success_rate()
                    
                    if self.gating.should_trigger_periodic_rfe(self.config, current_success_rate):
                        logger.info("Triggering periodic RFE...")
                        rfe_success = self._perform_rfe_feature_selection()
                        if rfe_success:
                            logger.info("Periodic RFE completed successfully")
                        else:
                            logger.warning("Periodic RFE failed")
                
                # Phase 2: Log hourly performance metrics
                self._log_hourly_performance()
                
                # Collect fresh data
                data = self._collect_data()
                
                # Process data
                features = self._process_data(data)
                
                # Generate trading signal (gating is now applied inside the model)
                active_features = self.gating.get_active_features()
                current_price = data.get('ohlcv', pd.DataFrame()).iloc[-1].close if not data.get('ohlcv', pd.DataFrame()).empty else 0
                
                signal = self.signal_generator.generate_signal(
                    features,  # Pass raw features; gating applied internally 
                    active_features,
                    current_price
                )
                
                # Process signal with position and risk management
                if signal:
                    self._process_signal(signal, current_price, data, gated_features, active_features)
                
                # Check for position management (stop loss, take profit adjustments)
                closed_positions = self._manage_positions(current_price)
                
                # Update learning system with trade outcomes
                if closed_positions:
                    self._update_learning_from_outcomes(closed_positions)
                
                # Save active features to database
                self.db_manager.update_active_features(active_features)
                
                # Update web interface with data
                self._update_web_interface(data, features, active_features, signal)
                
                # Sleep to prevent high CPU usage
                time.sleep(5)  # Check every 5 seconds
            
            except Exception as e:
                import traceback
                from datetime import datetime
                
                # Capture exception details
                exception_info = {
                    'timestamp': datetime.now().isoformat(),
                    'message': str(e),
                    'traceback': traceback.format_exc()[:1000]  # Limit traceback size
                }
                
                # Add to recent exceptions buffer
                self.recent_exceptions.append(exception_info)
                if len(self.recent_exceptions) > self.max_recent_exceptions:
                    self.recent_exceptions.pop(0)  # Keep only recent exceptions
                
                logger.error(f"Error in main loop: {str(e)}")
                logger.debug(f"Exception traceback: {traceback.format_exc()}")
                time.sleep(10)  # Wait longer on error
    
    def _process_signal(self, signal, current_price, data, gated_features=None, active_features=None):
        """
        Process a trading signal with position and risk management
        
        Args:
            signal: Signal dictionary
            current_price: Current market price
            data: Data dictionary
            gated_features: Gated features used for this signal
            active_features: Active features information
        """
        try:
            symbol = signal['symbol']
            signal_type = signal['signal_type']
            confidence = signal.get('confidence', 0.5)
            
            # Store signal for learning (regardless of execution)
            if gated_features and active_features:
                # Convert signal type to label
                label = {'BUY': 0, 'SELL': 1, 'HOLD': 2}.get(signal_type, 2)
                
                # Calculate feature contributions
                feature_contributions = {}
                for group_name, group_data in active_features.items():
                    if isinstance(group_data, dict) and 'weight' in group_data:
                        feature_contributions[group_name] = group_data['weight']
                    elif isinstance(group_data, dict):
                        for feature_name, feature_info in group_data.items():
                            if isinstance(feature_info, dict) and 'weight' in feature_info:
                                feature_contributions[f"{group_name}.{feature_name}"] = feature_info['weight']
                
                # Add to learning buffer
                self.learner.add_experience(
                    features=gated_features,
                    label=label,
                    confidence=confidence,
                    feature_contributions=feature_contributions
                )
            
            if signal_type == 'BUY':
                # Check if we already have a position
                if self.position_manager.has_position(symbol):
                    logger.info(f"Already have position for {symbol}, ignoring buy signal")
                    return
                
                # Calculate position parameters
                entry_price = current_price
                stop_loss = entry_price * (1 - self.config.get('trading', {}).get('initial_stop_loss', 0.03))
                
                # Calculate position size based on risk
                position_size = self.risk_manager.calculate_position_size(
                    symbol, 
                    entry_price, 
                    stop_loss, 
                    current_capital=self.executor.demo_balance if hasattr(self.executor, 'demo_balance') else None
                )
                
                # Check if position is allowed by risk manager
                can_open, reason = self.risk_manager.can_open_position(
                    symbol, 
                    entry_price, 
                    stop_loss, 
                    position_size, 
                    self.position_manager.get_all_positions()
                )
                
                if not can_open:
                    logger.warning(f"Risk manager rejected position: {reason}")
                    return
                
                # Add position to position manager
                self.position_manager.add_position(
                    symbol,
                    entry_price,
                    position_size,
                    signal.get('id')
                )
                
                # Mark signal as executed and start cooldown
                self.signal_generator.mark_signal_executed(signal_type, signal['timestamp'])
                
                logger.info(f"Processed BUY signal for {symbol} at {entry_price} (confidence: {confidence:.2f})")
                
            elif signal_type == 'SELL':
                # Check if we have a position to sell
                position = self.position_manager.get_position(symbol)
                
                if not position:
                    logger.info(f"No position for {symbol}, ignoring sell signal")
                    return
                
                # Check confidence threshold for SELL signal execution (70% required)
                if confidence < 0.7:
                    logger.info(f"SELL signal confidence {confidence:.4f} below execution threshold 0.7, keeping position")
                    return
                
                # Remove position from position manager
                closed_position = self.position_manager.remove_position(symbol)
                
                # Calculate P&L for learning
                if closed_position:
                    entry_price = closed_position['entry_price']
                    pnl_percent = (current_price - entry_price) / entry_price
                    
                    # This will be used by learning system
                    closed_position['exit_price'] = current_price
                    closed_position['pnl_percent'] = pnl_percent
                    closed_position['exit_reason'] = 'signal'
                
                # Mark signal as executed and start cooldown
                self.signal_generator.mark_signal_executed(signal_type, signal['timestamp'])
                
                logger.info(f"Processed SELL signal for {symbol} at {current_price} (confidence: {confidence:.2f})")
            
        except Exception as e:
            logger.error(f"Error processing signal: {str(e)}")
    
    def _manage_positions(self, current_price):
        """
        Manage existing positions (check for stop loss/take profit)
        
        Args:
            current_price: Current market price
            
        Returns:
            List of closed positions for learning feedback
        """
        closed_positions = []
        
        try:
            # Get all positions
            positions = self.position_manager.get_all_positions()
            
            for position in positions:
                symbol = position['symbol']
                
                # Update position with current price
                action, updated_position = self.position_manager.update_position_price(symbol, current_price)
                
                if action == 'stop_loss':
                    # Close position due to stop loss
                    closed_position = self.position_manager.remove_position(symbol)
                    
                    if closed_position:
                        entry_price = closed_position['entry_price']
                        pnl_percent = (current_price - entry_price) / entry_price
                        
                        closed_position['exit_price'] = current_price
                        closed_position['pnl_percent'] = pnl_percent
                        closed_position['exit_reason'] = 'stop_loss'
                        closed_positions.append(closed_position)
                    
                    # Mark as executed signal (stop loss is like a SELL signal)
                    self.signal_generator.mark_signal_executed('SELL')
                    
                    # Log the event
                    logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                    
                    # In a real system, you would execute the sell order here
                    
                elif action == 'tp_adjust':
                    # Stop loss was adjusted due to take profit level
                    logger.info(f"Take profit level reached for {symbol}, stop loss moved to {updated_position['stop_loss']}")
        
        except Exception as e:
            logger.error(f"Error managing positions: {str(e)}")
        
        return closed_positions
    
    def _update_learning_from_outcomes(self, closed_positions):
        """
        Update learning system with trade outcomes
        
        Args:
            closed_positions: List of closed position dictionaries
        """
        try:
            for position in closed_positions:
                pnl_percent = position.get('pnl_percent', 0)
                exit_reason = position.get('exit_reason', 'unknown')
                
                # Define success criteria
                was_successful = pnl_percent > 0.01  # More than 1% profit
                
                # Find the corresponding experience in learning buffer
                # This is a simplified approach - in a real system you'd want better tracking
                signal_id = position.get('signal_id')
                if signal_id and len(self.learner.experience_buffer['labels']) > 0:
                    # Update the most recent BUY experience (simplified)
                    # In a more sophisticated system, you'd match by signal_id
                    for i in range(len(self.learner.experience_buffer['labels']) - 1, -1, -1):
                        if (self.learner.experience_buffer['labels'][i] == 0 and  # BUY signal
                            self.learner.experience_buffer['outcomes'][i] is None):  # No outcome yet
                            
                            self.learner.update_trade_outcome(i, pnl_percent, was_successful)
                            break
                
                logger.info(f"Learning update: Position closed with {pnl_percent*100:.2f}% P&L, "
                           f"Success: {'Yes' if was_successful else 'No'}, Reason: {exit_reason}")
        
        except Exception as e:
            logger.error(f"Error updating learning from outcomes: {str(e)}")
    
    def _start_web_interface(self):
        """Start Flask web interface"""
        app.start_web_server(
            host=self.config.get('web', {}).get('host', '0.0.0.0'),
            port=self.config.get('web', {}).get('port', 5000)
        )
    
    def _process_messages(self):
        """Process messages from web interface"""
        while not self.message_queue.empty():
            try:
                message = self.message_queue.get_nowait()
                
                if message.get('type') == 'command':
                    command = message.get('command')
                    
                    if command == 'start':
                        if not self.is_running:
                            self.start()
                    elif command == 'stop':
                        if self.is_running:
                            self.stop()
                    elif command == 'restart':
                        self.stop()
                        time.sleep(1)
                        self.start()
                
                elif message.get('type') == 'settings_update':
                    settings = message.get('settings', {})
                    
                    # Update executor settings
                    if 'risk_per_trade' in settings:
                        risk = float(settings['risk_per_trade'])
                        self.executor.set_risk_per_trade(risk)
                        self.risk_manager.max_risk_per_trade = risk
                    
                    if 'max_open_trades' in settings:
                        self.executor.set_max_open_trades(int(settings['max_open_trades']))
                    
                    # Update signal generator settings
                    if 'confidence_threshold' in settings:
                        self.signal_generator.set_confidence_threshold(float(settings['confidence_threshold']))
            
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
    
    def _collect_data(self):
        """Collect data from database"""
        symbol = self.config.get('trading', {}).get('symbols', ['BTCUSDT'])[0]
        timeframe = self.config.get('trading', {}).get('timeframes', ['5m'])[0]
        
        # Collect OHLCV data
        ohlcv_data = self.db_manager.get_ohlcv(symbol, timeframe, limit=100)
        
        # Calculate indicators
        indicators = {}
        if not ohlcv_data.empty:
            profile_enabled = self.config.get('indicators', {}).get('profile', True)
            indicators_result = self.collectors['indicators'].calculate_indicators(symbol, timeframe, profile=profile_enabled)
            indicators = {name: values[-20:] if len(values) > 20 else values for name, values in indicators_result.items()}
        
        # Get latest sentiment
        sentiment = self.db_manager.get_latest_sentiment()
        
        # Get active features
        active_features = self.db_manager.get_active_features()
        
        return {
            'ohlcv': ohlcv_data,
            'indicators': indicators,
            'sentiment': sentiment,
            'active_features': active_features
        }
    
    def _process_data(self, data):
        """
        Process and encode data for the model
        
        Args:
            data: Dictionary with collected data
            
        Returns:
            Dictionary with encoded features
        """
        features = {}
        
        # Encode OHLCV data
        if 'ohlcv' in data and not data['ohlcv'].empty:
            features['ohlcv'] = self.encoders['ohlcv'].transform(data['ohlcv'])
        
        # Encode indicators
        if 'indicators' in data and data['indicators']:
            features['indicator'] = self.encoders['indicator'].transform(data['indicators'])
        
        # Encode sentiment
        if 'sentiment' in data and data['sentiment']:
            features['sentiment'] = self.encoders['sentiment'].transform(data['sentiment'])
        
        # We'll have orderbook and tick data when those collectors are running
        # For now, we'll proceed with what we have
        
        return features
    
    def _update_web_interface(self, data, features, active_features, signal):
        """Update web interface with current data"""
        # Get account summary
        account = self.executor.get_account_summary()
        
        # Create signals list
        session = self.db_manager.get_session()
        from database.models import Signal
        signals = session.query(Signal).order_by(Signal.timestamp.desc()).limit(10).all()
        signals_data = []
        
        for s in signals:
            signals_data.append({
                'timestamp': s.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'signal': s.signal_type,
                'price': s.price,
                'confidence': s.confidence
            })
        
        self.db_manager.close_session(session)
        
        # Calculate performance metrics
        performance = self._calculate_performance()
        
        # Prepare model stats with version and RFE info
        version_info = self.learner.get_version_info()
        rfe_summary = self.gating.get_rfe_summary() if hasattr(self.gating, 'get_rfe_summary') else {}
        
        model_stats = {
            'model_version': version_info.get('current_version', '1.0.0'),
            'updates_count': version_info.get('updates_count', 0),
            'feature_importance': {},
            'update_counts': {'model': self.learner.updates_counter},
            'market_regime': self._detect_market_regime(data.get('ohlcv', pd.DataFrame())),
            'rfe_info': {
                'enabled': self.gating.rfe_enabled if hasattr(self.gating, 'rfe_enabled') else False,
                'performed': rfe_summary.get('rfe_performed', False),
                'selected_features': rfe_summary.get('total_selected', 0),
                'target_features': rfe_summary.get('target_features', 15),
                'top_features': rfe_summary.get('top_features', [])[:5]  # Show top 5
            },
            'version_history': version_info.get('version_history', [])[-3:]  # Last 3 versions
        }
        
        # Extract feature importance from active features
        if active_features:
            for group, features in active_features.items():
                if isinstance(features, dict) and 'weight' in features:
                    model_stats['feature_importance'][group] = features['weight']
        
        # Create system stats
        import psutil
        system_stats = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'uptime': 0,  # We don't track this yet
            'errors': []  # We don't track this yet
        }
        
        # Update app.data_store
        app.data_store['bot_status'] = 'running' if self.is_running else 'stopped'
        app.data_store['last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        app.data_store['trading_data']['portfolio'] = account
        app.data_store['trading_data']['signals'] = signals_data
        app.data_store['trading_data']['performance'] = performance
        
        if 'ohlcv' in data:
            app.data_store['trading_data']['ohlcv'] = data['ohlcv']
        
        if 'indicators' in data and data['indicators']:
            app.data_store['trading_data']['indicators'] = data['indicators']
        
        app.data_store['system_stats'] = system_stats
        app.data_store['model_stats'] = model_stats
        app.data_store['active_features'] = active_features
        
        # Add learning statistics
        learning_summary = self.learner.get_learning_summary()
        app.data_store['learning_stats'] = learning_summary
    
    def _calculate_performance(self):
        """Calculate performance metrics"""
        session = self.db_manager.get_session()
        from database.models import Trade
        import pandas as pd
        
        try:
            # Get all trades
            trades = session.query(Trade).order_by(Trade.timestamp.asc()).all()
            
            if not trades:
                return {'daily': 0, 'weekly': 0, 'monthly': 0, 'all_time': 0}
            
            # Calculate PnL
            df = pd.DataFrame([{
                'timestamp': t.timestamp,
                'pnl': t.realized_pnl if t.realized_pnl else 0
            } for t in trades])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # Calculate performance metrics
            now = datetime.now()
            daily_cutoff = now - pd.Timedelta(days=1)
            weekly_cutoff = now - pd.Timedelta(days=7)
            monthly_cutoff = now - pd.Timedelta(days=30)
            
            daily_pnl = df[df.index >= daily_cutoff]['pnl'].sum()
            weekly_pnl = df[df.index >= weekly_cutoff]['pnl'].sum()
            monthly_pnl = df[df.index >= monthly_cutoff]['pnl'].sum()
            all_time_pnl = df['pnl'].sum()
            
            # Convert to percentage
            initial_balance = 100  # Assume we started with 100 USD
            daily_pct = (daily_pnl / initial_balance) * 100
            weekly_pct = (weekly_pnl / initial_balance) * 100
            monthly_pct = (monthly_pnl / initial_balance) * 100
            all_time_pct = (all_time_pnl / initial_balance) * 100
            
            return {
                'daily': daily_pct,
                'weekly': weekly_pct,
                'monthly': monthly_pct,
                'all_time': all_time_pct
            }
        
        except Exception as e:
            logger.error(f"Error calculating performance: {str(e)}")
            return {'daily': 0, 'weekly': 0, 'monthly': 0, 'all_time': 0}
        
        finally:
            self.db_manager.close_session(session)
    
    def _detect_market_regime(self, ohlcv):
        """Detect current market regime"""
        if ohlcv.empty:
            return 'neutral'
        
        try:
            # Simple detection based on recent price movement
            closes = ohlcv['close'].values
            if len(closes) < 20:
                return 'neutral'
            
            # Calculate returns
            returns = np.diff(closes) / closes[:-1]
            
            # Calculate volatility
            volatility = np.std(returns[-20:]) * 100
            
            # Calculate trend
            trend = (closes[-1] / closes[-20] - 1) * 100
            
            # Determine regime
            if volatility > 3:  # High volatility
                if trend > 5:
                    return 'bullish_volatile'
                elif trend < -5:
                    return 'bearish_volatile'
                else:
                    return 'choppy'
            else:  # Low volatility
                if trend > 3:
                    return 'bullish'
                elif trend < -3:
                    return 'bearish'
                else:
                    return 'neutral'
        
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return 'neutral'
    
    def _calculate_recent_success_rate(self):
        """
        Calculate recent trading success rate for periodic RFE triggering.
        
        Returns:
            float: Success rate (0.0-1.0) over last 20 trades
        """
        try:
            if not hasattr(self, 'recent_trades'):
                self.recent_trades = []
            
            # Get recent trade results from position manager or risk manager
            # This is a simplified implementation - in practice would track actual trade outcomes
            
            if len(self.recent_trades) < 5:
                return 0.5  # Default neutral success rate
            
            # Calculate success rate from last 20 trades
            recent = self.recent_trades[-20:]
            successful = sum(1 for trade in recent if trade.get('success', False))
            success_rate = successful / len(recent)
            
            # Update gating performance tracking
            if self.gating:
                self.gating.update_performance_tracking(success_rate)
            
            return success_rate
            
        except Exception as e:
            logger.error(f"Error calculating success rate: {str(e)}")
            return 0.5  # Return neutral rate on error
    
    def _calculate_rolling_performance_metrics(self):
        """
        Calculate rolling performance metrics for dashboard display (Phase 2).
        
        Returns:
            dict: Rolling metrics including win_rate_last_20, avg_rr_last_20, cumulative_pnl
        """
        try:
            if not hasattr(self, 'recent_trades'):
                self.recent_trades = []
            
            metrics = {
                'win_rate_last_20': 0.0,
                'avg_rr_last_20': 0.0,
                'cumulative_pnl': 0.0,
                'total_trades': len(self.recent_trades),
                'trades_last_20': 0
            }
            
            if len(self.recent_trades) == 0:
                return metrics
            
            # Get last 20 trades
            recent_trades = self.recent_trades[-20:]
            metrics['trades_last_20'] = len(recent_trades)
            
            # Calculate win rate
            winning_trades = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
            metrics['win_rate_last_20'] = (winning_trades / len(recent_trades)) * 100 if recent_trades else 0.0
            
            # Calculate average risk-reward ratio
            risk_rewards = []
            for trade in recent_trades:
                if trade.get('risk', 0) > 0 and trade.get('pnl', 0) != 0:
                    rr = abs(trade['pnl']) / trade['risk']
                    if trade['pnl'] > 0:  # Winning trade
                        risk_rewards.append(rr)
                    else:  # Losing trade
                        risk_rewards.append(-rr)
            
            metrics['avg_rr_last_20'] = np.mean(risk_rewards) if risk_rewards else 0.0
            
            # Calculate cumulative PnL
            metrics['cumulative_pnl'] = sum(trade.get('pnl', 0) for trade in self.recent_trades)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating rolling performance metrics: {str(e)}")
            return {
                'win_rate_last_20': 0.0,
                'avg_rr_last_20': 0.0,
                'cumulative_pnl': 0.0,
                'total_trades': 0,
                'trades_last_20': 0
            }
    
    def _log_hourly_performance(self):
        """
        Log performance metrics every hour (Phase 2 requirement).
        """
        try:
            if not hasattr(self, 'last_hourly_log'):
                self.last_hourly_log = time.time()
                return
            
            # Check if an hour has passed
            if time.time() - self.last_hourly_log < 3600:  # 1 hour in seconds
                return
            
            # Calculate and log metrics
            metrics = self._calculate_rolling_performance_metrics()
            
            logger.info("=" * 60)
            logger.info("📊 HOURLY PERFORMANCE SUMMARY")
            logger.info(f"Win Rate (Last 20): {metrics['win_rate_last_20']:.1f}%")
            logger.info(f"Avg Risk/Reward (Last 20): {metrics['avg_rr_last_20']:.2f}")
            logger.info(f"Cumulative PnL: ${metrics['cumulative_pnl']:.2f}")
            logger.info(f"Total Trades: {metrics['total_trades']}")
            logger.info("=" * 60)
            
            self.last_hourly_log = time.time()
            
        except Exception as e:
            logger.error(f"Error logging hourly performance: {str(e)}")


def _inject_config_defaults(config):
    """
    Inject default configuration values for training and indicators sections.
    
    Args:
        config: Configuration dictionary to modify in-place
    """
    # Training defaults
    if 'training' not in config:
        config['training'] = {}
    config['training'].setdefault('max_warmup_seconds', 120)
    
    # Indicators defaults  
    if 'indicators' not in config:
        config['indicators'] = {}
    config['indicators'].setdefault('profile', True)
    config['indicators'].setdefault('use_selected_only', False)
    config['indicators'].setdefault('min_active_features', 10)
    
    # RFE defaults
    if 'rfe' not in config:
        config['rfe'] = {}
    if 'periodic' not in config['rfe']:
        config['rfe']['periodic'] = {}
    config['rfe']['periodic'].setdefault('enabled', False)
    config['rfe']['periodic'].setdefault('interval_minutes', 30)
    
    if 'trigger' not in config['rfe']:
        config['rfe']['trigger'] = {}
    config['rfe']['trigger'].setdefault('performance_drop_pct', 15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        config = {
            'mysql': {
                'host': 'localhost',
                'user': 'root',
                'password': '',
                'database': 'trade'
            },
            'trading': {
                'symbols': ['BTCUSDT'],
                'timeframes': ['5m'],
                'exchange': 'coinex',
                'demo_mode': True,
                'initial_balance': 100,
                'risk_per_trade': 0.02,
                'max_open_trades': 1,
                'initial_stop_loss': 0.03,
                'take_profit_levels': [0.05, 0.1, 0.15],
                'max_open_risk': 0.06,
                'max_daily_loss': 0.05
            },
            'web': {
                'host': '0.0.0.0',
                'port': 5000
            },
            'model': {
                'hidden_dim': 128,
                'n_layers': 2,
                'n_heads': 4,
                'dropout': 0.1,
                'learning_rate': 1e-4,
                'update_interval': 60  # Reduced to 60 seconds for development testing
            }
        }
    
    # Inject config defaults for training and indicators sections
    _inject_config_defaults(config)
    
    # Create and start bot
    bot = TradingBot(config)
    
    try:
        bot.start()
        
        # Keep main thread alive
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("Stopping bot due to keyboard interrupt")
        bot.stop()
    
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        bot.stop()