import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import hashlib
from datetime import datetime
import tempfile
import os
from .locks import rfe_lock

logger = logging.getLogger("Gating")

# Optional scikit-learn import for RFE
try:
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. RFE feature selection will be disabled.")

class FeatureGatingModule(nn.Module):
    """
    Enhanced feature gating mechanism with RFE-style feature selection
    
    Features:
    - Pre-training RFE to find best 10-20 features from 100 available
    - Adaptively select important features based on market conditions
    - Weak features get minimum weight of 0.01 (effectively disabled but trackable)
    - Strong features maintain high impact
    - Context-aware gating for real-time adaptation
    """
    
    def __init__(self, feature_groups, hidden_dim=64, min_weight=0.01, adaptation_rate=0.1, 
                 rfe_enabled=True, rfe_n_features=15):
        """
        Initialize feature gating module with RFE capability
        
        Args:
            feature_groups: Dictionary with feature groups and their dimensions
                Example: {'ohlcv': 13, 'indicator': 100, 'sentiment': 1, 'orderbook': 42}
            hidden_dim: Hidden dimension for gating network
            min_weight: Minimum weight for weak features (default 0.01)
            adaptation_rate: Rate of adaptation for feature weights
            rfe_enabled: Whether to use RFE for feature selection
            rfe_n_features: Number of top features to select via RFE
        """
        super(FeatureGatingModule, self).__init__()
        self.feature_groups = feature_groups
        self.group_names = list(feature_groups.keys())
        self.min_weight = min_weight
        self.adaptation_rate = adaptation_rate
        self.rfe_enabled = rfe_enabled
        self.rfe_n_features = rfe_n_features
        
        # Calculate total input dimension
        self.total_dim = sum(feature_groups.values())
        
        # RFE results storage
        self.rfe_selected_features = {}
        self.rfe_feature_rankings = {}
        self.rfe_performed = False
        self.rfe_model = None
        
        # New unified RFE state attributes per requirements
        self.rfe_all_features = []  # List of cleaned feature names before selection
        self.selected_features = []  # List of selected feature names
        self.feature_categories = {}  # Dict: feature_name -> {selected, rank, category}
        self.rfe_counts = {'strong': 0, 'medium': 0, 'weak': 0}  # Category counts
        self.total_raw_features = 0  # Optional: raw count if differs from cleaned
        
        # Active feature masks for each group (Phase 2 requirement)
        self.active_feature_masks = {}  # Dict: group_name -> boolean mask of active features
        self.group_feature_indices = {}  # Dict: group_name -> original indices in the full feature matrix
        
        # Feature set versioning (Phase 2 requirement) 
        self.feature_set_version = 1  # Simple integer version increment
        self.last_rfe_time = None
        
        # Periodic RFE tracking (Phase 2 requirement)
        self.last_performance_check = time.time()
        self.recent_success_rates = []  # Rolling window for performance tracking
        
        # Gating adaptation control (Phase 2 requirement)
        self.freeze_adaptation = False  # When True, adaptation is disabled during warmup
        
        # Create gate networks for each feature group
        self.gate_networks = nn.ModuleDict()
        
        for group_name, dim in feature_groups.items():
            self.gate_networks[group_name] = nn.Sequential(
                nn.Linear(self.total_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, dim),
                nn.Sigmoid()
            )
        
        # Initialize last update time and performance tracking
        self.last_update_time = {}
        self.feature_performance = {}
        for group_name in feature_groups:
            self.last_update_time[group_name] = datetime.now()
            self.feature_performance[group_name] = {
                'success_count': 0,
                'total_count': 0,
                'avg_confidence': 0.0,
                'weight_history': [],
                'rfe_rank': 0,  # RFE ranking (0 = not ranked, 1 = best)
                'rfe_selected': False
            }
        
        # Store current gates for monitoring
        self.current_gates = {group: torch.ones(dim) for group, dim in feature_groups.items()}
        
        # Feature adaptation parameters
        self.update_frequency = 5  # seconds between updates
        
        logger.info(f"Initialized Enhanced FeatureGatingModule with {len(feature_groups)} feature groups")
        logger.info(f"Min weight for weak features: {min_weight}, Adaptation rate: {adaptation_rate}")
        logger.info(f"RFE enabled: {rfe_enabled}, Target features: {rfe_n_features}")
    
    def perform_rfe_selection(self, training_data, training_labels):
        """
        Perform RFE feature selection with proper DataFrame-based feature matrix construction
        
        Args:
            training_data: Training feature data as dict {group_name: DataFrame or features_array}
            training_labels: Training labels (0=BUY, 1=SELL, 2=HOLD)
            
        Returns:
            Dictionary with selected features and rankings
        """
        if not self.rfe_enabled or self.rfe_performed:
            logger.info("RFE already performed or disabled")
            return self.rfe_selected_features
        
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Disabling RFE feature selection.")
            self.rfe_enabled = False
            return {}
        
        logger.info("🔍 Starting RFE feature selection process...")
        
        try:
            # Log initial data info per group
            total_features_per_group = {}
            for group_name, group_data in training_data.items():
                if hasattr(group_data, 'shape'):
                    total_features_per_group[group_name] = group_data.shape[1] if group_data.ndim > 1 else 1
                elif hasattr(group_data, 'columns'):
                    total_features_per_group[group_name] = len(group_data.columns)
                else:
                    total_features_per_group[group_name] = 1
            
            logger.info(f"RFE input: Feature columns per group: {total_features_per_group}")
            total_combined_features = sum(total_features_per_group.values())
            logger.info(f"RFE input: Total combined features: {total_combined_features}")
            
            # Convert aligned DataFrames to numeric feature matrix
            feature_matrix = []
            feature_names = []
            samples_count = len(training_labels)
            
            for group_name, group_data in training_data.items():
                logger.debug(f"Processing group {group_name}: type={type(group_data)}")
                
                # Handle DataFrame format (from timestamp alignment)
                if hasattr(group_data, 'columns') and hasattr(group_data, 'index'):
                    # It's a DataFrame - select and convert numeric columns
                    numeric_cols = group_data.select_dtypes(include=[np.number]).columns.tolist()
                    
                    # For sentiment/orderbook columns, try to convert non-numeric to numeric
                    if group_name in ['sentiment', 'orderbook']:
                        for col in group_data.columns:
                            if col not in numeric_cols:
                                try:
                                    # Try to convert non-numeric columns to numeric
                                    converted_col = pd.to_numeric(group_data[col], errors='coerce')
                                    # Keep column if it has finite values and non-zero variance
                                    if not converted_col.isna().all() and converted_col.var() > 1e-10:
                                        group_data[col] = converted_col
                                        numeric_cols.append(col)
                                        logger.debug(f"Converted {group_name}.{col} to numeric")
                                except Exception as e:
                                    logger.debug(f"Could not convert {group_name}.{col} to numeric: {str(e)}")
                    
                    if len(numeric_cols) == 0:
                        logger.info(f"No valid numeric columns in group {group_name} (this is normal if group has no numeric data)")
                        continue
                    
                    numeric_data = group_data[numeric_cols]
                    
                    # Drop columns with all NaN or zero variance
                    valid_cols = []
                    for col in numeric_cols:
                        col_data = numeric_data[col]
                        # Fix pandas Series boolean comparison - use proper approach
                        try:
                            # Use .item() to convert scalar boolean to Python bool
                            has_valid_data = (~col_data.isna().all()).item() if hasattr((~col_data.isna().all()), "item") else bool(~col_data.isna().all())
                            has_variance = (col_data.var() > 1e-10).item() if hasattr((col_data.var() > 1e-10), "item") else bool(col_data.var() > 1e-10)
                            if has_valid_data and has_variance:
                                valid_cols.append(col)
                        except Exception as e:
                            logger.debug(f"Error processing column {col}: {e}")
                            continue
                    
                    if len(valid_cols) == 0:
                        logger.warning(f"No valid numeric columns in group {group_name} after filtering")
                        continue
                    
                    valid_data = numeric_data[valid_cols]
                    
                    # Fill NaN values
                    valid_data = valid_data.ffill().bfill().fillna(0)
                    
                    # Add each column as a feature
                    for col in valid_cols:
                        feature_matrix.append(valid_data[col].values)
                        feature_names.append(f"{group_name}.{col}")
                
                # Handle numpy array format (legacy)
                elif isinstance(group_data, np.ndarray):
                    if group_name == 'indicator' and group_data.ndim >= 2:
                        # For indicator group, add each feature individually
                        if group_data.ndim == 3:
                            # (samples, sequence, features) -> take mean over sequence
                            features_2d = np.mean(group_data, axis=1)
                        else:
                            features_2d = group_data
                            
                        # Add each indicator as separate feature
                        for i in range(features_2d.shape[1]):
                            col_data = features_2d[:, i]
                            # Check for valid data (not all NaN, has variance)
                            if not np.isnan(col_data).all() and np.var(col_data) > 1e-10:
                                feature_matrix.append(col_data)
                                feature_names.append(f"{group_name}.feature_{i}")
                    else:
                        # For other groups, treat as single feature or aggregate
                        if group_data.ndim == 1:
                            if not np.isnan(group_data).all() and np.var(group_data) > 1e-10:
                                feature_matrix.append(group_data)
                                feature_names.append(group_name)
                        elif group_data.ndim == 2:
                            # Take mean if multiple dimensions
                            agg_data = np.mean(group_data, axis=1)
                            if not np.isnan(agg_data).all() and np.var(agg_data) > 1e-10:
                                feature_matrix.append(agg_data)
                                feature_names.append(group_name)
                        else:
                            # Take mean over all dimensions except first
                            reshaped = group_data.reshape(group_data.shape[0], -1)
                            agg_data = np.mean(reshaped, axis=1)
                            if not np.isnan(agg_data).all() and np.var(agg_data) > 1e-10:
                                feature_matrix.append(agg_data)
                                feature_names.append(group_name)
                else:
                    logger.warning(f"Unsupported data type for group {group_name}: {type(group_data)}")
                    continue
            
            # Check if we have any features
            if not feature_matrix:
                logger.warning("RFE aborted: 0 numeric features after cleaning")
                return {}
            
            # Construct feature matrix X and labels y
            X = np.column_stack(feature_matrix)
            y = np.array(training_labels)
            
            # Remove samples with NaN/Inf after concatenation
            valid_samples = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y) | np.isinf(y))
            X_clean = X[valid_samples]
            y_clean = y[valid_samples]
            
            # Log prepared input
            logger.info(f"RFE input prepared: X shape=({X_clean.shape[0]}, {X_clean.shape[1]}), y shape=({y_clean.shape[0]})")
            
            if X_clean.shape[1] == 0:
                logger.warning("RFE aborted: 0 features after cleaning")
                return {}
            
            if len(X_clean) < 5:
                logger.warning(f"RFE aborted: only {len(X_clean)} samples after cleaning, need at least 5")
                return {}
            
            # Check label diversity to avoid single-class issues
            unique_labels = np.unique(y_clean)
            if len(unique_labels) < 2:
                logger.warning(f"Single-class labels detected: {unique_labels}. Need at least 2 classes for RFE.")
                logger.info("Attempting label regeneration with smaller thresholds...")
                
                # Try to regenerate labels with smaller thresholds if we have price data
                if 'ohlcv' in training_data and hasattr(training_data['ohlcv'], 'columns'):
                    ohlcv_df = training_data['ohlcv']
                    if 'close' in ohlcv_df.columns:
                        y_clean = self._generate_diverse_labels_from_prices(ohlcv_df['close'].values[valid_samples])
                        unique_labels = np.unique(y_clean)
                        logger.info(f"Regenerated labels: BUY={np.sum(y_clean==0)}, SELL={np.sum(y_clean==1)}, HOLD={np.sum(y_clean==2)}")
                
                if len(unique_labels) < 2:
                    logger.warning("Still single-class after regeneration → using fallback variance ranking")
                    return self._perform_fallback_selection(training_data, y_clean, len(X_clean))
            
            if len(X_clean) < 30:
                logger.info(f"RFE samples={len(X_clean)} (<30 threshold) → using fallback ranking")
                # For limited data, use fallback method
                return self._perform_fallback_selection(training_data, y_clean, len(X_clean))
            
            if len(X_clean) < 50:
                logger.info(f"Limited samples ({len(X_clean)}) - using simplified Random Forest")
                # Use simpler model for limited data
                rf_estimator = RandomForestClassifier(
                    n_estimators=10,
                    max_depth=3,
                    random_state=42,
                    n_jobs=1
                )
            else:
                # Use full model for sufficient data
                rf_estimator = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=1
                )
            
            logger.info(f"Starting RFE with {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
            
            # Perform RFE
            n_features_to_select = min(self.rfe_n_features, X_clean.shape[1])
            logger.info(f"Selecting top {n_features_to_select} features via RFE...")
            
            rfe = RFE(
                estimator=rf_estimator,
                n_features_to_select=n_features_to_select,
                step=1
            )
            
            rfe.fit(X_clean, y_clean)
            
            # Store results
            selected_indices = np.where(rfe.support_)[0]
            rankings = rfe.ranking_
            
            self.rfe_selected_features = {}
            self.rfe_feature_rankings = {}
            
            logger.info("🎯 RFE Feature Selection Results:")
            logger.info(f"Selected {len(selected_indices)} out of {len(feature_names)} features:")
            
            for i, (feature_name, rank) in enumerate(zip(feature_names, rankings)):
                self.rfe_feature_rankings[feature_name] = int(rank)  # Convert to native int
                is_selected = i in selected_indices
                
                if is_selected:
                    self.rfe_selected_features[feature_name] = {
                        'rank': int(rank),  # Convert to native int
                        'importance': float(rf_estimator.feature_importances_[i]) if hasattr(rf_estimator, 'feature_importances_') else 0.0,
                        'selected': True
                    }
                    logger.info(f"  ✅ {feature_name} (rank: {rank})")
                else:
                    # Still track unselected features but mark them
                    self.rfe_selected_features[feature_name] = {
                        'rank': int(rank),  # Convert to native int
                        'importance': 0.01,  # Low importance for unselected
                        'selected': False
                    }
            
            # Update feature performance tracking
            for group_name in self.feature_performance:
                if group_name == 'indicator':
                    # For indicators, check individual features
                    selected_indicators = [name for name in self.rfe_selected_features.keys() 
                                         if name.startswith('indicator.') and self.rfe_selected_features[name]['selected']]
                    self.feature_performance[group_name]['rfe_selected'] = len(selected_indicators) > 0
                    self.feature_performance[group_name]['rfe_rank'] = min([
                        self.rfe_selected_features[name]['rank'] for name in selected_indicators
                    ]) if selected_indicators else 999
                else:
                    # For other groups
                    if group_name in self.rfe_selected_features:
                        self.feature_performance[group_name]['rfe_selected'] = self.rfe_selected_features[group_name]['selected']
                        self.feature_performance[group_name]['rfe_rank'] = self.rfe_selected_features[group_name]['rank']
            
            self.rfe_performed = True
            self.last_rfe_time = time.time()
            self.feature_set_version += 1
            self.rfe_model = rfe
            
            # Populate new unified RFE state attributes
            self.rfe_all_features = feature_names.copy()  # Cleaned feature names before selection
            self.selected_features = [name for name, info in self.rfe_selected_features.items() if info['selected']]
            self._build_feature_categories()
            
            # Persist RFE results to JSON file
            self._save_rfe_results_to_file()
            
            # Single concise summary log (replaces old duplicate logging)
            logger.info(f"🚀 RFE completed! Selected {len(self.selected_features)} features")
            
            return self.rfe_selected_features
            
        except Exception as e:
            logger.error(f"Error during RFE: {str(e)}")
            import traceback
            logger.error(f"RFE traceback: {traceback.format_exc()}")
            return {}
    
    def _generate_diverse_labels_from_prices(self, close_prices):
        """
        Generate diverse labels from close prices using lookahead and smaller thresholds
        
        Args:
            close_prices: Array of close prices
            
        Returns:
            Array of diverse labels (0=BUY, 1=SELL, 2=HOLD)
        """
        try:
            # Use lookahead approach with smaller thresholds
            look_ahead = 3  # 3 candles forward
            up_threshold = 0.002  # +0.2%
            down_threshold = -0.002  # -0.2%
            
            labels = []
            for i in range(len(close_prices)):
                if i + look_ahead >= len(close_prices):
                    # Not enough future data - label as HOLD
                    labels.append(2)
                    continue
                
                current_price = close_prices[i]
                future_price = close_prices[i + look_ahead]
                
                if current_price == 0:  # Avoid division by zero
                    labels.append(2)
                    continue
                
                price_change = (future_price - current_price) / current_price
                
                if price_change > up_threshold:
                    labels.append(0)  # BUY
                elif price_change < down_threshold:
                    labels.append(1)  # SELL
                else:
                    labels.append(2)  # HOLD
            
            return np.array(labels)
            
        except Exception as e:
            logger.error(f"Error generating diverse labels: {str(e)}")
            return np.full(len(close_prices), 2)  # All HOLD as fallback
    
    def _build_feature_categories(self):
        """
        Build feature categories based on rank thresholds and update counts.
        
        Categories:
        - strong: rank <= target_features
        - medium: rank <= 2*target_features  
        - weak: rank > 2*target_features
        """
        self.feature_categories = {}
        strong_count = 0
        medium_count = 0
        weak_count = 0
        
        for feature_name, info in self.rfe_selected_features.items():
            rank = info.get('rank', 999)
            selected = info.get('selected', False)
            
            # Assign category based on rank thresholds
            if rank <= self.rfe_n_features:
                category = 'strong'
                strong_count += 1
            elif rank <= 2 * self.rfe_n_features:
                category = 'medium' 
                medium_count += 1
            else:
                category = 'weak'
                weak_count += 1
            
            self.feature_categories[feature_name] = {
                'selected': selected,
                'rank': rank,
                'category': category
            }
        
        # Update counts
        self.rfe_counts = {
            'strong': strong_count,
            'medium': medium_count, 
            'weak': weak_count
        }
    
    
    def _perform_fallback_selection(self, training_data, labels, n_samples):
        """
        Fallback feature selection using variance or correlation when RFE fails
        
        Args:
            training_data: Aligned training data dict
            labels: Target labels
            n_samples: Number of samples
            
        Returns:
            Dictionary with selected features
        """
        try:
            logger.info(f"Using fallback selection with {n_samples} samples")
            
            # Choose method based on label diversity
            unique_labels = np.unique(labels)
            if len(unique_labels) >= 2:
                method = 'correlation'
                logger.info("Fallback method: correlation (sufficient label diversity)")
            else:
                method = 'variance'
                logger.info("Fallback method: variance (insufficient label diversity)")
            
            feature_matrix = []
            feature_names = []
            
            # Rebuild feature matrix from aligned data - use same logic as main RFE
            for group_name, group_data in training_data.items():
                logger.debug(f"Fallback processing group {group_name}: type={type(group_data)}")
                
                # Handle DataFrame format (from timestamp alignment)
                if hasattr(group_data, 'columns') and hasattr(group_data, 'index'):
                    # It's a DataFrame - select and convert numeric columns
                    numeric_cols = group_data.select_dtypes(include=[np.number]).columns.tolist()
                    
                    # For sentiment/orderbook columns, try to convert non-numeric to numeric
                    if group_name in ['sentiment', 'orderbook']:
                        for col in group_data.columns:
                            if col not in numeric_cols:
                                try:
                                    # Try to convert non-numeric columns to numeric
                                    converted_col = pd.to_numeric(group_data[col], errors='coerce')
                                    # Keep column if it has finite values and non-zero variance
                                    if not converted_col.isna().all() and converted_col.var() > 1e-10:
                                        group_data[col] = converted_col
                                        numeric_cols.append(col)
                                        logger.debug(f"Converted {group_name}.{col} to numeric")
                                except Exception as e:
                                    logger.debug(f"Could not convert {group_name}.{col} to numeric: {str(e)}")
                    
                    if len(numeric_cols) == 0:
                        continue
                    
                    numeric_data = group_data[numeric_cols]
                    
                    # Drop columns with all NaN or zero variance
                    valid_cols = []
                    for col in numeric_cols:
                        col_data = numeric_data[col]
                        # Fix pandas Series boolean comparison - use proper approach
                        try:
                            # Use .item() to convert scalar boolean to Python bool
                            has_valid_data = (~col_data.isna().all()).item() if hasattr((~col_data.isna().all()), "item") else bool(~col_data.isna().all())
                            has_variance = (col_data.var() > 1e-10).item() if hasattr((col_data.var() > 1e-10), "item") else bool(col_data.var() > 1e-10)
                            if has_valid_data and has_variance:
                                valid_cols.append(col)
                        except Exception as e:
                            logger.debug(f"Error processing column {col}: {e}")
                            continue
                    
                    if len(valid_cols) == 0:
                        continue
                    
                    valid_data = numeric_data[valid_cols]
                    valid_data = valid_data.ffill().bfill().fillna(0)
                    
                    # Add each column as a feature
                    for col in valid_cols:
                        feature_matrix.append(valid_data[col].values)
                        feature_names.append(f"{group_name}.{col}")
                
                # Handle numpy array format (legacy)
                elif isinstance(group_data, np.ndarray):
                    if group_name == 'indicator' and group_data.ndim >= 2:
                        if group_data.ndim == 3:
                            features_2d = np.mean(group_data, axis=1)
                        else:
                            features_2d = group_data
                        
                        # Add each indicator as separate feature
                        for i in range(features_2d.shape[1]):
                            col_data = features_2d[:, i]
                            if not np.isnan(col_data).all() and np.var(col_data) > 1e-10:
                                feature_matrix.append(col_data)
                                feature_names.append(f"{group_name}.feature_{i}")
                    else:
                        # For other groups, treat as single feature or aggregate
                        if group_data.ndim == 1:
                            if not np.isnan(group_data).all() and np.var(group_data) > 1e-10:
                                feature_matrix.append(group_data)
                                feature_names.append(group_name)
                        elif group_data.ndim == 2:
                            agg_data = np.mean(group_data, axis=1)
                            if not np.isnan(agg_data).all() and np.var(agg_data) > 1e-10:
                                feature_matrix.append(agg_data)
                                feature_names.append(group_name)
                        else:
                            reshaped = group_data.reshape(group_data.shape[0], -1)
                            agg_data = np.mean(reshaped, axis=1)
                            if not np.isnan(agg_data).all() and np.var(agg_data) > 1e-10:
                                feature_matrix.append(agg_data)
                                feature_names.append(group_name)
            
            if not feature_matrix:
                logger.warning("No features for fallback selection")
                return {}
            
            X = np.column_stack(feature_matrix)
            
            # Remove NaN/Inf
            valid_samples = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
            X_clean = X[valid_samples]
            
            if len(X_clean) == 0:
                logger.warning("No valid samples after cleaning")
                return {}
            
            scores = []
            
            if method == 'variance':
                # Use variance for feature ranking
                for i in range(X_clean.shape[1]):
                    col_var = np.var(X_clean[:, i])
                    scores.append(col_var if not np.isnan(col_var) else 0.0)
                logger.info(f"Variance scores: min={min(scores):.6f}, max={max(scores):.6f}")
                
            else:  # correlation
                # Use correlation with labels
                y_clean = labels[valid_samples] if len(labels) == len(valid_samples) else labels[:len(X_clean)]
                
                for i in range(X_clean.shape[1]):
                    try:
                        # Calculate correlation, handle division by zero and NaN
                        col_data = X_clean[:, i]
                        if np.std(col_data) == 0:  # Constant feature
                            corr = 0.0
                        else:
                            corr_coef = np.corrcoef(col_data, y_clean)[0, 1]
                            corr = abs(corr_coef) if not np.isnan(corr_coef) else 0.0
                        scores.append(corr)
                    except:
                        scores.append(0.0)
                logger.info(f"Correlation scores: min={min(scores):.6f}, max={max(scores):.6f}")
            
            # Select top features
            n_features_to_select = min(self.rfe_n_features, len(scores))
            top_indices = np.argsort(scores)[-n_features_to_select:]
            
            # Create results
            self.rfe_selected_features = {}
            self.rfe_feature_rankings = {}
            
            for i, feature_name in enumerate(feature_names):
                is_selected = i in top_indices
                score = scores[i]
                
                self.rfe_selected_features[feature_name] = {
                    'rank': 1 if is_selected else 2,
                    'importance': score if is_selected else 0.01,
                    'selected': is_selected
                }
                self.rfe_feature_rankings[feature_name] = 1 if is_selected else 2
            
            # Update feature performance tracking
            for group_name in self.feature_performance:
                if group_name == 'indicator':
                    selected_indicators = [name for name in self.rfe_selected_features.keys() 
                                         if name.startswith('indicator.') and self.rfe_selected_features[name]['selected']]
                    self.feature_performance[group_name]['rfe_selected'] = len(selected_indicators) > 0
                    self.feature_performance[group_name]['rfe_rank'] = min([
                        self.rfe_selected_features[name]['rank'] for name in selected_indicators
                    ]) if selected_indicators else 999
                else:
                    if group_name in self.rfe_selected_features:
                        self.feature_performance[group_name]['rfe_selected'] = self.rfe_selected_features[group_name]['selected']
                        self.feature_performance[group_name]['rfe_rank'] = self.rfe_selected_features[group_name]['rank']
            
            self.rfe_performed = True
            self.last_rfe_time = time.time()
            self.feature_set_version += 1
            
            # Save results to file
            self._save_rfe_results_to_file()
            
            logger.info(f"Fallback selection completed using {method}: selected {len(top_indices)} features")
            
            return self.rfe_selected_features
            
        except Exception as e:
            logger.error(f"Error in fallback selection: {str(e)}")
            return {}
    
    def _perform_correlation_based_selection(self, X, y, feature_names):
        """
        Simplified feature selection for very limited data using correlation
        """
        try:
            from scipy.stats import pearsonr
            
            correlations = []
            for i in range(X.shape[1]):
                try:
                    # Calculate correlation with target
                    corr, _ = pearsonr(X[:, i], y)
                    correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
                except:
                    correlations.append(0.0)
            
            # Select top features based on correlation
            n_features_to_select = min(self.rfe_n_features, len(correlations))
            top_indices = np.argsort(correlations)[-n_features_to_select:]
            
            self.rfe_selected_features = {}
            self.rfe_feature_rankings = {}
            
            for i, feature_name in enumerate(feature_names):
                is_selected = i in top_indices
                correlation_score = correlations[i]
                
                self.rfe_selected_features[feature_name] = {
                    'rank': 1 if is_selected else 2,
                    'importance': correlation_score if is_selected else 0.01,
                    'selected': is_selected
                }
                self.rfe_feature_rankings[feature_name] = 1 if is_selected else 2
            
            logger.info(f"RFE fallback selected {len(top_indices)} features (method=correlation)")
            logger.info(f"RFE fallback correlation scores: min={min(correlations):.3f}, max={max(correlations):.3f}")
            
            # Update feature performance tracking
            for group_name in self.feature_performance:
                if group_name == 'indicator':
                    # For indicators, check individual features
                    selected_indicators = [name for name in self.rfe_selected_features.keys() 
                                         if name.startswith('indicator.') and self.rfe_selected_features[name]['selected']]
                    self.feature_performance[group_name]['rfe_selected'] = len(selected_indicators) > 0
                    self.feature_performance[group_name]['rfe_rank'] = min([
                        self.rfe_selected_features[name]['rank'] for name in selected_indicators
                    ]) if selected_indicators else 999
                else:
                    # For other groups
                    if group_name in self.rfe_selected_features:
                        self.feature_performance[group_name]['rfe_selected'] = self.rfe_selected_features[group_name]['selected']
                        self.feature_performance[group_name]['rfe_rank'] = self.rfe_selected_features[group_name]['rank']
            
            self.rfe_performed = True
            self.last_rfe_time = time.time() 
            self.feature_set_version += 1
            
            # Persist RFE results to JSON file  
            self._save_rfe_results_to_file()
            
            return self.rfe_selected_features
            
        except Exception as e:
            logger.error(f"Error in correlation-based selection: {str(e)}")
            return {}
    
    def get_rfe_weights(self):
        """
        Get feature weights based on RFE results with enhanced strong/medium/weak mapping
        
        Returns:
            Dictionary with weights for each feature group
        """
        if not self.rfe_performed or not self.rfe_selected_features:
            return {}
        
        weights = {}
        
        # Collect all selected features and their rankings for proper weight assignment
        all_selected = [(name, info) for name, info in self.rfe_selected_features.items() if info['selected']]
        all_selected.sort(key=lambda x: x[1]['importance'], reverse=True)  # Sort by importance
        
        # Categorize features: strong (top selected), medium (next 2x selected), weak (rest)
        n_selected = len(all_selected)
        strong_count = min(self.rfe_n_features, n_selected)  # Top selected features
        medium_count = min(strong_count * 2, n_selected - strong_count)  # Next 2x features
        
        strong_features = set(item[0] for item in all_selected[:strong_count])
        medium_features = set(item[0] for item in all_selected[strong_count:strong_count + medium_count])
        
        logger.info(f"Weight mapping: strong={len(strong_features)}, medium={len(medium_features)}, weak={len(self.rfe_selected_features) - len(strong_features) - len(medium_features)}")
        
        for group_name, group_dim in self.feature_groups.items():
            if group_name == 'indicator':
                # For indicators, create weight vector based on individual selections
                group_weights = torch.full((group_dim,), self.min_weight)
                
                indicator_names = self._get_indicator_names()
                
                # Count strong, medium, weak indicators for this group
                strong_indicators = []
                medium_indicators = []
                
                for i, ind_name in enumerate(indicator_names[:group_dim]):
                    feat_key = f"indicator.{ind_name}"
                    
                    if feat_key in strong_features:
                        strong_indicators.append(i)
                    elif feat_key in medium_features:
                        medium_indicators.append(i)
                
                # Apply weights with proper distribution
                if strong_indicators:
                    strong_weights = np.linspace(0.7, 0.9, len(strong_indicators))
                    for idx, i in enumerate(strong_indicators):
                        group_weights[i] = strong_weights[idx]
                
                if medium_indicators:
                    medium_weights = np.linspace(0.3, 0.7, len(medium_indicators))
                    for idx, i in enumerate(medium_indicators):
                        group_weights[i] = medium_weights[idx]
                
                weights[group_name] = group_weights
                
            else:
                # For other groups, assign based on category
                if group_name in strong_features:
                    weight_val = 0.8  # Strong weight
                elif group_name in medium_features:
                    weight_val = 0.5  # Medium weight
                else:
                    weight_val = self.min_weight  # Weak weight
                
                weights[group_name] = torch.full((group_dim,), weight_val)
        
        return weights
    
    def apply_rfe_weights(self):
        """
        Apply RFE weights to the gating module with single static mapping.
        
        Returns:
            bool: True if RFE weights were successfully applied, False otherwise
        """
        try:
            if not self.rfe_performed:
                logger.warning("RFE not performed, cannot apply weights")
                return False
            
            rfe_weights = self.get_rfe_weights()
            if not rfe_weights:
                logger.warning("No RFE weights available to apply")
                return False
            
            # Single mapping of categories to static weights
            weight_mapping = {
                'strong': 0.85,
                'medium': 0.55, 
                'weak': self.min_weight
            }
            
            # Single concise summary log
            logger.info(f"RFE weight mapping applied once: strong={weight_mapping['strong']:.2f} medium={weight_mapping['medium']:.2f} weak={weight_mapping['weak']:.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying RFE weights: {str(e)}")
            return False
    
    def _log_rfe_weight_application(self, rfe_weights):
        """Log the application of RFE weights in strong/medium/weak categories"""
        # Remove duplicate logging - this will be handled by main.py with consistent feature-based counting
        # to avoid the inconsistencies mentioned in the problem statement
        pass
    
    def _save_rfe_results_to_file(self):
        """Save RFE results to external JSON file with atomic write and thread safety"""
        try:
            import json
            
            # Use lock for thread safety
            with rfe_lock:
                # Determine method used
                method = 'variance' if not hasattr(self, 'rfe_model') else 'RandomForestRFE'
                
                # Create ranked features list
                ranked_features = []
                for name, info in self.rfe_selected_features.items():
                    ranked_features.append({
                        'name': name,
                        'rank': int(info['rank']),  # Convert to native int
                        'importance': float(info['importance']),  # Convert to native float
                        'selected': bool(info['selected'])  # Convert to native bool
                    })
                
                # Sort by rank (lower is better)
                ranked_features.sort(key=lambda x: x['rank'])
                
                # Selected features only
                selected_features = [f['name'] for f in ranked_features if f['selected']]
                
                # Weight mapping summary
                weights_mapping = {
                    'strong': len([f for f in ranked_features if f['selected'] and f['importance'] >= 0.7]),
                    'medium': len([f for f in ranked_features if f['selected'] and 0.3 <= f['importance'] < 0.7]),
                    'weak': len([f for f in ranked_features if not f['selected'] or f['importance'] < 0.3])
                }
                
                rfe_data = {
                    'last_run': datetime.now().isoformat(),
                    'timestamp': datetime.now().isoformat(),  # Keep for backward compatibility
                    'method': method,
                    'ranked_features': ranked_features,
                    'selected': selected_features,
                    'selected_features': {k: v for k, v in self.rfe_selected_features.items() if v['selected']},  # For backward compatibility
                    'weights_mapping': weights_mapping,
                    'n_features_selected': len(selected_features),
                    'total_features': len(self.rfe_selected_features),
                    'rfe_n_features_target': self.rfe_n_features,
                    'performance_summary': {
                        'selection_criteria': 'Top features based on predictive importance or variance',
                        'weak_feature_handling': f'Assigned minimum weight of {self.min_weight}',
                        'strong_feature_boost': 'High-impact features receive proportional weights 0.7-0.9'
                    }
                }
                
                # Atomic write using tempfile and os.replace
                rfe_file = 'rfe_results.json'
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
                    json.dump(rfe_data, tmp_file, indent=2, ensure_ascii=False)
                    tmp_filename = tmp_file.name
                
                # Atomic replacement
                os.replace(tmp_filename, rfe_file)
                logger.info(f"RFE results saved atomically to {rfe_file}")
            
        except Exception as e:
            logger.error(f"Failed to save RFE results: {str(e)}")
            # Clean up temp file if it exists
            try:
                if 'tmp_filename' in locals():
                    os.unlink(tmp_filename)
            except:
                pass
    
    def get_rfe_summary(self):
        """
        Get RFE summary for API and logging with consistent payload.
        
        Returns:
            Dictionary with RFE summary information including:
            - rfe_performed, total_available, total_cleaned, total_selected
            - strong, medium, weak counts
            - target_features, top_features (limited to 10)
        """
        if not self.rfe_performed:
            return {
                'rfe_performed': False,
                'total_available': 0,
                'total_cleaned': 0,
                'total_selected': 0,
                'strong': 0,
                'medium': 0, 
                'weak': 0,
                'target_features': self.rfe_n_features,
                'top_features': []
            }
        
        # Use unified state attributes
        total_available = len(self.rfe_all_features)
        total_cleaned = total_available  # Same unless tracking raw pre-clean length
        total_selected = len(self.selected_features)
        
        # Top features by rank (lower is better), limited to 10
        sorted_features = sorted(
            [(name, info) for name, info in self.rfe_selected_features.items() if info['selected']],
            key=lambda x: x[1]['rank']
        )
        top_features = [(name, {'rank': info['rank'], 'importance': info['importance']}) 
                       for name, info in sorted_features[:10]]
        
        return {
            'rfe_performed': True,
            'total_available': total_available,
            'total_cleaned': total_cleaned,
            'total_selected': total_selected,
            'strong': self.rfe_counts['strong'],
            'medium': self.rfe_counts['medium'],
            'weak': self.rfe_counts['weak'],
            'target_features': self.rfe_n_features,
            'top_features': top_features
        }
    
    def forward(self, features_dict):
        """
        Apply enhanced gating to features with RFE-based minimum weight enforcement
        
        Args:
            features_dict: Dictionary with feature tensors by group
                Example: {'ohlcv': tensor(...), 'indicator': tensor(...), ...}
        
        Returns:
            Dictionary with gated feature tensors
        """
        if not features_dict:
            logger.warning("No features provided to gating module")
            return {}
        
        # Get RFE weights if available
        rfe_weights = self.get_rfe_weights() if self.rfe_performed else {}
        
        # Concatenate all features for context - fixed tensor shape handling
        all_features = []
        
        for group in self.group_names:
            if group in features_dict:
                # Get the raw feature tensor
                raw_features = features_dict[group]
                
                # Handle different tensor shapes properly
                if len(raw_features.shape) == 1:
                    # 1D tensor: reshape to (1, features)
                    flat_features = raw_features.unsqueeze(0)
                elif len(raw_features.shape) == 2:
                    # 2D tensor: check if it's (batch, features) or (seq, features)
                    if raw_features.size(0) == 1:
                        # Already (1, features)
                        flat_features = raw_features
                    else:
                        # (seq, features) - take mean over sequence
                        flat_features = torch.mean(raw_features, dim=0, keepdim=True)
                elif len(raw_features.shape) == 3:
                    # 3D tensor: (batch, seq, features) - take mean over sequence dimension
                    flat_features = torch.mean(raw_features, dim=1)
                    if flat_features.size(0) > 1:
                        # Multiple batches - take mean over batch too
                        flat_features = torch.mean(flat_features, dim=0, keepdim=True)
                else:
                    # Higher dimensional - flatten completely then reshape
                    total_elements = raw_features.numel()
                    expected_dim = self.feature_groups[group]
                    # Take only the expected number of features
                    flat_features = raw_features.view(-1)[:expected_dim].unsqueeze(0)
                
                # Ensure the feature dimension matches expected
                expected_dim = self.feature_groups[group]
                if flat_features.size(1) != expected_dim:
                    if flat_features.size(1) > expected_dim:
                        # Truncate extra features
                        flat_features = flat_features[:, :expected_dim]
                    else:
                        # Pad with zeros if needed
                        padding = torch.zeros(1, expected_dim - flat_features.size(1))
                        flat_features = torch.cat([flat_features, padding], dim=1)
                
                all_features.append(flat_features)
            else:
                # If group is missing, use zeros
                dim = self.feature_groups[group]
                all_features.append(torch.zeros((1, dim)))
        
        # Concatenate along feature dimension
        context = torch.cat(all_features, dim=1)
        
        # Apply gates to each group with enhanced RFE-based logic
        gated_features = {}
        
        # Log RFE weight application if performed
        if self.rfe_performed and rfe_weights:
            self._log_rfe_weight_application(rfe_weights)
        
        for group in self.group_names:
            if group in features_dict:
                # Calculate raw gates for this group
                raw_gates = self.gate_networks[group](context)
                
                # Apply RFE-based weighting if available
                if group in rfe_weights:
                    # Use RFE weights as base, then apply learned gating on top
                    rfe_base_weights = rfe_weights[group]
                    if len(rfe_base_weights.shape) == 1:
                        rfe_base_weights = rfe_base_weights.unsqueeze(0)
                    
                    # Combine RFE weights with learned gates
                    # RFE provides the base (0.01 for weak, 0.7+ for strong)
                    # Learned gates provide fine-tuning within those bounds
                    gates = rfe_base_weights * raw_gates
                else:
                    # Fallback to minimum weight constraint
                    gates = torch.clamp(raw_gates, min=self.min_weight, max=1.0)
                
                # Ensure gates have right shape for the feature group
                expected_dim = self.feature_groups[group]
                if gates.size(-1) != expected_dim:
                    if gates.size(-1) > expected_dim:
                        gates = gates[:, :expected_dim]
                    else:
                        # Pad with min_weight if needed
                        padding = torch.full((gates.size(0), expected_dim - gates.size(-1)), self.min_weight)
                        gates = torch.cat([gates, padding], dim=1)
                
                # Get original features and handle different shapes
                original_features = features_dict[group]
                
                # Apply gates based on original feature shape
                if len(original_features.shape) == 1:
                    # 1D features: gates should be 1D too
                    gates_to_apply = gates.squeeze(0)[:original_features.size(0)]
                    gated_features[group] = original_features * gates_to_apply
                    
                elif len(original_features.shape) == 2:
                    # 2D features: apply gates to last dimension
                    if original_features.size(0) == 1:
                        # (1, features) - apply gates directly
                        gates_to_apply = gates
                        if gates_to_apply.size(1) != original_features.size(1):
                            gates_to_apply = gates_to_apply[:, :original_features.size(1)]
                        gated_features[group] = original_features * gates_to_apply
                    else:
                        # (seq, features) - broadcast gates across sequence
                        gates_to_apply = gates.expand(original_features.size(0), -1)
                        if gates_to_apply.size(1) != original_features.size(1):
                            gates_to_apply = gates_to_apply[:, :original_features.size(1)]
                        gated_features[group] = original_features * gates_to_apply
                        
                elif len(original_features.shape) == 3:
                    # 3D features: (batch, seq, features) - broadcast gates
                    gates_to_apply = gates.unsqueeze(1).expand(-1, original_features.size(1), -1)
                    if gates_to_apply.size(-1) != original_features.size(-1):
                        gates_to_apply = gates_to_apply[:, :, :original_features.size(-1)]
                    gated_features[group] = original_features * gates_to_apply
                    
                else:
                    # Higher dimensional - apply to last dimension
                    gate_shape = [1] * (len(original_features.shape) - 1) + [gates.size(-1)]
                    gates_to_apply = gates.view(gate_shape).expand_as(original_features)
                    gated_features[group] = original_features * gates_to_apply
                
                # Store current gates for monitoring (use squeezed version)
                self.current_gates[group] = gates.detach().squeeze()
                
                # Update last update time
                self.last_update_time[group] = datetime.now()
                
                # Enhanced logging for RFE results
                avg_weight = float(torch.mean(gates.detach()))
                if group in rfe_weights:
                    rfe_selected_count = torch.sum(rfe_weights[group] > self.min_weight + 0.01).item()
                    if rfe_selected_count > 0:
                        logger.debug(f"Feature group {group}: {rfe_selected_count} RFE-selected features active (avg weight: {avg_weight:.3f})")
                    else:
                        logger.debug(f"Feature group {group}: No RFE-selected features (avg weight: {avg_weight:.3f})")
                elif avg_weight <= self.min_weight + 0.01:
                    logger.debug(f"Feature group {group} has low weight ({avg_weight:.3f}) - effectively disabled")
                elif avg_weight >= 0.7:
                    logger.debug(f"Feature group {group} has high weight ({avg_weight:.3f}) - strongly active")
                
            else:
                # Pass group with zeros if missing - maintain original expected shape
                expected_dim = self.feature_groups[group]
                if group == 'ohlcv':
                    gated_features[group] = torch.zeros((1, 20, expected_dim))
                elif group == 'indicator':
                    gated_features[group] = torch.zeros((1, 20, expected_dim))
                elif group == 'tick_data':
                    gated_features[group] = torch.zeros((1, 100, expected_dim))
                else:
                    gated_features[group] = torch.zeros((1, expected_dim))
        
        return gated_features
    
    def update_feature_performance(self, group_name, success, confidence):
        """
        Update feature performance tracking for adaptation
        
        Args:
            group_name: Name of feature group
            success: Boolean indicating if the feature contributed to successful prediction
            confidence: Confidence score of the prediction
        """
        if group_name in self.feature_performance:
            perf = self.feature_performance[group_name]
            perf['total_count'] += 1
            if success:
                perf['success_count'] += 1
            
            # Update average confidence with exponential moving average
            perf['avg_confidence'] = 0.9 * perf['avg_confidence'] + 0.1 * confidence
            
            # Store weight history for analysis
            if group_name in self.current_gates:
                avg_weight = float(torch.mean(self.current_gates[group_name]))
                perf['weight_history'].append(avg_weight)
                
                # Keep only last 100 weights
                if len(perf['weight_history']) > 100:
                    perf['weight_history'].pop(0)
    
    def get_feature_performance_summary(self):
        """
        Get performance summary for all features
        
        Returns:
            Dictionary with performance metrics
        """
        summary = {}
        
        for group_name, perf in self.feature_performance.items():
            if perf['total_count'] > 0:
                success_rate = perf['success_count'] / perf['total_count']
                
                summary[group_name] = {
                    'success_rate': success_rate,
                    'avg_confidence': perf['avg_confidence'],
                    'total_predictions': perf['total_count'],
                    'current_weight': float(torch.mean(self.current_gates[group_name])) if group_name in self.current_gates else 0.0,
                    'weight_trend': np.mean(perf['weight_history'][-10:]) if len(perf['weight_history']) >= 10 else 0.0
                }
        
        return summary
    
    def get_active_features(self, threshold=0.3):
        """
        Get currently active features with enhanced RFE reporting
        Shows all features including those with 0.01 weight (effectively disabled)
        
        Args:
            threshold: Threshold for considering a feature active
            
        Returns:
            Dictionary with active features and their weights, including RFE status
        """
        active_features = {}
        
        for group, gates in self.current_gates.items():
            # Convert to numpy
            if isinstance(gates, torch.Tensor):
                gates_np = gates.cpu().numpy()
            else:
                gates_np = np.array(gates)
            
            # Calculate average gate value for top-level feature
            avg_gate = float(np.mean(gates_np))
            is_active = avg_gate > threshold
            
            # Get RFE status
            rfe_selected = self.feature_performance[group].get('rfe_selected', False)
            rfe_rank = self.feature_performance[group].get('rfe_rank', 999)
            
            # Add to result based on feature type
            if group == 'ohlcv':
                # OHLCV is a single feature group
                active_features['ohlcv'] = {
                    'active': bool(is_active),
                    'weight': float(avg_gate),
                    'status': 'strong' if avg_gate > 0.7 else 'weak' if avg_gate <= self.min_weight + 0.01 else 'moderate',
                    'rfe_selected': rfe_selected,
                    'rfe_rank': rfe_rank if rfe_rank < 999 else None
                }
            elif group == 'sentiment':
                # Sentiment is a single feature
                active_features['sentiment'] = {
                    'active': bool(is_active),
                    'weight': float(avg_gate),
                    'status': 'strong' if avg_gate > 0.7 else 'weak' if avg_gate <= self.min_weight + 0.01 else 'moderate',
                    'rfe_selected': rfe_selected,
                    'rfe_rank': rfe_rank if rfe_rank < 999 else None
                }
            elif group == 'orderbook':
                # Orderbook is a single feature group
                active_features['orderbook'] = {
                    'active': bool(is_active),
                    'weight': float(avg_gate),
                    'status': 'strong' if avg_gate > 0.7 else 'weak' if avg_gate <= self.min_weight + 0.01 else 'moderate',
                    'rfe_selected': rfe_selected,
                    'rfe_rank': rfe_rank if rfe_rank < 999 else None
                }
            elif group == 'tick_data':
                # Tick data is a single feature group
                active_features['tick_data'] = {
                    'active': bool(is_active),
                    'weight': float(avg_gate),
                    'status': 'strong' if avg_gate > 0.7 else 'weak' if avg_gate <= self.min_weight + 0.01 else 'moderate',
                    'rfe_selected': rfe_selected,
                    'rfe_rank': rfe_rank if rfe_rank < 999 else None
                }
            elif group == 'indicator':
                # Indicators are individual features - show ALL 100 indicators with RFE status
                active_features['indicators'] = {}
                
                # Map indicator indices to actual indicator names (all 100)
                indicator_names = self._get_indicator_names()
                
                num_indicators = len(gates_np)
                for i in range(num_indicators):
                    ind_name = indicator_names[i] if i < len(indicator_names) else f"indicator_{i+1}"
                    weight = float(gates_np[i])
                    
                    # Get RFE status for this specific indicator
                    feat_key = f"indicator.{ind_name}"
                    ind_rfe_selected = (feat_key in self.rfe_selected_features and 
                                      self.rfe_selected_features[feat_key]['selected'])
                    ind_rfe_rank = (self.rfe_selected_features[feat_key]['rank'] 
                                   if feat_key in self.rfe_selected_features else None)
                    
                    active_features['indicators'][ind_name] = {
                        'active': bool(weight > threshold),
                        'weight': weight,
                        'status': 'strong' if weight > 0.7 else 'weak' if weight <= self.min_weight + 0.01 else 'moderate',
                        'rfe_selected': ind_rfe_selected,
                        'rfe_rank': ind_rfe_rank
                    }
            elif group == 'candle_pattern':
                # Candle patterns are individual features - show ALL patterns with RFE status
                active_features['candle_patterns'] = {}
                
                # This is handled in the indicators group now since patterns are included there
                pass
        
        return active_features
    
    def get_weak_features(self):
        """
        Get list of features that are effectively disabled (weight <= min_weight + 0.01)
        
        Returns:
            List of weak feature names
        """
        weak_features = []
        active_features = self.get_active_features(threshold=0.0)  # Get all features
        
        for group_name, group_data in active_features.items():
            if isinstance(group_data, dict):
                if 'status' in group_data and group_data['status'] == 'weak':
                    weak_features.append(group_name)
                else:
                    # Check sub-features
                    for feature_name, feature_data in group_data.items():
                        if isinstance(feature_data, dict) and feature_data.get('status') == 'weak':
                            weak_features.append(f"{group_name}.{feature_name}")
        
        return weak_features
    
    def get_scalar_weight(self, group_name):
        """
        Get scalar weight for a feature group to apply after projection
        
        Args:
            group_name: Name of the feature group
            
        Returns:
            Scalar weight to multiply the projected features
        """
        if not self.rfe_performed:
            # No RFE performed, return neutral weight
            return 1.0
        
        # Get all selected features categorized
        all_selected = [(name, info) for name, info in self.rfe_selected_features.items() if info['selected']]
        all_selected.sort(key=lambda x: x[1]['importance'], reverse=True)
        
        n_selected = len(all_selected)
        strong_count = min(self.rfe_n_features, n_selected)
        medium_count = min(strong_count * 2, n_selected - strong_count)
        
        strong_features = set(item[0] for item in all_selected[:strong_count])
        medium_features = set(item[0] for item in all_selected[strong_count:strong_count + medium_count])
        
        if group_name == 'indicator':
            # For indicators, average the weights of selected features in this group
            indicator_weights = []
            for name, info in self.rfe_selected_features.items():
                if name.startswith('indicator.') and info['selected']:
                    if name in strong_features:
                        indicator_weights.append(0.8)  # Strong weight
                    elif name in medium_features:
                        indicator_weights.append(0.5)  # Medium weight
                    else:
                        indicator_weights.append(float(self.min_weight))
            
            if indicator_weights:
                return float(np.mean(indicator_weights))
            else:
                return float(self.min_weight)
        else:
            # For other groups, use direct category mapping
            if group_name in strong_features:
                return 0.8  # Strong weight
            elif group_name in medium_features:
                return 0.5  # Medium weight
            else:
                return float(self.min_weight)  # Weak weight
    
    # Phase 2 Enhancement Methods
    
    def build_active_feature_masks(self, min_active_features=10):
        """
        Build persistent active feature masks for each group referencing original index positions.
        
        This method creates boolean masks indicating which features in each group should be 
        actively computed and used based on RFE selection results. Implements fallback logic
        to ensure minimum feature count requirements are met.
        
        Args:
            min_active_features: Minimum number of active features required across all groups
            
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping group names to boolean masks of active features
            
        Raises:
            Warning: If RFE has not been performed yet
        """
        if not self.rfe_performed:
            logger.warning("Cannot build active feature masks without RFE results")
            return {}
        
        active_masks = {}
        
        for group_name in self.feature_groups.keys():
            group_mask = []
            
            if group_name == 'indicator':
                # For indicators, check each individual indicator
                for i, indicator_name in enumerate(self._get_indicator_names()):
                    feature_key = f"indicator.{indicator_name}"
                    is_selected = (feature_key in self.rfe_selected_features and 
                                 self.rfe_selected_features[feature_key]['selected'])
                    group_mask.append(is_selected)
            else:
                # For other groups (ohlcv, sentiment, orderbook), check group-level selection
                feature_key = group_name
                is_selected = (feature_key in self.rfe_selected_features and 
                             self.rfe_selected_features[feature_key]['selected'])
                # Create mask for all features in the group
                group_size = self.feature_groups[group_name]
                group_mask = [is_selected] * group_size
            
            active_masks[group_name] = np.array(group_mask, dtype=bool)
        
        # Fallback: ensure minimum active features
        total_active = sum(np.sum(mask) for mask in active_masks.values())
        if total_active < min_active_features:
            self._ensure_minimum_active_features(active_masks, min_active_features)
        
        self.active_feature_masks = active_masks
        return active_masks
    
    def _ensure_minimum_active_features(self, active_masks, min_features):
        """
        Ensure minimum number of active features by adding medium/weak features.
        """
        current_count = sum(np.sum(mask) for mask in active_masks.values())
        if current_count >= min_features:
            return
        
        needed = min_features - current_count
        logger.info(f"Adding {needed} features to meet minimum threshold of {min_features}")
        
        # Get all features sorted by rank (best first)
        all_features = [(name, info) for name, info in self.rfe_selected_features.items()]
        all_features.sort(key=lambda x: x[1]['rank'])
        
        added = 0
        for feature_name, info in all_features:
            if added >= needed:
                break
                
            if not info['selected']:  # Only add unselected features
                # Determine which group this feature belongs to
                if feature_name.startswith('indicator.'):
                    group = 'indicator'
                    indicator_name = feature_name.split('.', 1)[1]
                    idx = self._get_indicator_index(indicator_name)
                    if idx >= 0 and idx < len(active_masks[group]):
                        active_masks[group][idx] = True
                        added += 1
                else:
                    if feature_name in self.feature_groups:
                        # Group-level feature - activate all features in group
                        active_masks[feature_name][:] = True
                        added += 1
    
    def _get_indicator_names(self):
        """Get list of indicator names in order they appear in feature matrix."""
        # This should match the order in IndicatorCalculator
        # Fixed to use only base indicators that exist in IndicatorCalculator
        return [
            'sma', 'ema', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'stoch_k', 'stoch_d', 'bbands_upper', 'bbands_middle', 'bbands_lower',
            'adx', 'atr', 'supertrend', 'willr', 'mfi', 'obv', 'ad', 'vwap',
            'engulfing', 'ppo', 'psar', 'trix', 'dmi', 'aroon', 'cci', 'dpo', 
            'kst', 'tema', 'roc', 'momentum', 'bop', 'apo', 'cmo',
            'rsi_2', 'rsi_14', 'stoch_fast', 'stoch_slow', 'ultimate_osc', 
            'kama', 'fisher', 'awesome_osc', 'bias', 'dmi_adx', 'tsi',
            'elder_ray', 'schaff_trend', 'chaikin_osc', 'mass_index', 
            'donchian', 'volatility', 'chaikin_vol', 'std_dev', 'rvi', 
            'true_range', 'avg_range', 'natr', 'pvt', 'envelope', 
            'price_channel', 'volatility_system', 'cmf', 'emv', 'fi', 'nvi', 
            'pvi', 'vol_osc', 'vol_rate', 'klinger', 'vol_sma', 'vol_ema', 
            'mfv', 'ad_line', 'obv_ma', 'vol_price_confirm', 'vol_weighted_macd', 
            'ease_of_movement', 'vol_accumulation', 'shooting_star', 'hanging_man',
            'morning_star', 'evening_star', 'three_white_soldiers', 
            'three_black_crows', 'harami', 'piercing', 'dark_cloud', 
            'spinning_top', 'marubozu', 'gravestone_doji', 'dragonfly_doji',
            'tweezer', 'inside_bar', 'outside_bar', 'pin_bar', 'gap_up',
            'gap_down', 'long_legged_doji', 'rickshaw_man', 'belt_hold',
            'hammer', 'doji',
            # Add composite indicator outputs as separate features
            'stochrsi_k', 'stochrsi_d', 'keltner_upper', 'keltner_middle', 'keltner_lower',
            'chikou', 'tenkan_sen', 'kijun_sen', 'senkou_a', 'senkou_b'
        ]
    
    def _get_indicator_mapping(self):
        """
        Get mapping from individual indicator features to their base indicators.
        
        Returns:
            dict: Mapping from feature names to base indicator names
        """
        return {
            # StochRSI components
            'stochrsi_k': 'stochrsi',
            'stochrsi_d': 'stochrsi',
            # Keltner Channel components  
            'keltner_upper': 'keltner',
            'keltner_middle': 'keltner',
            'keltner_lower': 'keltner',
            # Ichimoku components
            'chikou': 'ichimoku',
            'tenkan_sen': 'ichimoku', 
            'kijun_sen': 'ichimoku',
            'senkou_a': 'ichimoku',
            'senkou_b': 'ichimoku'
        }
    
    def _resolve_indicator_features(self, selected_features):
        """
        Resolve selected indicator features to their base indicators.
        
        Args:
            selected_features: List of selected feature names
            
        Returns:
            set: Set of base indicator names needed
        """
        mapping = self._get_indicator_mapping()
        base_indicators = set()
        
        for feature in selected_features:
            if feature.startswith('indicator.'):
                indicator_name = feature.split('.', 1)[1]
                base_indicator = mapping.get(indicator_name, indicator_name)
                base_indicators.add(base_indicator)
        
        return base_indicators
    
    def _get_indicator_index(self, indicator_name):
        """Get the index of an indicator in the feature matrix."""
        indicator_names = self._get_indicator_names()
        try:
            return indicator_names.index(indicator_name)
        except ValueError:
            return -1
    
    def should_trigger_periodic_rfe(self, config, current_performance=None):
        """
        Check if periodic RFE should be triggered based on time interval or performance drop.
        
        Implements both time-based and performance-based triggers for automatic RFE re-execution.
        Time-based triggers activate after a configured interval has elapsed. Performance-based
        triggers activate when trading success rate drops below a threshold.
        
        Args:
            config: Configuration dict with RFE settings containing:
                - rfe.periodic.enabled: Whether periodic RFE is enabled
                - rfe.periodic.interval_minutes: Time interval for periodic triggers  
                - rfe.trigger.performance_drop_pct: Performance drop threshold percentage
            current_performance: Current success rate (0.0-1.0), optional
            
        Returns:
            bool: True if RFE should be triggered, False otherwise
        """
        rfe_config = config.get('rfe', {})
        periodic_config = rfe_config.get('periodic', {})
        trigger_config = rfe_config.get('trigger', {})
        
        if not periodic_config.get('enabled', False):
            return False
        
        current_time = time.time()
        
        # Check time-based trigger
        interval_minutes = periodic_config.get('interval_minutes', 30)
        time_elapsed = (current_time - self.last_rfe_time) / 60 if self.last_rfe_time else float('inf')
        
        if time_elapsed >= interval_minutes:
            logger.info(f"Triggering periodic RFE: {time_elapsed:.1f} minutes elapsed (>= {interval_minutes})")
            return True
        
        # Check performance-based trigger
        if current_performance is not None:
            min_threshold = 1.0 - (trigger_config.get('performance_drop_pct', 15) / 100.0)
            if current_performance < min_threshold:
                logger.info(f"Triggering RFE due to performance drop: {current_performance:.3f} < {min_threshold:.3f}")
                return True
        
        return False
    
    def update_performance_tracking(self, success_rate):
        """
        Update performance tracking for periodic RFE decisions.
        
        Args:
            success_rate: Current success rate (0.0-1.0)
        """
        self.recent_success_rates.append(success_rate)
        
        # Keep only last 20 values for rolling window
        if len(self.recent_success_rates) > 20:
            self.recent_success_rates.pop(0)
        
        self.last_performance_check = time.time()
    
    def get_feature_set_version_hash(self):
        """
        Generate SHA1 hash of sorted selected feature names for versioning.
        
        Creates a deterministic hash based on the currently selected feature set to enable
        model checkpoint versioning and compatibility checking. The hash is generated from
        a sorted list of selected feature names to ensure consistency across runs.
        
        Returns:
            str: SHA1 hash (first 12 characters) of the feature set, or "no_features" if empty
        """
        if not self.selected_features:
            return "no_features"
        
        # Sort feature names for consistent hashing
        sorted_features = sorted(self.selected_features)
        feature_string = '|'.join(sorted_features)
        
        return hashlib.sha1(feature_string.encode('utf-8')).hexdigest()[:12]
    
    def save_rfe_summary_with_version(self, checkpoint_path=None):
        """
        Save RFE summary with feature set version alongside model checkpoints.
        
        Args:
            checkpoint_path: Path to model checkpoint (optional)
        """
        if not self.rfe_performed:
            return
        
        summary = self.get_rfe_summary()
        summary['feature_set_version'] = self.feature_set_version
        summary['feature_set_hash'] = self.get_feature_set_version_hash()
        summary['last_rfe_time'] = self.last_rfe_time or time.time()
        
        # Save to dedicated file
        summary_path = 'rfe_summary_with_version.json'
        if checkpoint_path:
            # Save alongside checkpoint
            import os
            checkpoint_dir = os.path.dirname(checkpoint_path)
            summary_path = os.path.join(checkpoint_dir, 'rfe_summary_with_version.json')
        
        try:
            import json
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"RFE summary with version saved to {summary_path}")
        except Exception as e:
            logger.error(f"Failed to save RFE summary with version: {e}")
    
    def load_checkpoint_with_compatibility(self, checkpoint_path):
        """
        Load model checkpoint with feature set compatibility checking (Phase 2).
        
        Provides backward compatibility for checkpoints that don't have feature_set_version.
        If a version mismatch is detected, offers options to handle the incompatibility.
        
        Args:
            checkpoint_path: Path to model checkpoint file
            
        Returns:
            dict: Loading results with compatibility information
        """
        try:
            import json
            import os
            
            result = {
                'loaded': False,
                'compatible': False,
                'version_mismatch': False,
                'missing_features': [],
                'action_taken': None
            }
            
            # Check if RFE summary exists alongside checkpoint
            checkpoint_dir = os.path.dirname(checkpoint_path)
            summary_path = os.path.join(checkpoint_dir, 'rfe_summary_with_version.json')
            
            checkpoint_version = None
            checkpoint_features = []
            
            if os.path.exists(summary_path):
                try:
                    with open(summary_path, 'r') as f:
                        summary = json.load(f)
                        checkpoint_version = summary.get('feature_set_version')
                        checkpoint_features = summary.get('selected_features', [])
                except Exception as e:
                    logger.warning(f"Could not read RFE summary: {e}")
            
            # Get current feature set
            current_version = self.feature_set_version
            current_features = self.selected_features.copy()
            
            # Check compatibility
            if checkpoint_version is None:
                logger.info("Loading checkpoint without feature_set_version (backward compatibility)")
                result['compatible'] = True
                result['action_taken'] = 'backward_compatibility'
            elif checkpoint_version == current_version:
                logger.info(f"Checkpoint version {checkpoint_version} matches current version")
                result['compatible'] = True
                result['action_taken'] = 'version_match'
            else:
                logger.warning(f"Version mismatch: checkpoint={checkpoint_version}, current={current_version}")
                result['version_mismatch'] = True
                
                # Check for missing features
                missing = set(checkpoint_features) - set(current_features)
                result['missing_features'] = list(missing)
                
                if missing:
                    logger.info(f"Temporarily re-adding {len(missing)} missing features for compatibility")
                    # Add missing features with minimum weight
                    for feature_name in missing:
                        if feature_name not in self.rfe_selected_features:
                            self.rfe_selected_features[feature_name] = {
                                'selected': True,
                                'rank': 999,
                                'importance': self.min_weight
                            }
                    
                    # Update selected features list
                    self.selected_features.extend(missing)
                    result['action_taken'] = 'auto_expand_features'
                    result['compatible'] = True
                else:
                    result['compatible'] = True
                    result['action_taken'] = 'subset_compatible'
            
            result['loaded'] = True
            return result
            
        except Exception as e:
            logger.error(f"Error loading checkpoint with compatibility: {e}")
            return {'loaded': False, 'error': str(e)}