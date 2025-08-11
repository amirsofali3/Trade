import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger("Gating")

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
        Perform RFE feature selection before training starts
        
        Args:
            training_data: Training feature data as dict {group_name: features_array}
            training_labels: Training labels (0=BUY, 1=SELL, 2=HOLD)
            
        Returns:
            Dictionary with selected features and rankings
        """
        if not self.rfe_enabled or self.rfe_performed:
            logger.info("RFE already performed or disabled")
            return self.rfe_selected_features
        
        logger.info("🔍 Starting RFE feature selection process...")
        
        try:
            # Flatten all features into a single matrix
            feature_matrix = []
            feature_names = []
            
            for group_name, features in training_data.items():
                if group_name == 'indicator':
                    # For indicator group, we want individual feature selection
                    if isinstance(features, np.ndarray) and features.ndim >= 2:
                        # Handle different shapes
                        if features.ndim == 3:
                            # (samples, sequence, features) -> take mean over sequence
                            features_2d = np.mean(features, axis=1)
                        else:
                            features_2d = features
                            
                        # Add each indicator as separate feature
                        indicator_names = [
                            'sma', 'ema', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                            'stoch_k', 'stoch_d', 'bbands_upper', 'bbands_middle', 'bbands_lower',
                            'adx', 'atr', 'supertrend', 'willr', 'mfi', 'obv', 'ad', 'vwap',
                            'engulfing', 'ppo', 'psar', 'trix', 'dmi', 'aroon', 'cci', 'dpo', 
                            'kst', 'ichimoku', 'tema', 'roc', 'momentum', 'bop', 'apo', 'cmo',
                            'rsi_2', 'rsi_14', 'stoch_fast', 'stoch_slow', 'ultimate_osc', 
                            'kama', 'fisher', 'awesome_osc', 'bias', 'dmi_adx', 'tsi',
                            'elder_ray', 'schaff_trend', 'chaikin_osc', 'mass_index', 'keltner',
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
                            'hammer', 'doji'
                        ]
                        
                        n_features = min(features_2d.shape[1], len(indicator_names))
                        for i in range(n_features):
                            feature_matrix.append(features_2d[:, i])
                            feat_name = indicator_names[i] if i < len(indicator_names) else f"indicator_{i+1}"
                            feature_names.append(f"{group_name}.{feat_name}")
                else:
                    # For other groups, treat as single feature group
                    if isinstance(features, np.ndarray):
                        if features.ndim == 1:
                            feature_matrix.append(features)
                        elif features.ndim == 2:
                            # Take mean if multiple dimensions
                            feature_matrix.append(np.mean(features, axis=1))
                        else:
                            # Take mean over all dimensions except first
                            reshaped = features.reshape(features.shape[0], -1)
                            feature_matrix.append(np.mean(reshaped, axis=1))
                        feature_names.append(group_name)
            
            if not feature_matrix:
                logger.warning("No features available for RFE")
                return {}
            
            # Transpose to get (samples, features) shape
            X = np.column_stack(feature_matrix)
            y = np.array(training_labels)
            
            # Remove samples with NaN/Inf
            valid_samples = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
            X_clean = X[valid_samples]
            y_clean = y[valid_samples]
            
            if len(X_clean) < 10:
                logger.warning("Insufficient clean samples for RFE")
                return {}
            
            logger.info(f"RFE input: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
            
            # Use Random Forest as the estimator for RFE
            rf_estimator = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=1  # Avoid multiprocessing issues
            )
            
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
                self.rfe_feature_rankings[feature_name] = rank
                is_selected = i in selected_indices
                
                if is_selected:
                    self.rfe_selected_features[feature_name] = {
                        'rank': rank,
                        'importance': rf_estimator.feature_importances_[i] if hasattr(rf_estimator, 'feature_importances_') else 0.0,
                        'selected': True
                    }
                    logger.info(f"  ✅ {feature_name} (rank: {rank})")
                else:
                    # Still track unselected features but mark them
                    self.rfe_selected_features[feature_name] = {
                        'rank': rank,
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
            self.rfe_model = rfe
            
            logger.info(f"🚀 RFE completed! Selected {len([f for f in self.rfe_selected_features.values() if f['selected']])} features")
            
            return self.rfe_selected_features
            
        except Exception as e:
            logger.error(f"Error during RFE: {str(e)}")
            return {}
    
    def get_rfe_weights(self):
        """
        Get feature weights based on RFE results
        
        Returns:
            Dictionary with weights for each feature group
        """
        if not self.rfe_performed or not self.rfe_selected_features:
            return {}
        
        weights = {}
        
        for group_name, group_dim in self.feature_groups.items():
            if group_name == 'indicator':
                # For indicators, create weight vector based on individual selections
                group_weights = torch.full((group_dim,), self.min_weight)
                
                indicator_names = [
                    'sma', 'ema', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                    'stoch_k', 'stoch_d', 'bbands_upper', 'bbands_middle', 'bbands_lower',
                    'adx', 'atr', 'supertrend', 'willr', 'mfi', 'obv', 'ad', 'vwap',
                    'engulfing'  # Plus 80 more...
                ]
                
                for i, ind_name in enumerate(indicator_names[:group_dim]):
                    feat_key = f"indicator.{ind_name}"
                    if feat_key in self.rfe_selected_features and self.rfe_selected_features[feat_key]['selected']:
                        # Strong weight for selected features
                        importance = self.rfe_selected_features[feat_key]['importance']
                        group_weights[i] = max(0.7, importance)  # At least 70% weight for selected
                
                weights[group_name] = group_weights
            else:
                # For other groups
                if group_name in self.rfe_selected_features and self.rfe_selected_features[group_name]['selected']:
                    importance = self.rfe_selected_features[group_name]['importance']
                    weight_val = max(0.7, importance)
                else:
                    weight_val = self.min_weight
                
                weights[group_name] = torch.full((group_dim,), weight_val)
        
        return weights
    
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
                indicator_names = [
                    'sma', 'ema', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                    'stoch_k', 'stoch_d', 'bbands_upper', 'bbands_middle', 'bbands_lower',
                    'adx', 'atr', 'supertrend', 'willr', 'mfi', 'obv', 'ad', 'vwap',
                    'engulfing', 'ppo', 'psar', 'trix', 'dmi', 'aroon', 'cci', 'dpo', 
                    'kst', 'ichimoku', 'tema', 'roc', 'momentum', 'bop', 'apo', 'cmo',
                    'rsi_2', 'rsi_14', 'stoch_fast', 'stoch_slow', 'ultimate_osc', 
                    'kama', 'fisher', 'awesome_osc', 'bias', 'dmi_adx', 'tsi',
                    'elder_ray', 'schaff_trend', 'chaikin_osc', 'mass_index', 'keltner',
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
                    'hammer', 'doji'
                ]
                
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
    
    def get_rfe_summary(self):
        """
        Get a summary of RFE feature selection results
        
        Returns:
            Dictionary with RFE summary statistics
        """
        if not self.rfe_performed:
            return {'rfe_performed': False, 'message': 'RFE not yet performed'}
        
        selected_features = [name for name, data in self.rfe_selected_features.items() 
                           if data.get('selected', False)]
        
        # Count by category
        trend_selected = len([f for f in selected_features if any(trend in f for trend in 
                            ['sma', 'ema', 'macd', 'adx', 'ppo', 'psar', 'trix', 'aroon', 'cci', 'dpo', 'kst', 'ichimoku', 'tema'])])
        momentum_selected = len([f for f in selected_features if any(mom in f for mom in 
                               ['rsi', 'stoch', 'mfi', 'willr', 'roc', 'momentum', 'bop', 'apo', 'cmo', 'ultimate_osc', 
                                'kama', 'fisher', 'awesome_osc', 'bias', 'tsi', 'elder_ray', 'schaff_trend', 'mass_index'])])
        volatility_selected = len([f for f in selected_features if any(vol in f for vol in 
                                 ['bbands', 'atr', 'keltner', 'donchian', 'volatility', 'chaikin_vol', 'std_dev', 'rvi', 
                                  'true_range', 'avg_range', 'natr', 'pvt', 'envelope', 'price_channel'])])
        volume_selected = len([f for f in selected_features if any(vol in f for vol in 
                             ['obv', 'ad', 'vwap', 'cmf', 'emv', 'fi', 'nvi', 'pvi', 'vol_', 'klinger', 'mfv', 'ease_of_movement'])])
        pattern_selected = len([f for f in selected_features if any(pat in f for pat in 
                              ['engulfing', 'hammer', 'doji', 'shooting_star', 'hanging_man', 'morning_star', 'evening_star', 
                               'three_', 'harami', 'piercing', 'dark_cloud', 'spinning_top', 'marubozu', 'tweezer', 
                               'inside_bar', 'outside_bar', 'pin_bar', 'gap_', 'belt_hold'])])
        
        return {
            'rfe_performed': True,
            'total_selected': len(selected_features),
            'target_features': self.rfe_n_features,
            'selection_breakdown': {
                'trend': trend_selected,
                'momentum': momentum_selected, 
                'volatility': volatility_selected,
                'volume': volume_selected,
                'pattern': pattern_selected,
                'other': len(selected_features) - (trend_selected + momentum_selected + volatility_selected + volume_selected + pattern_selected)
            },
            'top_features': sorted(selected_features, 
                                 key=lambda x: self.rfe_selected_features[x]['rank'])[:10]
        }
    
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