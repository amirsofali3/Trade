import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from datetime import datetime

logger = logging.getLogger("Gating")

class FeatureGatingModule(nn.Module):
    """
    Enhanced feature gating mechanism with dynamic feature selection
    
    Features:
    - Adaptively select important features based on market conditions
    - Weak features get minimum weight of 0.01 (effectively disabled but trackable)
    - Strong features maintain high impact
    - Context-aware gating for real-time adaptation
    """
    
    def __init__(self, feature_groups, hidden_dim=64, min_weight=0.01, adaptation_rate=0.1):
        """
        Initialize feature gating module
        
        Args:
            feature_groups: Dictionary with feature groups and their dimensions
                Example: {'ohlcv': 13, 'indicator': 20, 'sentiment': 1, 'orderbook': 42}
            hidden_dim: Hidden dimension for gating network
            min_weight: Minimum weight for weak features (default 0.01)
            adaptation_rate: Rate of adaptation for feature weights
        """
        super(FeatureGatingModule, self).__init__()
        self.feature_groups = feature_groups
        self.group_names = list(feature_groups.keys())
        self.min_weight = min_weight
        self.adaptation_rate = adaptation_rate
        
        # Calculate total input dimension
        self.total_dim = sum(feature_groups.values())
        
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
                'weight_history': []
            }
        
        # Store current gates for monitoring
        self.current_gates = {group: torch.ones(dim) for group, dim in feature_groups.items()}
        
        # Feature adaptation parameters
        self.update_frequency = 5  # seconds between updates
        
        logger.info(f"Initialized Enhanced FeatureGatingModule with {len(feature_groups)} feature groups")
        logger.info(f"Min weight for weak features: {min_weight}, Adaptation rate: {adaptation_rate}")
    
    def forward(self, features_dict):
        """
        Apply enhanced gating to features with minimum weight enforcement
        
        Args:
            features_dict: Dictionary with feature tensors by group
                Example: {'ohlcv': tensor(...), 'indicator': tensor(...), ...}
        
        Returns:
            Dictionary with gated feature tensors
        """
        if not features_dict:
            logger.warning("No features provided to gating module")
            return {}
        
        # Concatenate all features for context
        all_features = []
        
        for group in self.group_names:
            if group in features_dict:
                # Flatten features if multidimensional
                flat_features = features_dict[group].view(1, -1)
                
                # If completely flat, reshape to ensure 2D
                if flat_features.dim() == 1:
                    flat_features = flat_features.view(1, -1)
                
                # Average over time dimension if present
                if flat_features.size(0) > 1:
                    flat_features = torch.mean(flat_features, dim=0, keepdim=True)
                
                all_features.append(flat_features)
            else:
                # If group is missing, use zeros
                dim = self.feature_groups[group]
                all_features.append(torch.zeros((1, dim)))
        
        # Concatenate along feature dimension
        context = torch.cat(all_features, dim=1)
        
        # Apply gates to each group with enhanced logic
        gated_features = {}
        
        for group in self.group_names:
            if group in features_dict:
                # Calculate raw gates for this group
                raw_gates = self.gate_networks[group](context)
                
                # Apply minimum weight constraint
                # Weak features get min_weight, strong features keep their strength
                gates = torch.clamp(raw_gates, min=self.min_weight, max=1.0)
                
                # Ensure gates have right shape
                if gates.size(-1) != self.feature_groups[group]:
                    gates = gates[:, :self.feature_groups[group]]
                
                # Reshape gates to match features
                feature_shape = features_dict[group].shape
                if len(feature_shape) > 2:  # For features with time dimension
                    gates = gates.view(1, 1, -1).expand(-1, feature_shape[1], -1)
                
                # Apply gates element-wise
                gated_features[group] = features_dict[group] * gates
                
                # Store current gates for monitoring
                self.current_gates[group] = gates.detach().squeeze()
                
                # Update last update time
                self.last_update_time[group] = datetime.now()
                
                # Log feature status for debugging
                avg_weight = float(torch.mean(gates))
                if avg_weight <= self.min_weight + 0.01:
                    logger.debug(f"Feature group {group} has low weight ({avg_weight:.3f}) - effectively disabled")
                elif avg_weight >= 0.7:
                    logger.debug(f"Feature group {group} has high weight ({avg_weight:.3f}) - strongly active")
                
            else:
                # Pass group with zeros if missing
                gated_features[group] = torch.zeros((1, self.feature_groups[group]))
        
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
        Get currently active features with enhanced reporting
        Shows all features including those with 0.01 weight (effectively disabled)
        
        Args:
            threshold: Threshold for considering a feature active
            
        Returns:
            Dictionary with active features and their weights
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
            
            # Add to result based on feature type
            if group == 'ohlcv':
                # OHLCV is a single feature group
                active_features['ohlcv'] = {
                    'active': bool(is_active),
                    'weight': float(avg_gate),
                    'status': 'strong' if avg_gate > 0.7 else 'weak' if avg_gate <= self.min_weight + 0.01 else 'moderate'
                }
            elif group == 'sentiment':
                # Sentiment is a single feature
                active_features['sentiment'] = {
                    'active': bool(is_active),
                    'weight': float(avg_gate),
                    'status': 'strong' if avg_gate > 0.7 else 'weak' if avg_gate <= self.min_weight + 0.01 else 'moderate'
                }
            elif group == 'orderbook':
                # Orderbook is a single feature group
                active_features['orderbook'] = {
                    'active': bool(is_active),
                    'weight': float(avg_gate),
                    'status': 'strong' if avg_gate > 0.7 else 'weak' if avg_gate <= self.min_weight + 0.01 else 'moderate'
                }
            elif group == 'tick_data':
                # Tick data is a single feature group
                active_features['tick_data'] = {
                    'active': bool(is_active),
                    'weight': float(avg_gate),
                    'status': 'strong' if avg_gate > 0.7 else 'weak' if avg_gate <= self.min_weight + 0.01 else 'moderate'
                }
            elif group == 'indicator':
                # Indicators are individual features - show ALL indicators
                active_features['indicators'] = {}
                
                # Map indicator indices to actual indicator names
                indicator_names = [
                    'sma', 'ema', 'rsi', 'macd', 'macd_signal', 'macd_hist',
                    'stoch_k', 'stoch_d', 'bbands_upper', 'bbands_middle', 'bbands_lower',
                    'adx', 'atr', 'supertrend', 'willr', 'mfi', 'obv', 'ad', 'vwap',
                    'engulfing'  # Add more as needed
                ]
                
                num_indicators = len(gates_np)
                for i in range(num_indicators):
                    ind_name = indicator_names[i] if i < len(indicator_names) else f"indicator_{i+1}"
                    weight = float(gates_np[i])
                    active_features['indicators'][ind_name] = {
                        'active': bool(weight > threshold),
                        'weight': weight,
                        'status': 'strong' if weight > 0.7 else 'weak' if weight <= self.min_weight + 0.01 else 'moderate'
                    }
            elif group == 'candle_pattern':
                # Candle patterns are individual features - show ALL patterns
                active_features['candle_patterns'] = {}
                
                # Map pattern indices to actual pattern names
                pattern_names = [
                    'hammer', 'doji', 'engulfing', 'shooting_star', 'hanging_man',
                    'morning_star', 'evening_star', 'three_white_soldiers', 'three_black_crows',
                    'harami'  # Add more as needed
                ]
                
                num_patterns = len(gates_np)
                for i in range(num_patterns):
                    pattern_name = pattern_names[i] if i < len(pattern_names) else f"pattern_{i+1}"
                    weight = float(gates_np[i])
                    active_features['candle_patterns'][pattern_name] = {
                        'active': bool(weight > threshold),
                        'weight': weight,
                        'status': 'strong' if weight > 0.7 else 'weak' if weight <= self.min_weight + 0.01 else 'moderate'
                    }
        
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