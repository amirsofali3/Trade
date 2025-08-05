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
    Feature gating mechanism to adaptively select important features
    based on market conditions
    """
    
    def __init__(self, feature_groups, hidden_dim=64):
        """
        Initialize feature gating module
        
        Args:
            feature_groups: Dictionary with feature groups and their dimensions
                Example: {'ohlcv': 13, 'indicator': 20, 'sentiment': 1, 'orderbook': 42}
            hidden_dim: Hidden dimension for gating network
        """
        super(FeatureGatingModule, self).__init__()
        self.feature_groups = feature_groups
        self.group_names = list(feature_groups.keys())
        
        # Calculate total input dimension
        self.total_dim = sum(feature_groups.values())
        
        # Create gate networks for each feature group
        self.gate_networks = nn.ModuleDict()
        
        for group_name, dim in feature_groups.items():
            self.gate_networks[group_name] = nn.Sequential(
                nn.Linear(self.total_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, dim),
                nn.Sigmoid()
            )
        
        # Initialize last update time
        self.last_update_time = {}
        for group_name in feature_groups:
            self.last_update_time[group_name] = datetime.now()
        
        # Store current gates for monitoring
        self.current_gates = {group: torch.ones(dim) for group, dim in feature_groups.items()}
        
        logger.info(f"Initialized FeatureGatingModule with {len(feature_groups)} feature groups")
    
    def forward(self, features_dict):
        """
        Apply gating to features
        
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
        
        # Apply gates to each group
        gated_features = {}
        
        for group in self.group_names:
            if group in features_dict:
                # Calculate gates for this group
                gates = self.gate_networks[group](context)
                
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
            else:
                # Pass group with zeros if missing
                gated_features[group] = torch.zeros((1, self.feature_groups[group]))
        
        return gated_features
    
    def get_active_features(self, threshold=0.3):
        """
        Get currently active features
        
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
            
            # Add to result
            if group == 'ohlcv':
                # OHLCV is a single feature group
                active_features['ohlcv'] = {
                    'active': bool(is_active),
                    'weight': float(avg_gate)
                }
            elif group == 'sentiment':
                # Sentiment is a single feature
                active_features['sentiment'] = {
                    'active': bool(is_active),
                    'weight': float(avg_gate)
                }
            elif group == 'orderbook':
                # Orderbook is a single feature group
                active_features['orderbook'] = {
                    'active': bool(is_active),
                    'weight': float(avg_gate)
                }
            elif group == 'tick_data':
                # Tick data is a single feature group
                active_features['tick_data'] = {
                    'active': bool(is_active),
                    'weight': float(avg_gate)
                }
            elif group == 'indicator':
                # Indicators are individual features
                active_features['indicators'] = {}
                
                # Basic indicators (we don't know specific names so we use indices)
                num_indicators = len(gates_np)
                for i in range(min(num_indicators, 10)):  # Limit to 10 indicators for visualization
                    ind_name = f"indicator_{i+1}"
                    active_features['indicators'][ind_name] = {
                        'active': bool(gates_np[i] > threshold),
                        'weight': float(gates_np[i])
                    }
            elif group == 'candle_pattern':
                # Candle patterns are individual features
                active_features['candle_patterns'] = {}
                
                # Basic patterns (we don't know specific names so we use indices)
                num_patterns = len(gates_np)
                for i in range(min(num_patterns, 10)):  # Limit to 10 patterns for visualization
                    pattern_name = f"pattern_{i+1}"
                    active_features['candle_patterns'][pattern_name] = {
                        'active': bool(gates_np[i] > threshold),
                        'weight': float(gates_np[i])
                    }
        
        return active_features