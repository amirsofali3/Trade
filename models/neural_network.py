import logging
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import time
import threading
import tempfile
from .locks import version_lock

logger = logging.getLogger("NeuralNetwork")

class MarketTransformer(nn.Module):
    """
    Transformer-based model for market prediction with context-aware feature gating
    """
    
    def __init__(self, feature_dims, hidden_dim=128, n_layers=2, n_heads=4, dropout=0.1):
        """
        Initialize Market Transformer with projection layers for consistent dimensions
        
        Args:
            feature_dims: Dictionary with feature dimensions
                Example: {'ohlcv': (20, 13), 'indicator': (20, 100), 'sentiment': (1, 1), 'orderbook': (1, 42)}
                Format is (seq_len, feature_dim)
            hidden_dim: Hidden dimension for transformer
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(MarketTransformer, self).__init__()
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        
        # Calculate total feature dimension
        self.total_feature_dim = sum(dim[1] for dim in feature_dims.values())
        
        # Projection layers to map each feature group to hidden_dim (fixes 128 vs 100 issue)
        self.projections = nn.ModuleDict()
        
        for name, (seq_len, dim) in feature_dims.items():
            # Projection layer maps input features to hidden_dim
            self.projections[name] = nn.Linear(dim, hidden_dim)
            
            logger.debug(f"Created projection for {name}: {dim} → {hidden_dim}")
        
        # Position encoding
        max_seq_len = max(dim[0] for dim in feature_dims.values())
        self.position_encoding = self._create_position_encoding(max_seq_len, hidden_dim)
        
        # Common transformer encoder layer for all features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )
        
        # Final prediction layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 3)  # Outputs: [buy_prob, sell_prob, hold_prob]
        
        logger.info(f"Initialized MarketTransformer with hidden_dim={hidden_dim}, n_layers={n_layers}, n_heads={n_heads}")
    
    def _create_position_encoding(self, seq_len, d_model):
        """Create position encoding for transformer"""
        pos = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        return pe.unsqueeze(0)  # Shape: (1, seq_len, d_model)
    
    def forward(self, features_dict):
        """
        Forward pass
        
        Args:
            features_dict: Dictionary with feature tensors by group
                Example: {'ohlcv': tensor(...), 'indicator': tensor(...), ...}
        
        Returns:
            Tensor with prediction probabilities [buy_prob, sell_prob, hold_prob]
            and confidence score
        """
        if not features_dict:
            logger.warning("No features provided to model")
            return torch.tensor([[0.0, 0.0, 1.0]]), 0.0
        
        # Process and embed each feature type with improved shape handling
        embedded_features = []
        
        for name, (expected_seq_len, expected_feature_dim) in self.feature_dims.items():
            if name in features_dict:
                # Get features and ensure proper shape
                x = features_dict[name]
                
                # Handle different input formats more robustly
                if len(x.shape) == 1:
                    # 1D tensor: check if it's sequential or single features
                    if name in ['ohlcv', 'indicator', 'tick_data']:
                        # Sequential data flattened - reshape to (1, seq_len, feature_dim)
                        total_elements = x.size(0)
                        if total_elements == expected_seq_len * expected_feature_dim:
                            x = x.view(1, expected_seq_len, expected_feature_dim)
                        else:
                            # Truncate or pad to expected dimensions
                            if total_elements > expected_feature_dim:
                                # Treat as sequence and take recent values
                                x = x[-expected_feature_dim:].unsqueeze(0).unsqueeze(0)
                            else:
                                # Pad and treat as single timestep
                                padded = torch.zeros(expected_feature_dim)
                                padded[:total_elements] = x
                                x = padded.unsqueeze(0).unsqueeze(0)
                    else:
                        # Non-sequential data - add batch dimension
                        if x.size(0) == expected_feature_dim:
                            x = x.unsqueeze(0)  # Shape: (1, features)
                        else:
                            # Truncate or pad
                            padded = torch.zeros(expected_feature_dim)
                            padded[:min(x.size(0), expected_feature_dim)] = x[:expected_feature_dim]
                            x = padded.unsqueeze(0)
                
                elif len(x.shape) == 2:
                    # 2D tensor: could be (batch, features) or (seq, features)
                    if name in ['ohlcv', 'indicator', 'tick_data']:
                        # Sequential data: should be (seq, features)
                        if x.size(0) == 1:
                            # Actually (1, total_features) - reshape to sequence
                            total_features = x.size(1)
                            if total_features == expected_seq_len * expected_feature_dim:
                                x = x.view(1, expected_seq_len, expected_feature_dim)
                            else:
                                # Expand to sequence dimension
                                x = x.unsqueeze(1).expand(-1, expected_seq_len, -1)
                                # Adjust feature dimension
                                if x.size(-1) != expected_feature_dim:
                                    if x.size(-1) > expected_feature_dim:
                                        x = x[:, :, :expected_feature_dim]
                                    else:
                                        padding = torch.zeros(x.size(0), x.size(1), expected_feature_dim - x.size(-1))
                                        x = torch.cat([x, padding], dim=-1)
                        else:
                            # Proper (seq, features) format
                            # Ensure batch dimension
                            if x.size(0) != expected_seq_len or x.size(1) != expected_feature_dim:
                                # Adjust sequence length
                                if x.size(0) > expected_seq_len:
                                    x = x[-expected_seq_len:, :]
                                elif x.size(0) < expected_seq_len:
                                    padding = torch.zeros(expected_seq_len - x.size(0), x.size(1))
                                    x = torch.cat([padding, x], dim=0)
                                
                                # Adjust feature dimension
                                if x.size(1) != expected_feature_dim:
                                    if x.size(1) > expected_feature_dim:
                                        x = x[:, :expected_feature_dim]
                                    else:
                                        padding = torch.zeros(x.size(0), expected_feature_dim - x.size(1))
                                        x = torch.cat([x, padding], dim=1)
                            
                            # Add batch dimension
                            x = x.unsqueeze(0)
                    else:
                        # Non-sequential data: should be (batch, features)
                        if x.size(0) > 1:
                            # Multiple batches - take mean
                            x = torch.mean(x, dim=0, keepdim=True)
                        
                        # Adjust feature dimension
                        if x.size(1) != expected_feature_dim:
                            if x.size(1) > expected_feature_dim:
                                x = x[:, :expected_feature_dim]
                            else:
                                padding = torch.zeros(x.size(0), expected_feature_dim - x.size(1))
                                x = torch.cat([x, padding], dim=1)
                
                elif len(x.shape) == 3:
                    # 3D tensor: (batch, seq, features) - ideal format
                    batch_size, seq_len, feature_dim = x.shape
                    
                    # Handle batch dimension
                    if batch_size > 1:
                        x = torch.mean(x, dim=0, keepdim=True)
                    
                    # Adjust sequence length
                    if seq_len != expected_seq_len:
                        if seq_len > expected_seq_len:
                            x = x[:, -expected_seq_len:, :]
                        else:
                            padding = torch.zeros(x.size(0), expected_seq_len - seq_len, x.size(2))
                            x = torch.cat([padding, x], dim=1)
                    
                    # Adjust feature dimension
                    if feature_dim != expected_feature_dim:
                        if feature_dim > expected_feature_dim:
                            x = x[:, :, :expected_feature_dim]
                        else:
                            padding = torch.zeros(x.size(0), x.size(1), expected_feature_dim - feature_dim)
                            x = torch.cat([x, padding], dim=2)
                
                else:
                    # Higher dimensional - flatten and reshape
                    x = x.view(-1)
                    total_elements = expected_seq_len * expected_feature_dim
                    
                    if x.size(0) >= total_elements:
                        x = x[:total_elements].view(1, expected_seq_len, expected_feature_dim)
                    else:
                        padded = torch.zeros(total_elements)
                        padded[:x.size(0)] = x
                        x = padded.view(1, expected_seq_len, expected_feature_dim)
                
                # Project to hidden_dim first to ensure dimensional consistency
                if name in ['ohlcv', 'indicator', 'tick_data']:
                    # Sequential features: project each timestep
                    # x shape: (1, seq_len, feature_dim)
                    batch_size, seq_len, feature_dim = x.shape
                    x_flat = x.view(-1, feature_dim)  # (batch_size * seq_len, feature_dim)
                    
                    logger.debug(f"Processing {name}: input shape {x.shape} -> flat shape {x_flat.shape}")
                    
                    # Project to hidden_dim
                    projected_flat = self.projections[name](x_flat)  # (batch_size * seq_len, hidden_dim)
                    embedded = projected_flat.view(batch_size, seq_len, self.hidden_dim)
                    
                    logger.debug(f"After projection {name}: shape {embedded.shape}")
                else:
                    # Non-sequential features: project directly
                    # x might be (1, feature_dim) or (1, 1, feature_dim)
                    logger.debug(f"Processing {name}: input shape {x.shape}")
                    
                    # Ensure x is 2D for projection
                    if x.dim() > 2:
                        x_2d = x.view(-1, x.shape[-1])  # Flatten to (batch, feature_dim)
                    else:
                        x_2d = x
                    
                    # Project to hidden_dim
                    projected = self.projections[name](x_2d)  # (batch, hidden_dim)
                    
                    # Ensure we have the right shape: (batch, 1, hidden_dim)
                    if projected.dim() == 2:
                        embedded = projected.unsqueeze(1)  # (batch, 1, hidden_dim)
                    else:
                        embedded = projected
                    
                    logger.debug(f"After projection {name}: shape {embedded.shape}")
                
                # Add position encoding for sequential features
                if name == 'ohlcv' or name == 'indicator' or name == 'tick_data':
                    seq_len_actual = min(embedded.shape[1], self.position_encoding.shape[1])
                    embedded[:, :seq_len_actual, :] += self.position_encoding[:, :seq_len_actual, :]
                
                embedded_features.append(embedded)
        
        if not embedded_features:
            logger.warning("No valid features to process")
            return torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([0.0])
        
        # Combine features by concatenating along sequence dimension
        # All features now have shape (1, seq_len, hidden_dim) or (1, 1, hidden_dim)
        max_seq_len = max(feat.shape[1] for feat in embedded_features)
        
        # Expand all features to max sequence length
        expanded_features = []
        for feat in embedded_features:
            if feat.shape[1] < max_seq_len:
                # Expand by repeating the last timestep
                seq_diff = max_seq_len - feat.shape[1]
                last_timestep = feat[:, -1:, :]  # (batch, 1, hidden_dim)
                # Repeat the last timestep to fill sequence
                repeated = last_timestep.expand(-1, seq_diff, -1)  # Expand along sequence dim
                feat_expanded = torch.cat([feat, repeated], dim=1)
            else:
                feat_expanded = feat
            expanded_features.append(feat_expanded)
        
        # Concatenate along the hidden dimension (sum embeddings)
        # Alternative: could concatenate but that would increase hidden_dim
        x = torch.stack(expanded_features, dim=0).sum(dim=0)  # Sum across feature types
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Pool over sequence dimension (global average pooling)
        x = torch.mean(x, dim=1)  # Shape: (1, hidden_dim)
        
        # Final prediction layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Calculate confidence as the max probability
        confidence = torch.max(probs, dim=1)[0]
        
        return probs, confidence

class OnlineLearner:
    """
    Enhanced online learning mechanism with trade outcome feedback and semantic versioning
    
    Features:
    - Learns from trade outcomes (profit/loss, accuracy)
    - Adjusts feature weights based on success rates
    - Implements continuous model improvement
    - Saves learning progress with semantic versioning (major.minor.patch)
    """
    
    def __init__(self, model, gating_module=None, optimizer_cls=torch.optim.Adam, 
                lr=1e-4, buffer_size=1000, batch_size=32,
                update_interval=3600, save_dir='saved_models', 
                initial_version="1.0.0"):
        """
        Initialize enhanced online learner with version tracking
        
        Args:
            model: Neural network model
            gating_module: Feature gating module for feedback
            optimizer_cls: Optimizer class
            lr: Learning rate
            buffer_size: Maximum size of experience buffer
            batch_size: Batch size for training
            update_interval: Time interval between updates in seconds
            save_dir: Directory to save model checkpoints
            initial_version: Initial model version (semantic versioning)
        """
        self.model = model
        self.gating_module = gating_module
        self.optimizer = optimizer_cls(model.parameters(), lr=lr)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.save_dir = save_dir
        
        # Version tracking
        self.model_version = initial_version
        self.version_history = []
        self.last_major_update = datetime.now()
        
        # Create experience buffer with enhanced data
        self.experience_buffer = {
            'features': [],
            'labels': [],
            'timestamps': [],
            'confidences': [],
            'outcomes': [],  # Actual trade outcomes (profit/loss)
            'feature_contributions': []  # Which features contributed to decision
        }
        
        # Trade outcome tracking for learning
        self.trade_outcomes = {
            'successful_trades': 0,
            'failed_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'feature_success_rates': {}
        }
        
        # Statistics
        self.updates_counter = 0
        self.last_update_time = datetime.now()
        self.learning_rate_schedule = lr
        
        # Performance tracking for version bumping
        self.performance_window = []  # Track recent performance
        self.performance_threshold_major = 0.05  # 5% improvement for major version
        self.performance_threshold_minor = 0.02  # 2% improvement for minor version
        
        # Make sure save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Start update thread
        self.running = False
        self.update_thread = None
        
        logger.info(f"Initialized Enhanced OnlineLearner with feedback learning")
        logger.info(f"Buffer size: {buffer_size}, Batch size: {batch_size}, Update interval: {update_interval}s")
    
    def perform_warmup_training(self, warmup_data, max_batches=30):
        """
        Perform warmup training to initialize model weights and avoid stuck confidence
        
        Args:
            warmup_data: List of training samples [(features_dict, label), ...]
            max_batches: Maximum number of warmup batches to run
            
        Returns:
            Dictionary with warmup training results
        """
        logger.info(f"🔥 Starting warmup training with {len(warmup_data)} samples, max_batches={max_batches}")
        
        if len(warmup_data) < 5:
            logger.warning("Insufficient data for warmup training")
            return {'batches_completed': 0, 'final_loss': None}
        
        try:
            initial_loss = None
            final_loss = None
            batches_completed = 0
            
            # Set model to training mode
            self.model.train()
            
            # Adjust learning rate for warmup (typically higher)
            original_lr = self.optimizer.param_groups[0]['lr']
            warmup_lr = original_lr * 2.0  # Double the learning rate for warmup
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = warmup_lr
            
            logger.info(f"Warmup learning rate: {warmup_lr} (original: {original_lr})")
            
            for batch_idx in range(max_batches):
                try:
                    # Sample random batch from warmup data
                    batch_size = min(self.batch_size, len(warmup_data))
                    batch_indices = np.random.choice(len(warmup_data), size=batch_size, replace=False)
                    batch_samples = [warmup_data[i] for i in batch_indices]
                    
                    # Prepare batch
                    batch_features = {}
                    batch_labels = []
                    
                    for features_dict, label in batch_samples:
                        # Collect features
                        for name, tensor in features_dict.items():
                            if name not in batch_features:
                                batch_features[name] = []
                            
                            # Ensure tensor is on correct device and has right shape
                            if isinstance(tensor, torch.Tensor):
                                batch_features[name].append(tensor.detach())
                            else:
                                batch_features[name].append(torch.tensor(tensor))
                        
                        batch_labels.append(label)
                    
                    # Stack tensors
                    stacked_features = {}
                    for name in batch_features:
                        try:
                            stacked_features[name] = torch.stack(batch_features[name])
                        except Exception as e:
                            logger.warning(f"Could not stack {name} tensors: {str(e)}")
                            # Use first tensor with proper batch dimension
                            stacked_features[name] = batch_features[name][0].unsqueeze(0)
                    
                    batch_labels_tensor = torch.tensor(batch_labels, dtype=torch.long)
                    
                    # Forward pass
                    self.optimizer.zero_grad()
                    logits, confidences = self.model(stacked_features)
                    
                    # Calculate loss
                    loss = F.cross_entropy(logits, batch_labels_tensor)
                    
                    if batch_idx == 0:
                        initial_loss = loss.item()
                    final_loss = loss.item()
                    
                    # Backward pass
                    loss.backward()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    # Update weights
                    self.optimizer.step()
                    
                    batches_completed += 1
                    
                    # Log progress every 10 batches
                    if (batch_idx + 1) % 10 == 0:
                        avg_confidence = torch.mean(confidences).item()
                        logger.info(f"Warmup batch {batch_idx + 1}/{max_batches}: loss={loss.item():.4f}, avg_confidence={avg_confidence:.4f}")
                    
                    # Early stopping if loss becomes very low
                    if loss.item() < 0.1:
                        logger.info(f"Early stopping warmup at batch {batch_idx + 1} due to low loss")
                        break
                        
                except Exception as e:
                    logger.error(f"Error in warmup batch {batch_idx}: {str(e)}")
                    continue
            
            # Restore original learning rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = original_lr
            
            # Set model back to eval mode
            self.model.eval()
            
            warmup_results = {
                'batches_completed': batches_completed,
                'initial_loss': initial_loss,
                'final_loss': final_loss,
                'loss_improvement': (initial_loss - final_loss) if initial_loss and final_loss else 0
            }
            
            if batches_completed > 0:
                logger.info(f"🚀 Warmup training completed: {batches_completed} batches, loss: {initial_loss:.4f} → {final_loss:.4f}")
            else:
                logger.warning("⚠️ No warmup batches completed - model may have stuck confidence")
            
            return warmup_results
            
        except Exception as e:
            logger.error(f"Error during warmup training: {str(e)}")
            return {'batches_completed': 0, 'final_loss': None, 'error': str(e)}
    
    def check_prediction_diversity(self, recent_predictions, threshold=0.01):
        """
        Check if the model predictions are too similar (stuck confidence issue)
        
        Args:
            recent_predictions: List of recent prediction tensors or confidences
            threshold: Minimum variance required for healthy diversity
            
        Returns:
            Dictionary with diversity analysis
        """
        try:
            if len(recent_predictions) < 5:
                return {'diverse': True, 'reason': 'insufficient_data', 'variance': None}
            
            # Extract confidence values
            confidences = []
            logits_std = []
            
            for pred in recent_predictions:
                if isinstance(pred, tuple) and len(pred) == 2:
                    # (probs, confidence) tuple
                    probs, confidence = pred
                    confidences.append(confidence.item() if hasattr(confidence, 'item') else float(confidence))
                    
                    # Calculate standard deviation of logits (before softmax)
                    if hasattr(probs, 'std'):
                        logits_std.append(probs.std().item())
                    else:
                        logits_std.append(np.std(probs) if hasattr(probs, '__iter__') else 0.0)
                        
                elif isinstance(pred, (int, float)):
                    confidences.append(float(pred))
                    logits_std.append(0.0)
                else:
                    # Try to extract confidence from tensor
                    if hasattr(pred, 'max'):
                        conf = pred.max().item() if hasattr(pred.max(), 'item') else float(pred.max())
                        confidences.append(conf)
                        std_val = pred.std().item() if hasattr(pred.std(), 'item') else float(pred.std())
                        logits_std.append(std_val)
                    else:
                        confidences.append(0.5)  # Default
                        logits_std.append(0.0)
            
            # Calculate variance metrics
            confidence_std = np.std(confidences) if len(confidences) > 1 else 0.0
            avg_logits_std = np.mean(logits_std) if logits_std else 0.0
            confidence_range = max(confidences) - min(confidences) if confidences else 0.0
            
            # Determine if predictions are diverse enough
            is_diverse = (
                confidence_std > threshold or 
                avg_logits_std > threshold or
                confidence_range > threshold * 3
            )
            
            diversity_info = {
                'diverse': is_diverse,
                'confidence_std': confidence_std,
                'logits_std': avg_logits_std,
                'confidence_range': confidence_range,
                'threshold': threshold,
                'sample_count': len(confidences),
                'avg_confidence': np.mean(confidences) if confidences else 0.0
            }
            
            if not is_diverse:
                diversity_info['reason'] = 'low_variance'
                logger.info(f"Prediction diversity check: logits_std={avg_logits_std:.6f}, confidence_std={confidence_std:.6f} (threshold={threshold})")
            
            return diversity_info
            
        except Exception as e:
            logger.error(f"Error checking prediction diversity: {str(e)}")
            return {'diverse': True, 'reason': 'error', 'error': str(e)}
    
    def start(self):
        """Start the online learning process"""
        if self.running:
            logger.warning("Online learner is already running")
            return
        
        self.running = True
        
        # Start update thread
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        
        logger.info("Online learner started")
    
    def stop(self):
        """Stop the online learning process"""
        self.running = False
        logger.info("Online learner stopped")
    
    def add_experience(self, features, label, confidence=None, feature_contributions=None):
        """
        Add experience to buffer with enhanced data
        
        Args:
            features: Dictionary with feature tensors
            label: Ground truth label (0=buy, 1=sell, 2=hold)
            confidence: Model confidence for this prediction
            feature_contributions: Dictionary showing which features contributed most
        """
        try:
            # Convert tensors to numpy arrays for storage
            features_numpy = {}
            for name, tensor in features.items():
                features_numpy[name] = tensor.detach().cpu().numpy()
            
            # Add to buffer
            self.experience_buffer['features'].append(features_numpy)
            self.experience_buffer['labels'].append(label)
            self.experience_buffer['timestamps'].append(datetime.now())
            self.experience_buffer['confidences'].append(confidence if confidence is not None else 0.5)
            self.experience_buffer['outcomes'].append(None)  # Will be updated when trade completes
            self.experience_buffer['feature_contributions'].append(feature_contributions)
            
            # Remove oldest experiences if buffer is full
            if len(self.experience_buffer['labels']) > self.buffer_size:
                for key in self.experience_buffer:
                    self.experience_buffer[key].pop(0)
            
            logger.debug(f"Added experience to buffer (size: {len(self.experience_buffer['labels'])})")
        except Exception as e:
            logger.error(f"Error adding experience to buffer: {str(e)}")
    
    def update_trade_outcome(self, experience_index, profit_loss, was_successful):
        """
        Update trade outcome for learning from results
        
        Args:
            experience_index: Index of experience in buffer (or use timestamp matching)
            profit_loss: Actual profit/loss from trade
            was_successful: Boolean indicating if trade was successful
        """
        try:
            if 0 <= experience_index < len(self.experience_buffer['outcomes']):
                # Update outcome
                self.experience_buffer['outcomes'][experience_index] = {
                    'profit_loss': profit_loss,
                    'successful': was_successful
                }
                
                # Update global statistics
                if was_successful:
                    self.trade_outcomes['successful_trades'] += 1
                    self.trade_outcomes['total_profit'] += max(0, profit_loss)
                else:
                    self.trade_outcomes['failed_trades'] += 1
                    self.trade_outcomes['total_loss'] += abs(min(0, profit_loss))  
                
                # Update feature success rates if we have contribution data
                contributions = self.experience_buffer['feature_contributions'][experience_index]
                if contributions:
                    for feature_name, contribution in contributions.items():
                        if feature_name not in self.trade_outcomes['feature_success_rates']:
                            self.trade_outcomes['feature_success_rates'][feature_name] = {
                                'successful': 0, 'total': 0, 'avg_contribution': 0.0
                            }
                        
                        rates = self.trade_outcomes['feature_success_rates'][feature_name]
                        rates['total'] += 1
                        if was_successful:
                            rates['successful'] += 1
                        
                        # Update average contribution with exponential moving average
                        rates['avg_contribution'] = 0.9 * rates['avg_contribution'] + 0.1 * contribution
                
                # Update gating module if available
                if self.gating_module and contributions:
                    for feature_name, contribution in contributions.items():
                        self.gating_module.update_feature_performance(
                            feature_name, was_successful, contribution
                        )
                
                logger.info(f"Updated trade outcome: {'Success' if was_successful else 'Failure'}, P&L: {profit_loss:.4f}")
            
        except Exception as e:
            logger.error(f"Error updating trade outcome: {str(e)}")
    
    def get_learning_summary(self):
        """
        Get summary of learning progress
        
        Returns:
            Dictionary with learning statistics
        """
        total_trades = self.trade_outcomes['successful_trades'] + self.trade_outcomes['failed_trades']
        success_rate = (self.trade_outcomes['successful_trades'] / total_trades) if total_trades > 0 else 0.0
        
        # Calculate feature performance
        feature_performance = {}
        for feature_name, rates in self.trade_outcomes['feature_success_rates'].items():
            if rates['total'] > 0:
                feature_performance[feature_name] = {
                    'success_rate': rates['successful'] / rates['total'],
                    'avg_contribution': rates['avg_contribution'],
                    'total_trades': rates['total']
                }
        
        return {
            'total_trades': total_trades,
            'success_rate': success_rate,
            'total_profit': self.trade_outcomes['total_profit'],
            'total_loss': self.trade_outcomes['total_loss'],
            'net_profit': self.trade_outcomes['total_profit'] - self.trade_outcomes['total_loss'],
            'model_updates': self.updates_counter,
            'buffer_size': len(self.experience_buffer['labels']),
            'feature_performance': feature_performance
        }
    
    def _update_loop(self):
        """Main update loop"""
        while self.running:
            try:
                # Check if it's time to update
                time_since_update = (datetime.now() - self.last_update_time).total_seconds()
                
                if time_since_update >= self.update_interval and len(self.experience_buffer['labels']) >= self.batch_size:
                    self._perform_update()
                    self.last_update_time = datetime.now()
                
                # Sleep to prevent high CPU usage
                time.sleep(1)
            
            except Exception as e:
                logger.error(f"Error in update loop: {str(e)}")
                time.sleep(10)
    
    def _perform_update(self):
        """Perform enhanced model update with trade outcome weighting"""
        try:
            # Check if we have enough data
            if len(self.experience_buffer['labels']) < self.batch_size:
                logger.warning("Not enough data for update")
                return
            
            # Sample batch with preference for experiences with known outcomes
            indices_with_outcomes = [i for i, outcome in enumerate(self.experience_buffer['outcomes']) if outcome is not None]
            indices_without_outcomes = [i for i, outcome in enumerate(self.experience_buffer['outcomes']) if outcome is None]
            
            # Prefer samples with outcomes for learning
            if len(indices_with_outcomes) >= self.batch_size // 2:
                # Use half with outcomes, half without
                n_with_outcomes = min(len(indices_with_outcomes), self.batch_size // 2)
                n_without_outcomes = self.batch_size - n_with_outcomes
                
                batch_indices = (
                    np.random.choice(indices_with_outcomes, size=n_with_outcomes, replace=False).tolist() +
                    np.random.choice(indices_without_outcomes, size=n_without_outcomes, replace=False).tolist()
                )
            else:
                # Sample normally if not enough outcome data
                batch_indices = np.random.choice(len(self.experience_buffer['labels']), 
                                            size=self.batch_size, 
                                            replace=False)
            
            # Prepare batch
            batch_features = {}
            batch_labels = []
            batch_weights = []  # Weight samples based on outcomes
            
            for idx in batch_indices:
                # Convert numpy back to tensors
                for name, arr in self.experience_buffer['features'][idx].items():
                    if name not in batch_features:
                        batch_features[name] = []
                    
                    batch_features[name].append(torch.tensor(arr))
                
                batch_labels.append(self.experience_buffer['labels'][idx])
                
                # Calculate sample weight based on outcome
                outcome = self.experience_buffer['outcomes'][idx]
                if outcome is not None:
                    # Weight successful trades more heavily
                    if outcome['successful']:
                        weight = 1.0 + abs(outcome['profit_loss']) * 10  # Boost successful trades
                    else:
                        weight = 0.5 + abs(outcome['profit_loss']) * 5   # Reduce failed trades but keep for learning
                else:
                    weight = 1.0  # Default weight for unknown outcomes
                
                batch_weights.append(weight)
            
            # Stack tensors
            for name in batch_features:
                try:
                    batch_features[name] = torch.stack(batch_features[name])
                except:
                    # If tensors have different shapes, just use the first one
                    logger.warning(f"Could not stack tensors for {name}, using only first tensor")
                    batch_features[name] = batch_features[name][0].unsqueeze(0)
            
            batch_labels = torch.tensor(batch_labels)
            batch_weights = torch.tensor(batch_weights, dtype=torch.float32)
            
            # Train model
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, _ = self.model(batch_features)
            
            # Calculate weighted loss
            loss = F.cross_entropy(logits, batch_labels, reduction='none')
            weighted_loss = torch.mean(loss * batch_weights)
            
            # Backward pass
            weighted_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update statistics
            self.updates_counter += 1
            
            # Adaptive learning rate based on performance
            success_rate = self.trade_outcomes['successful_trades'] / max(1, 
                self.trade_outcomes['successful_trades'] + self.trade_outcomes['failed_trades'])
            
            if success_rate > 0.6:
                # Reduce learning rate when performing well
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = max(param_group['lr'] * 0.99, self.learning_rate_schedule * 0.1)
            elif success_rate < 0.4:
                # Increase learning rate when performing poorly
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = min(param_group['lr'] * 1.01, self.learning_rate_schedule * 2.0)
            
            # Save model periodically
            if self.updates_counter % 10 == 0:
                self._save_model()
            
            logger.info(f"Enhanced model update performed (updates: {self.updates_counter}, "
                       f"loss: {weighted_loss.item():.4f}, success_rate: {success_rate:.3f})")
        
        except Exception as e:
            logger.error(f"Error performing update: {str(e)}")
    
    def _increment_version(self, version_type='patch'):
        """
        Increment model version using semantic versioning
        
        Args:
            version_type: 'major', 'minor', or 'patch'
        """
        try:
            major, minor, patch = map(int, self.model_version.split('.'))
            
            if version_type == 'major':
                major += 1
                minor = 0
                patch = 0
                self.last_major_update = datetime.now()
                logger.info(f"🎯 Major model improvement! Version bump: {self.model_version} → {major}.{minor}.{patch}")
            elif version_type == 'minor':
                minor += 1
                patch = 0
                logger.info(f"📈 Minor model improvement! Version bump: {self.model_version} → {major}.{minor}.{patch}")
            else:  # patch
                patch += 1
                logger.debug(f"🔧 Model patch update: {self.model_version} → {major}.{minor}.{patch}")
            
            # Update version and track in history
            old_version = self.model_version
            self.model_version = f"{major}.{minor}.{patch}"
            self.version_history.append({
                'from_version': old_version,
                'to_version': self.model_version,
                'timestamp': datetime.now().isoformat(),
                'update_type': version_type,
                'updates_count': self.updates_counter
            })
            
            # Save version history to external JSON file
            self._save_version_history_to_file()
            
        except Exception as e:
            logger.error(f"Error incrementing version: {str(e)}")
    
    def _save_version_history_to_file(self):
        """Save version history to external JSON file with atomic write and thread safety"""
        try:
            
            # Use lock for thread safety
            with version_lock:
                version_file = 'version_history.json'
                
                # Read existing history if file exists
                existing_history = []
                try:
                    with open(version_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                        existing_history = existing_data.get('history', [])
                except (FileNotFoundError, json.JSONDecodeError):
                    pass
                
                # Merge with current history, avoiding duplicates
                combined_history = existing_history.copy()
                for entry in self.version_history:
                    if entry not in combined_history:
                        combined_history.append(entry)
                
                # Keep only last 20 entries
                combined_history = combined_history[-20:]
                
                version_data = {
                    'schema_version': 1,
                    'current_version': self.model_version,
                    'last_updated': datetime.now().isoformat(),
                    'total_entries': len(combined_history),
                    'history': combined_history
                }
                
                # Atomic write using tempfile and os.replace
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
                    json.dump(version_data, tmp_file, indent=2, ensure_ascii=False)
                    tmp_filename = tmp_file.name
                
                # Atomic replacement
                os.replace(tmp_filename, version_file)
                logger.info(f"Version history updated atomically (entries={len(combined_history)})")
            
        except Exception as e:
            logger.error(f"Failed to save version history: {str(e)}")
            # Clean up temp file if it exists
            try:
                if 'tmp_filename' in locals():
                    os.unlink(tmp_filename)
            except:
                pass
    
    def _should_increment_version(self):
        """
        Check if model version should be incremented based on performance
        
        Returns:
            String indicating version increment type or None
        """
        if len(self.performance_window) < 10:
            return None
        
        # Calculate recent performance trend
        recent_performance = np.mean(self.performance_window[-5:])
        older_performance = np.mean(self.performance_window[-10:-5])
        
        if older_performance == 0:
            return None
        
        improvement = (recent_performance - older_performance) / abs(older_performance)
        
        # Check for major improvement (RFE results, significant accuracy gain)
        if (improvement >= self.performance_threshold_major or 
            (self.gating_module and self.gating_module.rfe_performed and 
             (datetime.now() - self.last_major_update).days >= 1)):
            return 'major'
        
        # Check for minor improvement
        elif improvement >= self.performance_threshold_minor:
            return 'minor'
        
        # Default patch increment every few updates
        elif self.updates_counter % 10 == 0:
            return 'patch'
        
        return None

    def _save_model(self):
        """Save model checkpoint with semantic versioning and learning progress"""
        try:
            # Check if version should be incremented
            version_increment = self._should_increment_version()
            if version_increment:
                self._increment_version(version_increment)
            
            # Create checkpoint filename with version
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_filename = f"model_v{self.model_version}_{timestamp}.pt"
            checkpoint_path = os.path.join(self.save_dir, checkpoint_filename)
            
            # Also create a "latest" symlink/copy
            latest_path = os.path.join(self.save_dir, "model_latest.pt")
            
            # Save model state with enhanced metadata
            checkpoint_data = {
                'model_version': self.model_version,
                'version_history': self.version_history,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'updates_counter': self.updates_counter,
                'timestamp': datetime.now().isoformat(),
                'trade_outcomes': self.trade_outcomes,
                'learning_summary': self.get_learning_summary(),
                'gating_performance': self.gating_module.get_feature_performance_summary() if self.gating_module else {},
                'rfe_results': {
                    'rfe_performed': self.gating_module.rfe_performed if self.gating_module else False,
                    'selected_features': self.gating_module.rfe_selected_features if self.gating_module else {},
                    'rfe_summary': self.gating_module.get_rfe_summary() if self.gating_module else {}
                },
                'performance_metrics': {
                    'recent_performance': self.performance_window[-10:] if len(self.performance_window) >= 10 else self.performance_window,
                    'success_rate': self.trade_outcomes['successful_trades'] / max(1, self.trade_outcomes['successful_trades'] + self.trade_outcomes['failed_trades']) * 100,
                    'total_profit_loss': self.trade_outcomes['total_profit'] - self.trade_outcomes['total_loss']
                }
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            torch.save(checkpoint_data, latest_path)  # Also save as latest
            
            # Keep only the latest 10 versioned checkpoints
            checkpoint_files = [f for f in os.listdir(self.save_dir) if f.startswith('model_v') and f.endswith('.pt')]
            if len(checkpoint_files) > 10:
                # Sort by modification time
                checkpoint_files_with_time = [(f, os.path.getmtime(os.path.join(self.save_dir, f))) 
                                            for f in checkpoint_files]
                checkpoint_files_with_time.sort(key=lambda x: x[1])
                
                # Remove oldest files
                for old_file, _ in checkpoint_files_with_time[:-10]:
                    try:
                        os.remove(os.path.join(self.save_dir, old_file))
                    except:
                        pass
            
            logger.info(f"✅ Model v{self.model_version} saved to {checkpoint_filename}")
            logger.info(f"📊 Updates: {self.updates_counter}, Success Rate: {checkpoint_data['performance_metrics']['success_rate']:.1f}%")
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def get_version_info(self):
        """
        Get detailed version information for API endpoints
        
        Returns:
            Dictionary with version details
        """
        return {
            'current_version': self.model_version,
            'updates_count': self.updates_counter,
            'version_history': self.version_history[-5:],  # Last 5 versions
            'last_major_update': self.last_major_update.isoformat(),
            'total_versions': len(self.version_history)
        }
    
    def get_training_accuracy(self):
        """
        Calculate training accuracy based on trade outcomes
        
        Returns:
            Float: Training accuracy (0.0 to 1.0)
        """
        total_trades = self.trade_outcomes['successful_trades'] + self.trade_outcomes['failed_trades']
        if total_trades == 0:
            return 0.0
        return self.trade_outcomes['successful_trades'] / total_trades
    
    def get_validation_accuracy(self):
        """
        Calculate validation accuracy based on recent performance window
        
        Returns:
            Float: Validation accuracy (0.0 to 1.0)
        """
        if not self.performance_window:
            return 0.0
        # Use mean of recent performance as validation accuracy
        return max(0.0, min(1.0, np.mean(self.performance_window)))
    
    def get_model_stats(self):
        """
        Get comprehensive model statistics for API endpoints
        
        Returns:
            Dictionary with model statistics
        """
        return {
            'training_accuracy': self.get_training_accuracy(),
            'validation_accuracy': self.get_validation_accuracy(),
            'model_version': self.model_version,
            'updates_count': self.updates_counter,
            'total_trades': self.trade_outcomes['successful_trades'] + self.trade_outcomes['failed_trades'],
            'successful_trades': self.trade_outcomes['successful_trades'],
            'failed_trades': self.trade_outcomes['failed_trades'],
            'total_profit': self.trade_outcomes['total_profit'],
            'total_loss': abs(self.trade_outcomes['total_loss']),
            'performance_trend': np.mean(self.performance_window[-5:]) if len(self.performance_window) >= 5 else 0.0,
            'last_update': self.last_update_time.isoformat() if hasattr(self, 'last_update_time') else None
        }
    
    def load_model(self, checkpoint_path):
        """
        Load model checkpoint with learning progress and version info
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Boolean indicating success
        """
        try:
            checkpoint = torch.load(checkpoint_path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.updates_counter = checkpoint.get('updates_counter', 0)
            
            # Load version information
            self.model_version = checkpoint.get('model_version', '1.0.0')
            self.version_history = checkpoint.get('version_history', [])
            
            # Load learning progress if available
            if 'trade_outcomes' in checkpoint:
                self.trade_outcomes = checkpoint['trade_outcomes']
            
            # Load RFE results if available
            if 'rfe_results' in checkpoint and self.gating_module:
                rfe_data = checkpoint['rfe_results']
                if rfe_data.get('rfe_performed', False):
                    self.gating_module.rfe_performed = True
                    self.gating_module.rfe_selected_features = rfe_data.get('selected_features', {})
            
            # Load performance metrics
            if 'performance_metrics' in checkpoint:
                perf_data = checkpoint['performance_metrics']
                self.performance_window = perf_data.get('recent_performance', [])
            
            logger.info(f"✅ Model v{self.model_version} loaded from {os.path.basename(checkpoint_path)}")
            logger.info(f"📈 Updates: {self.updates_counter}, RFE: {'✓' if self.gating_module and self.gating_module.rfe_performed else '✗'}")
            
            return True
        
        except Exception as e:
            logger.error(f"Error loading model from {checkpoint_path}: {str(e)}")
            return False
    
    def get_version_info(self):
        """
        Get detailed version information
        
        Returns:
            Dictionary with version details
        """
        return {
            'current_version': self.model_version,
            'updates_count': self.updates_counter,
            'version_history': self.version_history,
            'last_major_update': self.last_major_update.isoformat() if hasattr(self, 'last_major_update') else None,
            'rfe_performed': self.gating_module.rfe_performed if self.gating_module else False,
            'performance_trend': {
                'recent_avg': np.mean(self.performance_window[-5:]) if len(self.performance_window) >= 5 else 0,
                'overall_avg': np.mean(self.performance_window) if self.performance_window else 0,
                'data_points': len(self.performance_window)
            }
        }