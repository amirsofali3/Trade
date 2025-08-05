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

logger = logging.getLogger("NeuralNetwork")

class MarketTransformer(nn.Module):
    """
    Transformer-based model for market prediction with context-aware feature gating
    """
    
    def __init__(self, feature_dims, hidden_dim=128, n_layers=2, n_heads=4, dropout=0.1):
        """
        Initialize Market Transformer
        
        Args:
            feature_dims: Dictionary with feature dimensions
                Example: {'ohlcv': (20, 13), 'indicator': (20, 30), 'sentiment': (1, 1), 'orderbook': (1, 42)}
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
        
        # Feature embedding layers
        self.embeddings = nn.ModuleDict()
        for name, (seq_len, dim) in feature_dims.items():
            self.embeddings[name] = nn.Linear(dim, hidden_dim)
        
        # Position encoding
        max_seq_len = max(dim[0] for dim in feature_dims.values())
        self.position_encoding = self._create_position_encoding(max_seq_len, hidden_dim)
        
        # Create transformers for each feature type and a merged transformer
        self.transformers = nn.ModuleDict()
        
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
                
                # Embed features to hidden_dim
                if name in ['ohlcv', 'indicator', 'tick_data']:
                    # Sequential features: embed each timestep
                    # x shape: (1, seq_len, feature_dim)
                    batch_size, seq_len, feature_dim = x.shape
                    x_flat = x.view(-1, feature_dim)  # (batch_size * seq_len, feature_dim)
                    embedded_flat = self.embeddings[name](x_flat)  # (batch_size * seq_len, hidden_dim)
                    embedded = embedded_flat.view(batch_size, seq_len, self.hidden_dim)
                else:
                    # Non-sequential features: embed directly
                    # x shape: (1, feature_dim)
                    embedded = self.embeddings[name](x)
                    # Expand to sequence length for compatibility
                    embedded = embedded.unsqueeze(1)  # (1, 1, hidden_dim)
                
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
                last_timestep = feat[:, -1:, :].expand(-1, max_seq_len - feat.shape[1], -1)
                feat_expanded = torch.cat([feat, last_timestep], dim=1)
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
    Enhanced online learning mechanism with trade outcome feedback
    
    Features:
    - Learns from trade outcomes (profit/loss, accuracy)
    - Adjusts feature weights based on success rates
    - Implements continuous model improvement
    - Saves learning progress for recovery
    """
    
    def __init__(self, model, gating_module=None, optimizer_cls=torch.optim.Adam, 
                lr=1e-4, buffer_size=1000, batch_size=32,
                update_interval=3600, save_dir='saved_models'):
        """
        Initialize enhanced online learner
        
        Args:
            model: Neural network model
            gating_module: Feature gating module for feedback
            optimizer_cls: Optimizer class
            lr: Learning rate
            buffer_size: Maximum size of experience buffer
            batch_size: Batch size for training
            update_interval: Time interval between updates in seconds
            save_dir: Directory to save model checkpoints
        """
        self.model = model
        self.gating_module = gating_module
        self.optimizer = optimizer_cls(model.parameters(), lr=lr)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.save_dir = save_dir
        
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
        
        # Make sure save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Start update thread
        self.running = False
        self.update_thread = None
        
        logger.info(f"Initialized Enhanced OnlineLearner with feedback learning")
        logger.info(f"Buffer size: {buffer_size}, Batch size: {batch_size}, Update interval: {update_interval}s")
    
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
    
    def _save_model(self):
        """Save model checkpoint with learning progress"""
        try:
            checkpoint_path = os.path.join(self.save_dir, f"model_checkpoint_{self.updates_counter}.pt")
            
            # Save model state with learning data
            checkpoint_data = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'updates_counter': self.updates_counter,
                'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S'),
                'trade_outcomes': self.trade_outcomes,
                'learning_summary': self.get_learning_summary(),
                'gating_performance': self.gating_module.get_feature_performance_summary() if self.gating_module else {}
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            
            # Keep only the latest 5 checkpoints
            checkpoint_files = [f for f in os.listdir(self.save_dir) if f.startswith('model_checkpoint_')]
            if len(checkpoint_files) > 5:
                checkpoint_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
                for old_file in checkpoint_files[:-5]:
                    try:
                        os.remove(os.path.join(self.save_dir, old_file))
                    except:
                        pass
            
            logger.info(f"Enhanced model saved to {checkpoint_path}")
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, checkpoint_path):
        """
        Load model checkpoint with learning progress
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.updates_counter = checkpoint['updates_counter']
            
            # Load learning progress if available
            if 'trade_outcomes' in checkpoint:
                self.trade_outcomes = checkpoint['trade_outcomes']
                logger.info(f"Loaded learning progress: {self.get_learning_summary()}")
            
            # Load gating performance if available
            if 'gating_performance' in checkpoint and self.gating_module:
                gating_perf = checkpoint['gating_performance']
                logger.info(f"Loaded feature performance data for {len(gating_perf)} features")
            
            logger.info(f"Enhanced model loaded from {checkpoint_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False