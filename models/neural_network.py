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
        
        # Process and embed each feature type
        embedded_features = []
        
        for name, (seq_len, _) in self.feature_dims.items():
            if name in features_dict:
                # Get features and ensure proper shape
                x = features_dict[name]
                
                # Handle different input formats
                if len(x.shape) == 1:
                    x = x.unsqueeze(0)  # Add batch dimension
                
                if x.shape[0] > 1 and name != 'ohlcv' and name != 'indicator':
                    # Average over batch dimension except for sequential features
                    x = torch.mean(x, dim=0, keepdim=True)
                
                # Embed features to hidden_dim
                embedded = self.embeddings[name](x)
                
                # Add position encoding for sequential features
                if name == 'ohlcv' or name == 'indicator':
                    seq_len_actual = min(embedded.shape[1], self.position_encoding.shape[1])
                    embedded[:, :seq_len_actual, :] += self.position_encoding[:, :seq_len_actual, :]
                
                embedded_features.append(embedded)
        
        if not embedded_features:
            logger.warning("No valid features to process")
            return torch.tensor([[0.0, 0.0, 1.0]]), 0.0
        
        # Combine features
        # For sequential features (OHLCV, indicators) we concatenate along time dimension
        # For point features (sentiment, orderbook) we replicate to match sequence length
        
        # First, find the maximum sequence length among all features
        max_seq_len = max(feat.shape[1] if len(feat.shape) > 2 else 1 for feat in embedded_features)
        
        # Process each feature to make it compatible
        processed_features = []
        
        for feat in embedded_features:
            if len(feat.shape) <= 2:  # Point feature
                # Replicate to match sequence length
                feat_expanded = feat.expand(-1, max_seq_len, -1)
                processed_features.append(feat_expanded)
            else:  # Sequential feature
                if feat.shape[1] < max_seq_len:
                    # Pad with zeros
                    padding = torch.zeros(feat.shape[0], max_seq_len - feat.shape[1], feat.shape[2])
                    feat_padded = torch.cat([feat, padding], dim=1)
                    processed_features.append(feat_padded)
                else:
                    # Truncate to max_seq_len
                    processed_features.append(feat[:, :max_seq_len, :])
        
        # Concatenate all features
        x = torch.cat(processed_features, dim=0)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        
        # Pool over sequence dimension
        x = torch.mean(x, dim=1)
        
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
    Online learning mechanism for continually training the model
    """
    
    def __init__(self, model, optimizer_cls=torch.optim.Adam, 
                lr=1e-4, buffer_size=1000, batch_size=32,
                update_interval=3600, save_dir='saved_models'):
        """
        Initialize online learner
        
        Args:
            model: Neural network model
            optimizer_cls: Optimizer class
            lr: Learning rate
            buffer_size: Maximum size of experience buffer
            batch_size: Batch size for training
            update_interval: Time interval between updates in seconds
            save_dir: Directory to save model checkpoints
        """
        self.model = model
        self.optimizer = optimizer_cls(model.parameters(), lr=lr)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_interval = update_interval
        self.save_dir = save_dir
        
        # Create experience buffer
        self.experience_buffer = {
            'features': [],
            'labels': [],
            'timestamps': []
        }
        
        # Statistics
        self.updates_counter = 0
        self.last_update_time = datetime.now()
        
        # Make sure save directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Start update thread
        self.running = False
        self.update_thread = None
        
        logger.info(f"Initialized OnlineLearner with buffer_size={buffer_size}, batch_size={batch_size}")
    
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
    
    def add_experience(self, features, label):
        """
        Add experience to buffer
        
        Args:
            features: Dictionary with feature tensors
            label: Ground truth label (0=buy, 1=sell, 2=hold)
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
            
            # Remove oldest experiences if buffer is full
            if len(self.experience_buffer['labels']) > self.buffer_size:
                self.experience_buffer['features'].pop(0)
                self.experience_buffer['labels'].pop(0)
                self.experience_buffer['timestamps'].pop(0)
            
            logger.debug(f"Added experience to buffer (size: {len(self.experience_buffer['labels'])})")
        except Exception as e:
            logger.error(f"Error adding experience to buffer: {str(e)}")
    
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
        """Perform model update"""
        try:
            # Check if we have enough data
            if len(self.experience_buffer['labels']) < self.batch_size:
                logger.warning("Not enough data for update")
                return
            
            # Sample batch
            batch_indices = np.random.choice(len(self.experience_buffer['labels']), 
                                        size=self.batch_size, 
                                        replace=False)
            
            # Prepare batch
            batch_features = {}
            batch_labels = []
            
            for idx in batch_indices:
                # Convert numpy back to tensors
                for name, arr in self.experience_buffer['features'][idx].items():
                    if name not in batch_features:
                        batch_features[name] = []
                    
                    batch_features[name].append(torch.tensor(arr))
                
                batch_labels.append(self.experience_buffer['labels'][idx])
            
            # Stack tensors
            for name in batch_features:
                try:
                    batch_features[name] = torch.stack(batch_features[name])
                except:
                    # If tensors have different shapes, just use the first one
                    logger.warning(f"Could not stack tensors for {name}, using only first tensor")
                    batch_features[name] = batch_features[name][0].unsqueeze(0)
            
            batch_labels = torch.tensor(batch_labels)
            
            # Train model
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            logits, _ = self.model(batch_features)
            
            # Calculate loss
            loss = F.cross_entropy(logits, batch_labels)
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Update statistics
            self.updates_counter += 1
            
            # Save model periodically
            if self.updates_counter % 10 == 0:
                self._save_model()
            
            logger.info(f"Model update performed (updates: {self.updates_counter}, loss: {loss.item():.4f})")
        
        except Exception as e:
            logger.error(f"Error performing update: {str(e)}")
    
    def _save_model(self):
        """Save model checkpoint"""
        try:
            checkpoint_path = os.path.join(self.save_dir, f"model_checkpoint_{self.updates_counter}.pt")
            
            # Save model state
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'updates_counter': self.updates_counter,
                'timestamp': datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            }, checkpoint_path)
            
            logger.info(f"Model saved to {checkpoint_path}")
        
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, checkpoint_path):
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            checkpoint = torch.load(checkpoint_path)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.updates_counter = checkpoint['updates_counter']
            
            logger.info(f"Model loaded from {checkpoint_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False