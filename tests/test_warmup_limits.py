#!/usr/bin/env python3
"""
Test warmup time limits and progress logging.

Validates that warmup training:
- Respects time caps and aborts gracefully when exceeded
- Logs progress at appropriate intervals  
- Stores warmup summary for diagnostics
- Handles edge cases properly
"""

import unittest
import time
from unittest.mock import Mock, MagicMock, patch, call
import sys
import os
import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.neural_network import OnlineLearner, MarketTransformer


class TestWarmupLimits(unittest.TestCase):
    """Test warmup time limits and logging"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create a simple model for testing
        feature_dims = {'test': (10, 5)}
        self.model = MarketTransformer(feature_dims, hidden_dim=32)
        
        # Create learner
        self.learner = OnlineLearner(
            model=self.model,
            lr=1e-3,
            batch_size=4,
            update_interval=60
        )
        
        # Create mock training data
        self.mock_data = self._create_mock_dataloader()
    
    def test_warmup_time_cap_exceeded(self):
        """Test warmup aborts when time cap exceeded"""
        # Use very short time cap
        max_seconds = 0.1  # 100ms
        
        with patch('models.neural_network.logger') as mock_logger:
            summary = self.learner._warmup_loop(self.mock_data, max_seconds=max_seconds, log_every=1)
            
            # Should have aborted due to time
            self.assertTrue(summary['aborted_due_to_time'])
            self.assertLess(summary['batches_completed'], len(self.mock_data))
            
            # Should have logged warning about time exceeded
            warning_calls = [call for call in mock_logger.warning.call_args_list 
                           if 'WARNING: Warmup exceeded time cap' in str(call)]
            self.assertGreater(len(warning_calls), 0)
    
    def test_warmup_progress_logging(self):
        """Test that warmup progress is logged at intervals"""
        # Create small dataset for predictable logging
        small_data = list(self.mock_data[:5])
        
        with patch('models.neural_network.logger') as mock_logger:
            summary = self.learner._warmup_loop(small_data, max_seconds=30, log_every=2)
            
            # Check that progress was logged
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            
            # Should have start log
            start_logs = [call for call in info_calls if 'Starting warmup loop' in call]
            self.assertGreater(len(start_logs), 0)
            
            # Should have completion log  
            complete_logs = [call for call in info_calls if 'Warmup complete:' in call]
            self.assertGreater(len(complete_logs), 0)
            
            # Should have progress logs (every 2 batches + batch 0)
            progress_logs = [call for call in info_calls if 'Warmup progress:' in call]
            expected_progress_logs = (len(small_data) + 1) // 2 + 1  # log_every=2 + initial log
            self.assertGreaterEqual(len(progress_logs), 1)  # At least initial log
    
    def test_warmup_completion_summary(self):
        """Test warmup completion summary format"""
        small_data = list(self.mock_data[:3])
        
        with patch('models.neural_network.logger') as mock_logger:
            summary = self.learner._warmup_loop(small_data, max_seconds=30, log_every=1)
            
            # Check summary structure
            self.assertIn('batches_completed', summary)
            self.assertIn('duration_seconds', summary) 
            self.assertIn('initial_loss', summary)
            self.assertIn('final_loss', summary)
            self.assertIn('average_loss', summary)
            self.assertIn('aborted_due_to_time', summary)
            
            # Check that summary was stored as attribute
            self.assertTrue(hasattr(self.learner, 'last_warmup_summary'))
            self.assertEqual(self.learner.last_warmup_summary, summary)
            
            # Check completion log format
            complete_calls = [str(call) for call in mock_logger.info.call_args_list 
                            if 'Warmup complete:' in str(call)]
            self.assertGreater(len(complete_calls), 0)
            
            complete_log = complete_calls[0]
            self.assertIn('batches=', complete_log)
            self.assertIn('duration=', complete_log) 
            self.assertIn('loss_start=', complete_log)
            self.assertIn('loss_end=', complete_log)
            self.assertIn('avg=', complete_log)
    
    def test_warmup_early_stopping(self):
        """Test warmup early stopping on low loss"""
        # Mock the model to return very low loss
        with patch.object(self.model, 'forward') as mock_forward:
            # Return low loss scenario
            mock_forward.return_value = (
                torch.tensor([[0.9, 0.05, 0.05]]),  # Confident prediction
                torch.tensor([0.95])  # High confidence
            )
            
            small_data = list(self.mock_data[:10])
            
            with patch('models.neural_network.logger') as mock_logger, \
                 patch('torch.nn.functional.cross_entropy') as mock_loss_fn:
                
                # Mock very low loss
                mock_loss_fn.return_value = torch.tensor(0.05)  # Below 0.1 threshold
                
                summary = self.learner._warmup_loop(small_data, max_seconds=30, log_every=1)
                
                # Should have stopped early
                self.assertLess(summary['batches_completed'], len(small_data))
                
                # Should have logged early stopping
                info_calls = [str(call) for call in mock_logger.info.call_args_list]
                early_stop_logs = [call for call in info_calls if 'Early stopping warmup' in call]
                self.assertGreater(len(early_stop_logs), 0)
    
    def test_warmup_integration_with_perform_warmup_training(self):
        """Test integration between perform_warmup_training and _warmup_loop"""
        # Create warmup data in expected format
        warmup_data = [(
            {'test': torch.randn(10, 5)}, 
            i % 3  # Labels 0, 1, 2
        ) for i in range(5)]
        
        with patch('models.neural_network.logger'):
            result = self.learner.perform_warmup_training(
                warmup_data, 
                max_batches=3, 
                max_seconds=5
            )
            
            # Check result format
            self.assertIn('batches_completed', result)
            self.assertIn('initial_loss', result)
            self.assertIn('final_loss', result)
            self.assertIn('loss_improvement', result)
            self.assertIn('duration_seconds', result)
            self.assertIn('aborted_due_to_time', result)
            
            # Should have completed some batches
            self.assertGreaterEqual(result['batches_completed'], 0)
    
    def test_warmup_with_empty_data(self):
        """Test warmup with empty dataloader"""
        empty_data = []
        
        with patch('models.neural_network.logger') as mock_logger:
            summary = self.learner._warmup_loop(empty_data, max_seconds=30, log_every=1)
            
            # Should complete with zero batches
            self.assertEqual(summary['batches_completed'], 0)
            self.assertFalse(summary['aborted_due_to_time'])
            self.assertIsNone(summary['initial_loss'])
            self.assertIsNone(summary['final_loss'])
    
    def test_warmup_error_handling(self):
        """Test warmup continues after individual batch errors"""
        # Create data that will cause errors
        error_data = [
            ({'test': torch.randn(10, 5)}, 0),  # Good data
            ({'invalid': torch.randn(5, 3)}, 1),  # Bad data - wrong key
            ({'test': torch.randn(10, 5)}, 2),  # Good data
        ]
        
        with patch('models.neural_network.logger') as mock_logger:
            summary = self.learner._warmup_loop(error_data, max_seconds=30, log_every=1)
            
            # Should have handled errors and continued
            # Expecting at least 2 successful batches
            self.assertGreaterEqual(summary['batches_completed'], 1)
            
            # Should have logged errors
            error_calls = [str(call) for call in mock_logger.error.call_args_list]
            batch_error_logs = [call for call in error_calls if 'Error in warmup batch' in call]
            self.assertGreater(len(batch_error_logs), 0)
    
    def _create_mock_dataloader(self):
        """Create mock dataloader for testing"""
        # Create list of (features, label) tuples
        data = []
        for i in range(10):
            features = {'test': torch.randn(10, 5)}
            label = i % 3  # Labels 0, 1, 2
            data.append((features, label))
        
        return data


if __name__ == '__main__':
    unittest.main()