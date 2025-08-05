#!/usr/bin/env python3
"""
Direct test to verify the 0.01 minimum weight constraint works properly
"""

import logging
import torch
import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MinWeightTest")

def test_minimum_weight_constraint():
    """Test that the torch.clamp ensures minimum weight of 0.01"""
    try:
        from models.gating import FeatureGatingModule
        
        logger.info("Testing torch.clamp minimum weight constraint...")
        
        # Create feature groups
        feature_groups = {
            'ohlcv': 13,
            'indicator': 5,  # Smaller for easier testing
            'sentiment': 1,
            'orderbook': 5,  # Smaller for easier testing
            'tick_data': 3
        }
        
        # Create gating module with 0.01 minimum weight
        gating = FeatureGatingModule(feature_groups, min_weight=0.01, adaptation_rate=0.1)
        
        # Create test features
        features = {
            'ohlcv': torch.randn(1, 20, 13),
            'indicator': torch.randn(1, 20, 5), 
            'sentiment': torch.randn(1, 1),
            'orderbook': torch.randn(1, 5),
            'tick_data': torch.randn(1, 100, 3)
        }
        
        # Manually create context tensor
        context = torch.randn(1, 27)  # Sum of all feature dimensions
        
        # Test the clamping directly
        logger.info("Testing torch.clamp with very negative values...")
        
        # Create extremely negative raw gate outputs (pre-clamp)
        raw_negative = torch.tensor([[-10.0, -15.0, -20.0, -25.0, -30.0]])
        
        # Apply the same clamp as in the forward method
        clamped = torch.clamp(raw_negative, min=gating.min_weight, max=1.0)
        
        logger.info(f"Raw negative values: {raw_negative[0].tolist()}")
        logger.info(f"After clamp(min=0.01): {clamped[0].tolist()}")
        
        # Verify all values are exactly 0.01
        all_at_minimum = torch.all(clamped == gating.min_weight)
        logger.info(f"All values clamped to minimum (0.01): {all_at_minimum}")
        
        # Test with very positive values  
        raw_positive = torch.tensor([[10.0, 15.0, 20.0, 25.0, 30.0]])
        clamped_positive = torch.clamp(raw_positive, min=gating.min_weight, max=1.0)
        
        logger.info(f"Raw positive values: {raw_positive[0].tolist()}")
        logger.info(f"After clamp(max=1.0): {clamped_positive[0].tolist()}")
        
        # Verify all values are exactly 1.0
        all_at_maximum = torch.all(clamped_positive == 1.0)
        logger.info(f"All values clamped to maximum (1.0): {all_at_maximum}")
        
        # Now force the gating network to output very small values
        logger.info("\nForcing gating networks to output near-zero values...")
        
        with torch.no_grad():
            # Replace all gate network outputs with very small sigmoid outputs
            # by setting extremely negative parameters
            for name, network in gating.gate_networks.items():
                for param in network.parameters():
                    param.data.fill_(-20.0)  # Extremely negative -> sigmoid ≈ 0
        
        # Apply gating with forced small outputs
        gated_features = gating(features)
        
        # Check the actual gate values stored
        logger.info("\nActual gate values after forced small outputs:")
        min_weights_found = []
        
        for group_name, gates in gating.current_gates.items():
            if torch.is_tensor(gates):
                min_weight = float(torch.min(gates))
                max_weight = float(torch.max(gates))
                avg_weight = float(torch.mean(gates))
                min_weights_found.append(min_weight)
                
                logger.info(f"{group_name}: min={min_weight:.6f}, max={max_weight:.6f}, avg={avg_weight:.6f}")
                
                # Check if any weight is below the minimum threshold
                below_min = torch.any(gates < gating.min_weight - 0.0001)  # Small tolerance
                if below_min:
                    logger.error(f"❌ {group_name} has weights below minimum!")
                else:
                    at_or_above_min = torch.all(gates >= gating.min_weight - 0.0001)
                    if at_or_above_min:
                        logger.info(f"✅ {group_name} all weights at or above minimum")
        
        # Overall verification
        overall_min = min(min_weights_found) if min_weights_found else 1.0
        expected_min = gating.min_weight
        
        logger.info(f"\nOverall Results:")
        logger.info(f"Expected minimum weight: {expected_min}")
        logger.info(f"Actual minimum weight found: {overall_min:.8f}")
        logger.info(f"Difference: {overall_min - expected_min:.8f}")
        
        if overall_min >= expected_min - 0.0001:  # Tolerance for floating point
            logger.info("✅ Minimum weight constraint working perfectly!")
            
            # Test that features with minimum weight still contribute to calculations
            logger.info("\nTesting that minimum weight features still contribute to calculations...")
            
            # Find a feature group with minimum weights
            for group_name, gates in gating.current_gates.items():
                min_gate = float(torch.min(gates))
                if abs(min_gate - expected_min) < 0.001:
                    logger.info(f"Found {group_name} with minimum weight {min_gate:.6f}")
                    
                    # Check that the gated feature is not zero
                    if group_name in gated_features:
                        gated_tensor = gated_features[group_name]
                        mean_val = float(torch.mean(torch.abs(gated_tensor)))
                        logger.info(f"  Gated feature mean absolute value: {mean_val:.6f}")
                        
                        if mean_val > 0:
                            logger.info(f"✅ {group_name} with min weight still contributes to calculations")
                        else:
                            logger.warning(f"⚠️ {group_name} with min weight produces zero output")
                    break
            
            return True
        else:
            logger.error(f"❌ Minimum weight constraint failed! Found {overall_min:.8f} < {expected_min}")
            return False
        
    except Exception as e:
        logger.error(f"❌ Minimum weight constraint test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("=== Minimum Weight Constraint Test ===")
    success = test_minimum_weight_constraint()
    
    if success:
        logger.info("🎉 Minimum weight constraint test passed!")
        exit(0)
    else:
        logger.error("❌ Minimum weight constraint test failed!")
        exit(1)