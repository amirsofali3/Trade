import os
import csv
import logging
from typing import Set, Dict, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("FeatureCatalog")

@dataclass
class FeatureInfo:
    """Information about a feature from the encyclopedia CSV"""
    indicator: str
    category: str
    required_inputs: str
    formula: str
    must_keep: bool
    rfe_eligible: bool
    prerequisite_for: str = ""
    parameters: str = ""
    outputs: str = ""


class FeatureCatalog:
    """
    Feature Catalog Loader for crypto_trading_feature_encyclopedia.csv
    
    This class loads and parses the feature encyclopedia CSV to build structured
    feature sets for RFE filtering and must-keep enforcement.
    """
    
    def __init__(self, csv_path: str = None):
        """
        Initialize the feature catalog
        
        Args:
            csv_path: Optional path to the CSV file. If None, looks for it in the current directory.
        """
        if csv_path is None:
            csv_path = "crypto_trading_feature_encyclopedia.csv"
        
        self.csv_path = csv_path
        self.features: Dict[str, FeatureInfo] = {}
        self.must_keep: Set[str] = set()
        self.rfe_pool: Set[str] = set() 
        self.prereq: Set[str] = set()
        self.meta: Dict[str, Dict[str, Any]] = {}
        
        self._load_catalog()
    
    def _load_catalog(self):
        """Load and parse the feature encyclopedia CSV"""
        try:
            if not os.path.exists(self.csv_path):
                logger.warning(f"Feature encyclopedia CSV not found at {self.csv_path}. Using fallback behavior.")
                self._create_fallback_catalog()
                return
                
            logger.info(f"Loading feature catalog from {self.csv_path}")
            
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    try:
                        # Parse the row into FeatureInfo
                        indicator = row.get('Indicator', '').strip()
                        if not indicator:
                            continue
                            
                        category = row.get('Category', '').strip()
                        required_inputs = row.get('Required Inputs', '').strip()
                        formula = row.get('Formula / Calculation', '').strip()
                        must_keep_str = row.get('Must Keep (Not in RFE)', '').strip().lower()
                        rfe_eligible_str = row.get('RFE Eligible', '').strip().lower()
                        prerequisite_for = row.get('Prerequisite For', '').strip()
                        parameters = row.get('Parameters', '').strip()
                        outputs = row.get('Outputs', '').strip()
                        
                        # Convert Yes/No to boolean
                        must_keep = must_keep_str in ['yes', 'y', '1', 'true']
                        rfe_eligible = rfe_eligible_str in ['yes', 'y', '1', 'true']
                        
                        # Create FeatureInfo object
                        feature_info = FeatureInfo(
                            indicator=indicator,
                            category=category,
                            required_inputs=required_inputs,
                            formula=formula,
                            must_keep=must_keep,
                            rfe_eligible=rfe_eligible,
                            prerequisite_for=prerequisite_for,
                            parameters=parameters,
                            outputs=outputs
                        )
                        
                        self.features[indicator] = feature_info
                        
                        # Build sets based on flags
                        if must_keep:
                            self.must_keep.add(indicator)
                        
                        if rfe_eligible and not must_keep:
                            self.rfe_pool.add(indicator)
                            
                        if category.lower() == 'prereq':
                            self.prereq.add(indicator)
                        
                        # Also add core data to must_keep (should already be marked, but safety)
                        if category in ['Core Price Data', 'Core Meta', 'Core Order Book']:
                            self.must_keep.add(indicator)
                            
                        # Build metadata dict
                        self.meta[indicator] = {
                            'category': category,
                            'must_keep': must_keep,
                            'rfe_eligible': rfe_eligible,
                            'parameters': parameters,
                            'outputs': outputs
                        }
                        
                    except Exception as e:
                        logger.warning(f"Error parsing row for indicator '{indicator}': {e}")
                        continue
            
            logger.info(f"Loaded feature catalog: {len(self.features)} features, "
                       f"{len(self.must_keep)} must-keep, {len(self.rfe_pool)} RFE-eligible, "
                       f"{len(self.prereq)} prerequisites")
                       
        except Exception as e:
            logger.error(f"Error loading feature catalog from {self.csv_path}: {e}")
            logger.warning("Falling back to default catalog behavior")
            self._create_fallback_catalog()
    
    def _create_fallback_catalog(self):
        """Create a fallback catalog with basic core features if CSV is missing"""
        logger.info("Creating fallback feature catalog with basic OHLCV core features")
        
        # Core OHLCV features that should always be must-keep
        core_features = [
            'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'
        ]
        
        for feature in core_features:
            self.must_keep.add(feature)
            self.features[feature] = FeatureInfo(
                indicator=feature,
                category='Core Price Data',
                required_inputs='OHLCV',
                formula=f'Raw {feature.lower()}',
                must_keep=True,
                rfe_eligible=False
            )
            self.meta[feature] = {
                'category': 'Core Price Data',
                'must_keep': True,
                'rfe_eligible': False,
                'parameters': '',
                'outputs': ''
            }
    
    def is_must_keep(self, feature_name: str) -> bool:
        """Check if a feature should be must-keep (never removed by RFE)"""
        base_name = self._base_indicator_name(feature_name)
        return base_name in self.must_keep
    
    def is_rfe_eligible(self, feature_name: str) -> bool:
        """Check if a feature is eligible for RFE selection"""
        base_name = self._base_indicator_name(feature_name)
        return base_name in self.rfe_pool
    
    def is_prereq(self, feature_name: str) -> bool:
        """Check if a feature is a prerequisite"""
        base_name = self._base_indicator_name(feature_name)
        return base_name in self.prereq
    
    def get_feature_info(self, feature_name: str) -> FeatureInfo:
        """Get detailed information about a feature"""
        base_name = self._base_indicator_name(feature_name)
        return self.features.get(base_name)
    
    def _base_indicator_name(self, feature_name: str) -> str:
        """
        Extract base indicator name from feature name.
        
        Examples:
        - 'indicator.SMA_5' -> 'SMA_5'
        - 'ohlcv.Close' -> 'Close'
        - 'SMA_5' -> 'SMA_5'
        """
        if '.' in feature_name:
            return feature_name.split('.', 1)[1]
        return feature_name
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics about the catalog"""
        return {
            'total_features': len(self.features),
            'must_keep_count': len(self.must_keep),
            'rfe_pool_count': len(self.rfe_pool),
            'prereq_count': len(self.prereq),
            'categories': list(set(info.category for info in self.features.values()))
        }


# Global instance for easy access
_catalog_instance = None

def get_feature_catalog(csv_path: str = None) -> FeatureCatalog:
    """Get the global feature catalog instance (singleton pattern)"""
    global _catalog_instance
    if _catalog_instance is None:
        _catalog_instance = FeatureCatalog(csv_path)
    return _catalog_instance


def reset_feature_catalog():
    """Reset the global catalog instance (for testing)"""
    global _catalog_instance
    _catalog_instance = None