import unittest
import os
import tempfile
import csv
from models.feature_catalog import FeatureCatalog, get_feature_catalog, reset_feature_catalog


class TestFeatureCatalog(unittest.TestCase):
    """Test feature catalog loading and parsing"""
    
    def setUp(self):
        """Set up test fixtures"""
        reset_feature_catalog()
    
    def tearDown(self):
        """Clean up after tests"""
        reset_feature_catalog()
    
    def test_load_real_csv(self):
        """Test loading the actual crypto_trading_feature_encyclopedia.csv"""
        csv_path = "crypto_trading_feature_encyclopedia.csv"
        if not os.path.exists(csv_path):
            self.skipTest(f"CSV file {csv_path} not found")
        
        catalog = FeatureCatalog(csv_path)
        
        # Basic validation
        self.assertGreater(len(catalog.features), 0, "Should load features from CSV")
        self.assertGreater(len(catalog.must_keep), 0, "Should have must-keep features")
        self.assertGreater(len(catalog.rfe_pool), 0, "Should have RFE-eligible features")
        
        # Core OHLCV features should be must-keep
        core_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Timestamp']
        for feature in core_features:
            self.assertIn(feature, catalog.must_keep, f"{feature} should be must-keep")
            self.assertTrue(catalog.is_must_keep(feature), f"{feature} should be must-keep")
            self.assertFalse(catalog.is_rfe_eligible(feature), f"{feature} should not be RFE eligible")
        
        # Verify some technical indicators are RFE eligible and not must-keep
        tech_indicators = ['SMA_5', 'RSI_14', 'EMA_20']
        for indicator in tech_indicators:
            if indicator in catalog.features:
                if indicator in catalog.rfe_pool:
                    self.assertNotIn(indicator, catalog.must_keep, f"{indicator} should not be must-keep if in RFE pool")
                    self.assertTrue(catalog.is_rfe_eligible(indicator), f"{indicator} should be RFE eligible")
    
    def test_fallback_behavior(self):
        """Test fallback behavior when CSV is missing"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            nonexistent_csv = os.path.join(tmp_dir, "nonexistent.csv")
            
            catalog = FeatureCatalog(nonexistent_csv)
            
            # Should have fallback core features
            self.assertGreater(len(catalog.must_keep), 0, "Should have fallback must-keep features")
            
            # Core OHLCV should be present
            core_features = ['Open', 'High', 'Low', 'Close', 'Volume']
            for feature in core_features:
                self.assertIn(feature, catalog.must_keep, f"{feature} should be in fallback must-keep")
    
    def test_feature_name_parsing(self):
        """Test base indicator name extraction"""
        catalog = FeatureCatalog()
        
        # Test various feature name formats
        test_cases = [
            ('indicator.SMA_5', 'SMA_5'),
            ('ohlcv.Close', 'Close'),
            ('sentiment.fear_greed', 'fear_greed'),
            ('SMA_5', 'SMA_5'),
            ('Close', 'Close')
        ]
        
        for input_name, expected_base in test_cases:
            actual_base = catalog._base_indicator_name(input_name)
            self.assertEqual(actual_base, expected_base, f"Failed for {input_name}")
    
    def test_custom_csv_parsing(self):
        """Test parsing with custom CSV content"""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.writer(f)
            writer.writerow([
                'Indicator', 'Category', 'Required Inputs', 'Formula / Calculation',
                'Must Keep (Not in RFE)', 'RFE Eligible', 'Prerequisite For', 
                'Parameters', 'Outputs'
            ])
            writer.writerow([
                'TestCore', 'Core Price Data', 'OHLCV', 'Raw test data',
                'Yes', 'No', '', '', ''
            ])
            writer.writerow([
                'TestIndicator', 'Trend', 'Close', 'Test calculation',
                'No', 'Yes', '', 'period=10', ''
            ])
            writer.writerow([
                'TestPrereq', 'Prereq', 'Close', 'Test prereq',
                'No', 'Yes', 'Some indicators', '', ''
            ])
            temp_path = f.name
        
        try:
            catalog = FeatureCatalog(temp_path)
            
            # Verify parsing
            self.assertIn('TestCore', catalog.must_keep)
            self.assertNotIn('TestCore', catalog.rfe_pool)
            
            self.assertIn('TestIndicator', catalog.rfe_pool)
            self.assertNotIn('TestIndicator', catalog.must_keep)
            
            self.assertIn('TestPrereq', catalog.prereq)
            self.assertTrue(catalog.is_prereq('TestPrereq'))
            
            # Test feature info
            info = catalog.get_feature_info('TestIndicator')
            self.assertIsNotNone(info)
            self.assertEqual(info.category, 'Trend')
            self.assertEqual(info.parameters, 'period=10')
            
        finally:
            os.unlink(temp_path)
    
    def test_singleton_pattern(self):
        """Test that get_feature_catalog returns the same instance"""
        catalog1 = get_feature_catalog()
        catalog2 = get_feature_catalog()
        
        self.assertIs(catalog1, catalog2, "Should return the same instance")
    
    def test_summary(self):
        """Test catalog summary generation"""
        catalog = FeatureCatalog()
        summary = catalog.get_summary()
        
        self.assertIn('total_features', summary)
        self.assertIn('must_keep_count', summary)
        self.assertIn('rfe_pool_count', summary)
        self.assertIn('prereq_count', summary)
        self.assertIn('categories', summary)
        
        self.assertIsInstance(summary['total_features'], int)
        self.assertIsInstance(summary['categories'], list)


if __name__ == '__main__':
    unittest.main()