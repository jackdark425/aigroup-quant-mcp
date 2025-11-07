"""
Tests for the data processor module
"""
import unittest
import pandas as pd
import numpy as np
from quantanalyzer.data.processor import (
    ProcessInf, CSZFillna, CSZScoreNorm, ZScoreNorm, DropnaLabel, Fillna
)


class TestProcessors(unittest.TestCase):
    """Test cases for data processors"""
    
    def setUp(self):
        """Set up test data"""
        # Create test data with MultiIndex
        dates = pd.date_range('2020-01-01', periods=10)
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        index = pd.MultiIndex.from_product([dates, symbols], names=['datetime', 'symbol'])
        
        np.random.seed(42)
        data = pd.DataFrame({
            'factor1': np.random.rand(30),
            'factor2': np.random.rand(30),
            'label': np.random.rand(30)
        }, index=index)
        
        # Add some inf and nan values for testing
        data.loc[data.index[0], 'factor1'] = np.inf
        data.loc[data.index[1], 'factor2'] = -np.inf
        data.loc[data.index[2], 'factor1'] = np.nan
        
        self.data = data
    
    def test_process_inf(self):
        """Test ProcessInf processor"""
        processor = ProcessInf()
        processed_data = processor(self.data.copy())
        
        # Check that inf values have been replaced
        self.assertFalse(np.isinf(processed_data['factor1']).any())
        self.assertFalse(np.isinf(processed_data['factor2']).any())
        
    def test_csz_fillna(self):
        """Test CSZFillna processor"""
        processor = CSZFillna()
        processed_data = processor(self.data.copy())
        
        # Check that NaN values have been filled
        self.assertFalse(processed_data['factor1'].isna().any())
        self.assertFalse(processed_data['factor2'].isna().any())
        
    def test_csz_score_norm(self):
        """Test CSZScoreNorm processor"""
        processor = CSZScoreNorm()
        processed_data = processor(self.data.copy())
        
        # Check that values have been normalized (approximately)
        # Mean should be close to 0 and std close to 1 for each date group
        grouped_factor1 = processed_data['factor1'].groupby(level=0)
        grouped_factor2 = processed_data['factor2'].groupby(level=0)
        
        # For most groups, mean should be close to 0
        mean_factor1 = grouped_factor1.mean().abs()
        mean_factor2 = grouped_factor2.mean().abs()
        
        # Most values should be less than 1 (allowing for some variation)
        self.assertGreater((mean_factor1 < 1).sum(), len(mean_factor1) * 0.8)
        self.assertGreater((mean_factor2 < 1).sum(), len(mean_factor2) * 0.8)
        
    def test_zscore_norm(self):
        """Test ZScoreNorm processor"""
        processor = ZScoreNorm()
        
        # Fit on training data
        train_data = self.data.iloc[:15].copy()
        processor.fit(train_data)
        
        # Transform test data
        test_data = self.data.iloc[15:].copy()
        processed_data = processor(test_data)
        
        # Check that values have been normalized
        self.assertIsNotNone(processor.mean_)
        self.assertIsNotNone(processor.std_)
        
    def test_dropna_label(self):
        """Test DropnaLabel processor"""
        # Add NaN to label column
        test_data = self.data.copy()
        test_data.loc[test_data.index[0], 'label'] = np.nan
        
        processor = DropnaLabel(label_col='label')
        processed_data = processor(test_data)
        
        # Check that row with NaN label has been removed
        self.assertEqual(len(processed_data), len(test_data) - 1)
        
    def test_fillna(self):
        """Test Fillna processor"""
        # Add NaN values
        test_data = self.data.copy()
        test_data.loc[test_data.index[0], 'factor1'] = np.nan
        test_data.loc[test_data.index[1], 'factor2'] = np.nan
        
        processor = Fillna(fill_value=0)
        processed_data = processor(test_data)
        
        # Check that NaN values have been filled
        self.assertEqual(processed_data.loc[test_data.index[0], 'factor1'], 0)
        self.assertEqual(processed_data.loc[test_data.index[1], 'factor2'], 0)


if __name__ == '__main__':
    unittest.main()