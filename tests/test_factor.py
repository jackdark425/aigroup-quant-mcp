"""
Tests for the factor module
"""
import unittest
import pandas as pd
import numpy as np
from quantanalyzer.factor import FactorLibrary


class TestFactorLibrary(unittest.TestCase):
    """Test cases for FactorLibrary"""
    
    def setUp(self):
        """Set up test data"""
        # 创建MultiIndex数据
        dates = pd.date_range('2020-01-01', periods=5)
        symbols = ['AAPL'] * 5
        index = pd.MultiIndex.from_arrays([dates, symbols], names=['datetime', 'symbol'])
        
        self.data = pd.DataFrame({
            'open': [1, 2, 3, 4, 5],
            'high': [1.5, 2.5, 3.5, 4.5, 5.5],
            'low': [0.5, 1.5, 2.5, 3.5, 4.5],
            'close': [1.2, 2.2, 3.2, 4.2, 5.2],
            'volume': [100, 200, 300, 400, 500]
        }, index=index)
        
        self.library = FactorLibrary()
    
    def test_sma_factor(self):
        """Test SMA factor calculation"""
        result = self.library.calculate_factor('sma', self.data, window=3)
        self.assertIn('sma_3', result.columns)
        # Check that first 2 values are NaN (due to window size)
        self.assertTrue(pd.isna(result['sma_3'].iloc[0]))
        self.assertTrue(pd.isna(result['sma_3'].iloc[1]))
        
    def test_returns_factor(self):
        """Test returns factor calculation"""
        result = self.library.calculate_factor('returns', self.data)
        self.assertIn('returns', result.columns)
        # First value should be NaN
        self.assertTrue(pd.isna(result['returns'].iloc[0]))
        
    def test_volatility_factor(self):
        """Test volatility factor calculation"""
        result = self.library.calculate_factor('volatility', self.data, window=3)
        self.assertIn('volatility_3', result.columns)
        # Check that first 2 values are NaN (due to window size)
        self.assertTrue(pd.isna(result['volatility_3'].iloc[0]))
        self.assertTrue(pd.isna(result['volatility_3'].iloc[1]))


if __name__ == '__main__':
    unittest.main()