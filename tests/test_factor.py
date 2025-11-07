"""
Tests for the factor module
"""
import unittest
import pandas as pd
import numpy as np
from quantanalyzer.factor.library import FactorLibrary
from quantanalyzer.factor.alpha158 import Alpha158Generator


class TestFactorLibrary(unittest.TestCase):
    """Test cases for FactorLibrary"""
    
    def setUp(self):
        """Set up test data"""
        # Create simple test data with MultiIndex
        dates = pd.date_range('2020-01-01', periods=20)
        symbols = ['AAPL', 'GOOGL']
        index = pd.MultiIndex.from_product([dates, symbols], names=['datetime', 'symbol'])
        
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.rand(40) * 100,
            'high': np.random.rand(40) * 100 + 10,
            'low': np.random.rand(40) * 100 - 10,
            'close': np.random.rand(40) * 100,
            'volume': np.random.rand(40) * 1000000
        }, index=index)
        
        self.data = data
    
    def test_momentum_factor(self):
        """Test momentum factor calculation"""
        factor = FactorLibrary.momentum(self.data, period=5)
        self.assertIsInstance(factor, pd.Series)
        self.assertEqual(len(factor), len(self.data))
        
    def test_volatility_factor(self):
        """Test volatility factor calculation"""
        factor = FactorLibrary.volatility(self.data, period=5)
        self.assertIsInstance(factor, pd.Series)
        self.assertEqual(len(factor), len(self.data))
        
    def test_volume_ratio_factor(self):
        """Test volume ratio factor calculation"""
        factor = FactorLibrary.volume_ratio(self.data, period=5)
        self.assertIsInstance(factor, pd.Series)
        self.assertEqual(len(factor), len(self.data))
        
    def test_rsi_factor(self):
        """Test RSI factor calculation"""
        factor = FactorLibrary.rsi(self.data, period=14)
        self.assertIsInstance(factor, pd.Series)
        self.assertEqual(len(factor), len(self.data))


class TestAlpha158Generator(unittest.TestCase):
    """Test cases for Alpha158Generator"""
    
    def setUp(self):
        """Set up test data"""
        # Create simple test data with MultiIndex
        dates = pd.date_range('2020-01-01', periods=20)
        symbols = ['AAPL', 'GOOGL']
        index = pd.MultiIndex.from_product([dates, symbols], names=['datetime', 'symbol'])
        
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.rand(40) * 100,
            'high': np.random.rand(40) * 100 + 10,
            'low': np.random.rand(40) * 100 - 10,
            'close': np.random.rand(40) * 100,
            'volume': np.random.rand(40) * 1000000,
            'vwap': np.random.rand(40) * 100
        }, index=index)
        
        self.data = data
    
    def test_alpha158_generator_initialization(self):
        """Test Alpha158Generator initialization"""
        generator = Alpha158Generator(self.data)
        self.assertIsNotNone(generator)
        self.assertEqual(generator.data.shape, self.data.shape)
        
    def test_alpha158_generate_all(self):
        """Test Alpha158 factor generation"""
        generator = Alpha158Generator(self.data)
        factors = generator.generate_all(
            kbar=True,
            price=True,
            volume=True,
            rolling=True,
            rolling_windows=[5, 10]
        )
        
        self.assertIsInstance(factors, pd.DataFrame)
        self.assertGreater(factors.shape[1], 0)  # Should have at least some factors
        self.assertEqual(factors.shape[0], self.data.shape[0])


if __name__ == '__main__':
    unittest.main()