"""
Tests for the data module
"""
import unittest
import pandas as pd
import numpy as np
from quantanalyzer.data import DataLoader


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader"""
    
    def test_load_from_dict(self):
        """Test loading data from dictionary"""
        loader = DataLoader()
        data = {
            'open': [1, 2, 3],
            'close': [1.5, 2.5, 3.5],
            'volume': [100, 200, 300]
        }
        df = loader.load_from_dict(data)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        
    def test_load_from_dataframe(self):
        """Test loading data from DataFrame"""
        loader = DataLoader()
        df_input = pd.DataFrame({
            'open': [1, 2, 3],
            'close': [1.5, 2.5, 3.5],
            'volume': [100, 200, 300]
        })
        df = loader.load_from_dataframe(df_input)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        pd.testing.assert_frame_equal(df, df_input)


if __name__ == '__main__':
    unittest.main()