"""
Tests for the model module
"""
import unittest
import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from quantanalyzer.model import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer"""
    
    def setUp(self):
        """Set up test data"""
        # Create simple test data
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.rand(100, 5))
        self.y_train = pd.Series(np.random.rand(100))
        self.X_test = pd.DataFrame(np.random.rand(20, 5))
        self.y_test = pd.Series(np.random.rand(20))
    
    def test_model_trainer_initialization(self):
        """Test ModelTrainer initialization"""
        trainer = ModelTrainer(model_type='lightgbm')
        self.assertEqual(trainer.model_type, 'lightgbm')
        
        trainer = ModelTrainer(model_type='xgboost')
        self.assertEqual(trainer.model_type, 'xgboost')
        
        trainer = ModelTrainer(model_type='linear')
        self.assertEqual(trainer.model_type, 'linear')
    
    @patch('lightgbm.train')
    def test_lightgbm_training(self, mock_lgb_train):
        """Test LightGBM model training"""
        # Mock the LightGBM model
        mock_model = Mock()
        mock_lgb_train.return_value = mock_model
        # Mock the feature importance method to return a proper array
        mock_model.feature_importance.return_value = np.array([1, 2, 3, 4, 5])
        
        trainer = ModelTrainer(model_type='lightgbm')
        trainer.train(self.X_train, self.y_train)
        
        # Check that lgb.train was called
        mock_lgb_train.assert_called_once()
        
    @patch('xgboost.train')
    @patch('xgboost.DMatrix')
    def test_xgboost_training(self, mock_dmatrix, mock_xgb_train):
        """Test XGBoost model training"""
        # Mock the XGBoost model
        mock_model = Mock()
        mock_xgb_train.return_value = mock_model
        # Mock the feature importance method to return a proper dict
        mock_model.get_score.return_value = {
            '0': 10,
            '1': 20, 
            '2': 30,
            '3': 40,
            '4': 50
        }
        
        # Mock DMatrix
        mock_dmatrix_instance = Mock()
        mock_dmatrix.return_value = mock_dmatrix_instance
        
        trainer = ModelTrainer(model_type='xgboost')
        trainer.train(self.X_train, self.y_train)
        
        # Check that xgb.train was called
        mock_xgb_train.assert_called_once()
        mock_dmatrix.assert_called()


if __name__ == '__main__':
    unittest.main()