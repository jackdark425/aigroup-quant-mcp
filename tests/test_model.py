"""
Tests for the model module
"""
import unittest
import numpy as np
from unittest.mock import Mock, patch
from quantanalyzer.model import ModelTrainer


class TestModelTrainer(unittest.TestCase):
    """Test cases for ModelTrainer"""
    
    def setUp(self):
        """Set up test data"""
        # Create simple test data
        np.random.seed(42)
        self.X_train = np.random.rand(100, 5)
        self.y_train = np.random.rand(100)
        self.X_test = np.random.rand(20, 5)
        self.y_test = np.random.rand(20)
    
    def test_model_trainer_initialization(self):
        """Test ModelTrainer initialization"""
        trainer = ModelTrainer(model_type='lightgbm')
        self.assertEqual(trainer.model_type, 'lightgbm')
        
        trainer = ModelTrainer(model_type='xgboost')
        self.assertEqual(trainer.model_type, 'xgboost')
        
        trainer = ModelTrainer(model_type='linear')
        self.assertEqual(trainer.model_type, 'linear')
    
    @patch('quantanalyzer.model.lgb')
    def test_lightgbm_training(self, mock_lgb):
        """Test LightGBM model training"""
        # Mock the LightGBM model
        mock_model = Mock()
        mock_lgb.train.return_value = mock_model
        
        trainer = ModelTrainer(model_type='lightgbm')
        trainer.fit(self.X_train, self.y_train)
        
        # Check that lgb.train was called
        mock_lgb.train.assert_called_once()
        
    @patch('quantanalyzer.model.xgb')
    def test_xgboost_training(self, mock_xgb):
        """Test XGBoost model training"""
        # Mock the XGBoost model
        mock_model = Mock()
        mock_xgb.XGBRegressor.return_value = mock_model
        
        trainer = ModelTrainer(model_type='xgboost')
        trainer.fit(self.X_train, self.y_train)
        
        # Check that XGBRegressor was called
        mock_xgb.XGBRegressor.assert_called_once()
        mock_model.fit.assert_called_once()


if __name__ == '__main__':
    unittest.main()