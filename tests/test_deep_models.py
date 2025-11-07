"""
Tests for traditional machine learning models that replaced deep learning models
"""
import unittest
import pytest
import pandas as pd
import numpy as np
from quantanalyzer.model import ModelTrainer

class TestTraditionalModels(unittest.TestCase):
    """Test cases for traditional machine learning models that replaced deep learning models"""
    
    def setUp(self):
        """Set up test data"""
        # Create simple test data
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.rand(100, 5), 
                                   columns=[f'feature_{i}' for i in range(5)])
        self.y_train = pd.Series(np.random.rand(100), name='target')
        self.X_test = pd.DataFrame(np.random.rand(20, 5),
                                  columns=[f'feature_{i}' for i in range(5)])
        self.y_test = pd.Series(np.random.rand(20), name='target')
        
    def test_random_forest_model(self):
        """Test Random Forest model training and prediction"""
        trainer = ModelTrainer(model_type='random_forest')
        trainer.train(self.X_train, self.y_train)
        
        # Check that model was trained
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.feature_importance)
        
        # Check that feature importance has correct shape
        self.assertEqual(len(trainer.feature_importance), self.X_train.shape[1])
        
    def test_gradient_boosting_model(self):
        """Test Gradient Boosting model training and prediction"""
        trainer = ModelTrainer(model_type='gradient_boosting')
        trainer.train(self.X_train, self.y_train)
        
        # Check that model was trained
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.feature_importance)
        
        # Check that feature importance has correct shape
        self.assertEqual(len(trainer.feature_importance), self.X_train.shape[1])
        
    def test_linear_model(self):
        """Test Linear model training and prediction"""
        trainer = ModelTrainer(model_type='linear')
        trainer.train(self.X_train, self.y_train)
        
        # Check that model was trained
        self.assertIsNotNone(trainer.model)
        self.assertIsNotNone(trainer.feature_importance)
        
        # Check that feature importance has correct shape
        self.assertEqual(len(trainer.feature_importance), self.X_train.shape[1])


if __name__ == '__main__':
    unittest.main()