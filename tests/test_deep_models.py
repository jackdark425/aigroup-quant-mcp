"""
Tests for deep learning models
"""
import unittest
import pandas as pd
import numpy as np
import torch
from quantanalyzer.model.deep_models import LSTMModel, GRUModel, TransformerModel


class TestDeepModels(unittest.TestCase):
    """Test cases for deep learning models"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        # Create test data with different input shapes
        self.X_flat = pd.DataFrame(np.random.rand(100, 6))  # Flat features
        self.X_sequence = pd.DataFrame(np.random.rand(100, 30))  # Sequence features (5 time steps * 6 features)
        self.y = pd.Series(np.random.rand(100))
    
    def test_lstm_model_flat_input(self):
        """Test LSTM model with flat input features"""
        model = LSTMModel(d_feat=6, hidden_size=32, n_epochs=1, batch_size=32)
        
        # Should handle flat input without error
        try:
            model.fit(self.X_flat, self.y)
            predictions = model.predict(self.X_flat)
            self.assertEqual(len(predictions), len(self.X_flat))
        except Exception as e:
            self.fail(f"LSTM model failed with flat input: {e}")
    
    def test_lstm_model_sequence_input(self):
        """Test LSTM model with sequence input features"""
        model = LSTMModel(d_feat=6, hidden_size=32, n_epochs=1, batch_size=32)
        
        # Should handle sequence input without error
        try:
            model.fit(self.X_sequence, self.y)
            predictions = model.predict(self.X_sequence)
            self.assertEqual(len(predictions), len(self.X_sequence))
        except Exception as e:
            self.fail(f"LSTM model failed with sequence input: {e}")
    
    def test_gru_model_flat_input(self):
        """Test GRU model with flat input features"""
        model = GRUModel(d_feat=6, hidden_size=32, n_epochs=1, batch_size=32)
        
        # Should handle flat input without error
        try:
            model.fit(self.X_flat, self.y)
            predictions = model.predict(self.X_flat)
            self.assertEqual(len(predictions), len(self.X_flat))
        except Exception as e:
            self.fail(f"GRU model failed with flat input: {e}")
    
    def test_transformer_model_flat_input(self):
        """Test Transformer model with flat input features"""
        model = TransformerModel(d_feat=6, d_model=32, n_epochs=1, batch_size=32)
        
        # Should handle flat input without error
        try:
            model.fit(self.X_flat, self.y)
            predictions = model.predict(self.X_flat)
            self.assertEqual(len(predictions), len(self.X_flat))
        except Exception as e:
            self.fail(f"Transformer model failed with flat input: {e}")


if __name__ == '__main__':
    unittest.main()