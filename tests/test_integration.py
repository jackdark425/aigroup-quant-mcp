"""
Integration tests for quantanalyzer
"""
import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from quantanalyzer.data.loader import DataLoader
from quantanalyzer.factor.alpha158 import Alpha158Generator
from quantanalyzer.factor.library import FactorLibrary
from quantanalyzer.factor.evaluator import FactorEvaluator
from quantanalyzer.model.trainer import ModelTrainer
from quantanalyzer.data.processor import CSZScoreNorm


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow"""
    
    def setUp(self):
        """Set up test data"""
        # Create sample data
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        symbols = ['AAPL', 'GOOGL', 'MSFT']
        index = pd.MultiIndex.from_product([dates, symbols], names=['datetime', 'symbol'])
        
        np.random.seed(42)
        # Generate realistic price data
        base_prices = np.random.rand(len(index)) * 100
        returns = np.random.randn(len(index)) * 0.02  # 2% daily volatility
        prices = base_prices * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.rand(len(index)) * 0.01),
            'high': prices * (1 + np.random.rand(len(index)) * 0.02),
            'low': prices * (1 - np.random.rand(len(index)) * 0.02),
            'close': prices,
            'volume': np.random.rand(len(index)) * 1000000
        }, index=index)
        
        self.data = data
        
    def test_complete_workflow(self):
        """Test the complete workflow from data loading to model training"""
        # Step 1: Data processing
        processor = CSZScoreNorm()
        processed_data = processor(self.data)
        
        # Step 2: Factor generation
        generator = Alpha158Generator(processed_data)
        factors = generator.generate_all(
            kbar=True,
            price=True,
            volume=True,
            rolling=True,
            rolling_windows=[5, 10],
            parallel=False  # Disable parallel processing for testing
        )
        
        # Check that factors were generated
        self.assertIsInstance(factors, pd.DataFrame)
        self.assertGreater(factors.shape[1], 0)
        self.assertEqual(factors.shape[0], self.data.shape[0])
        
        # Step 3: Factor evaluation
        # Generate forward returns for IC calculation
        forward_returns = self.data['close'].groupby(level=1).pct_change(5).shift(-5)
        forward_returns = forward_returns.reindex(factors.index)
        
        # Evaluate a sample factor
        sample_factor = factors.iloc[:, 0]  # First factor
        evaluator = FactorEvaluator(sample_factor, forward_returns)
        ic_metrics = evaluator.calculate_ic()
        
        # Check that IC metrics were calculated
        self.assertIn('ic_mean', ic_metrics)
        self.assertIn('ic_std', ic_metrics)
        self.assertIn('icir', ic_metrics)
        
        # Step 4: Model training
        # Prepare data for training
        # Remove rows with NaN values
        valid_mask = ~(factors.isna().any(axis=1) | forward_returns.isna())
        clean_factors = factors[valid_mask]
        clean_returns = forward_returns[valid_mask]
        
        if len(clean_factors) > 10:  # Only proceed if we have enough data
            # Split data
            split_idx = len(clean_factors) // 2
            X_train = clean_factors.iloc[:split_idx]
            y_train = clean_returns.iloc[:split_idx]
            X_test = clean_factors.iloc[split_idx:]
            y_test = clean_returns.iloc[split_idx:]
            
            # Train model
            trainer = ModelTrainer(model_type='lightgbm', model_id='test_model')
            trainer.train(X_train, y_train, use_cache=False)  # Disable cache for testing
            
            # Check that model was trained
            self.assertIsNotNone(trainer.model)
            self.assertIsNotNone(trainer.feature_importance)
            
            # Make predictions
            predictions = trainer.predict(X_test)
            
            # Check that predictions were made
            self.assertIsInstance(predictions, pd.Series)
            self.assertEqual(len(predictions), len(X_test))
            
    def test_data_loader_integration(self):
        """Test data loader integration with other components"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            # Write sample data to CSV
            self.data.to_csv(f.name)
            temp_file = f.name
        
        try:
            # Load data using DataLoader
            loader = DataLoader()
            loaded_data = loader.load_from_csv(temp_file)
            
            # Check that data was loaded correctly
            self.assertIsInstance(loaded_data, pd.DataFrame)
            self.assertEqual(loaded_data.shape, self.data.shape)
            
            # Process data
            processor = CSZScoreNorm()
            processed_data = processor(loaded_data)
            
            # Generate factors
            generator = Alpha158Generator(processed_data)
            factors = generator.generate_all(
                kbar=False,  # Simplify for testing
                price=True,
                volume=True,
                rolling=False,
                parallel=False
            )
            
            # Check that factors were generated
            self.assertIsInstance(factors, pd.DataFrame)
            self.assertGreater(factors.shape[1], 0)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
            
    def test_single_factor_workflow(self):
        """Test workflow with a single factor"""
        # Calculate a single factor
        factor = FactorLibrary.momentum(self.data, period=5)
        
        # Generate forward returns
        forward_returns = self.data['close'].groupby(level=1).pct_change(5).shift(-5)
        
        # Align factor and returns
        factor, forward_returns = factor.align(forward_returns, join='inner')
        
        # Evaluate factor
        evaluator = FactorEvaluator(factor, forward_returns)
        ic_metrics = evaluator.calculate_ic()
        
        # Check results
        self.assertIn('ic_mean', ic_metrics)
        self.assertIn('ic_std', ic_metrics)
        self.assertIn('icir', ic_metrics)


if __name__ == '__main__':
    unittest.main()