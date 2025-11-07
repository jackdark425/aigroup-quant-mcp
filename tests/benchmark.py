"""
Performance benchmark tests for quantanalyzer
"""
import time
import numpy as np
import pandas as pd
from quantanalyzer.factor.alpha158 import Alpha158Generator
from quantanalyzer.model.trainer import ModelTrainer
from quantanalyzer.data.processor import CSZScoreNorm


def generate_test_data(n_stocks=10, n_days=252):
    """Generate test data for benchmarking"""
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    symbols = [f'STOCK{i:04d}' for i in range(n_stocks)]
    index = pd.MultiIndex.from_product([dates, symbols], names=['datetime', 'symbol'])
    
    np.random.seed(42)
    data = pd.DataFrame({
        'open': np.random.rand(len(index)) * 100,
        'high': np.random.rand(len(index)) * 100 + 10,
        'low': np.random.rand(len(index)) * 100 - 10,
        'close': np.random.rand(len(index)) * 100,
        'volume': np.random.rand(len(index)) * 1000000,
        'vwap': np.random.rand(len(index)) * 100
    }, index=index)
    
    return data


def benchmark_alpha158_generation(data, name="Alpha158 Generation"):
    """Benchmark Alpha158 factor generation"""
    print(f"Starting {name} benchmark...")
    start_time = time.time()
    
    generator = Alpha158Generator(data)
    factors = generator.generate_all()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"{name} Results:")
    print(f"  - Data shape: {data.shape}")
    print(f"  - Factors generated: {factors.shape[1]}")
    print(f"  - Time taken: {elapsed_time:.2f} seconds")
    print(f"  - Performance: {data.shape[0] / elapsed_time:.0f} rows/second")
    print()
    
    return elapsed_time


def benchmark_model_training(factors, labels, model_type="lightgbm", name="Model Training"):
    """Benchmark model training"""
    print(f"Starting {name} benchmark with {model_type}...")
    start_time = time.time()
    
    # Create a simple train/test split
    split_point = len(factors) // 2
    X_train = factors.iloc[:split_point]
    y_train = labels.iloc[:split_point]
    X_test = factors.iloc[split_point:]
    y_test = labels.iloc[split_point:]
    
    # Train model
    trainer = ModelTrainer(model_type=model_type)
    trainer.train(X_train, y_train)
    
    # Make predictions
    predictions = trainer.predict(X_test)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"{name} Results ({model_type}):")
    print(f"  - Training data shape: {X_train.shape}")
    print(f"  - Test data shape: {X_test.shape}")
    print(f"  - Time taken: {elapsed_time:.2f} seconds")
    print(f"  - Performance: {X_train.shape[0] / elapsed_time:.0f} samples/second")
    print()
    
    return elapsed_time


def benchmark_data_processing(data, name="Data Processing"):
    """Benchmark data processing"""
    print(f"Starting {name} benchmark...")
    start_time = time.time()
    
    processor = CSZScoreNorm()
    processed_data = processor(data)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"{name} Results:")
    print(f"  - Data shape: {data.shape}")
    print(f"  - Time taken: {elapsed_time:.2f} seconds")
    print(f"  - Performance: {data.shape[0] / elapsed_time:.0f} rows/second")
    print()
    
    return elapsed_time


def run_benchmarks():
    """Run all benchmarks"""
    print("=" * 60)
    print("QuantAnalyzer Performance Benchmarks")
    print("=" * 60)
    print()
    
    # Test with different data sizes
    data_sizes = [
        (5, 100),    # Small dataset
        (20, 252),   # Medium dataset
        (50, 500),   # Large dataset
    ]
    
    results = []
    
    for n_stocks, n_days in data_sizes:
        print(f"Testing with {n_stocks} stocks and {n_days} days of data")
        print("-" * 50)
        
        # Generate test data
        data = generate_test_data(n_stocks, n_days)
        
        # Benchmark Alpha158 generation
        alpha158_time = benchmark_alpha158_generation(data)
        
        # Generate factors for model training
        generator = Alpha158Generator(data)
        factors = generator.generate_all()
        
        # Generate labels (future returns)
        labels = data['close'].groupby(level=1).pct_change(5).shift(-5).reindex(factors.index)
        
        # Remove rows with NaN labels
        valid_mask = ~labels.isna()
        factors_clean = factors[valid_mask]
        labels_clean = labels[valid_mask]
        
        # Benchmark model training
        model_time = benchmark_model_training(factors_clean, labels_clean)
        
        # Benchmark data processing
        processing_time = benchmark_data_processing(factors_clean)
        
        results.append({
            'n_stocks': n_stocks,
            'n_days': n_days,
            'n_rows': data.shape[0],
            'alpha158_time': alpha158_time,
            'model_time': model_time,
            'processing_time': processing_time
        })
        
        print()
    
    # Print summary
    print("Benchmark Summary")
    print("=" * 50)
    for result in results:
        print(f"Data size: {result['n_stocks']} stocks × {result['n_days']} days "
              f"({result['n_rows']} rows)")
        print(f"  - Alpha158 generation: {result['alpha158_time']:.2f}s")
        print(f"  - Model training: {result['model_time']:.2f}s")
        print(f"  - Data processing: {result['processing_time']:.2f}s")
        print()


if __name__ == "__main__":
    run_benchmarks()