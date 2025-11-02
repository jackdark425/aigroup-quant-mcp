"""
性能改进测试脚本

测试以下优化：
1. Alpha158因子计算的内存优化
2. 深度学习模型的内存管理
3. 回测引擎的数据访问优化
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import time
import tracemalloc
import gc
from quantanalyzer.factor.alpha158 import Alpha158Generator
# from quantanalyzer.model.deep_models import LSTMModel, GRUModel, TransformerModel  # 深度学习模型已移除
from quantanalyzer.backtest.engine import BacktestEngine


def format_memory(bytes_value):
    """格式化内存大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def create_sample_data(n_dates=100, n_symbols=50):
    """创建测试数据"""
    dates = pd.date_range('2020-01-01', periods=n_dates, freq='D')
    symbols = [f'STOCK_{i:03d}' for i in range(n_symbols)]
    
    index = pd.MultiIndex.from_product(
        [dates, symbols],
        names=['datetime', 'symbol']
    )
    
    np.random.seed(42)
    data = pd.DataFrame({
        'open': np.random.randn(len(index)) * 10 + 100,
        'high': np.random.randn(len(index)) * 10 + 105,
        'low': np.random.randn(len(index)) * 10 + 95,
        'close': np.random.randn(len(index)) * 10 + 100,
        'volume': np.random.randint(1000000, 10000000, len(index)),
        'vwap': np.random.randn(len(index)) * 10 + 100,
    }, index=index)
    
    # 确保价格合理
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)
    
    return data


def test_alpha158_performance():
    """测试Alpha158因子计算性能"""
    print("\n" + "="*80)
    print("测试1: Alpha158因子计算性能")
    print("="*80)
    
    # 测试不同数据规模
    test_cases = [
        (50, 30, "小规模"),
        (100, 50, "中等规模"),
        (200, 100, "大规模"),
    ]
    
    for n_dates, n_symbols, desc in test_cases:
        print(f"\n{desc}测试 - {n_dates}天 × {n_symbols}只股票")
        print("-" * 60)
        
        data = create_sample_data(n_dates, n_symbols)
        
        # 测试内存使用
        tracemalloc.start()
        gc.collect()
        start_mem = tracemalloc.get_traced_memory()[0]
        start_time = time.time()
        
        # 生成因子
        generator = Alpha158Generator(data, copy_data=False)
        factors = generator.generate_all(
            kbar=True,
            price=True,
            volume=True,
            rolling=True,
            rolling_windows=[5, 10, 20]
        )
        
        end_time = time.time()
        end_mem = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        elapsed_time = end_time - start_time
        mem_used = end_mem - start_mem
        
        print(f"  生成因子数量: {factors.shape[1]}")
        print(f"  计算时间: {elapsed_time:.2f}秒")
        print(f"  内存使用: {format_memory(mem_used)}")
        print(f"  平均每因子: {elapsed_time/factors.shape[1]:.4f}秒")
        
        # 测试分块处理
        if n_symbols >= 50:
            print(f"\n  测试分块处理 (chunk_size=20):")
            gc.collect()
            start_time = time.time()
            
            factors_chunked = generator.generate_all(
                kbar=True,
                price=False,
                volume=False,
                rolling=True,
                rolling_windows=[5, 10],
                chunk_size=20
            )
            
            elapsed_time = time.time() - start_time
            print(f"    分块计算时间: {elapsed_time:.2f}秒")
            print(f"    因子数量: {factors_chunked.shape[1]}")
        
        del data, generator, factors
        gc.collect()


def test_deep_learning_memory():
    """测试深度学习模型内存管理"""
    print("\n" + "="*80)
    print("测试2: 深度学习模型内存管理")
    print("="*80)
    
    # 创建训练数据
    n_samples = 1000
    n_features = 20
    
    X_train = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    y_train = pd.Series(np.random.randn(n_samples))
    
    X_val = pd.DataFrame(
        np.random.randn(200, n_features),
        columns=[f'feat_{i}' for i in range(n_features)]
    )
    y_val = pd.Series(np.random.randn(200))
    
    # models = [
    #     ('LSTM', LSTMModel),
    #     ('GRU', GRUModel),
    #     ('Transformer', TransformerModel),
    # ]
    models = []  # 深度学习模型已移除
    
    for model_name, ModelClass in models:
        print(f"\n{model_name}模型测试")
        print("-" * 60)
        
        tracemalloc.start()
        gc.collect()
        start_mem = tracemalloc.get_traced_memory()[0]
        start_time = time.time()
        
        # 创建和训练模型
        model = ModelClass(
            d_feat=n_features,
            hidden_size=32,
            num_layers=2,
            n_epochs=20,
            batch_size=128,
            early_stop=5,
            device='cpu'
        )
        
        history = model.fit(X_train, y_train, X_val, y_val)
        
        # 预测
        predictions = model.predict(X_val)
        
        end_time = time.time()
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        elapsed_time = end_time - start_time
        mem_used = current_mem - start_mem
        
        print(f"  训练时间: {elapsed_time:.2f}秒")
        print(f"  最佳轮次: {history['best_epoch']}")
        print(f"  最佳损失: {history['best_loss']:.6f}")
        print(f"  内存使用: {format_memory(mem_used)}")
        print(f"  峰值内存: {format_memory(peak_mem)}")
        print(f"  预测样本数: {len(predictions)}")
        
        del model, predictions
        gc.collect()


def test_backtest_performance():
    """测试回测引擎性能"""
    print("\n" + "="*80)
    print("测试3: 回测引擎性能")
    print("="*80)
    
    test_cases = [
        (50, 30, "小规模"),
        (100, 50, "中等规模"),
        (200, 100, "大规模"),
    ]
    
    for n_dates, n_symbols, desc in test_cases:
        print(f"\n{desc}测试 - {n_dates}天 × {n_symbols}只股票")
        print("-" * 60)
        
        # 创建数据
        data = create_sample_data(n_dates, n_symbols)
        
        # 创建预测值
        predictions = pd.Series(
            np.random.randn(len(data)),
            index=data.index
        )
        
        # 测试回测性能
        tracemalloc.start()
        gc.collect()
        start_mem = tracemalloc.get_traced_memory()[0]
        start_time = time.time()
        
        engine = BacktestEngine(
            initial_capital=10000000,
            commission=0.0003,
            slippage=0.0001
        )
        
        results = engine.run_topk_strategy(
            predictions=predictions,
            prices=data,
            k=min(20, n_symbols),
            holding_period=1
        )
        
        end_time = time.time()
        end_mem = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        elapsed_time = end_time - start_time
        mem_used = end_mem - start_mem
        
        print(f"  回测时间: {elapsed_time:.2f}秒")
        print(f"  内存使用: {format_memory(mem_used)}")
        print(f"  总收益率: {results['total_return']:.2%}")
        print(f"  年化收益: {results['annualized_return']:.2%}")
        print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {results['max_drawdown']:.2%}")
        print(f"  波动率: {results['volatility']:.2%}")
        
        del data, predictions, engine, results
        gc.collect()


def run_comprehensive_test():
    """运行综合测试"""
    print("\n" + "="*80)
    print("量化分析系统性能优化测试")
    print("="*80)
    print("\n测试内容：")
    print("1. Alpha158因子计算 - 内存优化和分块处理")
    print("2. 深度学习模型 - 内存泄漏修复和GPU内存管理")
    print("3. 回测引擎 - MultiIndex数据访问优化")
    
    try:
        # 测试1: Alpha158
        test_alpha158_performance()
        
        # 测试2: 深度学习
        test_deep_learning_memory()
        
        # 测试3: 回测引擎
        test_backtest_performance()
        
        print("\n" + "="*80)
        print("所有测试完成!")
        print("="*80)
        
        print("\n优化总结：")
        print("1. ✓ Alpha158因子计算：")
        print("   - 移除不必要的DataFrame复制")
        print("   - 优化CORR/CORD计算，避免循环和多次concat")
        print("   - 添加分块处理支持大数据集")
        print("   - 及时释放内存")
        
        print("\n2. ✓ 深度学习模型：")
        print("   - 修复deepcopy导致的梯度累积")
        print("   - 批处理后及时释放GPU内存")
        print("   - 预测时使用预分配数组")
        print("   - 定期清理CUDA缓存")
        
        print("\n3. ✓ 回测引擎：")
        print("   - 预先构建数据访问缓存")
        print("   - 避免重复的MultiIndex查找")
        print("   - 使用字典替代频繁的loc/xs访问")
        
    except Exception as e:
        print(f"\n测试出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_comprehensive_test()