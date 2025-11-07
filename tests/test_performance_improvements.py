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
    ]
    
    for days, stocks, desc in test_cases:
        print(f"\n{desc}测试 - {days}天 × {stocks}只股票")
        print("-" * 60)
        
        # 生成测试数据
        dates = pd.date_range('2020-01-01', periods=days, freq='D')
        symbols = [f'STOCK_{i:03d}' for i in range(stocks)]
        index = pd.MultiIndex.from_product([dates, symbols], names=['datetime', 'symbol'])
        
        np.random.seed(42)
        data = pd.DataFrame({
            'open': np.random.rand(len(index)) * 100,
            'high': np.random.rand(len(index)) * 100 + 10,
            'low': np.random.rand(len(index)) * 100 - 10,
            'close': np.random.rand(len(index)) * 100,
            'volume': np.random.rand(len(index)) * 1000000
        }, index=index)
        
        # 测试Alpha158因子生成性能
        generator = Alpha158Generator(data)
        
        start_time = time.time()
        factors = generator.generate_all(
            kbar=True,
            price=True,
            volume=True,
            rolling=True,
            rolling_windows=[5, 10, 20],
            chunk_size=5000,  # 使用分块处理避免内存问题
            parallel=False  # 禁用并行处理以避免测试复杂性
        )
        end_time = time.time()
        
        elapsed = end_time - start_time
        print(f"  因子数量: {factors.shape[1]}")
        print(f"  计算耗时: {elapsed:.2f}秒")
        print(f"  性能: {factors.shape[0] * factors.shape[1] / elapsed:.0f} 因子*股票/秒")
        
        # 验证结果正确性
        assert factors.shape[0] == data.shape[0], "因子行数应与数据行数一致"
        assert factors.shape[1] > 0, "应生成至少一个因子"
        
    print("\n✅ Alpha158因子计算性能测试通过")


def test_deep_learning_memory():
    """测试深度学习模型内存使用（已移除）"""
    print("\n" + "="*80)
    print("测试2: 深度学习模型内存使用")
    print("="*80)
    print("ℹ️  深度学习模型已从项目中移除，专注于传统机器学习算法")
    print("\n✅ 深度学习模型内存测试跳过")


def test_backtest_performance():
    """测试回测引擎性能"""
    print("\n" + "="*80)
    print("测试3: 回测引擎性能")
    print("="*80)
    
    # 生成测试数据
    days, stocks = 100, 20
    print(f"测试 - {days}天 × {stocks}只股票")
    print("-" * 60)
    
    dates = pd.date_range('2020-01-01', periods=days, freq='D')
    symbols = [f'STOCK_{i:03d}' for i in range(stocks)]
    index = pd.MultiIndex.from_product([dates, symbols], names=['datetime', 'symbol'])
    
    np.random.seed(42)
    prices = pd.DataFrame({
        'open': np.random.rand(len(index)) * 100,
        'high': np.random.rand(len(index)) * 100 + 10,
        'low': np.random.rand(len(index)) * 100 - 10,
        'close': np.random.rand(len(index)) * 100,
        'volume': np.random.rand(len(index)) * 1000000
    }, index=index)
    
    # 生成预测数据
    predictions = pd.Series(
        np.random.rand(len(index)) * 2 - 1,  # [-1, 1]范围的随机预测
        index=index
    )
    
    # 测试回测性能
    engine = BacktestEngine(
        initial_capital=10000000,
        commission=0.0003,
        slippage=0.0001
    )
    
    start_time = time.time()
    results = engine.run_topk_strategy(
        predictions=predictions,
        prices=prices,
        k=10,
        holding_period=5
    )
    end_time = time.time()
    
    elapsed = end_time - start_time
    print(f"  回测耗时: {elapsed:.2f}秒")
    print(f"  年化收益: {results.get('annual_return', 0):.2%}")
    print(f"  最大回撤: {results.get('max_drawdown', 0):.2%}")
    
    print("\n✅ 回测引擎性能测试通过")


if __name__ == "__main__":
    test_alpha158_performance()
    test_deep_learning_memory()
    test_backtest_performance()
    print("\n" + "="*80)
    print("所有性能测试完成!")
    print("="*80)