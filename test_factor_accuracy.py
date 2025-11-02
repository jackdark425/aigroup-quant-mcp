"""
测试因子计算准确性
"""
import sys
sys.path.insert(0, '.')

from quantanalyzer.factor import FactorLibrary, FactorEvaluator
import pandas as pd
import numpy as np

print("=" * 60)
print("因子计算准确性测试")
print("=" * 60)

# 创建测试数据
dates = pd.date_range('2023-01-01', periods=100)
symbols = ['AAPL'] * 100
index = pd.MultiIndex.from_arrays([dates, symbols], names=['datetime', 'symbol'])

data = pd.DataFrame({
    'open': np.random.randn(100).cumsum() + 100,
    'high': np.random.randn(100).cumsum() + 105,
    'low': np.random.randn(100).cumsum() + 95,
    'close': np.random.randn(100).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 100)
}, index=index)

print("测试数据创建成功:")
print(f"数据形状: {data.shape}")
print(f"数据索引类型: {type(data.index)}")
print(f"数据列名: {data.columns.tolist()}")

# 计算动量因子
library = FactorLibrary()
momentum_20 = library.momentum(data, period=20)

print("\n动量因子计算测试:")
print(f"动量因子形状: {momentum_20.shape}")
print(f"动量因子有效值数量: {momentum_20.notna().sum()}")
print(f"动量因子前5个值: {momentum_20.dropna().head(5).values}")

# 计算前向收益
forward_return = data['close'].groupby(level=1).pct_change(1).shift(-1)

print("\n前向收益计算:")
print(f"前向收益形状: {forward_return.shape}")
print(f"前向收益有效值数量: {forward_return.notna().sum()}")

# 评估IC
print("\nIC评估测试:")
try:
    evaluator = FactorEvaluator(momentum_20, forward_return)
    ic_metrics = evaluator.calculate_ic()
    
    print("IC评估结果:")
    print(f"IC均值: {ic_metrics['ic_mean']}")
    print(f"IC标准差: {ic_metrics['ic_std']}")
    print(f"ICIR: {ic_metrics['icir']}")
    print(f"IC正值占比: {ic_metrics['ic_positive_ratio']}")
    
    # 检查是否有NaN
    print("\nNaN检查:")
    print(f"动量因子NaN数量: {momentum_20.isna().sum()}")
    print(f"前向收益NaN数量: {forward_return.isna().sum()}")
    print(f"IC均值是否为NaN: {pd.isna(ic_metrics['ic_mean'])}")
    
    if not pd.isna(ic_metrics['ic_mean']):
        print("✅ 因子IC评估正常，无NaN问题")
    else:
        print("❌ 因子IC评估仍存在NaN问题")
        
except Exception as e:
    print(f"❌ IC评估失败: {e}")
    import traceback
    traceback.print_exc()

# 测试其他因子
print("\n" + "=" * 60)
print("其他因子计算测试")
print("=" * 60)

try:
    # 测试波动率因子
    volatility_20 = library.volatility(data, period=20)
    print(f"波动率因子有效值数量: {volatility_20.notna().sum()}")
    
    # 测试RSI因子
    rsi_14 = library.rsi(data, period=14)
    print(f"RSI因子有效值数量: {rsi_14.notna().sum()}")
    
    # 测试成交量比率因子
    volume_ratio_10 = library.volume_ratio(data, period=10)
    print(f"成交量比率因子有效值数量: {volume_ratio_10.notna().sum()}")
    
    print("✅ 所有因子计算正常")
    
except Exception as e:
    print(f"❌ 其他因子计算失败: {e}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)