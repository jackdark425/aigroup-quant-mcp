"""
基础使用示例
"""
import sys
sys.path.insert(0, '..')

from quantanalyzer import DataLoader, FactorLibrary, FactorEvaluator
import pandas as pd

print("=" * 60)
print("QuantAnalyzer 基础使用示例")
print("=" * 60)

# 1. 加载数据
print("\n1. 加载数据...")
loader = DataLoader()
data = loader.load_from_csv("../sample_stock_data.csv")
print(f"   ✅ Data loaded: {data.shape}")
print(f"   Columns: {data.columns.tolist()}")

# 2. 数据质量检查
print("\n2. 数据质量检查...")
report = loader.validate_data(data)
print(f"   Shape: {report['shape']}")
print(f"   Symbols: {report['symbols_count']}")
print(f"   Date range: {report['date_range']['start']} to {report['date_range']['end']}")

# 3. 计算因子
print("\n3. 计算因子...")
library = FactorLibrary()

momentum_10 = library.momentum(data, period=10)
momentum_20 = library.momentum(data, period=20)
volatility_20 = library.volatility(data, period=20)
volume_ratio = library.volume_ratio(data, period=20)
rsi = library.rsi(data, period=14)

print(f"   ✅ Momentum 10: {momentum_10.shape}")
print(f"   ✅ Momentum 20: {momentum_20.shape}")
print(f"   ✅ Volatility 20: {volatility_20.shape}")
print(f"   ✅ Volume Ratio: {volume_ratio.shape}")
print(f"   ✅ RSI: {rsi.shape}")

# 4. 因子评估
print("\n4. 因子评估...")
returns = data['close'].groupby(level=1).pct_change(1).shift(-1)

factors_to_evaluate = {
    'momentum_10': momentum_10,
    'momentum_20': momentum_20,
    'volatility_20': volatility_20,
    'volume_ratio': volume_ratio,
    'rsi': rsi
}

results = []
for factor_name, factor_series in factors_to_evaluate.items():
    evaluator = FactorEvaluator(factor_series, returns)
    ic_metrics = evaluator.calculate_ic()
    
    results.append({
        'factor': factor_name,
        'ic_mean': ic_metrics['ic_mean'],
        'icir': ic_metrics['icir'],
        'positive_ratio': ic_metrics['ic_positive_ratio']
    })

results_df = pd.DataFrame(results).sort_values('icir', ascending=False)
print("\n   因子IC评估结果:")
print(results_df.to_string(index=False))

print("\n" + "=" * 60)
print("✅ 测试完成!")
print("=" * 60)