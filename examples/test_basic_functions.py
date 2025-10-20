"""
基础功能测试
"""
import sys
sys.path.insert(0, '..')

from quantanalyzer import DataLoader, FactorLibrary
import pandas as pd
import numpy as np

print("=" * 60)
print("QuantAnalyzer 核心功能测试")
print("=" * 60)

# 测试1: 数据加载
print("\n【测试1】数据加载功能")
print("-" * 40)
try:
    loader = DataLoader()
    data = loader.load_from_csv("../sample_stock_data.csv")
    print(f"✅ 数据加载成功")
    print(f"   数据形状: {data.shape}")
    print(f"   索引类型: {type(data.index)}")
    print(f"   列名: {data.columns.tolist()}")
    print(f"   前3行数据:")
    print(data.head(3))
except Exception as e:
    print(f"❌ 数据加载失败: {e}")
    sys.exit(1)

# 测试2: 因子计算
print("\n【测试2】因子计算功能")
print("-" * 40)
try:
    library = FactorLibrary()
    
    # 计算动量因子
    momentum_10 = library.momentum(data, period=10)
    print(f"✅ 动量因子(10日)计算成功")
    print(f"   形状: {momentum_10.shape}")
    print(f"   有效值数量: {momentum_10.notna().sum()}")
    print(f"   示例值:")
    print(momentum_10.dropna().head(5))
    
    # 计算波动率因子
    volatility_20 = library.volatility(data, period=20)
    print(f"\n✅ 波动率因子(20日)计算成功")
    print(f"   形状: {volatility_20.shape}")
    print(f"   有效值数量: {volatility_20.notna().sum()}")
    print(f"   示例值:")
    print(volatility_20.dropna().head(5))
    
    # 计算RSI
    rsi = library.rsi(data, period=14)
    print(f"\n✅ RSI指标(14日)计算成功")
    print(f"   形状: {rsi.shape}")
    print(f"   有效值数量: {rsi.notna().sum()}")
    print(f"   示例值:")
    print(rsi.dropna().head(5))
    
except Exception as e:
    print(f"❌ 因子计算失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 数据统计
print("\n【测试3】因子统计信息")
print("-" * 40)
try:
    print("动量因子统计:")
    print(f"   均值: {momentum_10.mean():.6f}")
    print(f"   标准差: {momentum_10.std():.6f}")
    print(f"   最小值: {momentum_10.min():.6f}")
    print(f"   最大值: {momentum_10.max():.6f}")
    
    print("\n波动率因子统计:")
    print(f"   均值: {volatility_20.mean():.6f}")
    print(f"   标准差: {volatility_20.std():.6f}")
    print(f"   最小值: {volatility_20.min():.6f}")
    print(f"   最大值: {volatility_20.max():.6f}")
    
    print("\nRSI指标统计:")
    print(f"   均值: {rsi.mean():.6f}")
    print(f"   标准差: {rsi.std():.6f}")
    print(f"   最小值: {rsi.min():.6f}")
    print(f"   最大值: {rsi.max():.6f}")
    
    print("\n✅ 统计计算成功")
except Exception as e:
    print(f"❌ 统计计算失败: {e}")
    sys.exit(1)

# 测试4: 数据验证
print("\n【测试4】数据质量验证")
print("-" * 40)
try:
    report = loader.validate_data(data)
    print(f"数据形状: {report['shape']}")
    print(f"股票数量: {report['symbols_count']}")
    print(f"时间范围: {report['date_range']['start']} 至 {report['date_range']['end']}")
    print(f"重复记录: {report['duplicate_count']}")
    print("\n缺失值比例:")
    for col, ratio in report['missing_ratio'].items():
        print(f"   {col}: {ratio:.2%}")
    
    print("\n✅ 数据质量验证成功")
except Exception as e:
    print(f"❌ 数据质量验证失败: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 所有测试通过！")
print("=" * 60)
print("\n核心功能验证:")
print("  ✅ 数据加载")
print("  ✅ 因子计算（动量、波动率、RSI）")
print("  ✅ 统计分析")
print("  ✅ 数据质量检查")
print("\nQuantAnalyzer包运行正常！")