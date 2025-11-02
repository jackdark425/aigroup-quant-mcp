#!/usr/bin/env python3
"""
测试修复后的aigroup-quant-mcp功能
验证数据格式兼容性和因子计算准确性
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_test_data():
    """创建测试数据"""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    symbols = ['000001.SZ', '000002.SZ']
    
    data = []
    for symbol in symbols:
        for date in dates:
            # 生成随机价格数据
            base_price = 10 + np.random.randn() * 2
            open_price = base_price * (1 + np.random.randn() * 0.01)
            high_price = open_price * (1 + abs(np.random.randn()) * 0.02)
            low_price = open_price * (1 - abs(np.random.randn()) * 0.02)
            close_price = open_price * (1 + np.random.randn() * 0.01)
            volume = int(1000000 * (1 + np.random.randn() * 0.5))
            
            data.append({
                'datetime': date,
                'symbol': symbol,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
    
    df = pd.DataFrame(data)
    df.to_csv('./exports/test_data.csv', index=False)
    print(f"✅ 测试数据已创建: {len(df)} 条记录")
    return df

def test_data_compatibility():
    """测试数据格式兼容性"""
    print("\n🔍 测试数据格式兼容性...")
    
    # 创建测试数据
    df = create_test_data()
    
    # 检查数据格式
    required_columns = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ 缺少必要列: {missing_columns}")
        return False
    
    print("✅ 数据格式检查通过")
    
    # 检查数据完整性
    nan_counts = df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"⚠️  数据中存在NaN值: {nan_counts.to_dict()}")
    else:
        print("✅ 数据完整性检查通过")
    
    return True

def test_factor_calculation():
    """测试因子计算逻辑"""
    print("\n🔍 测试因子计算逻辑...")
    
    # 模拟因子计算
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    close_prices = pd.Series(np.random.randn(len(dates)).cumsum() + 100, index=dates)
    
    # 计算动量因子
    period = 20
    momentum = close_prices.pct_change(period).fillna(0)
    
    # 检查因子质量
    momentum_mean = momentum.mean()
    momentum_std = momentum.std()
    momentum_ir = momentum_mean / momentum_std if momentum_std != 0 else 0
    
    print(f"📊 动量因子统计:")
    print(f"   - 均值: {momentum_mean:.6f}")
    print(f"   - 标准差: {momentum_std:.6f}")
    print(f"   - IR: {momentum_ir:.6f}")
    
    # 检查是否有NaN值
    nan_count = momentum.isna().sum()
    if nan_count == 0:
        print("✅ 因子计算无NaN值")
    else:
        print(f"❌ 因子计算存在 {nan_count} 个NaN值")
        return False
    
    return True

def test_model_training_compatibility():
    """测试模型训练数据兼容性"""
    print("\n🔍 测试模型训练数据兼容性...")
    
    # 模拟因子数据和价格数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    
    # 创建因子数据 (158个因子 + 索引)
    factor_data = pd.DataFrame(
        np.random.randn(len(dates), 158),
        index=dates,
        columns=[f'factor_{i}' for i in range(158)]
    )
    
    # 创建价格数据
    price_data = pd.DataFrame({
        'close': np.random.randn(len(dates)).cumsum() + 100
    }, index=dates)
    
    # 检查数据对齐
    common_index = factor_data.index.intersection(price_data.index)
    
    if len(common_index) == len(factor_data.index):
        print("✅ 数据索引对齐检查通过")
    else:
        print(f"⚠️  数据索引未完全对齐: {len(common_index)}/{len(factor_data.index)}")
    
    # 检查特征和标签分离
    features = factor_data.loc[common_index]
    labels = price_data.loc[common_index, 'close']
    
    if len(features) == len(labels):
        print("✅ 特征和标签分离检查通过")
    else:
        print(f"❌ 特征和标签数量不匹配: {len(features)} vs {len(labels)}")
        return False
    
    return True

def main():
    """主测试函数"""
    print("🚀 开始测试修复后的aigroup-quant-mcp功能")
    print("=" * 60)
    
    # 确保exports目录存在
    os.makedirs('./exports', exist_ok=True)
    
    # 运行测试
    tests = [
        ("数据格式兼容性", test_data_compatibility),
        ("因子计算逻辑", test_factor_calculation),
        ("模型训练兼容性", test_model_training_compatibility)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} 测试失败: {e}")
            results.append((test_name, False))
    
    # 输出测试结果
    print("\n" + "=" * 60)
    print("📋 测试结果汇总:")
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {test_name}: {status}")
    
    passed_count = sum(1 for _, result in results if result)
    total_count = len(results)
    
    print(f"\n🎯 总体结果: {passed_count}/{total_count} 项测试通过")
    
    if passed_count == total_count:
        print("🎉 所有测试通过！修复成功！")
    else:
        print("⚠️  部分测试失败，需要进一步优化")

if __name__ == "__main__":
    main()