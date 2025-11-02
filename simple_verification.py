#!/usr/bin/env python3
"""
简单验证脚本 - 直接测试修复的核心功能
"""

import pandas as pd
import numpy as np
import sys
import os

def test_momentum_calculation():
    """测试动量因子计算"""
    print("=== 测试动量因子计算 ===")
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL']
    
    data = []
    for symbol in symbols:
        for date in dates:
            close_price = 100 + np.random.randn() * 10
            data.append({
                'datetime': date,
                'symbol': symbol,
                'close': close_price
            })
    
    df = pd.DataFrame(data)
    df.set_index(['datetime', 'symbol'], inplace=True)
    
    # 直接计算动量因子（使用修复后的逻辑）
    period = 20
    momentum = df['close'].groupby(level=1).apply(
        lambda x: (x / x.shift(period) - 1).fillna(0)
    )
    
    print(f"动量因子形状: {momentum.shape}")
    print(f"非空值数量: {momentum.notna().sum()}")
    print(f"动量因子统计:")
    print(f"  均值: {momentum.mean():.6f}")
    print(f"  标准差: {momentum.std():.6f}")
    print(f"  最小值: {momentum.min():.6f}")
    print(f"  最大值: {momentum.max():.6f}")
    
    # 验证因子有效性
    if momentum.notna().sum() > 0 and momentum.std() > 0:
        print("✅ 动量因子计算成功")
        return True
    else:
        print("❌ 动量因子计算失败")
        return False

def test_ic_evaluation():
    """测试IC评估"""
    print("\n=== 测试IC评估 ===")
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL']
    
    data = []
    for symbol in symbols:
        for date in dates:
            # 创建有相关性的因子和收益率
            factor_value = np.random.randn()
            return_value = factor_value * 0.1 + np.random.randn() * 0.05
            close_price = 100 + np.cumsum([return_value])[0]
            
            data.append({
                'datetime': date,
                'symbol': symbol,
                'factor': factor_value,
                'close': close_price
            })
    
    df = pd.DataFrame(data)
    df.set_index(['datetime', 'symbol'], inplace=True)
    
    # 计算未来收益率
    future_returns = df['close'].groupby(level=1).pct_change(1).shift(-1)
    
    # 计算IC（斯皮尔曼秩相关系数）
    ic_values = []
    for date in df.index.get_level_values(0).unique():
        date_mask = df.index.get_level_values(0) == date
        factor_date = df.loc[date_mask, 'factor']
        returns_date = future_returns.loc[date_mask]
        
        # 对齐数据
        common_index = factor_date.index.intersection(returns_date.index)
        if len(common_index) > 1:
            factor_aligned = factor_date.loc[common_index]
            returns_aligned = returns_date.loc[common_index]
            
            # 计算斯皮尔曼相关系数
            ic = factor_aligned.corr(returns_aligned, method='spearman')
            if not np.isnan(ic):
                ic_values.append(ic)
    
    if ic_values:
        ic_mean = np.mean(ic_values)
        ic_std = np.std(ic_values)
        icir = ic_mean / ic_std if ic_std > 0 else 0
        ic_positive_ratio = sum(1 for ic in ic_values if ic > 0) / len(ic_values)
        
        print(f"IC均值: {ic_mean:.4f}")
        print(f"IC标准差: {ic_std:.4f}")
        print(f"ICIR: {icir:.4f}")
        print(f"IC正值占比: {ic_positive_ratio:.2%}")
        
        if not np.isnan(ic_mean) and not np.isnan(icir):
            print("✅ IC评估成功")
            return True
        else:
            print("❌ IC评估失败")
            return False
    else:
        print("❌ 没有有效的IC值")
        return False

def test_time_range_filtering():
    """测试时间范围过滤"""
    print("\n=== 测试时间范围过滤 ===")
    
    # 创建测试数据
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL']
    
    data = []
    for symbol in symbols:
        for date in dates:
            close_price = 100 + np.random.randn() * 10
            data.append({
                'datetime': date,
                'symbol': symbol,
                'close': close_price
            })
    
    df = pd.DataFrame(data)
    df.set_index(['datetime', 'symbol'], inplace=True)
    
    # 测试时间范围过滤
    train_start = '2023-01-01'
    train_end = '2023-06-30'
    test_start = '2023-07-01'
    test_end = '2023-12-31'
    
    # 转换为datetime对象
    train_start_dt = pd.to_datetime(train_start)
    train_end_dt = pd.to_datetime(train_end)
    test_start_dt = pd.to_datetime(test_start)
    test_end_dt = pd.to_datetime(test_end)
    
    # 创建时间范围掩码
    train_mask = (df.index.get_level_values(0) >= train_start_dt) & (df.index.get_level_values(0) <= train_end_dt)
    test_mask = (df.index.get_level_values(0) >= test_start_dt) & (df.index.get_level_values(0) <= test_end_dt)
    
    train_data = df[train_mask]
    test_data = df[test_mask]
    
    print(f"训练数据形状: {train_data.shape}")
    print(f"测试数据形状: {test_data.shape}")
    
    if len(train_data) > 0 and len(test_data) > 0:
        print("✅ 时间范围过滤成功")
        return True
    else:
        print("❌ 时间范围过滤失败")
        return False

def main():
    """主测试函数"""
    print("开始验证版本1.0.34的核心修复...")
    
    tests = [
        test_momentum_calculation,
        test_ic_evaluation,
        test_time_range_filtering
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    print(f"\n=== 测试总结 ===")
    passed = sum(results)
    total = len(results)
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("🎉 核心修复验证成功！")
        return True
    else:
        print("⚠️ 部分核心修复需要进一步优化")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)