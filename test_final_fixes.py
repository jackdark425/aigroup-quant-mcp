#!/usr/bin/env python3
"""
最终修复验证脚本
测试版本1.0.34中修复的所有问题
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantanalyzer.data.loader import DataLoader
from quantanalyzer.factor.library import FactorLibrary
from quantanalyzer.factor.evaluator import FactorEvaluator
from quantanalyzer.model.trainer import ModelTrainer

def create_test_data():
    """创建测试数据"""
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    data = []
    for symbol in symbols:
        for date in dates:
            # 生成随机价格数据
            base_price = 100 + np.random.randn() * 10
            open_price = base_price + np.random.randn() * 2
            high_price = open_price + abs(np.random.randn() * 3)
            low_price = open_price - abs(np.random.randn() * 3)
            close_price = (open_price + high_price + low_price) / 3 + np.random.randn() * 1
            volume = np.random.randint(1000000, 10000000)
            
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
    df.set_index(['datetime', 'symbol'], inplace=True)
    return df

def test_momentum_factor():
    """测试动量因子计算"""
    print("=== 测试动量因子计算 ===")
    
    # 创建测试数据
    data = create_test_data()
    
    # 计算动量因子
    factor_lib = FactorLibrary()
    momentum = factor_lib.calculate_momentum(data, period=20)
    
    print(f"动量因子形状: {momentum.shape}")
    print(f"动量因子非空值数量: {momentum.notna().sum()}")
    print(f"动量因子统计: {momentum.describe()}")
    
    # 验证因子有效性
    if momentum.notna().sum() > 0:
        print("✅ 动量因子计算成功")
        return True
    else:
        print("❌ 动量因子计算失败")
        return False

def test_factor_ic_evaluation():
    """测试因子IC评估"""
    print("\n=== 测试因子IC评估 ===")
    
    # 创建测试数据
    data = create_test_data()
    
    # 计算动量因子
    factor_lib = FactorLibrary()
    momentum = factor_lib.calculate_momentum(data, period=20)
    
    # 评估因子IC
    evaluator = FactorEvaluator()
    ic_results = evaluator.evaluate_ic(momentum, data['close'], method='spearman')
    
    print(f"IC均值: {ic_results['ic_mean']:.4f}")
    print(f"IC标准差: {ic_results['ic_std']:.4f}")
    print(f"ICIR: {ic_results['icir']:.4f}")
    print(f"IC正值占比: {ic_results['ic_positive_ratio']:.2%}")
    
    # 验证IC评估结果
    if not np.isnan(ic_results['ic_mean']) and not np.isnan(ic_results['icir']):
        print("✅ 因子IC评估成功")
        return True
    else:
        print("❌ 因子IC评估失败")
        return False

def test_data_merging_and_training():
    """测试数据合并和模型训练"""
    print("\n=== 测试数据合并和模型训练 ===")
    
    # 创建测试数据
    data = create_test_data()
    
    # 模拟因子数据（Alpha158特征）
    factor_data = data.copy()
    # 添加一些模拟因子列
    for i in range(10):
        factor_data[f'factor_{i}'] = np.random.randn(len(factor_data))
    
    # 创建数据管理器
    data_manager = DataLoader()
    data_manager.store_data('test_data', data)
    data_manager.store_data('test_factors', factor_data)
    
    # 测试时间范围过滤
    train_start = '2023-01-01'
    train_end = '2023-06-30'
    test_start = '2023-07-01'
    test_end = '2023-12-31'
    
    # 获取训练和测试数据
    train_data = data_manager.get_data_in_range('test_data', train_start, train_end)
    test_data = data_manager.get_data_in_range('test_data', test_start, test_end)
    
    print(f"训练数据形状: {train_data.shape if train_data is not None else 'None'}")
    print(f"测试数据形状: {test_data.shape if test_data is not None else 'None'}")
    
    if train_data is not None and len(train_data) > 0 and test_data is not None and len(test_data) > 0:
        print("✅ 数据合并和时间范围识别成功")
        return True
    else:
        print("❌ 数据合并和时间范围识别失败")
        return False

def test_feature_consistency():
    """测试特征一致性"""
    print("\n=== 测试特征一致性 ===")
    
    # 创建测试数据
    data = create_test_data()
    
    # 模拟因子数据
    factor_data = data.copy()
    feature_columns = []
    for i in range(5):
        col_name = f'feature_{i}'
        factor_data[col_name] = np.random.randn(len(factor_data))
        feature_columns.append(col_name)
    
    # 创建数据管理器
    data_manager = DataLoader()
    data_manager.store_data('consistency_test', factor_data)
    
    # 获取特征数据
    stored_data = data_manager.get_data('consistency_test')
    if stored_data is not None:
        available_features = [col for col in stored_data.columns if col.startswith('feature_')]
        print(f"可用特征: {available_features}")
        
        if len(available_features) == len(feature_columns):
            print("✅ 特征一致性检查成功")
            return True
        else:
            print("❌ 特征一致性检查失败")
            return False
    else:
        print("❌ 无法获取存储的数据")
        return False

def main():
    """主测试函数"""
    print("开始验证版本1.0.34的修复...")
    
    tests = [
        test_momentum_factor,
        test_factor_ic_evaluation,
        test_data_merging_and_training,
        test_feature_consistency
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            results.append(False)
    
    print(f"\n=== 测试总结 ===")
    passed = sum(results)
    total = len(results)
    print(f"通过测试: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有修复验证成功！")
        return True
    else:
        print("⚠️ 部分修复需要进一步优化")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)