#!/usr/bin/env python3
"""
测试机器学习算法训练功能
验证15种传统机器学习算法的训练和预测功能
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
sys.path.insert(0, '.')

from quantanalyzer.model.trainer import ModelTrainer
from quantanalyzer.data.loader import DataLoader

def create_sample_data():
    """创建样本数据用于测试"""
    # 生成时间序列
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # 创建样本数据
    data = []
    for i, date in enumerate(dates):
        # 基础价格
        base_price = 100 + i * 0.1
        
        # 生成特征
        feature1 = np.sin(i * 0.1) + np.random.normal(0, 0.1)
        feature2 = np.cos(i * 0.05) + np.random.normal(0, 0.1)
        feature3 = np.random.normal(0, 1)
        feature4 = np.random.uniform(0, 1)
        feature5 = np.log(i + 1) + np.random.normal(0, 0.1)
        
        # 目标变量（收益率）
        target = feature1 * 0.3 + feature2 * 0.2 + feature3 * 0.1 + np.random.normal(0, 0.05)
        
        data.append({
            'datetime': date,
            'symbol': 'TEST',
            'open': base_price,
            'high': base_price + np.random.uniform(0, 2),
            'low': base_price - np.random.uniform(0, 2),
            'close': base_price + target,
            'volume': np.random.randint(1000, 10000),
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'feature4': feature4,
            'feature5': feature5,
            'target': target
        })
    
    df = pd.DataFrame(data)
    df.set_index(['datetime', 'symbol'], inplace=True)
    return df

def test_ml_algorithms():
    """测试所有机器学习算法"""
    print("🚀 开始测试机器学习算法训练功能")
    print("=" * 60)
    
    # 创建样本数据
    print("📊 创建样本数据...")
    sample_data = create_sample_data()
    print(f"样本数据形状: {sample_data.shape}")
    
    # 定义要测试的算法
    algorithms = [
        'linear', 'ridge', 'lasso', 'elasticnet', 'logistic',
        'lightgbm', 'xgboost', 'random_forest', 'gradient_boosting', 'decision_tree', 'catboost',
        'svm', 'svr', 'naive_bayes', 'knn'
    ]
    
    # 准备特征和目标
    feature_cols = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    X = sample_data[feature_cols]
    y = sample_data['target']
    
    # 分割训练集和测试集
    split_idx = int(len(X) * 0.7)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print()
    
    results = []
    
    for algo in algorithms:
        print(f"🧪 测试算法: {algo}")
        
        try:
            # 创建模型训练器
            trainer = ModelTrainer(model_type=algo)
            
            # 训练模型
            print(f"  📈 训练模型...")
            model = trainer.train(X_train, y_train)
            
            # 预测
            print(f"  🔮 进行预测...")
            y_pred = trainer.predict(X_test)
            
            # 计算特征重要性
            print(f"  📊 计算特征重要性...")
            feature_importance = trainer.feature_importance
            
            # 评估模型
            mse = np.mean((y_test - y_pred) ** 2)
            mae = np.mean(np.abs(y_test - y_pred))
            r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2)
            
            # 检查特征重要性
            if feature_importance is not None:
                importance_sum = feature_importance.sum()
                has_importance = importance_sum > 0
            else:
                has_importance = False
            
            results.append({
                'algorithm': algo,
                'status': '✅ 成功',
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'has_feature_importance': has_importance,
                'error': None
            })
            
            print(f"  ✅ {algo} - 训练成功")
            print(f"      MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            print(f"      特征重要性: {'✅ 有' if has_importance else '❌ 无'}")
            
        except Exception as e:
            results.append({
                'algorithm': algo,
                'status': '❌ 失败',
                'mse': None,
                'mae': None,
                'r2': None,
                'has_feature_importance': False,
                'error': str(e)
            })
            print(f"  ❌ {algo} - 训练失败: {str(e)}")
        
        print()
    
    # 输出总结报告
    print("=" * 60)
    print("📋 测试总结报告")
    print("=" * 60)
    
    successful = [r for r in results if r['status'] == '✅ 成功']
    failed = [r for r in results if r['status'] == '❌ 失败']
    
    print(f"✅ 成功: {len(successful)}/{len(algorithms)}")
    print(f"❌ 失败: {len(failed)}/{len(algorithms)}")
    
    if successful:
        print("\n📊 成功算法的性能统计:")
        successful_df = pd.DataFrame(successful)
        print(f"平均 MSE: {successful_df['mse'].mean():.4f}")
        print(f"平均 MAE: {successful_df['mae'].mean():.4f}")
        print(f"平均 R²: {successful_df['r2'].mean():.4f}")
        print(f"支持特征重要性的算法: {successful_df['has_feature_importance'].sum()}/{len(successful)}")
    
    if failed:
        print("\n❌ 失败的算法:")
        for fail in failed:
            print(f"  - {fail['algorithm']}: {fail['error']}")
    
    # 按性能排序
    if successful:
        print("\n🏆 性能排名 (按R²):")
        sorted_results = sorted(successful, key=lambda x: x['r2'], reverse=True)
        for i, result in enumerate(sorted_results[:5], 1):
            print(f"  {i}. {result['algorithm']}: R² = {result['r2']:.4f}")
    
    return results

def test_model_trainer_initialization():
    """测试ModelTrainer类的初始化"""
    print("🧪 测试ModelTrainer初始化...")
    
    test_cases = [
        ('lightgbm', 'LightGBM梯度提升树'),
        ('xgboost', 'XGBoost梯度提升树'),
        ('linear', '线性回归'),
        ('ridge', '岭回归'),
        ('lasso', 'Lasso回归'),
        ('elasticnet', '弹性网络'),
        ('logistic', '逻辑回归'),
        ('random_forest', '随机森林'),
        ('gradient_boosting', '梯度提升树'),
        ('decision_tree', '决策树'),
        ('catboost', 'CatBoost'),
        ('svm', '支持向量机'),
        ('svr', '支持向量回归'),
        ('naive_bayes', '朴素贝叶斯'),
        ('knn', 'K-最近邻')
    ]
    
    for model_type, description in test_cases:
        try:
            trainer = ModelTrainer(model_type=model_type)
            print(f"  ✅ {model_type} - {description} - 初始化成功")
        except Exception as e:
            print(f"  ❌ {model_type} - {description} - 初始化失败: {str(e)}")

if __name__ == "__main__":
    print("🤖 机器学习算法功能验证")
    print("=" * 60)
    
    # 测试ModelTrainer初始化
    test_model_trainer_initialization()
    print()
    
    # 测试所有算法训练功能
    results = test_ml_algorithms()
    
    # 总结
    successful_count = len([r for r in results if r['status'] == '✅ 成功'])
    total_count = len(results)
    
    print("\n" + "=" * 60)
    if successful_count == total_count:
        print("🎉 所有算法测试通过！机器学习训练功能正常。")
    else:
        print(f"⚠️  {successful_count}/{total_count} 个算法测试通过。")
        print("建议检查失败的算法是否需要额外的依赖或参数调整。")