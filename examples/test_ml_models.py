#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试机器学习模型功能
演示如何使用LightGBM/XGBoost/sklearn进行训练和预测
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import pandas as pd
import numpy as np
from quantanalyzer.data import DataLoader
from quantanalyzer.data.processor import ProcessorChain, ProcessInf, CSZFillna, CSZScoreNorm
from quantanalyzer.factor import Alpha158Generator
from quantanalyzer.model.trainer import ModelTrainer


def create_sample_data():
    """创建示例数据用于测试"""
    print("📊 创建示例数据...")
    
    # 创建日期范围
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    symbols = ['TEST001']
    
    # 创建多索引
    index = pd.MultiIndex.from_product(
        [dates, symbols],
        names=['datetime', 'symbol']
    )
    
    # 生成模拟价格数据
    np.random.seed(42)
    n = len(index)
    
    # 生成趋势价格
    base_price = 100
    trend = np.linspace(0, 20, n)
    noise = np.random.randn(n) * 2
    close_prices = base_price + trend + noise
    
    data = pd.DataFrame({
        'open': close_prices * (1 + np.random.randn(n) * 0.01),
        'high': close_prices * (1 + np.abs(np.random.randn(n)) * 0.02),
        'low': close_prices * (1 - np.abs(np.random.randn(n)) * 0.02),
        'close': close_prices,
        'volume': np.random.randint(1000000, 10000000, n),
    }, index=index)
    
    print(f"✅ 创建了 {len(data)} 条数据")
    return data


def test_lightgbm_model():
    """测试LightGBM模型"""
    print("\n" + "="*60)
    print("🧪 测试 1: LightGBM模型")
    print("="*60)
    
    try:
        # 1. 创建数据
        data = create_sample_data()
        
        # 2. 生成Alpha158因子
        print("\n📊 生成Alpha158因子...")
        generator = Alpha158Generator(data)
        factors = generator.generate_all(
            kbar=True,
            price=True,
            volume=True,
            rolling=True,
            rolling_windows=[5, 10, 20]
        )
        print(f"✅ 生成了 {len(factors.columns)} 个因子")
        
        # 3. 数据清洗（单商品数据使用ZScoreNorm而不是CSZScoreNorm）
        print("\n🧹 数据清洗...")
        from quantanalyzer.data.processor import ZScoreNorm
        processor_chain = ProcessorChain([
            ProcessInf(),
            CSZFillna(),
            ZScoreNorm()  # 单商品数据使用ZScoreNorm，多商品数据使用CSZScoreNorm
        ])
        factors_clean = processor_chain.fit_transform(factors)
        print(f"✅ 清洗完成，NaN比例: {factors_clean.isna().sum().sum() / (factors_clean.shape[0] * factors_clean.shape[1]) * 100:.2f}%")
        
        # 4. 准备标签
        print("\n📈 准备训练标签...")
        labels = data['close'].groupby(level=1).pct_change().shift(-1)
        
        # 5. 训练LightGBM模型
        print("\n🤖 训练LightGBM模型...")
        trainer = ModelTrainer(model_type='lightgbm')
        
        X_train, y_train, X_test, y_test = trainer.prepare_dataset(
            factors_clean, labels,
            '2023-01-01', '2023-08-31',
            '2023-09-01', '2023-12-31'
        )
        
        print(f"训练集: {len(X_train)} 样本")
        print(f"测试集: {len(X_test)} 样本")
        
        trainer.train(X_train, y_train, X_test, y_test)
        
        # 6. 预测
        print("\n🔮 进行预测...")
        train_pred = trainer.predict(X_train)
        test_pred = trainer.predict(X_test)
        
        # 7. 评估
        from sklearn.metrics import mean_squared_error, r2_score
        train_mse = mean_squared_error(y_train, train_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        print("\n📊 模型性能:")
        print(f"训练集 - MSE: {train_mse:.6f}, R²: {train_r2:.4f}")
        print(f"测试集 - MSE: {test_mse:.6f}, R²: {test_r2:.4f}")
        
        # 8. 特征重要性
        if trainer.feature_importance is not None:
            print("\n🎯 Top 5 重要特征:")
            for i, (feat, imp) in enumerate(trainer.feature_importance.head(5).items(), 1):
                print(f"{i}. {feat}: {imp:.2f}")
        
        print("\n✅ LightGBM测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ LightGBM测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_xgboost_model():
    """测试XGBoost模型"""
    print("\n" + "="*60)
    print("🧪 测试 2: XGBoost模型")
    print("="*60)
    
    try:
        # 简化的测试流程
        data = create_sample_data()
        labels = data['close'].groupby(level=1).pct_change().shift(-1)
        
        print("\n🤖 训练XGBoost模型...")
        trainer = ModelTrainer(model_type='xgboost')
        
        # 使用简单特征
        simple_features = data[['open', 'high', 'low', 'close', 'volume']].copy()
        
        X_train, y_train, X_test, y_test = trainer.prepare_dataset(
            simple_features, labels,
            '2023-01-01', '2023-08-31',
            '2023-09-01', '2023-12-31'
        )
        
        print(f"训练集: {len(X_train)} 样本, 特征数: {X_train.shape[1]}")
        
        trainer.train(X_train, y_train, X_test, y_test, params={
            'max_depth': 3,
            'learning_rate': 0.1
        })
        
        # 预测
        test_pred = trainer.predict(X_test)
        
        from sklearn.metrics import r2_score
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\n📊 测试集R²: {test_r2:.4f}")
        print("✅ XGBoost测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ XGBoost测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_linear_model():
    """测试线性回归模型"""
    print("\n" + "="*60)
    print("🧪 测试 3: 线性回归模型")
    print("="*60)
    
    try:
        data = create_sample_data()
        labels = data['close'].groupby(level=1).pct_change().shift(-1)
        
        print("\n🤖 训练线性回归模型...")
        trainer = ModelTrainer(model_type='linear')
        
        # 使用简单特征
        simple_features = data[['open', 'high', 'low', 'volume']].copy()
        
        X_train, y_train, X_test, y_test = trainer.prepare_dataset(
            simple_features, labels,
            '2023-01-01', '2023-08-31',
            '2023-09-01', '2023-12-31'
        )
        
        trainer.train(X_train, y_train, params={'alpha': 1.0})
        
        # 预测
        test_pred = trainer.predict(X_test)
        
        from sklearn.metrics import r2_score
        test_r2 = r2_score(y_test, test_pred)
        
        print(f"\n📊 测试集R²: {test_r2:.4f}")
        print("✅ 线性回归测试通过!")
        return True
        
    except Exception as e:
        print(f"\n❌ 线性回归测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("🚀 机器学习模型功能测试")
    print("="*60)
    print("\n本脚本将测试以下功能:")
    print("1. LightGBM模型训练和预测")
    print("2. XGBoost模型训练和预测")
    print("3. 线性回归模型训练和预测")
    
    results = {
        'LightGBM': test_lightgbm_model(),
        'XGBoost': test_xgboost_model(),
        'Linear': test_linear_model()
    }
    
    # 总结
    print("\n" + "="*60)
    print("📋 测试总结")
    print("="*60)
    
    for model, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{model}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有测试通过!")
        print("\n💡 提示:")
        print("- LightGBM、XGBoost和sklearn已成功集成")
        print("- 不再依赖torch包")
        print("- 可以通过MCP工具使用这些模型")
    else:
        print("\n⚠️ 部分测试失败，请检查错误信息")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())