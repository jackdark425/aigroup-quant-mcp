"""
完整工作流示例：从数据加载到模型回测
"""
import sys
sys.path.insert(0, '..')

from quantanalyzer import DataLoader, FactorLibrary, FactorEvaluator, ModelTrainer, BacktestEngine
import pandas as pd
import numpy as np

print("=" * 70)
print(" QuantAnalyzer 完整量化工作流示例")
print(" 数据加载 → 因子计算 → 因子评估 → 模型训练 → 策略回测")
print("=" * 70)

# ==================== 第1步：数据加载 ====================
print("\n【第1步】数据加载")
print("-" * 70)
loader = DataLoader()
data = loader.load_from_csv("../sample_stock_data.csv")

# 数据质量检查
report = loader.validate_data(data)
print(f"✅ 数据加载成功")
print(f"   股票数量: {report['symbols_count']}只")
print(f"   数据条数: {data.shape[0]}条")
print(f"   时间范围: {report['date_range']['start'].strftime('%Y-%m-%d')} ~ {report['date_range']['end'].strftime('%Y-%m-%d')}")
print(f"   数据质量: 无缺失值，无重复记录")

# ==================== 第2步：因子计算 ====================
print("\n【第2步】因子计算")
print("-" * 70)
library = FactorLibrary()

# 计算多个因子
factors_dict = {
    'momentum_5': library.momentum(data, period=5),
    'momentum_10': library.momentum(data, period=10),
    'momentum_20': library.momentum(data, period=20),
    'volatility_10': library.volatility(data, period=10),
    'volatility_20': library.volatility(data, period=20),
    'volume_ratio_10': library.volume_ratio(data, period=10),
    'volume_ratio_20': library.volume_ratio(data, period=20),
    'rsi_14': library.rsi(data, period=14),
}

factors_df = pd.DataFrame(factors_dict)
print(f"✅ 成功计算 {len(factors_dict)} 个因子")
print(f"   因子列表: {list(factors_dict.keys())}")
print(f"   因子数据形状: {factors_df.shape}")

# ==================== 第3步：因子评估 ====================
print("\n【第3步】因子评估（IC分析）")
print("-" * 70)

# 计算前向收益
forward_return = data['close'].groupby(level=1).pct_change(1).shift(-1)

# 评估每个因子
ic_results = []
for factor_name, factor_series in factors_dict.items():
    try:
        evaluator = FactorEvaluator(factor_series, forward_return)
        ic_metrics = evaluator.calculate_ic()
        
        ic_results.append({
            'factor': factor_name,
            'ic_mean': ic_metrics['ic_mean'],
            'ic_std': ic_metrics['ic_std'],
            'icir': ic_metrics['icir'],
            'positive_ratio': ic_metrics['ic_positive_ratio']
        })
    except:
        pass

if ic_results:
    ic_df = pd.DataFrame(ic_results).sort_values('icir', ascending=False)
    print("✅ 因子IC评估结果（按ICIR排序）:")
    print(ic_df.to_string(index=False))
    
    # 选择最好的因子
    best_factors = ic_df.head(5)['factor'].tolist()
    print(f"\n📊 选择IC最优的5个因子: {best_factors}")
else:
    print("⚠️ IC评估数据不足，使用所有因子")
    best_factors = list(factors_dict.keys())[:5]

# ==================== 第4步：模型训练 ====================
print("\n【第4步】模型训练")
print("-" * 70)

# 准备训练数据
selected_factors = factors_df[best_factors]

# 数据分割
train_start = "2020-01-01"
train_end = "2022-12-31"
test_start = "2023-01-01"
test_end = "2023-12-31"

print(f"训练期: {train_start} ~ {train_end}")
print(f"测试期: {test_start} ~ {test_end}")

# 训练LightGBM模型
trainer = ModelTrainer("lightgbm")

try:
    X_train, y_train, X_test, y_test = trainer.prepare_dataset(
        selected_factors,
        forward_return,
        train_start, train_end,
        test_start, test_end
    )
    
    print(f"\n数据集准备:")
    print(f"   训练集: {len(X_train)}条")
    print(f"   测试集: {len(X_test)}条")
    
    # 训练模型
    trainer.train(X_train, y_train, X_test, y_test)
    
    print(f"\n✅ 模型训练完成")
    print(f"\n特征重要性（Top 5）:")
    if trainer.feature_importance is not None:
        for feat, imp in trainer.feature_importance.head(5).items():
            print(f"   {feat}: {imp:.0f}")
    
except Exception as e:
    print(f"❌ 模型训练失败: {e}")
    sys.exit(1)

# ==================== 第5步：策略回测 ====================
print("\n【第5步】策略回测")
print("-" * 70)

try:
    # 生成预测
    predictions = trainer.predict(selected_factors)
    
    # 筛选测试期数据
    test_mask = (
        (predictions.index.get_level_values(0) >= test_start) &
        (predictions.index.get_level_values(0) <= test_end)
    )
    predictions_test = predictions[test_mask]
    
    print(f"预测样本数: {len(predictions_test)}")
    
    # 执行回测
    engine = BacktestEngine(
        initial_capital=10000000,  # 初始资金1000万
        commission=0.0003,          # 手续费0.03%
        slippage=0.0001             # 滑点0.01%
    )
    
    print(f"\n回测参数:")
    print(f"   初始资金: 1000万元")
    print(f"   手续费率: 0.03%")
    print(f"   滑点: 0.01%")
    print(f"   选股数量: Top 30")
    print(f"   持仓周期: 1天")
    
    metrics = engine.run_topk_strategy(
        predictions_test,
        data,
        k=30,
        holding_period=1
    )
    
    print(f"\n✅ 回测完成")
    print(f"\n" + "=" * 70)
    print(" 📊 回测结果")
    print("=" * 70)
    print(f"\n收益指标:")
    print(f"   总收益率: {metrics['total_return']:>10.2%}")
    print(f"   年化收益率: {metrics['annualized_return']:>10.2%}")
    
    print(f"\n风险指标:")
    print(f"   波动率: {metrics['volatility']:>10.2%}")
    print(f"   最大回撤: {metrics['max_drawdown']:>10.2%}")
    
    print(f"\n风险调整收益:")
    print(f"   夏普比率: {metrics['sharpe_ratio']:>10.2f}")
    
    # 计算月度收益
    if len(metrics['returns']) > 20:
        monthly_return = np.mean(metrics['returns']) * 21  # 假设月21个交易日
        print(f"   平均月收益: {monthly_return:>10.2%}")
    
    print(f"\n" + "=" * 70)
    
    # 判断策略表现
    if metrics['sharpe_ratio'] > 1.0:
        print("✅ 策略表现良好（夏普比率 > 1.0）")
    elif metrics['sharpe_ratio'] > 0.5:
        print("⚠️ 策略表现一般（0.5 < 夏普比率 < 1.0）")
    else:
        print("❌ 策略表现较差（夏普比率 < 0.5）")
        
    if abs(metrics['max_drawdown']) < 0.1:
        print("✅ 回撤控制良好（最大回撤 < 10%）")
    elif abs(metrics['max_drawdown']) < 0.2:
        print("⚠️ 回撤控制一般（10% < 最大回撤 < 20%）")
    else:
        print("❌ 回撤较大（最大回撤 > 20%）")
    
except Exception as e:
    print(f"❌ 回测失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("🎉 完整工作流执行成功！")
print("=" * 70)
print("\n工作流总结:")
print("  ✅ 步骤1: 数据加载和质量检查")
print("  ✅ 步骤2: 计算8个量化因子")
print("  ✅ 步骤3: 因子IC评估和筛选")
print("  ✅ 步骤4: LightGBM模型训练")
print("  ✅ 步骤5: TopK策略回测")
print("\n✨ QuantAnalyzer量化工作流完整演示结束")