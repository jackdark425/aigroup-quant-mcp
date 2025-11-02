"""
测试Alpha158因子和深度学习模型
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from quantanalyzer.factor import Alpha158Generator
# from quantanalyzer.model import LSTMModel, GRUModel, TransformerModel  # 深度学习模型已移除

print("=" * 80)
print("测试Alpha158因子和深度学习模型")
print("=" * 80)

# 1. 加载真实数据
print("\n1. 加载真实数据...")
data_path = os.path.join(os.path.dirname(__file__), '..', 'real_data_2stocks.csv')
if not os.path.exists(data_path):
    print(f"错误: 数据文件不存在 {data_path}")
    sys.exit(1)

df = pd.read_csv(data_path)
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index(['datetime', 'symbol']).sort_index()

print(f"数据形状: {df.shape}")
print(f"列: {list(df.columns)}")
print(f"数据样本:\n{df.head()}")

# 2. 测试Alpha158因子生成
print("\n" + "=" * 80)
print("2. 测试Alpha158因子生成")
print("=" * 80)

print("\n2.1 生成K线形态因子...")
generator = Alpha158Generator(df)
kbar_factors = generator._generate_kbar_features()
print(f"K线形态因子数量: {kbar_factors.shape[1]}")
print(f"因子名称: {list(kbar_factors.columns)}")
print(f"统计信息:\n{kbar_factors.describe()}")

print("\n2.2 生成完整Alpha158因子（小窗口测试）...")
alpha158_small = generator.generate_all(
    kbar=True,
    price=True,
    volume=True,
    rolling=True,
    rolling_windows=[5, 10]  # 仅用小窗口测试
)
print(f"Alpha158因子总数: {alpha158_small.shape[1]}")
print(f"因子形状: {alpha158_small.shape}")
print(f"前10个因子: {list(alpha158_small.columns[:10])}")

# 检查空值
null_counts = alpha158_small.isna().sum()
print(f"\n空值统计:")
print(f"  总空值数: {null_counts.sum()}")
print(f"  有空值的因子数: {(null_counts > 0).sum()}")

# 3. 准备深度学习数据
print("\n" + "=" * 80)
print("3. 准备深度学习训练数据")
print("=" * 80)

# 使用K线形态因子作为特征
features = kbar_factors.copy()

# 生成标签（未来1日收益率）
labels = df['close'].groupby(level=1).pct_change(1).shift(-1)

# 删除NaN
valid_idx = ~(features.isna().any(axis=1) | labels.isna())
X = features[valid_idx]
y = labels[valid_idx]

print(f"有效样本数: {len(X)}")
print(f"特征维度: {X.shape[1]}")

# 划分训练集和测试集
split_idx = int(len(X) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")

# 4. 测试LSTM模型
print("\n" + "=" * 80)
print("4. 测试LSTM模型")
print("=" * 80)

print("\n4.1 创建LSTM模型...")
# lstm_model = LSTMModel(
#     d_feat=X_train.shape[1],  # 深度学习模型已移除
    hidden_size=32,
    num_layers=1,
    n_epochs=20,
    batch_size=16,
    lr=0.001,
    early_stop=5,
    device='cpu'
)

print(f"模型参数:")
print(f"  输入维度: {lstm_model.d_feat}")
print(f"  隐藏层大小: {lstm_model.hidden_size}")
print(f"  层数: {lstm_model.num_layers}")

print("\n4.2 训练LSTM模型...")
history = lstm_model.fit(X_train, y_train, X_test, y_test)

print(f"\n训练完成:")
print(f"  最佳轮次: {history['best_epoch']}")
print(f"  最佳验证损失: {history['best_loss']:.6f}")

print("\n4.3 LSTM预测...")
lstm_pred = lstm_model.predict(X_test)
print(f"预测结果:")
print(f"  均值: {lstm_pred.mean():.6f}")
print(f"  标准差: {lstm_pred.std():.6f}")
print(f"  最小值: {lstm_pred.min():.6f}")
print(f"  最大值: {lstm_pred.max():.6f}")

# 计算预测与实际的相关性
corr = lstm_pred.corr(y_test)
print(f"  与真实值相关性: {corr:.4f}")

# 5. 测试GRU模型
print("\n" + "=" * 80)
print("5. 测试GRU模型")
print("=" * 80)

print("\n5.1 创建GRU模型...")
# gru_model = GRUModel(
#     d_feat=X_train.shape[1],  # 深度学习模型已移除
    hidden_size=32,
    num_layers=1,
    n_epochs=20,
    batch_size=16,
    lr=0.001,
    early_stop=5,
    device='cpu'
)

print("\n5.2 训练GRU模型...")
history = gru_model.fit(X_train, y_train, X_test, y_test)

print(f"\n训练完成:")
print(f"  最佳轮次: {history['best_epoch']}")
print(f"  最佳验证损失: {history['best_loss']:.6f}")

print("\n5.3 GRU预测...")
gru_pred = gru_model.predict(X_test)
corr = gru_pred.corr(y_test)
print(f"  与真实值相关性: {corr:.4f}")

# 6. 测试Transformer模型
print("\n" + "=" * 80)
print("6. 测试Transformer模型")
print("=" * 80)

print("\n6.1 创建Transformer模型...")
# transformer_model = TransformerModel(
#     d_feat=X_train.shape[1],  # 深度学习模型已移除
    d_model=32,
    nhead=4,
    num_layers=1,
    n_epochs=20,
    batch_size=16,
    lr=0.001,
    early_stop=5,
    device='cpu'
)

print("\n6.2 训练Transformer模型...")
history = transformer_model.fit(X_train, y_train, X_test, y_test)

print(f"\n训练完成:")
print(f"  最佳轮次: {history['best_epoch']}")
print(f"  最佳验证损失: {history['best_loss']:.6f}")

print("\n6.3 Transformer预测...")
transformer_pred = transformer_model.predict(X_test)
corr = transformer_pred.corr(y_test)
print(f"  与真实值相关性: {corr:.4f}")

# 7. 模型对比
print("\n" + "=" * 80)
print("7. 模型性能对比")
print("=" * 80)

models = {
    'LSTM': lstm_pred,
    'GRU': gru_pred,
    'Transformer': transformer_pred
}

print("\n预测相关性对比:")
for name, pred in models.items():
    corr = pred.corr(y_test)
    mse = ((pred - y_test) ** 2).mean()
    mae = (pred - y_test).abs().mean()
    print(f"{name:12s} - 相关性: {corr:7.4f}, MSE: {mse:.6f}, MAE: {mae:.6f}")

print("\n" + "=" * 80)
print("✅ 所有测试完成！")
print("=" * 80)