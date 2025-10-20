# QuantAnalyzer - 量化分析工具包 v2.0

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> 参考Qlib架构的轻量级量化分析Python包，支持Alpha158因子库和深度学习模型

## 🌟 v2.0 新特性

### 📊 Alpha158因子库
- ✨ **158个技术指标因子** - 完整的Qlib级因子库
- 🎯 K线形态（9个）+ 价格（5个）+ 成交量（5个）+ 滚动统计（139个）
- 🔧 灵活配置，支持自定义窗口

### 🤖 深度学习模型
- 🚀 **LSTM** - 长期依赖建模
- ⚡ **GRU** - 高效序列学习
- 🎨 **Transformer** - 并行注意力机制

### 🛠️ 扩展MCP服务
- 📈 从5个工具扩展到**11个专业量化工具**
- 🔌 完整的因子生成、模型训练、预测流程
- 💡 即插即用的Roo-Code集成

---

## 📦 快速开始

### 安装

```bash
# 使用pip安装
pip install aigroup-quant-mcp

# 或者使用uv安装
uv pip install aigroup-quant-mcp
```

### 基础使用

```python
from quantanalyzer.data import DataLoader
from quantanalyzer.factor import Alpha158Generator
from quantanalyzer.model import LSTMModel

# 1. 加载数据
loader = DataLoader()
data = loader.load_from_csv("your_data.csv")

# 2. 生成Alpha158因子
generator = Alpha158Generator(data)
factors = generator.generate_all(rolling_windows=[5, 10, 20])

# 3. 训练LSTM模型
model = LSTMModel(d_feat=factors.shape[1], n_epochs=50)
history = model.fit(X_train, y_train, X_valid, y_valid)

# 4. 预测
predictions = model.predict(X_test)
```

### 使用MCP服务

```bash
# 使用python直接运行
python -m quantanalyzer.mcp

# 或者使用uvx运行（推荐）
uvx aigroup-quant-mcp

# 或者安装后直接运行
aigroup-quant-mcp
```

---

## 🎯 核心功能

### 1️⃣ 数据处理

```python
from quantanalyzer.data import DataLoader, DataProcessor

# 加载CSV
loader = DataLoader()
data = loader.load_from_csv("stock_data.csv")

# 数据预处理
processor = DataProcessor()
processed = processor.fillna(data)
normalized = processor.normalize(processed)
```

### 2️⃣ 因子计算

#### 基础因子库（v1.0）
```python
from quantanalyzer.factor import FactorLibrary

library = FactorLibrary()

# 6个基础因子
momentum = library.momentum(data, period=20)
volatility = library.volatility(data, period=20)
volume_ratio = library.volume_ratio(data, period=20)
rsi = library.rsi(data, period=14)
macd = library.macd(data)
bb = library.bollinger_bands(data, period=20)
```

#### Alpha158因子库（v2.0 新增）
```python
from quantanalyzer.factor import Alpha158Generator

# 生成158个技术指标
generator = Alpha158Generator(data)

# 完整因子集
alpha158_full = generator.generate_all()

# 自定义配置
alpha158_custom = generator.generate_all(
    kbar=True,                    # K线形态（9个）
    price=True,                   # 价格（5个）
    volume=True,                  # 成交量（5个）
    rolling=True,                 # 滚动统计（139个）
    rolling_windows=[5, 10, 20]   # 自定义窗口
)

# 仅K线因子
kbar_only = generator.generate_all(
    kbar=True, price=False, volume=False, rolling=False
)
```

### 3️⃣ 因子评估

```python
from quantanalyzer.factor import FactorEvaluator

# 计算收益率
returns = data['close'].groupby(level=1).pct_change(1).shift(-1)

# 评估IC
evaluator = FactorEvaluator(factor, returns)
ic_metrics = evaluator.calculate_ic(method='spearman')

print(f"IC均值: {ic_metrics['ic_mean']:.4f}")
print(f"ICIR: {ic_metrics['icir']:.4f}")
```

### 4️⃣ 模型训练

#### 传统模型（v1.0）
```python
from quantanalyzer.model import ModelTrainer

# LightGBM/XGBoost/Linear
trainer = ModelTrainer(model_type='lightgbm')
trainer.fit(X_train, y_train)
predictions = trainer.predict(X_test)
```

#### 深度学习模型（v2.0 新增）
```python
from quantanalyzer.model import LSTMModel, GRUModel, TransformerModel

# LSTM
lstm = LSTMModel(d_feat=158, hidden_size=64, n_epochs=100)
lstm_history = lstm.fit(X_train, y_train, X_valid, y_valid)
lstm_pred = lstm.predict(X_test)

# GRU
gru = GRUModel(d_feat=158, hidden_size=64)
gru_history = gru.fit(X_train, y_train, X_valid, y_valid)

# Transformer
transformer = TransformerModel(d_feat=158, d_model=64, nhead=4)
transformer_history = transformer.fit(X_train, y_train, X_valid, y_valid)
```

### 5️⃣ 回测引擎

```python
from quantanalyzer.backtest import BacktestEngine

# TopK策略回测
engine = BacktestEngine(
    data=data,
    predictions=predictions,
    top_k=10,
    commission=0.0015,
    slippage=0.001
)

result = engine.run()
print(f"总收益: {result['total_return']:.2%}")
print(f"夏普比率: {result['sharpe_ratio']:.2f}")
```

---

## 🔌 MCP服务

### 启动MCP服务

MCP服务支持多种启动方式：

#### 使用uvx（推荐）
```bash
# 直接运行，无需安装
uvx aigroup-quant-mcp

# 指定版本运行
uvx aigroup-quant-mcp==1.0.0
```

#### 使用Python直接运行
```bash
# 克隆项目后直接运行
python -m quantanalyzer.mcp

# 或者
python mcp_server.py
```

#### 安装后运行
```bash
# 安装
pip install aigroup-quant-mcp

# 运行
aigroup-quant-mcp
```

MCP服务已自动配置到Roo-Code，包含**11个专业量化工具**：

#### 数据工具
1. `load_csv_data` - 加载CSV数据

#### 因子工具
2. `calculate_factor` - 计算基础因子（6种）
3. `generate_alpha158` - 生成Alpha158因子集（158个）
4. `evaluate_factor_ic` - 评估因子IC

#### 模型工具
5. `train_lstm_model` - 训练LSTM模型
6. `train_gru_model` - 训练GRU模型
7. `train_transformer_model` - 训练Transformer模型
8. `predict_with_model` - 模型预测

#### 信息工具
9. `get_data_info` - 数据信息
10. `list_factors` - 因子列表
11. `list_models` - 模型列表

### MCP使用流程

```bash
# 使用uvx启动服务后，可通过MCP工具进行完整分析流程：

1. load_csv_data          # 加载数据
2. generate_alpha158      # 生成因子
3. train_lstm_model       # 训练模型
4. predict_with_model     # 预测
5. evaluate_factor_ic     # 评估
```

---

## 📂 项目结构

```
quantanalyzer-project/
├── quantanalyzer/              # 核心包
│   ├── data/                   # 数据层
│   │   ├── loader.py          # 数据加载
│   │   └── processor.py       # 数据处理
│   ├── factor/                 # 因子层
│   │   ├── library.py         # 基础因子库（6个）
│   │   ├── alpha158.py        # ⭐ Alpha158因子（158个）
│   │   └── evaluator.py       # 因子评估
│   ├── model/                  # 模型层
│   │   ├── trainer.py         # 传统模型
│   │   └── deep_models.py     # ⭐ 深度学习模型
│   └── backtest/               # 回测层
│       └── engine.py          # 回测引擎
├── examples/                   # 示例脚本
│   ├── test_basic_functions.py
│   ├── complete_workflow.py
│   ├── analyze_real_data.py
│   └── test_alpha158_and_dl.py  # ⭐ 新增
├── quantanalyzer/mcp.py       # ⭐ MCP服务器（11工具）
├── pyproject.toml             # ⭐ 项目配置（支持uv/uvx）
├── setup.py                   # ⭐ 安装配置
└── 功能扩展说明.md            # ⭐ 新增文档
```

---

## ▶️ 测试运行说明

### 运行完整测试

项目包含完整的测试脚本，用于验证所有核心功能：

```bash
# 运行Alpha158因子和深度学习模型测试
python examples/test_alpha158_and_dl.py
```

### 预期输出

**Alpha158因子测试结果**（44条真实行情数据）：
- ✅ K线形态因子：9个，正常
- ✅ Alpha158因子：72个（小窗口），正常
- ✅ 空值率：~8.1%（符合预期）

**深度学习模型性能对比**：

| 模型 | 训练轮次 | 验证损失 | 测试相关性 | 推荐 |
|------|---------|----------|-----------|------|
| **LSTM** | 20轮早停 | 1049.49 | **0.8655** | ⭐⭐⭐ |
| GRU | 20轮早停 | 1048.97 | -0.3271 | ⚠️ 需更多数据 |
| Transformer | 20轮早停 | 1045.70 | -0.5907 | ⚠️ 需更多数据 |

> **提示**：在小样本数据上，LSTM表现最佳。如需提升GRU和Transformer性能，建议使用更大规模的数据集。

### 其他测试示例

```bash
# 基础功能测试
python examples/test_basic_functions.py

# 完整工作流测试
python examples/complete_workflow.py

# 真实数据分析测试
python examples/analyze_real_data.py
```

### 测试环境要求

- Python 3.8+
- PyTorch 2.0+
- pandas, numpy, scikit-learn
- matplotlib (可视化)

### 故障排除

如果测试失败，请检查：
1. 依赖包是否完整安装
2. 数据文件路径是否正确
3. GPU可用性（如使用CUDA）
4. 内存是否充足

---

## 📊 功能验证

**性能对比**：

| 模型 | 训练轮次 | 验证损失 | 测试相关性 | 推荐 |
|------|---------|----------|-----------|------|
| **LSTM** | 20轮早停 | 1049.49 | **0.8655** | ⭐⭐⭐ |
| GRU | 20轮早停 | 1048.97 | -0.3271 | ⚠️ 需更多数据 |
| Transformer | 20轮早停 | 1045.70 | -0.5907 | ⚠️ 需更多数据 |

**结论**：LSTM在小样本下表现最佳。

---

## 🎓 使用场景

### 场景1：Alpha因子挖掘

```python
# 生成Alpha158因子
generator = Alpha158Generator(data)
factors = generator.generate_all()

# 评估每个因子
for factor_name in factors.columns:
    evaluator = FactorEvaluator(factors[factor_name], returns)
    ic_metrics = evaluator.calculate_ic()
    print(f"{factor_name}: IC={ic_metrics['ic_mean']:.4f}")

# 选择高IC因子构建组合
```

### 场景2：机器学习选股

```python
# 使用Alpha158因子训练模型
X, y = factors, returns
model = LSTMModel(d_feat=X.shape[1])
model.fit(X_train, y_train, X_valid, y_valid)

# 预测未来收益
predictions = model.predict(X_test)

# 构建投资组合
top_stocks = predictions.nlargest(10)
```

### 场景3：策略回测

```python
# 基于预测结果进行回测
engine = BacktestEngine(
    data=test_data,
    predictions=predictions,
    top_k=20,
    commission=0.001
)

results = engine.run()
print(f"年化收益: {results['annual_return']:.2%}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
```

---

## 🔧 配置参数

### Alpha158生成配置

```python
config = {
    'kbar': True,                    # K线形态因子
    'price': True,                   # 价格因子
    'volume': True,                  # 成交量因子
    'rolling': True,                 # 滚动统计因子
    'rolling_windows': [5, 10, 20, 30, 60]  # 窗口大小
}
```

### 深度学习模型配置

#### LSTM配置
```python
lstm_config = {
    'd_feat': 158,          # 输入特征维度
    'hidden_size': 64,      # 隐藏层大小
    'num_layers': 2,        # LSTM层数
    'dropout': 0.1,         # Dropout率
    'n_epochs': 100,        # 训练轮数
    'lr': 0.001,            # 学习率
    'batch_size': 800,      # 批次大小
    'early_stop': 20,       # 早停轮数
    'device': 'cuda'        # 使用GPU
}
```

#### Transformer配置
```python
transformer_config = {
    'd_feat': 158,          # 输入特征维度
    'd_model': 64,          # Transformer维度
    'nhead': 4,             # 注意力头数
    'num_layers': 2,        # Transformer层数
    'dropout': 0.1,
    'n_epochs': 100,
    'lr': 0.001,
    'batch_size': 800,
    'early_stop': 20
}
```

---

## 📚 文档

### 核心文档
- [功能扩展说明.md](功能扩展说明.md) - v2.0新功能详细说明
- [MCP_使用说明.md](MCP_使用说明.md) - MCP服务使用指南
- [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) - 项目总结

### 示例代码
- `examples/test_basic_functions.py` - 基础功能测试
- `examples/complete_workflow.py` - 完整工作流
- `examples/analyze_real_data.py` - 真实数据分析
- `examples/test_alpha158_and_dl.py` - Alpha158和深度学习测试

---

## 🎯 核心模块

### 数据模块 (`quantanalyzer.data`)
- `DataLoader` - CSV数据加载
- `DataProcessor` - 数据预处理（填充、标准化、异常值）

### 因子模块 (`quantanalyzer.factor`)
- `FactorLibrary` - 6个基础因子
- `Alpha158Generator` - 158个Alpha因子 ⭐ v2.0
- `FactorEvaluator` - IC/ICIR评估

### 模型模块 (`quantanalyzer.model`)
- `ModelTrainer` - 传统ML模型（LightGBM/XGBoost/Linear）
- `LSTMModel` - LSTM深度学习 ⭐ v2.0
- `GRUModel` - GRU深度学习 ⭐ v2.0
- `TransformerModel` - Transformer ⭐ v2.0

### 回测模块 (`quantanalyzer.backtest`)
- `BacktestEngine` - TopK策略回测

---

## 🔬 技术规格

### 支持的因子类型

| 类别 | 数量 | 示例 |
|------|------|------|
| K线形态 | 9 | KMID, KLEN, KUP, KLOW |
| 价格 | 5 | OPEN0, HIGH0, LOW0, CLOSE0 |
| 成交量 | 5 | VOLUME0-4 |
| 趋势 | 30+ | ROC, MA, BETA, RSQR |
| 波动 | 10+ | STD, RESI |
| 极值 | 20+ | MAX, MIN, QTLU, QTLD |
| 位置 | 15+ | RANK, RSV, IMAX, IMIN |
| 相关 | 10+ | CORR, CORD |
| 统计 | 40+ | CNTP, SUMP, VMA, WVMA |

### 模型架构对比

| 特性 | LSTM | GRU | Transformer |
|------|------|-----|-------------|
| 参数量 | 中 | 小 | 大 |
| 训练速度 | 慢 | 快 | 中 |
| 长期记忆 | 强 | 中 | 强 |
| 并行化 | 否 | 否 | 是 |
| 小样本 | ⭐⭐⭐ | ⭐⭐ | ⭐ |
| 大样本 | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ |

---

## 🚀 性能优化

### GPU加速

```python
# 启用GPU训练
model = LSTMModel(
    d_feat=158,
    device='cuda'  # 使用GPU
)

# 或自动检测
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = LSTMModel(d_feat=158, device=device)
```

### 批处理优化

```python
# 大数据集建议增加批次大小
model = LSTMModel(
    d_feat=158,
    batch_size=2000,  # 增大批次
    n_epochs=50       # 减少轮数
)
```

---

## 📈 实际应用

### 应用案例1：选股策略

```python
# 1. 生成Alpha158因子
factors = Alpha158Generator(data).generate_all()

# 2. 训练LSTM选股模型
model = LSTMModel(d_feat=158, n_epochs=100)
model.fit(X_train, y_train, X_valid, y_valid)

# 3. 预测未来收益
predictions = model.predict(X_test)

# 4. TopK回测
from quantanalyzer.backtest import BacktestEngine
engine = BacktestEngine(data, predictions, top_k=20)
result = engine.run()
```

### 应用案例2：因子挖掘

```python
# 生成所有因子
alpha158 = Alpha158Generator(data).generate_all()

# 评估每个因子
results = []
for col in alpha158.columns:
    evaluator = FactorEvaluator(alpha158[col], returns)
    ic = evaluator.calculate_ic()
    results.append({
        'factor': col,
        'ic_mean': ic['ic_mean'],
        'icir': ic['icir']
    })

# 选择Top因子
top_factors = sorted(results, key=lambda x: abs(x['icir']), reverse=True)[:20]
```

---

## 🎨 版本历史

### v2.0 (2025-01-16) - 重大功能扩展

**新增**：
- ✨ Alpha158因子库（158个技术指标）
- 🤖 深度学习模型（LSTM/GRU/Transformer）
- 🛠️ MCP工具扩展（11个工具）
- 📦 PyTorch集成
- 📚 完整测试和文档

**改进**：
- 🔧 因子计算性能优化
- 📊 索引对齐问题修复
- 📝 文档完善

### v1.0 (2025-01-15) - 初始版本

**功能**：
- ✅ 基础因子库（6个）
- ✅ 传统ML模型（LightGBM/XGBoost/Linear）
- ✅ 因子评估（IC/ICIR）
- ✅ TopK回测
- ✅ MCP服务（5个工具）

---

## 🧪 测试

项目包含单元测试，确保各模块功能正常。

### 运行测试

```bash
# 使用Python unittest运行所有测试
python -m unittest discover tests

# 或者运行特定模块的测试
python -m unittest tests.test_data
python -m unittest tests.test_factor
python -m unittest tests.test_model
```

### 测试覆盖率

```bash
# 使用coverage.py运行测试并生成覆盖率报告
pip install coverage
coverage run -m unittest discover tests
coverage report
coverage html  # 生成HTML格式的详细报告
```

---

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

---

## 🙏 鸣谢

- [Qlib](https://github.com/microsoft/qlib) - 参考架构
- [MCP](https://github.com/microsoft/mcp) - 工具协议
- [PyTorch](https://pytorch.org/) - 深度学习框架
```

```

```

```
