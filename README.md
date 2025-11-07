# aigroup-quant-mcp - AI量化分析MCP服务

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![MCP](https://img.shields.io/badge/MCP-1.0+-green.svg)](https://modelcontextprotocol.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PyPI](https://img.shields.io/badge/PyPI-v1.0.34-blue.svg)](https://pypi.org/project/aigroup-quant-mcp/)

> 🎯 **专业量化分析MCP服务** - 提供完整的量化分析工作流，支持15种机器学习算法和Alpha158因子库

---

## ✨ 核心特性

### 🚀 完整量化工作流
- **数据预处理**: 智能中英文列名识别，自动数据清洗
- **因子计算**: 支持单因子和Alpha158因子库（158个技术因子）
- **因子评估**: IC评估和因子质量分析
- **机器学习建模**: 15种传统机器学习算法
- **模型预测**: 样本外预测和结果导出

### 🤖 丰富的机器学习算法
- **线性模型**: linear, ridge, lasso, elasticnet, logistic
- **基于树的模型**: lightgbm, xgboost, catboost, random_forest, gradient_boosting, decision_tree
- **支持向量机**: svm, svr
- **其他算法**: naive_bayes, knn

### 🧠 智能数据处理
- **智能标准化**: 自动识别单商品/多商品数据，优化标准化方法
- **中英文列名识别**: 自动识别和转换各种列名格式
- **数据质量检查**: 自动检测和处理缺失值、异常值

---

## 🛠️ 快速开始

### 安装方式

#### 方式1: 使用uvx（推荐）
```bash
uvx aigroup-quant-mcp
```

#### 方式2: 使用pip
```bash
# 基础安装（快速启动）
pip install aigroup-quant-mcp

# 安装机器学习支持
pip install aigroup-quant-mcp[ml]

# 完整安装（所有功能）
pip install aigroup-quant-mcp[full]
```

### MCP配置

在RooCode的设置中添加以下配置：

```json
{
  "mcpServers": {
    "aigroup-quant-mcp": {
      "command": "uvx",
      "args": [
        "aigroup-quant-mcp"
      ],
      "env": {},
      "alwaysAllow": [
        "preprocess_data",
        "calculate_factor",
        "generate_alpha158",
        "merge_factor_data",
        "evaluate_factor_ic",
        "apply_processor_chain",
        "train_ml_model",
        "predict_ml_model",
        "list_factors"
      ]
    }
  }
}
```

---

## 📊 完整工作流示例

### 步骤1: 数据预处理
```python
preprocess_data(
    file_path="./data/stock_data.csv",
    data_id="stock_2023",
    auto_clean=True,
    export_path="./exports/cleaned_data.csv"
)
```

### 步骤2: 生成Alpha158因子
```python
generate_alpha158(
    data_id="stock_2023",
    result_id="alpha158_factors",
    kbar=True,
    price=True,
    volume=True,
    rolling=True,
    rolling_windows=[5, 10, 20, 30, 60],
    export_path="./exports/alpha158_factors.csv"
)
```

### 步骤3: 智能标准化
```python
apply_processor_chain(
    data_id="alpha158_factors",
    result_id="alpha158_normalized",
    processors=[{"name": "CSZScoreNorm"}],  # 智能标准化，自动适配单/多商品
    export_path="./exports/factors_normalized.csv"
)
```

### 步骤4: 合并因子和价格数据
```python
merge_factor_data(
    factor_data_id="alpha158_normalized",
    price_data_id="stock_2023",
    result_id="merged_for_training",
    export_path="./exports/training_data.csv"
)
```

### 步骤5: 训练机器学习模型
```python
train_ml_model(
    data_id="merged_for_training",  # 或使用原始数据ID
    model_id="lgb_model_v1",
    model_type="lightgbm",
    train_start="2023-01-01",
    train_end="2023-06-30",
    test_start="2023-07-01",
    test_end="2023-12-31",
    params={
        "learning_rate": 0.05,
        "num_leaves": 31
    }
)
```

### 步骤6: 模型预测
```python
predict_ml_model(
    model_id="lgb_model_v1",
    data_id="alpha158_normalized",
    export_path="./exports/predictions.csv"
)
```

---

## 🔧 可用工具

### 数据预处理
- **`preprocess_data`**: 数据加载和智能清洗
  - 中英文列名自动识别
  - 自动处理缺失值和异常值
  - 支持数据导出

### 因子计算
- **`calculate_factor`**: 单因子计算
  - 支持动量、波动率、成交量比率、RSI、MACD、布林带等因子
- **`generate_alpha158`**: Alpha158因子库
  - 158个技术因子完整实现
  - 支持K线形态、价格特征、成交量特征、滚动统计

### 数据处理
- **`apply_processor_chain`**: 数据处理器链
  - 智能标准化（自动适配单/多商品）
  - 支持CSZScoreNorm、ZScoreNorm、CSZFillna、ProcessInf等处理器
- **`merge_factor_data`**: 合并因子和价格数据
  - 解决因子数据无法直接训练的问题

### 因子评估
- **`evaluate_factor_ic`**: 因子IC评估
  - 计算IC均值、IC标准差、ICIR
  - 生成详细的Markdown评估报告

### 机器学习
- **`train_ml_model`**: 训练机器学习模型
  - 支持15种传统机器学习算法
  - 完整的训练和评估指标
  - 特征重要性分析
- **`predict_ml_model`**: 使用模型预测
  - 样本外预测
  - 结果导出

### 辅助工具
- **`list_factors`**: 查看已加载的数据和因子
  - 内存状态管理
  - 数据ID列表

---

## 🎯 算法选择指南

### 首次使用（推荐）
- **LightGBM**: 速度快，效果好，内存占用小
- **随机森林**: 稳定性好，参数简单

### 特征选择
- **Lasso回归**: 自动特征筛选
- **弹性网络**: 平衡特征选择

### 高精度需求
- **XGBoost**: 性能强大，精度最高
- **CatBoost**: 处理类别特征

### 快速验证
- **线性回归**: 速度极快，基线模型
- **决策树**: 可解释性强

---

## 📈 性能指标

### 因子质量评估
| IC绝对值 | 预测能力 | 推荐建议 |
|---------|---------|---------|
| > 0.10 | 非常强 | 强烈推荐使用 |
| 0.08-0.10 | 强 | 推荐使用 |
| 0.05-0.08 | 较强 | 可以使用 |
| 0.03-0.05 | 有效 | 谨慎使用，建议组合 |
| < 0.03 | 无效 | 不推荐使用 |

### 模型性能评估
| R²值 | 模型质量 | IC值 | 预测能力 |
|-----|---------|------|---------|
| > 0.7 | 优秀 | > 0.08 | 强 |
| 0.5-0.7 | 良好 | 0.05-0.08 | 较强 |
| 0.3-0.5 | 一般 | 0.03-0.05 | 有效 |
| < 0.3 | 较差 | < 0.03 | 无效 |

---

## 🧪 示例和测试

项目包含完整的示例代码：

```bash
# 运行完整工作流示例
python examples/complete_workflow.py

# 测试机器学习模型
python examples/test_ml_models.py

# 测试Alpha158因子
python examples/test_alpha158_and_dl.py
```

---

## 📚 文档资源

- **[ML模型使用指南](docs/ML_MODELS_GUIDE.md)**: 详细的机器学习模型说明
- **[因子合并指南](docs/MERGE_FACTOR_DATA_GUIDE.md)**: 因子数据合并说明
- **[开发文档](docs/development.md)**: 开发相关说明
- **[更新日志](CHANGELOG.md)**: 版本更新历史

---

## 🐛 常见问题

### Q1: 单商品数据标准化失败？
**A**: 系统自动识别单商品数据，将CSZScoreNorm切换为ZScoreNorm，避免NaN问题。

### Q2: 因子数据无法训练模型？
**A**: 使用`merge_factor_data`工具合并因子和价格数据，确保包含close列。

### Q3: 模型性能较差？
**A**: 
1. 增加训练数据量
2. 调整模型参数
3. 检查因子质量（使用`evaluate_factor_ic`）
4. 尝试不同的模型类型

### Q4: 数据列名不匹配？
**A**: 系统支持中英文智能列名识别，自动转换各种格式的列名。

---

## 🔄 版本更新

### v1.0.34 最新特性
- ✨ 扩展机器学习算法支持：从6种扩展到15种
- 🐛 修复因子IC评估结果为NaN问题
- 🔧 移除深度学习工具，专注于传统机器学习
- 📊 优化因子计算准确性和数据对齐

### 主要版本历史
- **v1.0.28**: 中英文智能列名识别
- **v1.0.27**: 新增因子数据合并工具
- **v1.0.24**: 新增机器学习模型支持
- **v1.0.14**: 智能标准化功能

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 开启Pull Request

---

## 📄 许可证

MIT License - 查看 [LICENSE](LICENSE) 了解详情

---

## 📞 支持

- 💬 提交 [Issues](https://github.com/jackdark425/aigroup-quant-mcp/issues)
- 📧 邮件：jackdark425@gmail.com
- 📚 文档：查看项目文档和示例

---

**立即开始**: `uvx aigroup-quant-mcp` 🚀
