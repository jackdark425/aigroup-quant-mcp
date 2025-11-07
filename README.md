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
pip install aigroup-quant-mcp[full]

# 完整安装（所有功能）
pip install aigroup-quant-mcp[full,viz]
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
    export_path="./exports/alpha158_factors.csv"
)
```

### 步骤3: 训练模型
```python
train_ml_model(
    data_id="stock_2023",
    model_type="lightgbm",
    train_start="2023-01-01",
    train_end="2023-10-31",
    test_start="2023-11-01",
    test_end="2023-12-31",
    export_path="./exports/trained_model.pkl"
)
```

### 步骤4: 模型预测
```python
predict_ml_model(
    data_id="stock_2023",
    model_path="./exports/trained_model.pkl",
    predict_start="2023-12-01",
    predict_end="2023-12-31",
    export_path="./exports/predictions.csv"
)
```

---

## 📚 文档

- [开发指南](DEVELOPING.md)
- [性能优化指南](docs/PERFORMANCE_OPTIMIZATION.md)
- [API参考](docs/API.md)

---

## 📃 许可证

本项目采用MIT许可证，详见[LICENSE](LICENSE)文件。