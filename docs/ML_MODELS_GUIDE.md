# 机器学习模型使用指南

## 📋 概述

aigroup-quant-mcp现在提供完整的机器学习建模功能，支持LightGBM、XGBoost和scikit-learn，**无需安装torch包**。

## ✨ 主要特性

- ✅ **LightGBM**: 快速高效的梯度提升树模型（推荐）
- ✅ **XGBoost**: 性能强大的梯度提升树模型
- ✅ **线性回归**: sklearn线性回归基线模型
- ✅ **自动特征重要性分析**
- ✅ **完整的训练和评估指标**
- ✅ **无需torch依赖**

## 🚀 快速开始

### 1. 安装依赖

```bash
# 核心依赖（必需）
pip install pandas numpy scipy lightgbm xgboost scikit-learn mcp

# 不再需要torch！
```

### 2. 完整工作流程

```python
# 通过MCP工具使用

# 步骤1: 加载数据
preprocess_data(
    file_path="./data/stock_data.csv",
    data_id="stock_data",
    auto_clean=True
)

# 步骤2: 生成Alpha158因子
generate_alpha158(
    data_id="stock_data",
    result_id="alpha158_factors"
)

# 步骤3: 数据标准化（智能标准化）
apply_processor_chain(
    data_id="alpha158_factors",
    result_id="alpha158_normalized",
    processors=[{"name": "CSZScoreNorm"}]
)

# 步骤4: 训练LightGBM模型
train_ml_model(
    data_id="stock_data",  # 必须包含close列
    model_id="lgb_model_v1",
    model_type="lightgbm",
    train_start="2023-01-01",
    train_end="2023-06-30",
    test_start="2023-07-01",
    test_end="2023-12-31"
)

# 步骤5: 使用模型预测
predict_ml_model(
    model_id="lgb_model_v1",
    data_id="alpha158_normalized",
    export_path="./exports/predictions.csv"
)
```

## 🤖 支持的模型类型

### LightGBM（推荐）

**优势**:
- 训练速度快
- 内存占用小
- 效果好
- 支持分类特征

**默认参数**:
```python
{
    "objective": "regression",
    "metric": "mse",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "verbose": -1
}
```

**自定义参数示例**:
```python
train_ml_model(
    data_id="stock_data",
    model_id="lgb_custom",
    model_type="lightgbm",
    train_start="2023-01-01",
    train_end="2023-06-30",
    test_start="2023-07-01",
    test_end="2023-12-31",
    params={
        "learning_rate": 0.1,
        "num_leaves": 63,
        "max_depth": 8
    }
)
```

### XGBoost

**优势**:
- 性能强大
- 参数丰富
- 社区活跃

**默认参数**:
```python
{
    "objective": "reg:squarederror",
    "learning_rate": 0.05,
    "max_depth": 6
}
```

### 线性回归（基线）

**优势**:
- 速度极快
- 可解释性强
- 适合快速验证

**默认参数**:
```python
{
    "alpha": 1.0  # L2正则化系数
}
```

## 📊 模型评估指标

训练完成后，系统会返回以下指标：

- **MSE** (Mean Squared Error): 均方误差，越小越好
- **MAE** (Mean Absolute Error): 平均绝对误差，越小越好
- **R²** (R-squared): 决定系数，越接近1越好
- **IC** (Information Coefficient): 信息系数，衡量预测能力

### 指标解读

| R² 值 | 模型质量 |
|-------|---------|
| > 0.7 | 优秀 |
| 0.5-0.7 | 良好 |
| 0.3-0.5 | 一般 |
| < 0.3 | 较差 |

| IC 绝对值 | 预测能力 |
|----------|---------|
| > 0.10 | 非常强 |
| 0.08-0.10 | 强 |
| 0.05-0.08 | 较强 |
| 0.03-0.05 | 有效 |
| < 0.03 | 无效 |

## 🎯 最佳实践

### 1. 数据准备

```python
# ✅ 推荐：使用标准化的因子数据
# 1. 加载数据并清洗
preprocess_data(file_path="data.csv", data_id="raw_data", auto_clean=True)

# 2. 生成因子
generate_alpha158(data_id="raw_data", result_id="factors")

# 3. 标准化（智能标准化，自动适配单/多商品）
apply_processor_chain(
    data_id="factors",
    result_id="factors_norm",
    processors=[{"name": "CSZScoreNorm"}]
)

# 4. 使用标准化后的因子训练
# 注意：需要使用原始数据（包含close列）生成标签
train_ml_model(data_id="raw_data", ...)  # 使用原始数据
```

### 2. 时间范围划分

```python
# ✅ 推荐：训练集70-80%，测试集20-30%
train_ml_model(
    data_id="stock_data",
    model_id="model_v1",
    model_type="lightgbm",
    train_start="2020-01-01",  # 3年训练集
    train_end="2022-12-31",
    test_start="2023-01-01",   # 1年测试集
    test_end="2023-12-31"
)
```

### 3. 模型选择

```python
# 首次尝试：LightGBM（速度快，效果好）
train_ml_model(model_type="lightgbm", ...)

# 追求性能：XGBoost（参数多，可调优）
train_ml_model(model_type="xgboost", ...)

# 快速验证：线性回归（基线对比）
train_ml_model(model_type="linear", ...)
```

### 4. 参数调优

```python
# LightGBM调优示例
train_ml_model(
    model_type="lightgbm",
    params={
        "learning_rate": 0.05,      # 学习率（0.01-0.3）
        "num_leaves": 31,           # 叶子数（31-127）
        "max_depth": -1,            # 最大深度（-1为不限制）
        "min_data_in_leaf": 20,     # 叶子最小样本数
        "feature_fraction": 0.8,    # 特征采样比例
        "bagging_fraction": 0.8,    # 样本采样比例
        "bagging_freq": 5           # 采样频率
    }
)
```

## ⚠️ 常见问题

### Q1: 训练数据为空怎么办？

**原因**: 单商品数据使用CSZScoreNorm导致100% NaN

**解决**: 
```python
# 单商品数据：系统会自动将CSZScoreNorm切换为ZScoreNorm
# 用户无需关心，直接使用CSZScoreNorm即可
apply_processor_chain(
    data_id="single_stock_factors",
    result_id="normalized",
    processors=[{"name": "CSZScoreNorm"}]  # 自动优化
)
```

### Q2: 如何使用因子数据训练？

**解决**: 需要原始数据（包含close列）生成标签
```python
# ❌ 错误：使用因子数据（无close列）
train_ml_model(data_id="alpha158_factors", ...)

# ✅ 正确：使用原始数据（有close列）
train_ml_model(data_id="raw_stock_data", ...)
```

### Q3: 模型性能较差怎么办？

**优化方向**:
1. 增加训练数据量
2. 调整模型参数
3. 尝试不同的模型类型
4. 检查因子质量（使用evaluate_factor_ic）
5. 使用特征选择（基于feature_importance）

## 📝 完整示例

查看 `examples/test_ml_models.py` 获取完整的代码示例。

```bash
# 运行测试
python examples/test_ml_models.py
```

## 🔄 从深度学习迁移

如果之前使用torch深度学习模型，现在可以轻松迁移到机器学习：

```python
# 之前：深度学习模型（已移除）
# train_lstm_model(...)  # 需要torch，已移除

# 现在：机器学习模型（无需torch）
train_ml_model(model_type="lightgbm", ...)  # 速度更快，效果相当
```

**优势对比**:
- ✅ 无需安装torch（节省空间）
- ✅ 训练速度更快
- ✅ 内存占用更小
- ✅ 可解释性更强（特征重要性）
- ✅ 参数调优更简单

## 📚 参考资源

- [LightGBM官方文档](https://lightgbm.readthedocs.io/)
- [XGBoost官方文档](https://xgboost.readthedocs.io/)
- [scikit-learn官方文档](https://scikit-learn.org/)

## 🆕 更新日志

**v1.0.24** (2025-01-25)
- ✨ 新增LightGBM模型支持
- ✨ 新增XGBoost模型支持
- ✨ 新增线性回归模型支持
- ✨ 新增train_ml_model工具
- ✨ 新增predict_ml_model工具
- 🔧 torch改为可选依赖
- 📝 添加完整的使用文档和测试

---

💡 **提示**: 机器学习模型通常比深度学习模型更快、更稳定，且不需要GPU！