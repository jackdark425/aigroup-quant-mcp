# 数据合并功能使用指南

## 📋 问题背景

在之前的版本中，存在一个设计限制：

- `train_ml_model` 工具需要数据包含 `close` 列来生成预测标签
- `generate_alpha158` 生成的因子数据**不包含** `close` 列
- 导致无法直接使用Alpha158因子训练机器学习模型

## ✨ 解决方案：merge_factor_data

新增的 `merge_factor_data` 工具可以将因子数据和价格数据合并，生成包含所有因子列和close列的完整数据集。

## 🎯 核心功能

### 功能特点

✅ **智能合并**：自动对齐因子数据和价格数据的索引  
✅ **数据验证**：确保两个数据集有时间重叠  
✅ **灵活导出**：支持CSV和JSON格式导出  
✅ **完整兼容**：合并后的数据可直接用于 `train_ml_model`

### 工作原理

```
因子数据 (159个因子列)  +  价格数据 (close列)
         ↓                          ↓
    [索引对齐 + 合并]
         ↓
合并数据 (159个因子列 + 1个close列)
```

## 📖 使用流程

### 完整工作流程

```
1️⃣ preprocess_data
   └─ 加载并清洗原始价格数据
       ↓
2️⃣ generate_alpha158
   └─ 生成Alpha158因子（159个因子）
       ↓
3️⃣ apply_processor_chain (可选)
   └─ 标准化因子数据
       ↓
4️⃣ merge_factor_data 👈 新增步骤
   └─ 合并因子数据和价格数据
       ↓
5️⃣ train_ml_model
   └─ 训练机器学习模型
       ↓
6️⃣ predict_ml_model
   └─ 使用模型进行预测
```

## 💡 使用示例

### 示例1：基础使用（最简单）

```json
{
  "factor_data_id": "alpha158_normalized",
  "price_data_id": "stock_data_2023",
  "result_id": "merged_alpha158"
}
```

**说明**：
- `factor_data_id`: 已生成的因子数据（通常是generate_alpha158或apply_processor_chain的输出）
- `price_data_id`: 原始价格数据（必须包含close列）
- `result_id`: 合并后数据的唯一标识

### 示例2：合并并导出CSV

```json
{
  "factor_data_id": "alpha158_stock",
  "price_data_id": "raw_data",
  "result_id": "alpha158_with_price",
  "export_path": "./exports/merged_data.csv",
  "export_format": "csv"
}
```

**优势**：
- 导出后可在Excel中查看验证
- 便于数据质量检查
- 支持外部工具分析

### 示例3：完整流程示例

```python
# 步骤1: 加载原始数据
{
  "file_path": "D:/data/stock_2023.csv",
  "data_id": "stock_2023",
  "auto_clean": true
}

# 步骤2: 生成Alpha158因子
{
  "data_id": "stock_2023",
  "result_id": "alpha158_raw",
  "kbar": true,
  "price": true,
  "volume": true,
  "rolling": true
}

# 步骤3: 标准化因子（可选但推荐）
{
  "data_id": "alpha158_raw",
  "result_id": "alpha158_normalized",
  "processors": [
    {"name": "CSZScoreNorm"}
  ]
}

# 步骤4: 合并因子和价格 ⭐ 关键步骤
{
  "factor_data_id": "alpha158_normalized",
  "price_data_id": "stock_2023",
  "result_id": "merged_for_training"
}

# 步骤5: 训练模型
{
  "data_id": "merged_for_training",
  "model_id": "lgb_model_v1",
  "model_type": "lightgbm",
  "train_start": "2023-01-01",
  "train_end": "2023-06-30",
  "test_start": "2023-07-01",
  "test_end": "2023-12-31"
}
```

## ⚠️ 常见问题

### Q1: 为什么需要合并数据？

**A**: `train_ml_model` 需要 `close` 列来生成预测标签（未来收益率），但因子数据中没有这一列。合并后的数据包含：
- 因子列（用作特征）
- close列（用于生成标签）

### Q2: 合并时提示"索引没有重叠"怎么办？

**A**: 确保因子数据和价格数据来源于同一个原始数据：
```json
// ✅ 正确做法
preprocess_data → data_id: "stock_2023"
generate_alpha158 → data_id: "stock_2023", result_id: "factors"
merge_factor_data → factor_data_id: "factors", price_data_id: "stock_2023"

// ❌ 错误做法
// 使用了两个不同的原始数据源
```

### Q3: 必须先标准化因子吗？

**A**: 不是必须的，但**强烈推荐**：
- 标准化后的因子训练效果更好
- 避免不同量纲的因子权重失衡
- 推荐使用 `apply_processor_chain` + `CSZScoreNorm`

### Q4: 可以合并自定义因子吗？

**A**: 可以！只要因子数据是DataFrame格式，都可以使用merge_factor_data：
```json
{
  "factor_data_id": "my_custom_factors",  // 自定义因子
  "price_data_id": "stock_2023",
  "result_id": "merged_custom_factors"
}
```

## 📊 数据结构说明

### 合并前

**因子数据** (alpha158_normalized):
```
Index: MultiIndex(datetime, symbol)
Columns: [KMID, KLEN, KMID2, ..., VSTD60]  # 159个因子列
```

**价格数据** (stock_2023):
```
Index: MultiIndex(datetime, symbol)
Columns: [open, high, low, close, volume, ...]
```

### 合并后

**合并数据** (merged_for_training):
```
Index: MultiIndex(datetime, symbol)
Columns: [KMID, KLEN, ..., VSTD60, close]  # 159因子 + 1价格
```

## 🔧 高级用法

### 导出不同格式

```json
// CSV格式 - 便于Excel查看
{
  "export_path": "./exports/merged.csv",
  "export_format": "csv"
}

// JSON格式 - 便于程序处理
{
  "export_path": "./exports/merged.json",
  "export_format": "json"
}
```

### 查看可用数据

使用 `list_factors` 工具查看当前内存中所有可用的数据和因子：

```json
{}  // 无需参数
```

返回示例：
```json
{
  "data_count": 1,
  "data_ids": ["stock_2023"],
  "factor_count": 3,
  "factors": {
    "alpha158_raw": {"type": "DataFrame", "shape": [10000, 159]},
    "alpha158_normalized": {"type": "DataFrame", "shape": [10000, 159]},
    "merged_for_training": {"type": "DataFrame", "shape": [10000, 160]}
  }
}
```

## 📈 性能说明

| 数据量 | 预计耗时 |
|--------|----------|
| < 10万行 | < 1秒 |
| 10-100万行 | 1-5秒 |
| > 100万行 | 5-15秒 |

## 🎓 最佳实践

1. **数据来源一致**：确保因子和价格来自同一原始数据
2. **先标准化再合并**：获得更好的模型效果
3. **导出验证**：合并后导出CSV检查数据质量
4. **命名规范**：使用描述性的result_id便于管理
5. **工作流顺序**：严格按照推荐工作流执行

## 🔄 版本更新

### v1.0.24 新增功能
- ✨ 新增 `merge_factor_data` 工具
- 🔧 解决因子数据无法直接训练模型的问题
- 📝 完整的文档和示例

## 📚 相关文档

- [ML_MODELS_GUIDE.md](./ML_MODELS_GUIDE.md) - 机器学习模型使用指南
- [README.md](../README.md) - 项目主文档
- [CHANGELOG.md](../CHANGELOG.md) - 版本更新记录

## 🆘 技术支持

如遇问题，请检查：
1. 数据ID是否正确（使用list_factors查看）
2. 两个数据集的时间范围是否有重叠
3. 因子数据是否已正确生成
4. 价格数据是否包含close列

---

📧 如有其他问题，欢迎提Issue或联系开发团队。