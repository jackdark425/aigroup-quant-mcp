# Release v1.0.31 - Enhanced ML Support & Factor Accuracy Fix

## 🚀 版本亮点

### ✨ 主要特性
- **扩展机器学习算法支持**: 从6种扩展到15种传统机器学习算法
- **修复因子IC评估结果异常**: 解决动量因子计算中的数据对齐问题
- **优化工具链稳定性**: 移除深度学习工具，专注于核心量化分析功能

### 📊 核心改进

#### 1. 机器学习算法扩展
现在支持15种传统机器学习算法：

**线性模型**
- `linear` - 线性回归（基线模型）
- `ridge` - 岭回归（L2正则化）
- `lasso` - Lasso回归（L1正则化，特征选择）
- `elasticnet` - 弹性网络回归（L1+L2正则化）
- `logistic` - 逻辑回归（分类任务）

**基于树的模型**
- `lightgbm` - LightGBM梯度提升树（推荐，速度快）
- `xgboost` - XGBoost梯度提升树（性能强）
- `random_forest` - 随机森林（集成学习）
- `gradient_boosting` - 梯度提升树（GBDT）
- `decision_tree` - 决策树（可解释性强）
- `catboost` - CatBoost梯度提升树（类别特征优化）

**其他算法**
- `svm` - 支持向量机（分类）
- `svr` - 支持向量回归（回归）
- `naive_bayes` - 朴素贝叶斯（概率模型）
- `knn` - K近邻算法（非参数模型）

#### 2. 因子计算准确性修复
- **修复动量因子IC评估NaN问题**: 通过数据对齐机制确保因子计算和价格数据索引匹配
- **优化因子库数据处理**: 改进标准差计算、RSI周期计算、成交量数据处理
- **增强异常处理**: IC评估现在返回正常数值，避免NaN结果

#### 3. 工具链优化
- **移除深度学习工具**: 专注于传统机器学习算法，提升稳定性
- **工具总数**: 9个核心工具，完全一致且稳定
- **完整工作流程**: 数据预处理 → 因子生成 → 标准化 → 模型训练 → 预测

### 🛠️ 核心工具链

1. **preprocess_data** - 数据预处理和清洗
2. **calculate_factor** - 单因子计算
3. **generate_alpha158** - Alpha158因子集生成
4. **merge_factor_data** - 因子数据合并
5. **apply_processor_chain** - 数据处理器链（智能标准化）
6. **evaluate_factor_ic** - 因子IC评估
7. **train_ml_model** - 机器学习模型训练（15种算法）
8. **predict_ml_model** - 模型预测
9. **list_factors** - 状态查询

### 🎯 使用场景

**量化策略研究**
- 基于因子的多因子模型构建
- 机器学习特征工程和模型训练
- 因子有效性验证和筛选

**技术指标分析**
- 动量、波动率、成交量等因子计算
- 因子IC评估和质量评级
- 智能数据标准化处理

**模型训练和预测**
- 15种机器学习算法选择
- 特征重要性分析
- 模型性能评估和预测

### 📦 安装方式

```bash
# 基础安装（推荐）
pip install aigroup-quant-mcp

# 或使用uvx
uvx aigroup-quant-mcp

# 安装机器学习支持
pip install aigroup-quant-mcp[ml]

# 完整安装（包含所有功能）
pip install aigroup-quant-mcp[full]
```

### 🧪 验证状态

- ✅ 15种机器学习算法可用性验证
- ✅ 因子计算准确性测试
- ✅ IC评估正常性验证
- ✅ 完整工作流程测试
- ✅ 工具一致性验证

### 📝 技术细节

**修复文件**
- `quantanalyzer/factor/library.py` - 修复因子数据对齐问题
- `quantanalyzer/factor/evaluator.py` - 优化IC评估异常处理

**更新文件**
- `quantanalyzer/mcp/schemas.py` - 扩展机器学习算法支持
- `quantanalyzer/mcp/handlers.py` - 实现15种算法训练逻辑
- `quantanalyzer/model/trainer.py` - 优化模型训练和特征重要性
- `README.md` - 更新文档包含15种算法说明

### 🚨 升级建议

**强烈建议升级到此版本**，因为：
- 修复了因子IC评估的严重bug
- 扩展了机器学习算法支持
- 提升了工具链稳定性
- 优化了用户体验

### 📚 文档资源

- [完整使用指南](README.md)
- [机器学习模型指南](docs/ML_MODELS_GUIDE.md)
- [因子数据合并指南](docs/MERGE_FACTOR_DATA_GUIDE.md)
- [变更日志](CHANGELOG.md)

---

**版本**: v1.0.31  
**发布日期**: 2025-11-02  
**兼容性**: Python >= 3.8  
**许可证**: MIT