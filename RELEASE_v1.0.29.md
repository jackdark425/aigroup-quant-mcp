# Release v1.0.29 - Critical Bug Fixes & Tool Consistency

## 🚀 发布概述

**版本**: v1.0.29  
**发布日期**: 2025-11-02  
**发布状态**: ✅ 已成功发布到PyPI  
**PyPI链接**: https://pypi.org/project/aigroup-quant-mcp/1.0.29/

## 🐛 修复的关键问题

### 1. 预测模块数据类型错误
- **问题**: `predict_ml_model` 出现 `keys must be str, int, float, bool or None, not tuple` 错误
- **原因**: MultiIndex DataFrame 的元组键序列化问题
- **修复**: 完善了 `_convert_index_to_string` 函数，正确处理元组键
- **影响**: 现在可以正常使用机器学习模型进行预测

### 2. 工具名称不一致问题
- **问题**: 文档中提到的深度学习工具不存在
- **移除的工具**:
  - `train_lstm_model`
  - `train_gru_model` 
  - `train_transformer_model`
  - `predict_with_model`
- **修复**: 从所有文档和代码中移除这些不存在的工具引用
- **影响**: 避免用户调用不存在工具导致的错误

### 3. MCP工具一致性验证
- **验证结果**: 9个工具在schema定义、服务器注册和实际处理中完全一致
- **可用工具列表**:
  1. `preprocess_data` - 数据预处理和清洗
  2. `calculate_factor` - 单因子计算
  3. `generate_alpha158` - Alpha158因子生成
  4. `merge_factor_data` - 因子数据合并
  5. `apply_processor_chain` - 数据处理器链（智能标准化）
  6. `evaluate_factor_ic` - 因子IC评估
  7. `list_factors` - 查看已加载数据和因子
  8. `train_ml_model` - 训练机器学习模型
  9. `predict_ml_model` - 使用模型进行预测

## 📊 数据兼容性优化

### 数据量要求
- **快速测试模式**: 仅需10条记录
- **标准模式**: 建议100条以上记录
- **生产模式**: 支持百万级数据

### 列名兼容性
- **英文格式**: datetime, symbol, open, high, low, close, volume
- **中文格式**: 时间, 股票代码, 开盘价, 最高价, 最低价, 收盘价, 成交量
- **混合格式**: Date, 股票名称, Open, High, Low, Close, Volume
- **智能识别**: 自动检测和转换不同格式的列名

## 🔧 技术细节

### 修改的文件
- `pyproject.toml` - 版本号更新到1.0.29
- `CHANGELOG.md` - 添加v1.0.29发布说明
- `quantanalyzer/mcp/handlers.py` - 完善预测数据序列化
- `quantanalyzer/mcp/schemas.py` - 移除不存在的工具引用
- `quantanalyzer/mcp/server.py` - 确保工具路由一致性
- `quantanalyzer/mcp/resources.py` - 移除不存在的工具引用
- `docs/ML_MODELS_GUIDE.md` - 更新文档与实际工具一致

### 构建和发布
- **构建状态**: ✅ 成功
- **包检查**: ✅ 通过
- **上传状态**: ✅ 成功
- **安装验证**: ✅ 成功

## 🧪 测试验证

### 验证项目
- ✅ 预测模块数据类型错误修复验证
- ✅ MCP工具名称一致性验证
- ✅ 数据量要求灵活性测试
- ✅ 列名兼容性测试
- ✅ 完整工作流程测试
- ✅ 包构建和发布测试
- ✅ 新版本安装验证

### 工作流程测试
1. `preprocess_data` - 数据预处理
2. `generate_alpha158` - 因子生成
3. `apply_processor_chain` - 智能标准化
4. `merge_factor_data` - 因子数据合并
5. `train_ml_model` - 模型训练
6. `predict_ml_model` - 模型预测
7. `evaluate_factor_ic` - 因子评估

## 🎯 影响评估

### 用户体验提升
- **修复严重bug**: 预测功能现在可以正常工作
- **提升稳定性**: 工具列表完全一致，避免调用错误
- **增强兼容性**: 支持更广泛的数据格式和规模
- **改善用户体验**: 更清晰的错误提示和文档

### 向后兼容性
- ✅ 完全向后兼容
- ✅ 现有代码无需修改
- ✅ 所有现有功能保持稳定

## 📦 安装方式

### 基础安装
```bash
pip install aigroup-quant-mcp
```

### 机器学习支持
```bash
pip install aigroup-quant-mcp[ml]
```

### 完整安装
```bash
pip install aigroup-quant-mcp[full]
```

### 使用uvx
```bash
uvx aigroup-quant-mcp
```

## 🔄 升级建议

**强烈建议所有用户升级到此版本**，特别是：
- 使用 `predict_ml_model` 的用户
- 遇到工具调用错误的用户
- 需要稳定性和一致性的用户

## 📝 后续计划

- 持续监控用户反馈
- 优化性能和用户体验
- 添加更多量化分析功能
- 完善文档和示例

---

**发布完成时间**: 2025-11-02 15:10 (UTC+8)  
**发布状态**: ✅ 成功  
**质量评级**: ⭐⭐⭐⭐⭐