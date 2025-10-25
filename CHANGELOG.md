## [1.0.27] - 2025-01-25 🚀 Data Merge & ML Training Enhancement

### ✨ New Feature
- **新增 `merge_factor_data` 工具**: 解决因子数据无法直接用于机器学习模型训练的问题
  - 🔗 **智能合并**: 自动对齐Alpha158因子数据和原始价格数据的索引
  - 📊 **数据验证**: 确保两个数据集有时间重叠，防止数据不匹配错误
  - 💾 **灵活导出**: 支持CSV/JSON格式导出合并后的完整数据集
  - 🤖 **完整兼容**: 合并后的数据可直接用于 `train_ml_model` 训练
  - 📈 **工作流程**: preprocess_data → generate_alpha158 → merge_factor_data → train_ml_model

### 🔧 Core Implementation
- **智能索引对齐**: 自动处理因子数据和价格数据的时间索引匹配
- **数据完整性验证**: 检查数据重叠范围，提供详细的错误提示和建议
- **内存优化**: 高效的DataFrame合并，避免不必要的内存消耗
- **错误处理**: 完善的异常处理和用户友好的错误信息

### 📚 Documentation
- **完整使用指南**: 新增 `docs/MERGE_FACTOR_DATA_GUIDE.md`
  - 详细的问题背景和解决方案说明
  - 完整的工作流程和使用示例
  - 常见问题解答和最佳实践
  - 数据结构说明和技术细节
- **主文档更新**: 更新README.md包含新工具说明和示例
- **Schema文档**: 完整的参数说明、类型定义和使用示例

### 🎯 Impact
- **解决关键限制**: 现在可以直接使用Alpha158因子训练机器学习模型
- **完整工作流**: 从数据加载到模型训练的完整量化分析流程
- **用户体验**: 消除数据格式不匹配的困惑，提供清晰的指导
- **向后兼容**: 不影响现有功能，新增工具可选使用

### 🧪 Testing & Validation
- ✅ 因子数据和价格数据合并测试
- ✅ 索引对齐和数据验证测试
- ✅ CSV/JSON导出功能测试
- ✅ 与train_ml_model的集成测试
- ✅ 错误处理和边界情况测试

### 📝 Technical Details
- 新增文件: `docs/MERGE_FACTOR_DATA_GUIDE.md`
- 修改文件:
  - `quantanalyzer/mcp/handlers.py` - 添加merge_factor_data处理函数
  - `quantanalyzer/mcp/schemas.py` - 添加工具schema定义
  - `quantanalyzer/mcp/server.py` - 注册新工具
  - `README.md` - 更新文档说明
- 代码质量: 完整的类型提示、错误处理、文档字符串

---

## [1.0.26] - 2025-01-25 📦 Package Enhancement

### 📦 Package Updates
- **依赖优化**: 进一步优化依赖配置，提升安装和启动速度
- **文档完善**: 更新README和使用指南
- **代码质量**: 改进错误处理和用户体验

---

## [1.0.23] - 2024-10-24 🐛 IC Evaluation Fix

### 修复
- **修复evaluate_factor_ic的"Must specify axis=0 or 1"错误**
  - 问题：处理DataFrame类型的价格或因子数据时未正确转换为Series
  - 症状：IC评估时报错"Must specify axis=0 or 1"
  - 解决：添加DataFrame到Series的自动转换逻辑
  - 影响：Alpha158等多列因子现在可以正常评估IC

### 改进
- 更新启动日志版本号到v1.0.23
- 优化因子数据类型处理逻辑

---

## [1.0.19] - 2024-10-24 🚀 Lightweight Version

### 移除
- 移除所有深度学习相关功能（LSTM/GRU/Transformer模型训练工具）
- 移除quick_start_lstm快捷工具
- 完全移除torch依赖引用，解决uvx启动卡住问题

### 改进  
- MCP服务专注于核心数据处理和因子分析功能
- 启动速度进一步优化，从"安装36包后卡住"到"秒级启动"
- 服务更轻量、更稳定、更专注

### ✅ 可用功能
- preprocess_data - 数据预处理和清洗
- calculate_factor - 单因子计算
- generate_alpha158 - Alpha158因子生成  
- apply_processor_chain - 数据处理器链（智能标准化）
- evaluate_factor_ic - 因子IC评估（含报告生成）✨ v1.0.16新增
- list_factors - 查看已加载数据和因子

### 📝 说明
深度学习功能已移除，专注于核心的数据处理和因子分析。如需深度学习功能，建议使用其他专业工具。

---

# Changelog

All notable changes to this project will be documented in this file.

## [1.0.18] - 2024-10-24 🐛 Critical Import Fix

### 🐛 Critical Bug Fix
- **修复torch导入错误导致uvx启动失败**
  - 问题：虽然torch已移到可选依赖，但代码仍强制导入
  - 症状：`uvx aigroup-quant-mcp` 报错 `ModuleNotFoundError: No module named 'torch'`
  - 修复：深度学习模型导入改为可选，无torch时提供友好错误提示
  - 影响文件：quantanalyzer/model/__init__.py

### 💡 用户体验改进
- 无torch时给出清晰的安装提示
- 基础MCP功能无需torch即可正常使用
- 需要深度学习功能时按需安装

### 🎯 Impact
- ✅ uvx aigroup-quant-mcp 现在可以正常启动
- ✅ 基础功能（数据处理、因子计算、因子评估）完全可用
- ✅ 深度学习功能按需安装

---

## [1.0.17] - 2024-10-24 ⚡ Performance Optimization

### ⚡ Performance Improvement
- **优化依赖配置，解决uvx安装卡住问题**
  - 问题：torch>=2.0.0 等重量级依赖导致uvx安装极慢或卡死
  - 解决方案：将重量级依赖移到可选依赖
  - 核心依赖（必需）：
    - pandas>=2.0.0
    - numpy>=1.24.0
    - scipy>=1.10.0
    - mcp>=1.0.0
  - 可选依赖：
    - `[ml]`: lightgbm, xgboost, scikit-learn（机器学习模型）
    - `[dl]`: torch（深度学习模型）
    - `[full]`: 所有机器学习和深度学习依赖
    - `[viz]`: matplotlib（可视化）
    - `[dev]`: pytest（开发测试）

### 📦 安装方式

```bash
# 基础安装（推荐，快速）
uvx aigroup-quant-mcp

# 或使用pip基础安装
pip install aigroup-quant-mcp

# 安装机器学习支持
pip install aigroup-quant-mcp[ml]

# 安装深度学习支持
pip install aigroup-quant-mcp[dl]

# 完整安装（包含所有功能）
pip install aigroup-quant-mcp[full]
```

### 🎯 Impact
- **显著提升安装速度**：基础安装从几分钟降至几秒
- **按需安装**：用户可根据需求选择安装哪些依赖
- **向后兼容**：不影响现有功能使用

---

## [1.0.16] - 2024-10-24 🐛 Critical Bug Fix & Feature Enhancement

### 🐛 Critical Bug Fix
- **修复evaluate_factor_ic工具返回NoneType错误**
  - 问题根源：函数定义与实现被错误分离
    - 第506-508行：只有空的函数定义
    - 第609-825行：实际实现被错误嵌套在handle_quick_start_lstm函数内部
  - 修复方案：
    - ✅ 将函数实现合并到正确位置（506-724行）
    - ✅ 删除错误嵌套的重复代码（826-942行）
    - ✅ 函数现在正常返回评估结果
  - 影响范围：所有使用evaluate_factor_ic工具的用户
  - 修复文件：quantanalyzer/mcp/handlers.py

### ✨ New Feature
- **因子评估报告生成功能**
  - 新增 `report_path` 可选参数
  - 自动生成Markdown格式的详细评估报告
  - 报告内容包括：
    - 📋 基本信息（因子名称、数据源、评估方法、时间）
    - 📊 IC指标（IC均值、IC标准差、ICIR、IC正值占比）
    - ⭐ 因子质量评级和推荐建议
    - 📖 详细指标解读（IC均值、ICIR、IC正值占比）
    - 🎯 预测方向和预测能力分析
    - 💡 使用建议和后续步骤指引
  - 报告特点：
    - Markdown格式，易读易分享
    - UTF-8编码，兼容性好
    - 自动创建目录
    - 可作为策略文档

### 📝 Schema Updates
- 更新 `evaluate_factor_ic` 工具Schema
- 添加 `report_path` 参数说明和使用示例
- 完善参数文档和最佳实践指引

### 🎯 Impact
- **强烈建议立即升级**：修复了因子评估功能完全无法使用的严重bug
- 向后兼容：report_path为可选参数，不影响现有调用
- 提升用户体验：支持生成专业的评估报告

### 📚 Technical Details
- 修复文件：
  - quantanalyzer/mcp/handlers.py (506-724行)
  - quantanalyzer/mcp/schemas.py (1018-1050行)
- 代码质量：函数结构清晰，无重复代码
- 测试状态：核心功能修复完成，可正常使用

---

## [1.0.15] - 2024-10-24 📚 Documentation Update

### 📝 Documentation
- **更新apply_processor_chain的MCP Schema文档**
  - 明确说明智能标准化功能和使用方法
  - 添加详细的处理器说明和适用场景
  - 更新示例展示单商品/多商品的智能处理
  - 强调推荐直接使用CSZScoreNorm，系统自动优化

### 💡 User Guidance
- 用户现在可以通过MCP工具描述了解智能标准化功能
- 明确推荐使用CSZScoreNorm，无需手动判断数据类型
- 提供更清晰的使用示例和预期结果

### 🎯 Impact
- 提升用户体验，避免传入错误参数
- 文档与代码功能保持同步
- 帮助用户正确使用智能标准化功能

---

## [1.0.14] - 2024-10-24 🎯 Smart Normalization

### 🚀 Major Feature
- **智能标准化功能**: 自动识别单商品/多商品数据并选择最佳标准化方法
  - 自动检测数据中的商品数量（symbol_count）
  - 单商品数据：CSZScoreNorm **自动切换** 为 ZScoreNorm
  - 多商品数据：CSZScoreNorm **正常使用**
  - 返回详细的智能调整信息
  
### 🐛 Critical Bug Fix
- **修复CSZScoreNorm在单商品数据上导致100% NaN的问题**
  - 问题原因：CSZScoreNorm是截面标准化，需要多个商品才能计算标准差
  - 单商品时每个时间点只有1个样本，标准差为NaN
  - 现在自动切换为ZScoreNorm（时间序列标准化）避免此问题

### 💡 User Experience
- **用户无需关心数据类型**：直接使用CSZScoreNorm，系统自动优化
- **透明性**：明确告知用户发生了什么调整及原因
- **向后兼容**：API保持不变，现有代码无需修改

### 📊 Smart Decision Logic
| 数据类型 | 请求Processor | 实际应用 | 原因 |
|---------|--------------|---------|------|
| 单商品 | CSZScoreNorm | ZScoreNorm | 避免NaN |
| 多商品 | CSZScoreNorm | CSZScoreNorm | 正常工作 |

### 🧪 Testing
- ✅ 单商品数据测试：自动切换 + NaN比例0%
- ✅ 多商品数据测试：保持不变 + NaN比例0%
- ✅ 完整自动化测试通过

### 📚 Documentation
- 新增 `BUG_REPORT_CSZScoreNorm.md` - 详细的Bug诊断报告
- 新增 `SMART_NORMALIZATION_FEATURE.md` - 智能功能实现总结
- 新增 `diagnose_normalization.py` - 问题诊断脚本
- 新增 `test_auto_normalization.py` - 自动化测试脚本

### 🎯 Impact
- 解决了单商品/单股票数据标准化失败的问题
- 提升用户体验，无需手动选择标准化方法
- 强烈建议所有用户升级到此版本

---

## [1.0.13] - 2024-10-24

### ✨ Feature Enhancement
- **generate_alpha158导出功能**: 添加因子数据导出支持
  - 新增 `export_path` 参数：指定导出文件路径
  - 新增 `export_format` 参数：支持CSV和JSON两种格式
  - CSV格式便于Excel查看所有158个因子的数值
  - JSON格式便于程序处理和集成
  - 返回导出文件信息（路径、大小、格式、因子数量、行数）

### 🎯 使用场景
- 直接查看所有158个因子的具体数值
- 数据质量检查和验证
- 外部工具进一步分析
- 因子数据持久化保存
- 建立因子数据库

### 📝 Documentation
- 更新 schema 添加导出参数说明
- 新增 RELEASE_v1.0.13.md 发布说明
- 完善使用示例和最佳实践

---

## [1.0.12] - 2024-10-24

### 🐛 Critical Bug Fix
- **修复generate_alpha158的NameError**: 修正了未定义的export_info变量引用
  - 移除了handlers.py中对不存在变量的错误引用 (第438-439行)
  - 修复了调用generate_alpha158时的NameError错误
  - 现在所有7个工具都能正常工作

### 技术细节
- 问题根源: 复制粘贴错误，从preprocess_data复制代码时保留了不必要的export_info引用
- 影响版本: v1.0.11
- 修复文件: quantanalyzer/mcp/handlers.py

### ✅ Verification
- ✅ generate_alpha158工具现在正常工作
- ✅ 所有7个工具验证通过

---

## [1.0.11] - 2024-10-24

### 🐛 Bug Fixes
- **修复Schema语法错误**: 修正了v1.0.10中导致MCP工具无法加载的语法错误
  - get_apply_processor_chain_schema函数位置错误
  - 现在所有7个工具正常显示

### ✅ Verification
- 验证7个工具全部可用
- 测试导出功能正常工作

---

## [1.0.10] - 2024-10-24

### ✨ Enhancement
- **apply_processor_chain导出功能**: 添加数据导出支持
  - 支持CSV格式导出（便于Excel查看）
  - 支持JSON格式导出（便于程序处理）
  - 可选参数：export_path、export_format
  - 返回导出文件信息（路径、大小、格式）

### 🎯 使用场景
- 标准化后的因子数据质量检查
- 导出供外部工具分析
- 数据持久化保存
- 调试和验证

### 🧪 Testing
- ✅ CSV导出测试
- ✅ JSON导出测试
- ✅ 可选导出测试

---

## [1.0.9] - 2024-10-24

### ✨ New Features
- **apply_processor_chain工具**: 对数据（因子）应用处理器链
  - 支持7种处理器：CSZScoreNorm、CSZFillna、ProcessInf、ZScoreNorm、RobustZScoreNorm、CSRankNorm、MinMaxNorm
  - 支持链式处理：可组合多个处理器
  - 完整的参数配置和错误处理

### 📊 Workflow
现在可以完成完整的数据→因子→标准化流程：
1. preprocess_data - 数据预处理
2. generate_alpha158 - 因子生成
3. apply_processor_chain - 因子标准化 ⭐ 新增
4-6. 模型训练、预测、评估（待实现）

### 📚 Documentation
- 新增 `IMPLEMENTATION_ROADMAP.md` - 完整开发路线图
- 更新 `QLIB_WORKFLOW_GUIDE.md` - 工作流程指南
- 新增 `test_processor_chain.py` - 完整测试脚本

### 🧪 Testing
- ✅ 单处理器测试
- ✅ 多处理器链测试
- ✅ 完整工作流程测试

---

## [1.0.8] - 2024-10-24 🔥 Critical Fix

### 🐛 Critical Bug Fix
- **修复预处理流程错误**: 移除了对原始OHLCV数据的错误标准化
  - 旧版本错误地在`preprocess_data`中对原始数据进行`CSZScoreNorm`标准化
  - 现在只进行异常值和缺失值清洗（ProcessInf + CSZFillna）
  - 标准化应该在因子生成后通过`apply_processor_chain`进行
  
### 📚 Documentation
- 新增 `QLIB_WORKFLOW_GUIDE.md` - 详细的Qlib工作流程指南
  - 解释正确的三步流程：数据清洗 → 因子生成 → 因子标准化
  - 说明为什么不能提前标准化原始数据
  - 提供错误vs正确流程对比
  
### 📝 Schema Updates
- 更新 `preprocess_data` 工具文档
- 明确标准化应该用于因子而非原始数据

### ⚠️ Impact
此版本修复了一个严重的数据处理错误，强烈建议所有v1.0.7用户立即升级。

---

## [1.0.7] - 2024-10-24

### 🔄 Breaking Changes
- **重命名工具**: `load_csv_data` → `preprocess_data`
  - 更准确地反映数据清洗功能
  - 所有引用需要更新为新名称

### ✨ New Features
- **数据导出功能**: 清洗后的数据自动保存到本地
  - 默认导出路径: `./exports/{data_id}_cleaned_{timestamp}.csv`
  - 支持自定义导出路径通过 `export_path` 参数
  - 返回导出文件信息（路径、大小）
  
### 📝 Parameters
- 新增 `export_path` 参数用于自定义导出路径
- 保留 `auto_clean` 参数控制是否清洗数据

### 🐛 Bug Fixes
- 优化数据清洗流程
- 改进错误处理和反馈

### 📚 Documentation
- 更新工具描述和使用示例
- 添加 RELEASE_v1.0.7.md 发布说明
- 添加 REFACTORING_SUMMARY.md 重构总结
- 创建测试脚本 test_preprocess_data.py

### 🧪 Testing
- 所有功能测试通过
- 验证自动导出功能
- 验证自定义路径导出
- 验证不清洗模式

---

## [1.0.6] - 2024-10-21

### Features
- 完善MCP服务器功能
- 优化数据清洗流程

---

## [1.0.5] - 2024-10-20

### Features
- 添加自动数据清洗功能
- 优化因子计算性能

---

## [1.0.4] - 2024-10-19

### Features
- 初始MCP服务器实现
- 基础量化分析功能