# RooCode MCP配置指南

## 📋 概述

本文档说明如何在RooCode中配置aigroup-quant-mcp服务，以便在RooCode中使用量化分析功能。

## 🚀 快速配置

### 方式1：使用uvx（推荐）

最简单的方式是使用`uvx`命令，无需本地安装：

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

### 方式2：本地安装

如果您已经通过pip安装了aigroup-quant-mcp：

```json
{
  "mcpServers": {
    "aigroup-quant-mcp": {
      "command": "aigroup-quant-mcp",
      "env": {},
      "alwaysAllow": [
        "preprocess_data",
        "calculate_factor",
        "generate_alpha158",
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

## ⚙️ 配置参数说明

### command
- **uvx**: 使用uvx运行，无需本地安装（推荐）
- **aigroup-quant-mcp**: 如果已通过pip安装

### args
- 启动参数，通常为空数组或包含服务名称

### env
- 环境变量，可留空或添加特定环境变量

### alwaysAllow
允许RooCode访问的工具列表。当前支持的工具：

| 工具 | 功能 | 权限说明 |
|-----|------|----------|
| `preprocess_data` | 数据预处理 | ✅ 安全 |
| `calculate_factor` | 单因子计算 | ✅ 安全 |
| `generate_alpha158` | Alpha158因子生成 | ✅ 安全 |
| `evaluate_factor_ic` | 因子评估 | ✅ 安全 |
| `apply_processor_chain` | 数据标准化 | ✅ 安全 |
| `train_ml_model` | 机器学习训练 | ✅ 安全 |
| `predict_ml_model` | 模型预测 | ✅ 安全 |
| `list_factors` | 查看因子 | ✅ 安全 |

## 🔧 在RooCode中配置

### 步骤1：打开RooCode设置

1. 在RooCode中按 `Ctrl+,` (Windows/Linux) 或 `Cmd+,` (Mac)
2. 搜索 "MCP" 或 "Model Context Protocol"
3. 找到MCP服务器配置部分

### 步骤2：添加配置

1. 点击 "添加MCP服务器" 或类似的按钮
2. 选择 "JSON配置" 方式
3. 粘贴上述JSON配置
4. 保存配置

### 步骤3：重启RooCode

配置保存后，需要重启RooCode才能生效。

## ✅ 验证配置

配置完成后，您可以通过以下方式验证：

1. **检查服务状态**: RooCode应该显示aigroup-quant-mcp服务正在运行
2. **测试工具**: 在对话中尝试使用 `list_factors` 工具
3. **查看工具列表**: 应该能看到所有8个可用工具

## 🔍 故障排除

### 服务无法启动

**症状**: MCP服务显示"未连接"或"错误"

**解决**:
1. 检查uvx是否可用: `uvx --version`
2. 尝试清除缓存: `uvx --no-cache aigroup-quant-mcp`
3. 检查网络连接
4. 查看RooCode的详细错误日志

### 工具不可用

**症状**: 某些工具在alwaysAllow列表中但无法使用

**解决**:
1. 检查alwaysAllow列表是否包含该工具
2. 重启RooCode
3. 确认工具名称拼写正确

### 权限问题

**症状**: 工具执行被拒绝

**解决**:
1. 检查alwaysAllow列表
2. 添加缺失的工具到alwaysAllow
3. 重启RooCode

## 📝 配置示例

### 完整配置示例

```json
{
  "mcpServers": {
    "aigroup-quant-mcp": {
      "command": "uvx",
      "args": ["aigroup-quant-mcp"],
      "env": {
        "PYTHONPATH": "/path/to/your/python"
      },
      "alwaysAllow": [
        "preprocess_data",
        "calculate_factor",
        "generate_alpha158",
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

### 最小配置示例

```json
{
  "mcpServers": {
    "aigroup-quant-mcp": {
      "command": "uvx",
      "args": ["aigroup-quant-mcp"],
      "alwaysAllow": ["list_factors"]
    }
  }
}
```

## 🚀 最佳实践

1. **使用uvx**: 推荐使用uvx，无需本地安装，自动更新
2. **包含所有工具**: 在alwaysAllow中包含所有需要使用的工具
3. **定期更新**: 通过uvx自动获取最新版本
4. **备份配置**: 保存您的MCP配置到安全位置

## 📚 相关文档

- [README.md](../README.md) - 主要文档
- [docs/ML_MODELS_GUIDE.md](ML_MODELS_GUIDE.md) - 机器学习使用指南
- [examples/](../examples/) - 示例代码

---

**配置完成后，您就可以在RooCode中享受完整的量化分析功能了！** 🎉