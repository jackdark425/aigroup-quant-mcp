"""
QuantAnalyzer MCP Service - 向后兼容入口
此文件保留用于向后兼容，实际实现已移至 quantanalyzer/mcp/ 模块

新的组件化结构:
- quantanalyzer/mcp/errors.py: 错误处理系统
- quantanalyzer/mcp/schemas.py: 工具Schema定义
- quantanalyzer/mcp/handlers.py: 工具处理函数
- quantanalyzer/mcp/utils.py: 工具函数
- quantanalyzer/mcp/server.py: MCP服务器主程序
"""

# 导入所有公开接口
from quantanalyzer.mcp import (
    app,
    main,
    MCPError,
    data_store,
    factor_store,
    model_store,
    processor_store
)

# 向后兼容
__all__ = [
    'app',
    'main',
    'MCPError',
    'data_store',
    'factor_store',
    'model_store',
    'processor_store'
]

if __name__ == "__main__":
    main()