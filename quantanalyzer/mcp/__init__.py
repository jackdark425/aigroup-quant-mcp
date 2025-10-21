"""
MCP Service for QuantAnalyzer
组件化的MCP服务

目录结构:
- errors.py: 错误处理系统（MCPError类 + 验证函数）
- schemas.py: 工具Schema定义
- handlers.py: 工具处理函数（业务逻辑）
- utils.py: 工具函数（序列化等）
- server.py: MCP服务器主程序
"""

from .server import app, main
from .errors import MCPError
from .handlers import data_store, factor_store, model_store, processor_store

__all__ = [
    'app',
    'main',
    'MCPError',
    'data_store',
    'factor_store',
    'model_store',
    'processor_store'
]