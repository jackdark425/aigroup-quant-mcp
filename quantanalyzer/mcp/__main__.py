"""
MCP服务入口点
当使用 python -m quantanalyzer.mcp 时执行
"""

from .server import main

if __name__ == "__main__":
    main()