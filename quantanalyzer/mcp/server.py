"""
MCP服务器主程序
整合所有组件
"""

import sys
from typing import Any, Dict, List
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from .handlers import (
    handle_preprocess_data,
    handle_generate_alpha158
)

# 创建MCP server实例
app = Server("aigroup-quant-mcp")


@app.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """列出所有可用工具"""
    from .schemas import (
        get_preprocess_data_schema,
        get_calculate_factor_schema,
        get_generate_alpha158_schema,
        get_merge_factor_data_schema,
        get_apply_processor_chain_schema,
        get_evaluate_factor_ic_schema,
        get_list_factors_schema,
        get_train_ml_model_schema,
        get_predict_ml_model_schema
    )
    
    # 返回所有可用工具（仅包含机器学习功能）
    tools = [
        get_preprocess_data_schema(),
        get_calculate_factor_schema(),
        get_generate_alpha158_schema(),
        get_merge_factor_data_schema(),
        get_apply_processor_chain_schema(),
        get_evaluate_factor_ic_schema(),
        get_list_factors_schema(),
        get_train_ml_model_schema(),
        get_predict_ml_model_schema(),
    ]
    
    return tools


@app.list_resources()
async def handle_list_resources() -> List[types.Resource]:
    """列出可用资源"""
    from .resources import get_faq_resources
    return get_faq_resources()


@app.read_resource()
async def handle_read_resource(uri: str) -> str:
    """读取资源内容"""
    from .resources import read_faq_resource
    return read_faq_resource(uri)


@app.call_tool()
async def handle_call_tool(
    name: str,
    arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """处理工具调用"""
    
    # 路由到对应的处理函数
    if name == "preprocess_data":
        return await handle_preprocess_data(arguments)
    elif name == "generate_alpha158":
        return await handle_generate_alpha158(arguments)
    else:
        from .errors import MCPError
        return [types.TextContent(
            type="text",
            text=MCPError.format_error(
                error_code=MCPError.INVALID_PARAMETER,
                message=f"未知工具: {name}",
                suggestions=["使用 list_tools 查看可用工具"]
            )
        )]


def main():
    """主函数入口"""
    import asyncio
    
    if "--help" in sys.argv:
        print("aigroup-quant-mcp - AI Group Quantitative Analysis MCP Service")
        print("Usage: aigroup-quant-mcp [options]")
        print("Options:")
        print("  --help     Show this help message")
        return
    
    # 添加启动日志到stderr（不影响MCP通信）
    sys.stderr.write("aigroup-quant-mcp v1.0.23 starting...\n")
    sys.stderr.flush()
    
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        sys.stderr.write("\nService stopped by user\n")
        pass
    except Exception as e:
        sys.stderr.write(f"\nError: {e}\n")
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise


async def _main():
    """异步主函数"""
    sys.stderr.write("MCP server initialized, waiting for connections...\n")
    sys.stderr.flush()
    
    async with stdio_server() as (read_stream, write_stream):
        sys.stderr.write("Stdio server started\n")
        sys.stderr.flush()
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    main()