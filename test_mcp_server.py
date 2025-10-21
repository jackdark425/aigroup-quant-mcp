"""
测试 aigroup-quant-mcp MCP 服务器
"""
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_server():
    """测试MCP服务器"""
    
    # 创建服务器参数
    server_params = StdioServerParameters(
        command="python",
        args=["-m", "quantanalyzer.mcp"],
        env=None
    )
    
    print("正在连接到 aigroup-quant-mcp 服务器...")
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化
            await session.initialize()
            print("✅ 服务器连接成功!\n")
            
            # 列出可用工具
            tools = await session.list_tools()
            print(f"📋 可用工具数量: {len(tools.tools)}\n")
            
            for tool in tools.tools:
                print(f"工具名称: {tool.name}")
                print(f"描述: {tool.description[:100]}..." if len(tool.description) > 100 else f"描述: {tool.description}")
                print("-" * 50)
            
            # 测试 list_factors 工具
            if len(tools.tools) > 0:
                print("\n🧪 测试 list_factors 工具...")
                result = await session.call_tool("list_factors", arguments={})
                print("结果:")
                for content in result.content:
                    if hasattr(content, 'text'):
                        data = json.loads(content.text)
                        print(json.dumps(data, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    asyncio.run(test_server())