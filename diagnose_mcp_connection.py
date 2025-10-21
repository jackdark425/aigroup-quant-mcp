"""
MCP连接诊断脚本
测试MCP服务器能否正常启动并保持运行
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def diagnose_mcp():
    """诊断MCP服务"""
    
    print("=" * 80)
    print("🔍 MCP连接诊断")
    print("=" * 80)
    print()
    
    try:
        # 测试1: 导入模块
        print("1️⃣ 测试模块导入...")
        from quantanalyzer.mcp import app, main
        from quantanalyzer.mcp.server import handle_list_tools, handle_list_resources
        print("   ✅ 模块导入成功")
        print()
        
        # 测试2: 测试工具列表
        print("2️⃣ 测试工具列表...")
        tools = await handle_list_tools()
        print(f"   ✅ 工具数量: {len(tools)}")
        for tool in tools:
            print(f"      - {tool.name}")
        print()
        
        # 测试3: 测试资源列表
        print("3️⃣ 测试资源列表...")
        resources = await handle_list_resources()
        print(f"   ✅ 资源数量: {len(resources)}")
        for res in resources:
            print(f"      - {res.name} ({res.uri})")
        print()
        
        # 测试4: 测试工具调用路由
        print("4️⃣ 测试工具调用路由...")
        from quantanalyzer.mcp.server import handle_call_tool
        from quantanalyzer.mcp.errors import MCPError
        
        # 测试list_factors（最简单的工具）
        try:
            result = await handle_call_tool("list_factors", {})
            print("   ✅ list_factors调用成功")
        except Exception as e:
            print(f"   ❌ list_factors调用失败: {e}")
        print()
        
        # 测试5: 检查MCP服务器是否可以初始化
        print("5️⃣ 测试MCP服务器初始化...")
        try:
            options = app.create_initialization_options()
            print(f"   ✅ 初始化选项创建成功")
        except Exception as e:
            print(f"   ❌ 初始化失败: {e}")
        print()
        
        # 测试6: 检查可能的问题
        print("6️⃣ 检查潜在问题...")
        issues = []
        
        # 检查工具数量
        if len(tools) < 5:
            issues.append("工具数量不足（当前{len(tools)}个，应该至少5个）")
        
        # 检查是否有import错误
        try:
            from quantanalyzer.data import DataLoader
            from quantanalyzer.factor import Alpha158Generator
            print("   ✅ QuantAnalyzer核心模块导入正常")
        except Exception as e:
            issues.append(f"QuantAnalyzer模块导入失败: {e}")
        
        if issues:
            print("   发现问题:")
            for issue in issues:
                print(f"      ⚠️  {issue}")
        else:
            print("   ✅ 未发现明显问题")
        print()
        
        print("=" * 80)
        print("📊 诊断结果")
        print("=" * 80)
        print()
        
        if not issues:
            print("✅ MCP服务器代码正常，可以正常运行")
            print()
            print("💡 如果Roo客户端仍然报错 'Connection closed'，可能原因:")
            print("  1. MCP服务器启动后立即退出（检查是否有异常）")
            print("  2. stdio通信管道问题（检查.roo/mcp.json配置）")
            print("  3. Roo客户端缓存问题（尝试重启Roo应用）")
            print("  4. Python环境问题（确认使用正确的Python解释器）")
            print()
            print("🔧 建议排查步骤:")
            print("  1. 完全关闭并重启Roo应用")
            print("  2. 检查.roo/mcp.json中的command和args配置")
            print("  3. 确认Python环境中安装了mcp库: pip install mcp")
            print("  4. 查看Roo的MCP日志（如果有的话）")
        else:
            print("❌ 发现潜在问题，需要修复")
            for issue in issues:
                print(f"  - {issue}")
        
        print()
        return len(issues) == 0
        
    except Exception as e:
        print(f"❌ 诊断过程出错: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(diagnose_mcp())
    sys.exit(0 if result else 1)