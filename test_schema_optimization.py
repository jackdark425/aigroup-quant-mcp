"""
测试MCP工具Schema优化是否生效
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from quantanalyzer.mcp import app

async def test_schema_optimization():
    """测试Schema优化"""
    
    print("=" * 80)
    print("🔍 MCP工具Schema优化验证")
    print("=" * 80)
    print()
    
    # 获取工具列表
    try:
        tools = await app._tool_handlers['list_tools']()
    except:
        # 尝试直接调用
        from quantanalyzer import mcp
        tools = await mcp.handle_list_tools()
    
    print(f"✅ 成功获取工具列表")
    print(f"📊 工具总数: {len(tools)}")
    print()
    
    # 检查load_csv_data工具
    load_csv_tool = None
    for tool in tools:
        if tool.name == "load_csv_data":
            load_csv_tool = tool
            break
    
    if not load_csv_tool:
        print("❌ 未找到load_csv_data工具！")
        return False
    
    print("🔍 检查 load_csv_data 工具：")
    print("-" * 80)
    
    # 检查description长度
    desc_length = len(load_csv_tool.description)
    print(f"📝 Description长度: {desc_length} 字符")
    
    if desc_length < 100:
        print(f"❌ FAILED: Description太短（{desc_length}字符），优化未生效")
        return False
    else:
        print(f"✅ PASSED: Description足够详细（{desc_length}字符）")
    
    # 检查关键内容
    key_markers = {
        "分类标签": "[📥 数据加载",
        "功能概述": "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "适用场景": "🎯 适用场景",
        "工作流程": "🎬 典型工作流",
        "性能建议": "⚡ 性能建议",
        "预计耗时": "⏱️ 预计耗时"
    }
    
    print()
    print("🔍 检查关键内容：")
    all_passed = True
    for name, marker in key_markers.items():
        if marker in load_csv_tool.description:
            print(f"  ✅ {name}: 存在")
        else:
            print(f"  ❌ {name}: 缺失")
            all_passed = False
    
    # 检查参数说明
    print()
    print("🔍 检查参数Schema：")
    
    schema = load_csv_tool.inputSchema
    
    # 检查file_path参数
    if 'properties' in schema and 'file_path' in schema['properties']:
        file_path_desc = schema['properties']['file_path'].get('description', '')
        if len(file_path_desc) > 100:
            print(f"  ✅ file_path参数说明: {len(file_path_desc)} 字符")
        else:
            print(f"  ❌ file_path参数说明太简单: {len(file_path_desc)} 字符")
            all_passed = False
        
        # 检查examples
        if 'examples' in schema['properties']['file_path']:
            examples = schema['properties']['file_path']['examples']
            print(f"  ✅ file_path有{len(examples)}个示例")
        else:
            print(f"  ❌ file_path缺少examples")
            all_passed = False
    else:
        print("  ❌ file_path参数定义缺失")
        all_passed = False
    
    # 检查data_id参数
    if 'properties' in schema and 'data_id' in schema['properties']:
        data_id_desc = schema['properties']['data_id'].get('description', '')
        if len(data_id_desc) > 100:
            print(f"  ✅ data_id参数说明: {len(data_id_desc)} 字符")
        else:
            print(f"  ❌ data_id参数说明太简单: {len(data_id_desc)} 字符")
            all_passed = False
        
        # 检查examples
        if 'examples' in schema['properties']['data_id']:
            examples = schema['properties']['data_id']['examples']
            print(f"  ✅ data_id有{len(examples)}个示例")
        else:
            print(f"  ❌ data_id缺少examples")
            all_passed = False
    else:
        print("  ❌ data_id参数定义缺失")
        all_passed = False
    
    # 检查工具级别的examples
    if 'examples' in schema:
        tool_examples = schema['examples']
        print(f"  ✅ 工具级别有{len(tool_examples)}个完整示例")
    else:
        print(f"  ❌ 缺少工具级别的使用示例")
        all_passed = False
    
    print()
    print("=" * 80)
    
    if all_passed:
        print("✅ 所有检查通过！优化已成功应用！")
        print()
        print("📋 优化效果总结：")
        print(f"  - Description长度: {desc_length} 字符（优化前约12字符）")
        print(f"  - 包含6个关键章节（功能概述、适用场景、工作流程等）")
        print(f"  - 参数说明详细且有示例")
        print(f"  - 提供完整的使用示例")
        return True
    else:
        print("❌ 部分检查未通过，优化可能未完全生效")
        print()
        print("💡 建议：")
        print("  1. 重启MCP服务器")
        print("  2. 清除Python缓存")
        print("  3. 重新加载模块")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_schema_optimization())
    sys.exit(0 if result else 1)