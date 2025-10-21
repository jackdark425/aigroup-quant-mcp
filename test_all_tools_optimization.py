"""
测试所有MCP工具Schema优化是否生效
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from quantanalyzer.mcp import app
from quantanalyzer import mcp

async def test_all_tools():
    """测试所有工具Schema优化"""
    
    print("=" * 100)
    print("🔍 MCP工具Schema优化全面验证")
    print("=" * 100)
    print()
    
    # 获取工具列表
    tools = await mcp.handle_list_tools()
    
    print(f"✅ 成功获取工具列表")
    print(f"📊 工具总数: {len(tools)}")
    print()
    
    # 定义每个工具的最低要求
    tool_requirements = {
        "load_csv_data": {
            "min_desc_length": 1500,
            "must_contain": ["[📥 数据加载", "功能概述", "适用场景", "工作流程", "性能建议"],
            "min_params": 2,
            "param_checks": {
                "file_path": {"min_desc": 300, "has_examples": True},
                "data_id": {"min_desc": 400, "has_examples": True}
            }
        },
        "calculate_factor": {
            "min_desc_length": 2000,
            "must_contain": ["[🔬 单因子计算", "功能概述", "支持的因子类型", "工作流程"],
            "min_params": 4,
            "param_checks": {
                "data_id": {"min_desc": 200, "has_examples": True},
                "factor_name": {"min_desc": 400, "has_examples": True},
                "factor_type": {"min_desc": 800, "has_examples": True},
                "period": {"min_desc": 600, "has_examples": True}
            }
        },
        "generate_alpha158": {
            "min_desc_length": 1800,
            "must_contain": ["[🔬 因子生成", "功能概述", "因子分类", "工作流程"],
            "min_params": 6,
            "param_checks": {
                "data_id": {"min_desc": 200, "has_examples": True},
                "result_id": {"min_desc": 200, "has_examples": True},
                "rolling_windows": {"min_desc": 500, "has_examples": True}
            }
        },
        "evaluate_factor_ic": {
            "min_desc_length": 2000,
            "must_contain": ["[📊 因子评估", "功能概述", "IC指标解读", "工作流程"],
            "min_params": 3,
            "param_checks": {
                "factor_name": {"min_desc": 300, "has_examples": True},
                "data_id": {"min_desc": 400, "has_examples": True},
                "method": {"min_desc": 600, "has_examples": True}
            }
        },
        "list_factors": {
            "min_desc_length": 1000,
            "must_contain": ["[📋 状态查询", "功能概述", "使用场景", "返回信息说明"],
            "min_params": 0,
            "param_checks": {}
        }
    }
    
    all_passed = True
    results = []
    
    for tool in tools:
        tool_name = tool.name
        print(f"\n{'=' * 100}")
        print(f"🔍 检查工具: {tool_name}")
        print(f"{'=' * 100}")
        
        if tool_name not in tool_requirements:
            print(f"  ⚠️  工具 '{tool_name}' 不在优化列表中（可能是其他工具）")
            continue
        
        req = tool_requirements[tool_name]
        tool_passed = True
        
        # 检查description长度
        desc_length = len(tool.description)
        min_length = req["min_desc_length"]
        
        print(f"\n📝 Description检查:")
        if desc_length >= min_length:
            print(f"  ✅ 长度: {desc_length} 字符 (要求≥{min_length})")
        else:
            print(f"  ❌ 长度不足: {desc_length} 字符 (要求≥{min_length})")
            tool_passed = False
            all_passed = False
        
        # 检查关键内容
        print(f"\n🔍 关键内容检查:")
        for marker in req["must_contain"]:
            if marker in tool.description:
                print(f"  ✅ '{marker}': 存在")
            else:
                print(f"  ❌ '{marker}': 缺失")
                tool_passed = False
                all_passed = False
        
        # 检查参数数量
        schema = tool.inputSchema
        if 'properties' in schema:
            param_count = len(schema['properties'])
        else:
            param_count = 0
        
        print(f"\n📋 参数检查:")
        if param_count >= req["min_params"]:
            print(f"  ✅ 参数数量: {param_count} (要求≥{req['min_params']})")
        else:
            print(f"  ❌ 参数数量不足: {param_count} (要求≥{req['min_params']})")
            tool_passed = False
            all_passed = False
        
        # 检查具体参数
        if req["param_checks"]:
            print(f"\n🔍 参数详细检查:")
            for param_name, param_req in req["param_checks"].items():
                if param_name in schema.get('properties', {}):
                    param = schema['properties'][param_name]
                    param_desc = param.get('description', '')
                    param_examples = param.get('examples', [])
                    
                    # 检查参数描述长度
                    if len(param_desc) >= param_req["min_desc"]:
                        print(f"  ✅ {param_name} 描述: {len(param_desc)} 字符 (要求≥{param_req['min_desc']})")
                    else:
                        print(f"  ❌ {param_name} 描述不足: {len(param_desc)} 字符 (要求≥{param_req['min_desc']})")
                        tool_passed = False
                        all_passed = False
                    
                    # 检查examples
                    if param_req["has_examples"]:
                        if param_examples and len(param_examples) > 0:
                            print(f"  ✅ {param_name} 有{len(param_examples)}个示例")
                        else:
                            print(f"  ❌ {param_name} 缺少示例")
                            tool_passed = False
                            all_passed = False
                else:
                    print(f"  ❌ 参数 '{param_name}' 不存在")
                    tool_passed = False
                    all_passed = False
        
        # 检查工具级examples
        print(f"\n🔍 工具示例检查:")
        if 'examples' in schema and schema['examples']:
            example_count = len(schema['examples'])
            print(f"  ✅ 工具级别有{example_count}个完整示例")
        else:
            print(f"  ⚠️  工具级别缺少完整示例（建议添加）")
            # 不算严重错误，只是警告
        
        # 记录结果
        results.append({
            "tool": tool_name,
            "passed": tool_passed,
            "desc_length": desc_length
        })
        
        if tool_passed:
            print(f"\n✅ {tool_name}: 所有检查通过")
        else:
            print(f"\n❌ {tool_name}: 部分检查未通过")
    
    # 总结
    print(f"\n{'=' * 100}")
    print(f"📊 优化验证总结")
    print(f"{'=' * 100}")
    
    passed_count = sum(1 for r in results if r["passed"])
    total_count = len(results)
    
    print(f"\n通过工具: {passed_count}/{total_count}")
    print(f"\n详细结果:")
    for r in results:
        status = "✅ PASSED" if r["passed"] else "❌ FAILED"
        print(f"  {status}: {r['tool']} (描述长度: {r['desc_length']} 字符)")
    
    print(f"\n{'=' * 100}")
    
    if all_passed:
        print("✅ 所有工具Schema优化验证通过！")
        print("\n📋 优化总结:")
        print(f"  - 优化了 {total_count} 个工具")
        print(f"  - 所有工具description都包含详细说明")
        print(f"  - 所有参数都有详细说明和示例")
        print(f"  - 提供了工作流程和使用建议")
        return True
    else:
        print("❌ 部分工具优化验证未通过")
        print("\n💡 建议:")
        print("  1. 检查未通过的工具")
        print("  2. 补充缺失的内容")
        print("  3. 重新运行验证")
        return False

if __name__ == "__main__":
    result = asyncio.run(test_all_tools())
    sys.exit(0 if result else 1)