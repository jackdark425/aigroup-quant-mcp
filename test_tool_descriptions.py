"""
测试MCP工具描述信息是否正常显示
"""

import sys
import os

# 添加项目路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from quantanalyzer.mcp.schemas import (
    get_load_csv_data_schema,
    get_calculate_factor_schema,
    get_generate_alpha158_schema,
    get_evaluate_factor_ic_schema,
    get_list_factors_schema,
    get_quick_start_lstm_schema
)

def test_tool_descriptions():
    """测试所有工具的描述信息"""
    
    tools = [
        ("load_csv_data", get_load_csv_data_schema()),
        ("calculate_factor", get_calculate_factor_schema()),
        ("generate_alpha158", get_generate_alpha158_schema()),
        ("evaluate_factor_ic", get_evaluate_factor_ic_schema()),
        ("list_factors", get_list_factors_schema()),
        ("quick_start_lstm", get_quick_start_lstm_schema()),
    ]
    
    print("=" * 80)
    print("MCP工具描述信息测试")
    print("=" * 80)
    
    for tool_name, tool_schema in tools:
        print(f"\n🔧 工具: {tool_name}")
        print("-" * 40)
        
        # 检查描述信息
        description = tool_schema.description
        if description:
            # 显示描述的前几行
            lines = description.strip().split('\n')[:10]
            for line in lines:
                print(f"  {line}")
            
            # 检查描述长度
            if len(description) > 100:
                print(f"  ✅ 描述信息详细 ({len(description)} 字符)")
            else:
                print(f"  ⚠️ 描述信息可能过短 ({len(description)} 字符)")
        else:
            print(f"  ❌ 缺少描述信息")
        
        # 检查输入schema
        input_schema = tool_schema.inputSchema
        if input_schema:
            print(f"  ✅ 输入Schema完整")
        else:
            print(f"  ❌ 缺少输入Schema")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

if __name__ == "__main__":
    test_tool_descriptions()