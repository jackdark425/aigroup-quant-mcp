"""
检查工具可用性
"""
import sys
sys.path.insert(0, '.')

from quantanalyzer.mcp.schemas import get_all_tool_schemas

print("=" * 60)
print("工具可用性检查")
print("=" * 60)

# 检查所有可用工具
tool_schemas = get_all_tool_schemas()
print("可用工具列表:")
available_tools = []
for tool_schema in tool_schemas:
    tool_name = tool_schema.name
    print(f"  - {tool_name}")
    available_tools.append(tool_name)

print(f"\n总共 {len(tool_schemas)} 个工具可用")

# 检查深度学习工具是否存在
deep_learning_tools = ['train_lstm_model', 'train_gru_model', 'train_transformer_model', 'predict_with_model']
print("\n深度学习工具检查:")
for tool in deep_learning_tools:
    if tool in available_tools:
        print(f"  ✅ {tool} - 可用")
    else:
        print(f"  ❌ {tool} - 缺失")

# 检查因子计算工具
factor_tools = ['calculate_factor', 'generate_alpha158', 'evaluate_factor_ic']
print("\n因子计算工具检查:")
for tool in factor_tools:
    if tool in available_tools:
        print(f"  ✅ {tool} - 可用")
    else:
        print(f"  ❌ {tool} - 缺失")

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)