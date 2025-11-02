#!/usr/bin/env python3
"""
最终验证脚本 - 检查aigroup-quant-mcp项目修复完成状态
"""

import sys
sys.path.insert(0, '.')

try:
    from quantanalyzer.mcp.schemas import get_all_tool_schemas
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

def main():
    print('🎯 最终验证 - 项目修复完成状态')
    print('=' * 60)
    
    # 检查所有可用工具
    tool_schemas = get_all_tool_schemas()
    
    # 检查核心工具是否正常
    core_tools = [
        'preprocess_data', 
        'calculate_factor', 
        'generate_alpha158', 
        'apply_processor_chain',
        'evaluate_factor_ic', 
        'train_ml_model', 
        'predict_ml_model',
        'merge_factor_data',
        'list_factors'
    ]
    
    print('📋 核心工具状态:')
    available_tools = [tool.name for tool in tool_schemas]
    all_core_tools_ok = True
    for tool in core_tools:
        if tool in available_tools:
            print(f'  ✅ {tool} - 正常')
        else:
            print(f'  ❌ {tool} - 缺失')
            all_core_tools_ok = False
    
    print(f'\n📊 工具统计:')
    print(f'  总工具数: {len(tool_schemas)}')
    print(f'  核心工具: {len([t for t in core_tools if t in available_tools])}/{len(core_tools)}')
    
    # 检查机器学习算法支持
    print(f'\n🤖 机器学习算法支持:')
    ml_tool = next((t for t in tool_schemas if t.name == 'train_ml_model'), None)
    if ml_tool:
        model_type_param = next((p for p in ml_tool.inputSchema.properties if p.name == 'model_type'), None)
        if model_type_param:
            print(f'  支持的算法: {len(model_type_param.enum)} 种')
            print(f'  算法类型: {model_type_param.enum}')
            ml_algorithms_ok = len(model_type_param.enum) == 15
        else:
            ml_algorithms_ok = False
    else:
        ml_algorithms_ok = False
        print('  ❌ train_ml_model 工具未找到')
    
    # 检查深度学习工具是否完全移除
    deep_learning_tools = ['train_lstm_model', 'train_gru_model', 'train_transformer_model', 'predict_with_model']
    dl_tools_removed = True
    for tool in deep_learning_tools:
        if tool in available_tools:
            dl_tools_removed = False
            break
    
    print(f'\n🧠 深度学习工具状态:')
    if dl_tools_removed:
        print(f'  深度学习工具: ✅ 已完全移除')
    else:
        print(f'  深度学习工具: ❌ 仍然存在')
    
    print(f'\n🎉 最终验证结果:')
    if all_core_tools_ok:
        print(f'  核心工具完整性: ✅ 通过')
    else:
        print(f'  核心工具完整性: ❌ 失败')
        
    if ml_algorithms_ok:
        print(f'  机器学习算法: ✅ 15/15 完全支持')
    else:
        print(f'  机器学习算法: ❌ 算法支持不完整')
        
    if dl_tools_removed:
        print(f'  深度学习工具移除: ✅ 完成')
    else:
        print(f'  深度学习工具移除: ❌ 未完成')
    
    if all_core_tools_ok and ml_algorithms_ok and dl_tools_removed:
        print(f'\n🎊 所有修复任务已完成！项目现在可以正常使用。')
        return True
    else:
        print(f'\n⚠️ 部分修复任务未完成，需要进一步检查。')
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)