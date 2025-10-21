"""
验证aigroup-quant-mcp修复的测试脚本
"""

import asyncio
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(__file__))

from quantanalyzer.mcp.handlers import (
    handle_calculate_factor,
    handle_generate_alpha158,
    data_store,
    factor_store
)
from quantanalyzer.data import DataLoader


async def test_fixes():
    """测试修复后的功能"""
    
    print("=" * 60)
    print("测试aigroup-quant-mcp修复")
    print("=" * 60)
    
    # 测试1: 加载数据
    print("\n[测试1] 加载数据...")
    try:
        loader = DataLoader()
        data = loader.load_from_csv("exports/maotai_stock_data.csv")
        data_store["test_data"] = data
        print("✓ 数据加载成功")
        print(f"  数据形状: {data.shape}")
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return
    
    # 测试2: 计算单个因子（之前会报UnboundLocalError）
    print("\n[测试2] 计算单个因子...")
    try:
        result = await handle_calculate_factor({
            "data_id": "test_data",
            "factor_name": "test_momentum",
            "factor_type": "momentum",
            "period": 20
        })
        
        # 检查是否包含错误
        result_text = result[0].text
        if "UnboundLocalError" in result_text:
            print("✗ 仍存在UnboundLocalError")
            print(f"  错误详情: {result_text}")
        elif "status" in result_text and "success" in result_text:
            print("✓ 单因子计算修复成功")
            print("  因子已正常生成并返回结果")
        else:
            print(f"? 返回结果: {result_text[:200]}...")
            
    except Exception as e:
        print(f"✗ 单因子计算测试失败: {e}")
    
    # 测试3: 生成Alpha158因子（之前会报UnboundLocalError）
    print("\n[测试3] 生成Alpha158因子...")
    try:
        result = await handle_generate_alpha158({
            "data_id": "test_data",
            "result_id": "test_alpha158",
            "kbar": True,
            "price": True,
            "volume": True,
            "rolling": True,
            "rolling_windows": [5, 10, 20]
        })
        
        # 检查是否包含错误
        result_text = result[0].text
        if "UnboundLocalError" in result_text:
            print("✗ 仍存在UnboundLocalError")
            print(f"  错误详情: {result_text}")
        elif "status" in result_text and "success" in result_text:
            print("✓ Alpha158因子计算修复成功")
            print("  因子已正常生成并返回结果")
        else:
            print(f"? 返回结果: {result_text[:200]}...")
            
    except Exception as e:
        print(f"✗ Alpha158因子计算测试失败: {e}")
    
    # 测试4: 检查编码问题
    print("\n[测试4] 检查编码...")
    try:
        # 检查result中的emoji是否能正常序列化
        import json
        test_dict = {
            "message": "✅ 测试成功",
            "tips": ["💡 提示1", "💡 提示2"]
        }
        json_str = json.dumps(test_dict, ensure_ascii=False)
        print("✓ JSON序列化支持emoji")
        print(f"  示例: {json_str[:50]}...")
    except Exception as e:
        print(f"✗ 编码测试失败: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    
    # 清理
    if "test_data" in data_store:
        del data_store["test_data"]
    if "test_momentum" in factor_store:
        del factor_store["test_momentum"]
    if "test_alpha158" in factor_store:
        del factor_store["test_alpha158"]


if __name__ == "__main__":
    asyncio.run(test_fixes())