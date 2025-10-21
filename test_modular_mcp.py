"""
测试组件化MCP服务
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

async def test_modular_structure():
    """测试组件化结构"""
    
    print("=" * 80)
    print("🔍 测试组件化MCP服务")
    print("=" * 80)
    print()
    
    try:
        # 测试导入新模块
        print("1️⃣ 测试模块导入...")
        from quantanalyzer.mcp import app, main, MCPError
        from quantanalyzer.mcp.errors import validate_period, validate_window_size
        from quantanalyzer.mcp.utils import serialize_response
        from quantanalyzer.mcp.handlers import data_store, factor_store
        print("   ✅ 所有模块导入成功")
        print()
        
        # 测试错误处理
        print("2️⃣ 测试错误处理...")
        test_error = MCPError.format_error(
            error_code=MCPError.DATA_NOT_FOUND,
            message="测试错误",
            details={"test": "data"},
            suggestions=["建议1", "建议2"]
        )
        assert "DATA_NOT_FOUND" in test_error
        assert "测试错误" in test_error
        print("   ✅ 错误格式化正常")
        print()
        
        # 测试参数验证
        print("3️⃣ 测试参数验证...")
        error = validate_period(300)  # 超出范围
        assert error is not None
        assert "2-250" in error
        print("   ✅ period验证正常")
        
        error = validate_window_size([1, 300])  # 包含无效值
        assert error is not None
        print("   ✅ window_size验证正常")
        print()
        
        # 测试序列化
        print("4️⃣ 测试序列化...")
        import pandas as pd
        test_data = {
            "status": "success",
            "timestamp": pd.Timestamp("2023-01-01"),
            "value": 123
        }
        result = serialize_response(test_data)
        assert "success" in result
        print("   ✅ 序列化正常")
        print()
        
        # 测试存储
        print("5️⃣ 测试全局存储...")
        assert isinstance(data_store, dict)
        assert isinstance(factor_store, dict)
        print("   ✅ 全局存储正常")
        print()
        
        # 测试向后兼容
        print("6️⃣ 测试向后兼容...")
        import quantanalyzer.mcp as old_mcp
        assert hasattr(old_mcp, 'app')
        assert hasattr(old_mcp, 'main')
        assert hasattr(old_mcp, 'MCPError')
        print("   ✅ 向后兼容正常")
        print()
        
        print("=" * 80)
        print("✅ 所有测试通过！组件化结构正常工作")
        print("=" * 80)
        print()
        
        # 显示模块结构
        print("📁 组件化模块结构:")
        print("quantanalyzer/")
        print("├── mcp/")
        print("│   ├── __init__.py      (模块初始化)")
        print("│   ├── errors.py        (错误处理 - 239行)")
        print("│   ├── schemas.py       (Schema定义 - 233行)")
        print("│   ├── handlers.py      (工具处理 - 368行)")
        print("│   ├── utils.py         (工具函数 - 51行)")
        print("│   └── server.py        (主服务器 - 113行)")
        print("└── mcp.py               (兼容入口 - 36行)")
        print()
        print("总计: ~1040行代码，分布在7个文件中")
        print("原mcp.py: ~1100行 → 现在36行向后兼容入口")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_modular_structure())
    sys.exit(0 if result else 1)