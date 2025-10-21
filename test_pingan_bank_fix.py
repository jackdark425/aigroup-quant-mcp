#!/usr/bin/env python3
"""
测试平安银行CSV文件修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantanalyzer.data.loader import DataLoader

def test_pingan_bank_csv():
    """测试平安银行CSV文件加载"""
    file_path = "exports/pingan_bank_direct.csv"

    print("🧪 测试平安银行CSV文件加载...")
    print(f"📁 文件路径: {file_path}")

    try:
        loader = DataLoader()

        # 先预览数据格式
        print("\n📋 预览数据格式:")
        preview_info = loader.preview_data_format(file_path)
        print(f"检测到的格式: {preview_info.get('detected_format', 'unknown')}")
        print(f"列名: {preview_info.get('columns', [])}")

        # 尝试加载数据
        print(f"\n🚀 正在加载数据...")
        data = loader.load_from_csv(file_path, target_symbol="000001.SZ")

        print("✅ 数据加载成功！")
        print(f"📊 数据形状: {data.shape}")
        print(f"📅 日期范围: {data.index.get_level_values(0).min()} 到 {data.index.get_level_values(0).max()}")
        print(f"🏢 股票代码: {data.index.get_level_values(1).unique().tolist()}")

        # 显示前3行数据
        print("\n📈 前3行数据预览:")
        print(data.head(3))

        print("\n✅ 测试通过！修复成功！")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        print(f"❌ 错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pingan_bank_csv()
    sys.exit(0 if success else 1)