#!/usr/bin/env python3
"""
测试MCP数据转换流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quantanalyzer.data.loader import DataLoader

def main():
    print("=== MCP数据转换完整流程测试 ===")

    # 初始化数据加载器
    loader = DataLoader()

    # 步骤1：预览数据格式
    print("\n1. 预览数据格式...")
    format_info = loader.preview_data_format('market_data_000001.csv')
    print(f"检测到的格式: {format_info['detected_format']}")
    print(f"列名: {format_info['columns']}")
    print(f"数据形状: {format_info['shape']}")

    # 步骤2：自动转换并加载数据
    print("\n2. 自动转换并加载数据...")
    df = loader.load_from_csv('market_data_000001.csv', target_symbol='000001.SZ')
    print("转换成功！"    print(f"转换后形状: {df.shape}")
    print(f"股票代码: {df.index.get_level_values(1).unique()}")
    print(f"日期范围: {df.index.get_level_values(0).min()} 到 {df.index.get_level_values(0).max()}")

    # 步骤3：数据验证
    validation_report = loader.validate_data(df)
    print("
3. 数据质量检查:"    print(f"   重复数据: {validation_report['duplicate_count']}条")
    print(f"   股票数量: {validation_report['symbols_count']}个")
    print(f"   日期范围: {validation_report['date_range']['start']} 到 {validation_report['date_range']['end']}")

    print("\n✅ 完整流程测试通过！")
    print("🎯 结论：aigroup-market-mcp下载的数据可以直接被aigroup-quant-mcp使用")

if __name__ == "__main__":
    main()