#!/usr/bin/env python3
"""
测试数据格式转换功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from quantanalyzer.data.loader import DataLoader
from quantanalyzer.data.converter import DataFormatConverter


def test_converter():
    """测试数据转换器"""
    print("测试数据转换器...")

    converter = DataFormatConverter()

    # 创建测试数据（模拟aigroup-market格式）
    test_data = {
        '交易日期': ['20251020', '20251017', '20251016'],
        '开盘': [9.13, 9.18, 8.93],
        '收盘': [9.25, 9.22, 9.19],
        '最高': [9.29, 9.34, 9.21],
        '最低': [9.04, 9.14, 8.93],
        '成交量': [1424353.31, 1701796.95, 2020122.21],
        '成交额(万元)': [130469.69, 157589.75, 184051.71],
        '股票代码': ['601939.SH', '601939.SH', '601939.SH']
    }

    df = pd.DataFrame(test_data)
    print(f"原始数据形状: {df.shape}")
    print(f"原始列名: {df.columns.tolist()}")

    # 检测格式
    detected_format = converter.detect_data_format(df)
    print(f"检测到的格式: {detected_format}")

    # 转换格式
    converted_df = converter.convert_to_standard_format(df, target_symbol='601939.SH')
    print(f"转换后数据形状: {converted_df.shape}")
    print(f"转换后列名: {converted_df.columns.tolist()}")

    # 验证转换结果
    assert 'datetime' in converted_df.columns
    assert 'symbol' in converted_df.columns
    assert 'open' in converted_df.columns
    assert 'close' in converted_df.columns
    assert converted_df['symbol'].iloc[0] == '601939.SH'

    print("✓ 数据转换器测试通过！")
    print()


def test_loader():
    """测试数据加载器"""
    print("测试数据加载器...")

    loader = DataLoader()

    # 测试自动转换功能
    try:
        # 创建临时测试文件
        test_data = {
            '交易日期': ['20251020', '20251017'],
            '开盘': [9.13, 9.18],
            '收盘': [9.25, 9.22],
            '最高': [9.29, 9.34],
            '最低': [9.04, 9.14],
            '成交量': [1424353.31, 1701796.95],
            '成交额(万元)': [130469.69, 157589.75],
            '股票代码': ['601939.SH', '601939.SH']
        }

        test_df = pd.DataFrame(test_data)
        temp_file = 'test_market_data.csv'
        test_df.to_csv(temp_file, index=False)

        # 测试加载
        df = loader.load_from_csv(temp_file, target_symbol='601939.SH')

        print(f"加载后数据形状: {df.shape}")
        print(f"索引级别: {df.index.nlevels}")
        print(f"股票代码: {df.index.get_level_values(1).unique()}")

        # 验证结果
        # MultiIndex DataFrame的shape[1]表示列数，不包括索引列
        expected_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        for col in expected_columns:
            assert col in df.columns, f"缺少列: {col}"
        assert df.index.get_level_values(1).unique()[0] == '601939.SH'

        # 清理临时文件
        os.remove(temp_file)

        print("✓ 数据加载器测试通过！")

    except Exception as e:
        print(f"✗ 数据加载器测试失败: {e}")
        raise
    print()


def test_format_detection():
    """测试格式检测功能"""
    print("测试格式检测功能...")

    loader = DataLoader()

    # 测试现有文件
    if os.path.exists('construction_bank_data.csv'):
        format_info = loader.preview_data_format('construction_bank_data.csv')
        print(f"文件格式检测结果: {format_info['detected_format']}")
        print(f"列名: {format_info['columns']}")

        assert format_info['detected_format'] == 'aigroup_market'
        print("✓ 格式检测测试通过！")
    else:
        print("测试文件不存在，跳过格式检测测试")
    print()


def test_supported_formats():
    """测试支持的格式列表"""
    print("测试支持的格式列表...")

    loader = DataLoader()
    formats = loader.get_supported_formats()

    print("支持的格式:")
    for format_name, format_info in formats.items():
        print(f"  {format_name}: {format_info['description']}")

    assert 'aigroup_market' in formats
    assert 'standard' in formats
    print("✓ 支持格式列表测试通过！")
    print()


def main():
    """运行所有测试"""
    print("=== 数据格式转换功能测试 ===\n")

    try:
        test_converter()
        test_loader()
        test_format_detection()
        test_supported_formats()

        print("🎉 所有测试通过！数据格式转换功能正常工作。")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    main()