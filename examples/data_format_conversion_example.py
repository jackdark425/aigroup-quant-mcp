#!/usr/bin/env python3
"""
数据格式转换示例
演示如何使用aigroup-quant-mcp处理不同来源的股票数据
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantanalyzer.data.loader import DataLoader
from quantanalyzer.data.converter import DataFormatConverter


def main():
    """主函数"""
    print("=== aigroup-quant-mcp 数据格式转换示例 ===\n")

    # 初始化数据加载器
    loader = DataLoader()

    # 示例1：自动检测和转换数据格式
    print("示例1：自动检测和转换数据格式")
    print("-" * 40)

    # 使用一个来自aigroup-market-mcp的数据文件
    # 这里使用项目中的示例文件
    sample_file = "construction_bank_data.csv"

    if os.path.exists(sample_file):
        try:
            # 预览数据格式
            print("正在预览数据格式...")
            format_info = loader.preview_data_format(sample_file)
            print(f"检测到的格式: {format_info['detected_format']}")
            print(f"列名: {format_info['columns']}")
            print(f"数据形状: {format_info['shape']}")
            print()

            # 自动转换并加载数据
            print("正在自动转换并加载数据...")
            df = loader.load_from_csv(sample_file, target_symbol="601939.SH")
            print("数据加载成功！")
            print(f"转换后的数据形状: {df.shape}")
            print(f"股票代码: {df.index.get_level_values(1).unique()}")
            print(f"日期范围: {df.index.get_level_values(0).min()} 到 {df.index.get_level_values(0).max()}")
            print()

            # 数据验证
            validation_report = loader.validate_data(df)
            print("数据验证报告:")
            print(f"  形状: {validation_report['shape']}")
            print(f"  重复数据: {validation_report['duplicate_count']}")
            print(f"  股票数量: {validation_report['symbols_count']}")
            print()

        except Exception as e:
            print(f"处理失败: {e}")
            print("尝试手动指定格式...")
            try:
                df = loader.load_from_csv(
                    sample_file,
                    source_format='aigroup_market',
                    target_symbol="601939.SH"
                )
                print("手动指定格式成功！")
            except Exception as e2:
                print(f"手动指定格式也失败了: {e2}")
    else:
        print(f"示例文件不存在: {sample_file}")
    print()

    # 示例2：手动转换数据格式
    print("示例2：手动转换数据格式")
    print("-" * 40)

    converter = DataFormatConverter()

    # 显示支持的格式
    print("支持的数据格式:")
    supported_formats = converter.get_supported_formats()
    for format_name, format_info in supported_formats.items():
        print(f"  {format_name}: {format_info['description']}")
    print()

    # 示例3：便捷的market格式加载方法
    print("示例3：使用便捷的market格式加载方法")
    print("-" * 40)

    if os.path.exists(sample_file):
        try:
            # 使用便捷方法加载market格式数据
            df_market = loader.load_from_market_csv(sample_file, target_symbol="601939.SH")
            print("使用便捷方法加载成功！")
            print(f"数据形状: {df_market.shape}")
            print(f"列名: {df_market.columns.tolist()}")
            print()

            # 查看前几行数据
            print("转换后的数据预览:")
            sample_data = df_market.reset_index().head(3)
            for _, row in sample_data.iterrows():
                print(f"  {row['datetime'].strftime('%Y-%m-%d')} | {row['symbol']} | 开盘: {row['open']} | 收盘: {row['close']}")
            print()

        except Exception as e:
            print(f"便捷方法加载失败: {e}")
    print()

    # 示例4：数据验证
    print("示例4：数据验证和质量检查")
    print("-" * 40)

    if os.path.exists(sample_file):
        try:
            df = loader.load_from_csv(sample_file, target_symbol="601939.SH")

            # 使用转换器进行数据验证
            validation_report = loader.converter.validate_converted_data(
                df.reset_index()
            )

            print("数据质量报告:")
            print(f"  数据形状: {validation_report['shape']}")
            print(f"  缺失值情况: {validation_report['missing_values']}")
            print(f"  数据类型: {validation_report['data_types']}")
            print(f"  股票代码: {validation_report['symbols']}")
            print(f"  价格范围: 开盘 {validation_report['price_range']['open_min']:.2f} - {validation_report['price_range']['open_max']:.2f}")
            print()

        except Exception as e:
            print(f"数据验证失败: {e}")

    print("=== 示例完成 ===")
    print()
    print("使用提示:")
    print("1. 对于aigroup-market-mcp下载的数据，可以直接使用load_from_csv()方法")
    print("2. 系统会自动检测数据格式并进行转换")
    print("3. 如果自动检测失败，可以手动指定source_format='aigroup_market'")
    print("4. 使用load_from_market_csv()方法可以更便捷地加载market格式数据")
    print("5. 转换后的数据完全兼容aigroup-quant-mcp的所有因子计算和模型训练功能")


if __name__ == "__main__":
    main()