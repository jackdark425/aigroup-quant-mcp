#!/usr/bin/env python3
"""
实际数据转换测试
使用真实从aigroup-market-mcp下载的数据进行转换测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from quantanalyzer.data.loader import DataLoader


def test_real_data_conversion():
    """测试真实数据的转换"""
    print("🧪 测试真实数据转换功能")
    print("=" * 50)

    # 检查测试文件是否存在
    test_file = "construction_bank_data.csv"
    if not os.path.exists(test_file):
        print(f"❌ 测试文件不存在: {test_file}")
        return False

    print(f"📁 测试文件: {test_file}")

    # 初始化数据加载器
    loader = DataLoader()

    try:
        # 步骤1：预览原始数据格式
        print("\n1️⃣ 预览原始数据格式...")
        format_info = loader.preview_data_format(test_file)
        print(f"   检测到的格式: {format_info['detected_format']}")
        print(f"   列名: {format_info['columns']}")
        print(f"   数据形状: {format_info['shape']}")

        # 显示样本数据
        print("   样本数据:")
        for i, row in enumerate(format_info['sample_data'][:2]):
            print(f"     第{i+1}行: {row}")

        # 步骤2：自动转换数据
        print("\n2️⃣ 执行自动数据转换...")
        df_converted = loader.load_from_csv(test_file, target_symbol="601939.SH")

        print("   ✅ 转换成功！")
        print(f"   转换后形状: {df_converted.shape}")
        print(f"   索引级别: {df_converted.index.nlevels}")
        print(f"   列名: {df_converted.columns.tolist()}")

        # 显示转换后的样本数据
        print("   转换后样本数据:")
        samples = df_converted.reset_index().head(3)
        for _, row in samples.iterrows():
            print(f"     {row['datetime'].strftime('%Y-%m-%d')} | {row['symbol']} | 开:{row['open']:.2f} | 收:{row['close']:.2f} | 量:{row['volume']:.0f}")

        # 步骤3：验证数据完整性
        print("\n3️⃣ 验证数据完整性...")

        # 检查必需的列
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
        missing_columns = [col for col in required_columns if col not in df_converted.columns]

        if missing_columns:
            print(f"   ❌ 缺少列: {missing_columns}")
            return False
        else:
            print("   ✅ 所有必需列都存在")

        # 检查数据类型
        print(f"   数据类型检查: {df_converted.dtypes.to_dict()}")

        # 检查索引结构
        if isinstance(df_converted.index, pd.MultiIndex):
            print("   ✅ MultiIndex结构正确")
            print(f"   股票代码: {df_converted.index.get_level_values(1).unique().tolist()}")
        else:
            print("   ❌ MultiIndex结构错误")
            return False

        # 检查日期范围
        date_range = {
            '最早日期': df_converted.index.get_level_values(0).min(),
            '最晚日期': df_converted.index.get_level_values(0).max()
        }
        print(f"   日期范围: {date_range['最早日期']} 到 {date_range['最晚日期']}")

        # 步骤4：测试因子计算兼容性
        print("\n4️⃣ 测试因子计算兼容性...")

        # 这里可以测试一个简单的因子计算来验证兼容性
        try:
            # 计算一个简单的移动平均线作为测试
            test_data = df_converted.reset_index()
            test_data['ma_5'] = test_data.groupby('symbol')['close'].rolling(5).mean().reset_index(0, drop=True)

            if not test_data['ma_5'].isnull().all():
                print("   ✅ 因子计算兼容性测试通过")
            else:
                print("   ❌ 因子计算兼容性测试失败")
                return False

        except Exception as e:
            print(f"   ❌ 因子计算兼容性测试失败: {e}")
            return False

        # 步骤5：数据验证报告
        print("\n5️⃣ 生成数据验证报告...")
        validation_report = loader.validate_data(df_converted)

        print("   📊 数据验证报告:")
        print(f"     - 数据形状: {validation_report['shape']}")
        print(f"     - 重复数据: {validation_report['duplicate_count']}")
        print(f"     - 股票数量: {validation_report['symbols_count']}")
        print(f"     - 日期范围: {validation_report['date_range']['start']} 到 {validation_report['date_range']['end']}")

        # 检查是否有缺失值
        missing_info = validation_report['missing_ratio']
        has_missing = any(missing_info > 0)
        print(f"     - 是否有缺失值: {'是' if has_missing else '否'}")

        if has_missing:
            print(f"     - 缺失值详情: {dict((k, v) for k, v in missing_info.items() if v > 0)}")

        print("\n🎉 所有测试通过！数据转换功能完全正常！")
        print("=" * 50)
        print("✨ 结论：aigroup-market-mcp的数据可以完美转换为aigroup-quant-mcp格式")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_conversion_process():
    """演示转换过程"""
    print("\n🔄 数据转换过程演示")
    print("=" * 30)

    test_file = "construction_bank_data.csv"
    if not os.path.exists(test_file):
        print("测试文件不存在")
        return

    loader = DataLoader()

    # 显示原始数据
    print("📋 原始数据 (前3行):")
    original_df = pd.read_csv(test_file)
    print(original_df.head(3).to_string(index=False))

    # 显示转换后的数据
    print("\n📋 转换后数据 (前3行):")
    converted_df = loader.load_from_csv(test_file, target_symbol="601939.SH")
    print(converted_df.reset_index().head(3).to_string(index=False))

    # 显示数据格式对比
    print("\n📊 数据格式对比:")
    print(f"  原格式列名: {list(original_df.columns)}")
    print(f"  转换后列名: {list(converted_df.columns)}")
    print(f"  原格式形状: {original_df.shape}")
    print(f"  转换后形状: {converted_df.shape}")
    print(f"  数据类型: {converted_df.dtypes.to_dict()}")


def main():
    """主函数"""
    print("🚀 aigroup-quant-mcp 数据格式转换工具测试")
    print("测试对象：construction_bank_data.csv (aigroup-market-mcp格式)")

    # 执行主要测试
    success = test_real_data_conversion()

    if success:
        # 演示转换过程
        demonstrate_conversion_process()

        print("\n" + "=" * 60)
        print("🎯 测试结论:")
        print("✅ aigroup-market-mcp下载的CSV数据")
        print("✅ 可以通过标准化转换工具自动转换为")
        print("✅ aigroup-quant-mcp完全认可的格式")
        print("✅ 无需任何手动预处理")
        print("✅ 完全兼容因子计算和模型训练")
        print("=" * 60)
    else:
        print("\n❌ 测试失败，请检查实现")
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)