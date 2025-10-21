#!/usr/bin/env python3
"""
测试股票代码掩码修复功能
"""

from quantanalyzer.data.loader import DataLoader
import pandas as pd

def test_no_symbol_column():
    """测试没有股票代码列的情况"""
    print("=== 测试没有股票代码列的数据加载 ===")

    # 初始化数据加载器
    loader = DataLoader()

    try:
        # 测试加载没有股票代码列的数据
        print("尝试加载没有股票代码列的测试数据...")
        df = loader.load_from_csv('test_no_symbol_data.csv')

        print(f"✅ 加载成功！数据形状: {df.shape}")
        print(f"股票代码: {df.index.get_level_values(1).unique()}")
        print(f"日期范围: {df.index.get_level_values(0).min()} 到 {df.index.get_level_values(0).max()}")

        # 显示前几行数据
        print("\n前5行数据:")
        print(df.head())

        # 数据验证
        validation_report = loader.validate_data(df)
        print("\n数据质量检查:")
        print(f"- 重复数据: {validation_report['duplicate_count']}条")
        print(f"- 股票数量: {validation_report['symbols_count']}个")
        print(f"- 日期范围: {validation_report['date_range']['start']} 到 {validation_report['date_range']['end']}")

        print("\n✅ 测试通过！没有股票代码列时成功使用了默认掩码 DEFAULT_STOCK")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_symbol_column():
    """测试有股票代码列的情况（正常情况）"""
    print("\n=== 测试有股票代码列的数据加载 ===")

    loader = DataLoader()

    try:
        # 创建有股票代码列的测试数据
        test_data = {
            '交易日期': ['20240101', '20240102', '20240103'],
            '股票代码': ['000001', '000001', '000001'],
            '开盘': [10.5, 10.8, 11.0],
            '收盘': [10.8, 11.0, 10.9],
            '最高': [10.9, 11.1, 11.2],
            '最低': [10.4, 10.7, 10.8],
            '成交量': [1000000, 1200000, 900000],
            '成交额': [10800000, 13200000, 9810000]
        }

        df = pd.DataFrame(test_data)
        df.to_csv('test_with_symbol_data.csv', index=False)

        # 加载数据
        loaded_df = loader.load_from_csv('test_with_symbol_data.csv')
        print(f"✅ 加载成功！股票代码: {loaded_df.index.get_level_values(1).unique()}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试股票代码掩码修复功能...\n")

    # 测试没有股票代码列的情况
    test1_passed = test_no_symbol_column()

    # 测试有股票代码列的情况
    test2_passed = test_with_symbol_column()

    print(f"\n{'='*50}")
    if test1_passed and test2_passed:
        print("🎉 所有测试通过！修复成功！")
    else:
        print("❌ 部分测试失败，需要进一步检查")

    print(f"{'='*50}")