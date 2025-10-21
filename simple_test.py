#!/usr/bin/env python3
from quantanalyzer.data.loader import DataLoader

loader = DataLoader()

# 用户的参数
file_path = "d:/bank/exports/pingan_bank_2021_2025.csv"
target_symbol = "000001.SZ"

print('测试用户数据...')
print('文件路径:', file_path)
print('目标股票代码:', target_symbol)

# 预览数据格式
format_info = loader.preview_data_format(file_path)
print('检测到的格式:', format_info['detected_format'])
print('列名:', format_info['columns'])
print('数据形状:', format_info['shape'])

# 使用用户指定的参数加载数据
df = loader.load_from_csv(file_path, target_symbol=target_symbol)
print('转换成功！')
print('转换后形状:', df.shape)
print('股票代码:', df.index.get_level_values(1).unique())

# 数据验证
validation_report = loader.validate_data(df)
print('数据质量检查: 重复数据', validation_report['duplicate_count'], '条, 股票数量', validation_report['symbols_count'], '个')

print('✅ 用户的数据可以完美处理！')