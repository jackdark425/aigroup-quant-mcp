#!/usr/bin/env python3
from quantanalyzer.data.loader import DataLoader

loader = DataLoader()

# 预览数据格式
format_info = loader.preview_data_format('market_data_000001.csv')
print('检测到的格式:', format_info['detected_format'])
print('列名:', format_info['columns'])
print('数据形状:', format_info['shape'])

# 自动转换并加载数据
# 测试便捷方法
df = loader.load_from_market_csv('market_data_000001.csv', target_symbol='000001.SZ')
print('转换成功！')
print('转换后形状:', df.shape)
print('股票代码:', df.index.get_level_values(1).unique())

# 数据验证
validation_report = loader.validate_data(df)
print('数据质量检查: 重复数据', validation_report['duplicate_count'], '条, 股票数量', validation_report['symbols_count'], '个')

print('✅ 转换成功！aigroup-market-mcp数据已完美转换为aigroup-quant-mcp格式')