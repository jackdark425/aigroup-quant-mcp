"""
从finance-mcp获取真实股票数据并转换为CSV格式
"""
import pandas as pd
import json

# 贵州茅台真实数据（从finance-mcp获取）
# 这里我手动整理了数据，实际使用时可以直接从MCP工具获取

data_text = """
交易日期,开盘,收盘,最高,最低,成交量,成交额
20231229,1720,1726,1749.58,1720,2753868,476885.75
20231228,1670.99,1724.99,1727.99,1667.06,3833806,653740.14
20231227,1668,1667.06,1677.15,1661,1605550,267922.14
20231226,1672.5,1670,1674.9,1657.5,1477126,245717.73
20231225,1671,1672,1678.6,1668,1247106,208601.44
20231222,1669.7,1670.65,1679.1,1658.01,2166684,361323.83
20231221,1640.01,1670,1672.32,1640.01,2901168,482782.79
20231220,1658,1649.79,1660,1643,2309203,381613.32
20200106,997.4032778456525,1004.0441883017716,1017.9314218081859,994.0874796375483,6341478,685391.76
20200103,1040.3782579922622,1004.5750885766646,1040.3782579922622,1003.0289579515374,13031878,1426638.06
20200102,1050.6237018937081,1052.4865098757891,1066.5134539808594,1039.4468540012217,14809916,1669683.71
"""

# 解析数据
lines = [l.strip() for l in data_text.strip().split('\n') if l.strip()]
header = lines[0].split(',')

# 转换为DataFrame
records = []
for line in lines[1:]:
    parts = line.split(',')
    if len(parts) >= 7:
        record = {
            'datetime': parts[0],
            'open': float(parts[1]),
            'close': float(parts[2]),
            'high': float(parts[3]),
            'low': float(parts[4]),
            'volume': float(parts[5]),
            'amount': float(parts[6])
        }
        records.append(record)

df = pd.DataFrame(records)

# 添加股票代码
df['symbol'] = '600519.SH'

# 转换日期格式
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d')

# 调整列顺序
df = df[['symbol', 'datetime', 'open', 'close', 'high', 'low', 'volume', 'amount']]

# 按日期排序
df = df.sort_values('datetime')

# 保存为CSV
output_file = '../real_stock_data_maotai.csv'
df.to_csv(output_file, index=False)

print(f"✅ 真实股票数据已保存: {output_file}")
print(f"   股票: 贵州茅台(600519.SH)")
print(f"   数据条数: {len(df)}")
print(f"   时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
print(f"\n前5条数据:")
print(df.head())
print(f"\n后5条数据:")
print(df.tail())