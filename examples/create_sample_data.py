"""
创建示例数据
"""
import pandas as pd
import numpy as np

# 生成模拟股票数据
dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600036.SH']

data = []
for symbol in symbols:
    np.random.seed(hash(symbol) % 10000)
    
    # 随机游走生成价格
    returns = np.random.randn(len(dates)) * 0.02
    prices = 10 * np.exp(returns.cumsum())
    
    for i, date in enumerate(dates):
        data.append({
            'symbol': symbol,
            'datetime': date,
            'open': prices[i] * (1 + np.random.randn() * 0.01),
            'close': prices[i],
            'high': prices[i] * (1 + abs(np.random.randn()) * 0.01),
            'low': prices[i] * (1 - abs(np.random.randn()) * 0.01),
            'volume': int(abs(np.random.randn()) * 1000000)
        })

df = pd.DataFrame(data)
df.to_csv('../sample_stock_data.csv', index=False)
print("✅ Sample data created: sample_stock_data.csv")
print(f"   Shape: {df.shape}")
print(f"   Symbols: {df['symbol'].unique().tolist()}")
print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")