#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中国建设银行股票数据量化分析脚本
分析时间范围：2021年1月至2025年10月
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建样本数据（实际应该从API获取）
data = [
    {'交易日期': '20251020', '收盘': 9.25, '成交量': 1424353.31, '成交额(万元)': 130469.69},
    {'交易日期': '20251017', '收盘': 9.22, '成交量': 1701796.95, '成交额(万元)': 157589.75},
    {'交易日期': '20251016', '收盘': 9.19, '成交量': 2020122.21, '成交额(万元)': 184051.71},
    {'交易日期': '20251015', '收盘': 8.95, '成交量': 1291223.39, '成交额(万元)': 115278.40},
    {'交易日期': '20251014', '收盘': 8.89, '成交量': 1910550.08, '成交额(万元)': 168450.26},
    # 添加更多历史数据...
    {'交易日期': '20210104', '收盘': 4.736, '成交量': 1955596.81, '成交额(万元)': 121850.99},
    {'交易日期': '20210105', '收盘': 4.699, '成交量': 1816628.75, '成交额(万元)': 112421.13},
    {'交易日期': '20210106', '收盘': 4.744, '成交量': 1346543.94, '成交额(万元)': 83837.64},
]

# 创建DataFrame
df = pd.DataFrame(data)
df['交易日期'] = pd.to_datetime(df['交易日期'], format='%Y%m%d')
df = df.sort_values('交易日期').reset_index(drop=True)

# 计算收益率
df['daily_return'] = df['收盘'].pct_change()
df['log_return'] = np.log(df['收盘']) - np.log(df['收盘'].shift(1))

# 删除缺失值
df = df.dropna()

print("="*80)
print("中国建设银行股票量化分析报告")
print("="*80)

# 1. 基本统计分析
print("\n1. 基本统计分析")
print("-" * 50)
basic_stats = {
    '样本数量': len(df),
    '数据时间范围': f"{df['交易日期'].min().strftime('%Y-%m-%d')} 至 {df['交易日期'].max().strftime('%Y-%m-%d')}",
    '收盘价均值': f"{df['收盘'].mean()".3f"}",
    '收盘价中位数': f"{df['收盘'].median()".3f"}",
    '收盘价标准差': f"{df['收盘'].std()".3f"}",
    '收盘价最小值': f"{df['收盘'].min()".3f"}",
    '收盘价最大值': f"{df['收盘'].max()".3f"}",
    '年化波动率': f"{(df['daily_return'].std() * np.sqrt(252) * 100)".2f"}%",
    '累计收益率': f"{((df['收盘'].iloc[-1] / df['收盘'].iloc[0] - 1) * 100)".2f"}%"
}

for key, value in basic_stats.items():
    print(f"{key}: {value}")

# 2. 收益率分布分析
print("\n2. 收益率分布分析")
print("-" * 50)
returns_stats = {
    '日收益率均值': f"{(df['daily_return'].mean() * 100)".4f"}%",
    '日收益率标准差': f"{(df['daily_return'].std() * 100)".4f"}%",
    '日收益率偏度': f"{df['daily_return'].skew()".4f"}",
    '日收益率峰度': f"{df['daily_return'].kurtosis()".4f"}",
    '最大单日涨幅': f"{(df['daily_return'].max() * 100)".2f"}%",
    '最大单日跌幅': f"{(df['daily_return'].min() * 100)".2f"}%",
    '正收益率天数占比': f"{(len(df[df['daily_return'] > 0]) / len(df) * 100)".2f"}%",
    '夏普比率': f"{(df['daily_return'].mean() / df['daily_return'].std() * np.sqrt(252))".4f"}"
}

for key, value in returns_stats.items():
    print(f"{key}: {value}")

# 3. 波动性分析
print("\n3. 波动性分析")
print("-" * 50)
volatility_analysis = {
    '20日滚动波动率': f"{(df['daily_return'].rolling(20).std() * np.sqrt(252) * 100).iloc[-1]".2f"}%",
    '60日滚动波动率': f"{(df['daily_return'].rolling(60).std() * np.sqrt(252) * 100).iloc[-1]".2f"}%",
    '250日滚动波动率': f"{(df['daily_return'].rolling(250).std() * np.sqrt(252) * 100).iloc[-1]".2f"}%",
    '最大回撤': f"{((df['收盘'].expanding().max() - df['收盘']) / df['收盘'].expanding().max()).max() * 100".2f"}%",
    'Calmar比率': f"{abs(df['daily_return'].mean() / ((df['收盘'].expanding().max() - df['收盘']) / df['收盘'].expanding().max()).max()) * 252".4f"}"
}

for key, value in volatility_analysis.items():
    print(f"{key}: {value}")

# 4. 成交量分析
print("\n4. 成交量分析")
print("-" * 50)
volume_stats = {
    '平均日成交量': f"{df['成交量'].mean()",.0f"}",
    '平均日成交额': f"{df['成交额(万元)'].mean()",.2f"}万元",
    '最大日成交量': f"{df['成交量'].max()",.0f"}",
    '最大日成交额': f"{df['成交额(万元)'].max()",.2f"}万元",
    '成交量放大倍数': f"{df['成交量'].iloc[-1] / df['成交量'].mean()".2f"}倍"
}

for key, value in volume_stats.items():
    print(f"{key}: {value}")

# 5. 技术指标统计
print("\n5. 技术指标统计")
print("-" * 50)

# 计算移动平均线
df['MA5'] = df['收盘'].rolling(5).mean()
df['MA20'] = df['收盘'].rolling(20).mean()
df['MA60'] = df['收盘'].rolling(60).mean()

technical_stats = {
    'MA5当前值': f"{df['MA5'].iloc[-1]".3f"}",
    'MA20当前值': f"{df['MA20'].iloc[-1]".3f"}",
    'MA60当前值': f"{df['MA60'].iloc[-1]".3f"}",
    '金叉死叉信号': "待计算",  # 需要更多数据计算交叉信号
    '均线排列状态': "待分析"  # 需要分析均线排列
}

for key, value in technical_stats.items():
    print(f"{key}: {value}")

print("\n" + "="*80)
print("分析完成！")
print("="*80)