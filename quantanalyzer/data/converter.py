"""
数据格式转换器
用于将不同来源的股票数据转换为统一格式
"""

import pandas as pd
import numpy as np
from typing import Union, Dict, Optional, Tuple
from pathlib import Path


class DataFormatConverter:
    """数据格式转换器"""

    # 支持的数据源格式映射
    SOURCE_FORMATS = {
        'aigroup_market': {
            'date_column': '交易日期',
            'open_column': '开盘',
            'close_column': '收盘',
            'high_column': '最高',
            'low_column': '最低',
            'volume_column': '成交量',
            'amount_column': '成交额',  # 兼容'成交额'和'成交额(万元)'
            'symbol_column': '股票代码',
            'date_format': '%Y%m%d',
            'amount_unit': '万元'
        },
        'standard': {
            'date_column': 'datetime',
            'open_column': 'open',
            'close_column': 'close',
            'high_column': 'high',
            'low_column': 'low',
            'volume_column': 'volume',
            'amount_column': 'amount',
            'symbol_column': 'symbol',
            'date_format': '%Y-%m-%d',
            'amount_unit': '元'
        }
    }

    def __init__(self):
        self.detected_format = None

    def detect_data_format(self, df: pd.DataFrame) -> str:
        """
        自动检测数据格式

        Args:
            df: 输入的DataFrame

        Returns:
            检测到的格式名称
        """
        columns = df.columns.tolist()

        # 检测aigroup-market格式
        # 检查核心必需列（股票代码列是可选的）
        core_indicators = ['交易日期', '开盘', '收盘', '最高', '最低', '成交量']
        if not all(col in columns for col in core_indicators):
            pass  # 不满足基本条件，继续检查其他格式
        else:
            # 检查成交额列（可能是'成交额'或'成交额(万元)'）
            has_amount_column = '成交额' in columns or '成交额(万元)' in columns
            if has_amount_column:
                return 'aigroup_market'

        # 检测标准格式
        standard_indicators = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        if all(col in columns for col in standard_indicators):
            return 'standard'

        # 无法识别的格式
        return 'unknown'

    def convert_to_standard_format(
        self,
        df: Union[pd.DataFrame, str, Path],
        source_format: Optional[str] = None,
        target_symbol: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        转换数据到标准格式

        Args:
            df: 输入数据（DataFrame或文件路径）
            source_format: 源数据格式，如果不指定则自动检测
            target_symbol: 目标股票代码，如果不指定则从数据中获取
            **kwargs: 其他转换参数

        Returns:
            转换后的DataFrame（标准格式）
        """
        # 加载数据，尝试多种编码
        if isinstance(df, (str, Path)):
            try:
                df = pd.read_csv(df, **kwargs)
            except UnicodeDecodeError:
                # 如果UTF-8失败，尝试GBK编码（常用于中文文件）
                try:
                    df = pd.read_csv(df, encoding='gbk', **kwargs)
                except UnicodeDecodeError:
                    # 如果GBK也失败，尝试gb18030编码
                    df = pd.read_csv(df, encoding='gb18030', **kwargs)

        df = df.copy()

        # 检测数据格式
        if source_format is None:
            source_format = self.detect_data_format(df)

        if source_format == 'unknown':
            raise ValueError("无法识别的数据格式，请手动指定source_format参数")

        # 获取格式配置
        format_config = self.SOURCE_FORMATS.get(source_format)
        if not format_config:
            raise ValueError(f"不支持的数据格式: {source_format}")

        # 执行转换
        converted_df = self._convert_format(df, format_config, target_symbol)

        return converted_df

    def _convert_format(
        self,
        df: pd.DataFrame,
        format_config: Dict,
        target_symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        执行格式转换

        Args:
            df: 原始数据
            format_config: 格式配置
            target_symbol: 目标股票代码

        Returns:
            转换后的DataFrame
        """
        # 重命名列
        column_mapping = {
            format_config['date_column']: 'datetime',
            format_config['open_column']: 'open',
            format_config['high_column']: 'high',
            format_config['low_column']: 'low',
            format_config['close_column']: 'close',
            format_config['volume_column']: 'volume',
        }

        # 添加股票代码列
        if format_config['symbol_column'] in df.columns:
            column_mapping[format_config['symbol_column']] = 'symbol'

        # 重命名列
        df = df.rename(columns=column_mapping)

        # 确保必需的列存在
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必需的列: {col}")

        # 处理股票代码
        if target_symbol:
            df['symbol'] = target_symbol
        elif 'symbol' not in df.columns:
            # 如果没有股票代码列，使用默认值
            df['symbol'] = 'DEFAULT'

        # 转换日期格式
        df['datetime'] = pd.to_datetime(df['datetime'], format=format_config['date_format'])

        # 转换数据类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 处理成交额列（如果存在）
        # 检查成交额列的各种可能名称
        amount_column_names = ['amount', '成交额', '成交额(万元)']
        amount_column = None

        for col_name in amount_column_names:
            if col_name in df.columns:
                amount_column = col_name
                break

        if amount_column:
            df['amount'] = pd.to_numeric(df[amount_column], errors='coerce')

            # 如果源格式是万元，需要转换为元
            if format_config['amount_unit'] == '万元':
                df['amount'] = df['amount'] * 10000
        else:
            # 如果没有成交额列，设置为NaN
            df['amount'] = np.nan

        # 重新排列列的顺序
        final_columns = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']
        existing_columns = [col for col in final_columns if col in df.columns]
        df = df[existing_columns]

        # 排序
        df = df.sort_values(['datetime', 'symbol']).reset_index(drop=True)

        return df

    def validate_converted_data(self, df: pd.DataFrame) -> Dict:
        """
        验证转换后的数据质量

        Args:
            df: 转换后的DataFrame

        Returns:
            验证报告
        """
        report = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'date_range': {
                'start': df['datetime'].min() if 'datetime' in df.columns else None,
                'end': df['datetime'].max() if 'datetime' in df.columns else None
            },
            'symbols': df['symbol'].unique().tolist() if 'symbol' in df.columns else [],
            'price_range': {
                'open_min': df['open'].min() if 'open' in df.columns else None,
                'open_max': df['open'].max() if 'open' in df.columns else None,
                'close_min': df['close'].min() if 'close' in df.columns else None,
                'close_max': df['close'].max() if 'close' in df.columns else None,
            }
        }

        return report

    def get_supported_formats(self) -> Dict:
        """
        获取支持的数据格式说明

        Returns:
            格式说明字典
        """
        return {
            'aigroup_market': {
                'description': 'aigroup-market-mcp下载的CSV格式',
                'columns': self.SOURCE_FORMATS['aigroup_market'],
                'features': ['中文列名', '万元单位成交额', 'YYYYMMDD日期格式']
            },
            'standard': {
                'description': 'aigroup-quant-mcp标准格式',
                'columns': self.SOURCE_FORMATS['standard'],
                'features': ['英文列名', '元单位成交额', 'YYYY-MM-DD日期格式']
            }
        }