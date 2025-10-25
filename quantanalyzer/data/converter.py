"""
数据格式转换器
用于将不同来源的股票数据转换为统一格式

功能特性：
1. 智能列名识别 - 自动识别多种列名变体（中英文、大小写等）
2. 自动删除空列 - 转换过程中自动移除完全为空的列（如空的持仓量列）
3. 多编码支持 - 自动尝试 UTF-8、GBK、GB18030 等编码
4. 灵活的日期格式 - 支持多种日期格式自动转换
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
    
    # 列名映射表 - 支持多种变体的列名自动识别
    COLUMN_NAME_VARIANTS = {
        'datetime': ['交易日期', '日期', 'date', 'datetime', 'time', '时间', 'trade_date', 'trading_date'],
        'symbol': ['股票代码', '代码', 'symbol', 'code', 'stock_code', 'ts_code', '合约代码', 'contract_code', 'instrument'],
        'open': ['开盘', '开盘价', 'open', 'Open', 'OPEN', 'open_price', '开'],
        'high': ['最高', '最高价', 'high', 'High', 'HIGH', 'high_price', '高'],
        'low': ['最低', '最低价', 'low', 'Low', 'LOW', 'low_price', '低'],
        'close': ['收盘', '收盘价', 'close', 'Close', 'CLOSE', 'close_price', '收'],
        'volume': ['成交量', '量', 'volume', 'Volume', 'VOLUME', 'vol', 'Vol', 'VOL', 'trade_volume'],
        'amount': ['成交额', '成交额(万元)', '额', 'amount', 'Amount', 'AMOUNT', 'turnover', 'trade_amount']
    }

    def __init__(self):
        self.detected_format = None
        self._build_reverse_mapping()
    
    def _build_reverse_mapping(self):
        """构建反向映射表，用于快速查找标准列名"""
        self.reverse_mapping = {}
        for standard_name, variants in self.COLUMN_NAME_VARIANTS.items():
            for variant in variants:
                # 存储原始名称
                self.reverse_mapping[variant] = standard_name
                # 存储去除空格后的名称
                self.reverse_mapping[variant.strip()] = standard_name
                # 存储小写版本
                self.reverse_mapping[variant.lower().strip()] = standard_name

    def _normalize_column_name(self, col_name: str) -> str:
        """
        标准化列名，去除空格并转小写
        
        Args:
            col_name: 原始列名
            
        Returns:
            标准化后的列名
        """
        return col_name.strip().lower() if isinstance(col_name, str) else str(col_name)
    
    def _find_standard_column(self, columns: list, standard_name: str) -> Optional[str]:
        """
        从列名列表中找到对应标准名称的列
        
        Args:
            columns: 列名列表
            standard_name: 标准列名（如'open', 'close'等）
            
        Returns:
            找到的原始列名，如果未找到则返回None
        """
        variants = self.COLUMN_NAME_VARIANTS.get(standard_name, [])
        
        # 首先尝试精确匹配
        for col in columns:
            if col in variants:
                return col
        
        # 然后尝试去空格匹配
        for col in columns:
            col_stripped = col.strip()
            if col_stripped in variants:
                return col
        
        # 最后尝试不区分大小写匹配
        for col in columns:
            col_normalized = self._normalize_column_name(col)
            if col_normalized in [self._normalize_column_name(v) for v in variants]:
                return col
        
        return None
    
    def detect_data_format(self, df: pd.DataFrame) -> str:
        """
        自动检测数据格式（支持智能列名识别）

        Args:
            df: 输入的DataFrame

        Returns:
            检测到的格式名称
        """
        columns = df.columns.tolist()
        
        # 尝试智能识别必需的列
        required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        found_columns = {}
        
        for required_col in required_columns:
            found_col = self._find_standard_column(columns, required_col)
            if found_col:
                found_columns[required_col] = found_col
        
        # 如果找到了所有必需列，判断是否为标准格式或需要转换的格式
        if len(found_columns) == len(required_columns):
            # 检查是否已经是标准格式
            standard_columns = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            if all(col in columns for col in standard_columns):
                return 'standard'
            
            # 检查是否包含中文列名（说明是aigroup_market格式或类似格式）
            has_chinese = any(
                any('\u4e00' <= char <= '\u9fff' for char in str(col))
                for col in found_columns.values()
            )
            
            if has_chinese:
                return 'aigroup_market'
            else:
                # 英文列名但不是标准格式，也归类为需要转换的格式
                return 'aigroup_market'  # 使用相同的转换逻辑
        
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
        执行格式转换（使用智能列名识别）

        Args:
            df: 原始数据
            format_config: 格式配置
            target_symbol: 目标股票代码

        Returns:
            转换后的DataFrame
        """
        df = df.copy()
        
        # 🔥 新增：自动删除完全为空的列
        # 检测每列的空值情况
        empty_columns = []
        for col in df.columns:
            # 检查列是否完全为空（全是NaN或空字符串）
            if df[col].isna().all() or (df[col].astype(str).str.strip() == '').all():
                empty_columns.append(col)
        
        # 删除空列
        if empty_columns:
            df = df.drop(columns=empty_columns)
        
        columns = df.columns.tolist()
        
        # 使用智能列名识别构建映射
        column_mapping = {}
        
        # 必需的列
        required_standards = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        for standard_name in required_standards:
            found_col = self._find_standard_column(columns, standard_name)
            if found_col:
                column_mapping[found_col] = standard_name
            else:
                raise ValueError(f"缺少必需的列: {standard_name} (未找到对应的列名变体)")
        
        # 可选的股票代码列
        symbol_col = self._find_standard_column(columns, 'symbol')
        if symbol_col:
            column_mapping[symbol_col] = 'symbol'
        
        # 可选的成交额列
        amount_col = self._find_standard_column(columns, 'amount')
        if amount_col:
            column_mapping[amount_col] = 'amount'
        
        # 重命名列
        df = df.rename(columns=column_mapping)
        
        # 处理股票代码
        if target_symbol:
            df['symbol'] = target_symbol
        elif 'symbol' not in df.columns:
            # 如果没有股票代码列，使用默认值
            df['symbol'] = 'DEFAULT'
        
        # 转换日期格式 - 尝试多种格式
        try:
            # 首先尝试配置的格式
            df['datetime'] = pd.to_datetime(df['datetime'], format=format_config.get('date_format', '%Y%m%d'))
        except:
            # 如果失败，让pandas自动推断
            df['datetime'] = pd.to_datetime(df['datetime'])
        
        # 转换数据类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 处理成交额列
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            # 如果源格式是万元，需要转换为元
            if format_config.get('amount_unit') == '万元':
                df['amount'] = df['amount'] * 10000
        # 🔥 修改：不再创建全是NaN的amount列
        
        # 重新排列列的顺序，只保留实际存在的列
        final_columns = ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'amount']
        existing_columns = [col for col in final_columns if col in df.columns]
        df = df[existing_columns]
        
        # 🔥 新增：删除转换后仍然全是NaN的列（二次清理）
        columns_to_drop = []
        for col in df.columns:
            if col not in ['datetime', 'symbol'] and df[col].isna().all():
                columns_to_drop.append(col)
        
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
        
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