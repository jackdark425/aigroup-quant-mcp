"""
数据加载器
支持多种数据格式的自动转换和加载
"""
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional
from pathlib import Path

from .converter import DataFormatConverter


class DataLoader:
    """数据加载器"""

    def __init__(self):
        self.data_cache = {}
        self.converter = DataFormatConverter()

    def load_from_csv(
        self,
        file_path: Union[str, Path],
        symbol_col: str = "symbol",
        datetime_col: str = "datetime",
        auto_convert: bool = True,
        source_format: Optional[str] = None,
        target_symbol: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        从CSV加载数据，支持多种格式自动转换

        Args:
            file_path: CSV文件路径
            symbol_col: 股票代码列名（用于标准格式）
            datetime_col: 日期列名（用于标准格式）
            auto_convert: 是否自动转换数据格式
            source_format: 源数据格式，如果不指定则自动检测
            target_symbol: 目标股票代码，如果不指定则从数据中获取
            **kwargs: 传递给pd.read_csv的其他参数

        Returns:
            MultiIndex DataFrame (datetime, symbol)
        """
        # 读取原始数据，尝试多种编码
        try:
            raw_df = pd.read_csv(file_path, **kwargs)
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试GBK编码（常用于中文文件）
            try:
                raw_df = pd.read_csv(file_path, encoding='gbk', **kwargs)
            except UnicodeDecodeError:
                # 如果GBK也失败，尝试gb18030编码
                raw_df = pd.read_csv(file_path, encoding='gb18030', **kwargs)

        # 保存检测到的格式，用于回退逻辑
        detected_format = 'unknown'
        conversion_occurred = False

        # 如果启用自动转换且数据格式不是标准格式，则进行转换
        if auto_convert:
            try:
                # 检测数据格式
                detected_format = self.converter.detect_data_format(raw_df)
                if detected_format != 'standard' and detected_format != 'unknown':
                    # 转换为标准格式
                    raw_df = self.converter.convert_to_standard_format(
                        raw_df,
                        source_format=detected_format,
                        target_symbol=target_symbol
                    )
                    conversion_occurred = True
            except Exception as e:
                # 如果转换失败，回退到原始方法
                # 不使用print避免Windows编码问题
                conversion_occurred = False

        # 根据是否进行了转换来决定使用的列名
        if conversion_occurred:
            # 如果进行了转换，使用标准列名
            expected_datetime_col = 'datetime'
            expected_symbol_col = 'symbol'
        else:
            # 如果没有转换，尝试多种列名格式
            datetime_candidates = ['交易日期', 'datetime', 'date', '时间', 'trade_date']
            symbol_candidates = ['股票代码', 'symbol', 'code', 'stock_code', 'ticker']
            
            # 找到第一个存在的日期列
            expected_datetime_col = None
            for candidate in datetime_candidates:
                if candidate in raw_df.columns:
                    expected_datetime_col = candidate
                    break
            
            # 找到第一个存在的股票代码列
            expected_symbol_col = None
            for candidate in symbol_candidates:
                if candidate in raw_df.columns:
                    expected_symbol_col = candidate
                    break
            
            # 如果找不到日期列，报错
            if expected_datetime_col is None:
                raise ValueError(f"找不到日期列，尝试的列名: {datetime_candidates}")
            
            # 如果找不到股票代码列，使用默认值
            if expected_symbol_col is None:
                if target_symbol:
                    expected_symbol_col = 'symbol'  # 创建新列
                    raw_df[expected_symbol_col] = target_symbol
                else:
                    # 使用默认股票代码
                    expected_symbol_col = 'symbol'
                    default_symbol = "DEFAULT_STOCK"
                    raw_df[expected_symbol_col] = default_symbol

        # 确保日期列存在且格式正确
        if expected_datetime_col not in raw_df.columns:
            raise ValueError(f"找不到日期列: {expected_datetime_col}")

        if not pd.api.types.is_datetime64_any_dtype(raw_df[expected_datetime_col]):
            raw_df[expected_datetime_col] = pd.to_datetime(raw_df[expected_datetime_col])

        # 确保股票代码列存在
        if expected_symbol_col not in raw_df.columns:
            if target_symbol:
                raw_df[expected_symbol_col] = target_symbol
            else:
                # 使用默认股票代码，避免报错
                default_symbol = "DEFAULT_STOCK"
                raw_df[expected_symbol_col] = default_symbol

        # 设置MultiIndex
        df = raw_df.set_index([expected_datetime_col, expected_symbol_col])
        df = df.sort_index()

        return df

    def load_from_market_csv(
        self,
        file_path: Union[str, Path],
        target_symbol: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        便捷方法：直接加载aigroup-market-mcp格式的CSV文件

        Args:
            file_path: CSV文件路径
            target_symbol: 股票代码，如果不指定则从数据中获取
            **kwargs: 其他参数

        Returns:
            MultiIndex DataFrame
        """
        return self.load_from_csv(
            file_path,
            auto_convert=True,
            source_format='aigroup_market',
            target_symbol=target_symbol,
            **kwargs
        )

    def preview_data_format(self, file_path: Union[str, Path], **kwargs) -> Dict:
        """
        预览文件数据格式

        Args:
            file_path: CSV文件路径
            **kwargs: 传递给pd.read_csv的参数

        Returns:
            数据格式信息
        """
        try:
            df = pd.read_csv(file_path, **kwargs)
            detected_format = self.converter.detect_data_format(df)

            info = {
                'detected_format': detected_format,
                'columns': df.columns.tolist(),
                'shape': df.shape,
                'sample_data': df.head(3).to_dict('records'),
                'format_details': self.converter.get_supported_formats().get(detected_format, {})
            }

            return info
        except Exception as e:
            return {
                'error': str(e),
                'detected_format': 'unknown'
            }

    def convert_and_save(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        source_format: Optional[str] = None,
        target_symbol: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        转换数据格式并保存到新文件

        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            source_format: 源格式，如果不指定则自动检测
            target_symbol: 目标股票代码
            **kwargs: 其他参数

        Returns:
            输出文件路径
        """
        try:
            # 转换数据
            converted_df = self.convert_to_standard_format(
                input_path,
                source_format=source_format,
                target_symbol=target_symbol,
                **kwargs
            )

            # 保存转换后的数据
            converted_df.to_csv(output_path, index=False)

            return str(output_path)
        except Exception as e:
            raise ValueError(f"数据转换失败: {e}")

    def convert_to_standard_format(
        self,
        df: Union[pd.DataFrame, str, Path],
        source_format: Optional[str] = None,
        target_symbol: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        转换数据到标准格式（便捷方法）

        Args:
            df: 输入数据或文件路径
            source_format: 源格式
            target_symbol: 目标股票代码
            **kwargs: 其他参数

        Returns:
            转换后的DataFrame
        """
        return self.converter.convert_to_standard_format(
            df,
            source_format=source_format,
            target_symbol=target_symbol,
            **kwargs
        )

    def get_supported_formats(self) -> Dict:
        """
        获取支持的数据格式说明

        Returns:
            格式说明字典
        """
        return self.converter.get_supported_formats()

    def load_from_dataframe(
        self,
        df: pd.DataFrame,
        symbol_col: str = "symbol",
        datetime_col: str = "datetime"
    ) -> pd.DataFrame:
        """
        从DataFrame加载数据

        Args:
            df: 原始DataFrame
            symbol_col: 股票代码列名
            datetime_col: 日期列名

        Returns:
            MultiIndex DataFrame
        """
        df = df.copy()

        # 确保日期格式正确
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col])

        # 设置MultiIndex
        if not isinstance(df.index, pd.MultiIndex):
            df = df.set_index([datetime_col, symbol_col])
            df = df.sort_index()

        return df

    def load_from_dict(
        self,
        data: Dict[str, List],
        symbol_col: str = "symbol",
        datetime_col: str = "datetime"
    ) -> pd.DataFrame:
        """
        从字典加载数据

        Args:
            data: 包含列名和数据的字典
            symbol_col: 股票代码列名
            datetime_col: 日期列名

        Returns:
            MultiIndex DataFrame
        """
        # 创建DataFrame
        df = pd.DataFrame(data)

        # 添加默认的symbol和datetime列
        if symbol_col not in df.columns:
            df[symbol_col] = 'default_symbol'
        if datetime_col not in df.columns:
            df[datetime_col] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')

        # 使用现有的load_from_dataframe方法
        return self.load_from_dataframe(df, symbol_col, datetime_col)

    def load_multiple_files(
        self,
        file_pattern: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        批量加载文件

        Args:
            file_pattern: 文件匹配模式，如 "data/*.csv"

        Returns:
            合并后的DataFrame
        """
        from glob import glob

        files = glob(file_pattern)
        dfs = []

        for file in files:
            df = self.load_from_csv(file, **kwargs)
            dfs.append(df)

        return pd.concat(dfs, axis=0).sort_index()

    def validate_data(self, df: pd.DataFrame) -> Dict:
        """
        数据质量检查

        Returns:
            验证报告
        """
        report = {
            "shape": df.shape,
            "missing_ratio": df.isnull().sum() / len(df),
            "duplicate_count": df.index.duplicated().sum(),
            "date_range": {
                "start": df.index.get_level_values(0).min(),
                "end": df.index.get_level_values(0).max()
            },
            "symbols_count": df.index.get_level_values(1).nunique()
        }

        return report