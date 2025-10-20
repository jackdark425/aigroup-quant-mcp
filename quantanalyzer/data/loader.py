"""
数据加载器
"""
import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional
from pathlib import Path


class DataLoader:
    """数据加载器"""
    
    def __init__(self):
        self.data_cache = {}
        
    def load_from_csv(
        self,
        file_path: Union[str, Path],
        symbol_col: str = "symbol",
        datetime_col: str = "datetime",
        **kwargs
    ) -> pd.DataFrame:
        """
        从CSV加载数据
        
        Args:
            file_path: CSV文件路径
            symbol_col: 股票代码列名
            datetime_col: 日期列名
            
        Returns:
            MultiIndex DataFrame (datetime, symbol)
        """
        df = pd.read_csv(file_path, **kwargs)
        
        # 转换日期
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        
        # 设置MultiIndex
        df = df.set_index([datetime_col, symbol_col])
        df = df.sort_index()
        
        return df
    
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