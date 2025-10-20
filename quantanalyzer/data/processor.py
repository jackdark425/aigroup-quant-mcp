"""
数据处理器(Processor)系统 - 参考Qlib设计
提供标准化的数据预处理流程，支持Learn/Infer分离避免数据泄露
"""

import abc
import numpy as np
import pandas as pd
from typing import Union, List, Optional


EPS = 1e-12  # 防止除零


class Processor(abc.ABC):
    """
    基础Processor抽象类
    
    核心方法：
    - fit(): 在训练集上学习参数（Learn阶段）
    - __call__() / transform(): 应用转换（Infer阶段）
    """
    
    def fit(self, df: pd.DataFrame):
        """
        在训练集上学习处理参数
        
        Parameters
        ----------
        df : pd.DataFrame
            训练数据，用于学习统计参数
            
        Returns
        -------
        self
            返回自身以支持链式调用
        """
        # 默认不需要学习参数
        return self
    
    @abc.abstractmethod
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        应用数据转换
        
        Parameters
        ----------
        df : pd.DataFrame
            需要处理的数据
            
        Returns
        -------
        pd.DataFrame
            处理后的数据
            
        NOTE
        ----
        Processor可能会**就地修改**df的内容！
        用户应该在外部保留数据副本
        """
        pass
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        transform的别名，兼容sklearn风格
        """
        return self(df)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        fit和transform的组合操作
        """
        self.fit(df)
        return self.transform(df)
    
    def is_for_infer(self) -> bool:
        """
        是否可用于推理阶段
        
        某些Processor（如DropnaLabel）只用于训练，不能用于推理
        """
        return True
    
    def readonly(self) -> bool:
        """
        是否为只读操作（不修改输入数据）
        
        了解只读信息有助于Handler避免不必要的拷贝
        """
        return False


# ========== 缺失值处理 Processors ==========

class DropnaLabel(Processor):
    """
    删除标签为空的样本
    
    使用场景：监督学习必须，删除无法计算loss的样本
    
    Parameters
    ----------
    label_col : str
        标签列名，默认'label'
    
    Examples
    --------
    >>> proc = DropnaLabel(label_col='return')
    >>> clean_data = proc(data)  # 删除return列为NaN的行
    """
    
    def __init__(self, label_col: str = 'label'):
        self.label_col = label_col
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.label_col in df.columns:
            return df.dropna(subset=[self.label_col])
        else:
            # 如果列不存在，返回原数据
            return df
    
    def is_for_infer(self) -> bool:
        """只用于训练，推理时没有标签"""
        return False
    
    def readonly(self) -> bool:
        return True


class Fillna(Processor):
    """
    填充缺失值
    
    Parameters
    ----------
    fields : list or None
        要填充的列名列表，None表示所有列
    fill_value : float
        填充值，默认0
        
    Examples
    --------
    >>> proc = Fillna(fields=['factor1', 'factor2'], fill_value=0)
    >>> filled_data = proc(data)
    """
    
    def __init__(self, fields: Optional[List[str]] = None, fill_value: float = 0):
        self.fields = fields
        self.fill_value = fill_value
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.fields is None:
            # 填充所有列
            df = df.fillna(self.fill_value)
        else:
            # 填充指定列
            for field in self.fields:
                if field in df.columns:
                    df[field] = df[field].fillna(self.fill_value)
        return df


# ========== 标准化 Processors ==========

class CSZScoreNorm(Processor):
    """
    截面Z-score标准化（Cross-Sectional ZScore Normalization）
    
    最常用的Processor！在每个时间截面上对所有股票进行标准化。
    
    公式: (x - mean) / std  (按日期分组计算)
    
    Parameters
    ----------
    fields : list or None
        要标准化的列名列表，None表示除datetime和symbol外的所有列
    method : str
        'zscore' - 普通Z-score
        'robust' - 鲁棒Z-score（使用中位数和MAD）
        
    Examples
    --------
    >>> proc = CSZScoreNorm(fields=['factor1', 'factor2'])
    >>> normalized_data = proc(data)
    
    Notes
    -----
    这是最重要的Processor，能消除不同股票间的量纲差异，
    让模型能够公平比较所有股票。
    """
    
    def __init__(self, fields: Optional[List[str]] = None, method: str = 'zscore'):
        self.fields = fields
        self.method = method
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        # 确定要处理的列
        if self.fields is None:
            # 排除datetime和symbol列
            cols = [col for col in df.columns 
                   if col not in ['datetime', 'symbol']]
        else:
            cols = self.fields
        
        # 检查是否有MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            # MultiIndex: (datetime, symbol)
            datetime_level = 0
            
            for col in cols:
                if col in df.columns:
                    if self.method == 'zscore':
                        # 普通Z-score
                        df[col] = df.groupby(level=datetime_level)[col].transform(
                            lambda x: (x - x.mean()) / (x.std() + EPS)
                        )
                    elif self.method == 'robust':
                        # 鲁棒Z-score
                        df[col] = df.groupby(level=datetime_level)[col].transform(
                            lambda x: (x - x.median()) / (x.mad() * 1.4826 + EPS)
                        )
        else:
            # 单层Index，假设有datetime列
            if 'datetime' in df.columns:
                for col in cols:
                    if col in df.columns:
                        if self.method == 'zscore':
                            df[col] = df.groupby('datetime')[col].transform(
                                lambda x: (x - x.mean()) / (x.std() + EPS)
                            )
                        elif self.method == 'robust':
                            df[col] = df.groupby('datetime')[col].transform(
                                lambda x: (x - x.median()) / (x.mad() * 1.4826 + EPS)
                            )
        
        return df


class ZScoreNorm(Processor):
    """
    时间序列Z-score标准化
    
    在整个时间序列上进行标准化（需要先fit学习参数）
    
    Parameters
    ----------
    fields : list or None
        要标准化的列名列表
        
    Examples
    --------
    >>> proc = ZScoreNorm(fields=['factor1'])
    >>> proc.fit(train_data)  # 学习训练集的均值和标准差
    >>> norm_train = proc(train_data)
    >>> norm_test = proc(test_data)  # 使用相同的均值和标准差
    
    Notes
    -----
    必须先fit再transform，避免数据泄露
    """
    
    def __init__(self, fields: Optional[List[str]] = None):
        self.fields = fields
        self.mean_ = None
        self.std_ = None
    
    def fit(self, df: pd.DataFrame):
        """学习均值和标准差"""
        if self.fields is None:
            cols = [col for col in df.columns 
                   if col not in ['datetime', 'symbol']]
        else:
            cols = self.fields
        
        # 计算每列的均值和标准差
        self.mean_ = {}
        self.std_ = {}
        for col in cols:
            if col in df.columns:
                self.mean_[col] = df[col].mean()
                self.std_[col] = df[col].std()
                # 避免标准差为0
                if self.std_[col] == 0:
                    self.std_[col] = 1.0
        
        return self
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.mean_ is None:
            raise ValueError("Must call fit() before transform()")
        
        for col in self.mean_.keys():
            if col in df.columns:
                df[col] = (df[col] - self.mean_[col]) / (self.std_[col] + EPS)
        
        return df


class RobustZScoreNorm(Processor):
    """
    鲁棒Z-score标准化
    
    使用中位数和MAD (Median Absolute Deviation) 代替均值和标准差，
    对异常值更鲁棒。
    
    公式: (x - median) / (MAD * 1.4826)
    
    Parameters
    ----------
    fields : list or None
        要标准化的列名列表
    clip_outlier : bool
        是否裁剪异常值到[-3, 3]范围
        
    Examples
    --------
    >>> proc = RobustZScoreNorm(fields=['factor1'], clip_outlier=True)
    >>> proc.fit(train_data)
    >>> norm_data = proc(test_data)
    
    Notes
    -----
    适合有异常值的数据，比普通Z-score更稳定
    """
    
    def __init__(self, fields: Optional[List[str]] = None, clip_outlier: bool = True):
        self.fields = fields
        self.clip_outlier = clip_outlier
        self.median_ = None
        self.mad_ = None
    
    def fit(self, df: pd.DataFrame):
        """学习中位数和MAD"""
        if self.fields is None:
            cols = [col for col in df.columns 
                   if col not in ['datetime', 'symbol']]
        else:
            cols = self.fields
        
        self.median_ = {}
        self.mad_ = {}
        for col in cols:
            if col in df.columns:
                values = df[col].values
                self.median_[col] = np.nanmedian(values)
                self.mad_[col] = np.nanmedian(
                    np.abs(values - self.median_[col])
                ) * 1.4826 + EPS
        
        return self
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.median_ is None:
            raise ValueError("Must call fit() before transform()")
        
        for col in self.median_.keys():
            if col in df.columns:
                df[col] = (df[col] - self.median_[col]) / self.mad_[col]
                
                if self.clip_outlier:
                    df[col] = np.clip(df[col], -3, 3)
        
        return df


class MinMaxNorm(Processor):
    """
    最小最大归一化
    
    将数据缩放到[0, 1]区间
    
    公式: (x - min) / (max - min)
    
    Parameters
    ----------
    fields : list or None
        要归一化的列名列表
        
    Examples
    --------
    >>> proc = MinMaxNorm(fields=['price', 'volume'])
    >>> proc.fit(train_data)
    >>> norm_data = proc(test_data)
    """
    
    def __init__(self, fields: Optional[List[str]] = None):
        self.fields = fields
        self.min_ = None
        self.max_ = None
    
    def fit(self, df: pd.DataFrame):
        """学习最小值和最大值"""
        if self.fields is None:
            cols = [col for col in df.columns 
                   if col not in ['datetime', 'symbol']]
        else:
            cols = self.fields
        
        self.min_ = {}
        self.max_ = {}
        for col in cols:
            if col in df.columns:
                self.min_[col] = df[col].min()
                self.max_[col] = df[col].max()
                # 避免最大值等于最小值
                if self.max_[col] == self.min_[col]:
                    self.max_[col] = self.min_[col] + 1.0
        
        return self
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.min_ is None:
            raise ValueError("Must call fit() before transform()")
        
        for col in self.min_.keys():
            if col in df.columns:
                df[col] = (df[col] - self.min_[col]) / (self.max_[col] - self.min_[col] + EPS)
        
        return df


class CSRankNorm(Processor):
    """
    截面排名标准化
    
    在每个时间截面上将数据转换为百分位排名，然后标准化。
    
    公式: (rank_pct - 0.5) * 3.46
    
    Parameters
    ----------
    fields : list or None
        要处理的列名列表
        
    Examples
    --------
    >>> proc = CSRankNorm(fields=['factor1'])
    >>> ranked_data = proc(data)
    
    Notes
    -----
    对异常值不敏感，因为使用排名而非原始值
    """
    
    def __init__(self, fields: Optional[List[str]] = None):
        self.fields = fields
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.fields is None:
            cols = [col for col in df.columns 
                   if col not in ['datetime', 'symbol']]
        else:
            cols = self.fields
        
        # 检查是否有MultiIndex
        if isinstance(df.index, pd.MultiIndex):
            datetime_level = 0
            for col in cols:
                if col in df.columns:
                    # 按日期分组排名
                    df[col] = df.groupby(level=datetime_level)[col].transform(
                        lambda x: (x.rank(pct=True) - 0.5) * 3.46
                    )
        else:
            if 'datetime' in df.columns:
                for col in cols:
                    if col in df.columns:
                        df[col] = df.groupby('datetime')[col].transform(
                            lambda x: (x.rank(pct=True) - 0.5) * 3.46
                        )
        
        return df


# ========== Processor链工具 ==========

class ProcessorChain:
    """
    Processor链，方便组合多个Processor
    
    Examples
    --------
    >>> chain = ProcessorChain([
    ...     DropnaLabel(),
    ...     CSZScoreNorm(),
    ...     Fillna(fill_value=0)
    ... ])
    >>> chain.fit(train_data)
    >>> processed_train = chain.transform(train_data)
    >>> processed_test = chain.transform(test_data)
    """
    
    def __init__(self, processors: List[Processor]):
        self.processors = processors
    
    def fit(self, df: pd.DataFrame):
        """按顺序fit所有processor"""
        for proc in self.processors:
            proc.fit(df)
            df = proc(df)  # fit后的数据传给下一个
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """按顺序应用所有processor"""
        for proc in self.processors:
            df = proc(df)
        return df
    
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """fit并transform"""
        self.fit(df)
        return self.transform(df)


# 导出常用Processor
__all__ = [
    'Processor',
    'DropnaLabel',
    'Fillna',
    'CSZScoreNorm',
    'ZScoreNorm',
    'RobustZScoreNorm',
    'MinMaxNorm',
    'CSRankNorm',
    'ProcessorChain',
]