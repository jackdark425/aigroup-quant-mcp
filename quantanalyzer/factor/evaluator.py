"""
因子评估器
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from typing import Dict


class FactorEvaluator:
    """因子评估器"""
    
    def __init__(self, factor: pd.Series, returns: pd.Series):
        """
        初始化因子评估器
        
        Args:
            factor: 因子值Series
            returns: 收益率Series
        """
        self.factor = factor
        self.returns = returns
        
    def calculate_ic(self, method: str = "spearman") -> Dict:
        """
        计算IC指标
        
        Args:
            method: pearson or spearman
            
        Returns:
            IC指标字典
        """
        # 确保因子和收益率对齐
        aligned_factor, aligned_return = self.factor.align(self.returns, join='inner')
        
        ic_series = []
        dates = aligned_factor.index.get_level_values(0).unique()
        
        for date in dates:
            factor_slice = aligned_factor.xs(date, level=0)
            return_slice = aligned_return.xs(date, level=0)
            
            # 去除NaN
            mask = ~(factor_slice.isna() | return_slice.isna())
            factor_slice = factor_slice[mask]
            return_slice = return_slice[mask]
            
            if len(factor_slice) < 10:  # 最少样本数
                continue
            
            if method == "spearman":
                ic, _ = spearmanr(factor_slice, return_slice)
            else:
                ic, _ = pearsonr(factor_slice, return_slice)
            
            ic_series.append(ic)
        
        ic_array = np.array(ic_series)
        
        return {
            "ic_mean": np.mean(ic_array),
            "ic_std": np.std(ic_array),
            "icir": np.mean(ic_array) / np.std(ic_array) if np.std(ic_array) > 0 else 0,
            "ic_positive_ratio": np.sum(ic_array > 0) / len(ic_array),
            "ic_series": ic_series
        }
    
    def calculate_group_return(
        self,
        n_groups: int = 10
    ) -> pd.DataFrame:
        """
        分组收益分析
        
        Args:
            n_groups: 分组数量
            
        Returns:
            各分组收益率
        """
        aligned_factor, aligned_return = self.factor.align(self.returns, join='inner')
        
        # 按因子值分组
        factor_groups = aligned_factor.groupby(level=0).apply(
            lambda x: pd.qcut(x, n_groups, labels=False, duplicates='drop')
        )
        
        # 计算各组平均收益
        group_returns = []
        for group in range(n_groups):
            mask = (factor_groups == group)
            group_return = aligned_return[mask].groupby(level=0).mean()
            group_returns.append(group_return)
        
        return pd.DataFrame(group_returns).T
    
    def calculate_turnover(self) -> float:
        """计算换手率"""
        # 计算因子排名变化
        rank = self.factor.groupby(level=0).rank(pct=True)
        rank_change = rank.groupby(level=1).diff().abs()
        
        return rank_change.mean()