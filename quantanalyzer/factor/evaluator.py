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
        # 修复：使用更稳健的数据对齐方法
        # 确保因子和收益率对齐，使用inner join避免NaN问题
        aligned_factor, aligned_return = self.factor.align(self.returns, join='inner')
        
        # 进一步去除NaN值
        mask = ~(aligned_factor.isna() | aligned_return.isna())
        aligned_factor = aligned_factor[mask]
        aligned_return = aligned_return[mask]
        
        if len(aligned_factor) == 0:
            return {
                "ic_mean": 0,
                "ic_std": 0,
                "icir": 0,
                "ic_positive_ratio": 0,
                "ic_series": [],
                "valid_dates": 0,
                "total_dates": 0,
                "data_quality": "poor",
                "error_message": "因子和收益率数据没有共同的有效样本"
            }

        ic_series = []
        dates = aligned_factor.index.get_level_values(0).unique()
        valid_dates = 0
        error_details = []

        for date in dates:
            try:
                factor_slice = aligned_factor.xs(date, level=0)
                return_slice = aligned_return.xs(date, level=0)
                
                # 去除NaN
                mask = ~(factor_slice.isna() | return_slice.isna())
                factor_slice = factor_slice[mask]
                return_slice = return_slice[mask]
                
                if len(factor_slice) < 3:  # 进一步降低最少样本数要求
                    error_details.append(f"日期 {date}: 样本数不足 ({len(factor_slice)} < 3)")
                    continue
                
                # 检查数据变异性，避免计算无效的相关系数
                if factor_slice.std() == 0:
                    error_details.append(f"日期 {date}: 因子标准差为0")
                    continue
                if return_slice.std() == 0:
                    error_details.append(f"日期 {date}: 收益率标准差为0")
                    continue
                
                if method == "spearman":
                    ic, _ = spearmanr(factor_slice, return_slice)
                else:
                    ic, _ = pearsonr(factor_slice, return_slice)
                
                # 检查相关系数是否有效
                if not np.isnan(ic) and not np.isinf(ic):
                    ic_series.append(ic)
                    valid_dates += 1
                else:
                    error_details.append(f"日期 {date}: 相关系数无效 (NaN或Inf)")
                    
            except (ValueError, TypeError) as e:
                # 跳过计算失败的日期
                error_details.append(f"日期 {date}: 计算错误 - {str(e)}")
                continue
        
        if len(ic_series) == 0:
            return {
                "ic_mean": 0,
                "ic_std": 0,
                "icir": 0,
                "ic_positive_ratio": 0,
                "ic_series": [],
                "valid_dates": 0,
                "total_dates": len(dates),
                "data_quality": "poor",
                "error_details": error_details[:10],  # 只显示前10个错误详情
                "suggestions": [
                    "检查因子计算是否正确",
                    "确认数据时间范围匹配",
                    "尝试使用不同的因子类型",
                    "确保数据量足够（建议100+条）"
                ]
            }

        ic_array = np.array(ic_series)
        
        # 评估数据质量
        if valid_dates / len(dates) > 0.8:
            data_quality = "excellent"
        elif valid_dates / len(dates) > 0.5:
            data_quality = "good"
        elif valid_dates / len(dates) > 0.2:
            data_quality = "fair"
        else:
            data_quality = "poor"

        return {
            "ic_mean": np.mean(ic_array),
            "ic_std": np.std(ic_array),
            "icir": np.mean(ic_array) / np.std(ic_array) if np.std(ic_array) > 0 else 0,
            "ic_positive_ratio": np.sum(ic_array > 0) / len(ic_array),
            "ic_series": ic_series,
            "valid_dates": valid_dates,
            "total_dates": len(dates),
            "data_quality": data_quality,
            "coverage_rate": valid_dates / len(dates),
            "error_details": error_details[:5] if error_details else None  # 只显示前5个错误详情
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