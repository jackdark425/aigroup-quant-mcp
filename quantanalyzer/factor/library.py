"""
内置因子库
"""
import pandas as pd
import numpy as np


class FactorLibrary:
    """内置因子库"""
    
    @staticmethod
    def momentum(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        动量因子
        
        Args:
            data: 包含close列的DataFrame
            period: 回望期
            
        Returns:
            动量因子值
        """
        close = data['close']
        return close / close.groupby(level=1).shift(period) - 1
    
    @staticmethod
    def volatility(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        波动率因子
        
        Args:
            data: 包含close列的DataFrame
            period: 计算周期
            
        Returns:
            波动率因子值
        """
        returns = data['close'].pct_change()
        return returns.groupby(level=1).rolling(period).std().droplevel(0)
    
    @staticmethod
    def volume_ratio(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        成交量比率
        
        Args:
            data: 包含volume列的DataFrame
            period: 计算周期
            
        Returns:
            成交量比率
        """
        volume = data['volume']
        ma_volume = volume.groupby(level=1).rolling(period).mean().droplevel(0)
        return volume / ma_volume
    
    @staticmethod
    def rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        RSI指标
        
        Args:
            data: 包含close列的DataFrame
            period: 计算周期
            
        Returns:
            RSI值
        """
        close = data['close']
        delta = close.groupby(level=1).diff()
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.groupby(level=1).rolling(period).mean().droplevel(0)
        avg_loss = loss.groupby(level=1).rolling(period).mean().droplevel(0)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(
        data: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        MACD指标
        
        Args:
            data: 包含close列的DataFrame
            fast: 快速EMA周期
            slow: 慢速EMA周期
            signal: 信号线周期
            
        Returns:
            包含macd, signal, histogram三列的DataFrame
        """
        close = data['close']
        
        exp1 = close.groupby(level=1).ewm(span=fast, adjust=False).mean()
        exp2 = close.groupby(level=1).ewm(span=slow, adjust=False).mean()
        
        macd_line = exp1 - exp2
        signal_line = macd_line.groupby(level=1).ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    @staticmethod
    def bollinger_bands(
        data: pd.DataFrame,
        period: int = 20,
        num_std: float = 2.0
    ) -> pd.DataFrame:
        """
        布林带指标
        
        Args:
            data: 包含close列的DataFrame
            period: 计算周期
            num_std: 标准差倍数
            
        Returns:
            包含upper, middle, lower三列的DataFrame
        """
        close = data['close']
        
        middle = close.groupby(level=1).rolling(period).mean().droplevel(0)
        std = close.groupby(level=1).rolling(period).std().droplevel(0)
        
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        
        return pd.DataFrame({
            'upper': upper,
            'middle': middle,
            'lower': lower
        })