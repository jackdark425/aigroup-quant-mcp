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
        
        # 修复：改进动量因子计算，使其产生更有意义的信号
        # 计算过去period天的累计收益率作为动量
        momentum = close.groupby(level=1).apply(
            lambda x: (x / x.shift(period) - 1).fillna(0)
        )
        
        # 确保索引正确
        if isinstance(momentum, pd.Series):
            momentum = momentum.droplevel(0)
        else:
            momentum = momentum.reset_index(level=0, drop=True)
        
        return momentum
    
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
        close = data['close']
        returns = close.groupby(level=1).pct_change()
        volatility = returns.groupby(level=1).rolling(period).std().droplevel(0)
        
        # 确保波动率因子与原始数据索引对齐
        volatility = volatility.reindex(close.index)
        
        return volatility
    
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
        volume_ratio = volume / ma_volume
        
        # 确保成交量比率与原始数据索引对齐
        volume_ratio = volume_ratio.reindex(volume.index)
        
        return volume_ratio
    
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
        
        # 确保RSI与原始数据索引对齐
        rsi = rsi.reindex(close.index)
        
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
    
    def calculate_factor(self, factor_name: str, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        计算指定因子
        
        Args:
            factor_name: 因子名称 ('sma', 'returns', 'volatility', 'momentum', 'volume_ratio', 'rsi', 'macd', 'bollinger_bands')
            data: 输入数据DataFrame
            **kwargs: 因子参数
            
        Returns:
            包含因子值的DataFrame
        """
        factor_name = factor_name.lower()
        
        if factor_name == 'sma':
            window = kwargs.get('window', 20)
            close = data['close']
            sma = close.groupby(level=1).rolling(window).mean().droplevel(0)
            return pd.DataFrame({f'sma_{window}': sma})
            
        elif factor_name == 'returns':
            close = data['close']
            returns = close.groupby(level=1).pct_change()
            return pd.DataFrame({'returns': returns})
            
        elif factor_name == 'volatility':
            window = kwargs.get('window', 20)
            close = data['close']
            returns = close.groupby(level=1).pct_change()
            volatility = returns.groupby(level=1).rolling(window).std().droplevel(0)
            return pd.DataFrame({f'volatility_{window}': volatility})
            
        elif factor_name == 'momentum':
            period = kwargs.get('period', 20)
            return pd.DataFrame({'momentum': self.momentum(data, period)})
            
        elif factor_name == 'volume_ratio':
            period = kwargs.get('period', 20)
            return pd.DataFrame({'volume_ratio': self.volume_ratio(data, period)})
            
        elif factor_name == 'rsi':
            period = kwargs.get('period', 14)
            return pd.DataFrame({'rsi': self.rsi(data, period)})
            
        elif factor_name == 'macd':
            fast = kwargs.get('fast', 12)
            slow = kwargs.get('slow', 26)
            signal = kwargs.get('signal', 9)
            return self.macd(data, fast, slow, signal)
            
        elif factor_name == 'bollinger_bands':
            period = kwargs.get('period', 20)
            num_std = kwargs.get('num_std', 2.0)
            return self.bollinger_bands(data, period, num_std)
            
        else:
            raise ValueError(f"不支持的因子类型: {factor_name}")