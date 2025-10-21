"""
回测引擎

优化内容：
1. 预先缓存MultiIndex数据切片，避免重复索引查找
2. 使用字典缓存提升数据访问速度
3. 向量化操作替代循环访问
"""
import pandas as pd
import numpy as np
from typing import Dict, List


class BacktestEngine:
    """回测引擎"""
    
    def __init__(
        self,
        initial_capital: float = 10000000,
        commission: float = 0.0003,
        slippage: float = 0.0001
    ):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission: 手续费率
            slippage: 滑点
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        self.positions = {}
        self.trades = []
        self.portfolio_value = []
    
    def run_topk_strategy(
        self,
        predictions: pd.Series,
        prices: pd.DataFrame,
        k: int = 50,
        holding_period: int = 1
    ) -> Dict:
        """
        TopK策略回测
        
        Args:
            predictions: 模型预测值
            prices: 价格数据
            k: 选择top k只股票
            holding_period: 持仓周期（天）
            
        Returns:
            回测结果
        """
        dates = predictions.index.get_level_values(0).unique()
        
        # 优化：预先构建数据访问缓存
        pred_cache = self._build_prediction_cache(predictions)
        price_cache = self._build_price_cache(prices)
        
        current_capital = self.initial_capital
        current_positions = {}
        
        portfolio_values = []
        returns_series = []
        
        for i, date in enumerate(dates[:-holding_period]):
            # 使用缓存获取当期预测
            pred_dict = pred_cache.get(date, {})
            
            if not pred_dict:
                continue
            
            # 选择TopK
            topk_items = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)[:k]
            topk_stocks = [item[0] for item in topk_items]
            
            # 平仓
            if current_positions:
                date_prices = price_cache.get(date, {})
                for symbol, shares in current_positions.items():
                    sell_price = date_prices.get(symbol)
                    if sell_price is not None:
                        # 考虑滑点和手续费
                        sell_price = sell_price * (1 - self.slippage)
                        sell_value = shares * sell_price
                        commission_fee = sell_value * self.commission
                        current_capital += (sell_value - commission_fee)
                
                current_positions = {}
            
            # 开仓
            position_value = current_capital / k
            date_prices = price_cache.get(date, {})
            
            for symbol in topk_stocks:
                buy_price = date_prices.get(symbol)
                if buy_price is not None:
                    # 考虑滑点和手续费
                    buy_price = buy_price * (1 + self.slippage)
                    shares = int(position_value / buy_price)
                    
                    if shares > 0:
                        cost = shares * buy_price
                        commission_fee = cost * self.commission
                        current_capital -= (cost + commission_fee)
                        current_positions[symbol] = shares
            
            # 计算持仓市值
            next_date = dates[i + holding_period]
            next_prices = price_cache.get(next_date, {})
            position_market_value = 0
            
            for symbol, shares in current_positions.items():
                price = next_prices.get(symbol)
                if price is not None:
                    position_market_value += shares * price
            
            total_value = current_capital + position_market_value
            portfolio_values.append({
                'date': next_date,
                'value': total_value
            })
            
            if i > 0:
                ret = (total_value - portfolio_values[-2]['value']) / portfolio_values[-2]['value']
                returns_series.append(ret)
        
        # 计算性能指标
        returns_array = np.array(returns_series)
        
        metrics = {
            "total_return": (portfolio_values[-1]['value'] - self.initial_capital) / self.initial_capital,
            "annualized_return": self._annualize_return(returns_array),
            "sharpe_ratio": self._calculate_sharpe(returns_array),
            "max_drawdown": self._calculate_max_drawdown(portfolio_values),
            "volatility": np.std(returns_array) * np.sqrt(252),
            "portfolio_values": portfolio_values,
            "returns": returns_series
        }
        
        return metrics
    
    def _build_prediction_cache(self, predictions: pd.Series) -> Dict[pd.Timestamp, Dict]:
        """
        构建预测值缓存
        
        Args:
            predictions: 预测序列（MultiIndex: datetime, symbol）
            
        Returns:
            {date: {symbol: prediction}}
        """
        cache = {}
        for (date, symbol), value in predictions.items():
            if date not in cache:
                cache[date] = {}
            cache[date][symbol] = value
        return cache
    
    def _build_price_cache(self, prices: pd.DataFrame) -> Dict[pd.Timestamp, Dict]:
        """
        构建价格缓存
        
        Args:
            prices: 价格DataFrame（MultiIndex: datetime, symbol）
            
        Returns:
            {date: {symbol: close_price}}
        """
        cache = {}
        if 'close' in prices.columns:
            for (date, symbol), row in prices.iterrows():
                if date not in cache:
                    cache[date] = {}
                cache[date][symbol] = row['close']
        return cache
    
    def _annualize_return(self, returns: np.ndarray) -> float:
        """年化收益率"""
        if len(returns) == 0:
            return 0.0
        cum_return = (1 + returns).prod() - 1
        n_periods = len(returns)
        return (1 + cum_return) ** (252 / n_periods) - 1
    
    def _calculate_sharpe(self, returns: np.ndarray, rf: float = 0.03) -> float:
        """夏普比率"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        excess_return = returns.mean() * 252 - rf
        return excess_return / (returns.std() * np.sqrt(252))
    
    def _calculate_max_drawdown(self, portfolio_values: List[Dict]) -> float:
        """最大回撤"""
        if not portfolio_values:
            return 0.0
            
        values = [pv['value'] for pv in portfolio_values]
        peak = values[0]
        max_dd = 0
        
        for value in values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd