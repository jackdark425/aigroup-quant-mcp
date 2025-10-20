"""
Alpha158 因子库 - 参考Qlib实现
包含158个技术指标因子的完整实现

因子分类：
1. KBAR (K线形态) - 9个因子
2. PRICE (价格) - 5个因子  
3. VOLUME (成交量) - 5个因子
4. ROLLING (滚动统计) - 139个因子
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class Alpha158Generator:
    """
    Alpha158因子生成器
    
    生成158个技术指标因子，包括：
    - KBAR: K线形态特征
    - PRICE: 价格特征  
    - VOLUME: 成交量特征
    - ROLLING: 滚动统计特征
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化Alpha158因子生成器
        
        Parameters:
        -----------
        data : pd.DataFrame
            MultiIndex DataFrame (datetime, symbol)
            必须包含列: open, high, low, close, volume, vwap(可选)
        """
        self.data = data.copy()
        self._validate_data()
        
    def _validate_data(self):
        """验证输入数据"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f"缺少必需列: {missing}")
            
    def generate_all(self, 
                     kbar: bool = True,
                     price: bool = True, 
                     volume: bool = True,
                     rolling: bool = True,
                     rolling_windows: List[int] = None) -> pd.DataFrame:
        """
        生成所有Alpha158因子
        
        Parameters:
        -----------
        kbar : bool
            是否生成K线形态因子
        price : bool
            是否生成价格因子
        volume : bool  
            是否生成成交量因子
        rolling : bool
            是否生成滚动统计因子
        rolling_windows : List[int]
            滚动窗口大小列表，默认[5, 10, 20, 30, 60]
            
        Returns:
        --------
        pd.DataFrame
            包含所有生成因子的DataFrame
        """
        if rolling_windows is None:
            rolling_windows = [5, 10, 20, 30, 60]
            
        result = pd.DataFrame(index=self.data.index)
        
        # 1. K线形态因子 (9个)
        if kbar:
            kbar_factors = self._generate_kbar_features()
            result = pd.concat([result, kbar_factors], axis=1)
            
        # 2. 价格因子 (5个)
        if price:
            price_factors = self._generate_price_features()
            result = pd.concat([result, price_factors], axis=1)
            
        # 3. 成交量因子 (5个)
        if volume:
            volume_factors = self._generate_volume_features()
            result = pd.concat([result, volume_factors], axis=1)
            
        # 4. 滚动统计因子 (最多139个，取决于窗口数量)
        if rolling:
            rolling_factors = self._generate_rolling_features(rolling_windows)
            result = pd.concat([result, rolling_factors], axis=1)
            
        return result
    
    def _generate_kbar_features(self) -> pd.DataFrame:
        """
        生成K线形态特征 (9个因子)
        
        Returns:
        --------
        pd.DataFrame
            K线形态因子
        """
        close = self.data['close']
        open_ = self.data['open']
        high = self.data['high']
        low = self.data['low']
        
        features = {}
        
        # KMID: (close - open) / open
        features['KMID'] = (close - open_) / open_
        
        # KLEN: (high - low) / open
        features['KLEN'] = (high - low) / open_
        
        # KMID2: (close - open) / (high - low + 1e-12)
        features['KMID2'] = (close - open_) / (high - low + 1e-12)
        
        # KUP: (high - max(open, close)) / open
        features['KUP'] = (high - np.maximum(open_, close)) / open_
        
        # KUP2: (high - max(open, close)) / (high - low + 1e-12)
        features['KUP2'] = (high - np.maximum(open_, close)) / (high - low + 1e-12)
        
        # KLOW: (min(open, close) - low) / open
        features['KLOW'] = (np.minimum(open_, close) - low) / open_
        
        # KLOW2: (min(open, close) - low) / (high - low + 1e-12)
        features['KLOW2'] = (np.minimum(open_, close) - low) / (high - low + 1e-12)
        
        # KSFT: (2*close - high - low) / open
        features['KSFT'] = (2 * close - high - low) / open_
        
        # KSFT2: (2*close - high - low) / (high - low + 1e-12)
        features['KSFT2'] = (2 * close - high - low) / (high - low + 1e-12)
        
        return pd.DataFrame(features, index=self.data.index)
    
    def _generate_price_features(self, windows: List[int] = [0]) -> pd.DataFrame:
        """
        生成价格特征
        
        Parameters:
        -----------
        windows : List[int]
            时间窗口，默认只用当前值[0]
            
        Returns:
        --------
        pd.DataFrame
            价格因子
        """
        close = self.data['close']
        features = {}
        
        price_fields = ['open', 'high', 'low', 'close']
        if 'vwap' in self.data.columns:
            price_fields.append('vwap')
            
        for field in price_fields:
            for d in windows:
                if d == 0:
                    name = f'{field.upper()}0'
                    features[name] = self.data[field] / close
                else:
                    name = f'{field.upper()}{d}'
                    features[name] = self.data[field].groupby(level=1).shift(d) / close
                    
        return pd.DataFrame(features, index=self.data.index)
    
    def _generate_volume_features(self, windows: List[int] = [0]) -> pd.DataFrame:
        """
        生成成交量特征
        
        Parameters:
        -----------
        windows : List[int]
            时间窗口，默认只用当前值[0]
            
        Returns:
        --------
        pd.DataFrame
            成交量因子
        """
        volume = self.data['volume']
        features = {}
        
        for d in windows:
            if d == 0:
                name = 'VOLUME0'
                features[name] = volume / (volume + 1e-12)
            else:
                name = f'VOLUME{d}'
                features[name] = volume.groupby(level=1).shift(d) / (volume + 1e-12)
                
        return pd.DataFrame(features, index=self.data.index)
    
    def _generate_rolling_features(self, windows: List[int]) -> pd.DataFrame:
        """
        生成滚动统计特征
        使用transform确保索引对齐
        
        Parameters:
        -----------
        windows : List[int]
            滚动窗口大小列表
            
        Returns:
        --------
        pd.DataFrame
            滚动统计因子
        """
        close = self.data['close']
        high = self.data['high']
        low = self.data['low']
        volume = self.data['volume']
        
        features = {}
        
        for d in windows:
            # ROC: 变化率
            features[f'ROC{d}'] = close.groupby(level=1).transform(
                lambda x: x.shift(d) / x
            )
            
            # MA: 移动平均
            ma_rolling = close.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).mean()
            )
            features[f'MA{d}'] = ma_rolling / close
            
            # STD: 标准差  
            std_rolling = close.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).std()
            )
            features[f'STD{d}'] = std_rolling / close
            
            # BETA: 线性回归斜率
            beta_rolling = close.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=2).apply(
                    lambda y: np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) >= 2 else np.nan,
                    raw=True
                )
            )
            features[f'BETA{d}'] = beta_rolling / close
            
            # RSQR: R方值
            features[f'RSQR{d}'] = close.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=2).apply(
                    self._calc_rsquare,
                    raw=True
                )
            )
            
            # RESI: 线性回归残差
            resi_rolling = close.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=2).apply(
                    self._calc_resi,
                    raw=True
                )
            )
            features[f'RESI{d}'] = resi_rolling / close
            
            # MAX: 最大值
            max_rolling = high.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).max()
            )
            features[f'MAX{d}'] = max_rolling / close
            
            # MIN: 最小值
            min_rolling = low.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).min()
            )
            features[f'MIN{d}'] = min_rolling / close
            
            # QTLU: 80%分位数
            qtlu_rolling = close.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).quantile(0.8)
            )
            features[f'QTLU{d}'] = qtlu_rolling / close
            
            # QTLD: 20%分位数
            qtld_rolling = close.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).quantile(0.2)
            )
            features[f'QTLD{d}'] = qtld_rolling / close
            
            # RANK: 百分位排名
            features[f'RANK{d}'] = close.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).apply(
                    lambda y: pd.Series(y).rank(pct=True).iloc[-1] if len(y) > 0 else np.nan,
                    raw=False
                )
            )
            
            # RSV: 相对强弱位置
            rsv_min = low.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).min()
            )
            rsv_max = high.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).max()
            )
            features[f'RSV{d}'] = (close - rsv_min) / (rsv_max - rsv_min + 1e-12)
            
            # IMAX/IMIN: 最大最小值索引
            features[f'IMAX{d}'] = high.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).apply(
                    lambda y: (np.argmax(y) + 1) / d if len(y) > 0 else np.nan,
                    raw=True
                )
            )
            
            features[f'IMIN{d}'] = low.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).apply(
                    lambda y: (np.argmin(y) + 1) / d if len(y) > 0 else np.nan,
                    raw=True
                )
            )
            
            # IMXD: 索引差
            features[f'IMXD{d}'] = (features[f'IMAX{d}'] * d - features[f'IMIN{d}'] * d) / d
            
            # CORR: 价格与成交量相关性
            log_vol = np.log(volume + 1)
            corr_result = []
            for symbol in close.index.get_level_values(1).unique():
                symbol_close = close.xs(symbol, level=1)
                symbol_vol = log_vol.xs(symbol, level=1)
                symbol_corr = symbol_close.rolling(d, min_periods=1).corr(symbol_vol)
                corr_result.append(pd.DataFrame({f'CORR{d}': symbol_corr}, 
                                               index=pd.MultiIndex.from_product([symbol_close.index, [symbol]],
                                                                              names=['datetime', 'symbol'])))
            features[f'CORR{d}'] = pd.concat(corr_result).sort_index()[f'CORR{d}']
            
            # CORD: 价格变化与成交量变化相关性  
            close_pct = close.groupby(level=1).pct_change()
            vol_pct = volume.groupby(level=1).pct_change()
            cord_result = []
            for symbol in close.index.get_level_values(1).unique():
                symbol_cpct = close_pct.xs(symbol, level=1)
                symbol_vpct = vol_pct.xs(symbol, level=1)
                symbol_cord = symbol_cpct.rolling(d, min_periods=1).corr(symbol_vpct)
                cord_result.append(pd.DataFrame({f'CORD{d}': symbol_cord},
                                               index=pd.MultiIndex.from_product([symbol_cpct.index, [symbol]],
                                                                              names=['datetime', 'symbol'])))
            features[f'CORD{d}'] = pd.concat(cord_result).sort_index()[f'CORD{d}']
            
            # CNTP: 上涨天数占比
            up = (close > close.groupby(level=1).shift(1)).astype(float)
            features[f'CNTP{d}'] = up.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).mean()
            )
            
            # CNTN: 下跌天数占比
            down = (close < close.groupby(level=1).shift(1)).astype(float)
            features[f'CNTN{d}'] = down.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).mean()
            )
            
            # CNTD: 涨跌天数差
            features[f'CNTD{d}'] = features[f'CNTP{d}'] - features[f'CNTN{d}']
            
            # SUMP/SUMN/SUMD: 涨跌幅统计
            gain = np.maximum(close - close.groupby(level=1).shift(1), 0)
            loss = np.maximum(close.groupby(level=1).shift(1) - close, 0)
            abs_change = np.abs(close - close.groupby(level=1).shift(1))
            
            sum_gain = gain.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).sum()
            )
            sum_loss = loss.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).sum()
            )
            sum_abs = abs_change.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).sum()
            )
            
            features[f'SUMP{d}'] = sum_gain / (sum_abs + 1e-12)
            features[f'SUMN{d}'] = sum_loss / (sum_abs + 1e-12)
            features[f'SUMD{d}'] = (sum_gain - sum_loss) / (sum_abs + 1e-12)
            
            # VMA/VSTD: 成交量统计
            vma_rolling = volume.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).mean()
            )
            features[f'VMA{d}'] = vma_rolling / (volume + 1e-12)
            
            vstd_rolling = volume.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).std()
            )
            features[f'VSTD{d}'] = vstd_rolling / (volume + 1e-12)
            
            # WVMA: 加权波动率
            weighted_vol = np.abs(close.groupby(level=1).pct_change()) * volume
            wvma_std = weighted_vol.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).std()
            )
            wvma_mean = weighted_vol.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).mean()
            )
            features[f'WVMA{d}'] = wvma_std / (wvma_mean + 1e-12)
            
            # VSUMP/VSUMN/VSUMD: 成交量涨跌统计
            vol_gain = np.maximum(volume - volume.groupby(level=1).shift(1), 0)
            vol_loss = np.maximum(volume.groupby(level=1).shift(1) - volume, 0)
            vol_abs = np.abs(volume - volume.groupby(level=1).shift(1))
            
            vsum_gain = vol_gain.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).sum()
            )
            vsum_loss = vol_loss.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).sum()
            )
            vsum_abs = vol_abs.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).sum()
            )
            
            features[f'VSUMP{d}'] = vsum_gain / (vsum_abs + 1e-12)
            features[f'VSUMN{d}'] = vsum_loss / (vsum_abs + 1e-12)
            features[f'VSUMD{d}'] = (vsum_gain - vsum_loss) / (vsum_abs + 1e-12)
            
        return pd.DataFrame(features, index=self.data.index)
    
    def _calc_rsquare(self, y):
        """计算R方值"""
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-12))
    
    def _calc_resi(self, y):
        """计算残差"""
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        return y[-1] - y_pred[-1]
    
    def _rolling_slope(self, series: pd.Series, window: int) -> pd.Series:
        """计算滚动线性回归斜率"""
        def slope(y):
            if len(y) < 2:
                return np.nan
            x = np.arange(len(y))
            return np.polyfit(x, y, 1)[0]
            
        return series.rolling(window, min_periods=2).apply(slope, raw=True)
    
    def _rolling_rsquare(self, series: pd.Series, window: int) -> pd.Series:
        """计算滚动R方值"""
        return series.rolling(window, min_periods=2).apply(self._calc_rsquare, raw=True)
    
    def _rolling_resi(self, series: pd.Series, window: int) -> pd.Series:
        """计算滚动线性回归残差"""
        return series.rolling(window, min_periods=2).apply(self._calc_resi, raw=True)


def get_alpha158_config(
    kbar: bool = True,
    price: bool = True,
    volume: bool = True,
    rolling: bool = True,
    rolling_windows: List[int] = None
) -> Dict:
    """
    获取Alpha158配置
    
    Parameters:
    -----------
    kbar : bool
        是否包含K线形态因子
    price : bool
        是否包含价格因子
    volume : bool
        是否包含成交量因子
    rolling : bool
        是否包含滚动统计因子
    rolling_windows : List[int]
        滚动窗口列表，默认[5, 10, 20, 30, 60]
        
    Returns:
    --------
    Dict
        配置字典
    """
    if rolling_windows is None:
        rolling_windows = [5, 10, 20, 30, 60]
        
    return {
        'kbar': kbar,
        'price': price,
        'volume': volume,
        'rolling': rolling,
        'rolling_windows': rolling_windows
    }