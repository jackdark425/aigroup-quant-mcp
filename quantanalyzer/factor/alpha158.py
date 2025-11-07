"""
Alpha158 因子库 - 参考Qlib实现
包含158个技术指标因子的完整实现

因子分类：
1. KBAR (K线形态) - 9个因子
2. PRICE (价格) - 5个因子  
3. VOLUME (成交量) - 5个因子
4. ROLLING (滚动统计) - 139个因子

优化内容：
1. 移除不必要的DataFrame复制，直接使用引用
2. 优化滚动计算，使用向量化操作替代循环
3. 分块处理大数据集，及时释放内存
4. 优化CORR/CORD计算，避免多次concat
5. 使用并行处理加速计算
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import gc
import logging
import time

from ..logger import get_logger
from ..config import get_config
from ..utils import parallelize_dataframe_operation, get_progress_bar


class Alpha158Generator:
    """
    Alpha158因子生成器
    
    生成158个技术指标因子，包括：
    - KBAR: K线形态特征
    - PRICE: 价格特征  
    - VOLUME: 成交量特征
    - ROLLING: 滚动统计特征
    """
    
    def __init__(self, data: pd.DataFrame, copy_data: bool = False):
        """
        初始化Alpha158因子生成器
        
        Parameters:
        -----------
        data : pd.DataFrame
            MultiIndex DataFrame (datetime, symbol)
            必须包含列: open, high, low, close, volume, vwap(可选)
        copy_data : bool
            是否复制数据，默认False节省内存
        """
        self.logger = get_logger(__name__)
        self.config = get_config()
        # 优化：避免不必要的复制，节省内存
        self.data = data.copy() if copy_data else data
        self._validate_data()
        self.logger.info(f"Alpha158Generator initialized with data shape: {self.data.shape}")
        
    def _validate_data(self):
        """验证输入数据"""
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in self.data.columns]
        if missing:
            raise ValueError(f"缺少必需列: {missing}")
        self.logger.debug("Data validation passed")
            
    def generate_all(self, 
                     kbar: bool = True,
                     price: bool = True, 
                     volume: bool = True,
                     rolling: bool = True,
                     rolling_windows: List[int] = None,
                     chunk_size: Optional[int] = None,
                     parallel: bool = True,
                     show_progress: bool = False) -> pd.DataFrame:
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
        chunk_size : int, optional
            分块处理大小，用于大数据集优化
        parallel : bool
            是否使用并行处理
            
        Returns:
        --------
        pd.DataFrame
            包含所有生成因子的DataFrame
        """
        if rolling_windows is None:
            rolling_windows = self.config.get('alpha158_rolling_windows', [5, 10, 20, 30, 60])
        
        chunk_size = chunk_size or self.config.get('chunk_size', 10000)
        
        self.logger.info(f"Generating Alpha158 factors with parameters: "
                        f"kbar={kbar}, price={price}, volume={volume}, rolling={rolling}")
        self.logger.info(f"Rolling windows: {rolling_windows}")
        
        # 如果指定分块大小，使用分块处理
        if chunk_size is not None and len(self.data) > chunk_size:
            self.logger.info(f"Using chunked processing with chunk_size={chunk_size}")
            return self._generate_all_chunked(
                kbar, price, volume, rolling, rolling_windows, chunk_size
            )
            
        # 使用列表收集因子，最后一次性合并
        factor_list = []
        
        # 1. K线形态因子 (9个)
        if kbar:
            self.logger.debug("Generating KBAR factors")
            kbar_factors = self._generate_kbar_factors()
            factor_list.append(kbar_factors)
        
        # 2. 价格因子 (5个)
        if price:
            self.logger.debug("Generating PRICE factors")
            price_factors = self._generate_price_factors()
            factor_list.append(price_factors)
            
        # 3. 成交量因子 (5个)
        if volume:
            self.logger.debug("Generating VOLUME factors")
            volume_factors = self._generate_volume_factors()
            factor_list.append(volume_factors)
            
        # 4. 滚动统计因子 (139个)
        if rolling:
            self.logger.debug("Generating ROLLING factors")
            if parallel:
                # 使用并行处理生成滚动因子
                rolling_factors = self._generate_rolling_factors_parallel(
                    rolling_windows, 
                    chunk_size=chunk_size if len(self.data) > chunk_size else None,
                    show_progress=show_progress
                )
            else:
                rolling_factors = self._generate_rolling_factors(rolling_windows)
            factor_list.append(rolling_factors)
            
        # 合并所有因子
        self.logger.debug(f"Concatenating {len(factor_list)} factor groups")
        if factor_list:
            # 确保所有因子具有相同的索引类型和顺序
            for i, factor_df in enumerate(factor_list):
                if not isinstance(factor_df.index, type(self.data.index)):
                    # 重新索引以确保类型一致
                    factor_list[i] = factor_df.reindex(self.data.index)
                elif not factor_df.index.equals(self.data.index):
                    # 如果索引不匹配，则重新索引
                    factor_list[i] = factor_df.reindex(self.data.index)
            
            factors = pd.concat(factor_list, axis=1)
            # 最终确保返回的DataFrame索引与输入数据一致
            factors = factors.reindex(self.data.index)
            self.logger.info(f"Generated {factors.shape[1]} Alpha158 factors with shape {factors.shape}")
            return factors
        else:
            self.logger.warning("No factors generated")
            return pd.DataFrame()

    def _generate_rolling_factors_parallel(self, windows: List[int], chunk_size=None, show_progress=False) -> pd.DataFrame:
        """
        使用并行处理生成滚动统计因子
        
        Parameters:
        -----------
        windows : List[int]
            滚动窗口大小列表
        chunk_size : int, optional
            分块大小
        show_progress : bool
            是否显示进度条
            
        Returns:
        --------
        pd.DataFrame
            滚动统计因子
        """
        self.logger.debug("Generating rolling factors with parallel processing")
        
        # 定义在每个股票组上计算因子的函数
        def compute_rolling_factors_for_group(group_data):
            # 获取原始索引信息
            original_index = group_data.index
            
            # 提取所需的价格和成交量数据
            open_prices = group_data['open']
            high_prices = group_data['high']
            low_prices = group_data['low']
            close_prices = group_data['close']
            volume = group_data['volume']
            
            # 计算对数成交量和收益率
            log_vol = np.log(volume + 1)
            close_pct = close_prices.pct_change()
            volume_pct = volume.pct_change()
            
            # 存储因子的字典
            features = {}
            
            # 对每个窗口大小计算因子
            for d in windows:
                # 价格比率
                features[f'OP{d}'] = open_prices / close_prices
                features[f'HP{d}'] = high_prices / close_prices
                features[f'LP{d}'] = low_prices / close_prices
                
                # 均值和标准差
                ma = close_prices.rolling(window=d, min_periods=1).mean()
                std = close_prices.rolling(window=d, min_periods=1).std()
                
                features[f'MA{d}'] = ma / close_prices
                features[f'STD{d}'] = std / close_prices
                
                # 变化率
                features[f'ROC{d}'] = close_prices.pct_change(d)
                
                # BOLL
                features[f'BOLL{d}'] = (close_prices - ma) / (std + 1e-12)
                
                # RSI
                delta = close_prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=d, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=d, min_periods=1).mean()
                rs = gain / (loss + 1e-12)
                features[f'RSI{d}'] = 100 - (100 / (1 + rs))
                
                # MACD (仅对较大窗口)
                if d >= 20:
                    ema12 = close_prices.ewm(span=12, min_periods=1).mean()
                    ema26 = close_prices.ewm(span=26, min_periods=1).mean()
                    macd = ema12 - ema26
                    signal = macd.ewm(span=9, min_periods=1).mean()
                    features[f'MACD{d}'] = macd - signal
                
                # 最大值和最小值
                max_rolling = high_prices.rolling(window=d, min_periods=1).max()
                min_rolling = low_prices.rolling(window=d, min_periods=1).min()
                features[f'MAX{d}'] = max_rolling / close_prices
                features[f'MIN{d}'] = min_rolling / close_prices
                
                # 成交量相关
                vol_ma = volume.rolling(window=d, min_periods=1).mean()
                vol_std = volume.rolling(window=d, min_periods=1).std()
                features[f'VMA{d}'] = vol_ma / (volume + 1e-12)
                features[f'VSTD{d}'] = vol_std / (volume + 1e-12)
                
                # 相关性
                corr_window = min(d, len(close_prices))
                if corr_window > 1:
                    features[f'CORR{d}'] = close_prices.rolling(window=corr_window, min_periods=1).corr(log_vol)
                    features[f'CORD{d}'] = close_pct.rolling(window=corr_window, min_periods=1).corr(volume_pct)
            
            # 转换为DataFrame，并保持原始索引
            result = pd.DataFrame(features, index=original_index)
            return result
        
        # 使用并行处理
        max_workers = self.config.get('parallel_workers', 4)
        
        # 显示进度条（如果需要）
        if show_progress:
            progress_callback = get_progress_bar(
                len(self.data.index.get_level_values(1).unique()),
                "Generating Alpha158 factors"
            )
        else:
            progress_callback = None
        
        try:
            result = parallelize_dataframe_operation(
                self.data[['open', 'high', 'low', 'close', 'volume']],
                compute_rolling_factors_for_group,
                groupby_level=1,  # 按股票分组
                max_workers=max_workers,
                chunk_size=chunk_size
            )
            
            # 确保结果的索引与原始数据一致
            if not result.index.equals(self.data.index):
                result = result.reindex(self.data.index)
                
            # 更新进度
            if progress_callback:
                progress_callback(len(self.data.index.get_level_values(1).unique()))
                
            return result
        finally:
            if progress_callback:
                progress_callback.close()

    def _compute_price_volume_stats(self, open_prices, high_prices, low_prices, close_prices, 
                                   volume, log_vol, close_pct, volume_pct, window):
        """计算价格和成交量相关统计"""
        features = {}
        
        # 价格比率
        features[f'OP{window}'] = open_prices / close_prices
        features[f'HP{window}'] = high_prices / close_prices
        features[f'LP{window}'] = low_prices / close_prices
        
        # 峰度和偏度
        def rolling_kurt(x):
            return x.rolling(window, min_periods=1).apply(
                lambda y: pd.Series(y).kurt() if len(y) > 3 else np.nan,
                raw=True
            )
        
        def rolling_skew(x):
            return x.rolling(window, min_periods=1).apply(
                lambda y: pd.Series(y).skew() if len(y) > 2 else np.nan,
                raw=True
            )
        
        features[f'KURT{window}'] = rolling_kurt(close_prices)
        features[f'SKEW{window}'] = rolling_skew(close_prices)
        
        # 相关性
        features[f'CORR{window}'] = self._calc_rolling_corr(close_prices, log_vol, window)
        features[f'CORD{window}'] = self._calc_rolling_corr(close_pct, volume_pct, window)
        
        return features

    def _compute_technical_indicators(self, close_prices, high_prices, low_prices, volume, window):
        """计算技术指标"""
        features = {}
        
        # RSI
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        rs = gain / (loss + 1e-12)
        features[f'RSI{window}'] = 100 - (100 / (1 + rs))
        
        # BOLL
        ma = close_prices.rolling(window=window, min_periods=1).mean()
        std = close_prices.rolling(window=window, min_periods=1).std()
        features[f'BOLL{window}'] = (close_prices - ma) / (std + 1e-12)
        
        # MACD
        if window >= 20:  # 只有较大的窗口才计算MACD
            ema12 = close_prices.ewm(span=12, min_periods=1).mean()
            ema26 = close_prices.ewm(span=26, min_periods=1).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, min_periods=1).mean()
            features[f'MACD{window}'] = (macd - signal).rolling(window, min_periods=1).mean()
        
        # KDJ
        low_min = low_prices.rolling(window=window, min_periods=1).min()
        high_max = high_prices.rolling(window=window, min_periods=1).max()
        rsv = (close_prices - low_min) / (high_max - low_min + 1e-12) * 100
        features[f'K{window}'] = rsv.rolling(window=3, min_periods=1).mean()
        features[f'D{window}'] = features[f'K{window}'].rolling(window=3, min_periods=1).mean()
        features[f'J{window}'] = 3 * features[f'K{window}'] - 2 * features[f'D{window}']
        
        return features

    def _compute_statistical_measures(self, close_prices, volume, window):
        """计算统计指标"""
        features = {}
        
        # 累积收益和波动率
        returns = close_prices.pct_change()
        features[f'RET{window}'] = (close_prices / close_prices.shift(window) - 1).rolling(window, min_periods=1).mean()
        features[f'VRET{window}'] = returns.rolling(window, min_periods=1).std()
        
        # 成交量相关
        features[f'VMA{window}'] = volume.rolling(window, min_periods=1).mean()
        features[f'VSTD{window}'] = volume.rolling(window, min_periods=1).std()
        features[f'VSUMP{window}'] = (volume / features[f'VMA{window}']).fillna(1)
        
        # 价格动量
        features[f'ROC{window}'] = close_prices.pct_change(window)
        
        return features

    def _generate_all_chunked(
        self, 
        kbar: bool, 
        price: bool, 
        volume: bool, 
        rolling: bool,
        rolling_windows: List[int],
        chunk_size: int
    ) -> pd.DataFrame:
        """分块生成因子，用于大数据集"""
        start_time = time.time()
        symbols = self.data.index.get_level_values(1).unique()
        chunks = [symbols[i:i+chunk_size] for i in range(0, len(symbols), chunk_size)]
        
        self.logger.info(f"Starting chunked processing: {len(chunks)} chunks, chunk_size={chunk_size}")
        
        all_results = []
        total_symbols = 0
        
        for i, symbol_chunk in enumerate(chunks):
            chunk_start = time.time()
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}, symbols: {len(symbol_chunk)}")
            
            # 提取当前分块数据
            chunk_data = self.data[self.data.index.get_level_values(1).isin(symbol_chunk)]
            
            # 创建临时生成器
            temp_gen = Alpha158Generator(chunk_data, copy_data=False)
            
            # 生成因子
            chunk_result = temp_gen.generate_all(
                kbar=kbar, price=price, volume=volume, 
                rolling=rolling, rolling_windows=rolling_windows,
                chunk_size=None  # 避免递归分块
            )
            
            all_results.append(chunk_result)
            total_symbols += len(symbol_chunk)
            
            # 清理临时对象
            del temp_gen, chunk_data, chunk_result
            gc.collect()
            
            self.logger.debug(f"Chunk {i+1} processed in {time.time() - chunk_start:.2f} seconds")
        
        # 合并所有分块结果
        concat_start = time.time()
        result = pd.concat(all_results, axis=0).sort_index()
        self.logger.debug(f"All chunks concatenated in {time.time() - concat_start:.2f} seconds")
        
        del all_results
        gc.collect()
        
        self.logger.info(f"Chunked processing completed: {total_symbols} symbols, "
                        f"total time: {time.time() - start_time:.2f} seconds")
        
        return result
    
    def _generate_kbar_factors(self) -> pd.DataFrame:
        """
        生成K线形态因子 (9个因子)
        
        Returns:
        --------
        pd.DataFrame
            K线形态因子
        """
        close = self.data['close']
        open_ = self.data['open']
        high = self.data['high']
        low = self.data['low']
        
        # 预计算常用中间结果
        high_low_diff = high - low + 1e-12
        close_open_diff = close - open_
        max_open_close = np.maximum(open_, close)
        min_open_close = np.minimum(open_, close)
        
        features = {}
        
        # KMID: (close - open) / open
        features['KMID'] = close_open_diff / open_
        
        # KLEN: (high - low) / open
        features['KLEN'] = high_low_diff / open_
        
        # KMID2: (close - open) / (high - low + 1e-12)
        features['KMID2'] = close_open_diff / high_low_diff
        
        # KUP: (high - max(open, close)) / open
        features['KUP'] = (high - max_open_close) / open_
        
        # KUP2: (high - max(open, close)) / (high - low + 1e-12)
        features['KUP2'] = (high - max_open_close) / high_low_diff
        
        # KLOW: (min(open, close) - low) / open
        features['KLOW'] = (min_open_close - low) / open_
        
        # KLOW2: (min(open, close) - low) / (high - low + 1e-12)
        features['KLOW2'] = (min_open_close - low) / high_low_diff
        
        # KSFT: (2*close - high - low) / open
        features['KSFT'] = (2 * close - high - low) / open_
        
        # KSFT2: (2*close - high - low) / (high - low + 1e-12)
        features['KSFT2'] = (2 * close - high - low) / high_low_diff
        
        return pd.DataFrame(features, index=self.data.index)
    
    def _generate_price_factors(self, windows: List[int] = [0]) -> pd.DataFrame:
        """
        生成价格因子
        
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
                    # 使用更高效的shift操作
                    shifted = self.data[field].groupby(level=1).shift(d)
                    features[name] = shifted / close
                    
        return pd.DataFrame(features, index=self.data.index)
    
    def _generate_volume_factors(self, windows: List[int] = [0]) -> pd.DataFrame:
        """
        生成成交量因子
        
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
                shifted = volume.groupby(level=1).shift(d)
                features[name] = shifted / (volume + 1e-12)
                
        return pd.DataFrame(features, index=self.data.index)
    
    def _generate_rolling_factors(self, windows: List[int]) -> pd.DataFrame:
        """
        生成滚动统计因子
        优化：减少临时对象创建，使用向量化操作
        
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
        
        # 预计算常用序列
        close_pct = close.groupby(level=1).pct_change()
        volume_pct = volume.groupby(level=1).pct_change()
        log_vol = np.log(volume + 1)
        
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
            
            # CORR: 价格与成交量相关性 - 优化版本，避免多次concat
            features[f'CORR{d}'] = self._calc_rolling_corr(close, log_vol, d)
            
            # CORD: 价格变化与成交量变化相关性 - 优化版本
            features[f'CORD{d}'] = self._calc_rolling_corr(close_pct, volume_pct, d)
            
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
            weighted_vol = np.abs(close_pct) * volume
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
            
            vol_sum_gain = vol_gain.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).sum()
            )
            vol_sum_loss = vol_loss.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).sum()
            )
            vol_sum_abs = vol_abs.groupby(level=1).transform(
                lambda x: x.rolling(d, min_periods=1).sum()
            )
            
            features[f'VSUMP{d}'] = vol_sum_gain / (vol_sum_abs + 1e-12)
            features[f'VSUMN{d}'] = vol_sum_loss / (vol_sum_abs + 1e-12)
            features[f'VSUMD{d}'] = (vol_sum_gain - vol_sum_loss) / (vol_sum_abs + 1e-12)
            
            # 清理临时变量
            del ma_rolling, std_rolling, beta_rolling, resi_rolling
            del max_rolling, min_rolling, qtlu_rolling, qtld_rolling
            del rsv_min, rsv_max, up, down
            del gain, loss, abs_change, sum_gain, sum_loss, sum_abs
            del vma_rolling, vstd_rolling, weighted_vol, wvma_std, wvma_mean
            del vol_gain, vol_loss, vol_abs, vol_sum_gain, vol_sum_loss, vol_sum_abs
            gc.collect()
            
        return pd.DataFrame(features, index=self.data.index)
    
    def _calc_rolling_corr(self, series1: pd.Series, series2: pd.Series, window: int) -> pd.Series:
        """
        优化的滚动相关系数计算
        避免循环和多次concat，直接使用groupby+rolling
        
        Parameters:
        -----------
        series1 : pd.Series
            第一个序列
        series2 : pd.Series
            第二个序列
        window : int
            窗口大小
            
        Returns:
        --------
        pd.Series
            滚动相关系数
        """
        # 使用groupby确保按symbol分组计算
        # 优化：使用更高效的rolling correlation计算
        def calc_corr(group):
            s1 = group[series1.name]
            s2 = group[series2.name]
            return s1.rolling(window, min_periods=1).corr(s2)
        
        # 重置为单层索引进行计算，然后恢复MultiIndex
        df = pd.DataFrame({series1.name: series1, series2.name: series2})
        result = df.groupby(level=1, group_keys=False).apply(calc_corr)
        
        return result
    
    @staticmethod
    def _calc_rsquare(y):
        """计算R方值"""
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / (ss_tot + 1e-12))
    
    @staticmethod
    def _calc_resi(y):
        """计算线性回归残差"""
        if len(y) < 2:
            return np.nan
        x = np.arange(len(y))
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        return y[-1] - y_pred[-1]