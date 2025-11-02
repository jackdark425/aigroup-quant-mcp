"""
深度学习模型 - LSTM/GRU/Transformer
参考Qlib实现的深度学习模型

注意：当前版本已移除torch依赖，深度学习工具暂时不可用
如需使用深度学习功能，请安装torch包
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple
from pathlib import Path
import gc

# 标记torch不可用
TORCH_AVAILABLE = False


class LSTMModel:
    """
    LSTM模型用于股票价格预测
    
    注意：当前版本已移除torch依赖，此模型暂时不可用
    如需使用深度学习功能，请安装torch包
    """
    
    def __init__(self, **kwargs):
        raise ImportError(
            "LSTM模型需要torch包支持。请安装torch: pip install torch\n"
            "或者使用其他机器学习模型如LightGBM、XGBoost"
        )


class GRUModel:
    """
    GRU模型用于股票价格预测
    
    注意：当前版本已移除torch依赖，此模型暂时不可用
    如需使用深度学习功能，请安装torch包
    """
    
    def __init__(self, **kwargs):
        raise ImportError(
            "GRU模型需要torch包支持。请安装torch: pip install torch\n"
            "或者使用其他机器学习模型如LightGBM、XGBoost"
        )


class TransformerModel:
    """
    Transformer模型用于股票价格预测
    
    注意：当前版本已移除torch依赖，此模型暂时不可用
    如需使用深度学习功能，请安装torch包
    """
    
    def __init__(self, **kwargs):
        raise ImportError(
            "Transformer模型需要torch包支持。请安装torch: pip install torch\n"
            "或者使用其他机器学习模型如LightGBM、XGBoost"
        )