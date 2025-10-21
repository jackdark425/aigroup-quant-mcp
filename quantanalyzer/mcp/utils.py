"""
MCP工具函数
包含序列化、转换等通用函数
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime

# ===== JSON序列化优化 =====

try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False


def serialize_response(data: dict) -> str:
    """优化的JSON序列化函数"""
    # 先转换为可序列化格式
    data = convert_to_serializable(data)
    
    if USE_ORJSON:
        return orjson.dumps(
            data,
            option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS
        ).decode('utf-8')
    else:
        return json.dumps(data, ensure_ascii=False, indent=2)


def convert_to_serializable(obj):
    """转换对象为JSON可序列化格式"""
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    elif isinstance(obj, (pd.DatetimeIndex, pd.PeriodIndex, pd.TimedeltaIndex)):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    return obj