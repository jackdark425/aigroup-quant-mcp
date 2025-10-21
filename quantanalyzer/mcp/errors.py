"""
MCP错误处理系统
"""

import json
import pandas as pd
from datetime import datetime


class MCPError:
    """MCP错误类型定义"""
    
    # 错误码定义
    DATA_NOT_FOUND = "DATA_NOT_FOUND"
    FACTOR_NOT_FOUND = "FACTOR_NOT_FOUND"
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    INVALID_PARAMETER = "INVALID_PARAMETER"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    COMPUTATION_ERROR = "COMPUTATION_ERROR"
    FILE_ERROR = "FILE_ERROR"
    
    @staticmethod
    def format_error(error_code: str, message: str, details: dict = None, suggestions: list = None):
        """格式化错误信息"""
        error_response = {
            "status": "error",
            "error_code": error_code,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
            "suggestions": suggestions or []
        }
        return json.dumps(error_response, ensure_ascii=False, indent=2)


# ===== 参数验证函数 =====

def validate_data_id(data_id: str, data_store: dict) -> str:
    """验证数据ID是否存在"""
    if data_id not in data_store:
        return MCPError.format_error(
            error_code=MCPError.DATA_NOT_FOUND,
            message=f"数据 '{data_id}' 未找到",
            details={
                "requested_id": data_id,
                "available_ids": list(data_store.keys()),
                "count": len(data_store)
            },
            suggestions=[
                "请先使用 load_csv_data 工具加载数据",
                "使用 list_factors 查看已加载的数据列表",
                f"可用的数据ID: {', '.join(list(data_store.keys())[:5])}" if data_store else "当前没有已加载的数据"
            ]
        )
    return None


def validate_factor_name(factor_name: str, factor_store: dict) -> str:
    """验证因子名称是否存在"""
    if factor_name not in factor_store:
        return MCPError.format_error(
            error_code=MCPError.FACTOR_NOT_FOUND,
            message=f"因子 '{factor_name}' 未找到",
            details={
                "requested_factor": factor_name,
                "available_factors": list(factor_store.keys()),
                "count": len(factor_store)
            },
            suggestions=[
                "请先使用 calculate_factor 或 generate_alpha158 生成因子",
                "使用 list_factors 查看已计算的因子列表",
                f"可用的因子: {', '.join(list(factor_store.keys())[:5])}" if factor_store else "当前没有已计算的因子"
            ]
        )
    return None


def validate_required_columns(data: pd.DataFrame, required_cols: list, data_id: str) -> str:
    """验证数据是否包含必需列"""
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        return MCPError.format_error(
            error_code=MCPError.INVALID_PARAMETER,
            message=f"数据缺少必需列: {missing_cols}",
            details={
                "data_id": data_id,
                "missing_columns": missing_cols,
                "required_columns": required_cols,
                "available_columns": list(data.columns)
            },
            suggestions=[
                "请确保CSV文件包含 open, high, low, close, volume 列",
                "检查CSV文件的列名是否正确（不区分大小写）",
                "查看数据加载返回的columns列表",
                "参考文档中的数据格式示例"
            ]
        )
    return None


def validate_data_length(data: pd.DataFrame, min_length: int, data_id: str) -> str:
    """验证数据量是否充足"""
    if len(data) < min_length:
        return MCPError.format_error(
            error_code=MCPError.INSUFFICIENT_DATA,
            message=f"数据量不足: 仅有 {len(data)} 条记录",
            details={
                "data_id": data_id,
                "current_rows": len(data),
                "minimum_required": min_length,
                "recommended": max(min_length * 10, 1000)
            },
            suggestions=[
                f"当前操作需要至少 {min_length} 条历史数据",
                f"建议使用至少 {max(min_length * 10, 1000)} 条数据以获得更好的效果",
                "检查数据加载是否完整",
                "考虑扩大数据的时间范围"
            ]
        )
    return None


def validate_window_size(windows: list) -> str:
    """验证滚动窗口参数"""
    if not isinstance(windows, list):
        return MCPError.format_error(
            error_code=MCPError.INVALID_PARAMETER,
            message="rolling_windows 必须是列表类型",
            details={"provided_type": str(type(windows).__name__)},
            suggestions=["使用列表格式，如 [5, 10, 20]"]
        )
    
    if not all(isinstance(x, int) for x in windows):
        return MCPError.format_error(
            error_code=MCPError.INVALID_PARAMETER,
            message="rolling_windows 列表元素必须是整数",
            details={"provided_values": windows},
            suggestions=["确保所有窗口大小都是整数"]
        )
    
    if not all(2 <= x <= 250 for x in windows):
        invalid = [x for x in windows if not (2 <= x <= 250)]
        return MCPError.format_error(
            error_code=MCPError.INVALID_PARAMETER,
            message="窗口大小必须在 2-250 之间",
            details={
                "invalid_windows": invalid,
                "valid_range": "2-250"
            },
            suggestions=[
                "使用有效的窗口大小，如 [5, 10, 20, 30, 60]",
                "窗口太小(<2)会导致统计不稳定",
                "窗口太大(>250)会需要过多历史数据"
            ]
        )
    
    if len(windows) > 10:
        return MCPError.format_error(
            error_code=MCPError.INVALID_PARAMETER,
            message=f"窗口数量过多: {len(windows)} 个",
            details={
                "window_count": len(windows),
                "maximum_allowed": 10
            },
            suggestions=[
                "最多支持10个窗口",
                "过多窗口会显著增加计算时间",
                "建议使用3-5个代表性窗口"
            ]
        )
    
    return None


def validate_period(period: int) -> str:
    """验证周期参数"""
    if not isinstance(period, int):
        return MCPError.format_error(
            error_code=MCPError.INVALID_PARAMETER,
            message="period 必须是整数",
            details={"provided_type": str(type(period).__name__)},
            suggestions=["使用整数值，如 20"]
        )
    
    if not (2 <= period <= 250):
        return MCPError.format_error(
            error_code=MCPError.INVALID_PARAMETER,
            message=f"period 必须在 2-250 之间，当前值: {period}",
            details={
                "provided_value": period,
                "valid_range": "2-250"
            },
            suggestions=[
                "常用周期: 5(周), 10(两周), 20(月), 60(季度)",
                "周期太小会导致噪音过大",
                "周期太大需要更多历史数据"
            ]
        )
    
    return None


def validate_file_path(file_path: str) -> str:
    """验证文件路径"""
    import os
    
    if not file_path.endswith('.csv'):
        return MCPError.format_error(
            error_code=MCPError.FILE_ERROR,
            message="文件必须是CSV格式",
            details={"provided_path": file_path},
            suggestions=["使用 .csv 扩展名的文件"]
        )
    
    if not os.path.exists(file_path):
        return MCPError.format_error(
            error_code=MCPError.FILE_ERROR,
            message=f"文件不存在: {file_path}",
            details={"provided_path": file_path},
            suggestions=[
                "检查文件路径是否正确",
                "确认文件名拼写",
                "使用绝对路径避免路径问题"
            ]
        )
    
    return None