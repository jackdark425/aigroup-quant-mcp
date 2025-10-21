"""
QuantAnalyzer MCP Service - aigroup-quant-mcp
提供量化分析MCP工具，支持Alpha158因子和深度学习模型
"""

import json
import sys
from typing import Any, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

# ===== 性能优化：JSON序列化 =====

try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False

def serialize_response(data: dict) -> str:
    """优化的JSON序列化函数"""
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


# 导入QuantAnalyzer模块
from quantanalyzer.data import DataLoader
from quantanalyzer.data.processor import (
    DropnaLabel, Fillna, CSZScoreNorm, ZScoreNorm,
    RobustZScoreNorm, MinMaxNorm, CSRankNorm, ProcessorChain
)
from quantanalyzer.factor import FactorLibrary, FactorEvaluator, Alpha158Generator
from quantanalyzer.model import LSTMModel, GRUModel, TransformerModel

# 全局存储
data_store = {}
factor_store = {}
model_store = {}
processor_store = {}

# 创建MCP server实例
app = Server("aigroup-quant-mcp")


@app.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """列出所有可用工具"""
    return [
        types.Tool(
            name="load_csv_data",
            description="加载CSV数据到内存",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "CSV文件路径"},
                    "data_id": {"type": "string", "description": "数据ID"}
                },
                "required": ["file_path", "data_id"]
            }
        ),
        types.Tool(
            name="calculate_factor",
            description="计算单个量化因子",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_id": {"type": "string", "description": "数据ID"},
                    "factor_name": {"type": "string", "description": "因子名称"},
                    "factor_type": {
                        "type": "string",
                        "enum": ["momentum", "volatility", "volume_ratio", "rsi", "macd", "bollinger_bands"],
                        "description": "因子类型"
                    },
                    "period": {"type": "integer", "default": 20, "description": "计算周期"}
                },
                "required": ["data_id", "factor_name", "factor_type"]
            }
        ),
        types.Tool(
            name="generate_alpha158",
            description="生成Alpha158因子集",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_id": {"type": "string", "description": "数据ID"},
                    "result_id": {"type": "string", "description": "结果ID"},
                    "kbar": {"type": "boolean", "default": True},
                    "price": {"type": "boolean", "default": True},
                    "volume": {"type": "boolean", "default": True},
                    "rolling": {"type": "boolean", "default": True},
                    "rolling_windows": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "default": [5, 10, 20, 30, 60]
                    }
                },
                "required": ["data_id", "result_id"]
            }
        ),
        types.Tool(
            name="evaluate_factor_ic",
            description="评估因子IC",
            inputSchema={
                "type": "object",
                "properties": {
                    "factor_name": {"type": "string", "description": "因子名称"},
                    "data_id": {"type": "string", "description": "数据ID"},
                    "method": {
                        "type": "string",
                        "enum": ["spearman", "pearson"],
                        "default": "spearman"
                    }
                },
                "required": ["factor_name", "data_id"]
            }
        ),
        types.Tool(
            name="list_factors",
            description="列出所有已加载的数据和因子",
            inputSchema={"type": "object", "properties": {}}
        ),
    ]


@app.list_resources()
async def handle_list_resources() -> List[types.Resource]:
    """列出可用资源"""
    return []


@app.read_resource()
async def handle_read_resource(uri: str) -> str:
    """读取资源内容"""
    return "资源未找到"


@app.call_tool()
async def handle_call_tool(
    name: str,
    arguments: Dict[str, Any]
) -> List[types.TextContent]:
    """处理工具调用"""
    
    try:
        if name == "load_csv_data":
            return await _load_csv_data(arguments)
        elif name == "calculate_factor":
            return await _calculate_factor(arguments)
        elif name == "generate_alpha158":
            return await _generate_alpha158(arguments)
        elif name == "evaluate_factor_ic":
            return await _evaluate_factor_ic(arguments)
        elif name == "list_factors":
            return await _list_factors(arguments)
        else:
            raise ValueError(f"未知工具: {name}")
            
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({"error": str(e)}, ensure_ascii=False, indent=2)
        )]


# ===== 工具实现函数 =====

async def _load_csv_data(args: Dict[str, Any]) -> List[types.TextContent]:
    """加载CSV数据"""
    file_path = args["file_path"]
    data_id = args["data_id"]
    
    try:
        loader = DataLoader()
        data = loader.load_from_csv(file_path)
        data_store[data_id] = data
        
        result = {
            "status": "success",
            "message": f"✅ 数据已成功加载为 '{data_id}'",
            "summary": {
                "data_id": data_id,
                "shape": {"rows": data.shape[0], "columns": data.shape[1]},
                "columns": list(data.columns),
                "date_range": {
                    "start": str(data.index.get_level_values(0).min()),
                    "end": str(data.index.get_level_values(0).max())
                }
            }
        }
        
        result = convert_to_serializable(result)
        return [types.TextContent(type="text", text=serialize_response(result))]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False, indent=2)
        )]


async def _calculate_factor(args: Dict[str, Any]) -> List[types.TextContent]:
    """计算因子"""
    data_id = args["data_id"]
    factor_name = args["factor_name"]
    factor_type = args["factor_type"]
    period = args.get("period", 20)
    
    if data_id not in data_store:
        return [types.TextContent(
            type="text",
            text=json.dumps({"status": "error", "message": f"数据 '{data_id}' 未找到"}, ensure_ascii=False)
        )]
    
    try:
        data = data_store[data_id]
        library = FactorLibrary()
        factor_func = getattr(library, factor_type)
        factor_values = factor_func(data, period)
        
        factor_store[factor_name] = factor_values
        
        result = {
            "status": "success",
            "message": f"✅ 因子 '{factor_name}' 计算完成",
            "factor_info": {
                "factor_name": factor_name,
                "factor_type": factor_type,
                "period": period,
                "data_rows": len(factor_values)
            }
        }
        
        return [types.TextContent(type="text", text=serialize_response(result))]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False, indent=2)
        )]


async def _generate_alpha158(args: Dict[str, Any]) -> List[types.TextContent]:
    """生成Alpha158因子"""
    data_id = args["data_id"]
    result_id = args["result_id"]
    kbar = args.get("kbar", True)
    price = args.get("price", True)
    volume = args.get("volume", True)
    rolling = args.get("rolling", True)
    rolling_windows = args.get("rolling_windows", [5, 10, 20, 30, 60])
    
    if data_id not in data_store:
        return [types.TextContent(
            type="text",
            text=json.dumps({"status": "error", "message": f"数据 '{data_id}' 未找到"}, ensure_ascii=False)
        )]
    
    try:
        data = data_store[data_id]
        generator = Alpha158Generator(data)
        
        alpha158 = generator.generate_all(
            kbar=kbar,
            price=price,
            volume=volume,
            rolling=rolling,
            rolling_windows=rolling_windows
        )
        
        factor_store[result_id] = alpha158
        
        result = {
            "status": "success",
            "message": f"✅ Alpha158因子已生成并存储为 '{result_id}'",
            "factor_info": {
                "factor_id": result_id,
                "total_factors": len(alpha158.columns),
                "shape": list(alpha158.shape)
            }
        }
        
        return [types.TextContent(type="text", text=serialize_response(result))]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False, indent=2)
        )]


async def _evaluate_factor_ic(args: Dict[str, Any]) -> List[types.TextContent]:
    """评估因子IC"""
    factor_name = args["factor_name"]
    data_id = args["data_id"]
    method = args.get("method", "spearman")
    
    if factor_name not in factor_store:
        return [types.TextContent(
            type="text",
            text=json.dumps({"status": "error", "message": f"因子 '{factor_name}' 未找到"}, ensure_ascii=False)
        )]
    
    if data_id not in data_store:
        return [types.TextContent(
            type="text",
            text=json.dumps({"status": "error", "message": f"数据 '{data_id}' 未找到"}, ensure_ascii=False)
        )]
    
    try:
        factor_data = factor_store[factor_name]
        price_data = data_store[data_id]["close"]
        
        returns = price_data.groupby(level=1).pct_change().shift(-1)
        aligned_factor = factor_data.dropna()
        aligned_returns = returns.reindex(aligned_factor.index)
        
        evaluator = FactorEvaluator(aligned_factor, aligned_returns)
        ic_result = evaluator.calculate_ic(method=method)
        
        result = {
            "status": "success",
            "message": f"✅ 因子 '{factor_name}' IC评估完成",
            "ic_metrics": ic_result
        }
        
        return [types.TextContent(type="text", text=serialize_response(result))]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False, indent=2)
        )]


async def _list_factors(args: Dict[str, Any]) -> List[types.TextContent]:
    """列出所有因子"""
    factors_info = {}
    for factor_id, factor in factor_store.items():
        factors_info[factor_id] = {
            "type": str(type(factor).__name__),
            "shape": list(factor.shape) if hasattr(factor, 'shape') else None
        }
    
    result = {
        "data_count": len(data_store),
        "data_ids": list(data_store.keys()),
        "factor_count": len(factor_store),
        "factors": factors_info
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
    )]


def main():
    """主函数入口"""
    import asyncio
    
    if "--help" in sys.argv:
        print("aigroup-quant-mcp - AI Group Quantitative Analysis MCP Service")
        print("Usage: aigroup-quant-mcp [options]")
        print("Options:")
        print("  --help     Show this help message")
        return
    
    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass


async def _main():
    """异步主函数"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    main()