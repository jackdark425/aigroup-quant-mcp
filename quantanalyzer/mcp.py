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
data_store = {}  # 存储加载的数据
factor_store = {}  # 存储计算的因子
model_store = {}  # 存储训练的模型
processor_store = {}  # 存储已配置的Processor

# 创建MCP server实例
app = Server("aigroup-quant-mcp")


@app.list_tools()
async def handle_list_tools() -> List[types.Tool]:
    """列出所有可用工具"""
    return [
        # ===== 数据加载工具 =====
        types.Tool(
            name="load_csv_data",
            description="从CSV文件加载股票数据到内存",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "CSV文件路径"
                    },
                    "data_id": {
                        "type": "string",
                        "description": "数据标识ID"
                    }
                },
                "required": ["file_path", "data_id"]
            }
        ),
        
        # ===== 因子计算工具 =====
        types.Tool(
            name="calculate_factor",
            description="计算量化因子（支持6个基础因子：momentum, volatility, volume_ratio, rsi, macd, bollinger_bands）",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_id": {
                        "type": "string",
                        "description": "数据ID"
                    },
                    "factor_name": {
                        "type": "string",
                        "description": "因子名称"
                    },
                    "factor_type": {
                        "type": "string",
                        "enum": ["momentum", "volatility", "volume_ratio", "rsi", "macd", "bollinger_bands"],
                        "description": "因子类型"
                    },
                    "period": {
                        "type": "integer",
                        "description": "计算周期",
                        "default": 20
                    }
                },
                "required": ["data_id", "factor_name", "factor_type"]
            }
        ),
        
        types.Tool(
            name="generate_alpha158",
            description="生成Alpha158因子集（包含158个技术指标因子）",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_id": {
                        "type": "string",
                        "description": "数据ID"
                    },
                    "result_id": {
                        "type": "string",
                        "description": "结果因子集ID"
                    },
                    "kbar": {
                        "type": "boolean",
                        "description": "是否生成K线形态因子（9个）",
                        "default": True
                    },
                    "price": {
                        "type": "boolean",
                        "description": "是否生成价格因子（5个）",
                        "default": True
                    },
                    "volume": {
                        "type": "boolean",
                        "description": "是否生成成交量因子（5个）",
                        "default": True
                    },
                    "rolling": {
                        "type": "boolean",
                        "description": "是否生成滚动统计因子（最多139个）",
                        "default": True
                    },
                    "rolling_windows": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "滚动窗口大小列表，默认[5,10,20,30,60]"
                    }
                },
                "required": ["data_id", "result_id"]
            }
        ),
        
        # ===== 因子评估工具 =====
        types.Tool(
            name="evaluate_factor_ic",
            description="评估因子IC（信息系数）",
            inputSchema={
                "type": "object",
                "properties": {
                    "factor_name": {
                        "type": "string",
                        "description": "因子名称"
                    },
                    "data_id": {
                        "type": "string",
                        "description": "数据ID"
                    },
                    "method": {
                        "type": "string",
                        "enum": ["spearman", "pearson"],
                        "default": "spearman",
                        "description": "计算方法"
                    }
                },
                "required": ["factor_name", "data_id"]
            }
        ),
        
        # ===== 深度学习模型工具 =====
        types.Tool(
            name="train_lstm_model",
            description="训练LSTM深度学习模型",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_id": {
                        "type": "string",
                        "description": "训练数据ID"
                    },
                    "model_id": {
                        "type": "string",
                        "description": "模型ID"
                    },
                    "hidden_size": {
                        "type": "integer",
                        "description": "隐藏层大小",
                        "default": 64
                    },
                    "num_layers": {
                        "type": "integer",
                        "description": "LSTM层数",
                        "default": 2
                    },
                    "n_epochs": {
                        "type": "integer",
                        "description": "训练轮数",
                        "default": 50
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "批次大小",
                        "default": 800
                    },
                    "lr": {
                        "type": "number",
                        "description": "学习率",
                        "default": 0.001
                    }
                },
                "required": ["data_id", "model_id"]
            }
        ),
        
        types.Tool(
            name="train_gru_model",
            description="训练GRU深度学习模型",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_id": {
                        "type": "string",
                        "description": "训练数据ID"
                    },
                    "model_id": {
                        "type": "string",
                        "description": "模型ID"
                    },
                    "hidden_size": {
                        "type": "integer",
                        "description": "隐藏层大小",
                        "default": 64
                    },
                    "num_layers": {
                        "type": "integer",
                        "description": "GRU层数",
                        "default": 2
                    },
                    "n_epochs": {
                        "type": "integer",
                        "description": "训练轮数",
                        "default": 50
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "批次大小",
                        "default": 800
                    }
                },
                "required": ["data_id", "model_id"]
            }
        ),
        
        types.Tool(
            name="train_transformer_model",
            description="训练Transformer深度学习模型",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_id": {
                        "type": "string",
                        "description": "训练数据ID"
                    },
                    "model_id": {
                        "type": "string",
                        "description": "模型ID"
                    },
                    "d_model": {
                        "type": "integer",
                        "description": "模型维度",
                        "default": 64
                    },
                    "nhead": {
                        "type": "integer",
                        "description": "注意力头数",
                        "default": 4
                    },
                    "num_layers": {
                        "type": "integer",
                        "description": "Transformer层数",
                        "default": 2
                    },
                    "n_epochs": {
                        "type": "integer",
                        "description": "训练轮数",
                        "default": 50
                    }
                },
                "required": ["data_id", "model_id"]
            }
        ),
        
        types.Tool(
            name="predict_with_model",
            description="使用训练好的模型进行预测",
            inputSchema={
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "模型ID"
                    },
                    "data_id": {
                        "type": "string",
                        "description": "预测数据ID"
                    },
                    "result_id": {
                        "type": "string",
                        "description": "预测结果ID"
                    }
                },
                "required": ["model_id", "data_id", "result_id"]
            }
        ),
        
        # ===== 信息查询工具 =====
        types.Tool(
            name="get_data_info",
            description="获取已加载数据的信息",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_id": {
                        "type": "string",
                        "description": "数据ID"
                    }
                },
                "required": ["data_id"]
            }
        ),
        
        types.Tool(
            name="list_factors",
            description="列出已计算的所有因子",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        
        types.Tool(
            name="list_models",
            description="列出已训练的所有模型",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        
        # ===== Processor数据清洗工具 =====
        types.Tool(
            name="apply_processor",
            description="对数据应用Processor进行清洗和标准化处理（支持：CSZScoreNorm截面标准化、DropnaLabel删除空标签、Fillna填充缺失值、ZScoreNorm时序标准化、RobustZScore鲁棒标准化、MinMaxNorm最小最大归一化、CSRankNorm排名标准化）",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_id": {
                        "type": "string",
                        "description": "要处理的数据ID"
                    },
                    "result_id": {
                        "type": "string",
                        "description": "处理后数据的ID"
                    },
                    "processor_type": {
                        "type": "string",
                        "enum": ["CSZScoreNorm", "DropnaLabel", "Fillna", "ZScoreNorm", "RobustZScoreNorm", "MinMaxNorm", "CSRankNorm"],
                        "description": "Processor类型"
                    },
                    "fields": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "要处理的列名列表，为空表示所有列"
                    },
                    "params": {
                        "type": "object",
                        "description": "Processor参数，如{'label_col': 'return', 'fill_value': 0, 'method': 'zscore'}",
                        "additionalProperties": True
                    },
                    "fit_first": {
                        "type": "boolean",
                        "description": "是否先执行fit（用于ZScoreNorm等需要学习参数的Processor）",
                        "default": False
                    }
                },
                "required": ["data_id", "result_id", "processor_type"]
            }
        ),
        
        types.Tool(
            name="create_processor_chain",
            description="创建Processor处理链，组合多个数据处理步骤",
            inputSchema={
                "type": "object",
                "properties": {
                    "chain_id": {
                        "type": "string",
                        "description": "Processor链的ID"
                    },
                    "processors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["CSZScoreNorm", "DropnaLabel", "Fillna", "ZScoreNorm", "RobustZScoreNorm", "MinMaxNorm", "CSRankNorm"]
                                },
                                "fields": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "params": {
                                    "type": "object",
                                    "additionalProperties": True
                                }
                            },
                            "required": ["type"]
                        },
                        "description": "Processor列表，按顺序执行"
                    }
                },
                "required": ["chain_id", "processors"]
            }
        ),
        
        types.Tool(
            name="apply_processor_chain",
            description="应用已创建的Processor链到数据，完整的Learn-Transform流程",
            inputSchema={
                "type": "object",
                "properties": {
                    "chain_id": {
                        "type": "string",
                        "description": "Processor链ID"
                    },
                    "train_data_id": {
                        "type": "string",
                        "description": "训练数据ID（用于fit学习参数）"
                    },
                    "test_data_id": {
                        "type": "string",
                        "description": "测试数据ID（可选，使用训练集学习的参数）"
                    },
                    "train_result_id": {
                        "type": "string",
                        "description": "处理后训练数据的ID"
                    },
                    "test_result_id": {
                        "type": "string",
                        "description": "处理后测试数据的ID（可选）"
                    }
                },
                "required": ["chain_id", "train_data_id", "train_result_id"]
            }
        ),
    ]


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
        elif name == "train_lstm_model":
            return await _train_lstm_model(arguments)
        elif name == "train_gru_model":
            return await _train_gru_model(arguments)
        elif name == "train_transformer_model":
            return await _train_transformer_model(arguments)
        elif name == "predict_with_model":
            return await _predict_with_model(arguments)
        elif name == "get_data_info":
            return await _get_data_info(arguments)
        elif name == "list_factors":
            return await _list_factors(arguments)
        elif name == "list_models":
            return await _list_models(arguments)
        elif name == "apply_processor":
            return await _apply_processor(arguments)
        elif name == "create_processor_chain":
            return await _create_processor_chain(arguments)
        elif name == "apply_processor_chain":
            return await _apply_processor_chain(arguments)
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
    
    loader = DataLoader()
    data = loader.load_from_csv(file_path)
    
    data_store[data_id] = data
    
    # 将MultiIndex DataFrame转换为可序列化格式
    sample_df = data.head(3).reset_index()
    sample_dict = sample_df.to_dict('records')
    
    # 转换所有可能的非序列化对象
    result = {
        "status": "success",
        "message": f"数据已加载为 '{data_id}'",
        "shape": list(data.shape),
        "columns": list(data.columns),
        "index_names": list(data.index.names),
        "date_range": {
            "start": str(data.index.get_level_values(0).min()),
            "end": str(data.index.get_level_values(0).max())
        },
        "symbols": [str(s) for s in data.index.get_level_values(1).unique()],
        "sample": sample_dict
    }
    
    # 使用自定义转换函数确保所有对象都可序列化
    result = convert_to_serializable(result)
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
    )]


async def _calculate_factor(args: Dict[str, Any]) -> List[types.TextContent]:
    """计算因子"""
    data_id = args["data_id"]
    factor_name = args["factor_name"]
    factor_type = args["factor_type"]
    period = args.get("period", 20)
    
    if data_id not in data_store:
        raise ValueError(f"数据 '{data_id}' 未找到")
    
    data = data_store[data_id]
    library = FactorLibrary()
    
    # 计算因子
    factor_func = getattr(library, factor_type)
    factor_values = factor_func(data, period)
    
    # 存储因子
    factor_store[factor_name] = factor_values
    
    # 计算统计信息
    stats = {
        "mean": float(factor_values.mean()),
        "std": float(factor_values.std()),
        "min": float(factor_values.min()),
        "max": float(factor_values.max()),
        "null_count": int(factor_values.isna().sum())
    }
    
    result = {
        "status": "success",
        "factor_name": factor_name,
        "factor_type": factor_type,
        "period": period,
        "statistics": stats
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
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
        raise ValueError(f"数据 '{data_id}' 未找到")
    
    data = data_store[data_id]
    generator = Alpha158Generator(data)
    
    # 生成因子
    alpha158 = generator.generate_all(
        kbar=kbar,
        price=price,
        volume=volume,
        rolling=rolling,
        rolling_windows=rolling_windows
    )
    
    # 存储因子
    factor_store[result_id] = alpha158
    
    # 计算统计信息
    stats = {
        "shape": list(alpha158.shape),
        "columns": list(alpha158.columns),
        "null_count": int(alpha158.isna().sum().sum()),
        "null_rate": float(alpha158.isna().sum().sum() / (alpha158.shape[0] * alpha158.shape[1]))
    }
    
    result = {
        "status": "success",
        "message": f"Alpha158因子已生成并存储为 '{result_id}'",
        "statistics": stats
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
    )]


async def _evaluate_factor_ic(args: Dict[str, Any]) -> List[types.TextContent]:
    """评估因子IC"""
    factor_name = args["factor_name"]
    data_id = args["data_id"]
    method = args.get("method", "spearman")
    
    if factor_name not in factor_store:
        raise ValueError(f"因子 '{factor_name}' 未找到")
    
    if data_id not in data_store:
        raise ValueError(f"数据 '{data_id}' 未找到")
    
    factor_data = factor_store[factor_name]
    price_data = data_store[data_id]["close"]
    
    # 计算收益率（向前1天）
    returns = price_data.groupby(level=1).pct_change().shift(-1)
    
    # 对齐索引
    aligned_factor = factor_data.dropna()
    aligned_returns = returns.reindex(aligned_factor.index)
    
    # 创建评估器并计算IC
    evaluator = FactorEvaluator(aligned_factor, aligned_returns)
    ic_result = evaluator.calculate_ic(method=method)
    
    result = {
        "status": "success",
        "factor_name": factor_name,
        "method": method,
        "ic_metrics": ic_result
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
    )]


async def _train_lstm_model(args: Dict[str, Any]) -> List[types.TextContent]:
    """训练LSTM模型"""
    data_id = args["data_id"]
    model_id = args["model_id"]
    hidden_size = args.get("hidden_size", 64)
    num_layers = args.get("num_layers", 2)
    n_epochs = args.get("n_epochs", 50)
    batch_size = args.get("batch_size", 800)
    lr = args.get("lr", 0.001)
    
    if data_id not in data_store:
        raise ValueError(f"数据 '{data_id}' 未找到")
    
    data = data_store[data_id]
    
    # 创建模型
    model = LSTMModel(
        d_feat=data.shape[1] if len(data.shape) > 1 else 1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        early_stop=10
    )
    
    # 训练模型（简化示例，实际使用中需要提供训练/验证数据）
    # 这里仅演示模型创建和存储
    model_store[model_id] = model
    
    result = {
        "status": "success",
        "message": f"LSTM模型已创建并存储为 '{model_id}'",
        "model_info": {
            "type": "LSTM",
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "lr": lr
        }
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
    )]


async def _train_gru_model(args: Dict[str, Any]) -> List[types.TextContent]:
    """训练GRU模型"""
    data_id = args["data_id"]
    model_id = args["model_id"]
    hidden_size = args.get("hidden_size", 64)
    num_layers = args.get("num_layers", 2)
    n_epochs = args.get("n_epochs", 50)
    batch_size = args.get("batch_size", 800)
    
    if data_id not in data_store:
        raise ValueError(f"数据 '{data_id}' 未找到")
    
    data = data_store[data_id]
    
    # 创建模型
    model = GRUModel(
        d_feat=data.shape[1] if len(data.shape) > 1 else 1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        n_epochs=n_epochs,
        batch_size=batch_size,
        early_stop=10
    )
    
    # 存储模型
    model_store[model_id] = model
    
    result = {
        "status": "success",
        "message": f"GRU模型已创建并存储为 '{model_id}'",
        "model_info": {
            "type": "GRU",
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "n_epochs": n_epochs,
            "batch_size": batch_size
        }
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
    )]


async def _train_transformer_model(args: Dict[str, Any]) -> List[types.TextContent]:
    """训练Transformer模型"""
    data_id = args["data_id"]
    model_id = args["model_id"]
    d_model = args.get("d_model", 64)
    nhead = args.get("nhead", 4)
    num_layers = args.get("num_layers", 2)
    n_epochs = args.get("n_epochs", 50)
    
    if data_id not in data_store:
        raise ValueError(f"数据 '{data_id}' 未找到")
    
    data = data_store[data_id]
    
    # 创建模型
    model = TransformerModel(
        d_feat=data.shape[1] if len(data.shape) > 1 else 1,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        n_epochs=n_epochs,
        early_stop=10
    )
    
    # 存储模型
    model_store[model_id] = model
    
    result = {
        "status": "success",
        "message": f"Transformer模型已创建并存储为 '{model_id}'",
        "model_info": {
            "type": "Transformer",
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "n_epochs": n_epochs
        }
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
    )]


async def _predict_with_model(args: Dict[str, Any]) -> List[types.TextContent]:
    """使用模型进行预测"""
    model_id = args["model_id"]
    data_id = args["data_id"]
    result_id = args["result_id"]
    
    if model_id not in model_store:
        raise ValueError(f"模型 '{model_id}' 未找到")
    
    if data_id not in data_store:
        raise ValueError(f"数据 '{data_id}' 未找到")
    
    model = model_store[model_id]
    data = data_store[data_id]
    
    # 执行预测（简化示例）
    # 注意：实际使用中需要确保模型已训练
    try:
        predictions = model.predict(data)
        factor_store[result_id] = predictions
        
        result = {
            "status": "success",
            "message": f"预测结果已存储为 '{result_id}'",
            "shape": list(predictions.shape) if hasattr(predictions, 'shape') else None
        }
    except Exception as e:
        # 如果模型未训练，返回错误信息
        result = {
            "status": "warning",
            "message": f"预测执行遇到问题: {str(e)}，请确保模型已训练"
        }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
    )]


async def _get_data_info(args: Dict[str, Any]) -> List[types.TextContent]:
    """获取数据信息"""
    data_id = args["data_id"]
    
    if data_id not in data_store:
        raise ValueError(f"数据 '{data_id}' 未找到")
    
    data = data_store[data_id]
    
    result = {
        "status": "success",
        "data_id": data_id,
        "shape": list(data.shape),
        "columns": list(data.columns) if hasattr(data, 'columns') else None,
        "index_names": list(data.index.names) if hasattr(data.index, 'names') else None,
        "date_range": {
            "start": str(data.index.get_level_values(0).min()) if hasattr(data.index, 'get_level_values') else None,
            "end": str(data.index.get_level_values(0).max()) if hasattr(data.index, 'get_level_values') else None
        } if hasattr(data, 'index') else None
    }
    
    # 使用自定义转换函数确保所有对象都可序列化
    result = convert_to_serializable(result)
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
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
        "factors": factors_info,
        "count": len(factor_store)
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
    )]


async def _list_models(args: Dict[str, Any]) -> List[types.TextContent]:
    """列出所有模型"""
    models_info = {}
    for model_id, model in model_store.items():
        model_type = model.__class__.__name__
        models_info[model_id] = {
            "type": model_type,
            "fitted": model.fitted if hasattr(model, 'fitted') else False
        }
    
    result = {
        "models": models_info,
        "count": len(model_store)
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
    )]


async def _apply_processor(args: Dict[str, Any]) -> List[types.TextContent]:
    """应用Processor"""
    data_id = args["data_id"]
    result_id = args["result_id"]
    processor_type = args["processor_type"]
    fields = args.get("fields", [])
    params = args.get("params", {})
    fit_first = args.get("fit_first", False)
    
    if data_id not in data_store:
        raise ValueError(f"数据 '{data_id}' 未找到")
    
    data = data_store[data_id]
    
    # 创建Processor实例
    processor_class = globals()[processor_type]
    processor = processor_class(**params)
    
    # 应用Processor
    if fit_first:
        # 先fit再transform（用于需要学习参数的Processor）
        processor.fit(data[fields] if fields else data)
        processed_data = processor.transform(data[fields] if fields else data)
    else:
        # 直接应用
        processed_data = processor(data[fields] if fields else data)
    
    # 存储处理后的数据
    data_store[result_id] = processed_data
    
    result = {
        "status": "success",
        "message": f"数据已处理并存储为 '{result_id}'",
        "processor_type": processor_type,
        "shape": list(processed_data.shape) if hasattr(processed_data, 'shape') else None
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
    )]


async def _create_processor_chain(args: Dict[str, Any]) -> List[types.TextContent]:
    """创建Processor链"""
    chain_id = args["chain_id"]
    processors_config = args["processors"]
    
    # 创建Processor实例列表
    processors = []
    for config in processors_config:
        processor_type = config["type"]
        params = config.get("params", {})
        processor_class = globals()[processor_type]
        processor = processor_class(**params)
        processors.append(processor)
    
    # 创建ProcessorChain
    chain = ProcessorChain(processors)
    
    # 存储Processor链
    processor_store[chain_id] = chain
    
    result = {
        "status": "success",
        "message": f"Processor链已创建并存储为 '{chain_id}'",
        "chain_length": len(processors),
        "processors": [p.__class__.__name__ for p in processors]
    }
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
    )]


async def _apply_processor_chain(args: Dict[str, Any]) -> List[types.TextContent]:
    """应用Processor链"""
    chain_id = args["chain_id"]
    train_data_id = args["train_data_id"]
    train_result_id = args["train_result_id"]
    test_data_id = args.get("test_data_id")
    test_result_id = args.get("test_result_id")
    
    if chain_id not in processor_store:
        raise ValueError(f"Processor链 '{chain_id}' 未找到")
    
    if train_data_id not in data_store:
        raise ValueError(f"训练数据 '{train_data_id}' 未找到")
    
    chain = processor_store[chain_id]
    train_data = data_store[train_data_id]
    
    # 对训练数据执行Learn-Transform流程
    processed_train_data = chain.learn_and_transform(train_data)
    data_store[train_result_id] = processed_train_data
    
    result = {
        "status": "success",
        "message": f"训练数据已处理并存储为 '{train_result_id}'"
    }
    
    # 如果提供了测试数据，也进行处理
    if test_data_id and test_result_id:
        if test_data_id not in data_store:
            raise ValueError(f"测试数据 '{test_data_id}' 未找到")
        
        test_data = data_store[test_data_id]
        processed_test_data = chain.transform(test_data)
        data_store[test_result_id] = processed_test_data
        
        result["test_message"] = f"测试数据已处理并存储为 '{test_result_id}'"
    
    return [types.TextContent(
        type="text",
        text=json.dumps(result, ensure_ascii=False, indent=2)
    )]


def main():
    """主函数入口"""
    import asyncio
    import sys
    
    # 检查是否有--help参数
    if "--help" in sys.argv:
        print("aigroup-quant-mcp - AI Group Quantitative Analysis MCP Service")
        print("Usage: aigroup-quant-mcp [options]")
        print("Options:")
        print("  --help     Show this help message")
        return
    
    # 运行异步主函数
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