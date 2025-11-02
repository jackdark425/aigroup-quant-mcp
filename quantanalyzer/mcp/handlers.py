"""
MCP工具处理函数
包含所有工具的业务逻辑和错误处理
"""

from typing import Any, Dict, List
from mcp import types
import json
import pandas as pd

from .errors import (
    MCPError,
    validate_data_id,
    validate_factor_name,
    validate_required_columns,
    validate_data_length,
    validate_window_size,
    validate_period,
    validate_file_path
)
from .utils import serialize_response, convert_to_serializable

def _convert_index_to_string(data_dict: dict) -> dict:
    """将字典中的元组索引转换为字符串格式"""
    if not isinstance(data_dict, dict):
        return data_dict
    
    converted = {}
    for key, value in data_dict.items():
        if isinstance(key, tuple):
            # 将元组索引转换为字符串格式，如 "(2023-01-01, 000001.SZ)"
            converted[str(key)] = value
        else:
            converted[str(key)] = value
    return converted

from quantanalyzer.data import DataLoader
from quantanalyzer.data.processor import (
    ProcessInf, CSZFillna, CSZScoreNorm, ZScoreNorm,
    RobustZScoreNorm, CSRankNorm, MinMaxNorm, ProcessorChain
)
from quantanalyzer.factor import FactorLibrary, FactorEvaluator, Alpha158Generator
# 深度学习模型已移除，专注于传统机器学习算法
# from quantanalyzer.model.deep_models import LSTMModel, GRUModel, TransformerModel
# 全局存储
data_store = {}
factor_store = {}
model_store = {}
processor_store = {}


async def handle_preprocess_data(args: Dict[str, Any]) -> List[types.TextContent]:
    """数据预处理和清洗"""
    file_path = args["file_path"]
    data_id = args["data_id"]
    auto_clean = args.get("auto_clean", True)  # 默认开启自动清洗
    export_path = args.get("export_path", None)  # 导出路径（可选）
    
    # 验证文件路径
    error = validate_file_path(file_path)
    if error:
        return [types.TextContent(type="text", text=error)]
    
    try:
        loader = DataLoader()
        data = loader.load_from_csv(file_path)
        
        # 验证必需列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        error = validate_required_columns(data, required_cols, data_id)
        if error:
            return [types.TextContent(type="text", text=error)]
        
        # 验证数据量 - 支持快速测试模式
        min_data_length = 30  # 降低快速测试门槛
        error = validate_data_length(data, min_length=min_data_length, data_id=data_id, operation_type="quick_test")
        if error:
            return [types.TextContent(type="text", text=error)]
        
        # 记录原始数据质量
        original_null_count = int(data.isna().sum().sum())
        original_null_rate = original_null_count / (data.shape[0] * data.shape[1]) if data.shape[0] * data.shape[1] > 0 else 0
        
        # 自动数据清洗
        cleaned = False
        cleaning_methods = []
        if auto_clean:
            try:
                # 检查股票数量
                symbol_count = len(data.index.get_level_values(1).unique())
                
                # 按照Qlib标准流程清洗数据
                # 注意：这里只清洗异常值和缺失值，不进行标准化
                # 标准化应该在因子生成后进行（对因子标准化，而非原始OHLCV数据）
                processor_chain = ProcessorChain([
                    ProcessInf(),          # 处理无穷值（用截面均值替换）
                    CSZFillna(),           # 截面填充缺失值（用截面均值）
                ])
                data = processor_chain.fit_transform(data.copy())
                
                cleaning_methods = [
                    "ProcessInf() - 用截面均值替换inf/-inf",
                    "CSZFillna() - 用截面均值填充NaN"
                ]
                
                # 多股票场景的提示信息
                if symbol_count > 1:
                    cleaning_methods.append(
                        "⚠️ 提示：原始数据未标准化，建议在因子生成后使用CSZScoreNorm标准化因子"
                    )
                
                cleaned = True
            except Exception as e:
                # 如果清洗失败，使用原始数据并记录警告
                cleaned = False
                cleaning_methods = []
        
        data_store[data_id] = data
        
        # 计算清洗后的数据质量
        current_null_count = int(data.isna().sum().sum())
        current_null_rate = current_null_count / (data.shape[0] * data.shape[1]) if data.shape[0] * data.shape[1] > 0 else 0
        
        # 导出清洗后的数据到本地
        export_info = None
        if export_path or cleaned:  # 如果指定导出路径或进行了清洗，则导出
            import os
            from pathlib import Path
            from datetime import datetime
            
            # 如果没有指定导出路径，使用默认路径
            if not export_path:
                # 默认保存到项目根目录的 exports 文件夹
                exports_dir = Path("exports")
                exports_dir.mkdir(exist_ok=True)
                
                # 生成文件名：data_id_timestamp.csv
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_filename = f"{data_id}_cleaned_{timestamp}.csv"
                export_path = exports_dir / export_filename
            else:
                export_path = Path(export_path)
                # 确保目录存在
                export_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 导出数据
            try:
                data.to_csv(export_path, encoding='utf-8')
                export_info = {
                    "exported": True,
                    "path": str(export_path.absolute()),
                    "size_mb": round(os.path.getsize(export_path) / (1024 * 1024), 2)
                }
            except Exception as e:
                export_info = {
                    "exported": False,
                    "error": f"导出失败: {str(e)}"
                }
        
        # 优化的响应格式
        result = {
            "status": "success",
            "message": f"✅ 数据已成功预处理为 '{data_id}'" + (f" (已自动清洗)" if cleaned else "") + (f" 并导出到本地" if export_info and export_info.get('exported') else ""),
            "summary": {
                "data_id": data_id,
                "shape": {"rows": data.shape[0], "columns": data.shape[1]},
                "columns": list(data.columns),
                "date_range": {
                    "start": str(data.index.get_level_values(0).min()),
                    "end": str(data.index.get_level_values(0).max())
                },
                "symbol_count": len(data.index.get_level_values(1).unique()),
                "total_records": len(data)
            },
            "preview": {
                "head_3": data.head(3).to_dict('records')[:3],
                "tail_3": data.tail(3).to_dict('records')[:3]
            },
            "data_quality": {
                "auto_cleaned": cleaned,
                "original_missing_values": original_null_count if cleaned else current_null_count,
                "original_missing_rate": f"{original_null_rate * 100:.2f}%" if cleaned else f"{current_null_rate * 100:.2f}%",
                "current_missing_values": current_null_count,
                "current_missing_rate": f"{current_null_rate * 100:.2f}%",
                "quality_score": "优秀" if current_null_rate < 0.01 else "良好" if current_null_rate < 0.05 else "需要清洗",
                "cleaning_applied": cleaning_methods
            },
            "next_steps": [
                f"使用 generate_alpha158 生成因子: result_id='alpha158_{data_id}'",
                f"或使用 calculate_factor 计算单个因子",
                f"使用 list_factors 查看已加载数据: 无需参数"
            ]
        }
        
        # 添加导出信息
        if export_info:
            result["export_info"] = export_info
        
        result = convert_to_serializable(result)
        return [types.TextContent(type="text", text=serialize_response(result))]
        
    except FileNotFoundError as e:
        return [types.TextContent(
            type="text",
            text=MCPError.format_error(
                error_code=MCPError.FILE_ERROR,
                message=f"文件未找到: {file_path}",
                details={"file_path": file_path, "error": str(e)},
                suggestions=[
                    "检查文件路径是否正确",
                    "确认文件是否存在",
                    "使用绝对路径"
                ]
            )
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=MCPError.format_error(
                error_code=MCPError.COMPUTATION_ERROR,
                message=f"数据加载失败: {str(e)}",
                details={
                    "file_path": file_path,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e)
                },
                suggestions=[
                    "检查CSV文件格式是否正确",
                    "确认文件编码为UTF-8",
                    "验证文件是否包含必需的列"
                ]
            )
        )]


async def handle_calculate_factor(args: Dict[str, Any]) -> List[types.TextContent]:
    """计算因子"""
    data_id = args["data_id"]
    factor_name = args["factor_name"]
    factor_type = args["factor_type"]
    period = args.get("period", 20)
    
    # 验证数据ID
    error = validate_data_id(data_id, data_store)
    if error:
        return [types.TextContent(type="text", text=error)]
    
    # 验证period参数
    error = validate_period(period)
    if error:
        return [types.TextContent(type="text", text=error)]
    
    try:
        data = data_store[data_id]
        library = FactorLibrary()
        factor_func = getattr(library, factor_type)
        factor_values = factor_func(data, period)
        
        factor_store[factor_name] = factor_values
        
        # 优化的响应格式
        null_count = int(factor_values.isna().sum()) if hasattr(factor_values, 'isna') else 0
        null_rate = null_count / len(factor_values) if len(factor_values) > 0 else 0
        
        # 先计算质量分数，避免在构建result时引用自身
        quality_score = "优秀" if null_rate < 0.01 else "良好" if null_rate < 0.05 else "需要清洗"
        
        result = {
            "status": "success",
            "message": f"✅ 因子 '{factor_name}' 计算完成",
            "factor_info": {
                "factor_name": factor_name,
                "factor_type": factor_type,
                "period": period,
                "data_rows": len(factor_values),
                "valid_values": len(factor_values) - null_count
            },
            "data_quality": {
                "null_count": null_count,
                "null_rate": f"{null_rate * 100:.2f}%",
                "quality_score": quality_score
            },
            "next_steps": [
            {
                "step": 1,
                "action": "评估因子有效性",
                "tool": "evaluate_factor_ic",
                "params_example": {
                    "factor_name": factor_name,
                    "data_id": data_id,
                    "method": "spearman"
                },
                "reason": "判断因子是否有预测能力"
            },
            {
                "step": 2,
                "action": "如果IC有效，可生成更多因子或直接训练模型",
                "tools": ["generate_alpha158", "train_ml_model"]
            }
        ],
            "tips": [
                f"💡 因子类型: {factor_type}，周期: {period}天",
                f"💡 数据质量: {quality_score}",
                "💡 建议先评估IC再决定是否使用此因子"
            ]
        }
        
        return [types.TextContent(type="text", text=serialize_response(result))]
        
    except AttributeError:
        return [types.TextContent(
            type="text",
            text=MCPError.format_error(
                error_code=MCPError.INVALID_PARAMETER,
                message=f"不支持的因子类型: {factor_type}",
                details={
                    "factor_type": factor_type,
                    "supported_types": ["momentum", "volatility", "volume_ratio", "rsi", "macd", "bollinger_bands"]
                },
                suggestions=[
                    "使用支持的因子类型之一",
                    "检查factor_type参数拼写"
                ]
            )
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=MCPError.format_error(
                error_code=MCPError.COMPUTATION_ERROR,
                message=f"因子计算失败: {str(e)}",
                details={
                    "factor_name": factor_name,
                    "factor_type": factor_type,
                    "period": period,
                    "exception_type": type(e).__name__
                },
                suggestions=[
                    "检查数据中是否有NaN或Inf值",
                    "确认period参数合理",
                    "尝试使用更小的period值"
                ]
            )
        )]


async def handle_generate_alpha158(args: Dict[str, Any]) -> List[types.TextContent]:
    """生成Alpha158因子"""
    data_id = args["data_id"]
    result_id = args["result_id"]
    kbar = args.get("kbar", True)
    price = args.get("price", True)
    volume = args.get("volume", True)
    rolling = args.get("rolling", True)
    rolling_windows = args.get("rolling_windows", [5, 10, 20, 30, 60])
    export_path = args.get("export_path", None)  # 导出路径（可选）
    export_format = args.get("export_format", "csv")  # csv 或 json
    
    # 验证数据ID
    error = validate_data_id(data_id, data_store)
    if error:
        return [types.TextContent(type="text", text=error)]
    
    # 验证rolling_windows参数
    if rolling:
        error = validate_window_size(rolling_windows)
        if error:
            return [types.TextContent(type="text", text=error)]
    
    try:
        data = data_store[data_id]
        
        # 验证必需列
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        error = validate_required_columns(data, required_cols, data_id)
        if error:
            return [types.TextContent(type="text", text=error)]
        
        # 验证数据量 - 支持快速测试模式
        min_required = max(rolling_windows) if rolling and rolling_windows else 50  # 降低门槛
        error = validate_data_length(data, min_length=min_required, data_id=data_id, operation_type="quick_test")
        if error:
            return [types.TextContent(type="text", text=error)]
        
        generator = Alpha158Generator(data)
        
        alpha158 = generator.generate_all(
            kbar=kbar,
            price=price,
            volume=volume,
            rolling=rolling,
            rolling_windows=rolling_windows
        )
        
        factor_store[result_id] = alpha158
        
        # 导出Alpha158因子数据
        export_info = None
        if export_path:
            import os
            from pathlib import Path
            from datetime import datetime
            
            export_path_obj = Path(export_path)
            # 确保目录存在
            export_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                if export_format.lower() == "csv":
                    alpha158.to_csv(export_path, encoding='utf-8')
                elif export_format.lower() == "json":
                    alpha158.to_json(export_path, orient='records', indent=2)
                else:
                    export_format = "csv"  # 默认使用CSV
                    alpha158.to_csv(export_path, encoding='utf-8')
                
                file_size = os.path.getsize(export_path) / (1024 * 1024)
                export_info = {
                    "exported": True,
                    "path": str(export_path_obj.absolute()),
                    "format": export_format,
                    "size_mb": round(file_size, 2),
                    "factor_count": len(alpha158.columns),
                    "row_count": len(alpha158)
                }
            except Exception as e:
                export_info = {
                    "exported": False,
                    "error": f"导出失败: {str(e)}"
                }
        
        # 优化的响应格式
        null_count = int(alpha158.isna().sum().sum())
        total_values = alpha158.shape[0] * alpha158.shape[1]
        null_rate = null_count / total_values if total_values > 0 else 0
        
        # 先计算质量分数，避免在构建result时引用自身
        quality_score = "优秀" if null_rate < 0.01 else "良好" if null_rate < 0.05 else "需要清洗"
        
        result = {
            "status": "success",
            "message": f"✅ Alpha158因子已生成并存储为 '{result_id}'",
            "factor_info": {
                "factor_id": result_id,
                "total_factors": len(alpha158.columns),
                "shape": list(alpha158.shape),
                "categories": {
                    "kbar": 9 if kbar else 0,
                    "price": 5 if price else 0,
                    "volume": 5 if volume else 0,
                    "rolling": len(alpha158.columns) - (9 if kbar else 0) - (5 if price else 0) - (5 if volume else 0)
                }
            },
            "data_quality": {
                "null_count": null_count,
                "null_rate": f"{null_rate * 100:.2f}%",
                "quality_score": quality_score,
                "recommendation": "数据质量良好，可直接用于模型训练" if null_rate < 0.01 else "建议使用 apply_processor_chain 进行数据清洗"
            },
            "next_steps": [
                {
                    "step": 1,
                    "action": "数据预处理（建议）" if null_rate > 0 else "数据预处理（可选）",
                    "tool": "apply_processor_chain",
                    "reason": "清洗缺失值并标准化" if null_rate > 0 else "标准化数据提升模型效果"
                },
                {
                    "step": 2,
                    "action": "训练机器学习模型",
                    "tools": ["train_ml_model"],
                    "params_example": {
                        "data_id": result_id,
                        "model_id": f"model_{result_id}"
                    }
                },
                {
                    "step": 3,
                    "action": "因子评估（可选）",
                    "tool": "evaluate_factor_ic",
                    "params_example": {
                        "factor_name": result_id,
                        "data_id": data_id
                    }
                }
            ],
            "tips": [
                f"💡 因子数量: {len(alpha158.columns)}个，建议使用LSTM或Transformer模型",
                f"💡 数据质量: {quality_score}",
                "💡 如果数据量不足1000条，建议使用更小的rolling_windows",
                f"💡 使用export_path参数导出因子数据便于查看" if not export_path else f"💡 因子数据已导出到: {export_path}"
            ]
        }
        
        # 添加导出信息
        if export_info:
            result["export_info"] = export_info
        
        return [types.TextContent(type="text", text=serialize_response(result))]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=MCPError.format_error(
                error_code=MCPError.COMPUTATION_ERROR,
                message=f"Alpha158因子计算失败: {str(e)}",
                details={
                    "data_id": data_id,
                    "result_id": result_id,
                    "exception_type": type(e).__name__
                },
                suggestions=[
                    "检查数据格式是否正确",
                    "确认数据中没有NaN或Inf值",
                    "尝试减少rolling_windows参数",
                    "确保数据量足够（建议1000+条）"
                ]
            )
        )]


async def handle_evaluate_factor_ic(args: Dict[str, Any]) -> List[types.TextContent]:
    """评估因子IC"""
    factor_name = args["factor_name"]
    data_id = args["data_id"]
    method = args.get("method", "spearman")
    report_path = args.get("report_path", None)  # 评估报告保存路径（可选）
    
    # 验证因子名称
    error = validate_factor_name(factor_name, factor_store)
    if error:
        return [types.TextContent(type="text", text=error)]
    
    # 验证数据ID
    error = validate_data_id(data_id, data_store)
    if error:
        return [types.TextContent(type="text", text=error)]
    
    try:
        factor_data = factor_store[factor_name]
        price_data = data_store[data_id]["close"]
        
        # 确保price_data是Series而不是DataFrame
        if isinstance(price_data, pd.DataFrame):
            # 如果是DataFrame，只取第一列
            price_data = price_data.iloc[:, 0]
        
        # 修复：使用更稳健的收益率计算方法，避免边界NaN问题
        # 计算下一期收益率，但避免使用shift(-1)产生边界NaN
        returns = price_data.groupby(level=1).pct_change()
        # 将收益率向前移动一期，表示因子对下一期收益率的预测能力
        returns = returns.groupby(level=1).shift(-1)
        
        # 处理因子数据
        if isinstance(factor_data, pd.DataFrame):
            # 如果因子是DataFrame（Alpha158），需要逐列计算IC
            # 暂时只使用第一个因子列
            factor_data = factor_data.iloc[:, 0]
        
        # 修复：使用更稳健的数据对齐方法
        # 确保因子和收益率有相同的索引
        common_index = factor_data.index.intersection(returns.index)
        if len(common_index) == 0:
            return [types.TextContent(
                type="text",
                text=MCPError.format_error(
                    error_code=MCPError.COMPUTATION_ERROR,
                    message="因子数据和收益率数据没有共同的时间索引",
                    details={
                        "factor_name": factor_name,
                        "data_id": data_id,
                        "factor_index_count": len(factor_data.index),
                        "returns_index_count": len(returns.index)
                    },
                    suggestions=[
                        "检查因子数据和价格数据的时间范围是否匹配",
                        "确认数据预处理是否正确"
                    ]
                )
            )]
        
        # 使用共同索引对齐数据
        aligned_factor = factor_data.loc[common_index].dropna()
        aligned_returns = returns.loc[common_index].dropna()
        
        # 进一步确保对齐
        final_common_index = aligned_factor.index.intersection(aligned_returns.index)
        if len(final_common_index) == 0:
            return [types.TextContent(
                type="text",
                text=MCPError.format_error(
                    error_code=MCPError.COMPUTATION_ERROR,
                    message="因子和收益率数据在去除NaN后没有共同的时间索引",
                    details={
                        "factor_name": factor_name,
                        "data_id": data_id,
                        "aligned_factor_count": len(aligned_factor),
                        "aligned_returns_count": len(aligned_returns)
                    },
                    suggestions=[
                        "检查因子计算是否正确",
                        "确认数据中没有过多的NaN值"
                    ]
                )
            )]
        
        aligned_factor = aligned_factor.loc[final_common_index]
        aligned_returns = aligned_returns.loc[final_common_index]
        
        evaluator = FactorEvaluator(aligned_factor, aligned_returns)
        ic_result = evaluator.calculate_ic(method=method)
        
        # 优化的响应格式
        ic_mean = ic_result.get('ic_mean', 0)
        ic_std = ic_result.get('ic_std', 0)
        icir = ic_result.get('icir', 0)
        ic_positive_ratio = ic_result.get('ic_positive_ratio', 0)
        
        # 判断因子质量
        abs_ic = abs(ic_mean)
        if abs_ic > 0.10:
            quality = "非常强"
            recommendation = "强烈推荐使用此因子"
        elif abs_ic > 0.08:
            quality = "强"
            recommendation = "推荐使用此因子"
        elif abs_ic > 0.05:
            quality = "较强"
            recommendation = "可以使用此因子"
        elif abs_ic > 0.03:
            quality = "有效"
            recommendation = "谨慎使用，建议与其他因子组合"
        else:
            quality = "无效"
            recommendation = "不推荐使用，预测能力不足"
        
        # 生成评估报告
        report_info = None
        if report_path:
            import os
            from pathlib import Path
            from datetime import datetime
            
            report_path_obj = Path(report_path)
            # 确保目录存在
            report_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # 生成Markdown格式的评估报告
                report_content = f"""# 因子评估报告

## 基本信息
- **因子名称**: {factor_name}
- **数据源**: {data_id}
- **评估方法**: {method}
- **评估时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## IC指标
- **IC均值**: {ic_mean:.4f}
- **IC标准差**: {ic_std:.4f}
- **ICIR (信息比率)**: {icir:.4f}
- **IC正值占比**: {ic_positive_ratio*100:.2f}%

## 因子质量评级
- **质量等级**: {quality}
- **推荐建议**: {recommendation}

## 指标解读
### IC均值 ({ic_mean:.4f})
- 因子与未来收益率的相关性均值
- 绝对值越大，预测能力越强
- 当前水平: {'优秀' if abs_ic > 0.08 else '良好' if abs_ic > 0.05 else '一般' if abs_ic > 0.03 else '较差'}

### ICIR ({icir:.4f})
- IC均值与IC标准差的比值，衡量因子稳定性
- 大于1.0为稳定，0.5-1.0为较稳定，小于0.5为不稳定
- 当前稳定性: {'稳定' if icir > 1.0 else '较稳定' if icir > 0.5 else '不稳定'}

### IC正值占比 ({ic_positive_ratio*100:.2f}%)
- IC值为正的时间段占比
- 大于60%说明因子方向性较好
- 当前方向性: {'优秀' if ic_positive_ratio > 0.6 else '良好' if ic_positive_ratio > 0.5 else '一般'}

## 预测方向
- **因子类型**: {'正向因子（因子值越大，收益越高）' if ic_mean > 0 else '反向因子（因子值越大，收益越低）'}
- **预测能力**: 每单位因子变化对应约{abs(ic_mean)*100:.2f}%的收益率变化

## 使用建议
{recommendation}

### 后续步骤
"""
                if abs_ic > 0.03:
                    report_content += """
1. ✅ 因子有效，可以使用
2. 建议与其他因子组合使用
3. 可以进行模型训练：使用train_ml_model等工具
4. 定期监控因子IC的变化
"""
                else:
                    report_content += """
1. ⚠️ 因子预测能力不足
2. 建议尝试其他因子类型
3. 或调整因子参数（如period）
4. 可使用calculate_factor生成其他因子
"""
                
                # 保存报告
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                file_size = os.path.getsize(report_path) / 1024  # KB
                report_info = {
                    "generated": True,
                    "path": str(report_path_obj.absolute()),
                    "size_kb": round(file_size, 2),
                    "format": "markdown"
                }
            except Exception as e:
                report_info = {
                    "generated": False,
                    "error": f"报告生成失败: {str(e)}"
                }
        
        result = {
            "status": "success",
            "message": f"✅ 因子 '{factor_name}' IC评估完成",
            "ic_metrics": {
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": icir,
                "ic_positive_ratio": ic_positive_ratio
            },
            "factor_quality": {
                "rating": quality,
                "ic_mean": f"{ic_mean:.4f}",
                "icir": f"{icir:.4f}",
                "recommendation": recommendation
            },
            "interpretation": {
                "direction": "正向因子（因子值越大，收益越高）" if ic_mean > 0 else "反向因子（因子值越大，收益越低）",
                "stability": "稳定" if icir > 1.0 else "较稳定" if icir > 0.5 else "不稳定",
                "predictive_power": f"每单位因子变化对应约{abs(ic_mean)*100:.2f}%的收益率变化"
            },
            "next_steps": [
                {
                    "step": 1,
                    "action": "使用有效因子" if abs_ic > 0.03 else "尝试其他因子",
                    "reason": recommendation
                },
                {
                    "step": 2,
                    "action": "生成更多因子或训练模型",
                    "tools": ["generate_alpha158", "train_ml_model"] if abs_ic > 0.05 else ["calculate_factor"],
                    "reason": "因子有效，可继续建模" if abs_ic > 0.05 else "当前因子效果不佳，建议尝试其他因子"
                }
            ],
            "tips": [
                f"💡 IC均值: {ic_mean:.4f} ({'强' if abs_ic > 0.08 else '较强' if abs_ic > 0.05 else '弱'})",
                f"💡 ICIR: {icir:.4f} ({'稳定' if icir > 1.0 else '较稳定' if icir > 0.5 else '不稳定'})",
                f"💡 {quality}因子，{recommendation}",
                f"💡 使用report_path参数生成详细评估报告" if not report_path else f"💡 评估报告已生成: {report_path}"
            ]
        }
        
        # 添加报告信息
        if report_info:
            result["report_info"] = report_info
        
        return [types.TextContent(type="text", text=serialize_response(result))]
        
    except KeyError as e:
        return [types.TextContent(
            type="text",
            text=MCPError.format_error(
                error_code=MCPError.INVALID_PARAMETER,
                message=f"数据缺少close列: {str(e)}",
                details={"data_id": data_id, "missing_column": "close"},
                suggestions=[
                    "确保数据包含close列",
                    "检查数据加载是否正确"
                ]
            )
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=MCPError.format_error(
                error_code=MCPError.COMPUTATION_ERROR,
                message=f"IC评估失败: {str(e)}",
                details={
                    "factor_name": factor_name,
                    "data_id": data_id,
                    "method": method,
                    "exception_type": type(e).__name__
                },
                suggestions=[
                    "检查因子和数据的时间范围是否匹配",
                    "确认数据中有足够的样本",
                    "尝试使用不同的method参数"
                ]
            )
        )]


async def handle_apply_processor_chain(args: Dict[str, Any]) -> List[types.TextContent]:
    """应用数据处理器链"""
    data_id = args["data_id"]
    result_id = args["result_id"]
    processors_config = args["processors"]
    export_path = args.get("export_path", None)
    export_format = args.get("export_format", "csv")  # csv 或 json
    
    # 验证数据ID
    error = validate_data_id(data_id, factor_store)  # 通常处理因子数据
    if error:
        # 如果不在factor_store，检查data_store
        error = validate_data_id(data_id, data_store)
        if error:
            return [types.TextContent(type="text", text=error)]
        data = data_store[data_id].copy()
    else:
        data = factor_store[data_id].copy()
    
    try:
        # 🆕 智能检测数据类型：单商品 vs 多商品
        symbol_count = 1
        if isinstance(data.index, pd.MultiIndex):
            symbol_count = len(data.index.get_level_values(1).unique())
        elif 'symbol' in data.columns:
            symbol_count = data['symbol'].nunique()
        
        data_type = "单商品" if symbol_count == 1 else f"多商品({symbol_count}个)"
        auto_adjusted = []  # 记录自动调整的处理器
        
        # 构建处理器链
        processors = []
        for proc_config in processors_config:
            proc_name = proc_config["name"]
            proc_params = proc_config.get("params", {})
            
            # 🆕 智能处理CSZScoreNorm：单商品自动切换为ZScoreNorm
            original_proc_name = proc_name
            if proc_name == "CSZScoreNorm" and symbol_count == 1:
                proc_name = "ZScoreNorm"
                auto_adjusted.append({
                    "original": "CSZScoreNorm",
                    "adjusted_to": "ZScoreNorm",
                    "reason": f"检测到单商品数据，CSZScoreNorm会导致100% NaN，自动切换为ZScoreNorm"
                })
            
            # 根据名称创建处理器
            if proc_name == "CSZScoreNorm":
                proc = CSZScoreNorm(**proc_params)
            elif proc_name == "ZScoreNorm":
                from quantanalyzer.data.processor import ZScoreNorm
                proc = ZScoreNorm(**proc_params)
            elif proc_name == "CSZFillna":
                proc = CSZFillna(**proc_params)
            elif proc_name == "ProcessInf":
                proc = ProcessInf()
            elif proc_name == "ZScoreNorm":
                proc = ZScoreNorm(**proc_params)
            elif proc_name == "RobustZScoreNorm":
                proc = RobustZScoreNorm(**proc_params)
            elif proc_name == "CSRankNorm":
                proc = CSRankNorm(**proc_params)
            elif proc_name == "MinMaxNorm":
                proc = MinMaxNorm(**proc_params)
            else:
                return [types.TextContent(
                    type="text",
                    text=MCPError.format_error(
                        error_code=MCPError.INVALID_PARAMETER,
                        message=f"不支持的处理器: {proc_name}",
                        details={"processor": proc_name},
                        suggestions=[
                            "支持的处理器: CSZScoreNorm, CSZFillna, ProcessInf, ZScoreNorm, RobustZScoreNorm, CSRankNorm, MinMaxNorm"
                        ]
                    )
                )]
            
            processors.append(proc)
        
        # 应用处理器链
        chain = ProcessorChain(processors)
        processed_data = chain.fit_transform(data)
        
        # 保存处理后的数据
        factor_store[result_id] = processed_data
        
        # 导出处理后的数据
        export_info = None
        if export_path:
            import os
            from pathlib import Path
            from datetime import datetime
            
            export_path_obj = Path(export_path)
            # 确保目录存在
            export_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                if export_format.lower() == "csv":
                    processed_data.to_csv(export_path, encoding='utf-8')
                elif export_format.lower() == "json":
                    processed_data.to_json(export_path, orient='records', indent=2)
                else:
                    export_format = "csv"  # 默认使用CSV
                    processed_data.to_csv(export_path, encoding='utf-8')
                
                file_size = os.path.getsize(export_path) / (1024 * 1024)
                export_info = {
                    "exported": True,
                    "path": str(export_path_obj.absolute()),
                    "format": export_format,
                    "size_mb": round(file_size, 2)
                }
            except Exception as e:
                export_info = {
                    "exported": False,
                    "error": f"导出失败: {str(e)}"
                }
        
        # 计算数据质量
        null_count = int(processed_data.isna().sum().sum())
        total_values = processed_data.shape[0] * processed_data.shape[1]
        null_rate = null_count / total_values if total_values > 0 else 0
        
        result = {
            "status": "success",
            "message": f"✅ 数据处理完成并存储为 '{result_id}'" + (f" (智能优化: {len(auto_adjusted)}个处理器)" if auto_adjusted else ""),
            "data_info": {
                "data_type": data_type,
                "symbol_count": symbol_count
            },
            "processing_info": {
                "input_id": data_id,
                "output_id": result_id,
                "processors_requested": [p["name"] for p in processors_config],
                "processors_applied": [p.name if hasattr(p, 'name') else type(p).__name__ for p in processors],
                "shape": list(processed_data.shape),
                "auto_adjustments": auto_adjusted if auto_adjusted else None
            },
            "data_quality": {
                "null_count": null_count,
                "null_rate": f"{null_rate * 100:.2f}%",
                "quality_score": "优秀" if null_rate < 0.01 else "良好" if null_rate < 0.05 else "一般"
            },
            "next_steps": [
                {
                    "step": 1,
                    "action": "训练机器学习模型",
                    "tools": ["train_ml_model"],
                    "params_example": {
                        "data_id": result_id,
                        "model_id": f"model_{result_id}"
                    }
                }
            ]
        }
        
        return [types.TextContent(type="text", text=serialize_response(result))]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=MCPError.format_error(
                error_code=MCPError.COMPUTATION_ERROR,
                message=f"数据处理失败: {str(e)}",
                details={
                    "data_id": data_id,
                    "processors": processors_config,
                    "exception_type": type(e).__name__
                },
                suggestions=[
                    "检查处理器参数是否正确",
                    "确认数据格式符合要求",
                    "尝试减少处理器数量"
                ]
            )
        )]


async def handle_merge_factor_data(args: Dict[str, Any]) -> List[types.TextContent]:
    """
    合并因子数据和价格数据
    将Alpha158等因子数据与原始价格数据合并，以便用于模型训练
    """
    factor_data_id = args["factor_data_id"]
    price_data_id = args["price_data_id"]
    result_id = args["result_id"]
    export_path = args.get("export_path", None)
    export_format = args.get("export_format", "csv")
    
    # 验证因子数据ID
    error = validate_data_id(factor_data_id, factor_store)
    if error:
        return [types.TextContent(type="text", text=error)]
    
    # 验证价格数据ID
    error = validate_data_id(price_data_id, data_store)
    if error:
        return [types.TextContent(type="text", text=error)]
    
    try:
        factor_data = factor_store[factor_data_id]
        price_data = data_store[price_data_id]
        
        # 确保索引对齐
        if not factor_data.index.equals(price_data.index):
            # 使用内连接确保索引匹配
            common_index = factor_data.index.intersection(price_data.index)
            if len(common_index) == 0:
                return [types.TextContent(
                    type="text",
                    text=MCPError.format_error(
                        error_code=MCPError.INVALID_PARAMETER,
                        message="因子数据和价格数据的索引没有重叠",
                        details={
                            "factor_data_id": factor_data_id,
                            "price_data_id": price_data_id,
                            "factor_index_range": f"{factor_data.index[0]} ~ {factor_data.index[-1]}",
                            "price_index_range": f"{price_data.index[0]} ~ {price_data.index[-1]}"
                        },
                        suggestions=[
                            "确认因子数据和价格数据来源于同一个原始数据",
                            "检查数据的时间范围是否匹配"
                        ]
                    )
                )]
            
            factor_data = factor_data.loc[common_index]
            price_data = price_data.loc[common_index]
        
        # 合并数据：因子列 + close列（用于生成标签）
        # 修复：确保正确的数据合并方式
        # 只取价格数据中的close列，并确保索引对齐
        close_data = price_data[['close']]
        
        # 使用内连接确保索引完全匹配
        common_index = factor_data.index.intersection(close_data.index)
        if len(common_index) == 0:
            return [types.TextContent(
                type="text",
                text=MCPError.format_error(
                    error_code=MCPError.INVALID_PARAMETER,
                    message="因子数据和价格数据的索引没有重叠",
                    details={
                        "factor_data_id": factor_data_id,
                        "price_data_id": price_data_id,
                        "factor_index_range": f"{factor_data.index[0]} ~ {factor_data.index[-1]}",
                        "price_index_range": f"{close_data.index[0]} ~ {close_data.index[-1]}"
                    },
                    suggestions=[
                        "确认因子数据和价格数据来源于同一个原始数据",
                        "检查数据的时间范围是否匹配"
                    ]
                )
            )]
        
        # 使用共同索引
        aligned_factor = factor_data.loc[common_index]
        aligned_close = close_data.loc[common_index]
        
        # 合并数据
        merged_data = pd.concat([
            aligned_factor,
            aligned_close
        ], axis=1)
        
        # 保存合并后的数据
        factor_store[result_id] = merged_data
        
        # 导出合并后的数据
        export_info = None
        if export_path:
            import os
            from pathlib import Path
            
            export_path_obj = Path(export_path)
            export_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                if export_format.lower() == "csv":
                    merged_data.to_csv(export_path, encoding='utf-8')
                elif export_format.lower() == "json":
                    merged_data.to_json(export_path, orient='records', indent=2)
                else:
                    export_format = "csv"
                    merged_data.to_csv(export_path, encoding='utf-8')
                
                file_size = os.path.getsize(export_path) / (1024 * 1024)
                export_info = {
                    "exported": True,
                    "path": str(export_path_obj.absolute()),
                    "format": export_format,
                    "size_mb": round(file_size, 2)
                }
            except Exception as e:
                export_info = {
                    "exported": False,
                    "error": f"导出失败: {str(e)}"
                }
        
        # 构建响应
        result = {
            "status": "success",
            "message": f"✅ 数据合并完成并存储为 '{result_id}'",
            "merge_info": {
                "result_id": result_id,
                "factor_data_id": factor_data_id,
                "price_data_id": price_data_id,
                "factor_count": len(factor_data.columns),
                "total_columns": len(merged_data.columns),
                "row_count": len(merged_data),
                "columns": list(merged_data.columns)
            },
            "data_quality": {
                "null_count": int(merged_data.isna().sum().sum()),
                "null_rate": f"{(merged_data.isna().sum().sum() / (merged_data.shape[0] * merged_data.shape[1]) * 100):.2f}%"
            },
            "next_steps": [
                {
                    "step": 1,
                    "action": "训练机器学习模型",
                    "tool": "train_ml_model",
                    "params_example": {
                        "data_id": result_id,
                        "model_id": f"model_{result_id}",
                        "train_start": "2023-01-01",
                        "train_end": "2023-06-30",
                        "test_start": "2023-07-01",
                        "test_end": "2023-12-31"
                    },
                    "reason": "合并后的数据包含因子和close列，可以直接用于训练"
                }
            ],
            "tips": [
                f"💡 合并了{len(factor_data.columns)}个因子 + 1个close列",
                f"💡 数据行数: {len(merged_data)}",
                f"💡 数据已准备好用于train_ml_model训练",
                f"💡 使用export_path参数可导出合并数据" if not export_path else f"💡 数据已导出到: {export_path}"
            ]
        }
        
        if export_info:
            result["export_info"] = export_info
        
        return [types.TextContent(type="text", text=serialize_response(result))]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=MCPError.format_error(
                error_code=MCPError.COMPUTATION_ERROR,
                message=f"数据合并失败: {str(e)}",
                details={
                    "factor_data_id": factor_data_id,
                    "price_data_id": price_data_id,
                    "exception_type": type(e).__name__
                },
                suggestions=[
                    "确认因子数据和价格数据的索引格式一致",
                    "检查两个数据集的时间范围是否重叠",
                    "尝试使用list_factors查看数据详情"
                ]
            )
        )]


async def handle_list_factors(args: Dict[str, Any]) -> List[types.TextContent]:
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


async def handle_train_ml_model(args: Dict[str, Any]) -> List[types.TextContent]:
    """训练机器学习模型（LightGBM/XGBoost/sklearn）"""
    data_id = args["data_id"]
    model_id = args["model_id"]
    model_type = args.get("model_type", "lightgbm")
    train_start = args["train_start"]
    train_end = args["train_end"]
    test_start = args["test_start"]
    test_end = args["test_end"]
    params = args.get("params", None)
    
    # 验证数据ID
    error = validate_data_id(data_id, factor_store)
    if error:
        error = validate_data_id(data_id, data_store)
        if error:
            return [types.TextContent(type="text", text=error)]
        data = data_store[data_id]
    else:
        data = factor_store[data_id]
    
    try:
        from quantanalyzer.model.trainer import ModelTrainer
        
        # 创建标签（下一日收益率）
        if 'close' in data.columns:
            # 使用close列创建标签
            labels = data['close'].groupby(level=1).pct_change().shift(-1)
            # 特征数据：排除close列
            features = data.drop('close', axis=1)
        else:
            # 如果是纯因子数据，需要从原始数据获取close列
            # 检查是否有对应的原始数据
            original_data_id = None
            for key in data_store.keys():
                if key in data_id or data_id in key:
                    original_data_id = key
                    break
            
            if original_data_id and 'close' in data_store[original_data_id].columns:
                # 从原始数据获取close列
                original_data = data_store[original_data_id]
                # 确保索引对齐
                common_index = data.index.intersection(original_data.index)
                if len(common_index) > 0:
                    labels = original_data.loc[common_index]['close'].groupby(level=1).pct_change().shift(-1)
                    features = data.loc[common_index]
                else:
                    return [types.TextContent(
                        type="text",
                        text=MCPError.format_error(
                            error_code=MCPError.INVALID_PARAMETER,
                            message="因子数据和原始价格数据的索引不匹配",
                            suggestions=[
                                "使用merge_factor_data工具合并因子和价格数据",
                                "确保因子数据和价格数据来源于同一原始数据"
                            ]
                        )
                    )]
            else:
                return [types.TextContent(
                    type="text",
                    text=MCPError.format_error(
                        error_code=MCPError.INVALID_PARAMETER,
                        message="因子数据需要提供对应的价格数据来生成标签",
                        suggestions=[
                            "使用merge_factor_data工具合并因子和价格数据",
                            "或使用包含close列的原始数据ID"
                        ]
                    )
                )]
        
        # 创建训练器
        trainer = ModelTrainer(model_type=model_type)
        
        # 修复：改进时间范围筛选逻辑
        # 将字符串日期转换为datetime对象
        from datetime import datetime
        train_start_dt = pd.to_datetime(train_start)
        train_end_dt = pd.to_datetime(train_end)
        test_start_dt = pd.to_datetime(test_start)
        test_end_dt = pd.to_datetime(test_end)
        
        # 获取时间索引
        time_index = features.index.get_level_values(0)
        
        # 筛选训练集和测试集
        train_mask = (time_index >= train_start_dt) & (time_index <= train_end_dt)
        test_mask = (time_index >= test_start_dt) & (time_index <= test_end_dt)
        
        X_train = features[train_mask]
        y_train = labels[train_mask]
        X_test = features[test_mask]
        y_test = labels[test_mask]
        
        # 检查数据集是否为空
        if len(X_train) == 0 or len(X_test) == 0:
            return [types.TextContent(
                type="text",
                text=MCPError.format_error(
                    error_code=MCPError.COMPUTATION_ERROR,
                    message="训练或测试数据集为空",
                    details={
                        "train_samples": len(X_train),
                        "test_samples": len(X_test),
                        "train_period": f"{train_start} ~ {train_end}",
                        "test_period": f"{test_start} ~ {test_end}",
                        "data_time_range": f"{time_index.min()} ~ {time_index.max()}",
                        "train_mask_count": train_mask.sum(),
                        "test_mask_count": test_mask.sum()
                    },
                    suggestions=[
                        "检查时间范围参数是否在数据时间范围内",
                        "调整train_start, train_end, test_start, test_end参数",
                        "使用list_factors查看数据的时间范围",
                        "确保时间格式为YYYY-MM-DD"
                    ]
                )
            )]
        
        # 训练模型
        trainer.train(X_train, y_train, X_test, y_test, params)
        
        # 预测
        train_pred = trainer.predict(X_train)
        test_pred = trainer.predict(X_test)
        
        # 计算评估指标
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        
        train_mse = mean_squared_error(y_train, train_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        train_r2 = r2_score(y_train, train_pred)
        
        test_mse = mean_squared_error(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # 计算IC (Information Coefficient)
        train_ic = np.corrcoef(y_train, train_pred)[0, 1]
        test_ic = np.corrcoef(y_test, test_pred)[0, 1]
        
        # 保存模型
        model_store[model_id] = {
            'trainer': trainer,
            'model_type': model_type,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': X_train.shape[1]
        }
        
        # 构建响应
        result = {
            "status": "success",
            "message": f"✅ {model_type.upper()}模型训练完成并存储为 '{model_id}'",
            "model_info": {
                "model_id": model_id,
                "model_type": model_type,
                "train_period": f"{train_start} ~ {train_end}",
                "test_period": f"{test_start} ~ {test_end}",
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_count": X_train.shape[1]
            },
            "performance": {
                "train": {
                    "mse": float(train_mse),
                    "mae": float(train_mae),
                    "r2": float(train_r2),
                    "ic": float(train_ic)
                },
                "test": {
                    "mse": float(test_mse),
                    "mae": float(test_mae),
                    "r2": float(test_r2),
                    "ic": float(test_ic)
                }
            },
            "feature_importance": {
                "top_10": trainer.feature_importance.head(10).to_dict() if trainer.feature_importance is not None else None
            },
            "next_steps": [
                {
                    "step": 1,
                    "action": "使用模型预测",
                    "tool": "predict_ml_model",
                    "params_example": {
                        "model_id": model_id,
                        "data_id": data_id
                    }
                }
            ],
            "tips": [
                f"💡 模型类型: {model_type}",
                f"💡 训练集R²: {train_r2:.4f}, 测试集R²: {test_r2:.4f}",
                f"💡 测试集IC: {test_ic:.4f} ({'有效' if abs(test_ic) > 0.03 else '较弱'})",
                "💡 使用predict_ml_model工具进行预测"
            ]
        }
        
        return [types.TextContent(type="text", text=serialize_response(result))]
        
    except ImportError as e:
        return [types.TextContent(
            type="text",
            text=MCPError.format_error(
                error_code=MCPError.COMPUTATION_ERROR,
                message=f"缺少必需的库: {str(e)}",
                suggestions=[
                    f"安装{model_type}: pip install {model_type}",
                    "或使用其他模型类型"
                ]
            )
        )]
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=MCPError.format_error(
                error_code=MCPError.COMPUTATION_ERROR,
                message=f"模型训练失败: {str(e)}",
                details={
                    "model_type": model_type,
                    "exception_type": type(e).__name__
                },
                suggestions=[
                    "检查数据格式是否正确",
                    "确认时间范围参数有效",
                    "尝试调整模型参数"
                ]
            )
        )]


async def handle_predict_ml_model(args: Dict[str, Any]) -> List[types.TextContent]:
    """使用训练好的机器学习模型进行预测"""
    model_id = args["model_id"]
    data_id = args["data_id"]
    export_path = args.get("export_path", None)
    
    # 验证模型ID
    if model_id not in model_store:
        return [types.TextContent(
            type="text",
            text=MCPError.format_error(
                error_code=MCPError.INVALID_PARAMETER,
                message=f"模型ID不存在: {model_id}",
                suggestions=[
                    "使用train_ml_model训练模型",
                    "检查model_id是否正确"
                ]
            )
        )]
    
    # 验证数据ID
    error = validate_data_id(data_id, factor_store)
    if error:
        error = validate_data_id(data_id, data_store)
        if error:
            return [types.TextContent(type="text", text=error)]
        data = data_store[data_id]
    else:
        data = factor_store[data_id]
    
    try:
        model_info = model_store[model_id]
        trainer = model_info['trainer']
        
        # 修复：确保预测数据特征与训练时一致
        # 如果数据包含close列，需要排除它（因为训练时只使用因子特征）
        if 'close' in data.columns:
            # 使用与训练时相同的特征列
            prediction_features = data.drop('close', axis=1)
        else:
            prediction_features = data
        
        # 预测
        predictions = trainer.predict(prediction_features)
        
        # 导出预测结果
        export_info = None
        if export_path:
            import os
            from pathlib import Path
            
            export_path_obj = Path(export_path)
            export_path_obj.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                predictions.to_csv(export_path, encoding='utf-8')
                file_size = os.path.getsize(export_path) / (1024 * 1024)
                export_info = {
                    "exported": True,
                    "path": str(export_path_obj.absolute()),
                    "size_mb": round(file_size, 2),
                    "prediction_count": len(predictions)
                }
            except Exception as e:
                export_info = {
                    "exported": False,
                    "error": f"导出失败: {str(e)}"
                }
        
        # 构建响应 - 修复数据类型问题
        result = {
            "status": "success",
            "message": f"✅ 使用模型 '{model_id}' 预测完成",
            "prediction_info": {
                "model_id": model_id,
                "model_type": model_info['model_type'],
                "data_id": data_id,
                "prediction_count": len(predictions),
                "statistics": {
                    "mean": float(predictions.mean()),
                    "std": float(predictions.std()),
                    "min": float(predictions.min()),
                    "max": float(predictions.max())
                }
            },
            "preview": {
                "head_5": _convert_index_to_string(predictions.head(5).to_dict()),
                "tail_5": _convert_index_to_string(predictions.tail(5).to_dict())
            },
            "tips": [
                f"💡 预测数量: {len(predictions)}条",
                f"💡 预测均值: {predictions.mean():.6f}",
                "💡 预测结果已保存在内存中" if not export_path else f"💡 已导出到: {export_path}"
            ]
        }
        
        if export_info:
            result["export_info"] = export_info
        
        return [types.TextContent(type="text", text=serialize_response(result))]
        
    except Exception as e:
        return [types.TextContent(
            type="text",
            text=MCPError.format_error(
                error_code=MCPError.COMPUTATION_ERROR,
                message=f"预测失败: {str(e)}",
                details={
                    "model_id": model_id,
                    "data_id": data_id,
                    "exception_type": type(e).__name__
                },
                suggestions=[
                    "检查数据格式是否与训练时一致",
                    "确认模型已正确训练"
                ]
            )
        )]

