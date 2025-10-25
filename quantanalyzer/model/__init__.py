"""
Model module for quantitative analysis
"""

from .trainer import ModelTrainer

__all__ = ['ModelTrainer']

# 深度学习模型是可选的，需要torch依赖
try:
    from .deep_models import LSTMModel, GRUModel, TransformerModel
    __all__.extend(['LSTMModel', 'GRUModel', 'TransformerModel'])
    _DL_AVAILABLE = True
except ImportError:
    _DL_AVAILABLE = False
    # 提供友好的错误提示
    class _DLModelPlaceholder:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "深度学习模型需要安装torch。\n"
                "请运行: pip install aigroup-quant-mcp[dl]\n"
                "或安装完整版本: pip install aigroup-quant-mcp[full]"
            )
    
    LSTMModel = _DLModelPlaceholder
    GRUModel = _DLModelPlaceholder
    TransformerModel = _DLModelPlaceholder