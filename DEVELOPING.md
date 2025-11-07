# 开发指南

本文档为希望参与 aigroup-quant-mcp 项目开发的开发者提供指导。

## 项目结构

```
aigroup-quant-mcp/
├── quantanalyzer/          # 核心代码
│   ├── data/              # 数据处理模块
│   ├── factor/            # 因子计算模块
│   ├── model/             # 模型训练模块
│   ├── backtest/          # 回测模块
│   ├── mcp/               # MCP协议实现
│   ├── logger.py          # 日志模块
│   ├── config.py          # 配置管理模块
│   ├── utils.py           # 工具函数模块
│   └── monitor.py         # 性能监控模块
├── tests/                 # 测试代码
├── examples/              # 使用示例
└── docs/                  # 文档
```

## 开发环境设置

1. 克隆项目：
```bash
git clone <repository-url>
cd aigroup-quant-mcp
```

2. 安装依赖：
```bash
pip install -e .[full,dev,viz]
```

## 代码规范

- 遵循 PEP 8 代码规范
- 使用类型提示
- 编写文档字符串
- 保持函数简洁，单一职责

## 测试

### 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试文件
python -m pytest tests/test_factor.py

# 详细输出
python -m pytest -v

# 运行集成测试
python -m pytest tests/test_integration.py
```

### 编写测试

1. 为新功能编写单元测试
2. 测试应覆盖正常情况和边界情况
3. 使用 `unittest` 框架
4. 测试文件应与被测试模块对应

## 日志记录

使用统一的日志模块：

```python
from ..logger import get_logger

logger = get_logger(__name__)

# 在关键操作中添加日志
logger.debug("调试信息")
logger.info("一般信息")
logger.warning("警告信息")
logger.error("错误信息")
```

## 配置管理

项目支持多种配置方式：

1. 默认配置
2. 环境变量
3. 配置文件

### 使用配置

```python
from quantanalyzer.config import get_config

config = get_config()
chunk_size = config.get('chunk_size', 10000)
```

### 配置优先级

1. 配置文件（最高优先级）
2. 环境变量
3. 默认值（最低优先级）

### 环境变量示例

```bash
export CHUNK_SIZE=5000
export LOG_LEVEL=DEBUG
```

### 配置文件示例

参考 [config.example.json](file:///d:/aigroup-quant-mcp/config.example.json)

## 性能优化

1. 避免不必要的数据复制
2. 使用向量化操作替代循环
3. 对大数据集使用分块处理
4. 及时释放不需要的内存
5. 使用并行处理加速计算

### 并行处理

项目提供了并行处理工具：

```python
from quantanalyzer.utils import parallelize_dataframe_operation

# 并行处理DataFrame操作
result = parallelize_dataframe_operation(
    df, 
    func, 
    groupby_level=0,
    max_workers=4
)
```

### 缓存机制

模型训练器支持缓存：

```python
trainer = ModelTrainer(model_type='lightgbm', model_id='my_model')
trainer.train(X_train, y_train, use_cache=True)
```

### 性能监控

项目提供了性能监控工具：

```python
from quantanalyzer.monitor import profile_function

@profile_function(track_memory=True)
def my_function():
    # 你的代码
    pass
```

## 添加新功能

### 添加新的因子

1. 在 `quantanalyzer/factor/library.py` 中添加因子计算函数
2. 编写相应的测试
3. 更新文档

### 添加新的模型

1. 在 `quantanalyzer/model/trainer.py` 中添加模型支持
2. 编写相应的测试
3. 更新文档

## 文档

- 更新 README.md 以反映新功能
- 为新功能编写专门的文档
- 保持示例代码的更新

## 提交代码

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 发起 Pull Request

## 发布流程

1. 更新版本号
2. 更新 CHANGELOG.md
3. 创建发布标签
4. 发布到 PyPI