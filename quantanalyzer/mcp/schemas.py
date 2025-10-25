
"""
MCP工具Schema定义
所有工具的输入schema和description
"""

from mcp import types


def get_preprocess_data_schema():
    """preprocess_data工具Schema"""
    return types.Tool(
        name="preprocess_data",
        description="""
[🧹 数据预处理 | 步骤1/6] 从CSV文件加载并清洗股票行情数据

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 功能概述
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这是使用aigroup-quant-mcp服务的第一步，将CSV格式的股票历史数据加载到内存并进行智能清洗。
系统会自动处理缺失值、异常值，并可选择性地将清洗后的数据导出到本地，便于后续深入分析。
为后续的因子计算、模型训练等操作提供高质量的数据基础。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 适用场景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 适合:
  - 量化策略回测
  - 因子挖掘研究
  - 机器学习模型训练
  - 历史数据分析

⚠️ 不适合:
  - 实时行情数据（需要流式数据接口）
  - 非结构化数据（仅支持CSV格式）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 数据格式要求
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CSV文件必须包含以下列（不区分大小写）：
- datetime: 日期时间（格式: YYYY-MM-DD 或 YYYY-MM-DD HH:MM:SS）
- symbol: 股票代码（如: 000001.SZ, 600000.SH）
- open: 开盘价
- high: 最高价
- low: 最低价
- close: 收盘价
- volume: 成交量

可选列：
- vwap: 成交量加权平均价
- amount: 成交额

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎬 典型工作流
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

preprocess_data 👈 当前步骤（自动清洗+导出）
    ↓
generate_alpha158 或 calculate_factor
    ↓
apply_processor_chain (可选，已清洗可跳过)
    ↓
train_lstm_model / train_gru_model / train_transformer_model
    ↓
predict_with_model

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ 性能建议
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 数据量 < 10万行: 直接加载
- 数据量 10-100万行: 正常，可能需要10-30秒
- 数据量 > 100万行: 考虑分批处理或数据采样

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱️ 预计耗时
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 1000条数据: < 1秒
- 1万条数据: 1-3秒
- 10万条数据: 5-15秒
- 100万条数据: 30-60秒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 注意事项
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 文件路径必须是绝对路径或相对于当前工作目录的路径
2. CSV文件编码建议使用UTF-8
3. 数据会完全加载到内存，注意内存使用
4. data_id必须唯一，重复ID会覆盖之前的数据
5. 加载后的数据会自动转换为MultiIndex格式(datetime, symbol)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": """
CSV文件的完整路径

📌 路径格式:
- Windows: "D:/data/stock_data.csv" 或 "D:\\data\\stock_data.csv"
- Linux/Mac: "/home/user/data/stock_data.csv"
- 相对路径: "./data/stock_data.csv" (相对于当前工作目录)

📌 文件要求:
- 必须是CSV格式（.csv扩展名）
- 文件必须存在且可读
- 建议使用UTF-8编码
- 文件大小建议 < 500MB

❌ 常见错误:
- 路径不存在或拼写错误
- 使用了反斜杠但未转义（Windows）
- 文件权限不足
- 文件正在被其他程序占用

💡 最佳实践:
- 使用绝对路径避免路径问题
- 文件名使用英文和数字，避免特殊字符
- 大文件建议先验证格式再加载
""",
                    "examples": [
                        "D:/data/stock_data_2023.csv",
                        "./data/training_set.csv",
                        "/home/user/quant/backtest_data.csv"
                    ]
                },
                "data_id": {
                    "type": "string",
                    "description": """
数据标识ID - 为加载的数据指定一个唯一标识符

📌 命名规则:
- 只能包含字母、数字、下划线、连字符
- 建议使用有意义的名称，便于后续引用
- 区分大小写
- 长度建议 3-50 个字符

📌 命名建议:
- 包含数据类型: stock_data, index_data
- 包含时间范围: data_2023, data_2020_2023
- 包含用途: training_set, test_set, backtest_data
- 组合命名: stock_training_2023

❌ 常见错误:
- 使用特殊字符（如空格、@、#等）
- ID过于简单（如: data, test）
- 重复使用相同ID（会覆盖之前的数据）

💡 最佳实践:
- 使用项目前缀: project1_stock_data
- 包含版本号: data_v1, data_v2
- 使用时间戳: data_20230101
- 描述性命名: hs300_daily_2023

⚠️ 重要提示:
- 此ID将用于后续所有工具调用
- 如果忘记ID，使用 list_factors 工具查看
- 相同ID会覆盖之前加载的数据
""",
                    "examples": [
                        "stock_data_2023",
                        "training_set",
                        "hs300_backtest",
                        "project1_data_v1"
                    ]
                },
                "auto_clean": {
                    "type": "boolean",
                    "default": True,
                    "description": """
是否自动清洗数据（默认开启）

📌 自动清洗流程（参考Qlib标准）:

⚠️ 重要：预处理阶段只清洗OHLCV原始数据的异常值和缺失值
1. ProcessInf() - 处理无穷值（用截面均值替换）
2. CSZFillna() - 截面填充缺失值（用截面均值）

⚠️ 标准化应该在因子生成后进行：
- 生成Alpha158因子后，使用apply_processor_chain工具
- 对因子进行CSZScoreNorm标准化（而非原始OHLCV数据）
- 标准化的是因子，不是价格/成交量数据

📌 Qlib清洗策略优势:
- ProcessInf: 用同一时间截面的均值替换inf/-inf，避免异常值
- CSZFillna: 用同一时间截面的均值填充NaN，比固定值更合理
- CSZScoreNorm: 应该用于因子标准化，不用于原始数据

📌 清洗效果:
- 妥善处理异常值（inf/-inf）
- 智能填充缺失值（截面均值）
- 统一数据量纲（多股票场景）
- 避免数据泄露（仅使用截面信息）
- 提高模型训练效果

💡 使用建议:
- 默认开启，遵循Qlib最佳实践
- 截面均值填充比固定值或前向填充更科学
- 如需自定义清洗流程，可设为false
- 清洗后的数据直接可用于模型训练

⚠️ 注意:
- 设为false时，原始数据可能包含缺失值和异常值
- 原始数据需要手动处理后才能用于模型
- 自动清洗使用Qlib标准流程，经过大量验证
- 单股票数据不进行标准化（截面标准化需要多个股票）
""",
                    "examples": [True, False]
                },
                "export_path": {
                    "type": "string",
                    "description": """
是否导出清洗后的数据到本地文件（可选）

📌 导出说明:
- 如果指定此参数，数据将导出到指定路径
- 如果不指定但开启了auto_clean，数据将自动导出到 ./exports/ 目录
- 导出文件格式为CSV，编码UTF-8
- 自动创建不存在的目录

📌 路径格式:
- 绝对路径: "D:/data/cleaned/stock_data.csv"
- 相对路径: "./exports/cleaned_data.csv"
- 默认路径: "./exports/{data_id}_cleaned_{timestamp}.csv"

📌 导出优势:
- 保存清洗后的数据，便于后续重复使用
- 避免重复清洗，节省计算时间
- 便于数据审查和质量检查
- 支持外部工具（如Excel）进一步分析
- 数据持久化，不依赖内存

💡 使用建议:
- 清洗后的数据建议导出保存
- 大数据集建议指定具体路径避免文件名冲突
- 导出路径建议使用项目相对路径便于管理

⚠️ 注意:
- 导出会覆盖同名文件
- 确保有足够的磁盘空间
- 大文件导出可能需要较长时间
""",
                    "examples": [
                        "./exports/cleaned_stock_data.csv",
                        "D:/data/processed/hs300_cleaned.csv",
                        "./data/train_set_cleaned.csv"
                    ]
                }
            },
            "required": ["file_path", "data_id"],
            "examples": [
                {
                    "name": "预处理2023年股票数据（自动清洗+自动导出）",
                    "description": "加载并清洗完整年度数据，自动导出到默认路径",
                    "input": {
                        "file_path": "D:/data/stock_data_2023.csv",
                        "data_id": "stock_2023",
                        "auto_clean": True
                    },
                    "expected_output": "数据成功预处理并自动清洗导出，返回数据摘要、清洗详情和导出路径"
                },
                {
                    "name": "预处理训练集数据（指定导出路径）",
                    "description": "加载并清洗训练数据，导出到指定位置",
                    "input": {
                        "file_path": "./data/training_set.csv",
                        "data_id": "training_data",
                        "export_path": "./exports/training_cleaned.csv"
                    },
                    "expected_output": "训练数据预处理完成，已清洗并导出到指定路径"
                },
                {
                    "name": "加载原始数据（不清洗不导出）",
                    "description": "仅加载原始数据用于自定义处理",
                    "input": {
                        "file_path": "/home/user/backtest/hs300_2020_2023.csv",
                        "data_id": "hs300_backtest",
                        "auto_clean": False
                    },
                    "expected_output": "原始数据加载完成，保留缺失值和原始数值，未导出"
                },
                {
                    "name": "预处理并导出大数据集",
                    "description": "处理大型数据集并导出以便重复使用",
                    "input": {
                        "file_path": "D:/bigdata/all_stocks_10years.csv",
                        "data_id": "large_dataset",
                        "auto_clean": True,
                        "export_path": "D:/processed/large_dataset_cleaned.csv"
                    },
                    "expected_output": "大数据集预处理完成，清洗后导出，后续可直接使用清洗后文件"
                }
            ]
        }
    )


def get_calculate_factor_schema():
    """calculate_factor工具Schema"""
    return types.Tool(
        name="calculate_factor",
        description="""
[🔬 单因子计算 | 步骤2/6] 计算单个量化因子

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 功能概述
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

计算单个技术指标或量化因子，用于股票价格预测和策略构建。
支持动量、波动率、成交量等多种因子类型。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 适用场景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 适合:
  - 单因子策略研究
  - 因子有效性验证
  - 技术指标分析
  - 快速原型验证

⚠️ 不适合:
  - 多因子组合（使用generate_alpha158）
  - 复杂因子计算（需要自定义实现）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 支持的因子类型
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- momentum: 动量因子（价格变化趋势）
- volatility: 波动率因子（价格波动程度）
- volume_ratio: 成交量比率因子
- rsi: 相对强弱指数（14日周期）
- macd: 移动平均收敛散度
- bollinger_bands: 布林带指标

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎬 典型工作流
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load_csv_data
    ↓
calculate_factor 👈 当前步骤
    ↓
evaluate_factor_ic
    ↓
如果IC有效 → 训练模型
    ↓
如果IC无效 → 尝试其他因子

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ 性能建议
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 数据量 < 10万行: 1-3秒
- 数据量 10-100万行: 5-15秒
- 数据量 > 100万行: 20-60秒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱️ 预计耗时
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 1000条数据: < 1秒
- 1万条数据: 1-3秒
- 10万条数据: 5-10秒
- 100万条数据: 20-40秒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 注意事项
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 必须先使用load_csv_data加载数据
2. factor_name必须唯一，重复会覆盖
3. period参数影响因子计算窗口
4. 建议先计算少量因子验证有效性
5. 因子计算后会自动存储，可通过list_factors查看

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""",
        inputSchema={
            "type": "object",
            "properties": {
                "data_id": {
                    "type": "string",
                    "description": """
数据标识ID - 指定要计算因子的数据源

📌 要求:
- 必须是已通过load_csv_data加载的数据ID
- 数据必须包含必需的OHLCV列
- 数据量建议 > 100条

❌ 常见错误:
- 数据ID不存在
- 数据格式不正确
- 数据量不足

💡 最佳实践:
- 使用有意义的data_id便于管理
- 确保数据已正确加载
- 使用list_factors查看可用数据
""",
                    "examples": ["stock_data_2023", "training_set", "backtest_data"]
                },
                "factor_name": {
                    "type": "string",
                    "description": """
因子名称 - 为计算的因子指定唯一标识

📌 命名规则:
- 只能包含字母、数字、下划线
- 建议包含因子类型和周期
- 区分大小写
- 长度建议 3-50 个字符

📌 命名建议:
- 包含因子类型: momentum_20, volatility_30
- 包含用途: signal_factor, alpha_factor
- 组合命名: rsi_14_signal, macd_trend

❌ 常见错误:
- 使用特殊字符
- 名称过于简单
- 重复使用相同名称

💡 最佳实践:
- 使用描述性名称
- 包含周期信息
- 便于后续引用
""",
                    "examples": ["momentum_20", "volatility_30", "rsi_14", "macd_signal"]
                },
                "factor_type": {
                    "type": "string",
                    "enum": ["momentum", "volatility", "volume_ratio", "rsi", "macd", "bollinger_bands"],
                    "description": """
因子类型 - 选择要计算的因子类型

📌 可用类型:
- momentum: 动量因子，衡量价格变化趋势
- volatility: 波动率因子，衡量价格波动程度
- volume_ratio: 成交量比率因子
- rsi: 相对强弱指数（14日周期）
- macd: 移动平均收敛散度
- bollinger_bands: 布林带指标

💡 选择建议:
- 趋势跟踪: momentum, macd
- 均值回归: rsi, bollinger_bands
- 波动率策略: volatility
- 成交量策略: volume_ratio
""",
                    "examples": ["momentum", "volatility", "rsi"]
                },
                "period": {
                    "type": "integer",
                    "default": 20,
                    "description": """
计算周期 - 因子计算的时间窗口（单位：天）

📌 周期范围:
- 建议范围: 5-60天
- 短期因子: 5-20天
- 中期因子: 20-40天
- 长期因子: 40-60天

💡 周期选择:
- 动量因子: 20-30天
- 波动率因子: 10-30天
- RSI: 固定14天
- MACD: 固定参数

⚠️ 注意:
- 周期不能超过数据长度
- 周期越大，数据要求越多
""",
                    "examples": [10, 20, 30, 50]
                }
            },
            "required": ["data_id", "factor_name", "factor_type"],
            "examples": [
                {
                    "name": "计算20日动量因子",
                    "description": "计算20日价格动量因子",
                    "input": {
                        "data_id": "stock_data_2023",
                        "factor_name": "momentum_20",
                        "factor_type": "momentum",
                        "period": 20
                    },
                    "expected_output": "动量因子计算完成，返回因子信息和质量评估"
                },
                {
                    "name": "计算30日波动率因子",
                    "description": "计算30日历史波动率",
                    "input": {
                        "data_id": "training_set",
                        "factor_name": "volatility_30",
                        "factor_type": "volatility",
                        "period": 30
                    },
                    "expected_output": "波动率因子计算完成，包含标准差和波动率指标"
                },
                {
                    "name": "计算RSI指标",
                    "description": "计算14日相对强弱指数",
                    "input": {
                        "data_id": "backtest_data",
                        "factor_name": "rsi_14",
                        "factor_type": "rsi"
                    },
                    "expected_output": "RSI指标计算完成，返回0-100范围内的数值"
                }
            ]
        }
    )


def get_generate_alpha158_schema():
    """generate_alpha158工具Schema"""
    return types.Tool(
        name="generate_alpha158",
        description="""
[🔬 因子生成 | 步骤2/6] 生成Alpha158因子集

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 功能概述
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

生成完整的Alpha158因子库，包含158个技术因子，
涵盖K线形态、价格特征、成交量特征和滚动统计特征。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 适用场景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 适合:
  - 多因子模型构建
  - 机器学习特征工程
  - 深度学习模型训练
  - 因子挖掘研究

⚠️ 不适合:
  - 单因子策略（使用calculate_factor）
  - 实时计算（计算量较大）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 因子分类
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- K线形态因子 (9个): 开盘、最高、最低、收盘相关特征
- 价格特征因子 (5个): 价格变化、收益率等
- 成交量特征因子 (5个): 成交量变化、成交额等
- 滚动统计因子 (139个): 不同窗口期的统计特征

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎬 典型工作流
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load_csv_data
    ↓
generate_alpha158 👈 当前步骤
    ↓
apply_processor_chain (可选)
    ↓
train_lstm_model / train_gru_model / train_transformer_model
    ↓
predict_with_model

━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ 性能建议
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 数据量 < 1万行: 10-30秒
- 数据量 1-10万行: 1-5分钟
- 数据量 > 10万行: 5-30分钟

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱️ 预计耗时
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 1000条数据: 10-30秒
- 1万条数据: 1-3分钟
- 10万条数据: 5-15分钟
- 100万条数据: 30-60分钟

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 注意事项
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 数据量建议 > 1000条，否则因子质量可能不佳
2. 滚动窗口越大，需要的数据越多
3. 生成因子会占用较多内存
4. result_id必须唯一，重复会覆盖
5. 建议使用apply_processor_chain进行数据清洗

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""",
        inputSchema={
            "type": "object",
            "properties": {
                "data_id": {
                    "type": "string",
                    "description": """
数据标识ID - 指定要生成因子的数据源

📌 要求:
- 必须是已通过load_csv_data加载的数据ID
- 数据必须包含必需的OHLCV列
- 数据量建议 > 1000条

❌ 常见错误:
- 数据ID不存在
- 数据格式不正确
- 数据量不足

💡 最佳实践:
- 确保数据已正确加载
- 数据量越大，因子质量越好
- 使用list_factors查看可用数据
""",
                    "examples": ["stock_data_2023", "training_set", "backtest_data"]
                },
                "result_id": {
                    "type": "string",
                    "description": """
结果标识ID - 为生成的Alpha158因子指定唯一标识

📌 命名规则:
- 只能包含字母、数字、下划线
- 建议包含数据类型和用途
- 区分大小写
- 长度建议 3-50 个字符

📌 命名建议:
- 包含数据类型: alpha158_stock, alpha158_index
- 包含用途: alpha158_training, alpha158_test
- 组合命名: alpha158_hs300_2023

❌ 常见错误:
- 使用特殊字符
- 名称过于简单
- 重复使用相同名称

💡 最佳实践:
- 使用描述性名称
- 便于后续引用
- 避免与data_id重复
""",
                    "examples": ["alpha158_stock", "alpha158_training", "alpha158_hs300"]
                },
                "kbar": {
                    "type": "boolean",
                    "default": True,
                    "description": """
是否生成K线形态因子

📌 包含9个因子:
- 开盘价相关特征
- 最高价相关特征
- 最低价相关特征
- 收盘价相关特征

💡 建议:
- 通常建议开启
- 对价格预测很重要
- 占用计算资源较少
""",
                    "examples": [True, False]
                },
                "price": {
                    "type": "boolean",
                    "default": True,
                    "description": """
是否生成价格特征因子

📌 包含5个因子:
- 价格变化特征
- 收益率特征
- 价格动量特征

💡 建议:
- 通常建议开启
- 核心价格特征
- 对趋势预测重要
""",
                    "examples": [True, False]
                },
                "volume": {
                    "type": "boolean",
                    "default": True,
                    "description": """
是否生成成交量特征因子

📌 包含5个因子:
- 成交量变化特征
- 成交额特征
- 量价关系特征

💡 建议:
- 通常建议开启
- 对波动率预测重要
- 提供流动性信息
""",
                    "examples": [True, False]
                },
                "rolling": {
                    "type": "boolean",
                    "default": True,
                    "description": """
是否生成滚动统计因子

📌 包含139个因子:
- 不同窗口期的统计特征
- 移动平均、标准差等
- 技术指标衍生

💡 建议:
- 通常建议开启
- 提供时间序列特征
- 计算量较大
""",
                    "examples": [True, False]
                },
                "rolling_windows": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "default": [5, 10, 20, 30, 60],
                    "description": """
滚动窗口大小列表（单位：天）

📌 窗口范围:
- 建议范围: 5-60天
- 短期窗口: 5-10天
- 中期窗口: 20-30天
- 长期窗口: 40-60天

💡 窗口选择:
- 数据量少时使用较小窗口
- 数据量多时可使用较大窗口
- 建议包含多个时间尺度

⚠️ 注意:
- 窗口越大，需要的数据越多
- 窗口数量影响计算时间
""",
                    "examples": [
                        [5, 10, 20],
                        [5, 10, 20, 30, 60],
                        [10, 20, 30, 50]
                    ]
                },
                "export_path": {
                    "type": "string",
                    "description": """
导出路径（可选） - 将生成的Alpha158因子数据导出到本地文件

📌 导出说明:
- 如果指定此参数，因子数据将导出到指定路径
- 支持CSV和JSON两种格式
- 导出文件包含所有158个因子的数值
- 便于在Excel或其他工具中查看和分析

📌 路径格式:
- 绝对路径: "D:/data/alpha158_factors.csv"
- 相对路径: "./exports/alpha158_factors.csv"
- 自动创建不存在的目录

📌 导出优势:
- 直接查看所有158个因子的数值
- 便于数据质量检查和验证
- 支持外部工具进一步分析
- 数据持久化保存

💡 使用建议:
- CSV格式便于Excel查看
- JSON格式便于程序处理
- 建议导出后检查因子质量
""",
                    "examples": [
                        "./exports/alpha158_factors.csv",
                        "D:/data/factors/alpha158_au2412.csv",
                        "./exports/alpha158_factors.json"
                    ]
                },
                "export_format": {
                    "type": "string",
                    "enum": ["csv", "json"],
                    "default": "csv",
                    "description": """
导出格式（可选，默认csv）

📌 格式说明:
- csv: CSV格式，便于Excel/WPS查看
- json: JSON格式，便于程序处理

💡 选择建议:
- 数据查看: 使用csv
- 程序处理: 使用json
- 默认推荐: csv
""",
                    "examples": ["csv", "json"]
                }
            },
            "required": ["data_id", "result_id"],
            "examples": [
                {
                    "name": "生成完整Alpha158因子并导出CSV",
                    "description": "生成包含所有158个因子的完整特征集并导出到CSV文件便于Excel查看",
                    "input": {
                        "data_id": "stock_data_2023",
                        "result_id": "alpha158_full",
                        "kbar": True,
                        "price": True,
                        "volume": True,
                        "rolling": True,
                        "rolling_windows": [5, 10, 20, 30, 60],
                        "export_path": "./exports/alpha158_full.csv",
                        "export_format": "csv"
                    },
                    "expected_output": "Alpha158因子生成完成并导出到CSV，可在Excel中查看所有158个因子的数值"
                },
                {
                    "name": "生成精简Alpha158因子并导出JSON",
                    "description": "只生成核心因子并导出为JSON格式便于程序处理",
                    "input": {
                        "data_id": "training_set",
                        "result_id": "alpha158_lite",
                        "kbar": True,
                        "price": True,
                        "volume": False,
                        "rolling": True,
                        "rolling_windows": [10, 20, 30],
                        "export_path": "./exports/alpha158_lite.json",
                        "export_format": "json"
                    },
                    "expected_output": "精简版Alpha158因子生成完成并导出为JSON格式"
                },
                {
                    "name": "生成基础因子（不导出）",
                    "description": "只生成基础K线和价格因子，保留在内存中不导出",
                    "input": {
                        "data_id": "backtest_data",
                        "result_id": "alpha158_basic",
                        "kbar": True,
                        "price": True,
                        "volume": False,
                        "rolling": False
                    },
                    "expected_output": "基础Alpha158因子生成完成，包含19个核心因子，保存在内存中"
                }
            ]
        }
    )


def get_evaluate_factor_ic_schema():
    """evaluate_factor_ic工具Schema"""
    return types.Tool(
        name="evaluate_factor_ic",
        description="""
[📊 因子评估 | 步骤3/6] 评估因子IC

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 功能概述
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

评估因子的信息系数（Information Coefficient），
衡量因子对未来收益率的预测能力。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 适用场景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 适合:
  - 因子有效性验证
  - 因子筛选
  - 策略优化
  - 模型特征选择

⚠️ 不适合:
  - 实时评估（需要历史数据）
  - 非数值型因子

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 IC指标说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- IC均值: 因子与未来收益的相关性均值
- IC标准差: 因子稳定性的衡量
- ICIR: 信息比率（IC均值/IC标准差）
- 因子质量评级: 基于IC的因子有效性评估

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎬 典型工作流
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

load_csv_data
    ↓
calculate_factor 或 generate_alpha158
    ↓
evaluate_factor_ic 👈 当前步骤
    ↓
如果IC有效 → 训练模型
    ↓
如果IC无效 → 尝试其他因子

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ 性能建议
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 数据量 < 10万行: 1-5秒
- 数据量 10-100万行: 5-20秒
- 数据量 > 100万行: 20-60秒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱️ 预计耗时
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 1000条数据: < 1秒
- 1万条数据: 1-3秒
- 10万条数据: 5-10秒
- 100万条数据: 20-40秒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 注意事项
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 因子和价格数据的时间范围必须匹配
2. 数据量建议 > 100条，否则IC不稳定
3. IC绝对值 > 0.03通常认为有效
4. ICIR > 0.5通常认为稳定
5. 建议使用spearman方法（对异常值更稳健）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""",
        inputSchema={
            "type": "object",
            "properties": {
                "factor_name": {
                    "type": "string",
                    "description": """
因子名称 - 指定要评估的因子

📌 要求:
- 必须是已通过calculate_factor或generate_alpha158生成的因子
- 因子数据必须为数值型
- 因子数据不能全为NaN

❌ 常见错误:
- 因子名称不存在
- 因子数据格式不正确
- 因子数据全为NaN

💡 最佳实践:
- 使用有意义的因子名称
- 确保因子已正确生成
- 使用list_factors查看可用因子
""",
                    "examples": ["momentum_20", "alpha158_stock", "volatility_30"]
                },
                "data_id": {
                    "type": "string",
                    "description": """
数据标识ID - 指定价格数据源

📌 要求:
- 必须是已通过load_csv_data加载的数据ID
- 数据必须包含close列
- 数据时间范围与因子匹配

❌ 常见错误:
- 数据ID不存在
- 数据缺少close列
- 时间范围不匹配

💡 最佳实践:
- 使用与因子相同的数据源
- 确保数据包含价格信息
- 时间范围要覆盖因子计算期
""",
                    "examples": ["stock_data_2023", "training_set", "backtest_data"]
                },
                "method": {
                    "type": "string",
                    "enum": ["spearman", "pearson"],
                    "default": "spearman",
                    "description": """
IC计算方法

📌 方法说明:
- spearman: 斯皮尔曼秩相关系数（推荐）
  - 对异常值不敏感
  - 适用于非线性关系
  - 更稳健
- pearson: 皮尔逊相关系数
  - 假设线性关系
  - 对异常值敏感
  - 计算更快

💡 选择建议:
- 通常建议使用spearman
- 如果确定线性关系可使用pearson
- spearman更稳健，推荐使用
""",
                    "examples": ["spearman", "pearson"]
                },
                "report_path": {
                    "type": "string",
                    "description": """
评估报告保存路径（可选） - 生成详细的Markdown格式因子评估报告

📌 报告说明:
- 如果指定此参数，将生成完整的因子评估报告
- 报告格式为Markdown（.md），易于阅读和分享
- 包含IC指标详解、因子质量评级、使用建议等
- 自动创建不存在的目录

📌 路径格式:
- 绝对路径: "D:/reports/factor_ic_report.md"
- 相对路径: "./reports/factor_evaluation.md"
- 默认保存到 ./exports/ 目录

📌 报告内容:
- 基本信息（因子名称、数据源、评估时间）
- IC指标（IC均值、IC标准差、ICIR、IC正值占比）
- 因子质量评级和推荐建议
- 详细指标解读和使用建议
- 后续步骤指引

💡 使用建议:
- 重要因子建议生成报告存档
- 报告便于团队共享和讨论
- 可作为策略文档的一部分
- 支持版本控制和对比

⚠️ 注意:
- 报告会覆盖同名文件
- 使用UTF-8编码保存
- 可用Markdown查看器或文本编辑器打开
""",
                    "examples": [
                        "./reports/factor_ic_evaluation.md",
                        "D:/quant/reports/alpha158_ic_report.md",
                        "./exports/momentum_20_report.md"
                    ]
                }
            },
            "required": ["factor_name", "data_id"],
            "examples": [
                {
                    "name": "评估动量因子IC",
                    "description": "评估20日动量因子的预测能力",
                    "input": {
                        "factor_name": "momentum_20",
                        "data_id": "stock_data_2023",
                        "method": "spearman"
                    },
                    "expected_output": "IC评估完成，返回IC均值、ICIR、因子质量评级和后续建议"
                },
                {
                    "name": "评估Alpha158因子IC并生成报告",
                    "description": "评估Alpha158因子集的整体有效性并生成详细报告",
                    "input": {
                        "factor_name": "alpha158_au2412_normalized",
                        "data_id": "au2412_data",
                        "method": "spearman",
                        "report_path": "./exports/alpha158_ic_report.md"
                    },
                    "expected_output": "Alpha158因子IC评估完成，返回综合IC指标和因子质量分析，并生成Markdown格式的详细评估报告"
                },
                {
                    "name": "评估波动率因子IC",
                    "description": "评估30日波动率因子的预测能力",
                    "input": {
                        "factor_name": "volatility_30",
                        "data_id": "backtest_data",
                        "method": "pearson"
                    },
                    "expected_output": "波动率因子IC评估完成，返回IC指标和稳定性分析"
                }
            ]
        }
    )


def get_list_factors_schema():
    """list_factors工具Schema"""
    return types.Tool(
        name="list_factors",
        description="""
[📋 状态查询 | 辅助工具] 列出所有已加载的数据和因子

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 功能概述
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

查看当前内存中所有已加载的数据和因子，
便于管理和引用。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 适用场景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 适合:
  - 查看当前工作状态
  - 查找可用的数据和因子
  - 管理内存中的对象
  - 调试和问题排查

⚠️ 不适合:
  - 数据持久化存储
  - 大量历史数据查询

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 返回信息
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 数据数量和数据ID列表
- 因子数量和因子信息
- 每个因子的类型和形状
- 内存使用情况概览

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎬 典型工作流
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

在任何步骤中都可使用此工具查看当前状态

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ 性能建议
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 瞬时完成，无性能影响

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⏱️ 预计耗时
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- < 100毫秒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 注意事项
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 只显示当前内存中的数据
2. 重启服务后数据会丢失
3. 数据量过大时建议及时清理
4. 可用于检查数据是否正确加载

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""",
        inputSchema={
            "type": "object",
            "properties": {},
            "examples": [
                {
                    "name": "查看当前状态",
                    "description": "查看所有已加载的数据和因子",
                    "input": {},
                    "expected_output": "返回数据数量、因子数量、详细列表和内存使用情况"
                }
            ]
        }
    )


def get_apply_processor_chain_schema():
    """apply_processor_chain工具Schema"""
    return types.Tool(
        name="apply_processor_chain",
        description="""
[🔧 数据处理 | 步骤3/6] 应用数据处理器链（智能标准化）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 功能概述
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

对数据（通常是因子数据）应用一系列处理器进行标准化、归一化等操作。
这是Qlib工作流程中的关键步骤 - 用于标准化因子而非原始数据。

✨ v1.0.14新特性：智能标准化
- 自动识别单商品/多商品数据
- 单商品数据：CSZScoreNorm自动切换为ZScoreNorm（避免100% NaN）
- 多商品数据：CSZScoreNorm正常使用（截面标准化）
- 无需手动选择，系统自动优化

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 典型用途
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 推荐: 直接使用CSZScoreNorm，系统会自动优化
✅ 正确: 对生成的Alpha158因子进行标准化
✅ 正确: 对因子数据应用处理器链
❌ 错误: 对原始OHLCV数据进行标准化

💡 智能处理示例:
- 单商品（AU2412）+ CSZScoreNorm → 自动使用ZScoreNorm
- 多商品（沪深300）+ CSZScoreNorm → 正常使用CSZScoreNorm
- 用户无需关心数据类型，一律使用CSZScoreNorm即可

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎬 典型工作流
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

preprocess_data (清洗原始数据)
    ↓
generate_alpha158 (生成因子)
    ↓
apply_processor_chain 👈 当前步骤（智能标准化因子）
    ↓
train_model (训练模型)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 处理器说明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CSZScoreNorm (推荐⭐):
  - 智能截面标准化
  - 多商品: 按时间截面标准化
  - 单商品: 自动切换为ZScoreNorm
  - 适用场景: 所有情况（系统自动优化）

ZScoreNorm:
  - 时间序列标准化
  - 适用场景: 单商品/单股票数据
  - 手动使用（通常不需要，CSZScoreNorm会自动处理）

RobustZScoreNorm:
  - 鲁棒标准化（对异常值不敏感）
  - 适用场景: 含异常值的数据

MinMaxNorm:
  - 最小最大归一化到[0,1]
  - 适用场景: 需要特定范围的数据

CSZFillna:
  - 截面填充缺失值
  - 适用场景: 数据清洗

ProcessInf:
  - 处理无穷值
  - 适用场景: 数据清洗

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""",
        inputSchema={
            "type": "object",
            "properties": {
                "data_id": {
                    "type": "string",
                    "description": "要处理的数据ID（通常是因子数据）"
                },
                "result_id": {
                    "type": "string",
                    "description": "处理后数据的存储ID"
                },
                "processors": {
                    "type": "array",
                    "description": """处理器配置列表

📌 推荐用法（v1.0.14+）:
- 直接使用 CSZScoreNorm，系统会自动识别数据类型并优化
- 单商品数据会自动切换为 ZScoreNorm（避免NaN问题）
- 多商品数据会正常使用 CSZScoreNorm（截面标准化）
- 无需手动判断，一律使用 CSZScoreNorm 即可

📌 可用处理器:
- CSZScoreNorm: 智能截面标准化（推荐⭐）
- ZScoreNorm: 时间序列标准化
- RobustZScoreNorm: 鲁棒标准化
- MinMaxNorm: 最小最大归一化
- CSRankNorm: 排名标准化
- CSZFillna: 截面填充缺失值
- ProcessInf: 处理无穷值
""",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": ["CSZScoreNorm", "CSZFillna", "ProcessInf", "ZScoreNorm", "RobustZScoreNorm", "CSRankNorm", "MinMaxNorm"],
                                "description": "处理器名称。推荐使用CSZScoreNorm，系统会自动优化"
                            },
                            "params": {
                                "type": "object",
                                "description": "处理器参数（可选）"
                            }
                        },
                        "required": ["name"]
                    }
                },
                "export_path": {
                    "type": "string",
                    "description": "导出路径（可选）",
                    "examples": [
                        "./exports/factors_normalized.csv",
                        "D:/data/processed/factors_norm.json"
                    ]
                },
                "export_format": {
                    "type": "string",
                    "enum": ["csv", "json"],
                    "default": "csv",
                    "description": "导出格式（csv或json）",
                    "examples": ["csv", "json"]
                }
            },
            "required": ["data_id", "result_id", "processors"],
            "examples": [
                {
                    "name": "智能标准化（推荐）- 单商品数据",
                    "description": "对单商品因子使用CSZScoreNorm，系统自动切换为ZScoreNorm",
                    "input": {
                        "data_id": "alpha158_au2412",
                        "result_id": "alpha158_au2412_norm",
                        "processors": [{"name": "CSZScoreNorm"}]
                    },
                    "expected_output": "检测到单商品，自动切换为ZScoreNorm，NaN比例0%"
                },
                {
                    "name": "智能标准化（推荐）- 多商品数据",
                    "description": "对多商品因子使用CSZScoreNorm，正常执行截面标准化",
                    "input": {
                        "data_id": "alpha158_hs300",
                        "result_id": "alpha158_hs300_norm",
                        "processors": [{"name": "CSZScoreNorm"}],
                        "export_path": "./exports/hs300_factors_norm.csv"
                    },
                    "expected_output": "多商品数据，CSZScoreNorm正常工作，导出到CSV"
                },
                {
                    "name": "多处理器链式处理",
                    "description": "组合多个处理器进行复杂数据清洗",
                    "input": {
                        "data_id": "alpha158_raw",
                        "result_id": "alpha158_clean_norm",
                        "processors": [
                            {"name": "ProcessInf"},
                            {"name": "CSZFillna"},
                            {"name": "CSZScoreNorm"}
                        ],
                        "export_path": "./exports/factors_clean_norm.json",
                        "export_format": "json"
                    },
                    "expected_output": "依次处理无穷值、填充缺失值、智能标准化，导出为JSON"
                }
            ]
        }
    )


def get_train_ml_model_schema():
    """train_ml_model工具Schema"""
    return types.Tool(
        name="train_ml_model",
        description="""
[🤖 机器学习训练 | 步骤4/6] 训练机器学习模型（LightGBM/XGBoost/sklearn）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 功能概述
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

使用LightGBM、XGBoost或scikit-learn训练机器学习模型进行股价预测。
支持多种模型类型和自定义参数，提供完整的训练和评估指标。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 适用场景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 适合:
  - 基于因子的量化策略
  - 中短期价格预测
  - 特征工程研究
  - 快速原型验证

⚠️ 不适合:
  - 序列建模（可考虑深度学习）
  - 实时高频交易
  - 需要复杂特征交互的场景

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 支持的模型类型
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- lightgbm: LightGBM梯度提升树（推荐，速度快）
- xgboost: XGBoost梯度提升树（性能强）
- linear: 线性回归（基线模型）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎬 典型工作流
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

preprocess_data
    ↓
generate_alpha158
    ↓
apply_processor_chain
    ↓
train_ml_model 👈 当前步骤
    ↓
predict_ml_model

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ 性能建议
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 数据量 < 1万: 1-10秒
- 数据量 1-10万: 10-60秒
- 数据量 > 10万: 1-10分钟

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 注意事项
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 数据必须包含close列用于生成标签
2. 训练和测试时间范围不能重叠
3. 建议训练集:测试集 = 7:3 或 8:2
4. 模型会自动保存在内存中
5. 特征重要性可用于特征筛选

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""",
        inputSchema={
            "type": "object",
            "properties": {
                "data_id": {
                    "type": "string",
                    "description": """
数据标识ID - 指定训练数据源

📌 要求:
- 必须包含close列（用于生成标签）
- 或使用已生成的因子数据
- 数据量建议 > 1000条

💡 最佳实践:
- 使用标准化后的因子数据效果更好
- 确保数据时间范围覆盖训练和测试期
""",
                    "examples": ["stock_data_2023", "alpha158_normalized"]
                },
                "model_id": {
                    "type": "string",
                    "description": """
模型标识ID - 为训练的模型指定唯一标识

📌 命名建议:
- 包含模型类型: lgb_model, xgb_model
- 包含数据信息: model_hs300_2023
- 包含用途: model_backtest_v1

💡 最佳实践:
- 使用描述性名称便于管理
- 不同实验使用不同model_id
""",
                    "examples": ["lgb_model_v1", "xgb_hs300", "model_test"]
                },
                "model_type": {
                    "type": "string",
                    "enum": ["lightgbm", "xgboost", "linear"],
                    "default": "lightgbm",
                    "description": """
模型类型

📌 可选类型:
- lightgbm: 推荐，速度快，效果好
- xgboost: 性能强，参数多
- linear: 基线模型，速度最快

💡 选择建议:
- 首次尝试: lightgbm
- 追求性能: xgboost
- 快速验证: linear
""",
                    "examples": ["lightgbm", "xgboost", "linear"]
                },
                "train_start": {
                    "type": "string",
                    "description": "训练集起始日期，格式YYYY-MM-DD",
                    "examples": ["2023-01-01", "2020-01-01"]
                },
                "train_end": {
                    "type": "string",
                    "description": "训练集结束日期，格式YYYY-MM-DD",
                    "examples": ["2023-06-30", "2022-12-31"]
                },
                "test_start": {
                    "type": "string",
                    "description": "测试集起始日期，格式YYYY-MM-DD",
                    "examples": ["2023-07-01", "2023-01-01"]
                },
                "test_end": {
                    "type": "string",
                    "description": "测试集结束日期，格式YYYY-MM-DD",
                    "examples": ["2023-12-31", "2023-06-30"]
                },
                "params": {
                    "type": "object",
                    "description": """
模型参数（可选）

📌 LightGBM参数示例:
- learning_rate: 学习率（默认0.05）
- num_leaves: 叶子数（默认31）
- max_depth: 最大深度

📌 XGBoost参数示例:
- learning_rate: 学习率（默认0.05）
- max_depth: 最大深度（默认6）

💡 使用建议:
- 不指定则使用默认参数
- 默认参数通常已经不错
""",
                    "examples": [
                        {"learning_rate": 0.05, "num_leaves": 31},
                        {"learning_rate": 0.1, "max_depth": 8}
                    ]
                }
            },
            "required": ["data_id", "model_id", "train_start", "train_end", "test_start", "test_end"],
            "examples": [
                {
                    "name": "训练LightGBM模型",
                    "description": "使用Alpha158因子训练LightGBM模型",
                    "input": {
                        "data_id": "alpha158_normalized",
                        "model_id": "lgb_model_v1",
                        "model_type": "lightgbm",
                        "train_start": "2023-01-01",
                        "train_end": "2023-06-30",
                        "test_start": "2023-07-01",
                        "test_end": "2023-12-31"
                    },
                    "expected_output": "模型训练完成，返回训练和测试性能指标、特征重要性"
                },
                {
                    "name": "训练XGBoost模型（自定义参数）",
                    "description": "使用自定义参数训练XGBoost模型",
                    "input": {
                        "data_id": "stock_data_2023",
                        "model_id": "xgb_custom",
                        "model_type": "xgboost",
                        "train_start": "2020-01-01",
                        "train_end": "2022-12-31",
                        "test_start": "2023-01-01",
                        "test_end": "2023-12-31",
                        "params": {
                            "learning_rate": 0.1,
                            "max_depth": 8
                        }
                    },
                    "expected_output": "XGBoost模型训练完成，包含完整性能评估"
                }
            ]
        }
    )


def get_predict_ml_model_schema():
    """predict_ml_model工具Schema"""
    return types.Tool(
        name="predict_ml_model",
        description="""
[🔮 模型预测 | 步骤5/6] 使用训练好的机器学习模型进行预测

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 功能概述
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

使用已训练的机器学习模型对新数据进行预测。
支持批量预测和结果导出。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 适用场景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 适合:
  - 回测验证
  - 样本外测试
  - 实盘信号生成
  - 模型性能评估

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎬 典型工作流
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

train_ml_model
    ↓
predict_ml_model 👈 当前步骤
    ↓
分析预测结果

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ 性能建议
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 预测速度快，通常 < 1秒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 注意事项
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 模型必须先训练
2. 预测数据特征必须与训练时一致
3. 预测结果为收益率预测值
4. 建议导出结果用于进一步分析

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""",
        inputSchema={
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": """
模型标识ID - 指定要使用的模型

📌 要求:
- 必须是已训练的模型ID
- 使用list_factors可查看可用模型

💡 最佳实践:
- 确认模型已训练完成
- 使用正确的model_id
""",
                    "examples": ["lgb_model_v1", "xgb_hs300"]
                },
                "data_id": {
                    "type": "string",
                    "description": """
数据标识ID - 指定要预测的数据

📌 要求:
- 特征必须与训练时一致
- 数据格式必须相同

💡 最佳实践:
- 使用相同的数据处理流程
- 确保特征完整
""",
                    "examples": ["alpha158_normalized", "test_data"]
                },
                "export_path": {
                    "type": "string",
                    "description": """
导出路径（可选） - 将预测结果导出到本地文件

📌 导出说明:
- CSV格式，包含预测值
- 便于后续分析和回测

💡 使用建议:
- 建议导出重要预测结果
- 便于结果对比和验证
""",
                    "examples": [
                        "./exports/predictions.csv",
                        "D:/results/model_pred.csv"
                    ]
                }
            },
            "required": ["model_id", "data_id"],
            "examples": [
                {
                    "name": "使用模型预测",
                    "description": "使用训练好的模型预测新数据",
                    "input": {
                        "model_id": "lgb_model_v1",
                        "data_id": "alpha158_normalized"
                    },
                    "expected_output": "预测完成，返回预测统计信息和样本预览"
                },
                {
                    "name": "预测并导出结果",
                    "description": "预测并将结果导出到文件",
                    "input": {
                        "model_id": "xgb_custom",
                        "data_id": "test_data",
                        "export_path": "./exports/predictions.csv"
                    },
                    "expected_output": "预测完成并导出到CSV文件"
                }
            ]
        }
    )


def get_merge_factor_data_schema():
    """merge_factor_data工具Schema"""
    return types.Tool(
        name="merge_factor_data",
        description="""
[🔗 数据合并 | 步骤2.5/6] 合并因子数据和价格数据

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 功能概述
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

将Alpha158等因子数据与原始价格数据合并，生成包含因子特征和close列的完整数据集。
解决了因子数据无法直接用于train_ml_model的问题。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 适用场景
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ 适合:
  - Alpha158因子训练机器学习模型
  - 自定义因子组合训练
  - 因子数据与价格数据分离的场景

⚠️ 不适合:
  - 原始数据已包含close列的情况
  - 深度学习模型（可直接使用因子数据）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎬 典型工作流
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

preprocess_data
    ↓
generate_alpha158
    ↓
apply_processor_chain (可选)
    ↓
merge_factor_data 👈 当前步骤（合并因子+价格）
    ↓
train_ml_model
    ↓
predict_ml_model

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ 性能建议
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- 数据量 < 10万行: < 1秒
- 数据量 10-100万行: 1-5秒
- 数据量 > 100万行: 5-15秒

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ 注意事项
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 因子数据和价格数据必须来源于同一原始数据
2. 两个数据集的时间范围必须有重叠
3. 合并后的数据包含所有因子列 + close列
4. result_id必须唯一，重复会覆盖
5. 合并后的数据可直接用于train_ml_model

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""",
        inputSchema={
            "type": "object",
            "properties": {
                "factor_data_id": {
                    "type": "string",
                    "description": """
因子数据ID - 指定要合并的因子数据

📌 要求:
- 必须是已生成的因子数据ID
- 通常是generate_alpha158或apply_processor_chain的输出
- 数据格式为DataFrame

❌ 常见错误:
- 因子数据ID不存在
- 使用了原始价格数据而非因子数据

💡 最佳实践:
- 使用标准化后的因子数据
- 确保因子已正确生成
- 使用list_factors查看可用因子
""",
                    "examples": ["alpha158_stock", "alpha158_normalized", "factors_processed"]
                },
                "price_data_id": {
                    "type": "string",
                    "description": """
价格数据ID - 指定原始价格数据源

📌 要求:
- 必须是已加载的原始数据ID
- 必须包含close列
- 时间范围与因子数据匹配

❌ 常见错误:
- 价格数据ID不存在
- 数据缺少close列
- 时间范围不匹配

💡 最佳实践:
- 使用生成因子时的原始数据ID
- 确保数据包含完整价格信息
- 时间范围要覆盖因子数据
""",
                    "examples": ["stock_data_2023", "raw_data", "price_data"]
                },
                "result_id": {
                    "type": "string",
                    "description": """
结果标识ID - 为合并后的数据指定唯一标识

📌 命名建议:
- 包含数据类型: merged_alpha158, merged_factors
- 包含用途: merged_for_training, merged_backtest
- 组合命名: alpha158_with_price

❌ 常见错误:
- 使用特殊字符
- 名称过于简单
- 与已有ID重复

💡 最佳实践:
- 使用描述性名称
- 便于后续引用
- 表明这是合并后的数据
""",
                    "examples": ["merged_alpha158", "alpha158_with_price", "merged_for_training"]
                },
                "export_path": {
                    "type": "string",
                    "description": """
导出路径（可选） - 将合并后的数据导出到本地文件

📌 导出说明:
- 支持CSV和JSON格式
- 包含所有因子列和close列
- 便于查看和验证合并结果

📌 路径格式:
- 绝对路径: "D:/data/merged_data.csv"
- 相对路径: "./exports/merged_factors.csv"
- 自动创建不存在的目录

💡 使用建议:
- CSV格式便于Excel查看
- JSON格式便于程序处理
- 建议导出后检查数据质量
""",
                    "examples": [
                        "./exports/merged_alpha158.csv",
                        "D:/data/merged_factors.csv",
                        "./exports/training_data.json"
                    ]
                },
                "export_format": {
                    "type": "string",
                    "enum": ["csv", "json"],
                    "default": "csv",
                    "description": """
导出格式（可选，默认csv）

📌 格式说明:
- csv: CSV格式，便于Excel查看
- json: JSON格式，便于程序处理

💡 选择建议:
- 数据查看: 使用csv
- 程序处理: 使用json
- 默认推荐: csv
""",
                    "examples": ["csv", "json"]
                }
            },
            "required": ["factor_data_id", "price_data_id", "result_id"],
            "examples": [
                {
                    "name": "合并Alpha158因子和价格数据",
                    "description": "将标准化后的Alpha158因子与原始价格数据合并",
                    "input": {
                        "factor_data_id": "alpha158_normalized",
                        "price_data_id": "stock_data_2023",
                        "result_id": "merged_alpha158"
                    },
                    "expected_output": "因子和价格数据合并完成，包含159个因子+1个close列，可直接用于训练"
                },
                {
                    "name": "合并并导出到CSV",
                    "description": "合并数据并导出到CSV文件便于查看",
                    "input": {
                        "factor_data_id": "alpha158_stock",
                        "price_data_id": "raw_data",
                        "result_id": "alpha158_with_price",
                        "export_path": "./exports/merged_data.csv",
                        "export_format": "csv"
                    },
                    "expected_output": "数据合并完成并导出到CSV，可在Excel中查看所有列"
                },
                {
                    "name": "合并处理后的因子",
                    "description": "合并经过多步处理的因子数据",
                    "input": {
                        "factor_data_id": "factors_clean_norm",
                        "price_data_id": "backtest_data",
                        "result_id": "merged_for_training",
                        "export_path": "./exports/training_ready.json",
                        "export_format": "json"
                    },
                    "expected_output": "清洗标准化后的因子与价格合并，导出为JSON格式"
                }
            ]
        }
    )


def get_all_tool_schemas():
    """获取所有工具Schema"""
    return [
        get_preprocess_data_schema(),
        get_calculate_factor_schema(),
        get_generate_alpha158_schema(),
        get_merge_factor_data_schema(),
        get_apply_processor_chain_schema(),
        get_evaluate_factor_ic_schema(),
        get_list_factors_schema(),
        get_train_ml_model_schema(),
        get_predict_ml_model_schema(),
    ]