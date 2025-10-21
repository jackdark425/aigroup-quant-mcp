"""
MCP资源定义 - FAQ文档系统
"""

from mcp import types
from typing import List


def get_faq_resources() -> List[types.Resource]:
    """获取所有FAQ资源"""
    return [
        types.Resource(
            uri="quant://faq/getting-started",
            name="快速入门指南",
            description="从零开始使用aigroup-quant-mcp的完整教程",
            mimeType="text/markdown"
        ),
        types.Resource(
            uri="quant://faq/common-errors",
            name="常见错误解决方案",
            description="常见错误及其解决方法",
            mimeType="text/markdown"
        ),
        types.Resource(
            uri="quant://faq/workflow-templates",
            name="工作流程模板",
            description="常见量化分析任务的完整工作流程",
            mimeType="text/markdown"
        ),
        types.Resource(
            uri="quant://faq/parameter-tuning",
            name="参数调优指南",
            description="各种模型和因子的参数调优建议",
            mimeType="text/markdown"
        ),
        types.Resource(
            uri="quant://faq/factor-library",
            name="因子库说明",
            description="Alpha158因子库详细说明",
            mimeType="text/markdown"
        )
    ]


def read_faq_resource(uri: str) -> str:
    """读取FAQ资源内容"""
    
    if uri == "quant://faq/getting-started":
        return """
# 快速入门指南

## 完整工作流程示例

### 场景: 构建LSTM选股模型

#### 第1步: 加载数据
```
工具: load_csv_data
参数:
  file_path: "./data/stock_data.csv"
  data_id: "my_stock_data"
  
预期结果: 数据成功加载，返回数据摘要信息
```

#### 第2步: 生成Alpha158因子
```
工具: generate_alpha158
参数:
  data_id: "my_stock_data"
  result_id: "alpha158_factors"
  kbar: true
  price: true
  volume: true
  rolling: true
  rolling_windows: [5, 10, 20, 30, 60]
  
预期结果: 生成158个技术指标因子
```

#### 第3步: 数据预处理(可选但推荐)
```
工具: create_processor_chain
参数:
  chain_id: "my_preprocessing"
  processors: [
    {"type": "DropnaLabel", "params": {"label_col": "return"}},
    {"type": "CSZScoreNorm", "params": {}},
    {"type": "Fillna", "params": {"fill_value": 0}}
  ]

然后应用:
工具: apply_processor_chain
参数:
  chain_id: "my_preprocessing"
  train_data_id: "alpha158_factors"
  train_result_id: "processed_factors"
```

#### 第4步: 训练LSTM模型
```
工具: train_lstm_model
参数:
  data_id: "processed_factors"
  model_id: "my_lstm_model"
  hidden_size: 64
  num_layers: 2
  n_epochs: 50
  batch_size: 800
  lr: 0.001
  
预期结果: LSTM模型训练完成，返回训练历史
```

#### 第5步: 模型预测
```
工具: predict_with_model
参数:
  model_id: "my_lstm_model"
  data_id: "processed_factors"
  result_id: "predictions"
  
预期结果: 生成股票收益预测
```

#### 第6步: 评估效果
```
工具: evaluate_factor_ic
参数:
  factor_name: "predictions"
  data_id: "my_stock_data"
  method: "spearman"
  
预期结果: 返回IC均值、ICIR等评估指标
```

## 注意事项

1. **数据格式**: CSV文件必须包含datetime、symbol、open、high、low、close、volume列
2. **数据量**: 建议至少1000条记录以获得稳定的因子计算结果
3. **ID命名**: 使用有意义的ID名称，便于后续引用和管理
4. **内存管理**: 大数据集建议分批处理，避免内存溢出

## 故障排查

### 问题: "数据未找到"
- 检查data_id是否正确
- 使用list_factors查看已加载的数据
- 确认load_csv_data是否成功执行

### 问题: "列缺失"
- 确认CSV文件包含必需的列
- 检查列名是否区分大小写
- 查看数据加载返回的columns列表

### 问题: "因子计算失败"
- 检查数据中是否有NaN或Inf值
- 确认数据量是否足够(至少100条)
- 尝试使用更小的rolling_windows
"""
    
    elif uri == "quant://faq/common-errors":
        return """
# 常见错误解决方案

## 错误1: DATA_NOT_FOUND
**错误信息**: "数据 'xxx' 未找到"

**原因**:
- 数据ID拼写错误
- 数据未成功加载
- 使用了错误的数据ID

**解决方法**:
1. 使用list_factors工具查看已加载的数据
2. 检查data_id参数的拼写
3. 重新执行load_csv_data加载数据

## 错误2: INVALID_PARAMETER
**错误信息**: "参数验证失败"

**常见原因和解决方法**:
- rolling_windows超出范围 → 使用2-250之间的值
- 列名不存在 → 检查CSV文件的列名
- 类型不匹配 → 确认参数类型正确(字符串/数字/布尔)

## 错误3: MODEL_NOT_TRAINED
**错误信息**: "模型尚未训练"

**原因**: 尝试用未训练的模型进行预测

**解决方法**:
1. 先使用train_*_model工具训练模型
2. 检查model_id是否正确
3. 确认训练是否成功完成

## 错误4: INSUFFICIENT_DATA
**错误信息**: "数据量不足"

**原因**: 数据行数少于最小要求

**解决方法**:
- Alpha158: 至少需要100条记录
- 模型训练: 建议1000条以上
- 增加数据范围或使用更多股票

## 错误5: COMPUTATION_ERROR
**错误信息**: "计算失败"

**常见原因**:
- 数据包含NaN或Inf值
- 内存不足
- 数值计算溢出

**解决方法**:
1. 使用Fillna processor处理缺失值
2. 减少rolling_windows数量
3. 使用更小的batch_size

## 错误6: FILE_ERROR
**错误信息**: "文件不存在"或"文件格式错误"

**解决方法**:
- 检查文件路径拼写
- 确认文件扩展名为.csv
- 使用绝对路径
- 检查文件权限
"""
    
    elif uri == "quant://faq/workflow-templates":
        return """
# 工作流程模板

## 模板1: 因子挖掘
**适用场景**: 寻找有效的量化因子

```
1. load_csv_data → 加载历史数据
2. generate_alpha158 → 生成158个候选因子
3. evaluate_factor_ic (循环) → 逐个评估因子IC
4. 筛选IC > 0.05的因子
5. 使用筛选后的因子训练模型
```

## 模板2: 深度学习选股
**适用场景**: 构建预测模型选股

```
1. load_csv_data → 加载数据
2. generate_alpha158 → 生成特征
3. create_processor_chain → 创建预处理链
4. apply_processor_chain → 应用预处理
5. train_lstm_model → 训练模型
6. predict_with_model → 生成预测
7. 根据预测值选股
```

## 模板3: 模型对比
**适用场景**: 对比不同模型的效果

```
1. load_csv_data → 加载数据
2. generate_alpha158 → 生成特征
3. apply_processor_chain → 预处理
4. train_lstm_model → LSTM模型
5. train_gru_model → GRU模型
6. train_transformer_model → Transformer模型
7. predict_with_model (×3) → 三个模型分别预测
8. evaluate_factor_ic (×3) → 评估对比
```

## 模板4: 快速验证
**适用场景**: 快速测试想法

```
1. load_csv_data → 加载数据
2. calculate_factor → 计算单个因子(速度快)
3. evaluate_factor_ic → 评估因子
4. 如果有效，再使用完整的Alpha158
```

## 模板5: 单因子策略回测
**适用场景**: 测试单个因子的选股效果

```
1. load_csv_data → 加载数据
2. calculate_factor → 计算目标因子
3. evaluate_factor_ic → 评估IC
4. 根据因子值排序选股
5. 计算收益率
```
"""
    
    elif uri == "quant://faq/parameter-tuning":
        return """
# 参数调优指南

## Alpha158因子参数

### rolling_windows (滚动窗口)

**默认值**: [5, 10, 20, 30, 60]

**调优建议**:
- 短期策略: [5, 10, 20] - 更灵敏
- 中期策略: [10, 20, 30, 60] - 平衡
- 长期策略: [30, 60, 120] - 更平滑
- 快速测试: [20] - 仅月线

**性能考虑**:
- 每增加1个窗口 → +27个因子
- 窗口数量越多，计算时间越长
- 建议: 3-5个窗口为最佳平衡点

## 因子计算参数

### period (计算周期)

**常用值**:
- momentum: 20-30 (月度动量)
- volatility: 20 (月波动率)
- rsi: 14 (标准参数)
- volume_ratio: 20-30

**调优原则**:
- 短周期(5-10): 更灵敏，但噪音大
- 中周期(20-30): 平衡，推荐
- 长周期(60-120): 更平滑，但滞后

## 模型训练参数

### LSTM模型

**hidden_size (隐藏层大小)**:
- 小模型: 32-64 (快速训练)
- 中模型: 64-128 (推荐)
- 大模型: 128-256 (复杂数据)

**num_layers (层数)**:
- 简单: 1层
- 标准: 2层 (推荐)
- 复杂: 3-4层

**n_epochs (训练轮数)**:
- 快速测试: 20-30
- 标准训练: 50-100
- 充分训练: 100-200

**batch_size (批次大小)**:
- 小数据集(<10000): 256-512
- 中数据集: 512-1024 (推荐)
- 大数据集: 1024-2048

**lr (学习率)**:
- 保守: 0.0001
- 标准: 0.001 (推荐)
- 激进: 0.01

## 数据预处理参数

### CSZScoreNorm (截面标准化)
**最常用**: 适合多股票横截面数据
**无需参数**

### ZScoreNorm (纵向标准化)
**适用**: 单股票时间序列
**无需参数**

### Fillna (填充缺失值)
**fill_value**: 
- 0 (推荐，保守)
- 均值 (平滑)
- 前值 (保持连续性)
"""
    
    elif uri == "quant://faq/factor-library":
        return """
# Alpha158因子库详细说明

## 因子分类 (共158个)

### 1. K线形态因子 (9个)

- **KMID**: (close-open)/open
  - 含义: 实体在K线中的位置
  - 取值: (-1, 1)
  - 正值: 阳线，负值: 阴线

- **KLEN**: (high-low)/open  
  - 含义: K线的总长度
  - 取值: (0, +∞)
  - 值越大波动越大

- **KUP**: (high-max(open,close))/open
  - 含义: 上影线长度
  - 取值: [0, +∞)
  - 值大: 上方抛压重

- **KLOW**: (min(open,close)-low)/open
  - 含义: 下影线长度
  - 取值: (-∞, 0]
  - 绝对值大: 下方支撑强

- **KSFT**: (2*close-high-low)/open
  - 含义: 重心位置
  - 取值: (-1, 1)

### 2. 价格因子 (5个)

- **OPEN0**: open/close
- **HIGH0**: high/close
- **LOW0**: low/close
- **CLOSE0**: 恒为1
- **VWAP0**: vwap/close (如果有vwap列)

### 3. 成交量因子 (5个)

- **VOLUME0-4**: 不同时间点的成交量相对值

### 4. 滚动统计因子 (139个)

**基于不同窗口(如5,10,20,30,60日)计算**:

**趋势类**:
- **ROC**: 变化率 = (price_t - price_t-n) / price_t-n
- **MA**: 移动平均
- **BETA**: 回归斜率，衡量趋势方向

**波动类**:
- **STD**: 标准差，衡量波动大小
- **RESI**: 回归残差

**极值类**:
- **MAX/MIN**: 最高价/最低价
- **QTLU**: 80%分位数
- **QTLD**: 20%分位数

**相对类**:
- **RANK**: 当前值在窗口内的排名百分位
- **RSV**: 相对强弱位置
- **IMAX/IMIN**: 最高/最低价出现的位置索引

**相关类**:
- **CORR**: 价格与成交量的相关系数
- **CORD**: 收益率与成交量变化率的相关系数

**统计类**:
- **CNTP**: 上涨天数占比
- **SUMP**: 上涨幅度之和
- **VMA**: 成交量移动平均

## 因子选择建议

### 趋势策略
推荐因子: ROC, MA, BETA, CORR
适合: 单边趋势市场

### 均值回归策略  
推荐因子: RSV, RANK, QTLU/QTLD
适合: 震荡市场

### 动量策略
推荐因子: ROC_5, ROC_10, CNTP
适合: 强趋势市场

### 波动率策略
推荐因子: STD, RESI, MAX-MIN
适合: 风险管理

## 因子组合建议

### 轻量级组合 (30-50个因子)
- 所有K线形态因子 (9个)
- 所有价格因子 (5个)
- 关键滚动因子: ROC, MA, STD, RANK (20日窗口)

### 标准组合 (100-120个因子)
- K线 + 价格 + 成交量 (19个)
- rolling_windows: [10, 20, 60]

### 完整组合 (158个因子)
- 全部因子
- rolling_windows: [5, 10, 20, 30, 60]
- 适合: 深度学习模型
"""
    
    return "资源未找到"