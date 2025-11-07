# 性能优化指南

本文档介绍如何优化 aigroup-quant-mcp 的性能，特别是在处理大数据集时。

## 1. 内存优化

### 避免不必要的数据复制

在处理大型数据集时，避免不必要的数据复制可以显著减少内存使用：

```python
# 不推荐 - 会创建数据副本
generator = Alpha158Generator(data, copy_data=True)

# 推荐 - 直接使用引用
generator = Alpha158Generator(data, copy_data=False)
```

### 及时释放内存

在处理完大数据集后，及时删除不需要的变量并调用垃圾回收：

```python
import gc

# 处理完数据后
del large_dataframe
gc.collect()
```

## 2. 并行处理

对于计算密集型任务，可以考虑使用并行处理：

```python
# 示例：并行计算多个因子
from concurrent.futures import ThreadPoolExecutor

def compute_factor(args):
    data, factor_func, params = args
    return factor_func(data, **params)

# 并行计算多个因子
with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(compute_factor, factor_args_list)
```

项目提供了专门的并行处理工具：

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

## 3. 分块处理大数据集

对于非常大的数据集，使用分块处理可以避免内存溢出：

```python
# 使用分块处理
generator = Alpha158Generator(large_data)
factors = generator.generate_all(
    chunk_size=10000  # 指定分块大小
)
```

## 4. 向量化操作

尽量使用 Pandas 和 NumPy 的向量化操作，而不是循环：

```python
# 不推荐 - 使用循环
for i in range(len(data)):
    result[i] = data[i] * 2

# 推荐 - 使用向量化操作
result = data * 2
```

## 5. 优化因子计算

### 使用高效的滚动窗口函数

```python
# 不推荐 - 使用 apply
rolling_mean = data.groupby(level=1)['close'].apply(lambda x: x.rolling(20).mean())

# 推荐 - 直接使用 rolling
rolling_mean = data.groupby(level=1)['close'].rolling(20).mean().droplevel(0)
```

### 预计算常用值

对于重复使用的计算结果，考虑将其缓存：

```python
class FactorCalculator:
    def __init__(self):
        self._cached_returns = None
        self._last_data_id = None
    
    def _calculate_returns(self, data, data_id):
        if self._last_data_id != data_id:
            self._cached_returns = data['close'].groupby(level=1).pct_change()
            self._last_data_id = data_id
        return self._cached_returns
```

## 6. 模型训练优化

### 选择合适的模型

不同模型在不同数据集上的性能差异很大：

- 对于大数据集：LightGBM 和 XGBoost 通常表现更好
- 对于小数据集：线性模型可能更合适
- 对于可解释性要求高的场景：决策树或线性模型

### 调整模型参数

```python
# LightGBM 参数优化示例
params = {
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_data_in_leaf': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8
}
```

### 使用模型缓存

项目支持模型缓存，避免重复训练：

```python
trainer = ModelTrainer(model_type='lightgbm', model_id='my_model')
trainer.train(X_train, y_train, use_cache=True)
```

## 7. 数据预处理优化

### 使用适当的数据类型

```python
# 优化数据类型以节省内存
from quantanalyzer.utils import optimize_dataframe_dtypes

data = optimize_dataframe_dtypes(data)
```

### 预先过滤数据

在进行复杂计算之前，先过滤掉不需要的数据：

```python
# 只处理特定时间段的数据
filtered_data = data.loc['2020-01-01':'2023-12-31']

# 只处理特定股票
filtered_data = data[data.index.get_level_values(1).isin(['AAPL', 'GOOGL'])]
```

## 8. 性能监控和分析

### 使用日志记录性能信息

项目中已集成日志记录功能，可以通过设置日志级别来监控性能：

```bash
# 设置日志级别为DEBUG以获取详细信息
export LOG_LEVEL=DEBUG
```

### 使用性能分析工具

```python
import cProfile
import pstats

# 分析函数性能
cProfile.run('your_function()', 'profile_output')
stats = pstats.Stats('profile_output')
stats.sort_stats('cumulative').print_stats(10)
```

### 使用性能监控装饰器

项目提供了性能监控装饰器，可以跟踪函数执行时间和资源使用情况：

```python
from quantanalyzer.monitor import profile_function

@profile_function(track_memory=True)
def compute_factors(data):
    # 因子计算逻辑
    pass
```

### 基准测试

项目包含基准测试脚本，可以用来评估性能：

```bash
# 运行基准测试
python tests/benchmark.py
```

这将运行一系列性能测试，包括：
- Alpha158因子生成
- 模型训练
- 数据处理

基准测试结果可以帮助您了解系统性能，并在进行优化后验证改进效果。

## 9. 配置优化

项目支持通过配置文件优化性能参数：

```json
{
  "chunk_size": 10000,
  "parallel_workers": 4,
  "enable_performance_monitoring": true,
  "track_memory_usage": true
}
```

您可以通过环境变量或配置文件调整这些参数以适应您的硬件环境。

## 10. 最佳实践总结

1. **内存管理**：避免不必要的数据复制，及时释放内存
2. **并行处理**：对于计算密集型任务，考虑使用并行处理
3. **分块处理**：处理大数据集时使用分块处理
4. **向量化操作**：使用 Pandas 和 NumPy 的向量化操作
5. **算法选择**：根据数据特点选择合适的算法
6. **参数调优**：根据数据特点调整模型参数
7. **数据类型优化**：使用适当的数据类型以节省内存
8. **预过滤数据**：在进行复杂计算前先过滤数据
9. **性能监控**：使用日志和基准测试监控性能
10. **配置优化**：根据硬件环境调整配置参数
11. **使用缓存**：避免重复计算，使用模型和数据缓存

通过遵循这些优化指南，可以显著提高 aigroup-quant-mcp 在处理量化分析任务时的性能。