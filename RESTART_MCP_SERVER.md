# 如何重启aigroup-quant-mcp服务器

## 问题说明

修改了Python代码后，MCP服务器仍在运行旧版本的代码，因为：
1. MCP服务器是独立运行的进程
2. 启动时会加载Python模块到内存
3. 代码修改后需要重启服务器才能生效

## 重启方法

### 方法1: 在VSCode/Roo中重启（推荐）

1. 打开VSCode或Roo
2. 找到MCP连接面板
3. 断开aigroup-quant-mcp连接
4. 重新连接aigroup-quant-mcp

### 方法2: 手动重启进程

1. 找到MCP服务器进程：
   ```bash
   # Windows
   tasklist | findstr python
   
   # Linux/Mac
   ps aux | grep "quantanalyzer.mcp"
   ```

2. 终止进程：
   ```bash
   # Windows
   taskkill /F /PID <进程ID>
   
   # Linux/Mac
   kill <进程ID>
   ```

3. 重新启动MCP服务器（通常由客户端自动重启）

### 方法3: 重启客户端应用

如果上述方法不起作用，可以：
1. 关闭VSCode/Roo/Claude Desktop
2. 重新打开应用
3. MCP服务器会自动启动新进程

## 验证修复是否生效

重启后，再次运行以下测试：

```bash
python test_fix_verification.py
```

或者通过MCP工具测试：

```python
# 测试单因子计算
{
  "data_id": "maotai_final_test",
  "factor_name": "test_momentum",
  "factor_type": "momentum", 
  "period": 20
}
```

## 预期结果

修复后应该返回：
```json
{
  "status": "success",
  "message": "✅ 因子 'test_momentum' 计算完成",
  "data_quality": {
    "quality_score": "优秀"
  },
  "tips": [
    "💡 因子类型: momentum，周期: 20天",
    "💡 数据质量: 优秀",
    "💡 建议先评估IC再决定是否使用此因子"
  ]
}
```

而不是：
```json
{
  "status": "error",
  "message": "因子计算失败: cannot access local variable 'result'..."
}
```

## 已修复的文件

- `quantanalyzer/mcp/handlers.py` (第158行, 第293行, 第389-408行)

## 修复内容

1. **UnboundLocalError修复**: 将`quality_score`提前计算，避免在构建result时引用自身
2. **GBK编码错误修复**: 移除了包含emoji的print语句

## 联系支持

如果重启后问题仍然存在，请检查：
1. 是否使用了正确的项目目录
2. Python包是否需要重新安装
3. 是否有多个Python环境导致加载了错误的版本