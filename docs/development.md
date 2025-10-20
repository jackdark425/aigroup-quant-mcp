# 开发指南

## 项目结构

```
aigroup-quant-mcp/
├── quantanalyzer/           # 核心模块
│   ├── __init__.py
│   ├── data/               # 数据处理模块
│   ├── factor/             # 因子计算模块
│   ├── model/              # 模型训练模块
│   ├── backtest/           # 回测引擎模块
│   └── mcp.py             # MCP服务入口
├── tests/                  # 测试文件
├── docs/                   # 文档
├── examples/               # 示例代码
├── dist/                   # 构建产物
├── README.md
├── pyproject.toml         # 项目配置
├── .gitignore
└── requirements.txt
```

## 开发环境设置

### 使用 uv (推荐)

```bash
# 克隆项目
git clone <repository-url>
cd aigroup-quant-mcp

# 创建虚拟环境并安装依赖
uv venv
uv pip install -e .

# 安装开发依赖
uv pip install -e .[dev]
```

### 使用 pip

```bash
# 克隆项目
git clone <repository-url>
cd aigroup-quant-mcp

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -e .
pip install -e .[dev]
```

## 代码规范

项目遵循 PEP 8 代码规范。使用 `flake8` 进行代码检查：

```bash
# 安装 flake8
pip install flake8

# 检查代码规范
flake8 quantanalyzer/
```

## 构建项目

使用 `uv` 构建项目：

```bash
uv build
```

构建产物将位于 `dist/` 目录下。

## 运行测试

```bash
# 运行所有测试
python -m unittest discover tests

# 运行特定测试
python -m unittest tests.test_data
```

## 运行 MCP 服务

```bash
# 直接运行
python -m quantanalyzer.mcp

# 使用 uvx 运行（无需安装）
uvx aigroup-quant-mcp
```

## 添加新功能

1. 在相应的模块中实现功能
2. 添加单元测试
3. 更新文档
4. 运行测试确保没有破坏现有功能
5. 提交 Pull Request