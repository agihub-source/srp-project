# 贡献指南

感谢你对丝路协议(SRP)项目的关注！本文档将指导你如何为项目贡献代码和文档。

## 目录

- [行为准则](#行为准则)
- [开始贡献](#开始贡献)
- [开发流程](#开发流程)
- [代码规范](#代码规范)
- [提交规范](#提交规范)
- [文档规范](#文档规范)
- [测试规范](#测试规范)
- [版本规范](#版本规范)
- [问题反馈](#问题反馈)

## 行为准则

我们希望所有贡献者能够：

- 保持友善和尊重
- 接受建设性的批评
- 关注问题本身
- 为社区创造价值

## 开始贡献

### 1. 环境准备

```bash
# 克隆仓库
git clone https://github.com/yourusername/srp.git
cd srp

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -e ".[dev]"
```

### 2. 分支管理

- `main`: 主分支，保持稳定
- `develop`: 开发分支，最新特性
- `feature/*`: 新功能分支
- `bugfix/*`: 修复分支
- `release/*`: 发布分支

### 3. 工作流程

1. Fork项目
2. 创建特性分支
3. 提交修改
4. 发起Pull Request
5. 等待审查和合并

## 开发流程

### 1. 功能开发

```bash
# 创建特性分支
git checkout -b feature/your-feature

# 开发完成后
git add .
git commit -m "feat: add new feature"
git push origin feature/your-feature
```

### 2. Bug修复

```bash
# 创建修复分支
git checkout -b bugfix/issue-number

# 修复完成后
git add .
git commit -m "fix: resolve issue #123"
git push origin bugfix/issue-number
```

## 代码规范

### 1. Python代码风格

- 使用[Black](https://github.com/psf/black)格式化代码
- 遵循[PEP 8](https://www.python.org/dev/peps/pep-0008/)规范
- 使用类型注解
- 编写文档字符串

```python
from typing import List, Optional

def example_function(param1: str, param2: Optional[int] = None) -> List[str]:
    """
    函数功能说明
    
    Args:
        param1: 参数1说明
        param2: 参数2说明
    
    Returns:
        返回值说明
    
    Raises:
        ValueError: 错误说明
    """
    pass
```

### 2. 代码质量要求

- 使用flake8进行代码检查
- 使用mypy进行类型检查
- 保持代码简洁明了
- 避免重复代码

## 提交规范

### 1. 提交消息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

### 2. Type类型

- feat: 新功能
- fix: 修复
- docs: 文档
- style: 格式
- refactor: 重构
- test: 测试
- chore: 其他

### 3. 示例

```
feat(network): add new routing algorithm

- Implement Kademlia DHT
- Add node discovery
- Optimize path finding

Closes #123
```

## 文档规范

### 1. 代码文档

- 模块级别文档
- 类和方法文档
- 复杂逻辑说明
- 示例代码

### 2. 项目文档

- README.md
- API文档
- 架构文档
- 示例文档

## 测试规范

### 1. 测试要求

- 单元测试覆盖率 > 80%
- 包含集成测试
- 性能测试
- 边界测试

### 2. 测试示例

```python
import unittest

class TestExample(unittest.TestCase):
    def setUp(self):
        # 测试准备
        pass
        
    def test_function(self):
        # 测试用例
        result = example_function()
        self.assertEqual(result, expected)
        
    def tearDown(self):
        # 测试清理
        pass
```

## 版本规范

遵循[语义化版本](https://semver.org/lang/zh-CN/)规范：

- 主版本号：不兼容的API修改
- 次版本号：向下兼容的功能性新增
- 修订号：向下兼容的问题修复

## 问题反馈

### 1. Issue模板

```markdown
## 问题描述

[详细描述问题]

## 复现步骤

1. [步骤1]
2. [步骤2]
3. [步骤3]

## 期望行为

[描述期望的行为]

## 实际行为

[描述实际的行为]

## 环境信息

- OS: [操作系统]
- Python版本: [版本号]
- SRP版本: [版本号]
```

### 2. Pull Request模板

```markdown
## 相关Issue

[例如: #123]

## 修改说明

[描述你的修改]

## 检查列表

- [ ] 代码格式化
- [ ] 通过所有测试
- [ ] 更新文档
- [ ] 添加测试用例
```

## 其他注意事项

1. 保持分支最新
```bash
git remote add upstream https://github.com/original/srp.git
git fetch upstream
git rebase upstream/main
```

2. 解决冲突
```bash
git rebase -i upstream/main
# 解决冲突后
git rebase --continue
```

3. 代码审查
- 回应所有评论
- 及时更新代码
- 保持礼貌和专业