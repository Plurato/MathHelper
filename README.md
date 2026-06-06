# MathCoach-Agent

基于多智能体协作的数学解题与教学辅导系统。系统将解题过程拆分为四个阶段，由不同 Agent 依次完成题目理解、解题规划、求解验证和教学讲解，最终输出结构化的解题过程与学习反馈。

> 详细系统设计见 [design.md](./design.md)。

## 功能概览

| Agent | 职责 | 输出要点 |
|-------|------|----------|
| **Problem Understanding**（题目理解） | 解析自然语言题目 | 题型、知识点、已知条件、求解目标、难度 |
| **Solving Planning**（解题规划） | 制定解题策略 | 推荐方法、分步计划、关键步骤、易错提醒 |
| **Solving Verification**（求解验证） | 执行详细求解并验证 | 解题步骤、最终答案、验证方式与可信度 |
| **Teaching Explanation**（教学讲解） | 生成教学化反馈 | 通俗讲解、核心知识点、常见错误、变式练习 |

流水线示意：

```
用户输入数学题
      ↓
题目理解 Agent  →  解题规划 Agent  →  求解验证 Agent  →  教学讲解 Agent
      ↓                  ↓                  ↓                  ↓
  ProblemUnderstanding  SolvingPlan   SolvingVerification  TeachingExplanation
```

## 开发进度

### 已完成

- [x] 四个 Agent 的核心实现（Prompt + Schema + Agent 类）
- [x] 基于 OpenRouter 的 LLM 调用与 JSON 结构化输出解析
- [x] 完整 4 Agent 流水线 CLI 演示（`scripts/demo_agents.py`）
- [x] 执行追踪（Trace）：Prompt、推理内容、原始响应、解析结果、Token 用量
- [x] Pydantic 数据模型与基础单元测试

### 进行中 / 待开发

- [ ] **SymPy / Python 工具调用**：求解验证 Agent 目前由 LLM 生成验证结论，尚未接入真实数学计算工具（依赖见 `pyproject.toml` 的 `[solve]` 可选包）
- [ ] **Web 前端界面**：题目输入、流程展示、结果可视化（见 `design.md` 第 10 节）
- [ ] **Agent 集成测试**：针对四个 Agent 的端到端测试（需配置 API Key）
- [ ] **扩展功能**：错题本、学生画像、RAG 知识库、多轮对话等（见 `design.md` 第 14 节）

## 快速开始

### 环境要求

- Python 3.10+
- OpenRouter API Key（[https://openrouter.ai](https://openrouter.ai)）

### 安装

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

复制环境变量模板并填入 API Key：

```bash
cp .env.example .env
# 编辑 .env，至少设置 OPENROUTER_API_KEY
```

如需后续接入 SymPy 数值验证，可额外安装：

```bash
pip install -e ".[dev,solve]"
```

### 运行演示

默认题目为一道函数最值题，会依次运行全部 4 个 Agent：

```bash
python scripts/demo_agents.py
```

指定题目与学生水平：

```bash
python scripts/demo_agents.py \
  --question "解方程 x^2 - 5x + 6 = 0。" \
  --student-level "高中"
```

指定模型（覆盖 `.env` 中的默认值）：

```bash
python scripts/demo_agents.py --model "openai/gpt-4o-mini"
```

### 演示输出说明

默认模式下，每个 Agent 会打印完整执行追踪：

- 发送给模型的 System / User Prompt
- 推理内容（模型支持时）
- LLM 原始响应
- 解析后的 JSON 与校验结果
- Token 用量

常用参数：

| 参数 | 说明 |
|------|------|
| `--question` | 待分析的数学题 |
| `--student-level` | 学生水平提示（如「高中」），供教学讲解 Agent 参考 |
| `--model` | 覆盖默认 OpenRouter 模型 |
| `--quiet` | 仅输出各 Agent 的最终 JSON 结果 |
| `--hide-prompts` | 隐藏 Prompt，保留推理与解析信息 |
| `--no-reasoning` | 不请求模型的 reasoning 内容 |

示例：

```bash
python scripts/demo_agents.py --quiet
python scripts/demo_agents.py --hide-prompts
python scripts/demo_agents.py --no-reasoning
```

## 在代码中使用

各 Agent 继承自 `BaseAgent`，可单独调用或自行编排流水线：

```python
from mathcoach.agents import (
    ProblemUnderstandingAgent,
    SolvingPlanningAgent,
    SolvingVerificationAgent,
    SolvingVerificationInput,
    TeachingExplanationAgent,
    TeachingExplanationInput,
)
from mathcoach.agents.solving_planning import SolvingPlanningInput
from mathcoach.schemas.inputs import UserQuery

query = UserQuery(question="解方程 x^2 - 5x + 6 = 0。", student_level="高中")

understanding = ProblemUnderstandingAgent().run(query)

planning = SolvingPlanningAgent().run(
    SolvingPlanningInput(analysis=understanding, original_question=query.question)
)

verification = SolvingVerificationAgent().run(
    SolvingVerificationInput(
        analysis=understanding,
        plan=planning,
        original_question=query.question,
    )
)

teaching = TeachingExplanationAgent().run(
    TeachingExplanationInput(
        analysis=understanding,
        plan=planning,
        verification=verification,
        original_question=query.question,
        student_level=query.student_level,
    )
)
```

需要调试时，使用 `run_with_trace()` 代替 `run()`，可获取完整执行追踪。

## 运行测试

```bash
pytest -q
```

当前测试覆盖 JSON 解析、Schema 校验与 Trace 打印工具，不依赖 LLM API。

## 项目结构

```
MathHelper/
├── design.md                 # 系统设计文档（架构、数据流、扩展方向）
├── scripts/
│   └── demo_agents.py        # 4 Agent 流水线 CLI 演示
├── src/mathcoach/
│   ├── agents/               # Agent 实现
│   │   ├── base.py           # BaseAgent 基类
│   │   ├── problem_understanding.py
│   │   ├── solving_planning.py
│   │   ├── solving_verification.py
│   │   └── teaching_explanation.py
│   ├── llm/                  # OpenRouter 客户端
│   ├── prompts/              # 各 Agent 的 Prompt 模板
│   ├── schemas/              # Pydantic 输入/输出模型
│   ├── utils/                # JSON 解析、Trace 打印等工具
│   └── config.py             # 环境变量配置
└── tests/                    # 单元测试
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENROUTER_API_KEY` | OpenRouter API Key | *(运行演示时必填)* |
| `OPENROUTER_BASE_URL` | OpenRouter API 地址 | `https://openrouter.ai/api/v1` |
| `MATHCOACH_DEFAULT_MODEL` | 默认 LLM 模型 | `openai/gpt-4o-mini` |
| `MATHCOACH_DEFAULT_TEMPERATURE` | 默认采样温度 | `0.2` |

## 扩展开发指南

新增或修改 Agent 时，建议遵循现有模式：

1. 在 `src/mathcoach/schemas/` 定义 Pydantic 输出模型
2. 在 `src/mathcoach/prompts/` 编写 System Prompt 与 Few-shot 示例
3. 在 `src/mathcoach/agents/` 继承 `BaseAgent`，实现 `build_user_prompt()`
4. 在 `src/mathcoach/agents/__init__.py` 导出公开 API
5. 在 `scripts/demo_agents.py` 或新的编排脚本中接入流水线
6. 补充 Schema 单元测试

下一步优先建议：

1. 为 **求解验证 Agent** 接入 SymPy，实现真实的符号计算与代入验证
2. 为新增 Schema（`SolvingVerification`、`TeachingExplanation`）补充单元测试
3. 视需求实现 Web 前端或 API 服务层

## 支持的题型范围

系统面向高中至大学低年级的基础数学题，设计上优先支持：

- 代数方程（因式分解、求根公式等）
- 函数与导数（单调性、闭区间最值等）
- 数列（等差/等比数列、求和）
- 概率与组合（古典概型等）

暂不保证处理复杂几何证明、竞赛级难题或需 OCR 的图片题目。详见 `design.md` 第 7、12 节。

## 分支说明

本项目以 **`main`** 分支为主开发分支，已合并 `master` 分支上的最新功能（完整 4 Agent 实现）。新贡献请基于 `main` 分支进行。
