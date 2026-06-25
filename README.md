# MathCoach-Agent

> 基于多智能体协作的数学解题与教学辅导系统。

MathCoach-Agent 将数学解题过程拆分为四个阶段，由不同 Agent 依次完成**题目理解 → 解题规划 → 求解验证 → 教学讲解**，最终输出结构化的解题过程与学习反馈。

系统的核心特点是 **「LLM 推理 + 符号计算双重保障」**：求解结果会经过 SymPy 符号/数值验证，一旦验证失败，流水线会自动携带失败原因**重新规划并重新求解**（自我纠错回路），从而显著提升答案可靠性。

项目同时提供两种使用方式：

- **CLI 演示**：命令行一键运行四 Agent 流水线，打印完整执行追踪。
- **本地 Web 应用**：浏览器输入题目，借助 SSE **实时流式**展示各阶段进度与中间结果，并通过 KaTeX 渲染数学公式。

> 完整测试套件包含 **128 个单元测试**，覆盖解析、Schema、验证器、流水线编排与 Web 路由，且不依赖任何 LLM API。详细系统设计见 [design.md](./design.md)。

## ✨ 主要特性

- 🧩 **四 Agent 协作流水线**：题目理解、解题规划、求解验证、教学讲解，各司其职、可单独调用。
- 🔁 **求解自我纠错回路**：验证失败时自动回退到规划阶段，携带失败断言重试（最多 2 次），无需人工介入。
- ✅ **SymPy 符号验证**：基于 `Assertion` 原语的多层断言校验（symbolic / numeric / sampling 三层兜底），用工具结果覆盖 LLM 自评置信度。
- 🌐 **本地 Web 应用**：FastAPI 后端 + 零依赖前端，SSE 实时推送各阶段进度，KaTeX 渲染数学公式。
- 🔍 **全链路执行追踪**：Prompt、推理内容、原始响应、解析结果、Token 用量逐步可见，便于调试与教学。
- 🧮 **LaTeX 输出规范化**：所有展示字段强制 `$...$` 包裹的 LaTeX，并自动修复 JSON 转义冲突。
- 📊 **端到端评测框架**：对 JSON 题集批量评测并生成报告。

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
题目理解 Agent ──→ 解题规划 Agent ──→ 求解验证 Agent ──→ 教学讲解 Agent
      │                  ↑                  │                  │
      │                  └──────────────────┘                  │
      │              验证失败时携带反馈重新规划（最多 2 次）      │
      ↓                  ↓                  ↓                  ↓
ProblemUnderstanding  SolvingPlan   SolvingVerification  TeachingExplanation
```

本地 Web 应用在同一套流水线上提供：

- 浏览器 UI：题目输入、学生水平 / 讲解风格、模型选择
- **SSE 实时流式输出**：各阶段开始 / 完成、验证重试、最终结果逐条推送
- 分阶段结果展示与执行追踪（Trace）
- KaTeX 数学公式渲染
- REST API：`GET /api/health`、`POST /api/solve`、`POST /api/solve/stream`

## 开发进度

**核心功能已全部开发完成 ✅**

| 模块 | 状态 | 说明 |
|------|------|------|
| 四 Agent 核心实现 | ✅ | Prompt + Schema + Agent 类，继承统一 `BaseAgent` |
| OpenRouter LLM 调用 | ✅ | 结构化 JSON 输出解析与 Schema 校验 |
| CLI 演示 | ✅ | `scripts/demo_agents.py`，完整 4 Agent 流水线 |
| Pipeline 编排层 | ✅ | `src/mathcoach/pipeline.py`，汇总阶段状态与 Token 用量 |
| **求解自我纠错回路** | ✅ | 验证失败自动携带失败断言重新规划 + 重新求解（最多 2 次） |
| **SymPy 符号验证** | ✅ | `Assertion` 原语多层断言（symbolic / numeric / sampling），覆盖 LLM 置信度 |
| 数学输出 LaTeX 规范化 | ✅ | 展示字段强制 `$...$` LaTeX；自动修复 JSON 转义冲突 |
| 本地 Web 应用 | ✅ | FastAPI 后端 + 静态前端（`scripts/serve_web.py`） |
| **Web SSE 实时流式输出** | ✅ | `POST /api/solve/stream` 逐阶段推送进度与中间结果 |
| 全链路执行追踪（Trace） | ✅ | Prompt、推理、原始响应、解析结果、Token 用量；区分 LLM 步与 Tool 步 |
| 端到端评测框架 | ✅ | `scripts/evaluate.py`、`src/mathcoach/eval/` |
| 单元测试套件 | ✅ | 128 个测试，覆盖解析 / Schema / 验证器 / 流水线 / Web 路由 |

### 未来扩展方向

以下为可选的后续增强项，不影响当前功能的完整性（详见 `design.md` 第 14 节）：

- 针对四个 Agent 的在线集成测试（需配置 API Key）
- 错题本、学生画像、RAG 知识库、多轮对话等教学增强能力
- Web UI 历史记录与会话管理

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

运行 Web 应用或 SymPy 验证时，建议安装可选依赖：

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

## 运行本地 Web 应用

本地 Web 应用会启动 FastAPI 服务，同时提供浏览器界面和 REST API。

```bash
source .venv/bin/activate
pip install -e ".[dev,solve]"
python scripts/serve_web.py
```

默认地址为 `http://127.0.0.1:8000`。运行前请确认 `.env` 中已设置 `OPENROUTER_API_KEY`。

常用启动参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 绑定地址 | `127.0.0.1` |
| `--port` | 绑定端口 | `8000` |
| `--reload` | 开发模式下代码变更自动重载 | 关闭 |

示例：

```bash
python scripts/serve_web.py --port 8001
python scripts/serve_web.py --reload
```

### Web API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 浏览器 UI |
| `/api/health` | GET | 检查服务状态与 API Key 是否已配置 |
| `/api/solve` | POST | 同步运行完整 4 Agent 流水线，一次性返回结果 |
| `/api/solve/stream` | POST | 以 SSE（`text/event-stream`）逐步推送各阶段进度与最终结果 |

`POST /api/solve` 与 `POST /api/solve/stream` 共用同一请求体：

```json
{
  "question": "解方程 x^2 - 5x + 6 = 0。",
  "student_level": "高中",
  "explanation_style": null,
  "model": null,
  "include_trace": true
}
```

`/api/solve/stream` 返回的事件类型包括：

| 事件 `type` | 触发时机 |
|-------------|----------|
| `stage_started` | 某个 Agent 阶段开始执行 |
| `stage_completed` | 某个阶段成功完成（携带该阶段输出） |
| `retry` | 求解验证失败，触发重新规划（携带失败断言） |
| `stage_failed` / `error` | 某阶段失败 |
| `done` | 全部完成，携带完整 `PipelineResult` |

## 运行评测

对 JSON 题集运行端到端评测，并生成报告：

```bash
python scripts/evaluate.py --limit 5
python scripts/evaluate.py --problems data/eval/dev.json --full
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--problems` | 题集 JSON 路径 | `data/eval/dev.json` |
| `--limit` | 仅运行前 N 题 | 全部 |
| `--full` | 包含教学讲解 Agent（完整 4 Agent） | 关闭 |
| `--output-dir` | 结果输出目录 | `output/eval` |
| `--model` | 覆盖默认 OpenRouter 模型 | 使用 `.env` 默认值 |

## 在代码中使用

### 使用 Pipeline 编排

推荐通过 `run_mathcoach_pipeline` 一次性运行完整流水线：

```python
from mathcoach.pipeline import run_mathcoach_pipeline
from mathcoach.schemas.inputs import UserQuery

query = UserQuery(question="解方程 x^2 - 5x + 6 = 0。", student_level="高中")
result = run_mathcoach_pipeline(query)

print(result.analysis)
print(result.plan)
print(result.verification)
print(result.teaching)
print(result.stages)            # 各阶段耗时与状态
print(result.usage)             # Token 汇总
print(result.planning_attempts) # 规划尝试次数（含自我纠错重试）
```

如需实时进度（如自建 UI 或日志），可传入 `on_event` 回调，它会在每个阶段开始 / 完成、验证重试以及最终完成时被调用：

```python
def on_event(event: dict) -> None:
    print(event["type"], event.get("key") or event.get("attempt"))

result = run_mathcoach_pipeline(query, on_event=on_event)
```

### 单独调用 Agent

各 Agent 继承自 `BaseAgent`，也可单独调用或自行编排：

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

当前共 **128 个单元测试**，覆盖 JSON 解析、Schema 校验、SymPy 验证器、Pipeline 编排（含自我纠错回路）、Web API 路由（含 SSE 流式）、评测加载 / 运行 / 评分 / 报告与 Trace 打印工具，全部不依赖 LLM API。

## 项目结构

```
MathHelper/
├── design.md                 # 系统设计文档（架构、数据流、扩展方向）
├── data/eval/                # 评测题集
├── scripts/
│   ├── demo_agents.py        # 4 Agent 流水线 CLI 演示
│   ├── evaluate.py           # 端到端评测入口
│   └── serve_web.py          # 本地 Web 应用启动脚本
├── src/mathcoach/
│   ├── agents/               # Agent 实现
│   │   ├── base.py           # BaseAgent 基类
│   │   ├── problem_understanding.py
│   │   ├── solving_planning.py
│   │   ├── solving_verification.py
│   │   └── teaching_explanation.py
│   ├── eval/                 # 评测加载、运行、评分与报告
│   ├── llm/                  # OpenRouter 客户端
│   ├── pipeline.py           # 四 Agent 流水线编排
│   ├── prompts/              # 各 Agent 的 Prompt 模板
│   ├── schemas/              # Pydantic 输入/输出模型
│   ├── tools/                # SymPy 验证器等工具
│   ├── utils/                # JSON 解析、Trace 打印等工具
│   ├── web/                  # FastAPI 应用与静态前端
│   └── config.py             # 环境变量配置
└── tests/                    # 单元测试
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENROUTER_API_KEY` | OpenRouter API Key | *(运行演示或 Web 应用时必填)* |
| `OPENROUTER_BASE_URL` | OpenRouter API 地址 | `https://openrouter.ai/api/v1` |
| `MATHCOACH_DEFAULT_MODEL` | 默认 LLM 模型 | `openai/gpt-4o-mini` |
| `MATHCOACH_DEFAULT_TEMPERATURE` | 默认采样温度 | `0.2` |

## 扩展开发指南

新增或修改 Agent 时，建议遵循现有模式：

1. 在 `src/mathcoach/schemas/` 定义 Pydantic 输出模型
2. 在 `src/mathcoach/prompts/` 编写 System Prompt 与 Few-shot 示例
3. 在 `src/mathcoach/agents/` 继承 `BaseAgent`，实现 `build_user_prompt()`
4. 在 `src/mathcoach/agents/__init__.py` 导出公开 API
5. 在 `src/mathcoach/pipeline.py` 或 `scripts/demo_agents.py` 中接入流水线
6. 补充 Schema 与 Pipeline 单元测试；Web 行为变更时更新 `tests/test_web_app.py`

可选的后续增强方向：

1. 补充 **Agent 集成测试**（需 API Key 的端到端用例）
2. 视需求扩展 Web UI（历史记录、多轮对话等）
3. 接入教学增强能力（错题本、学生画像、RAG 知识库）

## 支持的题型范围

系统面向高中至大学低年级的基础数学题，设计上优先支持：

- 代数方程（因式分解、求根公式等）
- 函数与导数（单调性、闭区间最值等）
- 数列（等差/等比数列、求和）
- 概率与组合（古典概型等）

暂不保证处理复杂几何证明、竞赛级难题或需 OCR 的图片题目。详见 `design.md` 第 7、12 节。

## 分支说明

本项目以 **`main`** 分支为主开发分支。功能开发完成后合并至 `main`；新贡献请基于 `main` 分支进行。
