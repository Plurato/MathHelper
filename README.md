# MathCoach-Agent

> 基于多智能体协作的数学解题与教学辅导系统。

MathCoach-Agent 将数学解题拆分为四个阶段，由不同 Agent 依次完成**题目理解 → 解题规划 → 求解验证 → 教学讲解**，并通过本地 Web 应用输出结构化的解题过程与学习反馈。

核心特点是 **「LLM 推理 + 符号计算双重保障」**：求解结果会经过 SymPy 符号 / 数值验证，一旦验证失败，流水线会自动携带失败原因**重新规划并重新求解**（自我纠错回路），显著提升答案可靠性。

> 详细系统设计见 [design.md](./design.md)。

## ✨ 主要特性

- 🧩 **四 Agent 协作流水线**：题目理解、解题规划、求解验证、教学讲解。
- 🔁 **求解自我纠错回路**：验证失败时自动携带失败断言重新规划并求解（最多 2 次）。
- ✅ **SymPy 符号验证**：基于 `Assertion` 原语的多层断言校验（symbolic / numeric / sampling 三层兜底）。
- 🌐 **本地 Web 应用**：FastAPI 后端 + 零依赖前端，SSE 实时推送各阶段进度，KaTeX 渲染公式。
- 🔍 **全链路执行追踪**：Prompt、推理、原始响应、解析结果、Token 用量逐步可见。
- 🧮 **LaTeX 输出规范化**：展示字段强制 `$...$` 包裹的 LaTeX，并自动修复 JSON 转义冲突。

> 核心功能已全部开发完成，并由 **128 个单元测试**覆盖（解析 / Schema / 验证器 / 流水线 / Web 路由 / 评测），均不依赖 LLM API。

## 流水线概览

| Agent | 职责 | 输出要点 |
|-------|------|----------|
| **题目理解** Problem Understanding | 解析自然语言题目 | 题型、知识点、已知条件、求解目标、难度 |
| **解题规划** Solving Planning | 制定解题策略 | 推荐方法、分步计划、关键步骤、易错提醒 |
| **求解验证** Solving Verification | 执行求解并用 SymPy 验证 | 解题步骤、最终答案、验证方式与可信度 |
| **教学讲解** Teaching Explanation | 生成教学化反馈 | 通俗讲解、核心知识点、常见错误、变式练习 |

```
用户输入数学题
      ↓
题目理解 ──→ 解题规划 ──→ 求解验证 ──→ 教学讲解
                ↑            │
                └────────────┘
        验证失败时携带反馈重新规划（最多 2 次）
```

## 快速开始

### 环境要求

- Python 3.10+
- OpenRouter API Key（[https://openrouter.ai](https://openrouter.ai)）

### 安装与配置

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev,solve]"

cp .env.example .env               # 编辑 .env，至少设置 OPENROUTER_API_KEY
```

### 启动 Web 应用

```bash
python scripts/serve_web.py
```

启动后访问 `http://127.0.0.1:8000`，在浏览器输入题目即可查看四 Agent 流水线的实时结果（含数学公式渲染与执行追踪）。

常用启动参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--host` | 绑定地址 | `127.0.0.1` |
| `--port` | 绑定端口 | `8000` |
| `--reload` | 开发模式下代码变更自动重载 | 关闭 |

## Web API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 浏览器 UI |
| `/api/health` | GET | 检查服务状态与 API Key 是否已配置 |
| `/api/solve` | POST | 同步运行完整流水线，一次性返回结果 |
| `/api/solve/stream` | POST | 以 SSE 逐步推送各阶段进度与最终结果 |

`/api/solve` 与 `/api/solve/stream` 共用同一请求体：

```json
{
  "question": "解方程 x^2 - 5x + 6 = 0。",
  "student_level": "高中",
  "explanation_style": null,
  "model": null,
  "include_trace": true
}
```

`/api/solve/stream` 推送的事件类型：`stage_started`（阶段开始）、`stage_completed`（阶段完成，含输出）、`retry`（验证失败触发重新规划）、`stage_failed` / `error`（失败）、`done`（全部完成，含完整结果）。

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `OPENROUTER_API_KEY` | OpenRouter API Key | *(运行 Web 应用时必填)* |
| `OPENROUTER_BASE_URL` | OpenRouter API 地址 | `https://openrouter.ai/api/v1` |
| `MATHCOACH_DEFAULT_MODEL` | 默认 LLM 模型 | `openai/gpt-4o-mini` |
| `MATHCOACH_DEFAULT_TEMPERATURE` | 默认采样温度 | `0.2` |

## 支持的题型范围

面向高中至大学低年级的基础数学题，优先支持：代数方程、函数与导数、数列、概率与组合。暂不保证复杂几何证明、竞赛级难题或需 OCR 的图片题目。详见 `design.md` 第 7、12 节。

## 开发

```bash
pytest -q                                   # 运行单元测试（128 个，不依赖 LLM API）
python scripts/evaluate.py --limit 5        # 对 data/eval/dev.json 运行端到端评测
```

项目采用 `src/` 布局，核心代码位于 `src/mathcoach/`：

```
src/mathcoach/
├── agents/      # 四个 Agent 实现（继承统一 BaseAgent）
├── prompts/     # 各 Agent 的 Prompt 模板
├── schemas/     # Pydantic 输入 / 输出模型
├── llm/         # OpenRouter 客户端
├── tools/       # SymPy 验证器等工具
├── eval/        # 评测加载、运行、评分与报告
├── web/         # FastAPI 应用与静态前端
├── utils/       # JSON 解析、Trace 打印等工具
├── pipeline.py  # 四 Agent 流水线编排（含自我纠错回路）
└── config.py    # 环境变量配置
```

新增或修改 Agent 时，依次在 `schemas/` 定义输出模型、在 `prompts/` 编写 Prompt、在 `agents/` 继承 `BaseAgent` 实现 `build_user_prompt()`、在 `pipeline.py` 中接入流水线，并补充对应单元测试。

## 分支说明

本项目以 **`main`** 分支为主开发分支，新贡献请基于 `main` 分支进行。
