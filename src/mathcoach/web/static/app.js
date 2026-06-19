const stages = [
  ["understanding", "题目理解", "ProblemUnderstandingAgent"],
  ["planning", "解题规划", "SolvingPlanningAgent"],
  ["verification", "求解验证", "SolvingVerificationAgent"],
  ["teaching", "教学讲解", "TeachingExplanationAgent"],
];

const els = {};

document.addEventListener("DOMContentLoaded", () => {
  Object.assign(els, {
    form: document.getElementById("solveForm"),
    question: document.getElementById("questionInput"),
    studentLevel: document.getElementById("studentLevel"),
    explanationStyle: document.getElementById("explanationStyle"),
    model: document.getElementById("modelInput"),
    solveButton: document.getElementById("solveButton"),
    agentFlow: document.getElementById("agentFlow"),
    resultWorkspace: document.getElementById("resultWorkspace"),
    errorPanel: document.getElementById("errorPanel"),
    healthStatus: document.getElementById("healthStatus"),
    canvas: document.getElementById("functionCanvas"),
  });

  renderStages();
  drawCurve();
  checkHealth();
  els.form.addEventListener("submit", handleSubmit);
});

async function checkHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    els.healthStatus.textContent = data.openrouter_api_key_present
      ? `环境就绪 · ${data.default_model}`
      : "缺少 API Key";
    els.healthStatus.className = data.openrouter_api_key_present
      ? "health-pill ready"
      : "health-pill warn";
  } catch (error) {
    els.healthStatus.textContent = "服务未就绪";
    els.healthStatus.className = "health-pill warn";
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  hideError();

  const payload = {
    question: els.question.value.trim(),
    student_level: els.studentLevel.value || null,
    explanation_style: els.explanationStyle.value || null,
    model: els.model.value.trim() || null,
  };

  if (!payload.question) {
    showError("请输入一道数学题。");
    els.question.focus();
    return;
  }

  els.solveButton.disabled = true;
  els.solveButton.textContent = "解题中";
  renderStages(null, "running");
  renderLoading();

  try {
    const result = await solveProblem(payload);
    renderStages(result.stages);
    renderResult(result);
  } catch (error) {
    renderStages(error.detail?.stages, "failed");
    showError(readError(error));
  } finally {
    els.solveButton.disabled = false;
    els.solveButton.textContent = "开始解题";
  }
}

async function solveProblem(payload) {
  const response = await fetch("/api/solve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw data;
  }
  return data;
}

function renderStages(resultStages = null, fallbackStatus = "idle") {
  const byKey = new Map((resultStages || []).map((stage) => [stage.key, stage]));
  els.agentFlow.innerHTML = stages
    .map(([key, label, agent], index) => {
      const stage = byKey.get(key);
      const status = stage?.status || fallbackStatus;
      const statusText = stage
        ? formatStageStatus(stage)
        : fallbackStatus === "running"
          ? "运行中"
          : fallbackStatus === "failed"
            ? "未完成"
            : "等待";
      return `
        <article class="stage-card ${escapeHtml(status)}">
          <div class="stage-index">${index + 1}</div>
          <div>
            <div class="stage-title">${label}</div>
            <div class="stage-agent">${agent}</div>
          </div>
          <div class="stage-status">${escapeHtml(statusText)}</div>
        </article>
      `;
    })
    .join("");
}

function formatStageStatus(stage) {
  if (stage.status === "succeeded") {
    return `完成 · ${formatSeconds(stage.duration_s)}`;
  }
  if (stage.status === "failed") {
    return "失败";
  }
  return stage.status;
}

function renderLoading() {
  els.resultWorkspace.innerHTML = `
    <div class="empty-state">
      <p class="eyebrow">Solving</p>
      <h2>Agent 正在协作</h2>
      <p>题目理解、解题规划、求解验证和教学讲解会顺序完成。</p>
    </div>
  `;
}

function renderResult(result) {
  const sections = [
    ["answer", "最终答案"],
    ["analysis", "题目分析"],
    ["plan", "解题规划"],
    ["solution", "详细步骤"],
    ["teaching", "教学讲解"],
    ["trace", "执行详情"],
  ];

  els.resultWorkspace.innerHTML = `
    <div class="workspace-grid">
      <nav class="workspace-nav" aria-label="结果导航">
        ${sections.map(([id, title]) => `<a href="#${id}">${title}</a>`).join("")}
      </nav>
      <div class="workspace-main">
        ${renderAnswerStrip(result)}
        ${renderAnswerSection(result)}
        ${renderAnalysisSection(result.analysis)}
        ${renderPlanSection(result.plan)}
        ${renderSolutionSection(result.verification)}
        ${renderTeachingSection(result.teaching)}
        ${renderTraceSection(result)}
      </div>
    </div>
  `;
}

function renderAnswerStrip(result) {
  const verification = result.verification.verification;
  return `
    <div class="answer-strip">
      <div class="metric-card">
        <div class="metric-label">验证状态</div>
        <div class="metric-value ${statusClass(verification.status)}">
          ${escapeHtml(verification.status)}
        </div>
      </div>
      <div class="metric-card">
        <div class="metric-label">可信度</div>
        <div class="metric-value">${formatConfidence(verification.confidence)}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">耗时</div>
        <div class="metric-value">${formatSeconds(result.duration_s)}</div>
      </div>
    </div>
  `;
}

function renderAnswerSection(result) {
  const answers = result.verification.answer || [];
  return `
    <section class="content-section" id="answer">
      <h3>最终答案</h3>
      ${
        answers.length
          ? `<div class="answer-list">${answers.map(renderAnswerItem).join("")}</div>`
          : "<p>本题未返回可结构化展示的答案。</p>"
      }
      <p>${escapeHtml(result.verification.verification.detail || "")}</p>
    </section>
  `;
}

function renderAnswerItem(item) {
  const value = item.latex || item.sympy || item.numeric || "";
  const meta = [item.sympy ? `SymPy: ${item.sympy}` : "", item.unit || ""]
    .filter(Boolean)
    .join(" · ");
  return `
    <div class="answer-item">
      <div class="answer-label">${escapeHtml(item.label)}</div>
      <div class="answer-value">${escapeHtml(String(value))}</div>
      ${meta ? `<div class="stage-agent">${escapeHtml(meta)}</div>` : ""}
    </div>
  `;
}

function renderAnalysisSection(analysis) {
  return `
    <section class="content-section" id="analysis">
      <h3>题目分析</h3>
      <div class="kv-grid">
        ${kv("题型", analysis.problem_type)}
        ${kv("难度", analysis.difficulty)}
        ${kv("目标", analysis.goal)}
        ${kv("知识点", (analysis.knowledge_points || []).join("、"))}
      </div>
      <h3 style="margin-top:18px;">已知条件</h3>
      ${renderObject(analysis.conditions)}
    </section>
  `;
}

function renderPlanSection(plan) {
  return `
    <section class="content-section" id="plan">
      <h3>解题规划</h3>
      ${kv("推荐方法", plan.method)}
      ${renderList(plan.steps, "section-list")}
      ${plan.alternative_method ? kv("备用方法", plan.alternative_method) : ""}
      ${renderNamedList("关键步骤", plan.key_steps)}
      ${renderNamedList("易错提醒", plan.warnings)}
    </section>
  `;
}

function renderSolutionSection(verification) {
  return `
    <section class="content-section" id="solution">
      <h3>详细步骤</h3>
      ${renderList(verification.solution_steps, "section-list")}
      <h3 style="margin-top:18px;">工具验证</h3>
      <div class="kv-grid">
        ${kv("方法", verification.verification.method)}
        ${kv("状态", verification.verification.status)}
        ${kv("可信度", formatConfidence(verification.verification.confidence))}
        ${kv("断言数量", String((verification.assertions || []).length))}
      </div>
    </section>
  `;
}

function renderTeachingSection(teaching) {
  return `
    <section class="content-section" id="teaching">
      <h3>教学讲解</h3>
      <p>${escapeHtml(teaching.explanation || "")}</p>
      ${renderNamedList("核心知识点", teaching.key_points)}
      ${renderNamedList("常见错误", teaching.common_mistakes)}
      ${renderNamedList("变式练习", teaching.practice_questions)}
      ${teaching.learning_advice ? kv("学习建议", teaching.learning_advice) : ""}
    </section>
  `;
}

function renderTraceSection(result) {
  return `
    <section class="content-section" id="trace">
      <h3>执行详情</h3>
      <div class="kv-grid">
        ${kv("Prompt Tokens", String(result.usage.prompt_tokens || 0))}
        ${kv("Completion Tokens", String(result.usage.completion_tokens || 0))}
      </div>
      ${(result.trace || []).map(renderTraceEntry).join("")}
    </section>
  `;
}

function renderTraceEntry(entry) {
  return `
    <details class="trace-block">
      <summary>${escapeHtml(entry.agent_name)}</summary>
      <pre>${escapeHtml(JSON.stringify(entry.steps || [], null, 2))}</pre>
    </details>
  `;
}

function renderObject(obj) {
  const entries = Object.entries(obj || {});
  if (!entries.length) {
    return "<p>无结构化条件。</p>";
  }
  return `<div class="kv-grid">${entries.map(([key, value]) => kv(key, value)).join("")}</div>`;
}

function renderNamedList(title, values) {
  if (!values || !values.length) {
    return "";
  }
  return `<h3 style="margin-top:18px;">${title}</h3>${renderList(values, "section-list")}`;
}

function renderList(values, className) {
  if (!values || !values.length) {
    return "<p>暂无。</p>";
  }
  return `<ol class="${className}">${values.map((item) => `<li>${escapeHtml(String(item))}</li>`).join("")}</ol>`;
}

function kv(key, value) {
  return `
    <div class="kv-item">
      <div class="kv-key">${escapeHtml(String(key))}</div>
      <div class="kv-value">${escapeHtml(String(value || "暂无"))}</div>
    </div>
  `;
}

function showError(message) {
  els.errorPanel.hidden = false;
  els.errorPanel.textContent = message;
}

function hideError() {
  els.errorPanel.hidden = true;
  els.errorPanel.textContent = "";
}

function readError(error) {
  const detail = error.detail || error;
  if (Array.isArray(detail)) {
    return detail.map((item) => item.msg).join("；");
  }
  return detail.message || detail.code || "解题失败，请稍后重试。";
}

function statusClass(status) {
  return `status-${String(status || "").toLowerCase()}`;
}

function formatConfidence(value) {
  if (typeof value !== "number") {
    return "暂无";
  }
  return `${Math.round(value * 100)}%`;
}

function formatSeconds(value) {
  if (typeof value !== "number") {
    return "0.00s";
  }
  return `${value.toFixed(2)}s`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function drawCurve() {
  const canvas = els.canvas;
  if (!canvas) {
    return;
  }
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfaf5";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d9ded8";
  ctx.lineWidth = 1;
  for (let x = 40; x < width; x += 48) {
    line(ctx, x, 20, x, height - 24);
  }
  for (let y = 28; y < height; y += 36) {
    line(ctx, 28, y, width - 24, y);
  }
  ctx.strokeStyle = "#69736f";
  line(ctx, 28, height / 2, width - 24, height / 2);
  line(ctx, width / 2, 20, width / 2, height - 24);

  ctx.beginPath();
  for (let px = 28; px <= width - 24; px += 4) {
    const x = (px - width / 2) / 54;
    const y = Math.sin(x) * 34 + 0.08 * x * x * 24;
    const py = height / 2 - y;
    if (px === 28) {
      ctx.moveTo(px, py);
    } else {
      ctx.lineTo(px, py);
    }
  }
  ctx.strokeStyle = "#1f7a6b";
  ctx.lineWidth = 3;
  ctx.stroke();

  ctx.fillStyle = "#315f8f";
  ctx.beginPath();
  ctx.arc(width * 0.67, height * 0.36, 4, 0, Math.PI * 2);
  ctx.fill();
}

function line(ctx, x1, y1, x2, y2) {
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
}
