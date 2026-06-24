const stages = [
  ["understanding", "题目理解", "ProblemUnderstandingAgent"],
  ["planning", "解题规划", "SolvingPlanningAgent"],
  ["verification", "求解验证", "SolvingVerificationAgent"],
  ["teaching", "教学讲解", "TeachingExplanationAgent"],
];

// Result section <-> agent stage mapping for progressive rendering.
const SECTION_FOR_STAGE = {
  understanding: "analysis",
  planning: "plan",
  verification: "solution",
  teaching: "teaching",
};

const els = {};
let state = freshState();

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

  renderFlow();
  renderIdleWorkspace();
  drawCurve();
  checkHealth();
  els.form.addEventListener("submit", handleSubmit);
});

function freshState() {
  return {
    phase: "idle", // idle | running | done | error
    stageStatus: {}, // key -> "running" | "succeeded" | "failed"
    stageMeta: {}, // key -> { attempt, duration_s, agent_name }
    current: null, // currently running stage key
    outputs: {}, // sectionId -> output payload
    retries: [], // retry events
    planningAttempts: 1,
    result: null,
    error: null,
  };
}

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

  state = freshState();
  state.phase = "running";
  els.solveButton.disabled = true;
  els.solveButton.textContent = "解题中";
  render();

  try {
    await streamSolve(payload);
  } catch (error) {
    state.phase = "error";
    state.error = { message: readError(error) };
    showError(readError(error));
    render();
  } finally {
    els.solveButton.disabled = false;
    els.solveButton.textContent = "开始解题";
  }
}

async function streamSolve(payload) {
  const response = await fetch("/api/solve/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    let data;
    try {
      data = await response.json();
    } catch (parseError) {
      data = { detail: { message: `请求失败（${response.status}）。` } };
    }
    throw data;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  for (;;) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    let separator;
    while ((separator = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, separator);
      buffer = buffer.slice(separator + 2);
      const event = parseEventFrame(frame);
      if (event) {
        handleEvent(event);
      }
    }
  }
}

function parseEventFrame(frame) {
  const dataLines = frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trimStart());
  if (!dataLines.length) {
    return null;
  }
  try {
    return JSON.parse(dataLines.join("\n"));
  } catch (error) {
    return null;
  }
}

function handleEvent(event) {
  switch (event.type) {
    case "stage_started":
      state.current = event.key;
      state.stageStatus[event.key] = "running";
      state.stageMeta[event.key] = {
        attempt: event.attempt,
        agent_name: event.agent_name,
      };
      break;
    case "stage_completed":
      state.stageStatus[event.key] = "succeeded";
      state.stageMeta[event.key] = {
        attempt: event.attempt,
        duration_s: event.duration_s,
        agent_name: event.agent_name,
      };
      if (state.current === event.key) {
        state.current = null;
      }
      {
        const sectionId = SECTION_FOR_STAGE[event.key];
        if (sectionId) {
          state.outputs[sectionId] = event.output;
        }
      }
      break;
    case "stage_failed":
      state.stageStatus[event.key] = "failed";
      if (state.current === event.key) {
        state.current = null;
      }
      break;
    case "retry":
      state.retries.push(event);
      state.planningAttempts = event.attempt;
      break;
    case "done":
      state.phase = "done";
      state.result = event.result;
      state.current = null;
      break;
    case "error":
      state.phase = "error";
      state.error = event;
      showError(event.message || "解题失败，请稍后重试。");
      break;
    default:
      break;
  }
  render();
}

function render() {
  renderFlow();
  if (state.phase === "idle") {
    renderIdleWorkspace();
  } else if (state.phase === "done" && state.result) {
    renderResult(state.result);
  } else {
    renderProgress();
  }
}

function renderFlow() {
  const retryCount = state.retries.length;
  const flowHtml = stages
    .map(([key, label, agent], index) => {
      const status = state.stageStatus[key] || "idle";
      const meta = state.stageMeta[key] || {};
      const attemptBadge =
        meta.attempt && meta.attempt > 1
          ? `<span class="stage-badge">第 ${meta.attempt} 次</span>`
          : "";
      return `
        <article class="stage-card ${escapeHtml(status)}">
          <div class="stage-index">${statusGlyph(status, index + 1)}</div>
          <div>
            <div class="stage-title">${label}${attemptBadge}</div>
            <div class="stage-agent">${agent}</div>
          </div>
          <div class="stage-status">${escapeHtml(flowStatusText(key, status))}</div>
        </article>
      `;
    })
    .join("");
  const retryNote =
    retryCount > 0
      ? `<div class="flow-retry-note">已触发 ${retryCount} 次重新规划</div>`
      : "";
  els.agentFlow.innerHTML = flowHtml + retryNote;
}

function statusGlyph(status, index) {
  if (status === "succeeded") {
    return "✓";
  }
  if (status === "failed") {
    return "!";
  }
  if (status === "running") {
    return '<span class="pulse-dot"></span>';
  }
  return String(index);
}

function flowStatusText(key, status) {
  const meta = state.stageMeta[key] || {};
  if (status === "succeeded") {
    return typeof meta.duration_s === "number"
      ? `完成 · ${formatSeconds(meta.duration_s)}`
      : "完成";
  }
  if (status === "running") {
    return "运行中…";
  }
  if (status === "failed") {
    return "失败";
  }
  return "等待";
}

function renderIdleWorkspace() {
  els.resultWorkspace.innerHTML = `
    <div class="empty-state">
      <p class="eyebrow">Ready</p>
      <h2>等待题目</h2>
      <p>提交后，这里会实时显示每个 Agent 的运行进度与中间结果。</p>
    </div>
  `;
}

function renderProgress() {
  els.resultWorkspace.innerHTML = `
    <div class="workspace-grid">
      <nav class="workspace-nav" aria-label="结果导航">
        ${workspaceNav()}
      </nav>
      <div class="workspace-main">
        ${renderStatusBanner()}
        ${renderRetryNotices()}
        ${renderSectionShell("analysis", "题目分析", "understanding", renderAnalysisBody)}
        ${renderSectionShell("plan", "解题规划", "planning", renderPlanBody)}
        ${renderSectionShell("solution", "详细步骤与验证", "verification", renderSolutionBody)}
        ${renderSectionShell("teaching", "教学讲解", "teaching", renderTeachingBody)}
      </div>
    </div>
  `;
  renderMath(els.resultWorkspace);
}

function workspaceNav() {
  const items = [
    ["analysis", "题目分析"],
    ["plan", "解题规划"],
    ["solution", "详细步骤"],
    ["teaching", "教学讲解"],
  ];
  return items.map(([id, title]) => `<a href="#${id}">${title}</a>`).join("");
}

function renderStatusBanner() {
  if (state.phase === "error") {
    return `
      <div class="status-banner error">
        <span class="status-banner-dot"></span>
        <div>
          <strong>执行中断</strong>
          <span>${escapeHtml(state.error?.message || "解题失败。")}</span>
        </div>
      </div>
    `;
  }
  const runningKey = state.current;
  if (runningKey) {
    const label = stageLabel(runningKey);
    const meta = state.stageMeta[runningKey] || {};
    const attempt =
      meta.attempt && meta.attempt > 1 ? `（第 ${meta.attempt} 次尝试）` : "";
    return `
      <div class="status-banner running">
        <span class="pulse-dot"></span>
        <div>
          <strong>${escapeHtml(label)} 正在运行${attempt}</strong>
          <span>Agent 正在生成中间结果，请稍候…</span>
        </div>
      </div>
    `;
  }
  return `
    <div class="status-banner">
      <span class="status-banner-dot"></span>
      <div>
        <strong>正在协作求解</strong>
        <span>各 Agent 顺序运行，结果将逐步呈现。</span>
      </div>
    </div>
  `;
}

function renderRetryNotices() {
  if (!state.retries.length) {
    return "";
  }
  const items = state.retries
    .map((retry) => {
      const failed = (retry.failed_assertions || [])
        .map(
          (item) =>
            `<li><code>${escapeHtml(item.expr || "")}</code> 期望 <code>${escapeHtml(
              String(item.expected ?? "")
            )}</code>${item.detail ? ` — ${escapeHtml(item.detail)}` : ""}</li>`
        )
        .join("");
      return `
        <div class="retry-item">
          <div class="retry-head">第 ${retry.attempt - 1} 次求解未通过验证，正在重新规划（第 ${retry.attempt} 次）</div>
          ${retry.detail ? `<p>${escapeHtml(retry.detail)}</p>` : ""}
          ${failed ? `<ul class="retry-list">${failed}</ul>` : ""}
        </div>
      `;
    })
    .join("");
  return `<div class="retry-panel"><div class="retry-title">验证反馈循环</div>${items}</div>`;
}

function renderSectionShell(id, title, stageKey, bodyRenderer) {
  const sectionId = SECTION_FOR_STAGE[stageKey];
  const status = sectionStatus(stageKey);
  const data = state.outputs[sectionId];
  let body;
  if (status === "done" && data) {
    body = bodyRenderer(data);
  } else if (status === "running") {
    body = renderSkeleton();
  } else {
    body = `<p class="section-pending">等待 ${escapeHtml(title)} Agent 运行…</p>`;
  }
  return `
    <section class="content-section ${status}" id="${id}">
      <div class="section-head">
        <h3>${title}</h3>
        <span class="section-pill ${status}">${sectionStatusText(status)}</span>
      </div>
      ${body}
    </section>
  `;
}

function sectionStatus(stageKey) {
  const sectionId = SECTION_FOR_STAGE[stageKey];
  if (state.outputs[sectionId]) {
    return "done";
  }
  if (state.current === stageKey || state.stageStatus[stageKey] === "running") {
    return "running";
  }
  if (state.stageStatus[stageKey] === "failed") {
    return "failed";
  }
  return "pending";
}

function sectionStatusText(status) {
  if (status === "done") {
    return "已完成";
  }
  if (status === "running") {
    return "运行中";
  }
  if (status === "failed") {
    return "失败";
  }
  return "等待";
}

function renderSkeleton() {
  return `
    <div class="skeleton-block">
      <span class="skeleton-line w-80"></span>
      <span class="skeleton-line w-60"></span>
      <span class="skeleton-line w-70"></span>
    </div>
  `;
}

function stageLabel(key) {
  const found = stages.find(([k]) => k === key);
  return found ? found[1] : key;
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
        ${renderResultSummaryBanner(result)}
        ${renderRetryNotices()}
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
  renderMath(els.resultWorkspace);
}

function renderResultSummaryBanner(result) {
  const status = result.verification.verification.status;
  const passed = status === "passed";
  const attempts = result.planning_attempts || 1;
  const attemptText = attempts > 1 ? `，共规划 ${attempts} 次` : "";
  return `
    <div class="status-banner ${passed ? "success" : "warn"}">
      <span class="status-banner-dot"></span>
      <div>
        <strong>${passed ? "求解完成且验证通过" : "求解完成，但验证未通过"}</strong>
        <span>四个 Agent 已全部运行完毕${attemptText} · 用时 ${formatSeconds(
          result.duration_s
        )}</span>
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
    <section class="content-section done" id="answer">
      <div class="section-head"><h3>最终答案</h3></div>
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
    <section class="content-section done" id="analysis">
      <div class="section-head"><h3>题目分析</h3></div>
      ${renderAnalysisBody(analysis)}
    </section>
  `;
}

function renderAnalysisBody(analysis) {
  return `
    <div class="kv-grid">
      ${kv("题型", analysis.problem_type)}
      ${kv("难度", analysis.difficulty)}
      ${kv("目标", analysis.goal)}
      ${kv("知识点", (analysis.knowledge_points || []).join("、"))}
    </div>
    <h4 class="sub-head">已知条件</h4>
    ${renderObject(analysis.conditions)}
  `;
}

function renderPlanSection(plan) {
  return `
    <section class="content-section done" id="plan">
      <div class="section-head"><h3>解题规划</h3></div>
      ${renderPlanBody(plan)}
    </section>
  `;
}

function renderPlanBody(plan) {
  return `
    ${kv("推荐方法", plan.method)}
    ${renderList(plan.steps, "section-list")}
    ${plan.alternative_method ? kv("备用方法", plan.alternative_method) : ""}
    ${renderNamedList("关键步骤", plan.key_steps)}
    ${renderNamedList("易错提醒", plan.warnings)}
  `;
}

function renderSolutionSection(verification) {
  return `
    <section class="content-section done" id="solution">
      <div class="section-head"><h3>详细步骤与验证</h3></div>
      ${renderSolutionBody(verification)}
    </section>
  `;
}

function renderSolutionBody(verification) {
  return `
    ${renderList(verification.solution_steps, "section-list")}
    <h4 class="sub-head">工具验证</h4>
    <div class="kv-grid">
      ${kv("方法", verification.verification.method)}
      ${kv("状态", verification.verification.status)}
      ${kv("可信度", formatConfidence(verification.verification.confidence))}
      ${kv("断言数量", String((verification.assertions || []).length))}
    </div>
  `;
}

function renderTeachingSection(teaching) {
  return `
    <section class="content-section done" id="teaching">
      <div class="section-head"><h3>教学讲解</h3></div>
      ${renderTeachingBody(teaching)}
    </section>
  `;
}

function renderTeachingBody(teaching) {
  return `
    <p>${escapeHtml(teaching.explanation || "")}</p>
    ${renderNamedList("核心知识点", teaching.key_points)}
    ${renderNamedList("常见错误", teaching.common_mistakes)}
    ${renderNamedList("变式练习", teaching.practice_questions)}
    ${teaching.learning_advice ? kv("学习建议", teaching.learning_advice) : ""}
  `;
}

function renderTraceSection(result) {
  return `
    <section class="content-section done" id="trace">
      <div class="section-head"><h3>执行详情</h3></div>
      <div class="kv-grid">
        ${kv("规划次数", String(result.planning_attempts || 1))}
        ${kv("Prompt Tokens", String(result.usage.prompt_tokens || 0))}
        ${kv("Completion Tokens", String(result.usage.completion_tokens || 0))}
        ${kv("总 Tokens", String(result.usage.total_tokens || 0))}
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
  return `<h4 class="sub-head">${title}</h4>${renderList(values, "section-list")}`;
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

function renderMath(container) {
  if (typeof window.renderMathInElement !== "function" || !container) {
    return;
  }
  window.renderMathInElement(container, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    ignoredTags: [
      "script",
      "noscript",
      "style",
      "textarea",
      "pre",
      "code",
      "option",
    ],
    throwOnError: false,
  });
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
