const frameCanvas = document.getElementById("frameCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const frameCtx = frameCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");
let currentFrameId = 0;
let overlay = null;
let overlayConfidenceLevel = "NONE";
let overlayConfidenceScore = 0;
let apiToken = null;
let localQuestionSha256 = null;
let localQuestions = [];
let localQuestionsWritable = false;

function websocketUrl(path) {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${location.host}${path}`;
}

function reconnectingSocket(path, configure) {
  let delay = 250;
  const connect = () => {
    const socket = new WebSocket(websocketUrl(path));
    configure(socket);
    socket.onopen = () => {
      socket.send(JSON.stringify({type: "authenticate", token: apiToken}));
      delay = 250;
    };
    socket.onclose = ({code}) => {
      if (code === 1008) {
        document.getElementById("errorMessage").textContent = "本机会话已失效，请重新双击 XYQQuiz.exe 打开页面";
        return;
      }
      window.setTimeout(connect, delay);
      delay = Math.min(delay * 2, 5000);
    };
  };
  connect();
}

async function decodeFrame(data) {
  const view = new DataView(data);
  const frameId = Number(view.getBigUint64(0, false));
  const bitmap = await createImageBitmap(new Blob([data.slice(8)], {type: "image/jpeg"}));
  return {frameId, bitmap};
}

function createLatestFrameDecoder(decodeFrame, renderFrame) {
  let activeFrameDecode = false;
  let pendingFrameBuffer = null;

  async function drain() {
    activeFrameDecode = true;
    try {
      while (pendingFrameBuffer !== null) {
        const data = pendingFrameBuffer;
        pendingFrameBuffer = null;
        const {frameId, bitmap} = await decodeFrame(data);
        if (pendingFrameBuffer !== null) {
          bitmap.close();
          continue;
        }
        renderFrame(frameId, bitmap);
      }
    } finally {
      activeFrameDecode = false;
      if (pendingFrameBuffer !== null) {
        void drain();
      }
    }
  }

  return {
    enqueue(data) {
      pendingFrameBuffer = data;
      if (!activeFrameDecode) void drain();
    },
  };
}

function renderFrame(frameId, bitmap) {
  currentFrameId = frameId;
  frameCanvas.width = bitmap.width;
  frameCanvas.height = bitmap.height;
  overlayCanvas.width = bitmap.width;
  overlayCanvas.height = bitmap.height;
  document.querySelector(".canvas-stack").style.aspectRatio = `${bitmap.width} / ${bitmap.height}`;
  frameCtx.drawImage(bitmap, 0, 0);
  bitmap.close();
  document.getElementById("previewHint").hidden = true;
  drawOverlay();
}

const frameDecoder = createLatestFrameDecoder(decodeFrame, renderFrame);

function drawOverlay() {
  overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  if (!overlay) return;
  const [x, y, width, height] = overlay;
  const presentation = confidencePresentation(
    overlayConfidenceLevel,
    overlayConfidenceScore,
  );
  const lineWidth = Math.max(3, overlayCanvas.width / 400);
  const left = x * overlayCanvas.width;
  const top = y * overlayCanvas.height;
  const boxWidth = width * overlayCanvas.width;
  const boxHeight = height * overlayCanvas.height;
  const label = `${presentation.label} · 评分 ${Math.round(presentation.score)}/100`;
  const fontSize = Math.max(14, overlayCanvas.width / 90);

  overlayCtx.save();
  overlayCtx.strokeStyle = presentation.color;
  overlayCtx.fillStyle = presentation.color;
  overlayCtx.globalAlpha = presentation.alpha;
  overlayCtx.lineWidth = presentation.solid ? lineWidth * 1.6 : lineWidth;
  overlayCtx.setLineDash(
    presentation.solid ? [] : [lineWidth * 2.5, lineWidth * 1.5],
  );
  overlayCtx.strokeRect(
    left,
    top,
    boxWidth,
    boxHeight,
  );
  overlayCtx.setLineDash([]);
  overlayCtx.globalAlpha = 1;
  overlayCtx.font = `700 ${fontSize}px "Microsoft YaHei", sans-serif`;
  const labelPadding = Math.max(5, fontSize * 0.35);
  const labelWidth = overlayCtx.measureText(label).width + labelPadding * 2;
  const labelHeight = fontSize + labelPadding * 1.5;
  const labelLeft = Math.min(
    Math.max(0, left),
    Math.max(0, overlayCanvas.width - labelWidth),
  );
  const labelTop = Math.max(0, top - labelHeight - lineWidth);
  overlayCtx.fillStyle = "rgba(2, 6, 23, 0.88)";
  overlayCtx.fillRect(labelLeft, labelTop, labelWidth, labelHeight);
  overlayCtx.fillStyle = presentation.color;
  overlayCtx.fillText(
    label,
    labelLeft + labelPadding,
    labelTop + fontSize + labelPadding * 0.25,
  );
  overlayCtx.restore();
}

function normalizeConfidenceLevel(state) {
  if (["NONE", "CANDIDATE", "HIGH"].includes(state.confidence_level)) {
    return state.confidence_level;
  }
  if (state.high_confidence) return "HIGH";
  return state.overlay ? "CANDIDATE" : "NONE";
}

function normalizeConfidenceScore(value, level) {
  const numeric = Number(value);
  if (Number.isFinite(numeric)) return Math.min(100, Math.max(0, numeric));
  if (level === "HIGH") return 100;
  if (level === "CANDIDATE") return 50;
  return 0;
}

function confidencePresentation(level, score) {
  const normalizedScore = normalizeConfidenceScore(score, level);
  const hue = 120 * (1 - normalizedScore / 100);
  return {
    score: normalizedScore,
    color: `hsl(${hue.toFixed(1)}, 85%, 52%)`,
    label: level === "HIGH" ? "高可信" : "候选",
    solid: level === "HIGH",
    alpha: level === "HIGH" ? 1 : 0.68,
  };
}

function score(value, runnerUp) {
  return `${Number(value || 0).toFixed(1)} / 次高 ${Number(runnerUp || 0).toFixed(1)}`;
}

function renderSidebar(state) {
  document.getElementById("phase").textContent = state.phase || "—";
  if (state.capture) document.getElementById("capturePhase").textContent = state.capture.phase || "—";
  document.getElementById("question").textContent = state.question_text || "等待识别";
  document.getElementById("answer").textContent = state.official_answer || "—";
  document.getElementById("questionScore").textContent = score(state.question_score, state.question_runner_up_score);
  document.getElementById("optionScore").textContent = score(state.option_score, state.option_runner_up_score);
  const confidenceLevel = normalizeConfidenceLevel(state);
  const confidenceScore = normalizeConfidenceScore(
    state.confidence_score,
    confidenceLevel,
  );
  const confidenceLevelElement = document.getElementById("confidenceLevel");
  confidenceLevelElement.dataset.level = confidenceLevel;
  confidenceLevelElement.textContent = {
    NONE: "未定位",
    CANDIDATE: "低可信候选",
    HIGH: "高可信答案",
  }[confidenceLevel];
  document.getElementById("confidenceScore").textContent = `${Math.round(confidenceScore)}/100`;
  document.getElementById("confidenceReason").textContent = state.confidence_reason || "—";
  const timings = state.timings;
  document.getElementById("timings").textContent = timings
    ? `布局 ${timings.layout_ms.toFixed(1)} · OCR ${timings.ocr_ms.toFixed(1)} · 匹配 ${timings.match_ms.toFixed(1)} · 总计 ${timings.total_ms.toFixed(1)} ms`
    : "—";
}

async function runAction(button, path) {
  const error = document.getElementById("errorMessage");
  button.disabled = true;
  error.textContent = "";
  try {
    const response = await apiFetch(path);
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
  } catch (caught) {
    error.textContent = caught.message;
  } finally {
    button.disabled = false;
  }
}

async function apiFetch(path, options = {}) {
  if (!apiToken) throw new Error("本机会话尚未建立，请重新双击 XYQQuiz.exe");
  const method = options.method || "POST";
  const request = {
    method,
    headers: {
      "Content-Type": "application/json",
      "X-XYQQuiz-Token": apiToken,
    },
  };
  if (method !== "GET" && method !== "HEAD") {
    request.body = JSON.stringify(options.body || {});
  }
  return fetch(path, request);
}

async function bootstrapSession() {
  const parameters = new URLSearchParams(location.hash.slice(1));
  const bootstrapToken = parameters.get("token");
  history.replaceState(null, "", `${location.pathname}${location.search}`);
  const endpoint = bootstrapToken ? "/api/session/bootstrap" : "/api/session/restore";
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(bootstrapToken ? {token: bootstrapToken} : {}),
  });
  const result = await response.json();
  if (!response.ok || !result.ok || !result.token) {
    if (!bootstrapToken) {
      throw new Error("浏览器会话已失效，请重新双击 XYQQuiz.exe 打开页面");
    }
    throw new Error(result.error || `HTTP ${response.status}`);
  }
  apiToken = result.token;
}

async function initialize() {
  await bootstrapSession();
  for (const button of document.querySelectorAll(".actions button")) button.disabled = false;

  reconnectingSocket("/ws/frames", (frameSocket) => {
    frameSocket.binaryType = "arraybuffer";
    frameSocket.onmessage = ({data}) => {
      frameDecoder.enqueue(data);
    };
  });

  reconnectingSocket("/ws/state", (stateSocket) => {
    stateSocket.onmessage = ({data}) => {
      const state = JSON.parse(data);
      overlay = state.overlay;
      overlayConfidenceLevel = normalizeConfidenceLevel(state);
      overlayConfidenceScore = normalizeConfidenceScore(
        state.confidence_score,
        overlayConfidenceLevel,
      );
      renderSidebar(state);
      drawOverlay();
    };
  });

  const response = await apiFetch("/api/status");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  renderSidebar(await response.json());
  await loadLocalQuestions();
}

async function saveRecognitionDiagnostics(button) {
  const accepted = window.confirm(
    "识别诊断会保存当前完整游戏画面、OCR 裁剪和日志尾部，可能包含角色名、聊天或其他个人信息。确认仅保存到本机吗？"
  );
  if (!accepted) return;
  await runAction(button, "/api/diagnostics");
}

function updateLocalModeFields() {
  const override = document.getElementById("localQuestionMode").value === "override";
  const field = document.getElementById("localQuestionTargetField");
  field.hidden = !override;
}

function resetLocalQuestionForm() {
  document.getElementById("localQuestionForm").reset();
  document.getElementById("localQuestionId").value = "";
  document.getElementById("localQuestionEnabled").checked = true;
  document.getElementById("localQuestionCancel").hidden = true;
  updateLocalModeFields();
}

function editLocalQuestion(record) {
  document.getElementById("localQuestionId").value = record.id;
  document.getElementById("localQuestionMode").value = record.mode;
  document.getElementById("localQuestionTarget").value = record.target_source_id || "";
  document.getElementById("localQuestionText").value = record.question;
  document.getElementById("localQuestionAnswer").value = record.answer;
  document.getElementById("localQuestionAliases").value = (record.answer_aliases || []).join("\n");
  document.getElementById("localQuestionEnabled").checked = record.enabled;
  document.getElementById("localQuestionCancel").hidden = false;
  updateLocalModeFields();
  document.querySelector(".local-bank-panel").open = true;
}

function renderLocalQuestions(result) {
  localQuestionSha256 = result.sha256 || null;
  localQuestions = result.records || [];
  localQuestionsWritable = Boolean(result.writable);
  document.getElementById("localQuestionCount").textContent = `${localQuestions.length} 条`;
  document.getElementById("localQuestionSave").disabled = !result.writable;
  const list = document.getElementById("localQuestionList");
  list.replaceChildren();

  for (const record of localQuestions) {
    const item = document.createElement("article");
    item.className = "local-question-item";
    item.dataset.enabled = String(record.enabled);
    const question = document.createElement("p");
    question.textContent = record.question;
    const answer = document.createElement("p");
    answer.className = "local-answer";
    answer.textContent = `答案：${record.answer}`;
    const meta = document.createElement("p");
    meta.className = "local-meta";
    const mode = record.mode === "override" ? "覆盖" : "补充";
    const target = record.target_source_id ? ` · ${record.target_source_id}` : "";
    meta.textContent = `${mode}${target} · ${record.enabled ? "已启用" : "已停用"}`;
    const actions = document.createElement("div");
    actions.className = "local-item-actions";
    const edit = document.createElement("button");
    edit.type = "button";
    edit.className = "secondary";
    edit.textContent = "编辑";
    edit.addEventListener("click", () => editLocalQuestion(record));
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "secondary";
    remove.textContent = "删除";
    remove.addEventListener("click", () => deleteLocalQuestion(record));
    actions.append(edit, remove);
    item.append(question, answer, meta, actions);
    list.append(item);
  }
}

function localQuestionWarnings(result) {
  return [...(result.conflicts || []), ...(result.issues || [])]
    .map((item) => item.message)
    .join("；");
}

async function loadLocalQuestions() {
  const error = document.getElementById("localQuestionError");
  try {
    const response = await apiFetch("/api/local-questions", {method: "GET"});
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
    renderLocalQuestions(result);
    error.textContent = localQuestionWarnings(result);
  } catch (caught) {
    error.textContent = caught.message;
    localQuestionsWritable = false;
    document.getElementById("localQuestionSave").disabled = true;
  }
}

function localQuestionFormPayload() {
  const mode = document.getElementById("localQuestionMode").value;
  return {
    sha256: localQuestionSha256,
    mode,
    question: document.getElementById("localQuestionText").value.trim(),
    answer: document.getElementById("localQuestionAnswer").value.trim(),
    target_source_id: mode === "override"
      ? document.getElementById("localQuestionTarget").value.trim()
      : null,
    answer_aliases: document.getElementById("localQuestionAliases").value
      .split(/\r?\n/)
      .map((value) => value.trim())
      .filter(Boolean),
    enabled: document.getElementById("localQuestionEnabled").checked,
  };
}

async function saveLocalQuestion(event) {
  event.preventDefault();
  const button = document.getElementById("localQuestionSave");
  const error = document.getElementById("localQuestionError");
  const recordId = document.getElementById("localQuestionId").value;
  button.disabled = true;
  error.textContent = "";
  try {
    const path = recordId
      ? `/api/local-questions/${encodeURIComponent(recordId)}`
      : "/api/local-questions";
    const response = await apiFetch(path, {
      method: recordId ? "PUT" : "POST",
      body: localQuestionFormPayload(),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
    renderLocalQuestions(result);
    resetLocalQuestionForm();
    error.textContent = localQuestionWarnings(result);
  } catch (caught) {
    const message = caught.message;
    await loadLocalQuestions();
    error.textContent = message;
  } finally {
    button.disabled = !localQuestionsWritable;
  }
}

async function deleteLocalQuestion(record) {
  if (!window.confirm(`确认删除本地题目“${record.question}”吗？`)) return;
  const error = document.getElementById("localQuestionError");
  try {
    const response = await apiFetch(
      `/api/local-questions/${encodeURIComponent(record.id)}`,
      {method: "DELETE", body: {sha256: localQuestionSha256}},
    );
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
    renderLocalQuestions(result);
    resetLocalQuestionForm();
    error.textContent = localQuestionWarnings(result);
  } catch (caught) {
    const message = caught.message;
    await loadLocalQuestions();
    error.textContent = message;
  }
}

document.getElementById("updateButton").addEventListener("click", ({currentTarget}) => runAction(currentTarget, "/api/question-bank/update"));
document.getElementById("diagnosticsButton").addEventListener("click", ({currentTarget}) => saveRecognitionDiagnostics(currentTarget));
document.getElementById("environmentDiagnosticsButton").addEventListener("click", ({currentTarget}) => runAction(currentTarget, "/api/environment-diagnostics"));
document.getElementById("shutdownButton").addEventListener("click", ({currentTarget}) => runAction(currentTarget, "/api/shutdown"));
document.getElementById("localQuestionMode").addEventListener("change", updateLocalModeFields);
document.getElementById("localQuestionForm").addEventListener("submit", saveLocalQuestion);
document.getElementById("localQuestionCancel").addEventListener("click", resetLocalQuestionForm);

initialize()
  .catch((error) => { document.getElementById("errorMessage").textContent = error.message; });
