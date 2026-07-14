const frameCanvas = document.getElementById("frameCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const frameCtx = frameCanvas.getContext("2d");
const overlayCtx = overlayCanvas.getContext("2d");
let currentFrameId = 0;
let overlay = null;
let apiToken = null;

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
  overlayCtx.strokeStyle = "#22c55e";
  overlayCtx.lineWidth = Math.max(3, overlayCanvas.width / 400);
  overlayCtx.strokeRect(
    x * overlayCanvas.width,
    y * overlayCanvas.height,
    width * overlayCanvas.width,
    height * overlayCanvas.height,
  );
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

async function apiFetch(path) {
  if (!apiToken) throw new Error("本机会话尚未建立，请重新双击 XYQQuiz.exe");
  return fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-XYQQuiz-Token": apiToken,
    },
    body: "{}",
  });
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
      renderSidebar(state);
      drawOverlay();
    };
  });

  const response = await apiFetch("/api/status");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  renderSidebar(await response.json());
}

async function saveRecognitionDiagnostics(button) {
  const accepted = window.confirm(
    "识别诊断会保存当前完整游戏画面、OCR 裁剪和日志尾部，可能包含角色名、聊天或其他个人信息。确认仅保存到本机吗？"
  );
  if (!accepted) return;
  await runAction(button, "/api/diagnostics");
}

document.getElementById("updateButton").addEventListener("click", ({currentTarget}) => runAction(currentTarget, "/api/question-bank/update"));
document.getElementById("diagnosticsButton").addEventListener("click", ({currentTarget}) => saveRecognitionDiagnostics(currentTarget));
document.getElementById("environmentDiagnosticsButton").addEventListener("click", ({currentTarget}) => runAction(currentTarget, "/api/environment-diagnostics"));
document.getElementById("shutdownButton").addEventListener("click", ({currentTarget}) => runAction(currentTarget, "/api/shutdown"));

initialize()
  .catch((error) => { document.getElementById("errorMessage").textContent = error.message; });
