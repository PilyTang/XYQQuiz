const frameCanvas = document.getElementById("frameCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const canvasStack = document.querySelector(".canvas-stack");
const previewHint = document.getElementById("previewHint");
const frameCtx = frameCanvas.getContext("2d", {alpha: false});
const overlayCtx = overlayCanvas.getContext("2d");
const sidebarElements = {
  phase: document.getElementById("phase"),
  capturePhase: document.getElementById("capturePhase"),
  question: document.getElementById("question"),
  answer: document.getElementById("answer"),
  questionScore: document.getElementById("questionScore"),
  optionScore: document.getElementById("optionScore"),
  confidenceLevel: document.getElementById("confidenceLevel"),
  confidenceScore: document.getElementById("confidenceScore"),
  confidenceReason: document.getElementById("confidenceReason"),
  timings: document.getElementById("timings"),
};
let currentFrameId = 0;
let overlay = null;
let overlayConfidenceLevel = "NONE";
let overlayConfidenceScore = 0;
let apiToken = null;
let localQuestionSha256 = null;
let localQuestions = [];
let localQuestionsWritable = false;
let performanceSnapshot = null;
let renderedFrames = 0;
let fpsWindowStarted = performance.now();
let lastCanvasFps = null;
let previewMode = "i420";
let videoDecoder = null;
const videoFrameIds = new Map();

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
        document.getElementById("errorMessage").textContent = "本机会话已失效，请关闭当前页面并重新打开 XYQQuiz";
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
  const jpeg = new Uint8Array(data, 8);
  const bitmap = await createImageBitmap(new Blob([jpeg], {type: "image/jpeg"}));
  return {frameId, bitmap};
}

async function decodeI420Frame(data) {
  if (!("VideoFrame" in window)) {
    throw new Error("当前 WebView2 不支持 WebCodecs VideoFrame");
  }
  const view = new DataView(data);
  if (view.byteLength <= 24) throw new Error("I420 预览帧数据不完整");
  const frameId = Number(view.getBigUint64(0, false));
  const timestamp = Number(view.getBigInt64(8, false));
  const width = view.getUint32(16, false);
  const height = view.getUint32(20, false);
  const payload = new Uint8Array(data, 24);
  if (payload.byteLength !== width * height * 3 / 2) {
    throw new Error("I420 预览帧尺寸不匹配");
  }
  const bitmap = new VideoFrame(payload, {
    format: "I420",
    codedWidth: width,
    codedHeight: height,
    timestamp,
  });
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
  const bitmapWidth = bitmap.displayWidth || bitmap.width;
  const bitmapHeight = bitmap.displayHeight || bitmap.height;
  const canvasSizeChanged = (
    frameCanvas.width !== bitmapWidth
    || frameCanvas.height !== bitmapHeight
    || overlayCanvas.width !== bitmapWidth
    || overlayCanvas.height !== bitmapHeight
  );
  if (canvasSizeChanged) {
    frameCanvas.width = bitmapWidth;
    frameCanvas.height = bitmapHeight;
    overlayCanvas.width = bitmapWidth;
    overlayCanvas.height = bitmapHeight;
    canvasStack.style.aspectRatio = `${bitmapWidth} / ${bitmapHeight}`;
  }
  frameCtx.drawImage(bitmap, 0, 0);
  bitmap.close();
  renderedFrames += 1;
  updateCanvasFps();
  if (!previewHint.hidden) previewHint.hidden = true;
  if (canvasSizeChanged) drawOverlay();
}

function closeVideoDecoder() {
  if (videoDecoder && videoDecoder.state !== "closed") videoDecoder.close();
  videoDecoder = null;
  videoFrameIds.clear();
}

function configureH264Decoder(codec) {
  closeVideoDecoder();
  if (!("VideoDecoder" in window) || !("EncodedVideoChunk" in window)) {
    throw new Error("当前 WebView2 不支持 WebCodecs，已无法启用硬件预览");
  }
  videoDecoder = new VideoDecoder({
    output(frame) {
      const frameId = videoFrameIds.get(frame.timestamp) || currentFrameId + 1;
      videoFrameIds.delete(frame.timestamp);
      renderFrame(frameId, frame);
    },
    error(error) {
      document.getElementById("errorMessage").textContent = `硬件预览解码失败：${error.message}`;
    },
  });
  videoDecoder.configure({
    codec,
    optimizeForLatency: true,
    hardwareAcceleration: "prefer-hardware",
  });
}

function decodeH264Frame(data) {
  if (!videoDecoder || videoDecoder.state !== "configured") return;
  const view = new DataView(data);
  if (view.byteLength <= 25) throw new Error("硬件预览帧数据不完整");
  const keyFrame = (view.getUint8(0) & 1) !== 0;
  const frameId = Number(view.getBigUint64(1, false));
  const timestamp = Number(view.getBigInt64(9, false));
  videoFrameIds.set(timestamp, frameId);
  videoDecoder.decode(new EncodedVideoChunk({
    type: keyFrame ? "key" : "delta",
    timestamp,
    data: new Uint8Array(data, 25),
  }));
}

async function decodeNV12Frame(data) {
  if (!("VideoFrame" in window)) {
    throw new Error("当前 WebView2 不支持 WebCodecs VideoFrame");
  }
  const view = new DataView(data);
  if (view.byteLength <= 25) throw new Error("NV12 预览帧数据不完整");
  const frameId = Number(view.getBigUint64(1, false));
  const timestamp = Number(view.getBigInt64(9, false));
  const width = view.getUint32(17, false);
  const height = view.getUint32(21, false);
  const payload = new Uint8Array(data, 25);
  if (payload.byteLength !== width * height * 3 / 2) {
    throw new Error("NV12 预览帧尺寸不匹配");
  }
  const bitmap = new VideoFrame(payload, {
    format: "NV12",
    codedWidth: width,
    codedHeight: height,
    timestamp,
  });
  return {frameId, bitmap};
}

async function decodeBGRAFrame(data) {
  if (!("VideoFrame" in window)) {
    throw new Error("当前 WebView2 不支持 WebCodecs VideoFrame");
  }
  const view = new DataView(data);
  if (view.byteLength <= 25) throw new Error("BGRA 预览帧数据不完整");
  const frameId = Number(view.getBigUint64(1, false));
  const timestamp = Number(view.getBigInt64(9, false));
  const width = view.getUint32(17, false);
  const height = view.getUint32(21, false);
  const payload = new Uint8Array(data, 25);
  if (payload.byteLength !== width * height * 4) {
    throw new Error("BGRA 预览帧尺寸不匹配");
  }
  const bitmap = new VideoFrame(payload, {
    format: "BGRA",
    codedWidth: width,
    codedHeight: height,
    timestamp,
  });
  return {frameId, bitmap};
}

function configurePreviewMessage(message) {
  if (message.type !== "preview-config") return;
  const nativePreview = message.mode === "native";
  canvasStack.classList.toggle("native-preview-active", nativePreview);
  previewHint.hidden = nativePreview;
  if (nativePreview) {
    closeVideoDecoder();
    previewMode = "native";
    lastCanvasFps = null;
    void reportNativePreviewLayout();
  } else if (message.mode === "h264") {
    configureH264Decoder(message.codec || "avc1.42E01E");
    previewMode = "h264";
  } else if (message.mode === "nv12") {
    closeVideoDecoder();
    previewMode = "nv12";
  } else if (message.mode === "bgra") {
    closeVideoDecoder();
    previewMode = "bgra";
  } else if (message.mode === "i420") {
    closeVideoDecoder();
    previewMode = "i420";
  } else {
    closeVideoDecoder();
    previewMode = "jpeg";
  }
}

let nativeLayoutTimer = null;

async function reportNativePreviewLayout() {
  // Report the slot before the native renderer is marked healthy.  Its first
  // successful Present is what promotes the backend to native mode, so gating
  // this request on previewMode would create a startup deadlock.
  if (!apiToken) return;
  const rect = canvasStack.getBoundingClientRect();
  const visible = (
    document.visibilityState === "visible"
    && rect.width > 0
    && rect.height > 0
  );
  try {
    await apiFetch("/api/preview/layout", {
      body: {
        x: Math.round(rect.left),
        y: Math.round(rect.top),
        width: Math.round(rect.width),
        height: Math.round(rect.height),
        scale: window.devicePixelRatio || 1,
        visible,
      },
    });
  } catch (error) {
    document.getElementById("errorMessage").textContent = error.message;
  }
}

function scheduleNativePreviewLayout() {
  if (nativeLayoutTimer !== null) window.clearTimeout(nativeLayoutTimer);
  nativeLayoutTimer = window.setTimeout(() => {
    nativeLayoutTimer = null;
    void reportNativePreviewLayout();
  }, 50);
}

async function reportNativeOverlay() {
  if (previewMode !== "native" || !apiToken) return;
  const levels = {NONE: 0, CANDIDATE: 1, HIGH: 2};
  try {
    await apiFetch("/api/preview/overlay", {
      body: {
        rect: overlay,
        score: overlayConfidenceScore,
        level: levels[overlayConfidenceLevel] || 0,
      },
    });
  } catch (_) {
    // The state WebSocket remains authoritative; a transient overlay update
    // failure must not interrupt recognition or preview rendering.
  }
}

function updateCanvasFps() {
  const now = performance.now();
  const elapsed = now - fpsWindowStarted;
  if (elapsed < 1000) return;
  lastCanvasFps = renderedFrames * 1000 / elapsed;
  renderedFrames = 0;
  fpsWindowStarted = now;
  renderBackendStatus();
  if (apiToken) {
    void apiFetch("/api/performance/canvas-fps", {
      body: {fps: lastCanvasFps},
    }).catch(() => {});
  }
}

function shortBackendLabel(state, capability) {
  if (!state) return capability === "ocr" ? "OCR —" : "预览 —";
  if (state.effective === "cpu") {
    return capability === "ocr" ? "OCR CPU" : "预览 CPU";
  }
  if (state.effective.startsWith("directml:")) return "OCR DirectML";
  if (state.effective.startsWith("windows_hardware:")) return "预览 硬件";
  return state.label || state.effective;
}

function renderBackendStatus() {
  const element = document.getElementById("backendStatus");
  if (!performanceSnapshot) {
    element.textContent = "OCR CPU · 预览 CPU · — FPS";
    return;
  }
  const fps = Number.isFinite(lastCanvasFps)
    ? lastCanvasFps
    : performanceSnapshot.canvas_fps;
  const fpsLabel = Number.isFinite(fps) ? `${fps.toFixed(1)} FPS` : "— FPS";
  element.textContent = [
    shortBackendLabel(performanceSnapshot.ocr, "ocr"),
    shortBackendLabel(performanceSnapshot.preview, "preview"),
    fpsLabel,
  ].join(" · ");
  const reasons = [
    performanceSnapshot.ocr?.fallback_reason,
    performanceSnapshot.preview?.fallback_reason,
  ].filter(Boolean);
  element.title = reasons.length
    ? `当前已回退：${reasons.join("；")}`
    : "当前实际执行后端与 Canvas 绘制帧率";
}

function fillBackendSelect(select, options, selected) {
  select.replaceChildren();
  for (const backend of options || []) {
    const option = document.createElement("option");
    option.value = backend.value;
    option.textContent = backend.available
      ? backend.label
      : `${backend.label}（不可用）`;
    option.disabled = !backend.selectable;
    option.title = backend.reason || "";
    select.append(option);
  }
  if ([...select.options].some((option) => option.value === selected)) {
    select.value = selected;
  }
}

function renderPerformanceDialog({preserveSelection = false} = {}) {
  if (!performanceSnapshot) return;
  const ocrSelect = document.getElementById("ocrBackendSelect");
  const previewSelect = document.getElementById("previewBackendSelect");
  const selectedOcr = preserveSelection
    ? ocrSelect.value
    : performanceSnapshot.pending_ocr;
  const selectedPreview = preserveSelection
    ? previewSelect.value
    : performanceSnapshot.pending_preview;
  fillBackendSelect(
    ocrSelect,
    performanceSnapshot.ocr_options,
    selectedOcr,
  );
  fillBackendSelect(
    previewSelect,
    performanceSnapshot.preview_options,
    selectedPreview,
  );
  const ocrReason = performanceSnapshot.ocr.fallback_reason
    ? `；回退：${performanceSnapshot.ocr.fallback_reason}`
    : "";
  const previewReason = performanceSnapshot.preview.fallback_reason
    ? `；回退：${performanceSnapshot.preview.fallback_reason}`
    : "";
  document.getElementById("ocrBackendCurrent").textContent = `当前：${performanceSnapshot.ocr.label}${ocrReason}`;
  document.getElementById("previewBackendCurrent").textContent = `当前：${performanceSnapshot.preview.label}${previewReason}`;
  const probeLabels = {
    idle: "等待自检",
    probing: "正在后台自检，不影响当前答题",
    ready: "可用设备自检完成",
  };
  document.getElementById("backendProbeStatus").textContent = `自检状态：${probeLabels[performanceSnapshot.benchmark_status] || performanceSnapshot.benchmark_status}`;
}

async function loadPerformanceStatus({
  renderDialog = false,
  preserveDialogSelection = false,
} = {}) {
  const response = await apiFetch("/api/performance", {method: "GET"});
  const result = await response.json();
  if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
  performanceSnapshot = result;
  renderBackendStatus();
  if (renderDialog) {
    renderPerformanceDialog({preserveSelection: preserveDialogSelection});
  }
}

async function openBackendSettings() {
  const message = document.getElementById("backendSettingsMessage");
  message.textContent = "";
  try {
    await loadPerformanceStatus({renderDialog: true});
    document.getElementById("backendSettingsDialog").showModal();
  } catch (error) {
    document.getElementById("errorMessage").textContent = error.message;
  }
}

function closeBackendSettings() {
  document.getElementById("backendSettingsDialog").close();
}

async function saveBackendSettings(action) {
  const message = document.getElementById("backendSettingsMessage");
  if (action === "apply" && !window.confirm("应用后 XYQQuiz 将立即重启，Windows 可能再次请求管理员权限。现在重启吗？")) return;
  const buttons = document.querySelectorAll(".dialog-actions button");
  for (const button of buttons) button.disabled = true;
  message.textContent = "";
  try {
    const response = await apiFetch("/api/performance/settings", {
      body: {
        action,
        ocr_backend: document.getElementById("ocrBackendSelect").value,
        preview_backend: document.getElementById("previewBackendSelect").value,
      },
    });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
    if (action === "apply") {
      message.textContent = "设置已保存，正在重启…";
      return;
    }
    await loadPerformanceStatus({renderDialog: true});
    message.textContent = "已保存，将在下次启动时生效。";
  } catch (error) {
    message.textContent = error.message;
  } finally {
    for (const button of buttons) button.disabled = false;
  }
}

const frameDecoder = createLatestFrameDecoder(decodeFrame, renderFrame);
const i420FrameDecoder = createLatestFrameDecoder(decodeI420Frame, renderFrame);
const nv12FrameDecoder = createLatestFrameDecoder(decodeNV12Frame, renderFrame);
const bgraFrameDecoder = createLatestFrameDecoder(decodeBGRAFrame, renderFrame);

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

function setText(element, value) {
  if (element.textContent !== value) element.textContent = value;
}

function renderSidebar(state) {
  setText(sidebarElements.phase, state.phase || "—");
  if (state.capture) setText(sidebarElements.capturePhase, state.capture.phase || "—");
  setText(sidebarElements.question, state.question_text || "等待识别");
  setText(sidebarElements.answer, state.official_answer || "—");
  setText(
    sidebarElements.questionScore,
    score(state.question_score, state.question_runner_up_score),
  );
  setText(
    sidebarElements.optionScore,
    score(state.option_score, state.option_runner_up_score),
  );
  const confidenceLevel = normalizeConfidenceLevel(state);
  const confidenceScore = normalizeConfidenceScore(
    state.confidence_score,
    confidenceLevel,
  );
  const confidenceLevelElement = sidebarElements.confidenceLevel;
  if (confidenceLevelElement.dataset.level !== confidenceLevel) {
    confidenceLevelElement.dataset.level = confidenceLevel;
  }
  setText(confidenceLevelElement, {
    NONE: "未定位",
    CANDIDATE: "低可信候选",
    HIGH: "高可信答案",
  }[confidenceLevel]);
  setText(sidebarElements.confidenceScore, `${Math.round(confidenceScore)}/100`);
  setText(sidebarElements.confidenceReason, state.confidence_reason || "—");
  const timings = state.timings;
  setText(sidebarElements.timings, timings
    ? `布局 ${timings.layout_ms.toFixed(1)} · OCR ${timings.ocr_ms.toFixed(1)} · 匹配 ${timings.match_ms.toFixed(1)} · 总计 ${timings.total_ms.toFixed(1)} ms`
    : "—");
}

function overlaysEqual(left, right) {
  if (left === right) return true;
  if (!Array.isArray(left) || !Array.isArray(right)) return false;
  return left.length === right.length
    && left.every((value, index) => value === right[index]);
}

function updateOverlayState(state) {
  const nextOverlay = state.overlay;
  const nextConfidenceLevel = normalizeConfidenceLevel(state);
  const nextConfidenceScore = normalizeConfidenceScore(
    state.confidence_score,
    nextConfidenceLevel,
  );
  const changed = (
    !overlaysEqual(overlay, nextOverlay)
    || overlayConfidenceLevel !== nextConfidenceLevel
    || overlayConfidenceScore !== nextConfidenceScore
  );
  overlay = nextOverlay;
  overlayConfidenceLevel = nextConfidenceLevel;
  overlayConfidenceScore = nextConfidenceScore;
  return changed;
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
  if (!apiToken) throw new Error("本机会话尚未建立，请重新打开 XYQQuiz");
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
      throw new Error("本机会话已失效，请关闭当前页面并重新打开 XYQQuiz");
    }
    throw new Error(result.error || `HTTP ${response.status}`);
  }
  apiToken = result.token;
}

async function initialize() {
  await bootstrapSession();
  await reportNativePreviewLayout();
  for (const button of document.querySelectorAll(".actions button")) button.disabled = false;
  document.getElementById("backendSettingsButton").disabled = false;

  reconnectingSocket("/ws/frames", (frameSocket) => {
    frameSocket.binaryType = "arraybuffer";
    frameSocket.onmessage = ({data}) => {
      try {
        if (typeof data === "string") {
          configurePreviewMessage(JSON.parse(data));
        } else if (previewMode === "h264") {
          decodeH264Frame(data);
        } else if (previewMode === "i420") {
          i420FrameDecoder.enqueue(data);
        } else if (previewMode === "nv12") {
          nv12FrameDecoder.enqueue(data);
        } else if (previewMode === "bgra") {
          bgraFrameDecoder.enqueue(data);
        } else {
          frameDecoder.enqueue(data);
        }
      } catch (error) {
        document.getElementById("errorMessage").textContent = error.message;
        frameSocket.close(1011, "preview decode failed");
      }
    };
  });

  reconnectingSocket("/ws/state", (stateSocket) => {
    stateSocket.onmessage = ({data}) => {
      const state = JSON.parse(data);
      const overlayChanged = updateOverlayState(state);
      renderSidebar(state);
      if (overlayChanged) drawOverlay();
      if (overlayChanged) void reportNativeOverlay();
    };
  });

  const response = await apiFetch("/api/status");
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  renderSidebar(await response.json());
  await loadPerformanceStatus();
  window.setInterval(() => {
    void loadPerformanceStatus({
      renderDialog: document.getElementById("backendSettingsDialog").open,
      preserveDialogSelection: true,
    }).catch(() => {});
  }, 2000);
  await loadLocalQuestions();
}

new ResizeObserver(scheduleNativePreviewLayout).observe(canvasStack);
window.addEventListener("resize", scheduleNativePreviewLayout);
document.addEventListener("visibilitychange", scheduleNativePreviewLayout);

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
document.getElementById("backendSettingsButton").addEventListener("click", openBackendSettings);
document.getElementById("backendSettingsClose").addEventListener("click", closeBackendSettings);
document.getElementById("backendSettingsCancel").addEventListener("click", closeBackendSettings);
document.getElementById("backendSettingsForm").addEventListener("submit", (event) => {
  event.preventDefault();
  void saveBackendSettings("save");
});
document.getElementById("backendSettingsApply").addEventListener("click", () => {
  void saveBackendSettings("apply");
});

initialize()
  .catch((error) => { document.getElementById("errorMessage").textContent = error.message; });
