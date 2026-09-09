const frameCanvas = document.getElementById("frameCanvas");
const overlayCanvas = document.getElementById("overlayCanvas");
const canvasStack = document.querySelector(".canvas-stack");
const previewHint = document.getElementById("previewHint");
const backendSettingsDialog = document.getElementById("backendSettingsDialog");
const confirmationDialog = document.getElementById("confirmationDialog");
const diagnosticsButton = document.getElementById("diagnosticsButton");
const frameCtx = frameCanvas.getContext("2d", {alpha: false});
const overlayCtx = overlayCanvas.getContext("2d");
const sidebarElements = {
  activityKind: document.getElementById("activityKind"),
  questionLabel: document.getElementById("questionLabel"),
  questionScoreLabel: document.getElementById("questionScoreLabel"),
  bankStatus: document.getElementById("bankStatus"),
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
let confirmationResolver = null;
let previewPaused = false;
let performanceRecording = {enabled: false, questions: 0};
let latestStateVersion = null;

function renderPerformanceRecording(value) {
  if (!value) return;
  performanceRecording = value;
  document.getElementById("performanceRecordingButton").textContent = value.enabled
    ? "停止性能记录" : value.questions ? "重新开始记录（清空上一轮）" : "开始性能记录";
  document.getElementById("performanceRecordingStatus").textContent =
    `${value.enabled ? "正在记录" : "记录已停止"} · ${value.questions} 题 · ${value.attempts || 0} 次识别。仅记录耗时和状态；退出前请导出。${value.stop_reason?.endsWith("_limit") ? "已达到记录上限。" : ""}`;
}

async function performanceRecordingAction(action) {
  const button = document.getElementById(action === "export" ? "performanceRecordingExportButton" : "performanceRecordingButton");
  button.disabled = true;
  try {
    const response = await apiFetch("/api/performance/recording", {body: {action}});
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || "性能记录操作失败");
    renderPerformanceRecording(result.recording);
    if (result.path) document.getElementById("errorMessage").textContent = `耗时报告已保存：${result.path}`;
  } catch (error) {
    document.getElementById("errorMessage").textContent = error.message;
  } finally { button.disabled = false; }
}

function acknowledgePerformance(trace, stage, started) {
  void apiFetch("/api/performance/recording/ack", {body: {
    ...trace, stage, elapsed_ms: performance.now() - started,
  }}).catch(() => {});
}

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
  if (previewPaused || document.visibilityState !== "visible") {
    bitmap.close();
    return;
  }
  currentFrameId = frameId;
  const bitmapWidth = bitmap.displayWidth || bitmap.width;
  const bitmapHeight = bitmap.displayHeight || bitmap.height;
  const canvasSizeChanged = (
    frameCanvas.width !== bitmapWidth
    || frameCanvas.height !== bitmapHeight
  );
  if (canvasSizeChanged) {
    frameCanvas.width = bitmapWidth;
    frameCanvas.height = bitmapHeight;
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
  // The native renderer is a separate HWND above WebView, so no DOM z-index
  // can cover it. Hide that window while any modal dialog is open.
  const noOpenDialog = document.querySelector("dialog[open]") === null;
  const visible = (
    document.visibilityState === "visible"
    && rect.width > 0
    && rect.height > 0
    && noOpenDialog
    && !previewPaused
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
    drawOverlay();
    void reportNativePreviewLayout();
  }, 50);
}

async function reportNativeOverlay() {
  if (previewMode !== "native" || !apiToken) return;
  const levels = {NONE: 0, CANDIDATE: 1, HIGH: 2};
  try {
    const response = await apiFetch("/api/preview/overlay", {
      body: {
        rect: overlay,
        score: overlayConfidenceScore,
        level: levels[overlayConfidenceLevel] || 0,
      },
    });
    return response.ok;
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
  const fpsLabel = previewPaused ? "已暂停" : Number.isFinite(fps) ? `${fps.toFixed(1)} FPS` : "— FPS";
  element.textContent = [
    shortBackendLabel(performanceSnapshot.ocr, "ocr"),
    shortBackendLabel(performanceSnapshot.preview, "preview"),
    fpsLabel,
    performanceSnapshot.low_resource_mode ? "低配模式" : "",
  ].filter(Boolean).join(" · ");
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
  if (!preserveSelection) {
    document.getElementById("lowResourceMode").checked = Boolean(performanceSnapshot.pending_low_resource_mode);
  }
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
  document.getElementById("resourceProfileReason").textContent = performanceSnapshot.resource_profile_reason
    || (performanceSnapshot.resource_profile_origin === "manual" ? "当前沿用手动保存的高低配选择。" : "首次运行将自动选择，之后保留你的设置。");
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
  renderPerformanceRecording(result.recording);
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
    backendSettingsDialog.showModal();
    await reportNativePreviewLayout();
  } catch (error) {
    document.getElementById("errorMessage").textContent = error.message;
  }
}

function closeBackendSettings() {
  backendSettingsDialog.close();
  void reportNativePreviewLayout();
}

async function saveBackendSettings(action) {
  const message = document.getElementById("backendSettingsMessage");
  if (action === "apply" && !await requestConfirmation({
    eyebrow: "性能设置",
    title: "立即重启并应用？",
    message: "应用后 XYQQuiz 将立即重启，Windows 可能再次请求管理员权限。",
    acceptLabel: "重启并应用",
  })) return;
  const buttons = backendSettingsDialog.querySelectorAll(".dialog-actions button");
  for (const button of buttons) button.disabled = true;
  message.textContent = "";
  try {
    const response = await apiFetch("/api/performance/settings", {
      body: {
        action,
        ocr_backend: document.getElementById("ocrBackendSelect").value,
        preview_backend: document.getElementById("previewBackendSelect").value,
        low_resource_mode: document.getElementById("lowResourceMode").checked,
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
  // Use display pixels, not the downscaled preview bitmap, so CPU and GPU
  // outlines retain the same thickness in low-resource mode and at high DPI.
  const bounds = canvasStack.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  const outputWidth = Math.max(1, Math.round(Math.round(bounds.width) * scale));
  const outputHeight = Math.max(1, Math.round(Math.round(bounds.height) * scale));
  if (overlayCanvas.width !== outputWidth) overlayCanvas.width = outputWidth;
  if (overlayCanvas.height !== outputHeight) overlayCanvas.height = outputHeight;
  overlayCtx.clearRect(0, 0, outputWidth, outputHeight);
  if (!overlay || overlayConfidenceLevel === "NONE") return;
  const inputWidth = frameCanvas.width;
  const inputHeight = frameCanvas.height;
  if (!inputWidth || !inputHeight) return;
  let contentWidth = outputWidth;
  let contentHeight = outputHeight;
  if (outputWidth * inputHeight <= outputHeight * inputWidth) {
    contentHeight = Math.floor(inputHeight * outputWidth / inputWidth);
  } else {
    contentWidth = Math.floor(inputWidth * outputHeight / inputHeight);
  }
  const offsetX = Math.floor((outputWidth - contentWidth) / 2);
  const offsetY = Math.floor((outputHeight - contentHeight) / 2);
  const [x, y, width, height] = overlay;
  const left = offsetX + Math.floor(x * contentWidth);
  const top = offsetY + Math.floor(y * contentHeight);
  const right = offsetX + Math.floor((x + width) * contentWidth);
  const bottom = offsetY + Math.floor((y + height) * contentHeight);
  const thickness = Math.max(3, Math.floor(outputWidth / 400));
  const dashed = overlayConfidenceLevel === "CANDIDATE";
  const dashLength = thickness * 3;
  const dashGap = thickness * 2;
  overlayCtx.fillStyle = "#ef4444";
  function horizontal(y0, y1) {
    if (!dashed) {
      overlayCtx.fillRect(left, y0, right - left, y1 - y0);
      return;
    }
    for (let x0 = left; x0 < right; x0 += dashLength + dashGap) {
      overlayCtx.fillRect(x0, y0, Math.min(right, x0 + dashLength) - x0, y1 - y0);
    }
  }
  function vertical(x0, x1) {
    if (!dashed) {
      overlayCtx.fillRect(x0, top, x1 - x0, bottom - top);
      return;
    }
    for (let y0 = top; y0 < bottom; y0 += dashLength + dashGap) {
      overlayCtx.fillRect(x0, y0, x1 - x0, Math.min(bottom, y0 + dashLength) - y0);
    }
  }
  horizontal(top, Math.min(bottom, top + thickness));
  horizontal(Math.max(top, bottom - thickness), bottom);
  vertical(left, Math.min(right, left + thickness));
  vertical(Math.max(left, right - thickness), right);
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


function score(value, runnerUp) {
  return `${Number(value || 0).toFixed(1)} / 次高 ${Number(runnerUp || 0).toFixed(1)}`;
}

function setText(element, value) {
  if (element.textContent !== value) element.textContent = value;
}

function renderSidebar(state) {
  const teacher = state.activity_kind === "teachers_day";
  setText(sidebarElements.activityKind, {
    keju: "科举", teachers_day: "教师节·看图说话", unknown: "正在判断题型",
  }[state.activity_kind] || "等待答题界面");
  setText(sidebarElements.questionLabel, teacher ? "题目提示" : "OCR 题目");
  setText(sidebarElements.questionScoreLabel, teacher ? "图标匹配分数" : "题目分数");
  if (state.question_banks) {
    const banks = state.question_banks;
    const lines = [["keju", "科举"], ["teachers_day", "教师节"]].map(([key, label]) => {
      const bank = banks[key];
      if (!bank?.available) return `${label}：不可用`;
      const updated = bank.updated_at ? new Date(bank.updated_at).toLocaleDateString("zh-CN") : "—";
      return `${label} ${bank.record_count} 条 · ${updated}${bank.message ? ` · ${bank.message}` : ""}`;
    });
    setText(sidebarElements.bankStatus, lines.join("；"));
  }
  setText(sidebarElements.phase, state.phase || "—");
  if (state.capture) setText(sidebarElements.capturePhase, state.capture.phase || "—");
  setText(sidebarElements.question, state.question_text || "等待识别");
  setText(sidebarElements.answer, state.official_answer || "—");
  setText(
    sidebarElements.questionScore,
    teacher ? score(state.image_score, state.image_runner_up_score) : score(state.question_score, state.question_runner_up_score),
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

async function runAction(button, path, options = {}) {
  const error = document.getElementById("errorMessage");
  button.disabled = true;
  error.dataset.kind = options.pendingMessage ? "pending" : "";
  error.textContent = options.pendingMessage || "";
  try {
    const response = await apiFetch(path);
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
    if (options.successMessage) {
      error.dataset.kind = "success";
      error.textContent = typeof options.successMessage === "function"
        ? options.successMessage(result)
        : options.successMessage;
    }
  } catch (caught) {
    error.dataset.kind = "error";
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
      const received = performance.now();
      const state = JSON.parse(data);
      latestStateVersion = state.version;
      const trace = state.performance_trace;
      if (trace) acknowledgePerformance(trace, "received", received);
      const overlayChanged = updateOverlayState(state);
      renderSidebar(state);
      if (overlayChanged) drawOverlay();
      if (overlayChanged || trace) {
        const nativeUpdate = reportNativeOverlay();
        if (trace && !previewPaused && document.visibilityState === "visible") {
          if (previewMode === "native") {
            void nativeUpdate.then(ok => {
              if (ok) acknowledgePerformance(trace, "native_command", received);
            });
          } else {
            requestAnimationFrame(() => {
              if (latestStateVersion === state.version && !previewPaused && document.visibilityState === "visible")
                acknowledgePerformance(trace, "canvas_submitted", received);
            });
          }
        }
      }
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
document.getElementById("previewPauseButton").addEventListener("click", () => {
  previewPaused = !previewPaused;
  canvasStack.classList.toggle("preview-paused", previewPaused);
  const button = document.getElementById("previewPauseButton");
  button.textContent = previewPaused ? "恢复预览" : "暂停预览（识别继续）";
  button.setAttribute("aria-pressed", String(previewPaused));
  renderBackendStatus();
  void reportNativePreviewLayout();
});

function requestConfirmation({eyebrow, title, message, acceptLabel = "确认"}) {
  if (confirmationResolver !== null) {
    return Promise.reject(new Error("已有确认操作正在进行"));
  }
  document.getElementById("confirmationEyebrow").textContent = eyebrow;
  document.getElementById("confirmationTitle").textContent = title;
  document.getElementById("confirmationMessage").textContent = message;
  document.getElementById("confirmationAccept").textContent = acceptLabel;
  confirmationDialog.returnValue = "";
  return new Promise((resolve, reject) => {
    confirmationResolver = resolve;
    try {
      confirmationDialog.showModal();
      scheduleNativePreviewLayout();
    } catch (error) {
      confirmationResolver = null;
      reject(error);
    }
  });
}

async function saveRecognitionDiagnostics() {
  const accepted = await requestConfirmation({
    eyebrow: "识别诊断",
    title: "保存当前识别现场？",
    message: "识别诊断会保存当前完整游戏画面、OCR 裁剪和日志尾部，可能包含角色名、聊天或其他个人信息。文件只会保存到本机 diagnostics 目录。",
    acceptLabel: "确认保存",
  });
  if (!accepted) return;
  await runAction(diagnosticsButton, "/api/diagnostics", {
    pendingMessage: "正在保存识别诊断，请稍候…",
    successMessage: (result) => `识别诊断已保存：${result.path}`,
  });
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
  if (!await requestConfirmation({
    eyebrow: "本地补题",
    title: "删除这条本地题目？",
    message: `确认删除本地题目“${record.question}”吗？`,
    acceptLabel: "确认删除",
  })) return;
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

document.getElementById("updateButton").addEventListener("click", ({currentTarget}) => runAction(currentTarget, "/api/question-bank/update", {
  pendingMessage: "正在更新科举和教师节题库…", successMessage: (result) => result.message || "题库更新完成",
}));
diagnosticsButton.addEventListener("click", () => { void saveRecognitionDiagnostics(); });
document.getElementById("environmentDiagnosticsButton").addEventListener("click", ({currentTarget}) => runAction(currentTarget, "/api/environment-diagnostics"));
document.getElementById("performanceRecordingButton").addEventListener("click", () => performanceRecordingAction(performanceRecording.enabled ? "stop" : "start"));
document.getElementById("performanceRecordingExportButton").addEventListener("click", () => performanceRecordingAction("export"));
document.getElementById("shutdownButton").addEventListener("click", ({currentTarget}) => runAction(currentTarget, "/api/shutdown"));
document.getElementById("localQuestionMode").addEventListener("change", updateLocalModeFields);
document.getElementById("localQuestionForm").addEventListener("submit", saveLocalQuestion);
document.getElementById("localQuestionCancel").addEventListener("click", resetLocalQuestionForm);
document.getElementById("backendSettingsButton").addEventListener("click", openBackendSettings);
document.getElementById("backendSettingsClose").addEventListener("click", closeBackendSettings);
document.getElementById("backendSettingsCancel").addEventListener("click", closeBackendSettings);
backendSettingsDialog.addEventListener("close", scheduleNativePreviewLayout);
backendSettingsDialog.addEventListener("cancel", scheduleNativePreviewLayout);
confirmationDialog.addEventListener("close", () => {
  scheduleNativePreviewLayout();
  const resolve = confirmationResolver;
  confirmationResolver = null;
  if (resolve !== null) resolve(confirmationDialog.returnValue === "confirm");
});
confirmationDialog.addEventListener("cancel", scheduleNativePreviewLayout);
document.getElementById("backendSettingsForm").addEventListener("submit", (event) => {
  event.preventDefault();
  void saveBackendSettings("save");
});
document.getElementById("backendSettingsApply").addEventListener("click", () => {
  void saveBackendSettings("apply");
});

initialize()
  .catch((error) => { document.getElementById("errorMessage").textContent = error.message; });
