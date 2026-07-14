from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import ctypes
import json
from pathlib import Path
import sys
import threading
from time import perf_counter, perf_counter_ns
from typing import Any, Sequence

import numpy as np

from xyq_quiz.acceptance.fixtures import (
    FixtureKind,
    ManifestError,
    RecognitionFixture,
    decode_png_or_jpeg,
    load_manifest,
)
from xyq_quiz.capture.hub import LatestFrameHub
from xyq_quiz.capture.models import CapturedFrame, CapturePhase, CaptureStatus, Rect
from xyq_quiz.config import AppConfig
from xyq_quiz.knowledge.matcher import QuestionMatcher
from xyq_quiz.knowledge.updater import load_current_generation
from xyq_quiz.knowledge.models import OptionMatch, QuestionMatch, QuestionRecord
from xyq_quiz.recognition.models import DetectedLayout, OCRText, RecognitionResult
from xyq_quiz.recognition.layout import LayoutProfile, build_layout_detector
from xyq_quiz.recognition.ocr import RapidOCREngine
from xyq_quiz.recognition.pipeline import RecognitionPipeline
from xyq_quiz.runtime.coordinator import RecognitionCoordinator
from xyq_quiz.runtime.state import RuntimePhase, RuntimeStore


FRAMEWORK_ONLY_NOTICE = "这是 synthetic/fake 管线的框架开销，不代表真实 OCR/WGC 性能。"
STATIC_FIXTURE_NOTICE = (
    "这是同一进程预热后的静态真实截图 OCR 基准，不是 WGC/FPS 或连续答题验收。"
)
_STAGES = (
    "capture_to_layout_ms",
    "ocr_ms",
    "match_ms",
    "state_publish_ms",
    "total_ms",
)
_LAYOUT = DetectedLayout(
    question_rect=Rect(0, 0, 10, 10),
    option_rects=(
        Rect(10, 0, 10, 10),
        Rect(20, 0, 10, 10),
        Rect(30, 0, 10, 10),
        Rect(40, 0, 10, 10),
    ),
    anchor_scores=(1.0, 1.0),
)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    mode: str
    iterations: int
    published_frames: int
    delivered_frames: int
    skipped_frames: int
    processed_frames: int
    last_processed_frame_id: int
    queued_frame_count: int
    clear_ms: float
    p50_ms: dict[str, float]
    p95_ms: dict[str, float]
    measurement_source: str
    targets_met: None
    notice: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StaticBenchmarkReport:
    mode: str
    fixture_count: int
    rounds: int
    measured_runs: int
    cold_start_ms: float
    end_to_end_p50_ms: float
    end_to_end_p95_ms: float
    pipeline_component_p50_ms: dict[str, float]
    pipeline_component_p95_ms: dict[str, float]
    rss_delta_mb: float | None
    rec_only_success_count: int
    fallback_count: int
    line_count_distribution: dict[int, int]
    targets_met: bool
    measurement_source: str
    notice: str

    def to_json(self) -> dict[str, object]:
        return asdict(self)


class _CaptureStatus:
    def status(self) -> CaptureStatus:
        return CaptureStatus(CapturePhase.CAPTURING)


class _TracingHub(LatestFrameHub):
    def __init__(self) -> None:
        super().__init__()
        self.published_ids: list[int] = []
        self.delivered_ids: list[int] = []
        self._trace_lock = threading.Lock()

    def publish(self, frame: CapturedFrame) -> None:
        with self._trace_lock:
            self.published_ids.append(frame.frame_id)
        super().publish(frame)

    def wait_after(self, frame_id: int, timeout: float) -> CapturedFrame | None:
        delivered = super().wait_after(frame_id, timeout)
        if delivered is not None:
            with self._trace_lock:
                self.delivered_ids.append(delivered.frame_id)
        return delivered

    def measured_backlog(self) -> int:
        latest = self.snapshot()
        with self._trace_lock:
            last_delivered = self.delivered_ids[-1] if self.delivered_ids else 0
        return int(latest is not None and latest.frame_id > last_delivered)


class _TracingLayout:
    def __init__(self) -> None:
        self.first_detect_ns: dict[int, int] = {}
        self._condition = threading.Condition()

    def detect(self, frame: np.ndarray) -> DetectedLayout | None:
        frame_id = _decode_frame_id(frame)
        with self._condition:
            self.first_detect_ns.setdefault(frame_id, perf_counter_ns())
            self._condition.notify_all()
        return None if int(frame[0, 0, 0]) == 0 else _LAYOUT

    def wait_for(self, frame_id: int, timeout: float = 2.0) -> None:
        with self._condition:
            if not self._condition.wait_for(
                lambda: frame_id in self.first_detect_ns,
                timeout=timeout,
            ):
                raise TimeoutError(f"layout did not receive frame {frame_id}")


class _ObservedLayoutDetector:
    def __init__(self, detector: Any) -> None:
        self._detector = detector
        self._calls = 0
        self._condition = threading.Condition()

    def detect(self, frame: np.ndarray) -> DetectedLayout | None:
        result = self._detector.detect(frame)
        with self._condition:
            self._calls += 1
            self._condition.notify_all()
        return result

    def call_count(self) -> int:
        with self._condition:
            return self._calls

    def wait_after(self, calls: int, timeout: float = 10.0) -> None:
        with self._condition:
            if not self._condition.wait_for(lambda: self._calls > calls, timeout):
                raise TimeoutError("layout did not receive static benchmark frame")


class _FakeOCR:
    _OPTIONS = {241: "甲", 242: "乙", 243: "丙", 244: "丁"}

    def recognize(self, image: np.ndarray) -> OCRText:
        started = perf_counter_ns()
        marker = int(image[0, 0, 0])
        text = self._OPTIONS.get(marker, f"题目{marker}")
        return OCRText(text, 1.0, _ms(perf_counter_ns() - started))


class _FakeMatcher:
    def match_question(self, text: str) -> QuestionMatch:
        marker = text.removeprefix("题目")
        record = QuestionRecord(marker, text, "乙", text)
        return QuestionMatch(100.0, 0.0, record)

    def map_answer(self, answer: str, options: Sequence[str]) -> OptionMatch | None:
        try:
            index = tuple(options).index(answer)
        except ValueError:
            return None
        return OptionMatch(100.0, 0.0, index)

    def is_high_confidence(
        self,
        question: QuestionMatch | None,
        option: OptionMatch | None,
    ) -> bool:
        return question is not None and option is not None


class _TracingPipeline:
    def __init__(self, pipeline: Any) -> None:
        self.pipeline = pipeline
        self.finished_ns: dict[int, int] = {}
        self.results: dict[int, RecognitionResult] = {}
        self._condition = threading.Condition()

    def recognize(self, frame: CapturedFrame, generation_id: int) -> RecognitionResult:
        result = self.pipeline.recognize(frame, generation_id)
        finished = perf_counter_ns()
        with self._condition:
            self.results[frame.frame_id] = result
            self.finished_ns[frame.frame_id] = finished
            self._condition.notify_all()
        return result


@dataclass(frozen=True, slots=True)
class _StaticCaseMeasurement:
    end_to_end_ms: float
    pipeline_layout_ms: float
    pipeline_ocr_ms: float
    pipeline_match_ms: float
    pipeline_total_ms: float


class _TracingStore(RuntimeStore):
    def __init__(self) -> None:
        super().__init__()
        self.completed_ns: dict[int, int] = {}
        self.cleared_ns: dict[str, int] = {}
        self._trace_condition = threading.Condition()

    def complete(self, generation_id: int, result: RecognitionResult) -> bool:
        accepted = super().complete(generation_id, result)
        if accepted:
            with self._trace_condition:
                self.completed_ns[result.frame_id] = perf_counter_ns()
                self._trace_condition.notify_all()
        return accepted

    def clear_question(
        self,
        message: str,
        *,
        phase: RuntimePhase = RuntimePhase.MONITORING,
    ) -> int:
        generation = super().clear_question(message, phase=phase)
        with self._trace_condition:
            self.cleared_ns[message] = perf_counter_ns()
            self._trace_condition.notify_all()
        return generation

    def wait_completed(self, frame_id: int, timeout: float = 2.0) -> int:
        with self._trace_condition:
            if not self._trace_condition.wait_for(
                lambda: frame_id in self.completed_ns,
                timeout=timeout,
            ):
                raise TimeoutError(f"state was not published for frame {frame_id}")
            return self.completed_ns[frame_id]

    def wait_cleared(self, message: str, timeout: float = 2.0) -> int:
        with self._trace_condition:
            if not self._trace_condition.wait_for(
                lambda: message in self.cleared_ns,
                timeout=timeout,
            ):
                raise TimeoutError(f"clear was not published for {message}")
            return self.cleared_ns[message]


def run_synthetic_replay(iterations: int = 200) -> BenchmarkReport:
    if iterations < 1 or iterations > 200:
        raise ValueError("iterations must be between 1 and 200")
    hub = _TracingHub()
    layout = _TracingLayout()
    real_pipeline = RecognitionPipeline(layout, _FakeOCR(), _FakeMatcher())  # type: ignore[arg-type]
    pipeline = _TracingPipeline(real_pipeline)
    store = _TracingStore()
    coordinator = RecognitionCoordinator(_CaptureStatus(), hub, layout, pipeline, store)
    captured_ns: dict[int, int] = {}
    measurements = {stage: [] for stage in _STAGES}
    next_frame_id = 1

    def publish(marker: int) -> int:
        nonlocal next_frame_id
        frame_id = next_frame_id
        next_frame_id += 1
        captured = perf_counter_ns()
        captured_ns[frame_id] = captured
        hub.publish(CapturedFrame.create(frame_id, captured, _synthetic_image(marker, frame_id)))
        return frame_id

    # These two frames are deterministically replaced before the consumer starts.
    publish(250)
    publish(249)
    first_candidate = publish(1)
    coordinator.start()
    try:
        layout.wait_for(first_candidate)
        for marker in range(1, iterations + 1):
            if marker > 1:
                candidate = publish(marker)
                layout.wait_for(candidate)
            recognition_frame = publish(marker)
            completed = store.wait_completed(recognition_frame)
            result = pipeline.results[recognition_frame]
            measurements["capture_to_layout_ms"].append(
                _ms(layout.first_detect_ns[recognition_frame] - captured_ns[recognition_frame])
            )
            measurements["ocr_ms"].append(result.timings.ocr_ms)
            measurements["match_ms"].append(result.timings.match_ms)
            measurements["state_publish_ms"].append(
                _ms(completed - pipeline.finished_ns[recognition_frame])
            )
            measurements["total_ms"].append(
                _ms(completed - captured_ns[recognition_frame])
            )

        clear_frame = publish(0)
        cleared = store.wait_cleared("dialog_missing")
        clear_ms = _ms(cleared - captured_ns[clear_frame])
        published = len(hub.published_ids)
        delivered = len(set(hub.delivered_ids))
        processed_ids = tuple(pipeline.results)
        return BenchmarkReport(
            mode="synthetic",
            iterations=iterations,
            published_frames=published,
            delivered_frames=delivered,
            skipped_frames=published - delivered,
            processed_frames=len(processed_ids),
            last_processed_frame_id=max(processed_ids),
            queued_frame_count=hub.measured_backlog(),
            clear_ms=clear_ms,
            p50_ms={name: _percentile(values, 50) for name, values in measurements.items()},
            p95_ms={name: _percentile(values, 95) for name, values in measurements.items()},
            measurement_source="perf_counter_ns at production-path event boundaries",
            targets_met=None,
            notice=FRAMEWORK_ONLY_NOTICE,
        )
    finally:
        coordinator.stop()
        real_pipeline.close()


def _run_static_round(
    cases: Sequence[RecognitionFixture],
    images: dict[str, np.ndarray],
    layout_detector: _ObservedLayoutDetector,
    pipeline: Any,
    *,
    next_frame_id: int,
) -> tuple[list[_StaticCaseMeasurement], int]:
    """Run one cache-cold coordinator lifetime while keeping OCR model hot."""
    measurements: list[_StaticCaseMeasurement] = []
    for case in cases:
        hub = LatestFrameHub()
        store = _TracingStore()
        traced_pipeline = _TracingPipeline(pipeline)
        coordinator = RecognitionCoordinator(
            _CaptureStatus(),
            hub,
            layout_detector,
            traced_pipeline,
            store,
        )
        coordinator.start()
        try:
            image = images[case.file]
            started_ns = perf_counter_ns()
            initial_version = store.snapshot().version
            calls = layout_detector.call_count()
            candidate_id = next_frame_id
            next_frame_id += 1
            hub.publish(CapturedFrame.create(candidate_id, started_ns, image))
            layout_detector.wait_after(calls)

            calls = layout_detector.call_count()
            recognition_id = next_frame_id
            next_frame_id += 1
            hub.publish(
                CapturedFrame.create(
                    recognition_id,
                    perf_counter_ns(),
                    image,
                )
            )
            layout_detector.wait_after(calls)

            if case.kind is FixtureKind.NEGATIVE:
                snapshot = _wait_for_snapshot(
                    store,
                    lambda current: (
                        current.version > initial_version
                        and (
                            current.message == "dialog_missing"
                            or (
                                current.frame_id == recognition_id
                                and current.phase
                                in {
                                    RuntimePhase.ANSWERED,
                                    RuntimePhase.UNCERTAIN,
                                    RuntimePhase.ERROR,
                                }
                            )
                        )
                    ),
                )
                if snapshot.high_confidence or snapshot.overlay is not None:
                    raise RuntimeError(
                        f"negative fixture produced overlay: {case.file}"
                    )
                if snapshot.message != "dialog_missing":
                    raise RuntimeError(
                        f"negative fixture reached recognition terminal: {case.file}: "
                        f"phase={snapshot.phase}, frame_id={snapshot.frame_id}"
                    )
                continue

            completed_ns = store.wait_completed(recognition_id, timeout=15.0)
            result = traced_pipeline.results[recognition_id]
            if (
                not result.high_confidence
                or result.source_id != case.expected_source_id
                or result.option_index != case.expected_option_index
            ):
                raise RuntimeError(
                    f"static fixture regression: {case.file}: "
                    f"source={result.source_id}, index={result.option_index}, "
                    f"question={result.question_text!r}, options={result.option_texts!r}"
                )
            measurements.append(
                _StaticCaseMeasurement(
                    end_to_end_ms=_ms(completed_ns - started_ns),
                    pipeline_layout_ms=result.timings.layout_ms,
                    pipeline_ocr_ms=result.timings.ocr_ms,
                    pipeline_match_ms=result.timings.match_ms,
                    pipeline_total_ms=result.timings.total_ms,
                )
            )
        finally:
            coordinator.stop()
    return measurements, next_frame_id


def _wait_for_snapshot(
    store: RuntimeStore,
    predicate: Any,
    timeout: float = 10.0,
):
    deadline = perf_counter() + timeout
    snapshot = store.snapshot()
    while not predicate(snapshot):
        remaining = deadline - perf_counter()
        if remaining <= 0:
            raise TimeoutError("static benchmark state was not published")
        update = store.wait_after(snapshot.version, min(remaining, 0.25))
        if update is not None:
            snapshot = update
    return snapshot


def run_static_fixture_benchmark(
    manifest_path: Path,
    layout_paths: tuple[Path, ...],
    rounds: int = 3,
) -> StaticBenchmarkReport:
    """Measure warmed OCR on checked static fixtures; this does not exercise WGC."""
    if rounds < 2:
        raise ValueError("static benchmark rounds must be at least 2")
    manifest = load_manifest(manifest_path, require_assets=True)
    positives = tuple(
        case for case in manifest.cases if case.kind is FixtureKind.POSITIVE
    )
    if len(positives) < 5:
        raise ManifestError("static benchmark requires at least 5 positive fixtures")
    if not manifest.cases or not all(case.human_verified for case in manifest.cases):
        raise ManifestError("static benchmark requires human-verified fixtures")
    if not layout_paths:
        raise ValueError("static benchmark requires at least one layout")

    config = AppConfig()
    current = load_current_generation(config.data_dir)
    matcher = QuestionMatcher(
        current.question_bank,
        config.match.question_score,
        config.match.question_gap,
        config.match.option_score,
    )
    detector = _ObservedLayoutDetector(
        build_layout_detector(
            tuple(LayoutProfile.load(path) for path in layout_paths)
        )
    )
    images = {
        case.file: decode_png_or_jpeg(manifest.path.parent / case.file)
        for case in manifest.cases
    }
    rss_before = _process_rss_mb()
    pipeline_ocr = RapidOCREngine()
    pipeline = RecognitionPipeline(detector, pipeline_ocr, matcher)
    measurements: list[_StaticCaseMeasurement] = []
    try:
        cold_started = perf_counter()
        pipeline.warm_up()
        cold_start_ms = (perf_counter() - cold_started) * 1000.0
        diagnostics_before = pipeline_ocr.diagnostics_snapshot()
        next_frame_id = 1
        for _round in range(rounds):
            round_measurements, next_frame_id = _run_static_round(
                manifest.cases,
                images,
                detector,
                pipeline,
                next_frame_id=next_frame_id,
            )
            if len(round_measurements) != len(positives):
                raise RuntimeError(
                    "static benchmark did not measure every positive fixture"
                )
            measurements.extend(round_measurements)
        rss_after = _process_rss_mb()
        diagnostics_after = pipeline_ocr.diagnostics_snapshot()
    finally:
        pipeline.close()

    end_to_end = [item.end_to_end_ms for item in measurements]
    components = {
        "layout_ms": [item.pipeline_layout_ms for item in measurements],
        "ocr_ms": [item.pipeline_ocr_ms for item in measurements],
        "match_ms": [item.pipeline_match_ms for item in measurements],
        "total_ms": [item.pipeline_total_ms for item in measurements],
    }
    end_to_end_p50 = _percentile(end_to_end, 50)
    end_to_end_p95 = _percentile(end_to_end, 95)
    return StaticBenchmarkReport(
        mode="static_fixture",
        fixture_count=len(positives),
        rounds=rounds,
        measured_runs=len(measurements),
        cold_start_ms=round(cold_start_ms, 4),
        end_to_end_p50_ms=end_to_end_p50,
        end_to_end_p95_ms=end_to_end_p95,
        pipeline_component_p50_ms={
            name: _percentile(values, 50)
            for name, values in components.items()
        },
        pipeline_component_p95_ms={
            name: _percentile(values, 95)
            for name, values in components.items()
        },
        rss_delta_mb=(
            round(max(0.0, rss_after - rss_before), 4)
            if rss_before is not None and rss_after is not None
            else None
        ),
        rec_only_success_count=(
            diagnostics_after.rec_only_success_count
            - diagnostics_before.rec_only_success_count
        ),
        fallback_count=(
            diagnostics_after.fallback_count - diagnostics_before.fallback_count
        ),
        line_count_distribution={
            line_count: count
            - diagnostics_before.line_count_distribution.get(line_count, 0)
            for line_count, count in diagnostics_after.line_count_distribution.items()
            if count
            - diagnostics_before.line_count_distribution.get(line_count, 0)
            > 0
        },
        targets_met=end_to_end_p50 <= 350.0 and end_to_end_p95 < 500.0,
        measurement_source=(
            "LatestFrameHub -> RecognitionCoordinator -> RecognitionPipeline "
            "-> RuntimeStore"
        ),
        notice=STATIC_FIXTURE_NOTICE,
    )


def _process_rss_mb() -> float | None:
    if sys.platform != "win32":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    try:
        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = ctypes.c_void_p
        get_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        get_memory_info.restype = ctypes.c_int
        ok = get_memory_info(
            current_process(),
            ctypes.byref(counters),
            counters.cb,
        )
    except (AttributeError, OSError):
        return None
    return counters.WorkingSetSize / (1024 * 1024) if ok else None


def _synthetic_image(marker: int, frame_id: int) -> np.ndarray:
    image = np.zeros((12, 52, 3), dtype=np.uint8)
    rng = np.random.default_rng(marker)
    image[:10, :10] = rng.integers(0, 256, size=(10, 10, 1), dtype=np.uint8)
    image[0, 0] = marker
    for index, option_marker in enumerate((241, 242, 243, 244), start=1):
        image[:10, index * 10 : (index + 1) * 10] = option_marker
    image[10, 50, 0] = frame_id & 0xFF
    image[10, 51, 0] = (frame_id >> 8) & 0xFF
    return image


def _decode_frame_id(image: np.ndarray) -> int:
    return int(image[10, 50, 0]) | (int(image[10, 51, 0]) << 8)


def _ms(nanoseconds: int) -> float:
    return nanoseconds / 1_000_000


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] * (1 - fraction) + ordered[upper] * fraction, 4)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="科举识别确定性回放/延迟统计")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--layout", type=Path, action="append")
    parser.add_argument("--rounds", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.static:
        if args.live:
            print("--static 与 --live 不能同时使用。", file=sys.stderr)
            return 2
        if args.manifest is None or not args.layout:
            print("静态真实截图基准必须显式传入 --manifest 和 --layout。", file=sys.stderr)
            return 2
        try:
            report = run_static_fixture_benchmark(
                args.manifest,
                tuple(args.layout),
                args.rounds,
            )
        except (ManifestError, OSError, RuntimeError, ValueError) as exc:
            print(f"静态真实截图基准失败：{exc}", file=sys.stderr)
            return 2
        payload = report.to_json()
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        print(report.notice)
        return 0
    if args.live:
        if args.manifest is None:
            print("等待真实截图：--live 必须显式传入 --manifest。未执行真实 WGC/OCR 验收。", file=sys.stderr)
            return 2
        try:
            manifest = load_manifest(args.manifest, require_assets=True)
        except ManifestError as exc:
            print(f"等待真实截图：{exc}", file=sys.stderr)
            return 2
        if not manifest.cases or not all(case.human_verified for case in manifest.cases):
            print("等待真实截图：manifest 为空或仍有未人工核验条目。", file=sys.stderr)
            return 2
        print("真实素材已就绪，但本机 WGC 真机验收属于 Task 10B；本阶段不打印达标结论。", file=sys.stderr)
        return 2
    report = run_synthetic_replay(args.iterations)
    payload = report.to_json()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(report.notice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
