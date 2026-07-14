from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from xyq_quiz.recognition.ocr import OCRRole, OCRUnavailable, RapidOCREngine


def test_rapidocr_recognizes_generated_high_contrast_chinese_text() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    assert font_path.exists(), "Windows Microsoft YaHei font is required by smoke test"
    canvas = Image.new("RGB", (720, 180), "white")
    ImageDraw.Draw(canvas).text(
        (30, 35),
        "梦幻西游",
        font=ImageFont.truetype(str(font_path), 80),
        fill="black",
    )
    bgr = np.asarray(canvas)[:, :, ::-1].copy()

    try:
        result = RapidOCREngine().recognize(bgr)
    except OCRUnavailable as exc:
        pytest.skip(f"RapidOCR optional model unavailable: {exc}")

    assert "梦幻西游" in result.text
    assert 0.0 < result.confidence <= 1.0
    assert result.elapsed_ms > 0


def test_dense_banner_with_small_question_uses_real_complete_detection() -> None:
    font_path = Path(r"C:\Windows\Fonts\msyh.ttc")
    canvas = Image.new("RGB", (800, 300), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((30, 55, 770, 145), fill="black")
    draw.text(
        (250, 72),
        "科举活动",
        font=ImageFont.truetype(str(font_path), 48),
        fill="white",
    )
    draw.text(
        (80, 195),
        "真正小题",
        font=ImageFont.truetype(str(font_path), 30),
        fill="black",
    )
    bgr = np.asarray(canvas)[:, :, ::-1].copy()
    engine = RapidOCREngine()

    try:
        result = engine.recognize_region(
            bgr,
            OCRRole.QUESTION,
            fallback_image=bgr,
        )
    except OCRUnavailable as exc:
        pytest.skip(f"RapidOCR optional model unavailable: {exc}")

    assert "真正小题" in result.text
    diagnostics = engine.diagnostics_snapshot()
    assert diagnostics.rec_only_success_count == 0
    assert diagnostics.fallback_count == 1


def test_rapidocr_adapter_is_lazy_and_orders_lines_by_position() -> None:
    calls = 0

    class FakeRapidOCR:
        def __call__(self, _image: np.ndarray) -> SimpleNamespace:
            return SimpleNamespace(
                boxes=np.array(
                    [
                        [[110, 80], [150, 80], [150, 100], [110, 100]],
                        [[10, 12], [50, 12], [50, 32], [10, 32]],
                        [[110, 10], [150, 10], [150, 30], [110, 30]],
                    ],
                    dtype=np.float32,
                ),
                txts=("右下", "左上", "右上"),
                scores=(0.7, 0.8, 0.9),
                elapse=0.012,
            )

    def factory() -> FakeRapidOCR:
        nonlocal calls
        calls += 1
        return FakeRapidOCR()

    engine = RapidOCREngine(engine_factory=factory)
    assert calls == 0

    result = engine.recognize(np.zeros((120, 180, 3), dtype=np.uint8))

    assert calls == 1
    assert result.text == "左上右上右下"
    assert result.confidence == pytest.approx(0.8)
    assert result.elapsed_ms == pytest.approx(12.0)


def test_rapidocr_adapter_returns_empty_text_for_empty_output() -> None:
    class EmptyRapidOCR:
        def __call__(self, _image: np.ndarray) -> SimpleNamespace:
            return SimpleNamespace(boxes=None, txts=None, scores=None, elapse=0)

    result = RapidOCREngine(engine_factory=EmptyRapidOCR).recognize(
        np.zeros((64, 256, 3), dtype=np.uint8)
    )

    assert result.text == ""
    assert result.confidence == 0.0
    assert result.elapsed_ms == 0.0


def test_rapidocr_initialization_error_explains_install_command() -> None:
    def unavailable() -> None:
        raise RuntimeError("missing model")

    engine = RapidOCREngine(engine_factory=unavailable)

    with pytest.raises(OCRUnavailable, match="pip install rapidocr onnxruntime"):
        engine.recognize(np.zeros((10, 10, 3), dtype=np.uint8))


def test_rapidocr_uses_one_engine_per_worker_without_result_pollution() -> None:
    call_barrier = threading.Barrier(5)
    read_barrier = threading.Barrier(5)
    instances: list[NonReentrantRapidOCR] = []
    instances_lock = threading.Lock()

    class NonReentrantRapidOCR:
        def __init__(self, instance_id: int) -> None:
            self.instance_id = instance_id
            self.concurrent_reentries = 0
            self._active = False
            self._active_lock = threading.Lock()
            self._result_marker = ""

        def __call__(self, image: np.ndarray) -> SimpleNamespace:
            marker = str(int(image[0, 0, 0]))
            with self._active_lock:
                if self._active:
                    self.concurrent_reentries += 1
                self._active = True
            self._result_marker = marker
            call_barrier.wait(timeout=2)
            observed_marker = self._result_marker
            read_barrier.wait(timeout=2)
            with self._active_lock:
                self._active = False
            return SimpleNamespace(
                boxes=None,
                txts=(observed_marker,),
                scores=(1.0,),
                elapse=0.001,
            )

    def factory() -> NonReentrantRapidOCR:
        with instances_lock:
            instance = NonReentrantRapidOCR(len(instances) + 1)
            instances.append(instance)
            return instance

    engine = RapidOCREngine(engine_factory=factory)
    images = [np.full((4, 4, 3), marker, np.uint8) for marker in range(1, 6)]

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(engine.recognize, images))

    assert (
        [result.text for result in results],
        len(instances),
        sum(instance.concurrent_reentries for instance in instances),
    ) == (["1", "2", "3", "4", "5"], 5, 0)


def test_rapidocr_reuses_the_engine_owned_by_the_current_thread() -> None:
    factory_calls = 0

    class IdentifiedRapidOCR:
        def __init__(self, instance_id: int) -> None:
            self.instance_id = instance_id

        def __call__(self, _image: np.ndarray) -> SimpleNamespace:
            return SimpleNamespace(
                boxes=None,
                txts=(str(self.instance_id),),
                scores=(1.0,),
                elapse=0.001,
            )

    def factory() -> IdentifiedRapidOCR:
        nonlocal factory_calls
        factory_calls += 1
        return IdentifiedRapidOCR(factory_calls)

    engine = RapidOCREngine(engine_factory=factory)

    first = engine.recognize(np.zeros((2, 2, 3), np.uint8))
    second = engine.recognize(np.zeros((2, 2, 3), np.uint8))

    assert (first.text, second.text, factory_calls) == ("1", "1", 1)


def test_rapidocr_initialization_failure_is_cached_across_workers() -> None:
    start_barrier = threading.Barrier(5)
    factory_calls = 0
    calls_lock = threading.Lock()

    def unavailable() -> None:
        nonlocal factory_calls
        with calls_lock:
            factory_calls += 1
        raise RuntimeError("missing model")

    engine = RapidOCREngine(engine_factory=unavailable)

    def recognize_after_barrier(_index: int) -> str:
        start_barrier.wait(timeout=2)
        with pytest.raises(OCRUnavailable) as caught:
            engine.recognize(np.zeros((2, 2, 3), np.uint8))
        return str(caught.value)

    with ThreadPoolExecutor(max_workers=5) as executor:
        errors = list(executor.map(recognize_after_barrier, range(5)))

    with pytest.raises(OCRUnavailable):
        engine.recognize(np.zeros((2, 2, 3), np.uint8))

    assert factory_calls == 1
    assert len(set(errors)) == 1
