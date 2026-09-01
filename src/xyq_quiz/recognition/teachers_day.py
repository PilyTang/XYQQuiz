from __future__ import annotations

import time
import cv2

from xyq_quiz.recognition.models import (
    ActivityKind, ConfidenceLevel, RecognitionResult, RecognitionTimings,
)


def recognize_teacher(frame, generation_id, layout, matcher, read_options, store_crops, layout_ms):
    started = time.perf_counter()
    def crop(rect):
        return frame.bgr[rect.y:rect.y+rect.height, rect.x:rect.x+rect.width]
    def result(**values):
        defaults = dict(
            generation_id=generation_id, frame_id=frame.frame_id,
            question_text="请指出图中所对应的技能或法术名称", option_texts=(),
            official_answer=None, question_score=0., question_runner_up_score=0.,
            option_score=0., option_runner_up_score=0., high_confidence=False,
            option_index=None, overlay_rect=None, activity_kind=ActivityKind.TEACHERS_DAY,
            timings=RecognitionTimings(layout_ms,0.,0.,layout_ms+(time.perf_counter()-started)*1000),
            confidence_score=0., bank_generation=matcher.bank.generation_id if matcher else None,
        )
        defaults.update(values)
        return RecognitionResult(**defaults)
    if layout.icon_rect is None or len(layout.option_rects) != 4:
        return result(confidence_reason="教师节题面尚未完整显示")
    icon = crop(layout.icon_rect)
    raw = tuple(crop(rect) for rect in (layout.option_text_rects or layout.option_rects))
    if icon.size == 0 or any(image.size == 0 for image in raw):
        return result(confidence_reason="教师节题面被截断")
    fallback = tuple(cv2.resize(image,None,fx=3.,fy=3.,interpolation=cv2.INTER_CUBIC) for image in raw)
    store_crops((icon,*fallback))
    if matcher is None:
        return result(confidence_reason="教师节题库不可用，请更新题库或检查安装资源")
    ocr_started=time.perf_counter()
    ocrs=read_options(raw,fallback)
    ocr_ms=(time.perf_counter()-ocr_started)*1000
    texts=tuple(item.text for item in ocrs)
    if any(not item.text.strip() or item.confidence < .8 for item in ocrs):
        return result(option_texts=texts,confidence_reason="选项文字尚未读清，正在重试")
    match_started=time.perf_counter()
    matched=matcher.match(icon,texts)
    match_ms=(time.perf_counter()-match_started)*1000
    record=matched.record
    return result(
        option_texts=texts,official_answer=record.name if record else None,
        source_id=record.source_id if record else None,
        option_score=matched.option_score,
        option_runner_up_score=matched.option_runner_up_score,
        high_confidence=matched.level is ConfidenceLevel.HIGH,
        confidence_level=matched.level,
        confidence_score=(
            matched.score * matched.option_score / 100 * min(item.confidence for item in ocrs)
            if record else 0.
        ),
        confidence_reason=matched.reason,
        image_score=matched.score,image_runner_up_score=matched.runner_up_score,
        image_candidates=matched.candidates,option_index=matched.option_index,
        overlay_rect=layout.option_rects[matched.option_index] if matched.option_index is not None else None,
        timings=RecognitionTimings(layout_ms,ocr_ms,match_ms,layout_ms+(time.perf_counter()-started)*1000),
    )
