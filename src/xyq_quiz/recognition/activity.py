from __future__ import annotations

import time

from xyq_quiz.recognition.models import ActivityDetection, ActivityKind


class ActivityLayoutDetector:
    def __init__(self, keju_detector, teacher_detector) -> None:
        self.keju = keju_detector
        self.teacher = teacher_detector
        self.observation = ActivityDetection(None)
        self._next_search = 0.
        self._shape = None

    def detect(self, frame):
        should_search = (
            frame.shape != self._shape
            or self.observation.activity_kind in {ActivityKind.TEACHERS_DAY, ActivityKind.UNKNOWN}
            or time.monotonic() >= self._next_search
        )
        self._shape = frame.shape
        layout = self.teacher.detect(frame) if self.teacher is not None and should_search else None
        if should_search:
            self._next_search = time.monotonic() + .2
        if layout is not None:
            self.observation = ActivityDetection(ActivityKind.TEACHERS_DAY, layout)
            return layout
        if should_search and self.observation.activity_kind is ActivityKind.TEACHERS_DAY:
            self.observation = ActivityDetection(ActivityKind.UNKNOWN, reason="教师节题面已变化，正在重新定位")
            return None
        if self.teacher is not None and should_search and self.teacher.suspected:
            self.observation = ActivityDetection(ActivityKind.UNKNOWN, reason="正在判断教师节答题界面")
            return None
        layout = self.keju.detect(frame)
        self.observation = ActivityDetection(ActivityKind.KEJU if layout else None, layout)
        return layout
