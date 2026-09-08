from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from xyq_quiz.capture.models import Rect
from xyq_quiz.recognition.models import ActivityKind, DetectedLayout


class TeacherLayoutDetector:
    """Find the dialog itself; desktop size does not determine dialog scale."""

    def __init__(self, profile_path: Path) -> None:
        path = Path(profile_path)
        self.profile = json.loads(path.read_text(encoding="utf-8"))
        self.templates = {}
        for name, anchor in self.profile["anchors"].items():
            data = (path.parent / anchor["template_path"]).read_bytes()
            image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError("教师节定位资源损坏")
            self.templates[name] = image
        self.suspected = False
        self._last = None
        self._scaled_templates = {}

    def _template(self, template, sx, sy):
        width, height = max(3, round(template.shape[1]*sx)), max(3, round(template.shape[0]*sy))
        key = (id(template), width, height, sx < 1)
        resized = self._scaled_templates.get(key)
        if resized is None:
            resized = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA if sx < 1 else cv2.INTER_LINEAR)
            if len(self._scaled_templates) >= 128:
                self._scaled_templates.pop(next(iter(self._scaled_templates)))
            self._scaled_templates[key] = resized
        return resized

    def _find(self, gray, template, sx, sy, bounds=None):
        frame_h, frame_w = gray.shape
        if bounds is None:
            left, top, right, bottom = 0, 0, frame_w, frame_h
        else:
            left, top, right, bottom = bounds
            left, top = max(0, int(left)), max(0, int(top))
            right, bottom = min(frame_w, int(right)), min(frame_h, int(bottom))
        width, height = max(3, round(template.shape[1]*sx)), max(3, round(template.shape[0]*sy))
        if right-left < width or bottom-top < height:
            return (-1., 0, 0, width, height)
        resized = self._template(template, sx, sy)
        result = cv2.matchTemplate(gray[top:bottom, left:right], resized, cv2.TM_CCOEFF_NORMED)
        _, score, _, (x, y) = cv2.minMaxLoc(result)
        return (float(score), x+left, y+top, width, height)

    def _title_candidates(self, gray):
        if self._last is not None and self._last[0] == gray.shape:
            _, x, y, sx, sy = self._last
            bounds = (x-45*sx, y-35*sy, x+115*sx, y+55*sy)
            matches = [self._find(gray, self.templates["title"], sx*s, sy*s, bounds) for s in (.97, 1., 1.03)]
            if max(item[0] for item in matches) >= .78:
                return sorted(matches, reverse=True)
            # Clear the previous answer immediately when its dialog disappears.
            # A following frame performs the slower global search for relocation.
            return []
        # Search a small image first, then score only promising neighborhoods
        # at original resolution. Coarse scores can nominate but never accept
        # a dialog; title/exit/prompt still pass the original fine thresholds.
        old_factor = min(1., 1280 / gray.shape[1])
        factor = min(1., 640 / gray.shape[1])
        analysis = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA) if factor < 1 else gray
        matches = []
        for scale in (.5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.35, 1.5, 1.75, 2.):
            native_scale = scale / old_factor
            template = self._template(self.templates["title"], native_scale*factor, native_scale*factor)
            h, w = template.shape
            if h > analysis.shape[0] or w > analysis.shape[1]:
                continue
            response = cv2.matchTemplate(analysis, template, cv2.TM_CCOEFF_NORMED)
            # Retain several peaks so an unrelated small label cannot hide
            # the real title after downsampling.
            for _ in range(3):
                _, score, _, (x, y) = cv2.minMaxLoc(response)
                if not np.isfinite(score) or score < .40:
                    break
                margin = 8 / factor
                left, top = x/factor, y/factor
                bounds = (left-margin, top-margin, left+w/factor+margin, top+h/factor+margin)
                for refinement in (.95, 1., 1.05):
                    fine_scale = native_scale * refinement
                    matches.append(self._find(gray, self.templates["title"], fine_scale, fine_scale, bounds))
                response[max(0,y-h):y+h+1,max(0,x-w):x+w+1] = -1
        return sorted(matches, reverse=True)[:4]

    def detect(self, frame: np.ndarray) -> DetectedLayout | None:
        self.suspected = False
        if frame.ndim != 3 or min(frame.shape[:2]) < 300:
            self._last = None
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        title_ref = self.profile["anchors"]["title"]["rect"]
        exit_ref = self.profile["anchors"]["exit"]["rect"]
        for title in self._title_candidates(gray):
            score, x, y, tw, th = title
            if score < .65:
                continue
            sx, sy = tw/title_ref[2], th/title_ref[3]
            ox, oy = x-title_ref[0]*sx, y-title_ref[1]*sy
            ex, ey = ox+exit_ref[0]*sx, oy+exit_ref[1]*sy
            margin = max(12., 22*max(sx,sy))
            bounds = (ex-margin, ey-margin, ex+exit_ref[2]*sx+margin, ey+exit_ref[3]*sy+margin)
            exit_match = max((self._find(gray,self.templates["exit"],sx*s,sy*s,bounds) for s in (.95,1.,1.05)), key=lambda item:item[0])
            if exit_match[0] < .68:
                # A generic blue header alone is not enough to block Keju or
                # force continuous teacher searches. Keep partial dialogs
                # guarded when their independent question prompt is present.
                px, py, pw, ph = self.profile["anchors"]["prompt"]["rect"]
                prompt_bounds = (ox+(px-10)*sx, oy+(py-8)*sy,
                                 ox+(px+pw+10)*sx, oy+(py+ph+8)*sy)
                if self._find(gray, self.templates["prompt"], sx, sy, prompt_bounds)[0] >= .62:
                    self.suspected = True
                continue
            self.suspected = True
            _, ex, ey, ew, eh = exit_match
            txc, tyc = title_ref[0]+title_ref[2]/2, title_ref[1]+title_ref[3]/2
            exc, eyc = exit_ref[0]+exit_ref[2]/2, exit_ref[1]+exit_ref[3]/2
            fit_x = (ex+ew/2-x-tw/2)/(exc-txc)
            fit_y = (ey+eh/2-y-th/2)/(eyc-tyc)
            if not (.75*sx <= fit_x <= 1.25*sx and .75*sy <= fit_y <= 1.25*sy):
                continue
            sx, sy = fit_x, fit_y
            ox, oy = x+tw/2-txc*sx, y+th/2-tyc*sy
            def mapped(raw, padding=0):
                rx,ry,rw,rh = raw
                left,top=round(ox+(rx-padding)*sx),round(oy+(ry-padding)*sy)
                right,bottom=round(ox+(rx+rw+padding)*sx),round(oy+(ry+rh+padding)*sy)
                if left<0 or top<0 or right>frame.shape[1] or bottom>frame.shape[0]:
                    raise ValueError("incomplete dialog")
                return Rect(left,top,max(1,right-left),max(1,bottom-top))
            try:
                panel=mapped((0,0,*self.profile["reference_size"]))
                prompt=mapped(self.profile["anchors"]["prompt"]["rect"])
                icon=mapped(self.profile["icon_rect"],3)
                options=tuple(mapped(raw) for raw in self.profile["option_rects"])
            except ValueError:
                continue
            bounds=(prompt.x-5*sx,prompt.y-5*sy,prompt.x+prompt.width+5*sx,prompt.y+prompt.height+5*sy)
            prompt_match=self._find(gray,self.templates["prompt"],sx,sy,bounds)
            if prompt_match[0] < .62:
                continue
            self._last=(gray.shape,x,y,sx,sy)
            return DetectedLayout(
                question_rect=prompt, option_rects=options,
                anchor_scores=(score,exit_match[0],prompt_match[0]), profile_name="teachers-day",
                activity_kind=ActivityKind.TEACHERS_DAY, panel_rect=panel, icon_rect=icon,
                option_text_rects=options,
            )
        self._last = None
        return None
