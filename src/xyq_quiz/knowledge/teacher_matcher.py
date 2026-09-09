from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np
from rapidfuzz.distance import OSA

from xyq_quiz.knowledge.models import normalize_text
from xyq_quiz.knowledge.teacher_bank import TeacherBankSnapshot, TeacherSkillRecord
from xyq_quiz.recognition.models import ConfidenceLevel


MATCHER_VERSION = "teachers-day-partial-options-border-2"


def _near_name(left: str, right: str) -> bool:
    """Allow one OCR edit, including an adjacent transposition, in real names."""
    return min(len(left), len(right)) >= 2 and OSA.distance(left, right) == 1


def _unit(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32).ravel()
    values = values - values.mean()
    return values / max(float(np.linalg.norm(values)), 1e-6)


def _feature(image: np.ndarray) -> np.ndarray:
    body = image[2:-2, 2:-2] if min(image.shape[:2]) > 12 else image
    small = cv2.resize(body, (24, 24), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edge = cv2.Laplacian(cv2.GaussianBlur(gray, (3, 3), 0), cv2.CV_32F)
    return np.concatenate((.85 * _unit(small), .15 * _unit(edge)))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(_unit(left), _unit(right)))


@dataclass(frozen=True, slots=True)
class TeacherMatch:
    record: TeacherSkillRecord | None
    option_index: int | None
    score: float
    runner_up_score: float
    level: ConfidenceLevel
    reason: str
    candidates: tuple[tuple[str, float], ...] = ()
    option_score: float = 0.
    option_runner_up_score: float = 0.


class TeacherIconMatcher:
    """Use image evidence AND the actual options; never force a four-way choice."""

    def __init__(self, bank: TeacherBankSnapshot) -> None:
        self.bank = bank
        self._features = np.stack([_feature(image) for image in bank.images])
        self._image_variants = []
        self._body_indexes = []
        body_features = []
        for index, image in enumerate(bank.images):
            variants = [image]
            if min(image.shape[:2]) > 12:
                # Apply the same thin/decorative-rim variants to every source,
                # including future supplements. No skill-name or website
                # allowlist; retain originals for already aligned assets.
                insets = sorted({max(1, round(min(image.shape[:2]) * fraction))
                                 for fraction in (.05, .08)})
                for inset in insets:
                    body = image[inset:-inset, inset:-inset]
                    variants.append(body)
                    self._body_indexes.append(index)
                    body_features.append(_feature(body))
            self._image_variants.append(tuple(variants))
        self._body_features = np.stack(body_features) if body_features else None

    @lru_cache(maxsize=4)
    def _templates(self, nominal: int):
        prepared = []
        for images in self._image_variants:
            variants = []
            for image in images:
                for side in range(max(8, nominal - 3), nominal + 4):
                    scale = side / max(image.shape[:2])
                    size = (max(8, round(image.shape[1] * scale)), max(8, round(image.shape[0] * scale)))
                    template = cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
                    variants.append(template)
            prepared.append(tuple(variants))
        return tuple(prepared)

    def _score(self, query: np.ndarray, variants) -> float:
        best = -1.0
        for template in variants:
            height, width = template.shape[:2]
            if height > query.shape[0] or width > query.shape[1]:
                continue
            response = cv2.matchTemplate(query, template, cv2.TM_CCOEFF_NORMED)
            _, color, _, (x, y) = cv2.minMaxLoc(response)
            if not np.isfinite(color):
                continue
            candidate = query[y:y+height, x:x+width]
            inset = max(1, round(min(height, width) * .07))
            a = cv2.cvtColor(candidate[inset:-inset, inset:-inset], cv2.COLOR_BGR2GRAY)
            b = cv2.cvtColor(template[inset:-inset, inset:-inset], cv2.COLOR_BGR2GRAY)
            body = _correlation(a, b)
            a_edge = cv2.Laplacian(cv2.GaussianBlur(a, (3, 3), 0), cv2.CV_32F)
            b_edge = cv2.Laplacian(cv2.GaussianBlur(b, (3, 3), 0), cv2.CV_32F)
            edge = _correlation(a_edge, b_edge)
            best = max(best, .55 * color + .30 * body + .15 * edge)
        return max(0.0, best)

    def _fuzzy_option(self, scores, ranked, names, options, candidates):
        """A strong image may support an approximate label, always as CANDIDATE."""
        index = ranked[0]
        record = self.bank.records[index]
        name = normalize_text(record.name)
        if name in names:
            return None
        image_score = scores[index]
        image_runner = max(
            (scores[i] for i in ranked if normalize_text(self.bank.records[i].name) != name),
            default=0.,
        )
        if image_score < .72 or image_score - image_runner < .10:
            return None
        similarities = [OSA.normalized_similarity(name, option) for option in names]
        order = sorted(range(len(names)), key=similarities.__getitem__, reverse=True)
        option, second = order[:2]
        text_score, text_runner = similarities[option], similarities[second]
        # Do not reinterpret an exact label for a different known skill. Compare
        # against ALL four options, so a tie cannot be hidden by eligibility.
        if (
            names[option] in self.bank.by_name
            or not _near_name(name, names[option])
            or text_score < .5
            or text_score - text_runner < .20
        ):
            return None
        return TeacherMatch(
            record, option, round(image_score * 100, 2), round(image_runner * 100, 2),
            ConfidenceLevel.CANDIDATE,
            f"图标匹配{record.name}；选项{options[option]}文字最接近"
            f"（{text_score * 100:.0f}分，次近{text_runner * 100:.0f}分），仅作低可信候选",
            candidates, round(text_score * 100, 2), round(text_runner * 100, 2),
        )

    def match(self, icon: np.ndarray, options: tuple[str, ...], *, allow_partial: bool = False) -> TeacherMatch:
        def reject(reason, score=0., runner=0., candidates=()):
            return TeacherMatch(None, None, round(score * 100, 2), round(runner * 100, 2), ConfidenceLevel.NONE, reason, candidates)
        names = tuple(normalize_text(text) for text in options)
        partial = allow_partial and len(names) == 4 and names.count("") == 1
        if len(names) != 4 or len(set(names)) != 4 or (not all(names) and not partial):
            return reject("四个选项尚未完整识别，正在重试")
        option_records = tuple(self.bank.by_name.get(name, ()) for name in names)
        mapped_options = [i for i, indexes in enumerate(option_records) if indexes]
        has_unknown_options = len(mapped_options) < 4
        if icon.size == 0 or min(icon.shape[:2]) < 16 or float(icon.std()) < 8:
            return reject("技能图标为空或不完整")
        # Layout crops include a three-reference-pixel guard around the icon.
        nominal = max(8, round(min(icon.shape[:2]) / 1.15))
        templates = self._templates(nominal)
        inner = max(1, round(nominal * .075))
        query_feature = _feature(icon[inner:-inner, inner:-inner])
        coarse = self._features @ query_feature
        if self._body_features is not None:
            # Several renditions can share one record. Scatter-max retains
            # the best score instead of overwriting a previous rendition.
            np.maximum.at(coarse, self._body_indexes, self._body_features @ query_feature)
        selected = set(np.argsort(coarse)[-min(12, len(coarse)):].tolist())
        for indexes in option_records:
            selected.update(indexes)
        # Misspelled options must not depend solely on coarse image retrieval.
        for option in names:
            if option not in self.bank.by_name:
                for name, indexes in self.bank.by_name.items():
                    if _near_name(name, option):
                        selected.update(indexes)
        scores = {index: self._score(icon, templates[index]) for index in selected}
        ranked = sorted(scores, key=scores.get, reverse=True)
        candidates = tuple((self.bank.records[index].name, round(scores[index] * 100, 2)) for index in ranked[:5])
        # With a hidden option, require an exact visible label. Approximate
        # text could otherwise point to a distractor while the answer is hidden.
        fuzzy = None if partial else self._fuzzy_option(scores, ranked, names, options, candidates)
        if fuzzy is not None:
            return fuzzy
        if not mapped_options:
            return reject("当前选项未能唯一对应题库技能", candidates=candidates)
        option_best = {
            option: max(option_records[option], key=lambda index: scores[index])
            for option in mapped_options
        }
        order = sorted(mapped_options, key=lambda option: scores[option_best[option]], reverse=True)
        option = order[0]
        index = option_best[option]
        score = scores[index]
        runner = scores[option_best[order[1]]] if len(order) > 1 else 0.
        if any(
            not option_records[i] and _near_name(names[option], name)
            for i, name in enumerate(names) if i != option
        ):
            return reject("多个选项文字接近同一技能，无法唯一定位", score, runner, candidates)
        outside = [i for i in ranked if normalize_text(self.bank.records[i].name) not in names]
        if outside and scores[outside[0]] >= .72 and scores[outside[0]] - score > .10:
            return reject("图标与当前选项冲突，正在重新识别", score, runner, candidates)
        if has_unknown_options:
            # An unknown distractor does not make a clearly matched answer invalid.
            # Require HIGH image evidence and a margin against every other skill
            # in the full-bank shortlist, not only the known options.
            other_scores = [
                scores[i] for i in ranked
                if normalize_text(self.bank.records[i].name) != names[option]
            ]
            runner = max(other_scores, default=0.)
            if score < .72 or score - runner < .10:
                return reject("存在未收录选项，图标尚不足以唯一确认答案", score, runner, candidates)
        if score < .58 or score - runner < .075:
            return reject("图标证据不足或选项之间仍有歧义", score, runner, candidates)
        level = ConfidenceLevel.HIGH if score >= .72 and score - runner >= .10 else ConfidenceLevel.CANDIDATE
        reason = (
            "一个选项未读清，当前图标与可读选项唯一对应"
            if partial else
            "图标与选项唯一对应" if level is ConfidenceLevel.HIGH else "候选唯一，图像清晰度不足以达到高可信"
        )
        return TeacherMatch(
            self.bank.records[index], option, round(score * 100, 2), round(runner * 100, 2), level,
            reason,
            candidates, 100.,
        )
