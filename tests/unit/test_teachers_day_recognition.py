from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from xyq_quiz.knowledge.teacher_bank import load_teacher_bank
from xyq_quiz.knowledge.teacher_matcher import TeacherIconMatcher
from xyq_quiz.recognition.activity import ActivityLayoutDetector
from xyq_quiz.recognition.models import ActivityKind, ConfidenceLevel
from xyq_quiz.recognition.teachers_day_layout import TeacherLayoutDetector
from xyq_quiz.runtime.coordinator import _quiz_stability_signature, _same_question_signature, _quiz_cache_identity, _same_quiz_identity


ROOT = Path(__file__).parents[2]
PROFILE = ROOT / "data/layouts/teachers-day.json"
CASES = [
    ("连环击", ("乾天罡气", "魂兮归来", "红袖添香", "连环击"), 3),
    ("健身术", ("养生之道", "健身术", "打造技巧", "巧匠之术"), 1),
    ("百毒不侵", ("血雨", "后发制人", "百毒不侵", "阎罗令"), 2),
    ("古董评估", ("变化之术", "打坐", "丹元济会", "古董评估"), 3),
    ("勘察令", ("推气过宫", "无穷妙道", "娉婷袅娜", "堪察令"), 3),
    ("炼金术", ("巧匠之术", "打造技巧", "中药医理", "炼金术"), 3),
    ("姐妹同心", ("以和为贵", "姐妹同心", "定身符", "五雷咒"), 1),
]


@pytest.fixture(scope="module")
def matcher():
    return TeacherIconMatcher(load_teacher_bank(ROOT / "data/teachers_day"))


def query_for(matcher, name, size=40, record_index=0):
    image = matcher.bank.images[matcher.bank.by_name[name][record_index]]
    scale = size / max(image.shape[:2])
    image = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
    query = np.full((size+6, size+6, 3), 157, np.uint8)
    query[3:3+image.shape[0], 3:3+image.shape[1]] = image
    return query


@pytest.mark.parametrize("name,options,index", CASES)
@pytest.mark.parametrize("size", [32, 40, 50])
def test_official_icons_map_to_current_option_order(matcher, name, options, index, size):
    query = query_for(matcher, name, size)
    match = matcher.match(query, options)
    assert match.record.name == name and match.option_index == index
    assert match.level is ConfidenceLevel.HIGH
    reordered = matcher.match(query, options[::-1])
    assert reordered.option_index == 3-index and reordered.record.name == name


def test_duplicate_name_records_are_kept_and_resolved_by_the_icon(matcher):
    options = ("养生之道", "破釜沉舟", "健身术", "打造技巧")
    ids = set()
    for index in range(2):
        result = matcher.match(query_for(matcher, "破釜沉舟", record_index=index), options)
        assert result.option_index == 1 and result.level is ConfidenceLevel.HIGH
        ids.add(result.record.source_id)
    assert len(ids) == 2


def test_unknown_distractors_are_allowed_only_with_unique_strong_image_evidence(matcher):
    options = ("未收录甲", "姐妹同心", "未收录乙", "未收录丙")
    result = matcher.match(query_for(matcher, "姐妹同心"), options)
    assert result.option_index == 1 and result.level is ConfidenceLevel.HIGH
    missing_answer = matcher.match(query_for(matcher, "连环击"), options)
    assert missing_answer.option_index is None
    all_unknown = matcher.match(query_for(matcher, "姐妹同心"), ("未收录甲", "未收录乙", "未收录丙", "未收录丁"))
    assert all_unknown.option_index is None


def test_aliases_never_choose_between_two_equivalent_options(matcher):
    options = ("堪察令", "勘察令", "推气过宫", "无穷妙道")
    result = matcher.match(query_for(matcher, "勘察令"), options)
    assert result.option_index is None and result.level is ConfidenceLevel.NONE


def test_verified_transposition_alias_can_be_the_answer(matcher):
    result = matcher.match(query_for(matcher, "中医药理"), ("巧匠之术", "中药医理", "炼金术", "打造技巧"))
    assert result.option_index == 1 and result.record.name == "中医药理"


def test_unknown_distractor_requires_separation_from_outside_bank_candidates(matcher):
    from types import MappingProxyType
    source = matcher.bank.by_name["姐妹同心"][0]
    record = matcher.bank.records[source]
    # An indistinguishable second skill must prevent using the one known option.
    bank = replace(matcher.bank,
        records=(record, replace(record, source_id="teachers_day:collision", name="相似技能")),
        images=(matcher.bank.images[source], matcher.bank.images[source]),
        by_name=MappingProxyType({"姐妹同心": (0,), "相似技能": (1,)}))
    ambiguous = TeacherIconMatcher(bank)
    result = ambiguous.match(query_for(matcher, "姐妹同心"), ("未收录甲", "姐妹同心", "未收录乙", "未收录丙"))
    assert result.option_index is None


@pytest.mark.parametrize("kind", ["empty", "noise", "absent", "partial-options", "duplicate-options", "unknown-option"])
def test_missing_or_conflicting_evidence_never_forces_an_answer(matcher, kind):
    options = CASES[0][1]
    query = query_for(matcher, "连环击")
    if kind == "empty":
        query[:] = 156
    elif kind == "noise":
        query = np.random.default_rng(42).integers(0,256,query.shape,dtype=np.uint8)
    elif kind == "absent":
        query = query_for(matcher, "古董评估")
    elif kind == "partial-options":
        options = options[:3]
    elif kind == "duplicate-options":
        options = (*options[:3], options[0])
    else:
        options = (*options[:3], "连环去")
    result = matcher.match(query, options)
    assert result.level is ConfidenceLevel.NONE and result.option_index is None


def dialog_canvas(size=(1024,768), scale=1., origin=None):
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    panel = np.full((375,629,3),170,np.uint8)
    for anchor in profile["anchors"].values():
        x,y,w,h = anchor["rect"]
        image = cv2.imdecode(np.frombuffer((PROFILE.parent/anchor["template_path"]).read_bytes(),np.uint8),1)
        panel[y:y+h,x:x+w] = image
    for i,(x,y,w,h) in enumerate(profile["option_rects"]):
        cv2.putText(panel, f"Skill {i}", (x+40,y+30), cv2.FONT_HERSHEY_SIMPLEX,.5,(15,15,15),1)
    panel[66:106,222:262] = np.random.default_rng(7).integers(0,256,(40,40,3),dtype=np.uint8)
    panel = cv2.resize(panel,None,fx=scale,fy=scale)
    frame = np.full((size[1],size[0],3),60,np.uint8)
    x,y = origin or ((size[0]-panel.shape[1])//2,(size[1]-panel.shape[0])//2)
    frame[y:y+panel.shape[0],x:x+panel.shape[1]] = panel
    return frame


@pytest.mark.parametrize("size,scale", [((1024,768),1.),((1280,960),1.),((1920,1080),1.),((1920,1080),1.5),((1280,960),.8)])
def test_dialog_localizes_independently_of_canvas_resolution(size, scale):
    detector = TeacherLayoutDetector(PROFILE)
    frame = dialog_canvas(size,scale)
    layout = detector.detect(frame)
    assert layout is not None and layout.activity_kind is ActivityKind.TEACHERS_DAY
    assert abs(layout.panel_rect.width-629*scale) <= 6
    assert abs(layout.icon_rect.width-46*scale) <= 3
    again = detector.detect(frame)
    assert again is not None
    assert abs(again.option_rects[3].x-layout.option_rects[3].x) <= 3


def test_partial_dialog_and_no_dialog_are_not_accepted():
    detector = TeacherLayoutDetector(PROFILE)
    frame = dialog_canvas()
    frame[540:620] = 60  # covers the required exit anchor
    assert detector.detect(frame) is None
    assert detector.suspected
    assert detector.detect(np.full_like(frame,60)) is None


def test_suspected_teacher_dialog_never_routes_to_keju():
    called = []
    keju = SimpleNamespace(detect=lambda frame: called.append(True))
    teacher = SimpleNamespace(detect=lambda frame: None, suspected=True)
    router = ActivityLayoutDetector(keju,teacher)
    assert router.detect(np.zeros((768,1024,3),np.uint8)) is None
    assert router.observation.activity_kind is ActivityKind.UNKNOWN
    assert not called


def test_teacher_identity_tracks_icon_and_option_order_ignoring_counter_and_background():
    frame = dialog_canvas()
    layout = TeacherLayoutDetector(PROFILE).detect(frame)
    signature = _quiz_stability_signature(frame,layout)
    identity = _quiz_cache_identity(frame,layout)
    changed = frame.copy()
    changed[:30] = 200
    r = layout.panel_rect
    changed[r.y+30:r.y+80,r.x+35:r.x+140] = 90
    assert _same_question_signature(signature,_quiz_stability_signature(changed,layout))
    assert _same_quiz_identity(identity,_quiz_cache_identity(changed,layout))
    a,b = layout.option_rects[:2]
    swapped = frame.copy()
    swapped[b.y:b.y+b.height,b.x:b.x+b.width] = cv2.resize(frame[a.y:a.y+a.height,a.x:a.x+a.width],(b.width,b.height))
    assert not _same_question_signature(signature,_quiz_stability_signature(swapped,layout))
    assert not _same_quiz_identity(identity,_quiz_cache_identity(swapped,layout))
    icon = layout.icon_rect
    changed = frame.copy()
    changed[icon.y:icon.y+icon.height,icon.x:icon.x+icon.width] = 160
    assert not _same_question_signature(signature,_quiz_stability_signature(changed,layout))
