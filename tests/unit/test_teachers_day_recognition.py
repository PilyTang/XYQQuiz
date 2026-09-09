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
from xyq_quiz.runtime.coordinator import _quiz_stability_signature, _same_question_signature, _quiz_cache_identity, _same_quiz_identity, _teacher_cursor_continuity


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


def test_official_icon_rim_matches_native_diagnostic_crop(matcher):
    path = ROOT / "tests/fixtures/teachers_day/feedback-6-icon.png"
    query = cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    options = ("牛刀小试","乙木仙遁","兵解符","吃茶去了")
    match = matcher.match(query,options)
    assert match.option_index == 0 and match.level is ConfidenceLevel.HIGH
    assert match.score > 90
    assert match.score-match.runner_up_score >= 10
    assert matcher.match(query,options[::-1]).option_index == 3


@pytest.mark.parametrize("name,options,index", CASES)
@pytest.mark.parametrize("size", [32, 40, 50])
def test_official_icons_map_to_current_option_order(matcher, name, options, index, size):
    query = query_for(matcher, name, size)
    match = matcher.match(query, options)
    assert match.record.name == name and match.option_index == index
    expected_level = ConfidenceLevel.HIGH if name in options else ConfidenceLevel.CANDIDATE
    assert match.level is expected_level
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


def test_near_duplicate_labels_never_choose_between_two_options(matcher):
    options = ("堪察令", "勘察令", "推气过宫", "无穷妙道")
    result = matcher.match(query_for(matcher, "勘察令"), options)
    assert result.option_index is None and result.level is ConfidenceLevel.NONE


def test_transposed_characters_give_only_a_candidate(matcher):
    result = matcher.match(query_for(matcher, "中医药理"), ("巧匠之术", "中药医理", "炼金术", "打造技巧"))
    assert result.option_index == 1 and result.record.name == "中医药理"
    assert result.level is ConfidenceLevel.CANDIDATE
    assert result.option_score == 75.


@pytest.mark.parametrize("name,observed", [
    ("鹰击", "鷹击"),
    ("鹰击", "鹰去"),
    ("连环击", "连环去"),
    ("连环击", "连击"),
    ("连环击", "连环环击"),
    ("勘察令", "堪察令"),
])
def test_single_ocr_edits_use_general_low_confidence_matching(matcher, name, observed):
    options = (observed, "龙卷雨击", "龙腾", "飘渺式")
    result = matcher.match(query_for(matcher, name), options)
    assert result.record.name == name and result.option_index == 0
    assert result.level is ConfidenceLevel.CANDIDATE
    assert 50. <= result.option_score < 100.
    assert observed in result.reason and name in result.reason
    reordered = matcher.match(query_for(matcher, name), options[::-1])
    assert reordered.option_index == 3 and reordered.level is ConfidenceLevel.CANDIDATE


def test_fuzzy_matching_can_recover_without_any_exactly_known_option(matcher):
    result = matcher.match(query_for(matcher, "鹰击"), ("鷹击", "未收录甲", "未收录乙", "未收录丙"))
    assert result.option_index == 0 and result.level is ConfidenceLevel.CANDIDATE


@pytest.mark.parametrize("options", [
    ("鷹击", "鹰去", "龙腾", "飘渺式"),  # Two equally close unknown labels.
    ("鷹击", "破击", "龙腾", "飘渺式"),  # Known distractors count in the text margin.
    ("鷹击", "龙卷击", "龙腾", "飘渺式"),  # Best label has too little separation.
    ("破击", "龙卷雨击", "龙腾", "飘渺式"),  # Do not reinterpret another known skill.
    ("击", "龙卷雨击", "龙腾", "飘渺式"),  # One character is too little evidence.
    ("飞天", "龙卷雨击", "龙腾", "飘渺式"),  # No close label.
])
def test_fuzzy_matching_rejects_text_ties_conflicts_and_weak_labels(matcher, options):
    result = matcher.match(query_for(matcher, "鹰击"), options)
    assert result.option_index is None and result.level is ConfidenceLevel.NONE


@pytest.mark.parametrize("image_kind", ["noise", "weak", "collision"])
def test_fuzzy_matching_requires_strong_unique_image_evidence(matcher, image_kind):
    from types import MappingProxyType
    source = matcher.bank.by_name["鹰击"][0]
    record = matcher.bank.records[source]
    bank = replace(matcher.bank,
        records=(record, replace(record, source_id="teachers_day:collision", name="其他技能")),
        images=(matcher.bank.images[source], matcher.bank.images[source]),
        by_name=MappingProxyType({"鹰击": (0,), "其他技能": (1,)}))
    local = TeacherIconMatcher(bank)
    query = query_for(matcher, "鹰击")
    if image_kind == "noise":
        query = np.random.default_rng(42).integers(0, 256, query.shape, dtype=np.uint8)
    elif image_kind == "weak":
        # Even a unique score below the image threshold cannot enable fuzzy text.
        values = iter((.71, .20))
        local._score = lambda *_: next(values)
    result = local.match(query, ("鷹击", "龙卷雨击", "龙腾", "飘渺式"))
    assert result.option_index is None and result.level is ConfidenceLevel.NONE


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
        options = (*options[:3], "无关技能")
    result = matcher.match(query, options)
    assert result.level is ConfidenceLevel.NONE and result.option_index is None


@pytest.mark.parametrize("hidden", range(4))
@pytest.mark.parametrize("reverse", [False, True])
def test_partial_options_use_visible_answer_without_guessing_hidden_one(matcher, hidden, reverse):
    options = CASES[1][1][::-1] if reverse else CASES[1][1]
    answer = options.index("健身术")
    visible = tuple("" if i == hidden else text for i,text in enumerate(options))
    query = query_for(matcher,"健身术")
    assert matcher.match(query,visible).option_index is None
    matched = matcher.match(query,visible,allow_partial=True)
    if hidden == answer:
        assert matched.option_index is None
    else:
        assert matched.option_index == answer
        assert matched.level is ConfidenceLevel.HIGH


@pytest.mark.parametrize("options", [
    ("", "", "龙腾", "飘渺式"),
    ("", "鹰击", "鹰击", "飘渺式"),
    ("", "鷹击", "龙腾", "飘渺式"),
])
def test_partial_options_reject_multiple_missing_duplicate_or_fuzzy_labels(matcher,options):
    assert matcher.match(query_for(matcher,"鹰击"),options,allow_partial=True).option_index is None


@pytest.mark.parametrize("kind", ["collision", "weak", "noise"])
def test_partial_options_require_strong_separated_image_evidence(matcher,kind):
    from types import MappingProxyType
    source = matcher.bank.by_name["鹰击"][0]
    record = matcher.bank.records[source]
    bank = replace(matcher.bank,records=(record,replace(record,source_id="collision",name="相似技能")),
                   images=(matcher.bank.images[source],matcher.bank.images[source]),
                   by_name=MappingProxyType({"鹰击":(0,),"相似技能":(1,)}))
    local = TeacherIconMatcher(bank)
    query = query_for(matcher,"鹰击")
    if kind == "weak":
        local._score = lambda *_: .71
    elif kind == "noise":
        query = np.random.default_rng(5).integers(0,256,query.shape,dtype=np.uint8)
    assert local.match(query,("", "鹰击", "龙腾", "飘渺式"),allow_partial=True).option_index is None


@pytest.mark.parametrize("hidden,confidence", [(0,0.),(0,.5),(1,.5)])
def test_teacher_pipeline_partial_ocr_uses_only_current_readable_options(matcher,hidden,confidence):
    from xyq_quiz.capture.models import CapturedFrame, Rect
    from xyq_quiz.recognition.models import DetectedLayout, OCRText
    from xyq_quiz.recognition.teachers_day import recognize_teacher
    query = query_for(matcher,"健身术")
    image = np.full((200,500,3),170,np.uint8)
    h,w = query.shape[:2]
    image[:h,:w] = query
    layout = DetectedLayout(question_rect=Rect(0,0,w,h),icon_rect=Rect(0,0,w,h),
                            option_rects=tuple(Rect(i*100,100,90,50) for i in range(4)),
                            anchor_scores=(1.,),activity_kind=ActivityKind.TEACHERS_DAY)
    ocrs = tuple(OCRText(text=("" if confidence == 0 else "健身术") if i==hidden else text,
                        confidence=confidence if i==hidden else .97,elapsed_ms=0.) for i,text in enumerate(CASES[1][1]))
    recognized = recognize_teacher(CapturedFrame.create(8,0,image),3,layout,matcher,
                                   lambda *_: ocrs,lambda *_: None,0.)
    assert recognized.frame_id == 8 and recognized.generation_id == 3
    if hidden == 1:
        assert recognized.option_index is None and recognized.overlay_rect is None
    else:
        assert recognized.option_index == 1 and recognized.high_confidence
        assert recognized.confidence_score > 70
        assert recognized.overlay_rect == layout.option_rects[1]


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


@pytest.mark.parametrize("size,scale", [((1024,768),.5), ((2560,1440),1.), ((3840,2160),1.5)])
def test_coarse_search_keeps_small_and_large_canvas_support(size, scale):
    frame = dialog_canvas(size, scale)
    layout = TeacherLayoutDetector(PROFILE).detect(frame)
    assert layout is not None
    assert abs(layout.panel_rect.width-629*scale) <= 8


def test_new_dialog_appears_without_waiting_for_search_backoff():
    detector = TeacherLayoutDetector(PROFILE)
    assert detector.detect(np.full((768,1024,3),60,np.uint8)) is None
    assert detector.detect(dialog_canvas()) is not None


def test_move_clears_old_layout_then_reacquires_new_location():
    detector = TeacherLayoutDetector(PROFILE)
    first = dialog_canvas((1920,1080),1.,origin=(100,100))
    second = dialog_canvas((1920,1080),1.,origin=(800,400))
    assert detector.detect(first) is not None
    assert detector.detect(second) is None
    moved = detector.detect(second)
    assert moved is not None and abs(moved.panel_rect.x-800) <= 5
    assert detector.detect(np.full_like(second,60)) is None


def test_cached_detection_uses_small_search_regions(monkeypatch):
    detector = TeacherLayoutDetector(PROFILE)
    frame = dialog_canvas((1920,1080),1.)
    assert detector.detect(frame) is not None
    original = cv2.matchTemplate
    shapes = []
    def tracked(image, template, method):
        shapes.append(image.shape)
        return original(image, template, method)
    monkeypatch.setattr(cv2,"matchTemplate",tracked)
    assert detector.detect(frame) is not None
    assert shapes and max(h*w for h,w in shapes) < 1920*1080/20


def test_title_without_exit_and_prompt_never_accepts_dialog():
    profile=json.loads(PROFILE.read_text(encoding="utf-8"))
    frame=np.full((768,1024,3),60,np.uint8)
    anchor=profile['anchors']['title']
    title=cv2.imdecode(np.frombuffer((PROFILE.parent/anchor['template_path']).read_bytes(),np.uint8),1)
    frame[200:200+title.shape[0],400:400+title.shape[1]]=title
    detector=TeacherLayoutDetector(PROFILE)
    assert detector.detect(frame) is None
    assert not detector.suspected
    expected = object()
    router = ActivityLayoutDetector(SimpleNamespace(detect=lambda _: expected), detector)
    assert router.detect(frame) is expected


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


@pytest.mark.parametrize("option", range(4))
@pytest.mark.parametrize("scale", [.8, 1., 1.5])
def test_teacher_cursor_occlusion_preserves_continuity_but_not_cache(option, scale):
    frame = dialog_canvas((1920,1080), scale)
    layout = TeacherLayoutDetector(PROFILE).detect(frame)
    baseline = _quiz_stability_signature(frame, layout)
    rect = layout.option_rects[option]
    changed = frame.copy()
    x, y = rect.x+round(rect.width*.30), rect.y+round(rect.height*.40)
    cv2.rectangle(changed, (x,y), (x+round(18*scale),y+round(15*scale)), (255,220,20), -1)
    signature = _quiz_stability_signature(changed, layout)
    assert _teacher_cursor_continuity(signature, baseline)
    assert not _same_question_signature(signature, baseline)
    assert not _same_quiz_identity(_quiz_cache_identity(changed,layout), _quiz_cache_identity(frame,layout))
    # No unoccluded baseline means no permission to reuse a hidden answer.
    assert not _teacher_cursor_continuity(baseline, signature)
    icon = layout.icon_rect
    changed[icon.y:icon.y+icon.height,icon.x:icon.x+icon.width] = 160
    assert not _teacher_cursor_continuity(_quiz_stability_signature(changed,layout), baseline)


def test_teacher_occlusion_does_not_hide_reordered_options_or_large_cover():
    frame = dialog_canvas()
    layout = TeacherLayoutDetector(PROFILE).detect(frame)
    baseline = _quiz_stability_signature(frame, layout)
    a, b = layout.option_rects[:2]
    changed = frame.copy()
    cv2.rectangle(changed, (a.x+50,a.y+20), (a.x+68,a.y+35), (255,220,20), -1)
    changed[b.y:b.y+b.height,b.x:b.x+b.width] = cv2.resize(frame[a.y:a.y+a.height,a.x:a.x+a.width],(b.width,b.height))
    assert not _teacher_cursor_continuity(_quiz_stability_signature(changed,layout),baseline)
    changed = frame.copy()
    changed[a.y:a.y+a.height,a.x:a.x+a.width] = (255,220,20)
    assert not _teacher_cursor_continuity(_quiz_stability_signature(changed,layout),baseline)
