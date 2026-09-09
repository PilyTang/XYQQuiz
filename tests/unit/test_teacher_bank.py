from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from xyq_quiz.knowledge.teacher_bank import load_teacher_bank, recover_teacher_bank
from xyq_quiz.knowledge.teacher_updater import TeacherBankUpdater, extract_teacher_records
from xyq_quiz.knowledge.updater import UpdaterParseError
from xyq_quiz.runtime.paths import initialize_teacher_assets


ROOT = Path(__file__).parents[2]


def rows():
    return [dict(Name="破釜沉舟", Id=i, TypeName=category, Type=str(i),
                 Pic=f"https://w.163.com/icon-{i}.jpg", PicName=f"{i}.jpg")
            for i, category in enumerate(("大唐官府", "坐骑技能"), 1)]


def icon(row):
    image = np.random.default_rng(row["Id"]).integers(0, 256, (40, 39, 3), dtype=np.uint8)
    return cv2.imencode(".png", image)[1].tobytes()


def test_parser_is_module_scoped_and_independent_of_names_and_field_order():
    payload = json.dumps(rows(), ensure_ascii=False)
    script = f'999:()=>{{var wrong=[{{Id:7,question:"q"}}];}},7345:()=>{{let renamed={payload};}}'
    parsed = extract_teacher_records(script, 7345)
    assert [row["Id"] for row in parsed] == [1, 2]
    assert [row["TypeName"] for row in parsed] == ["大唐官府", "坐骑技能"]
    with pytest.raises(UpdaterParseError):
        extract_teacher_records(script, 999)
    with pytest.raises(UpdaterParseError):
        extract_teacher_records(f'7345:()=>{{let a={payload},b={payload};}}', 7345)


def test_duplicate_ids_deduplicate_but_conflicts_fail():
    records = rows()
    assert len(extract_teacher_records(f'1:()=>{{var a={json.dumps(records+records)};}}', 1)) == 2
    conflict = [*records, {**records[0], "Name": "连环击"}]
    with pytest.raises(ValueError, match="冲突"):
        extract_teacher_records(f'1:()=>{{var a={json.dumps(conflict)};}}', 1)


def test_atomic_update_retains_pointer_and_snapshot_on_broken_image(tmp_path):
    updater = TeacherBankUpdater(tmp_path, minimum_records=2)
    old = updater.publish(rows(), icon)
    pointer = (tmp_path / "current.json").read_bytes()
    with pytest.raises(ValueError):
        updater.publish(rows(), lambda row: b"broken" if row["Id"] == 2 else icon(row))
    assert (tmp_path / "current.json").read_bytes() == pointer
    assert load_teacher_bank(tmp_path).generation_id == old.generation_id
    assert old.count == 2 and len(old.by_name["破釜沉舟"]) == 2
    assert not old.images[0].flags.writeable
    assert not list((tmp_path / "generations").glob(".tmp-*"))
    with pytest.raises(ValueError, match="过少"):
        updater.publish(rows()[:1], icon)


@pytest.mark.parametrize("broken", ["image", "metadata", "pointer", "path"])
def test_corruption_recovers_previous_valid_generation(tmp_path, broken):
    updater = TeacherBankUpdater(tmp_path, minimum_records=2)
    first = updater.publish(rows(), icon)
    second = updater.publish(rows(), icon)
    if broken == "image":
        (second.directory / second.records[0].image_path).write_bytes(b"broken")
    elif broken == "metadata":
        (second.directory / "metadata.json").write_text("[]")
    elif broken == "pointer":
        (tmp_path / "current.json").write_text("[]")
    else:
        (tmp_path / "current.json").write_text(json.dumps({"schema_version": 1, "generation_id": "../../elsewhere"}))
    recovered = recover_teacher_bank(tmp_path)
    assert recovered is not None and recovered.count == 2 and recovered.recovery_reason
    if broken in {"image", "metadata"}:
        assert recovered.generation_id == first.generation_id
    # Existing snapshots keep their own decoded images after disk changes.
    assert second.images[0].shape == (40, 39, 3)


def test_bundled_bank_and_old_custom_data_directory_seed_offline(tmp_path):
    bundled = ROOT / "data"
    bank = load_teacher_bank(bundled / "teachers_day")
    assert bank.count == 354 and len(bank.by_name) == 353
    (tmp_path / "current.json").write_text('"keep-keju-pointer"')
    initialize_teacher_assets(tmp_path, bundled)
    assert load_teacher_bank(tmp_path / "teachers_day").count == 354
    assert (tmp_path / "current.json").read_text() == '"keep-keju-pointer"'
    assert (tmp_path / "layouts/teachers-day.json").is_file()
    pointer = tmp_path / "teachers_day/current.json"
    pointer.write_text("broken-user-state")
    initialize_teacher_assets(tmp_path, bundled)
    assert pointer.read_text() == "broken-user-state"
    assert recover_teacher_bank(tmp_path / "teachers_day", bundled / "teachers_day") is not None


def test_remote_assets_must_stay_on_official_https_origin(tmp_path):
    updater = TeacherBankUpdater(tmp_path, minimum_records=2)
    for url in ("http://w.163.com/a.png", "https://example.com/a.png", "https://w.163.com:8080/a.png"):
        with pytest.raises(ValueError, match="来源"):
            updater.publish([{**row, "Pic": url} for row in rows()], icon)
    assert not (tmp_path / "current.json").exists()


def test_official_updates_preserve_supplements_and_restart(tmp_path):
    updater = TeacherBankUpdater(tmp_path, minimum_records=2)
    updater.publish(rows(), icon)
    shutil.copytree(ROOT / "data/teachers_day/supplements", tmp_path / "supplements")
    before = {p.relative_to(tmp_path): p.read_bytes() for p in (tmp_path / "supplements").rglob("*") if p.is_file()}
    for _ in range(2):
        updated = updater.publish(rows(), icon)
        assert updated.count == 20
        assert updated.metadata["official_record_count"] == 2
        assert updated.metadata["supplement_record_count"] == 18
        assert "吃茶去了" in load_teacher_bank(tmp_path).by_name
        assert all((tmp_path / p).read_bytes() == data for p, data in before.items())
    pointer = (tmp_path / "current.json").read_bytes()
    with pytest.raises(ValueError):
        updater.publish(rows(), lambda row: b"broken")
    assert (tmp_path / "current.json").read_bytes() == pointer
    assert load_teacher_bank(tmp_path).count == 20
    # A damaged official pointer still recovers a merged snapshot.
    (tmp_path / "current.json").write_text("broken")
    assert recover_teacher_bank(tmp_path).count == 20


def test_existing_install_gets_supplements_without_replacing_official_bank(tmp_path):
    target = tmp_path / "teachers_day"
    old = TeacherBankUpdater(target, minimum_records=2).publish(rows(), icon)
    pointer = (target / "current.json").read_bytes()
    initialize_teacher_assets(tmp_path, ROOT / "data")
    assert (target / "current.json").read_bytes() == pointer
    assert load_teacher_bank(target).count == old.count + 18
    initialize_teacher_assets(tmp_path, ROOT / "data")
    assert load_teacher_bank(target).count == old.count + 18


def test_supplement_corruption_does_not_publish_official_update(tmp_path):
    updater = TeacherBankUpdater(tmp_path, minimum_records=2)
    updater.publish(rows(), icon)
    shutil.copytree(ROOT / "data/teachers_day/supplements", tmp_path / "supplements")
    next((tmp_path / "supplements").glob("generations/*/icons/*.png")).write_bytes(b"broken")
    pointer = (tmp_path / "current.json").read_bytes()
    with pytest.raises(ValueError):
        updater.publish(rows(), icon)
    assert (tmp_path / "current.json").read_bytes() == pointer


def test_official_duplicate_of_supplement_is_merged_once(tmp_path):
    extra = load_teacher_bank(ROOT / "data/teachers_day/supplements")
    record = extra.records[0]
    added = dict(Name=record.name, Id=99, TypeName="其他技能", Type="qtjn", Pic="https://w.163.com/new.png", PicName="new.png")
    updater = TeacherBankUpdater(tmp_path, minimum_records=2)
    updater.publish(rows(), icon)
    shutil.copytree(ROOT / "data/teachers_day/supplements", tmp_path / "supplements")
    bank = updater.publish([*rows(), added], lambda row: (extra.directory / record.image_path).read_bytes() if row["Id"] == 99 else icon(row))
    assert bank.count == 20
    assert len(bank.by_name[record.name]) == 1


@pytest.mark.parametrize("side", [32,40,56])
@pytest.mark.parametrize("border", [0,3,5])
def test_all_supplement_icons_match_after_game_size_rescale(side,border):
    from xyq_quiz.knowledge.teacher_matcher import TeacherIconMatcher
    bank = load_teacher_bank(ROOT / "data/teachers_day")
    matcher = TeacherIconMatcher(bank)
    extra = load_teacher_bank(ROOT / "data/teachers_day/supplements")
    supplement_ids = {record.source_id for record in extra.records}
    count = 0
    for record, image in zip(bank.records, bank.images, strict=True):
        if record.source_id not in supplement_ids:
            continue
        count += 1
        body = image[border:-border,border:-border] if border else image
        padding = max(2,round(side*.075))
        query = cv2.copyMakeBorder(cv2.resize(body, (side, side)), padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(160,160,180))
        result = matcher.match(query, ("牛刀小试", record.name, "变化咒", "龙腾"))
        assert result.option_index == 1, (record.name,side,border,result.reason)
        assert result.record.name == record.name
    assert count == 18


def test_future_supplement_source_uses_same_image_normalization():
    from xyq_quiz.knowledge.teacher_matcher import TeacherIconMatcher
    bank = load_teacher_bank(ROOT / "data/teachers_day/supplements")
    renamed = replace(bank, records=tuple(replace(record,
        source_id=f"teachers_day:future:{index}", source_url="https://example.test/icon.png")
        for index,record in enumerate(bank.records)))
    original = TeacherIconMatcher(bank)
    future = TeacherIconMatcher(renamed)
    for index in range(bank.count):
        old, new = original._image_variants[index], future._image_variants[index]
        assert len(old) == len(new) >= 2
        assert all(np.array_equal(a,b) for a,b in zip(old,new,strict=True))
    assert np.array_equal(original._body_features,future._body_features)
