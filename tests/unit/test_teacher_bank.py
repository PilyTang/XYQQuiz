from __future__ import annotations

import json
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
    assert bank.count == 336 and len(bank.by_name) == 335
    (tmp_path / "current.json").write_text('"keep-keju-pointer"')
    initialize_teacher_assets(tmp_path, bundled)
    assert load_teacher_bank(tmp_path / "teachers_day").count == 336
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
