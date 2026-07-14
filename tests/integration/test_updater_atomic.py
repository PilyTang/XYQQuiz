from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Callable

import httpx
import pytest

from xyq_quiz.knowledge.updater import (
    QuestionBankUpdater,
    UpdateInProgressError,
    UpdateValidationError,
    load_current_generation,
)


BASE_URL = "https://w.163.com/h5/xyq/dtk/"
INDEX = '<script src="static/js/app.fixture.js"></script>'
APP = (
    '(()=>{var t={};t.u=e=>"static/js/"+'
    '({81:"shared",912:"keju-data"}[e]||e)+"."+'
    '{81:"11111111",912:"cafebabe"}[e]+".js";'
    'const routes=[{path:"/keju",component:()=>Promise.all('
    '[t.e(81),t.e(912)]).then(t.bind(t,4567))}];})();'
)


def _chunk(rows: list[dict[str, str]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
    return (
        "(self.webpackChunk=self.webpackChunk||[]).push([[912],{"
        f'4567:(e,t,n)=>{{const u={payload}}},'
        '9999:(e,t,n)=>{const u=[{Id:"signin",Q:"签到题",A:"签到"}]}'
        "}]);"
    )


def _rows(count: int) -> list[dict[str, str]]:
    rows = [
        {
            "Id": str(index + 1),
            "Q": f"第{index + 1}道结构测试题",
            "A": f"答案{index + 1}",
        }
        for index in range(count)
    ]
    rows[0]["A"] = "蝎子,蝎子精"
    return rows


def _transport(rows: list[dict[str, str]]) -> httpx.MockTransport:
    responses = {
        BASE_URL: INDEX,
        f"{BASE_URL}static/js/app.fixture.js": APP,
        f"{BASE_URL}static/js/keju-data.cafebabe.js": _chunk(rows),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        body = responses.get(str(request.url))
        if body is None:
            return httpx.Response(404, request=request)
        return httpx.Response(200, text=body, request=request)

    return httpx.MockTransport(handler)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def _seed_old_generation(data_dir: Path) -> bytes:
    generation_id = "old-generation"
    generation_dir = data_dir / "generations" / generation_id
    generation_dir.mkdir(parents=True)
    question_bytes = _json_bytes(
        [
            {
                "source_id": "old",
                "question": "旧问题",
                "answer": "旧答案",
                "normalized_question": "旧问题",
            }
        ]
    )
    (generation_dir / "keju_questions.json").write_bytes(question_bytes)
    (generation_dir / "metadata.json").write_bytes(
        _json_bytes(
            {
                "generation_id": generation_id,
                "source_url": "fixture://old",
                "raw_record_count": 1,
                "published_record_count": 1,
                "filtered_ids": [],
                "sha256": hashlib.sha256(question_bytes).hexdigest(),
                "marker": "old",
            }
        )
    )
    current_bytes = _json_bytes({"generation_id": generation_id})
    (data_dir / "current.json").write_bytes(current_bytes)
    return current_bytes


def _assert_old_current(data_dir: Path, current_bytes: bytes) -> None:
    assert (data_dir / "current.json").read_bytes() == current_bytes
    current = load_current_generation(data_dir)
    assert current.generation_id == "old-generation"
    assert current.question_bank.count == 1
    assert current.question_bank.records[0].answer == "旧答案"
    assert current.metadata["marker"] == "old"
    assert current.question_path.parent == current.metadata_path.parent


def test_valid_update_publishes_one_audited_generation(tmp_path: Path) -> None:
    _seed_old_generation(tmp_path)
    rows = _rows(2587)
    rows[1601]["Q"] = ""
    rows[1601]["A"] = ""
    updater = QuestionBankUpdater(tmp_path, transport=_transport(rows))

    result = updater.update()

    current = load_current_generation(tmp_path)
    assert result.raw_record_count == 2587
    assert result.published_record_count == 2586
    assert result.record_count == 2586
    assert result.filtered_ids == ("1602",)
    assert current.generation_id == result.generation_id
    assert current.question_bank.count == 2586
    assert current.question_bank.records[0].answer == "蝎子,蝎子精"
    assert current.metadata["raw_record_count"] == 2587
    assert current.metadata["published_record_count"] == 2586
    assert current.metadata["filtered_ids"] == ["1602"]
    assert current.metadata["sha256"] == hashlib.sha256(
        current.question_path.read_bytes()
    ).hexdigest()
    assert not (tmp_path / "keju_questions.json").exists()
    assert not (tmp_path / "metadata.json").exists()
    assert not list((tmp_path / "generations").glob(".tmp-*"))
    assert not list(tmp_path.glob(".current-*.tmp"))


@pytest.mark.parametrize(
    "failure_stage",
    [
        "question_write",
        "question_fsync",
        "metadata_write",
        "metadata_fsync",
        "generation_publish",
        "current_replace",
    ],
)
def test_failure_before_pointer_switch_keeps_old_current_readable(
    tmp_path: Path,
    failure_stage: str,
) -> None:
    current_bytes = _seed_old_generation(tmp_path)

    def inject(stage: str) -> None:
        if stage == failure_stage:
            raise OSError(f"injected {stage}")

    updater = QuestionBankUpdater(
        tmp_path,
        transport=_transport(_rows(2000)),
        fault_injector=inject,
    )

    with pytest.raises(OSError, match=f"injected {failure_stage}"):
        updater.update()

    _assert_old_current(tmp_path, current_bytes)
    assert not list((tmp_path / "generations").glob(".tmp-*"))
    assert not list(tmp_path.glob(".current-*.tmp"))


def test_next_update_removes_only_safe_staging_directories(tmp_path: Path) -> None:
    current_bytes = _seed_old_generation(tmp_path)
    stale_staging = tmp_path / "generations" / ".tmp-stale"
    stale_staging.mkdir()
    (stale_staging / "partial").write_text("partial", encoding="utf-8")
    orphan = tmp_path / "generations" / "orphan-generation"
    orphan.mkdir()

    def fail_question_write(stage: str) -> None:
        if stage == "question_write":
            raise OSError("stop after cleanup")

    updater = QuestionBankUpdater(
        tmp_path,
        transport=_transport(_rows(2000)),
        fault_injector=fail_question_write,
    )

    with pytest.raises(OSError, match="stop after cleanup"):
        updater.update()

    assert not stale_staging.exists()
    assert orphan.is_dir()
    _assert_old_current(tmp_path, current_bytes)


def test_same_updater_rejects_concurrent_update(tmp_path: Path) -> None:
    _seed_old_generation(tmp_path)
    entered_write = threading.Event()
    release_write = threading.Event()
    errors: list[BaseException] = []

    def pause_first_update(stage: str) -> None:
        if stage == "question_write":
            entered_write.set()
            if not release_write.wait(timeout=5):
                raise TimeoutError("test did not release update")

    updater = QuestionBankUpdater(
        tmp_path,
        transport=_transport(_rows(2000)),
        fault_injector=pause_first_update,
    )

    def run_update() -> None:
        try:
            updater.update()
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run_update)
    thread.start()
    assert entered_write.wait(timeout=5)
    try:
        with pytest.raises(UpdateInProgressError, match="already in progress"):
            updater.update()
    finally:
        release_write.set()
        thread.join(timeout=10)

    assert not thread.is_alive()
    assert errors == []


@pytest.mark.parametrize(
    ("rows_factory", "message"),
    [
        (lambda: _rows(3), "at least 2000"),
        (
            lambda: [
                dict(row, Q="重复问题" if index < 101 else row["Q"])
                for index, row in enumerate(_rows(2000))
            ],
            "duplicate rate",
        ),
        (
            lambda: [
                *(_rows(1999)),
                {"Id": "1", "Q": "最后一题", "A": "最后答案"},
            ],
            "unique source IDs",
        ),
    ],
)
def test_invalid_update_keeps_old_current(
    tmp_path: Path,
    rows_factory: Callable[[], list[dict[str, str]]],
    message: str,
) -> None:
    current_bytes = _seed_old_generation(tmp_path)
    updater = QuestionBankUpdater(tmp_path, transport=_transport(rows_factory()))

    with pytest.raises(UpdateValidationError, match=message):
        updater.update()

    _assert_old_current(tmp_path, current_bytes)
