from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from xyq_quiz.knowledge.updater import (
    UpdateValidationError,
    UpdaterParseError,
    extract_keju_records,
    resolve_keju_chunk,
)


@dataclass(frozen=True)
class UpdaterFixtures:
    index: str
    app: str
    exp: str


@pytest.fixture
def fixtures() -> UpdaterFixtures:
    root = Path(__file__).parents[1] / "fixtures" / "updater"
    return UpdaterFixtures(
        index=(root / "index.html").read_text(encoding="utf-8"),
        app=(root / "app.js").read_text(encoding="utf-8"),
        exp=(root / "exp.js").read_text(encoding="utf-8"),
    )


def test_resolves_current_keju_chunk_without_hardcoded_hash(
    fixtures: UpdaterFixtures,
) -> None:
    url, module_id = resolve_keju_chunk(
        fixtures.index,
        fixtures.app,
        "https://w.163.com/h5/xyq/dtk/",
    )

    assert url == "https://w.163.com/h5/xyq/dtk/static/js/exp.b65575c0.js"
    assert module_id == 8733


def test_resolution_follows_changed_chunk_module_name_and_hash(
    fixtures: UpdaterFixtures,
) -> None:
    changed_app = (
        fixtures.app.replace("306", "912")
        .replace("8733", "4567")
        .replace('"exp"', '"keju-data"')
        .replace("b65575c0", "cafebabe")
    )

    url, module_id = resolve_keju_chunk(
        fixtures.index,
        changed_app,
        "https://w.163.com/h5/xyq/dtk/",
    )

    assert url.endswith("static/js/keju-data.cafebabe.js")
    assert module_id == 4567


def test_keju_route_without_loader_does_not_use_following_route_loader(
    fixtures: UpdaterFixtures,
) -> None:
    app = (
        '(()=>{var t={};t.u=e=>"static/js/"+'
        '({17:"chunk-common",306:"exp"}[e]||e)+"."+'
        '{17:"11111111",306:"b65575c0"}[e]+".js";'
        'const routes=[{path:"/keju"},{path:"/qiandao",'
        'component:()=>Promise.all([t.e(17),t.e(306)]).then(t.bind(t,8733))}];'
        "})();"
    )

    with pytest.raises(UpdaterParseError, match="/keju.*loader"):
        resolve_keju_chunk(
            fixtures.index,
            app,
            "https://w.163.com/h5/xyq/dtk/",
        )


def test_extracts_only_exact_records_from_keju_module(
    fixtures: UpdaterFixtures,
) -> None:
    rows = extract_keju_records(fixtures.exp, 8733)

    assert [row.source_id for row in rows] == ["1", "2", "3"]
    assert all("签到" not in row.question for row in rows)
    assert rows[1].answer == "蝎子,蝎子精"
    assert rows[0].normalized_question


def test_extracts_changed_module_id_without_using_exp_name(
    fixtures: UpdaterFixtures,
) -> None:
    changed_exp = fixtures.exp.replace("8733", "4567", 1)

    rows = extract_keju_records(changed_exp, 4567)

    assert [row.source_id for row in rows] == ["1", "2", "3"]


@pytest.mark.parametrize(
    ("replacement", "missing_field"),
    [
        ('{Id:"4",Q:"",A:"只有答案"}', "question"),
        ('{Id:"4",Q:"只有问题",A:""}', "answer"),
    ],
)
def test_rejects_record_with_only_one_blank_value(
    fixtures: UpdaterFixtures,
    replacement: str,
    missing_field: str,
) -> None:
    chunk = fixtures.exp.replace('{Id:"4",Q:"",A:""}', replacement)

    with pytest.raises(UpdateValidationError, match=f"empty {missing_field}"):
        extract_keju_records(chunk, 8733)
