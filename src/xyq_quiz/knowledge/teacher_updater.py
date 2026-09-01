from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import httpx
import pyjson5

from xyq_quiz.knowledge.teacher_bank import (
    TeacherSkillRecord, decode_icon, load_teacher_bank, load_teacher_generation,
)
from xyq_quiz.knowledge.updater import (
    DEFAULT_SOURCE_URL, UpdateInProgressError, UpdaterParseError,
    _app_script_source, _balanced_end, resolve_keju_chunk,
)


def extract_teacher_records(script: str, module_id: int) -> list[dict]:
    match = re.search(rf"(?<![$\w])['\"]?{module_id}['\"]?\s*:", script)
    if match is None:
        raise UpdaterParseError("未找到教师节路由模块")
    start = script.find("{", match.end())
    module = script[start:_balanced_end(script, start, "{", "}")]
    arrays = []
    # Match array assignments structurally, independent of minified variable names.
    for candidate in re.finditer(r"(?:=|:)\s*\[\s*\{", module):
        begin = module.index("[", candidate.start())
        try:
            rows = pyjson5.decode(module[begin:_balanced_end(module, begin, "[", "]")])
        except (ValueError, TypeError, pyjson5.Json5DecoderException):
            continue
        if rows and all(isinstance(row, dict) and {"Id", "Name", "TypeName", "Type", "Pic", "PicName"} <= row.keys() for row in rows):
            arrays.append(rows)
    if len(arrays) != 1:
        raise UpdaterParseError("教师节图标数组缺失或不唯一")
    by_id = {}
    for row in arrays[0]:
        if not isinstance(row["Id"], (str, int)) or isinstance(row["Id"], bool):
            raise ValueError("教师节技能编号无效")
        if any(not isinstance(row[field], str) or not row[field].strip() for field in ("Name", "TypeName", "Type", "Pic")):
            raise ValueError("教师节技能数据不完整")
        key = str(row["Id"])
        if key in by_id and row != by_id[key]:
            raise ValueError("教师节技能编号冲突")
        by_id[key] = row
    return list(by_id.values())


def _write_json(path: Path, payload: object) -> None:
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with path.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


class TeacherBankUpdater:
    def __init__(self, data_dir: Path, *, source_url: str = DEFAULT_SOURCE_URL,
                 client_factory=None, minimum_records: int = 300) -> None:
        self.data_dir = Path(data_dir)
        self.source_url = source_url
        self.minimum_records = minimum_records
        self._client_factory = client_factory or (lambda: httpx.Client(timeout=20, follow_redirects=True))
        self._lock = threading.Lock()

    def _checked_url(self, url: str) -> str:
        url = urljoin(self.source_url, url)
        parsed, base = urlparse(url), urlparse(self.source_url)
        if parsed.scheme != "https" or parsed.hostname != base.hostname or parsed.username or parsed.password or parsed.port not in (None, 443):
            raise ValueError("教师节资源来源不符合预期")
        return url

    def _get(self, client, url: str) -> bytes:
        url = self._checked_url(url)
        for attempt in range(2):
            try:
                response = client.get(url)
                response.raise_for_status()
                self._checked_url(str(response.url))
                if len(response.content) > 5_000_000:
                    raise ValueError("教师节资源文件过大")
                return response.content
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt:
                    raise
        raise RuntimeError("教师节资源下载失败")

    def update(self):
        if not self._lock.acquire(blocking=False):
            raise UpdateInProgressError("教师节题库正在更新")
        try:
            with self._client_factory() as client:
                index = self._get(client, self.source_url).decode("utf-8")
                app = self._get(client, _app_script_source(index)).decode("utf-8")
                chunk_url, module_id = resolve_keju_chunk(index, app, self.source_url, route_path="/jiaoshijie")
                rows = extract_teacher_records(self._get(client, chunk_url).decode("utf-8"), module_id)
                return self.publish(rows, lambda row: self._get(client, row["Pic"]), chunk_url=chunk_url)
        finally:
            self._lock.release()

    def publish(self, rows, fetch_image, *, chunk_url: str = ""):
        if len(rows) < self.minimum_records:
            raise ValueError("教师节题库记录数过少，保留原题库")
        try:
            previous = load_teacher_bank(self.data_dir)
        except (OSError, ValueError, KeyError, TypeError):
            previous = None
        if previous is not None and len(rows) < previous.count * .8:
            raise ValueError("教师节题库记录数异常减少，保留原题库")
        generations = (self.data_dir / "generations").resolve()
        generations.mkdir(parents=True, exist_ok=True)
        generation = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid4().hex[:12]
        stage = generations / (".tmp-" + generation)
        final = generations / generation
        stage.mkdir()
        (stage / "icons").mkdir()
        try:
            def materialize(row):
                self._checked_url(row["Pic"])
                data = fetch_image(row)
                decode_icon(data)
                extension = "png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "jpg" if data.startswith(b"\xff\xd8") else None
                if extension is None:
                    raise ValueError("技能图标格式不受支持")
                digest = hashlib.sha256(data).hexdigest()
                record = TeacherSkillRecord(
                    "teachers_day:" + str(row["Id"]), row["Name"], row["TypeName"], row["Type"],
                    row["Pic"], f"icons/{digest}.{extension}", digest,
                )
                return record, data
            with ThreadPoolExecutor(max_workers=4) as executor:
                assets = list(executor.map(materialize, rows))
            for record, data in assets:
                (stage / record.image_path).write_bytes(data)
            _write_json(stage / "records.json", [asdict(record) for record, _ in assets])
            metadata = {
                "schema_version": 1, "generation_id": generation,
                "source_url": self.source_url, "chunk_url": chunk_url,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "record_count": len(assets), "image_count": len({record.image_path for record, _ in assets}),
                "records_sha256": hashlib.sha256((stage / "records.json").read_bytes()).hexdigest(),
            }
            _write_json(stage / "metadata.json", metadata)
            os.replace(stage, final)
            snapshot = load_teacher_generation(self.data_dir, generation)
            # A pointer is published only after every record and icon can be read.
            pointer = self.data_dir / (".current-" + uuid4().hex + ".tmp")
            try:
                _write_json(pointer, {"schema_version": 1, "generation_id": generation,
                                      "previous_generation_id": previous.generation_id if previous else None})
                os.replace(pointer, self.data_dir / "current.json")
            finally:
                pointer.unlink(missing_ok=True)
            return snapshot
        finally:
            if stage.exists():
                assert stage.resolve().parent == generations and stage.name.startswith(".tmp-")
                shutil.rmtree(stage)
