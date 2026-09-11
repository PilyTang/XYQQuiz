from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
import shutil
from pathlib import Path
import re
from types import MappingProxyType
from typing import Mapping
from uuid import uuid4

import cv2
import numpy as np

from xyq_quiz.knowledge.models import normalize_text


@dataclass(frozen=True, slots=True)
class TeacherSkillRecord:
    source_id: str
    name: str
    category: str
    category_code: str
    source_url: str
    image_path: str
    image_sha256: str


@dataclass(frozen=True, slots=True)
class TeacherBankSnapshot:
    generation_id: str
    records: tuple[TeacherSkillRecord, ...]
    images: tuple[np.ndarray, ...]
    by_name: Mapping[str, tuple[int, ...]]
    metadata: Mapping
    directory: Path
    recovery_reason: str | None = None

    @property
    def count(self) -> int:
        return len(self.records)


def decode_icon(data: bytes) -> np.ndarray:
    if not data or len(data) > 512_000:
        raise ValueError("技能图标文件为空或过大")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or not all(8 <= side <= 256 for side in image.shape[:2]):
        raise ValueError("技能图标无法解码或尺寸无效")
    image.setflags(write=False)
    return image


def merge_teacher_supplements(source: Path, target: Path) -> int:
    """Add verified bundled supplements without deleting local records.

    Write a complete immutable generation before atomically switching the
    supplement pointer. The official generation and pointer are never touched.
    """
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target:
        return 0
    incoming = load_teacher_bank(source)
    existing = load_teacher_bank(target)
    records = list(existing.records)
    payloads = {r.image_sha256:(existing.directory/r.image_path).read_bytes() for r in records}
    seen = {(normalize_text(r.name),r.image_sha256) for r in records}
    ids = {r.source_id for r in records}
    added = 0
    for record in incoming.records:
        identity = (normalize_text(record.name),record.image_sha256)
        if identity in seen:
            continue
        payloads[record.image_sha256]=(incoming.directory/record.image_path).read_bytes()
        if record.source_id in ids:
            record=replace(record,source_id=record.source_id+':'+record.image_sha256[:16])
        if record.source_id in ids:
            raise ValueError('补充题库编号冲突，保留现有题库')
        records.append(record)
        seen.add(identity)
        ids.add(record.source_id)
        added+=1
    if not added:
        return 0
    encoded=(json.dumps([asdict(r) for r in records],ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    digest=hashlib.sha256(encoded).hexdigest()
    generation='merged-'+digest[:24]
    generations=target/'generations'
    generations.mkdir(parents=True,exist_ok=True)
    staged=generations/('.seed-'+uuid4().hex)
    pointer=target/('.current-'+uuid4().hex+'.tmp')
    try:
        staged.mkdir()
        (staged/'icons').mkdir()
        for record in records:
            (staged/record.image_path).write_bytes(payloads[record.image_sha256])
        (staged/'records.json').write_bytes(encoded)
        metadata=dict(schema_version=1,generation_id=generation,record_count=len(records),
                      image_count=len(records),records_sha256=digest,
                      source_url=incoming.metadata.get('source_url'),
                      merged_from=[existing.generation_id,incoming.generation_id])
        (staged/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
        from xyq_quiz.runtime.paths import _publish_seed, _fsync_file
        _publish_seed(staged,generations/generation)
        load_teacher_generation(target,generation)
        pointer.write_text(json.dumps(dict(schema_version=1,generation_id=generation)),encoding='utf-8')
        _fsync_file(pointer)
        os.replace(pointer,target/'current.json')
    finally:
        if staged.exists():
            shutil.rmtree(staged,ignore_errors=True)
        pointer.unlink(missing_ok=True)
    return added


def load_teacher_generation(root: Path, generation_id: str) -> TeacherBankSnapshot:
    root = Path(root).resolve()
    if not isinstance(generation_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", generation_id):
        raise ValueError("教师节题库 generation 无效")
    directory = (root / "generations" / generation_id).resolve()
    if directory.parent != root / "generations":
        raise ValueError("教师节题库路径越界")
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    raw = (directory / "records.json").read_bytes()
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1 or metadata.get("generation_id") != generation_id:
        raise ValueError("教师节题库元数据不一致")
    if hashlib.sha256(raw).hexdigest() != metadata.get("records_sha256"):
        raise ValueError("教师节题库记录校验失败")
    rows = json.loads(raw)
    if not isinstance(rows, list) or not rows or len(rows) != metadata.get("record_count"):
        raise ValueError("教师节题库记录数无效")
    records, images, names, ids = [], [], {}, set()
    for row in rows:
        record = TeacherSkillRecord(**row)
        if not all(isinstance(value, str) and value.strip() for value in row.values()):
            raise ValueError("教师节题库包含空字段")
        if not record.source_id.startswith("teachers_day:") or record.source_id in ids:
            raise ValueError("教师节题库编号重复或无效")
        if not normalize_text(record.name):
            raise ValueError("技能名称无效")
        if not re.fullmatch(r"[0-9a-f]{64}", record.image_sha256):
            raise ValueError("技能图片摘要无效")
        if record.image_path not in {
            f"icons/{record.image_sha256}.png", f"icons/{record.image_sha256}.jpg"
        }:
            raise ValueError("技能图片路径无效")
        image_path = (directory / record.image_path).resolve()
        if image_path.parent != directory / "icons":
            raise ValueError("技能图片路径越界")
        data = image_path.read_bytes()
        if hashlib.sha256(data).hexdigest() != record.image_sha256:
            raise ValueError("技能图片校验失败")
        images.append(decode_icon(data))
        names.setdefault(normalize_text(record.name), []).append(len(records))
        ids.add(record.source_id)
        records.append(record)
    if metadata.get("image_count") != len({record.image_path for record in records}):
        raise ValueError("教师节题库图片数不一致")
    # Confirmed incorrect source rendition. Match name AND content so an
    # upstream corrected icon remains eligible, regardless of record ID.
    rejected = [i for i,r in enumerate(records)
                if normalize_text(r.name) == normalize_text('佛法无边')
                and r.image_sha256 == '11e47e325fe0b80d5729795472fb9a80731128be16dc4d0bbf81f9300e027970']
    if rejected:
        records = [r for i,r in enumerate(records) if i not in rejected]
        images = [image for i,image in enumerate(images) if i not in rejected]
        names = {}
        for i,r in enumerate(records):
            names.setdefault(normalize_text(r.name),[]).append(i)
        metadata = dict(metadata, source_record_count=metadata['record_count'],
                        excluded_record_count=len(rejected),record_count=len(records),
                        image_count=len({r.image_path for r in records}))
    return TeacherBankSnapshot(
        generation_id, tuple(records), tuple(images),
        MappingProxyType({name: tuple(indexes) for name, indexes in names.items()}),
        MappingProxyType(metadata), directory,
    )


def load_teacher_bank(root: Path) -> TeacherBankSnapshot:
    pointer = json.loads((Path(root) / "current.json").read_text(encoding="utf-8"))
    if not isinstance(pointer, dict) or pointer.get("schema_version") != 1:
        raise ValueError("教师节题库指针版本无效")
    return with_teacher_supplements(load_teacher_generation(root, pointer["generation_id"]), root)


def with_teacher_supplements(bank: TeacherBankSnapshot, root: Path) -> TeacherBankSnapshot:
    """Combine independently stored supplements with an official generation."""
    supplement_root = Path(root) / "supplements"
    if not supplement_root.exists():
        return bank
    pointer = json.loads((supplement_root / "current.json").read_text(encoding="utf-8"))
    if not isinstance(pointer, dict) or pointer.get("schema_version") != 1:
        raise ValueError("教师节补充题库指针版本无效")
    extra = load_teacher_generation(supplement_root, pointer["generation_id"])
    records, images = list(bank.records), list(bank.images)
    ids = {record.source_id for record in records}
    keys = {(normalize_text(record.name), record.image_sha256) for record in records}
    for record, image in zip(extra.records, extra.images, strict=True):
        key = (normalize_text(record.name), record.image_sha256)
        if key in keys:
            continue
        if record.source_id in ids:
            raise ValueError("教师节补充题库编号冲突")
        records.append(record)
        images.append(image)
        ids.add(record.source_id)
        keys.add(key)
    names: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        names.setdefault(normalize_text(record.name), []).append(index)
    metadata = dict(bank.metadata)
    metadata.update(record_count=len(records), official_record_count=bank.count,
                    supplement_record_count=len(records)-bank.count,
                    supplement_generation_id=extra.generation_id,
                    image_count=len({record.image_sha256 for record in records}))
    return replace(bank, generation_id=bank.generation_id+"+"+extra.generation_id,
                   records=tuple(records), images=tuple(images),
                   by_name=MappingProxyType({name: tuple(indexes) for name, indexes in names.items()}),
                   metadata=MappingProxyType(metadata))


def recover_teacher_bank(root: Path, default_root: Path | None = None) -> TeacherBankSnapshot | None:
    """Recover only validated snapshots; never overwrite the user's broken files."""
    from dataclasses import replace
    try:
        return load_teacher_bank(root)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    generations = Path(root) / "generations"
    for directory in sorted(generations.glob("*"), reverse=True):
        if not directory.is_dir() or directory.name.startswith("."):
            continue
        try:
            return replace(with_teacher_supplements(load_teacher_generation(root, directory.name), root), recovery_reason="已恢复上一有效教师节题库")
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if default_root is not None and Path(default_root).resolve() != Path(root).resolve():
        try:
            return replace(load_teacher_bank(default_root), recovery_reason="使用内置教师节题库")
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return None
