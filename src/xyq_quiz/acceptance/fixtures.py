from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
import re
from typing import Any

import cv2
import numpy as np


class ManifestError(ValueError):
    pass


class FixtureKind(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"


class Provenance(StrEnum):
    WEB = "web"
    LOCAL_WGC = "local_wgc"


_CASE_FIELDS = {
    "file",
    "kind",
    "expected_source_id",
    "expected_option_index",
    "window_size",
    "dpi",
    "provenance",
    "human_verified",
    "sha256",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class RecognitionFixture:
    file: str
    kind: FixtureKind
    expected_source_id: str | None
    expected_option_index: int | None
    window_size: tuple[int, int]
    dpi: tuple[int, int]
    provenance: Provenance
    human_verified: bool
    sha256: str

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["provenance"] = self.provenance.value
        payload["window_size"] = list(self.window_size)
        payload["dpi"] = list(self.dpi)
        return payload


@dataclass(frozen=True, slots=True)
class RecognitionManifest:
    schema_version: int
    cases: tuple[RecognitionFixture, ...]
    path: Path

    def write(self, path: Path | None = None) -> Path:
        destination = Path(path or self.path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.schema_version,
            "cases": [case.to_json() for case in self.cases],
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination


def load_manifest(path: Path, *, require_assets: bool = True) -> RecognitionManifest:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read recognition manifest: {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ManifestError("manifest must be a JSON object")
    unknown = set(payload) - {"schema_version", "cases"}
    if unknown:
        raise ManifestError(f"manifest has unknown fields: {sorted(unknown)}")
    if payload.get("schema_version") != 1:
        raise ManifestError("schema_version must be 1")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise ManifestError("cases must be a list")
    cases = tuple(_parse_case(raw, index) for index, raw in enumerate(raw_cases))
    names = [case.file.casefold() for case in cases]
    if len(names) != len(set(names)):
        raise ManifestError("fixture file names must be unique")
    manifest = RecognitionManifest(1, cases, manifest_path)
    if require_assets:
        for case in cases:
            _validate_asset(manifest_path.parent / case.file, case)
    return manifest


def _parse_case(raw: Any, index: int) -> RecognitionFixture:
    if not isinstance(raw, dict):
        raise ManifestError(f"case {index} must be an object")
    unknown = set(raw) - _CASE_FIELDS
    missing = _CASE_FIELDS - set(raw)
    if unknown:
        raise ManifestError(f"case {index} has unknown fields: {sorted(unknown)}")
    if missing:
        raise ManifestError(f"case {index} is missing fields: {sorted(missing)}")
    filename = raw["file"]
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or Path(filename).suffix.casefold() not in {".png", ".jpg", ".jpeg"}
    ):
        raise ManifestError(f"case {index} file must be a plain file name ending in PNG or JPEG")
    try:
        kind = FixtureKind(raw["kind"])
        provenance = Provenance(raw["provenance"])
    except ValueError as exc:
        raise ManifestError(f"case {index} has invalid kind or provenance") from exc
    expected_prefix = "keju-" if kind is FixtureKind.POSITIVE else "non-keju-"
    if not filename.casefold().startswith(expected_prefix):
        raise ManifestError(f"{kind.value} fixture file must start with {expected_prefix}")
    source_id = raw["expected_source_id"]
    if source_id is not None and (not isinstance(source_id, str) or not source_id.strip()):
        raise ManifestError(f"case {index} expected source id must be a non-empty string or null")
    option = raw["expected_option_index"]
    if option is not None and (not isinstance(option, int) or isinstance(option, bool) or option not in range(4)):
        raise ManifestError(f"case {index} expected option index must be 0 through 3 or null")
    window_size = _pair(raw["window_size"], "window_size", index)
    dpi = _pair(raw["dpi"], "dpi", index)
    verified = raw["human_verified"]
    if not isinstance(verified, bool):
        raise ManifestError(f"case {index} human_verified must be boolean")
    if verified and kind is FixtureKind.POSITIVE and source_id is None:
        raise ManifestError(f"case {index} verified positive requires expected source id")
    if verified and kind is FixtureKind.POSITIVE and option is None:
        raise ManifestError(f"case {index} verified positive requires expected option index")
    if kind is FixtureKind.NEGATIVE and (source_id is not None or option is not None):
        raise ManifestError(f"case {index} negative expectations must be null")
    sha256 = raw["sha256"]
    if not isinstance(sha256, str) or _SHA256.fullmatch(sha256) is None:
        raise ManifestError(f"case {index} sha256 must contain 64 lowercase hex characters")
    return RecognitionFixture(
        filename,
        kind,
        source_id,
        option,
        window_size,
        dpi,
        provenance,
        verified,
        sha256,
    )


def _pair(value: Any, name: str, index: int) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in value)
    ):
        raise ManifestError(f"case {index} {name} must contain two positive integers")
    return value[0], value[1]


def _validate_asset(path: Path, case: RecognitionFixture) -> None:
    try:
        image = decode_png_or_jpeg(path)
    except ValueError as exc:
        raise ManifestError(str(exc)) from exc
    height, width = image.shape[:2]
    if (width, height) != case.window_size:
        raise ManifestError(
            f"fixture {case.file} window_size is {case.window_size}, image is {(width, height)}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != case.sha256:
        raise ManifestError(f"fixture {case.file} sha256 does not match")


def decode_png_or_jpeg(path: Path):
    image_path = Path(path)
    try:
        contents = image_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"fixture is missing or unreadable: {image_path}") from exc
    suffix = image_path.suffix.casefold()
    is_png = contents.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = len(contents) >= 4 and contents[:3] == b"\xff\xd8\xff"
    expected_matches = (suffix == ".png" and is_png) or (
        suffix in {".jpg", ".jpeg"} and is_jpeg
    )
    if not expected_matches:
        raise ValueError(f"fixture must be an actual PNG or JPEG matching its extension: {image_path}")
    encoded = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"fixture is not a decodable actual PNG or JPEG: {image_path}")
    return image


__all__ = [
    "FixtureKind",
    "ManifestError",
    "Provenance",
    "RecognitionFixture",
    "RecognitionManifest",
    "decode_png_or_jpeg",
    "load_manifest",
]
