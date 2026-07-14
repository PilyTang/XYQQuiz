from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path


DEPENDENCIES = (
    "fastapi",
    "uvicorn",
    "websockets",
    "windows-capture",
    "numpy",
    "opencv-python",
    "rapidocr",
    "onnxruntime",
    "rapidfuzz",
    "httpx",
    "pyjson5",
    "pydantic",
    "pyinstaller",
)


def load_question_bank_manifest(package_root: Path) -> dict[str, object]:
    data_root = package_root / "_internal" / "defaults" / "data"
    pointer = json.loads((data_root / "current.json").read_text(encoding="utf-8"))
    if not isinstance(pointer, dict):
        raise ValueError("question-bank current.json must contain an object")
    generation_id = pointer.get("generation_id")
    if not isinstance(generation_id, str) or not generation_id:
        raise ValueError("question-bank generation_id is missing")
    generation_root = data_root / "generations" / generation_id
    metadata = json.loads((generation_root / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("question-bank metadata.json must contain an object")
    if metadata.get("generation_id") != generation_id:
        raise ValueError("question-bank metadata generation_id does not match current.json")
    required = {
        "source_url": str,
        "updated_at": str,
        "published_record_count": int,
        "sha256": str,
    }
    for name, expected_type in required.items():
        value = metadata.get(name)
        if not isinstance(value, expected_type) or isinstance(value, bool):
            raise ValueError(f"question-bank metadata field is invalid: {name}")
    question_path = generation_root / "keju_questions.json"
    digest = hashlib.sha256(question_path.read_bytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError("question-bank file SHA-256 does not match metadata")
    return {
        "generation_id": generation_id,
        "source_url": metadata["source_url"],
        "retrieved_at": metadata["updated_at"],
        "record_count": metadata["published_record_count"],
        "sha256": digest,
    }


def build_manifest(
    package_root: Path,
    output: Path,
    *,
    version: str,
    commit: str,
    signed: bool,
) -> dict[str, object]:
    package_root = package_root.resolve()
    output = output.resolve()
    files = []
    for path in sorted(package_root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file() or path == output:
            continue
        relative = path.relative_to(package_root).as_posix()
        payload = path.read_bytes()
        files.append(
            {
                "path": relative,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    versions = {}
    for name in DEPENDENCIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return {
        "schema_version": 1,
        "app_version": version,
        "git_commit": commit,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "build_system": "Windows x64 / PyInstaller onedir",
        "target": "Windows 10 1903+ / Windows 11 x64",
        "verified_platforms": ["Windows 11 x64"],
        "signed": signed,
        "question_bank": load_question_bank_manifest(package_root),
        "dependencies": versions,
        "files": files,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--signed", action="store_true")
    args = parser.parse_args()
    root = args.package_root.resolve()
    output = args.output.resolve()
    if root not in output.parents:
        parser.error("--output must be inside --package-root")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        root,
        output,
        version=args.version,
        commit=args.commit,
        signed=args.signed,
    )
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
