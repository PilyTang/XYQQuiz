from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from typing import Sequence
from uuid import uuid4

from xyq_quiz.acceptance.fixtures import (
    FixtureKind,
    Provenance,
    RecognitionFixture,
    RecognitionManifest,
    decode_png_or_jpeg,
    load_manifest,
)


def import_fixture(
    source: Path,
    destination: Path,
    *,
    kind: FixtureKind,
    provenance: str | Provenance,
    dpi: tuple[int, int],
    filename: str,
) -> RecognitionFixture:
    source_path = Path(source)
    try:
        image = decode_png_or_jpeg(source_path)
    except ValueError as exc:
        raise ValueError(
            "source must be a readable PNG or JPEG image and an actual PNG or JPEG"
        ) from exc
    height, width = image.shape[:2]
    digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    case = RecognitionFixture(
        file=filename,
        kind=FixtureKind(kind),
        expected_source_id=None,
        expected_option_index=None,
        window_size=(width, height),
        dpi=dpi,
        provenance=Provenance(provenance),
        human_verified=False,
        sha256=digest,
    )
    output_dir = Path(destination)
    manifest_path = output_dir / "manifest.json"
    existing = load_manifest(manifest_path, require_assets=False) if manifest_path.exists() else RecognitionManifest(1, (), manifest_path)
    merged = RecognitionManifest(1, (*existing.cases, case), manifest_path)
    # Validate the entire future manifest before creating or copying anything.
    with tempfile.TemporaryDirectory() as temporary:
        draft = Path(temporary) / "manifest.json"
        merged.write(draft)
        load_manifest(draft, require_assets=False)
    target = output_dir / filename
    if target.exists():
        raise FileExistsError(f"fixture already exists: {target}")
    created_directory = not output_dir.exists()
    output_dir.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    image_temp = output_dir / f".{filename}.{token}.tmp"
    manifest_temp = output_dir / f".manifest.{token}.tmp"
    image_published = False
    try:
        shutil.copy2(source_path, image_temp)
        merged.write(manifest_temp)
        os.replace(image_temp, target)
        image_published = True
        os.replace(manifest_temp, manifest_path)
    except BaseException:
        image_temp.unlink(missing_ok=True)
        manifest_temp.unlink(missing_ok=True)
        if image_published:
            target.unlink(missing_ok=True)
        if created_directory:
            try:
                output_dir.rmdir()
            except OSError:
                pass
        raise
    return case


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="校验并导入真实科举截图，生成待人工填写的 manifest 草稿")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--filename", required=True)
    parser.add_argument("--kind", choices=[item.value for item in FixtureKind], required=True)
    parser.add_argument("--provenance", choices=[item.value for item in Provenance], required=True)
    parser.add_argument("--dpi", type=int, default=96)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    case = import_fixture(
        args.source,
        args.output_dir,
        kind=FixtureKind(args.kind),
        provenance=args.provenance,
        dpi=(args.dpi, args.dpi),
        filename=args.filename,
    )
    print(f"已导入 {case.file}，尺寸 {case.window_size}，SHA-256 {case.sha256}")
    print("expected_source_id / expected_option_index 仍为 null；必须人工核验后填写。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
