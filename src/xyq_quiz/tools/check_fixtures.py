from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from xyq_quiz.acceptance.fixtures import ManifestError, load_manifest
from xyq_quiz.recognition.layout import LayoutProfile, validate_anchor_templates


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查真实科举 fixture 验收前置条件")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--layout", type=Path, action="append")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.manifest is None:
        print("必须显式传入 --manifest；默认不会运行或判定真实识别验收。", file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(args.manifest, require_assets=True)
    except ManifestError as exc:
        print(f"等待真实截图：{exc}", file=sys.stderr)
        return 2
    if not manifest.cases:
        print("等待真实截图：manifest 中尚无真实样本。", file=sys.stderr)
        return 2
    drafts = [case.file for case in manifest.cases if not case.human_verified]
    if drafts:
        print(f"等待人工核验：{', '.join(drafts)}", file=sys.stderr)
        return 2
    if not args.layout:
        print("等待真实校准：必须显式传入 --layout。", file=sys.stderr)
        return 2
    for layout_path in args.layout:
        try:
            profile = LayoutProfile.load(layout_path)
            missing = [str(anchor.template_path) for anchor in profile.anchors if not anchor.template_path.is_file()]
        except (OSError, ValueError) as exc:
            print(f"等待真实校准：{layout_path}：{exc}", file=sys.stderr)
            return 2
        if missing:
            print(f"等待真实 anchor：{', '.join(missing)}", file=sys.stderr)
            return 2
        try:
            validate_anchor_templates(profile)
        except ValueError as exc:
            print(f"等待真实 anchor：{exc}", file=sys.stderr)
            return 2
    print(f"真实 fixture 前置文件已就绪（{len(args.layout)} 套布局）；请运行带显式参数的 pytest。此检查本身不代表识别通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
