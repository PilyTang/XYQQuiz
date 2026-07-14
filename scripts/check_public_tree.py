from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tomllib
from typing import Iterable


FORBIDDEN_PATH_PREFIXES = (
    "build/",
    "dist/",
    "release/",
    "docs/superpowers/",
)
PRIVATE_FIXTURE_PREFIX = "tests/fixtures/recognition/"
PRIVATE_FIXTURE_NAMES = {"manifest.json"}
PRIVATE_FIXTURE_SUFFIXES = {".png", ".jpg", ".jpeg"}
FORBIDDEN_BINARY_SUFFIXES = {".exe", ".zip", ".pyd", ".dll"}
TEXT_PATTERNS = (
    (
        "personal Windows user path",
        re.compile(r"(?i)[a-z]:[\\/]users[\\/](?!<(?:name|userprofile)>)[^\\/\s]+"),
    ),
    (
        "personal Unix user path",
        re.compile(r"/(?:Users|home)/(?!<(?:name|userprofile)>)[^/\s]+"),
    ),
    (
        "private key material",
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ),
    (
        "GitHub access token",
        re.compile(r"(?:github" r"_pat_|gh[pousr]_[A-Za-z0-9_]{20,})"),
    ),
)
EXPECTED_LICENSE_ID = "PolyForm-Noncommercial-1.0.0"
EXPECTED_LICENSE_HEADER = "# PolyForm Noncommercial License 1.0.0"
EXPECTED_LICENSE_URL = "https://polyformproject.org/licenses/noncommercial/1.0.0"


def _git_executable() -> str:
    discovered = shutil.which("git")
    if discovered:
        return discovered
    candidates = (
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Git"
        / "cmd"
        / "git.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        / "Microsoft Visual Studio"
        / "2022"
        / "Community"
        / "Common7"
        / "IDE"
        / "CommonExtensions"
        / "Microsoft"
        / "TeamFoundation"
        / "Team Explorer"
        / "Git"
        / "cmd"
        / "git.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("git executable not found")


def tracked_paths(root: Path) -> tuple[PurePosixPath, ...]:
    result = subprocess.run(
        [_git_executable(), "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return tuple(
        PurePosixPath(value.decode("utf-8", errors="strict"))
        for value in result.stdout.split(b"\0")
        if value
    )


def audit_paths(root: Path, paths: Iterable[PurePosixPath]) -> list[str]:
    findings: list[str] = []
    folded_seen: dict[str, str] = {}
    for relative in paths:
        rendered = relative.as_posix()
        folded = rendered.casefold()
        previous = folded_seen.get(folded)
        if previous is not None and previous != rendered:
            findings.append(f"case-colliding tracked paths: {previous} / {rendered}")
        folded_seen[folded] = rendered

        if any(folded.startswith(prefix) for prefix in FORBIDDEN_PATH_PREFIXES):
            findings.append(f"private/generated path is tracked: {rendered}")
        if folded.startswith(PRIVATE_FIXTURE_PREFIX):
            name = relative.name.casefold()
            if name in PRIVATE_FIXTURE_NAMES or relative.suffix.casefold() in PRIVATE_FIXTURE_SUFFIXES:
                findings.append(f"private recognition fixture is tracked: {rendered}")
        if relative.suffix.casefold() in FORBIDDEN_BINARY_SUFFIXES:
            findings.append(f"release/runtime binary is tracked: {rendered}")

        path = root.joinpath(*relative.parts)
        try:
            payload = path.read_bytes()
        except OSError as error:
            findings.append(f"cannot read tracked file {rendered}: {error}")
            continue
        if b"\0" in payload[:8192]:
            continue
        text = payload.decode("utf-8", errors="replace")
        for label, pattern in TEXT_PATTERNS:
            if pattern.search(text):
                findings.append(f"{label} in {rendered}")
    return findings


def audit_project_license(root: Path) -> list[str]:
    findings: list[str] = []
    license_path = root / "LICENSE"
    if not license_path.is_file():
        findings.append("project LICENSE is missing")
    else:
        text = license_path.read_text(encoding="utf-8")
        if EXPECTED_LICENSE_HEADER not in text or EXPECTED_LICENSE_URL not in text:
            findings.append(
                "project LICENSE is not PolyForm Noncommercial License 1.0.0"
            )

    project_path = root / "pyproject.toml"
    if not project_path.is_file():
        findings.append("pyproject.toml is missing")
    else:
        try:
            project = tomllib.loads(project_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as error:
            findings.append(f"cannot parse pyproject.toml: {error}")
        else:
            if project.get("project", {}).get("license") != EXPECTED_LICENSE_ID:
                findings.append(
                    f"pyproject.toml license must be {EXPECTED_LICENSE_ID}"
                )
    return findings


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    findings = audit_project_license(root)
    findings.extend(audit_paths(root, tracked_paths(root)))
    if findings:
        print("Public-tree audit failed:", file=sys.stderr)
        for finding in findings:
            print(f"- {finding}", file=sys.stderr)
        return 1
    print("Public-tree audit passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
