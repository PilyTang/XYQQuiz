"""Publish a verified portable ZIP, then expose its update manifest on CNB.

Credentials are read from a private file or CNB_TOKEN, never stored in Git.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import traceback
from urllib.parse import urlparse
import zipfile

import httpx

from xyq_quiz.updates import REPOSITORY, MANIFEST_URL, validate_manifest, version_tuple

API = "https://api.cnb.cool/pilytang/XYQQuiz"


def read_token(path: Path | None) -> str:
    raw = path.read_text(encoding="utf-8-sig") if path else os.environ.get("CNB_TOKEN", "")
    candidates = [line.split(":", 1)[-1].strip() for line in raw.splitlines()]
    candidates = [value for value in candidates if re.fullmatch(r"[A-Za-z0-9_.-]{24,}", value)]
    if len(candidates) != 1:
        raise ValueError("请提供只包含一个有效令牌的私有文件或 CNB_TOKEN")
    return candidates[0]


def prepare(package: Path, notes_path: Path) -> dict:
    with zipfile.ZipFile(package) as archive:
        manifests = [name for name in archive.namelist() if name.endswith("/_internal/build-manifest.json")]
        if len(manifests) != 1:
            raise ValueError("包内必须有唯一构建清单")
        version = json.loads(archive.read(manifests[0]))["app_version"]
    version_tuple(version)
    expected_name = f"XYQQuiz-v{version}-win10-win11-x64.zip"
    if package.name != expected_name:
        raise ValueError("安装包名称与构建版本不一致")
    with package.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    checksum = Path(str(package) + ".sha256").read_text(encoding="ascii").split()[0]
    if checksum != digest:
        raise ValueError("安装包 SHA-256 校验失败")
    return validate_manifest({"schema_version": 1, "version": version,
        "download_url": f"{REPOSITORY}/-/releases/download/v{version}/{package.name}",
        "sha256": digest, "notes": notes_path.read_text(encoding="utf-8")})


def publish(package: Path, manifest: dict, token: str) -> None:
    # No token in URL, command arguments, stored remote config, or console output.
    env = os.environ.copy()
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "http.https://cnb.cool/.extraheader",
        "GIT_CONFIG_VALUE_0": "Authorization: Basic " + base64.b64encode(f"cnb:{token}".encode()).decode(),
        "GIT_CONFIG_KEY_1": "credential.helper", "GIT_CONFIG_VALUE_1": ""})
    for name in list(env):
        if name.startswith("GIT_TRACE") or name == "GIT_CURL_VERBOSE":
            env.pop(name)

    def git(root, *args):
        result = subprocess.run(["git", "-C", str(root), *args], env=env,
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)
        if result.returncode:
            safe = result.stderr.replace(token, "[redacted]").replace(
                env["GIT_CONFIG_VALUE_0"].removeprefix("Authorization: Basic "), "[redacted]")
            print(safe[:2000], flush=True)
            raise RuntimeError(f"Git {args[0]} 失败；请检查仓库内容读写权限（输出已隐藏以保护凭据）")
        return result.stdout.strip()

    with tempfile.TemporaryDirectory(prefix="xyqquiz-cnb-") as directory:
        print("Preparing CNB release repository...", flush=True)
        root = Path(directory)
        git(root, "clone", "--quiet", REPOSITORY + ".git", "repo")
        root /= "repo"
        git(root, "config", "user.name", "XYQQuiz release")
        git(root, "config", "user.email", "release@xyqquiz.invalid")
        branches = git(root, "branch", "--list", "main")
        if not branches:
            git(root, "checkout", "-b", "main")
        else:
            git(root, "checkout", "main")
        if not git(root, "ls-files"):
            (root / "README.md").write_text("# XYQQuiz 发布仓库\n\nWindows 便携包、更新说明及版本检查信息。\n", encoding="utf-8")
            git(root, "add", "README.md")
            git(root, "commit", "-m", "Initialize XYQQuiz releases")
            git(root, "push", "origin", "HEAD:main")
        latest_path = root / "latest.json"
        if latest_path.exists():
            previous = validate_manifest(json.loads(latest_path.read_text(encoding="utf-8")))
            if version_tuple(previous["version"]) > version_tuple(manifest["version"]):
                raise ValueError("拒绝将最新版本回退到旧版")
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.cnb.api+json"}
        with httpx.Client(headers=headers, timeout=30) as client:
            print("Preparing release and attachments...", flush=True)
            tag = "v" + manifest["version"]
            response = client.get(f"{API}/-/releases/tags/{tag}")
            if response.status_code == 404:
                response = client.post(f"{API}/-/releases", json={
                    "tag_name": tag, "target_commitish": git(root, "rev-parse", "HEAD"),
                    "name": "XYQQuiz " + manifest["version"], "body": manifest["notes"],
                    "draft": True, "prerelease": False, "make_latest": "false"})
            response.raise_for_status()
            release = response.json()
            release_id = release["id"]
            if release.get("draft"):
                for asset in (package, Path(str(package) + ".sha256")):
                    print(f"Uploading {asset.name}...", flush=True)
                    response = client.post(f"{API}/-/releases/{release_id}/asset-upload-url", json={
                        "asset_name": asset.name, "size": asset.stat().st_size,
                        "overwrite": True, "ttl": 0})
                    response.raise_for_status()
                    upload = response.json()
                    parsed = urlparse(upload["upload_url"])
                    if parsed.scheme != "https" or not parsed.hostname:
                        raise ValueError("无效上传地址")
                    # Dedicated client: never send the CNB token to object storage.
                    with asset.open("rb") as stream, httpx.Client(timeout=180) as uploader:
                        uploader.put(upload["upload_url"], content=stream,
                                     headers={"Content-Length": str(asset.stat().st_size)}).raise_for_status()
                    verify_url = upload["verify_url"]
                    if not verify_url.startswith(f"{API}/-/releases/{release_id}/asset-upload-confirmation/"):
                        raise ValueError("上传确认地址不属于此仓库")
                    client.post(verify_url, params={"ttl": 0}).raise_for_status()
                client.patch(f"{API}/-/releases/{release_id}", json={
                    "draft": False, "make_latest": "true"}).raise_for_status()
        # Verify the public package before changing the version-discovery pointer.
        with httpx.Client(follow_redirects=True, timeout=60) as public:
            print("Verifying anonymous package download...", flush=True)
            digest = hashlib.sha256()
            with public.stream("GET", manifest["download_url"]) as response:
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    digest.update(chunk)
            if digest.hexdigest() != manifest["sha256"]:
                raise ValueError("公开下载 SHA-256 不一致，未更新 latest.json")
        latest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Publishing latest.json...", flush=True)
        git(root, "add", "latest.json")
        if git(root, "diff", "--cached", "--name-only"):
            git(root, "commit", "-m", f"Publish update metadata for {tag}")
            git(root, "push", "origin", "HEAD:main")
        response = httpx.get(MANIFEST_URL, follow_redirects=True, timeout=20)
        response.raise_for_status()
        if validate_manifest(response.json()) != manifest:
            raise ValueError("公开版本信息尚未同步，请稍后重试")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--token-file", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        manifest = prepare(args.package, args.notes)
        output = args.package.parent / f"latest-{manifest['version']}.json"
        output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not args.prepare_only:
            publish(args.package, manifest, read_token(args.token_file))
        print(f"{'Prepared' if args.prepare_only else 'Published and verified'}: {manifest['version']}")
        print(f"Manifest: {output}")
        return 0
    except Exception as error:
        # HTTP exceptions may include signed URLs. Never print exception text.
        status = getattr(getattr(error, "response", None), "status_code", None)
        frames = [f"{frame.name}:{frame.lineno}" for frame in traceback.extract_tb(error.__traceback__)
                  if Path(frame.filename).name == "publish_cnb.py"]
        print(f"CNB publication failed: {type(error).__name__}, HTTP={status}; latest metadata may remain unchanged.")
        print("Location: " + ", ".join(frames))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
