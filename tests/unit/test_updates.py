import asyncio
import json

import httpx
import pytest

from xyq_quiz.updates import UpdateChecker, REPOSITORY, validate_manifest, version_tuple


def manifest(version="0.9.0"):
    return {"schema_version": 1, "version": version, "notes": "修复与更新",
            "sha256": "a" * 64, "download_url": f"{REPOSITORY}/-/releases/download/v{version}/XYQQuiz-v{version}-win10-win11-x64.zip"}


@pytest.mark.asyncio
async def test_daily_cache_survives_restart_and_manual_bypasses(tmp_path):
    calls = []
    def handle(request):
        calls.append(request)
        assert "authorization" not in request.headers
        return httpx.Response(200, json=manifest())
    path = tmp_path / "updates.json"
    transport = httpx.MockTransport(handle)
    checker = UpdateChecker(path, transport=transport, clock=lambda: 100000)
    assert (await checker.check())["available"]
    checker = UpdateChecker(path, transport=transport, clock=lambda: 100001)
    await checker.check()
    assert len(calls) == 1
    await checker.check(manual=True)
    assert len(calls) == 2
    checker = UpdateChecker(path, transport=transport, clock=lambda: 200000)
    await checker.check()
    assert len(calls) == 3


@pytest.mark.asyncio
async def test_disabled_persists_but_manual_works(tmp_path):
    calls = []
    transport = httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(200, json=manifest()))
    path = tmp_path / "updates.json"
    checker = UpdateChecker(path, transport=transport)
    await checker.set_enabled(False)
    checker = UpdateChecker(path, transport=transport)
    assert not (await checker.check())["enabled"]
    assert not calls
    assert (await checker.check(manual=True))["available"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_failure_retains_previous_success_and_throttles(tmp_path):
    responses = iter([httpx.Response(200, json=manifest()), httpx.Response(503)])
    checker = UpdateChecker(tmp_path / "state", transport=httpx.MockTransport(lambda _: next(responses)))
    await checker.check()
    previous = checker.checked_at
    result = await checker.check(manual=True)
    assert result["status"] == "unavailable"
    assert result["latest"] == manifest()
    assert result["checked_at"] == previous
    assert (await checker.check())["status"] == "unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("response", [httpx.Response(404), httpx.Response(200, text="<html>login</html>"),
    httpx.Response(200, content=b"x" * 32769), httpx.Response(200, json={"schema_version": 2})])
async def test_bad_remote_never_claims_current(tmp_path, response):
    checker = UpdateChecker(tmp_path / "state", transport=httpx.MockTransport(lambda _: response))
    result = await checker.check()
    assert result["status"] == "unavailable"
    assert result["latest"] is None


@pytest.mark.asyncio
async def test_concurrent_auto_checks_share_daily_result(tmp_path):
    calls = []
    async def handle(request):
        calls.append(request)
        await asyncio.sleep(.01)
        return httpx.Response(200, json=manifest())
    checker = UpdateChecker(tmp_path / "state", transport=httpx.MockTransport(handle))
    await asyncio.gather(checker.check(), checker.check())
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_older_remote_does_not_replace_known_newer_version(tmp_path):
    responses = iter([manifest("0.9.1"), manifest("0.9.0")])
    checker = UpdateChecker(tmp_path / "state", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=next(responses))))
    await checker.check()
    result = await checker.check(manual=True)
    assert result["status"] == "unavailable"
    assert result["latest"]["version"] == "0.9.1"


@pytest.mark.parametrize("raw", ["not-json", "[]", '{"latest":{}}'])
def test_bad_local_cache_recovers(tmp_path, raw):
    path = tmp_path / "state"
    path.write_text(raw)
    assert UpdateChecker(path).snapshot()["status"] == "unchecked"


def test_numeric_versions_and_download_boundary():
    assert version_tuple("0.10.0") > version_tuple("0.9.9")
    for version in ("0.9.0-beta.1", "v0.9.0", "1.2", None):
        with pytest.raises(ValueError):
            version_tuple(version)
    for url in ("javascript:alert(1)", "https://evil.example/update.zip", REPOSITORY + "/evil.zip"):
        with pytest.raises(ValueError):
            validate_manifest({**manifest(), "download_url": url})
