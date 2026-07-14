from pathlib import Path


ROOT = Path(__file__).parents[2]


def test_release_workflow_uses_tag_sha_and_requires_polyform_noncommercial() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "actions/checkout@v7" in workflow
    assert "actions/setup-python@v6" in workflow
    assert 'Test-Path -LiteralPath ".\\LICENSE"' in workflow
    assert "# PolyForm Noncommercial License 1.0.0" in workflow
    assert 'license = "PolyForm-Noncommercial-1.0.0"' in workflow
    assert "-Commit $env:GITHUB_SHA" in workflow
    assert "--require-hashes -r requirements-release.txt" in workflow
    assert "--self-test" in workflow
    assert "Get-FileHash" in workflow
    assert "gh release create" in workflow
