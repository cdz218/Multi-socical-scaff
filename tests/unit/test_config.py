from __future__ import annotations

import json
from pathlib import Path

import pytest

from multichannel.config import ConfigurationError, RuntimePaths, diagnostic_report, find_interpreter


def test_runtime_paths_are_repository_local_and_reject_escapes(tmp_path: Path) -> None:
    paths = RuntimePaths.from_repository(tmp_path)

    assert paths.models == tmp_path / ".runtime" / "models"
    with pytest.raises(ConfigurationError, match="outside"):
        paths.resolve_model_cache("../escape")


def test_runtime_paths_reject_symlink_escapes(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    (runtime / "models").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ConfigurationError, match="outside"):
        RuntimePaths.from_repository(tmp_path).resolve_model_cache("kokoro")


def test_runtime_paths_rejects_models_symlink_escape_during_construction(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside"
    outside.mkdir(exist_ok=True)
    runtime = tmp_path / ".runtime"
    runtime.mkdir()
    (runtime / "models").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ConfigurationError, match="outside"):
        RuntimePaths.from_repository(tmp_path)


def test_diagnostic_report_is_redacted_json() -> None:
    report = diagnostic_report(
        {
            "MULTICHANNEL_TOKEN": "secret",
            "DATABASE_URL": "postgresql://operator:password@db.example/app",
            "SAFE": "value",
        }
    )

    parsed = json.loads(report)
    assert parsed["environment"]["MULTICHANNEL_TOKEN"] == "[REDACTED]"
    assert parsed["environment"]["DATABASE_URL"] == "[REDACTED]"
    assert parsed["environment"]["SAFE"] == "value"
    assert "secret" not in report
    assert "postgresql://operator:password@db.example/app" not in report


def test_find_interpreter_requires_supported_version(tmp_path: Path) -> None:
    unsupported = tmp_path / "python"
    unsupported.write_text("#!/bin/sh\necho 'Python 3.10.0'\n")
    unsupported.chmod(0o755)

    with pytest.raises(ConfigurationError, match=">=3.11,<3.13"):
        find_interpreter((unsupported,))


def test_find_interpreter_skips_missing_candidate(tmp_path: Path) -> None:
    supported = tmp_path / "python"
    supported.write_text("#!/bin/sh\necho 'Python 3.11.15'\n")
    supported.chmod(0o755)

    assert find_interpreter((tmp_path / "missing-python", supported)) == supported
