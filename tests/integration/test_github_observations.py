from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from multichannel.db import connect, migrate
from multichannel.sources.base import SourceObservationInput
from multichannel.sources.github import GitHubRepository, persist_github_repository, persist_observation, write_capture


def _repository() -> GitHubRepository:
    return GitHubRepository(
        github_id="42", owner="acme", name="widgets", canonical_url="https://github.com/acme/widgets",
        api_url="https://api.github.com/repos/acme/widgets", default_branch="main", description=None,
        language=None, license_spdx=None, topics=(), readme_url=None, readme_ref=None, readme_text=None,
        readme_sha256=None, created_at="2026-08-20T00:00:00Z",
    )


def _observation(observed_at: datetime, raw_bytes: bytes) -> SourceObservationInput:
    return SourceObservationInput(
        source_kind="github_repository", source_id="github:42:repository", observed_at=observed_at,
        metrics={"stars": 12, "forks": 3}, raw_bytes=raw_bytes, rate_limit_remaining=10,
    )


def test_second_capture_is_a_second_observation_not_a_second_repository(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    assert migrate(connection) == 2
    captures = tmp_path / ".runtime" / "captures"
    first_time = datetime(2026, 8, 21, tzinfo=timezone.utc)
    repository_id = persist_github_repository(connection, _repository())
    persist_observation(connection, _observation(first_time, b'{"stars":12}'), captures, repository_id=repository_id)
    persist_observation(
        connection, _observation(first_time + timedelta(minutes=1), b'{"stars":13}'), captures,
        repository_id=repository_id,
    )

    assert connection.execute("SELECT COUNT(*) FROM github_repositories").fetchone() == (1,)
    assert connection.execute("SELECT COUNT(*) FROM source_observations").fetchone() == (2,)
    with pytest.raises(sqlite3.IntegrityError):
        persist_observation(connection, _observation(first_time, b'{"stars":12}'), captures, repository_id=repository_id)
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "INSERT INTO source_observations (id, source_kind, source_identity, observed_at, metrics_json, raw_sha256, raw_path, created_at) "
            "VALUES ('invalid', 'github_repository', 'invalid', '2026-08-21T00:00:00Z', '{}', ?, 'x', '2026-08-21T00:00:00Z')",
            ("a" * 64,),
        )
    capture_paths = [path.resolve() for path in captures.rglob("*") if path.is_file()]
    assert capture_paths and all(path.is_relative_to(captures.resolve()) for path in capture_paths)


def test_observation_kind_identity_and_parent_must_agree_without_mutation(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    repository_id = persist_github_repository(connection, _repository())
    observation = _observation(datetime(2026, 8, 21, tzinfo=timezone.utc), b'{"stars":12}')
    captures = tmp_path / ".runtime" / "captures"

    with pytest.raises(sqlite3.IntegrityError):
        persist_observation(
            connection,
            SourceObservationInput("github_release", "attacker-controlled", observation.observed_at, {}, observation.raw_bytes),
            captures,
            repository_id=repository_id,
        )
    assert connection.execute("SELECT COUNT(*) FROM source_observations").fetchone() == (0,)
    assert not captures.exists()

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """INSERT INTO source_observations
               (id, source_kind, source_identity, observed_at, github_repository_id, metrics_json,
                raw_sha256, raw_path, created_at)
               VALUES ('mismatch', 'github_release', 'github_repository:' || ?,
                       '2026-08-21T00:00:00Z', ?, '{}', ?, 'x', '2026-08-21T00:00:00Z')""",
            (repository_id, repository_id, "a" * 64),
        )
    assert connection.execute("SELECT COUNT(*) FROM source_observations").fetchone() == (0,)


def test_persistence_derives_repository_identity_and_duplicate_leaves_no_orphan(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    repository_id = persist_github_repository(connection, _repository())
    observed_at = datetime(2026, 8, 21, tzinfo=timezone.utc)
    observation = SourceObservationInput("github_repository", "untrusted-source-id", observed_at, {}, b'{"id":42}')
    captures = tmp_path / ".runtime" / "captures"
    persist_observation(connection, observation, captures, repository_id=repository_id)
    saved_identity, saved_path = connection.execute(
        "SELECT source_identity, raw_path FROM source_observations"
    ).fetchone()
    assert saved_identity == f"github_repository:{repository_id}"
    capture = captures / saved_path
    before = capture.read_bytes()

    with pytest.raises(sqlite3.IntegrityError):
        persist_observation(connection, observation, captures, repository_id=repository_id)

    assert capture.read_bytes() == before
    assert [path for path in captures.rglob("*") if path.is_file()] == [capture]


def test_observation_persistence_keeps_foreign_keys_enabled_without_toggles(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    repository_id = persist_github_repository(connection, _repository())
    observation = _observation(datetime(2026, 8, 21, tzinfo=timezone.utc), b'{"id":42}')
    captures = tmp_path / ".runtime" / "captures"
    traced_sql: list[str] = []
    connection.set_trace_callback(traced_sql.append)

    assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
    persist_observation(connection, observation, captures, repository_id=repository_id)
    assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)

    with pytest.raises(sqlite3.IntegrityError):
        persist_observation(connection, observation, captures, repository_id=repository_id)

    assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
    assert not any(
        statement.strip().casefold().startswith("pragma foreign_keys=")
        for statement in traced_sql
    )


@pytest.mark.parametrize(
    ("source_kind", "parent_kwargs"),
    [
        ("github_repository", {"repository_id": "missing-repository"}),
        ("github_release", {"release_id": "missing-release"}),
    ],
)
def test_unknown_github_parent_rejects_observation_without_a_capture_orphan(
    tmp_path: Path, source_kind: str, parent_kwargs: dict[str, str]
) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    captures = tmp_path / ".runtime" / "captures"
    observation = SourceObservationInput(
        source_kind=source_kind, source_id="untrusted", observed_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
        metrics={}, raw_bytes=b'{"id":"unknown"}',
    )

    with pytest.raises(sqlite3.IntegrityError, match="unknown GitHub"):
        persist_observation(connection, observation, captures, **parent_kwargs)

    assert connection.execute("SELECT COUNT(*) FROM source_observations").fetchone() == (0,)
    assert not captures.exists()


def test_write_capture_rejects_symlinked_source_directory_before_external_write(tmp_path: Path) -> None:
    captures = tmp_path / ".runtime" / "captures"
    outside = tmp_path / "outside"
    captures.mkdir(parents=True)
    outside.mkdir()
    (captures / "github_repository").symlink_to(outside, target_is_directory=True)
    observation = _observation(datetime(2026, 8, 21, tzinfo=timezone.utc), b'{"safe":true}')

    with pytest.raises(ValueError, match="symlink"):
        write_capture(captures, observation)

    assert list(outside.iterdir()) == []
