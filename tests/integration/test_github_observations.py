from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from multichannel.db import connect, migrate
from multichannel.sources.base import SourceObservationInput
import multichannel.sources.github as github_source
from multichannel.sources.github import (
    GitHubRelease,
    GitHubRepository,
    persist_github_repository,
    persist_observation,
    persist_release,
    write_capture,
)


class _ControlledConnection(sqlite3.Connection):
    fail_repository_lookup = False
    fail_release_lookup = False
    fail_commit = False

    def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
        if self.fail_repository_lookup and sql.startswith("SELECT id FROM github_repositories"):
            raise sqlite3.OperationalError("controlled repository lookup failure")
        if self.fail_release_lookup and sql.startswith("SELECT id FROM github_releases"):
            raise sqlite3.OperationalError("controlled release lookup failure")
        return super().execute(sql, parameters)

    def commit(self) -> None:
        if self.fail_commit:
            raise sqlite3.OperationalError("controlled commit failure")
        super().commit()


def _controlled_connection(path: Path) -> _ControlledConnection:
    connection = sqlite3.connect(path, factory=_ControlledConnection)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


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


def _release(repository_github_id: str = "42") -> GitHubRelease:
    return GitHubRelease(
        github_release_id="7",
        repository_github_id=repository_github_id,
        tag_name="v1.0.0",
        name=None,
        body=None,
        html_url="https://github.com/acme/widgets/releases/tag/v1.0.0",
        published_at="2026-08-20T00:00:00Z",
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


def test_concurrent_duplicate_observation_leaves_one_row_one_capture_and_no_temps(tmp_path: Path) -> None:
    database = tmp_path / "state.sqlite3"
    setup = connect(database)
    migrate(setup)
    repository_id = persist_github_repository(setup, _repository())
    setup.close()
    captures = tmp_path / ".runtime" / "captures"
    observation = _observation(datetime(2026, 8, 21, tzinfo=timezone.utc), b'{"stars":12}')

    def persist_from_independent_connection() -> str:
        connection = connect(database)
        try:
            return persist_observation(connection, observation, captures, repository_id=repository_id)
        finally:
            connection.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(persist_from_independent_connection) for _ in range(2)]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except sqlite3.IntegrityError:
                outcomes.append("integrity-error")

    assert sum(outcome == "integrity-error" for outcome in outcomes) == 1
    assert sum(outcome != "integrity-error" for outcome in outcomes) == 1
    check = connect(database)
    try:
        assert check.execute("SELECT COUNT(*) FROM source_observations").fetchone() == (1,)
    finally:
        check.close()
    capture_files = [path for path in captures.rglob("*") if path.is_file()]
    assert len(capture_files) == 1
    assert capture_files[0].is_file() and not capture_files[0].is_symlink()
    assert not [path for path in captures.rglob(".*.tmp") if path.is_file()]


def test_repository_and_release_helpers_preserve_an_active_caller_transaction(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    connection.execute("BEGIN")
    connection.execute(
        "INSERT INTO channels (id, slug, name, locale, enabled, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        ("caller-work", "caller-work", "Caller work", "en", 1, "2026-08-21T00:00:00Z"),
    )

    with pytest.raises(sqlite3.IntegrityError, match="no active transaction"):
        persist_github_repository(connection, _repository())
    with pytest.raises(sqlite3.IntegrityError, match="no active transaction"):
        persist_release(connection, _release(), "missing")

    assert connection.in_transaction
    assert connection.execute("SELECT COUNT(*) FROM channels").fetchone() == (1,)
    assert connection.execute("SELECT COUNT(*) FROM github_repositories").fetchone() == (0,)
    connection.rollback()


def test_repository_helper_rolls_back_its_own_failed_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _controlled_connection(tmp_path / "state.sqlite3")
    migrate(connection)
    connection.fail_repository_lookup = True

    with pytest.raises(sqlite3.OperationalError, match="controlled repository lookup failure"):
        persist_github_repository(connection, _repository())

    connection.fail_repository_lookup = False
    assert not connection.in_transaction
    assert connection.execute("SELECT COUNT(*) FROM github_repositories").fetchone() == (0,)


def test_release_helper_rolls_back_its_own_failed_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    connection = _controlled_connection(tmp_path / "state.sqlite3")
    migrate(connection)
    repository_id = persist_github_repository(connection, _repository())
    connection.fail_release_lookup = True

    with pytest.raises(sqlite3.OperationalError, match="controlled release lookup failure"):
        persist_release(connection, _release(), repository_id)

    connection.fail_release_lookup = False
    assert not connection.in_transaction
    assert connection.execute("SELECT COUNT(*) FROM github_releases").fetchone() == (0,)


def test_repository_identity_conflict_is_case_insensitive_and_does_not_rebind(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    first_id = persist_github_repository(connection, _repository())
    conflicting = GitHubRepository(
        github_id="99", owner="Acme", name="Widgets", canonical_url="https://github.com/Acme/Widgets",
        api_url="https://api.github.com/repos/Acme/Widgets", default_branch="main", description=None,
        language=None, license_spdx=None, topics=(), readme_url=None, readme_ref=None, readme_text=None,
        readme_sha256=None, created_at="2026-08-20T00:00:00Z",
    )

    with pytest.raises(sqlite3.IntegrityError):
        persist_github_repository(connection, conflicting)

    assert connection.execute("SELECT id, github_id FROM github_repositories").fetchall() == [(first_id, "42")]


def test_release_parent_identity_must_match_durable_repository(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    repository_id = persist_github_repository(connection, _repository())

    with pytest.raises(sqlite3.IntegrityError, match="release repository identity"):
        persist_release(connection, _release("99"), repository_id)

    assert connection.execute("SELECT COUNT(*) FROM github_releases").fetchone() == (0,)


@pytest.mark.parametrize(
    "html_url",
    [
        "https://github.com/other/project/releases/tag/v2.0.0",
        "https://github.com/acme%2Fother/widgets/releases/tag/v2.0.0",
    ],
)
def test_direct_release_persistence_rejects_wrong_parent_url_without_mutation(
    tmp_path: Path, html_url: str
) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    repository_id = persist_github_repository(connection, _repository())
    release_id = persist_release(connection, _release(), repository_id)
    before = connection.execute(
        "SELECT id, repository_id, github_release_id, tag_name, html_url FROM github_releases"
    ).fetchall()
    invalid = replace(_release(), tag_name="v2.0.0", html_url=html_url)

    with pytest.raises(sqlite3.IntegrityError, match="release URL did not match parent"):
        persist_release(connection, invalid, repository_id)

    assert connection.execute(
        "SELECT id, repository_id, github_release_id, tag_name, html_url FROM github_releases"
    ).fetchall() == before == [
        (release_id, repository_id, "7", "v1.0.0", "https://github.com/acme/widgets/releases/tag/v1.0.0")
    ]


def test_direct_release_persistence_accepts_a_valid_encoded_tag(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    repository_id = persist_github_repository(connection, _repository())
    release = replace(
        _release(),
        tag_name="release/2026",
        html_url="https://github.com/acme/widgets/releases/tag/release%2F2026",
    )

    release_id = persist_release(connection, release, repository_id)

    assert connection.execute(
        "SELECT id, repository_id, tag_name, html_url FROM github_releases"
    ).fetchall() == [
        (release_id, repository_id, "release/2026", "https://github.com/acme/widgets/releases/tag/release%2F2026")
    ]


def test_rollback_cleanup_keeps_a_replacement_capture_it_does_not_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    connection = _controlled_connection(tmp_path / "state.sqlite3")
    migrate(connection)
    repository_id = persist_github_repository(connection, _repository())
    captures = tmp_path / ".runtime" / "captures"
    observation = _observation(datetime(2026, 8, 21, tzinfo=timezone.utc), b'{"id":42}')
    original_finalize = github_source._finalize_capture

    def publish_then_replace(*args: object) -> object:
        publication = original_finalize(*args)
        assert publication
        target = args[2]
        assert isinstance(target, Path)
        target.unlink()
        target.write_bytes(b"replacement")
        return publication

    monkeypatch.setattr(github_source, "_finalize_capture", publish_then_replace)
    connection.fail_commit = True

    with pytest.raises(sqlite3.OperationalError, match="controlled commit failure"):
        persist_observation(connection, observation, captures, repository_id=repository_id)

    connection.fail_commit = False
    capture_files = [path for path in captures.rglob("*") if path.is_file()]
    assert [path.read_bytes() for path in capture_files] == [b"replacement"]
    assert connection.execute("SELECT COUNT(*) FROM source_observations").fetchone() == (0,)
    assert not [path for path in captures.rglob(".*.tmp") if path.is_file()]
