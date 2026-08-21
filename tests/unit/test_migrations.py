from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from uuid import UUID

import pytest

from multichannel import db
from multichannel.db import MigrationError, connect, migrate
import importlib.util


_FACTORIES = importlib.util.spec_from_file_location("test_factories", Path(__file__).parents[1] / "factories.py")
assert _FACTORIES and _FACTORIES.loader
factories = importlib.util.module_from_spec(_FACTORIES)
_FACTORIES.loader.exec_module(factories)


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def test_migrate_is_idempotent_and_creates_current_tables(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")

    assert migrate(connection) == 2
    assert migrate(connection) == 2
    assert _table_names(connection) == {
        "schema_migrations",
        "channels",
        "platform_accounts",
        "job_runs",
        "job_events",
        "requeue_requests",
        "github_repositories",
        "github_releases",
        "source_observations",
    }
    assert not _table_names(connection).intersection({"concepts", "publishers", "media"})
    assert connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone() == (2,)


def test_migrate_refuses_a_changed_recorded_migration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    migration = migrations / "0001_foundation.sql"
    migration.write_text(
        "CREATE TABLE schema_migrations (migration_id TEXT PRIMARY KEY, applied_at TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE);\n"
        "CREATE TABLE first_version (id TEXT PRIMARY KEY);",
        encoding="utf-8",
    )
    monkeypatch.setattr(db, "MIGRATIONS_DIRECTORY", migrations)
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    migration.write_text("CREATE TABLE changed_version (id TEXT PRIMARY KEY);", encoding="utf-8")

    with pytest.raises(MigrationError, match="checksum differs"):
        migrate(connection)


def test_migrate_rejects_checksum_mismatch_before_applying_later_migrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    first = migrations / "0001_first.sql"
    first.write_text("CREATE TABLE first_version (id TEXT PRIMARY KEY);", encoding="utf-8")
    (migrations / "0002_second.sql").write_text(
        "CREATE TABLE second_version (id TEXT PRIMARY KEY);", encoding="utf-8"
    )
    monkeypatch.setattr(db, "MIGRATIONS_DIRECTORY", migrations)
    connection = connect(tmp_path / "state.sqlite3")
    assert migrate(connection) == 2
    connection.execute("DROP TABLE second_version")
    connection.execute("DELETE FROM schema_migrations WHERE migration_id = '0002'")
    connection.commit()
    first.write_text("CREATE TABLE changed_first_version (id TEXT PRIMARY KEY);", encoding="utf-8")

    with pytest.raises(MigrationError, match="migration 0001 checksum differs"):
        migrate(connection)

    assert "second_version" not in _table_names(connection)
    assert connection.execute("SELECT migration_id FROM schema_migrations").fetchall() == [("0001",)]


def test_migrate_rejects_unknown_history_before_applying_pending_migrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_first.sql").write_text(
        "CREATE TABLE first_version (id TEXT PRIMARY KEY);", encoding="utf-8"
    )
    monkeypatch.setattr(db, "MIGRATIONS_DIRECTORY", migrations)
    connection = connect(tmp_path / "state.sqlite3")
    unknown_checksum = "a" * 64
    connection.execute(
        """
        CREATE TABLE schema_migrations (
          migration_id TEXT PRIMARY KEY,
          applied_at TEXT NOT NULL,
          sha256 TEXT NOT NULL UNIQUE
        )
        """
    )
    connection.execute(
        "INSERT INTO schema_migrations (migration_id, applied_at, sha256) VALUES (?, ?, ?)",
        ("0002", "2026-08-21T00:00:00Z", unknown_checksum),
    )
    connection.commit()
    before = connection.execute(
        "SELECT migration_id, applied_at, sha256 FROM schema_migrations"
    ).fetchall()

    with pytest.raises(MigrationError, match="migration history"):
        migrate(connection)

    assert "first_version" not in _table_names(connection)
    assert connection.execute(
        "SELECT migration_id, applied_at, sha256 FROM schema_migrations"
    ).fetchall() == before


@pytest.mark.parametrize("recorded_id", ["0002", "not-a-migration"])
def test_migrate_rejects_non_prefix_history_before_applying_pending_migrations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recorded_id: str
) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    for migration_id in ("0001", "0002"):
        (migrations / f"{migration_id}_version.sql").write_text(
            f"CREATE TABLE version_{migration_id} (id TEXT PRIMARY KEY);", encoding="utf-8"
        )
    monkeypatch.setattr(db, "MIGRATIONS_DIRECTORY", migrations)
    connection = connect(tmp_path / "state.sqlite3")
    connection.execute(
        """
        CREATE TABLE schema_migrations (
          migration_id TEXT PRIMARY KEY,
          applied_at TEXT NOT NULL,
          sha256 TEXT NOT NULL UNIQUE
        )
        """
    )
    connection.execute(
        "INSERT INTO schema_migrations (migration_id, applied_at, sha256) VALUES (?, ?, ?)",
        (recorded_id, "2026-08-21T00:00:00Z", "b" * 64),
    )
    connection.commit()
    before = connection.execute("SELECT * FROM schema_migrations").fetchall()

    with pytest.raises(MigrationError, match="migration history"):
        migrate(connection)

    assert "version_0001" not in _table_names(connection)
    assert "version_0002" not in _table_names(connection)
    assert connection.execute("SELECT * FROM schema_migrations").fetchall() == before


def test_failed_migration_rolls_back_schema_and_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_partial.sql").write_text(
        "CREATE TABLE incomplete (id TEXT PRIMARY KEY);\nTHIS IS NOT SQL;",
        encoding="utf-8",
    )
    monkeypatch.setattr(db, "MIGRATIONS_DIRECTORY", migrations)
    connection = connect(tmp_path / "state.sqlite3")

    with pytest.raises(sqlite3.OperationalError):
        migrate(connection)

    assert "incomplete" not in _table_names(connection)
    assert "schema_migrations" not in _table_names(connection)


@pytest.mark.parametrize(
    "transaction_statement",
    [
        "BEGIN;",
        "COMMIT;",
        "ROLLBACK;",
        "SAVEPOINT migration_savepoint;",
        "RELEASE migration_savepoint;",
        "END;",
        "END TRANSACTION;",
    ],
)
def test_migrate_rejects_transaction_control_sql_before_any_schema_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transaction_statement: str
) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_first.sql").write_text(
        "CREATE TABLE first_version (id TEXT PRIMARY KEY);", encoding="utf-8"
    )
    monkeypatch.setattr(db, "MIGRATIONS_DIRECTORY", migrations)
    connection = connect(tmp_path / "state.sqlite3")
    assert migrate(connection) == 1
    before_history = connection.execute(
        "SELECT migration_id, applied_at, sha256 FROM schema_migrations"
    ).fetchall()
    (migrations / "0002_second.sql").write_text(
        "CREATE TABLE second_version (id TEXT PRIMARY KEY);", encoding="utf-8"
    )
    invalid = migrations / "0003_partial.sql"
    invalid.write_text(
        "CREATE TABLE incomplete (id TEXT PRIMARY KEY); "
        f"{transaction_statement} "
        "THIS IS NOT SQL;",
        encoding="utf-8",
    )

    with pytest.raises(MigrationError, match="transaction-control"):
        migrate(connection)

    assert "first_version" in _table_names(connection)
    assert "second_version" not in _table_names(connection)
    assert "incomplete" not in _table_names(connection)
    assert connection.execute(
        "SELECT migration_id, applied_at, sha256 FROM schema_migrations"
    ).fetchall() == before_history


def test_migrate_rejects_malformed_sql_filename_before_any_schema_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_first.sql").write_text(
        "CREATE TABLE first_version (id TEXT PRIMARY KEY);", encoding="utf-8"
    )
    monkeypatch.setattr(db, "MIGRATIONS_DIRECTORY", migrations)
    connection = connect(tmp_path / "state.sqlite3")
    assert migrate(connection) == 1
    before_history = connection.execute(
        "SELECT migration_id, applied_at, sha256 FROM schema_migrations"
    ).fetchall()
    before_tables = _table_names(connection)
    (migrations / "0002_second.sql").write_text(
        "CREATE TABLE second_version (id TEXT PRIMARY KEY);", encoding="utf-8"
    )
    (migrations / "unversioned.sql").write_text(
        "CREATE TABLE ignored_version (id TEXT PRIMARY KEY);", encoding="utf-8"
    )

    with pytest.raises(MigrationError, match="malformed migration filename unversioned.sql"):
        migrate(connection)

    assert _table_names(connection) == before_tables
    assert "second_version" not in _table_names(connection)
    assert connection.execute(
        "SELECT migration_id, applied_at, sha256 FROM schema_migrations"
    ).fetchall() == before_history


def test_migrate_rejects_an_active_caller_transaction_without_changing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_first.sql").write_text(
        "CREATE TABLE migration_table (id TEXT PRIMARY KEY);", encoding="utf-8"
    )
    monkeypatch.setattr(db, "MIGRATIONS_DIRECTORY", migrations)
    connection = connect(tmp_path / "state.sqlite3")
    assert migrate(connection) == 1
    connection.execute("BEGIN")
    connection.execute("CREATE TABLE caller_table (id TEXT PRIMARY KEY)")
    connection.execute("INSERT INTO caller_table (id) VALUES ('caller-work')")
    before_history = connection.execute(
        "SELECT migration_id, applied_at, sha256 FROM schema_migrations"
    ).fetchall()
    before_tables = _table_names(connection)

    with pytest.raises(MigrationError, match="active transaction"):
        migrate(connection)

    assert connection.in_transaction
    assert _table_names(connection) == before_tables
    assert connection.execute("SELECT id FROM caller_table").fetchall() == [("caller-work",)]
    assert connection.execute(
        "SELECT migration_id, applied_at, sha256 FROM schema_migrations"
    ).fetchall() == before_history
    connection.rollback()


def test_duplicate_migration_numbers_are_rejected_before_any_schema_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_a.sql").write_text(
        "CREATE TABLE first_version (id TEXT PRIMARY KEY);", encoding="utf-8"
    )
    (migrations / "0001_b.sql").write_text(
        "CREATE TABLE conflicting_version (id TEXT PRIMARY KEY);", encoding="utf-8"
    )
    monkeypatch.setattr(db, "MIGRATIONS_DIRECTORY", migrations)
    connection = connect(tmp_path / "state.sqlite3")

    with pytest.raises(MigrationError, match="duplicate migration number 0001"):
        migrate(connection)

    assert _table_names(connection) == set()


def test_migration_sequence_gaps_are_rejected_before_any_schema_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_first.sql").write_text(
        "CREATE TABLE first_version (id TEXT PRIMARY KEY);", encoding="utf-8"
    )
    (migrations / "0003_skipped.sql").write_text(
        "CREATE TABLE skipped_version (id TEXT PRIMARY KEY);", encoding="utf-8"
    )
    monkeypatch.setattr(db, "MIGRATIONS_DIRECTORY", migrations)
    connection = connect(tmp_path / "state.sqlite3")

    with pytest.raises(MigrationError, match="must be contiguous starting at 0001"):
        migrate(connection)

    assert _table_names(connection) == set()


def test_migrate_executes_multiple_complete_statements_on_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_two_tables.sql").write_text(
        "CREATE TABLE first_table (id TEXT PRIMARY KEY); CREATE TABLE second_table (id TEXT PRIMARY KEY);",
        encoding="utf-8",
    )
    monkeypatch.setattr(db, "MIGRATIONS_DIRECTORY", migrations)
    connection = connect(tmp_path / "state.sqlite3")

    assert migrate(connection) == 1
    assert _table_names(connection) == {"schema_migrations", "first_table", "second_table"}


def test_migrate_preserves_trigger_statement_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migrations = tmp_path / "migrations"
    migrations.mkdir()
    (migrations / "0001_trigger.sql").write_text(
        "CREATE TABLE source (id INTEGER PRIMARY KEY); CREATE TABLE audit (value INTEGER); "
        "CREATE TRIGGER source_audit AFTER INSERT ON source BEGIN "
        "INSERT INTO audit (value) VALUES (NEW.id); INSERT INTO audit (value) VALUES (NEW.id + 1); END;",
        encoding="utf-8",
    )
    monkeypatch.setattr(db, "MIGRATIONS_DIRECTORY", migrations)
    connection = connect(tmp_path / "state.sqlite3")

    assert migrate(connection) == 1
    connection.execute("INSERT INTO source (id) VALUES (7)")
    assert connection.execute("SELECT value FROM audit ORDER BY value").fetchall() == [(7,), (8,)]


def test_current_database_is_unchanged_on_repeat_migration(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    assert migrate(connection) == 2
    before = connection.execute("SELECT migration_id, applied_at, sha256 FROM schema_migrations").fetchall()

    assert migrate(connection) == 2
    assert connection.execute("SELECT migration_id, applied_at, sha256 FROM schema_migrations").fetchall() == before


def test_migrate_upgrades_an_actual_0001_database_without_changing_0001_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    migration_tree = tmp_path / "migrations"
    migration_tree.mkdir()
    source_tree = Path(__file__).parents[2] / "multichannel" / "migrations"
    first = migration_tree / "0001_foundation.sql"
    first.write_bytes((source_tree / "0001_foundation.sql").read_bytes())
    monkeypatch.setattr(db, "MIGRATIONS_DIRECTORY", migration_tree)
    connection = connect(tmp_path / "state.sqlite3")
    assert migrate(connection) == 1
    before = connection.execute(
        "SELECT migration_id, applied_at, sha256 FROM schema_migrations WHERE migration_id = '0001'"
    ).fetchone()
    (migration_tree / "0002_github.sql").write_bytes((source_tree / "0002_github.sql").read_bytes())

    assert migrate(connection) == 2
    assert connection.execute(
        "SELECT migration_id, applied_at, sha256 FROM schema_migrations WHERE migration_id = '0001'"
    ).fetchone() == before
    assert "source_observations" in _table_names(connection)


def test_github_schema_defers_reddit_fks_and_keeps_github_parent_identity_agreement(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    schema = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'source_observations'"
    ).fetchone()[0]
    repository_schema = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'github_repositories'"
    ).fetchone()[0]

    foreign_keys = {
        (row[3], row[2])
        for row in connection.execute("PRAGMA foreign_key_list(source_observations)")
    }
    assert ("github_repository_id", "github_repositories") in foreign_keys
    assert ("github_release_id", "github_releases") in foreign_keys
    assert "reddit_post_id TEXT" in schema
    assert "reddit_comment_id TEXT" in schema
    assert "reddit_post_id TEXT REFERENCES" not in schema
    assert "reddit_comment_id TEXT REFERENCES" not in schema
    assert "source_identity='github_repository:' || github_repository_id" in schema
    assert "source_identity='github_release:' || github_release_id" in schema
    assert "source_identity='reddit_post:' || reddit_post_id" in schema
    assert "source_identity='reddit_comment:' || reddit_comment_id" in schema
    assert "length(readme_sha256)=64" in repository_schema

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """INSERT INTO source_observations
               (id, source_kind, source_identity, observed_at, github_repository_id, metrics_json,
                raw_sha256, raw_path, created_at)
               VALUES ('unknown-repository', 'github_repository',
                       'github_repository:missing', '2026-08-21T00:00:00Z', 'missing', '{}', ?,
                       'x', '2026-08-21T00:00:00Z')""",
            ("a" * 64,),
        )


@pytest.mark.parametrize("invalid_id", ["github-id", "+42", " 42", "42 ", "04"])
def test_github_schema_rejects_noncanonical_numeric_ids(tmp_path: Path, invalid_id: str) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    connection.execute(
        """INSERT INTO github_repositories
           (id, github_id, owner, name, canonical_url, api_url, default_branch, topics_json, created_at)
           VALUES ('parent', '42', 'acme', 'widgets', 'https://github.com/acme/widgets',
                   'https://api.github.com/repos/acme/widgets', 'main', '[]', '2026-08-21T00:00:00Z')"""
    )

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """INSERT INTO github_repositories
               (id, github_id, owner, name, canonical_url, api_url, default_branch, topics_json, created_at)
               VALUES ('invalid-repository', ?, 'acme', 'invalid-repository',
                       'https://github.com/acme/invalid-repository',
                       'https://api.github.com/repos/acme/invalid-repository', 'main', '[]',
                       '2026-08-21T00:00:00Z')""",
            (invalid_id,),
        )
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """INSERT INTO github_releases
               (id, repository_id, github_release_id, tag_name, html_url, created_at)
               VALUES ('invalid-release', 'parent', ?, 'v1',
                       'https://github.com/acme/widgets/releases/tag/v1', '2026-08-21T00:00:00Z')""",
            (invalid_id,),
        )

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """INSERT INTO github_repositories
               (id, github_id, owner, name, canonical_url, api_url, default_branch, topics_json,
                readme_sha256, created_at)
               VALUES ('bad-hash', '1', 'acme', 'bad', 'https://github.com/acme/bad',
                       'https://api.github.com/repos/acme/bad', 'main', '[]', 'ABC',
                       '2026-08-21T00:00:00Z')"""
        )


def test_factories_generate_uuid_ids_and_utc_z_timestamps() -> None:
    channels = (factories.channel(), factories.channel())
    accounts = tuple(factories.platform_account(channel.id) for channel in channels)
    runs = (factories.job_run(), factories.job_run())
    events = tuple(factories.job_event(run.id) for run in runs)
    requests = tuple(factories.requeue_request(run.id) for run in runs)
    values = (*channels, *accounts, *runs, *events, *requests)

    ids = [value.id for value in values]
    assert len(set(ids)) == len(ids)
    for value_id in ids:
        assert str(UUID(value_id)) == value_id
    timestamps = [value.created_at for value in values] + [run.updated_at for run in runs]
    for timestamp in timestamps:
        assert timestamp.endswith("Z")
        assert datetime.fromisoformat(timestamp.replace("Z", "+00:00")).tzinfo is not None


def test_repeated_factories_insert_schema_unique_defaults(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    channels = (factories.channel(), factories.channel())
    runs = (factories.job_run(), factories.job_run())
    requests = tuple(factories.requeue_request(run.id) for run in runs)

    connection.executemany(
        """
        INSERT INTO channels (id, slug, name, locale, enabled, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (channel.id, channel.slug, channel.name, channel.locale, channel.enabled, channel.created_at)
            for channel in channels
        ],
    )
    connection.executemany(
        """
        INSERT INTO job_runs (
          id, job_type, subject_type, subject_id, state, attempt_count, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                run.id,
                run.job_type,
                run.subject_type,
                run.subject_id,
                run.state,
                run.attempt_count,
                run.created_at,
                run.updated_at,
            )
            for run in runs
        ],
    )
    connection.executemany(
        """
        INSERT INTO requeue_requests (id, job_run_id, request_key, operator, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                request.id,
                request.job_run_id,
                request.request_key,
                request.operator,
                request.reason,
                request.created_at,
            )
            for request in requests
        ],
    )
    connection.commit()

    assert connection.execute("SELECT COUNT(*) FROM channels").fetchone() == (2,)
    assert connection.execute("SELECT COUNT(*) FROM requeue_requests").fetchone() == (2,)


def test_factory_child_rows_enforce_foreign_keys(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    account = factories.platform_account("missing-channel")

    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            """
            INSERT INTO platform_accounts (
              id, channel_id, platform, capability_state, media_capability, enabled, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                account.id,
                account.channel_id,
                account.platform,
                account.capability_state,
                account.media_capability,
                account.enabled,
                account.created_at,
            ),
        )
