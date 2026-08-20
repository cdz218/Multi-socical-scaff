from __future__ import annotations

import sqlite3
from pathlib import Path

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


def test_migrate_is_idempotent_and_creates_only_foundation_tables(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")

    assert migrate(connection) == 1
    assert migrate(connection) == 1
    assert _table_names(connection) >= {
        "schema_migrations",
        "channels",
        "platform_accounts",
        "job_runs",
        "job_events",
        "requeue_requests",
    }
    assert not _table_names(connection).intersection({"concepts", "publishers", "media"})
    assert connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone() == (1,)


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


def test_existing_0001_database_is_unchanged_on_repeat_migration(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")
    assert migrate(connection) == 1
    before = connection.execute("SELECT migration_id, applied_at, sha256 FROM schema_migrations").fetchall()

    assert migrate(connection) == 1
    assert connection.execute("SELECT migration_id, applied_at, sha256 FROM schema_migrations").fetchall() == before


def test_factories_generate_uuid_ids_and_utc_z_timestamps() -> None:
    channel = factories.channel()
    account = factories.platform_account(channel.id)
    run = factories.job_run()
    event = factories.job_event(run.id)
    request = factories.requeue_request(run.id)

    values = (channel, account, run, event, request)
    assert all(value.id != "1" for value in values)
    assert len({value.id for value in values}) == len(values)
    assert all(value.created_at.endswith("Z") for value in values)


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
