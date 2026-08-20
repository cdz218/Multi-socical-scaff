from __future__ import annotations

from pathlib import Path

from multichannel.db import connect, migrate


def test_connect_enables_foreign_keys_wal_and_busy_timeout(tmp_path: Path) -> None:
    connection = connect(tmp_path / "state.sqlite3")

    assert connection.execute("PRAGMA foreign_keys").fetchone() == (1,)
    assert connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    assert connection.execute("PRAGMA busy_timeout").fetchone() == (5000,)


def test_empty_database_upgrades_through_only_current_foundation_migration(tmp_path: Path) -> None:
    """Later migrations are intentionally owned by later tasks and do not exist yet."""
    connection = connect(tmp_path / "state.sqlite3")

    assert migrate(connection) == 1
    assert connection.execute("SELECT migration_id FROM schema_migrations").fetchall() == [
        ("0001",)
    ]
