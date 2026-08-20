from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from multichannel.db import connect, migrate


_FOUNDATION_TABLES = {
    "schema_migrations",
    "channels",
    "platform_accounts",
    "job_runs",
    "job_events",
    "requeue_requests",
}


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


def test_concurrent_connections_migrate_an_empty_database_once(tmp_path: Path) -> None:
    for iteration in range(10):
        path = tmp_path / f"state-{iteration}.sqlite3"
        start = Barrier(2)

        def migrate_from_new_connection() -> int:
            start.wait()
            connection = connect(path)
            try:
                return migrate(connection)
            finally:
                connection.close()

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: migrate_from_new_connection(), range(2)))

        connection = connect(path)
        try:
            tables = {
                row[0]
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
            }
            assert results == [1, 1]
            assert tables == _FOUNDATION_TABLES
            assert connection.execute("SELECT migration_id FROM schema_migrations").fetchall() == [
                ("0001",)
            ]
        finally:
            connection.close()
