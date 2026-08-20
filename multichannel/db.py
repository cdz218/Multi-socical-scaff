"""SQLite connection and versioned migration engine."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

MIGRATIONS_DIRECTORY = Path(__file__).parent / "migrations"
_MIGRATION_NAME = re.compile(r"^(\d+)_.*\.sql$")
_LEADING_SQL_COMMENTS = re.compile(
    r"^(?:\s*(?:(?:--[^\n]*(?:\n|$))|(?:/\*.*?\*/)))*\s*", re.DOTALL
)
_TRANSACTION_CONTROL = re.compile(
    r"(?:BEGIN|COMMIT|ROLLBACK|SAVEPOINT|RELEASE|END)(?:\s|;|$)", re.IGNORECASE
)


class MigrationError(RuntimeError):
    """Raised when migration history is inconsistent."""


def connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def _migration_files() -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for path in MIGRATIONS_DIRECTORY.glob("*.sql"):
        match = _MIGRATION_NAME.match(path.name)
        if not match:
            raise MigrationError(f"malformed migration filename {path.name}")
        files.append((int(match.group(1)), path))
    return sorted(files)


def _validate_migration_sequence(migrations: list[tuple[int, Path]]) -> None:
    migration_numbers = [number for number, _ in migrations]
    duplicate_numbers = {
        number for number in migration_numbers if migration_numbers.count(number) > 1
    }
    if duplicate_numbers:
        duplicate = min(duplicate_numbers)
        raise MigrationError(f"duplicate migration number {duplicate:04d}")
    expected_numbers = list(range(1, len(migration_numbers) + 1))
    if migration_numbers != expected_numbers:
        raise MigrationError("migration numbers must be contiguous starting at 0001")


def _ensure_history(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
          migration_id TEXT PRIMARY KEY,
          applied_at TEXT NOT NULL,
          sha256 TEXT NOT NULL UNIQUE
            CHECK(length(sha256)=64 AND sha256 NOT GLOB '*[^0-9a-f]*')
        )
        """
    )


def _has_history(connection: sqlite3.Connection) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'"
        ).fetchone()
        is not None
    )


def _statements(sql: str) -> list[str]:
    statements: list[str] = []
    statement = ""
    for character in sql:
        statement += character
        if sqlite3.complete_statement(statement):
            if statement.strip():
                statements.append(statement)
            statement = ""
    if statement.strip():
        statements.append(statement)
    return statements


def _validate_migration_sql(path: Path, sql: str) -> None:
    for statement in _statements(sql):
        sql_without_comments = _LEADING_SQL_COMMENTS.sub("", statement)
        if _TRANSACTION_CONTROL.match(sql_without_comments):
            raise MigrationError(f"migration {path.name} contains transaction-control SQL")


def _execute_statements(connection: sqlite3.Connection, sql: str) -> None:
    for statement in _statements(sql):
        connection.execute(statement)


def _creates_history(sql: str) -> bool:
    return bool(re.search(r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?schema_migrations\b", sql, re.I))


def _validate_history(
    connection: sqlite3.Connection, migrations: list[tuple[int, Path]]
) -> dict[str, str]:
    """Validate recorded history before applying pending migrations."""
    rows = connection.execute(
        "SELECT migration_id, sha256 FROM schema_migrations ORDER BY migration_id"
    ).fetchall()
    migration_ids = [f"{number:04d}" for number, _ in migrations]
    recorded_ids: list[str] = []
    checksums: dict[str, str] = {}
    for migration_id, checksum in rows:
        if not isinstance(migration_id, str) or not re.fullmatch(r"\d{4}", migration_id):
            raise MigrationError(f"migration history contains malformed migration ID {migration_id!r}")
        recorded_ids.append(migration_id)
        checksums[migration_id] = checksum
    if recorded_ids != migration_ids[: len(recorded_ids)]:
        raise MigrationError("migration history must be a prefix of discovered migration IDs")
    for migration_number, path in migrations[: len(recorded_ids)]:
        migration_id = f"{migration_number:04d}"
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksums[migration_id] != checksum:
            raise MigrationError(f"migration {migration_id} checksum differs")
    return checksums


def migrate(connection: sqlite3.Connection) -> int:
    if connection.in_transaction:
        raise MigrationError("cannot migrate with an active transaction")
    migrations = _migration_files()
    _validate_migration_sequence(migrations)
    migration_sql = {path: path.read_bytes().decode("utf-8") for _, path in migrations}
    for _, path in migrations:
        _validate_migration_sql(path, migration_sql[path])
    history_exists = _has_history(connection)
    recorded_checksums = _validate_history(connection, migrations) if history_exists else {}
    for migration_number, path in migrations:
        migration_id = f"{migration_number:04d}"
        sql = path.read_bytes()
        checksum = hashlib.sha256(sql).hexdigest()
        if migration_id in recorded_checksums:
            continue
        try:
            connection.execute("BEGIN IMMEDIATE")
            statement_sql = migration_sql[path]
            if not history_exists and not _creates_history(statement_sql):
                _ensure_history(connection)
            _execute_statements(connection, statement_sql)
            connection.execute(
                "INSERT INTO schema_migrations (migration_id, applied_at, sha256) VALUES (?, ?, ?)",
                (migration_id, datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), checksum),
            )
            connection.commit()
            history_exists = True
        except Exception:
            connection.rollback()
            raise
    if not history_exists:
        _ensure_history(connection)
    count = connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()
    if count is None:
        raise MigrationError("migration history count was unavailable")
    return int(count[0])
