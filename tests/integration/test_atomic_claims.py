from __future__ import annotations

import threading
from pathlib import Path

import pytest

from multichannel.db import connect, migrate
from multichannel.jobs import claim_job, transition_job


NOW = "2026-08-21T00:00:00Z"


def _setup(path: Path) -> None:
    connection = connect(path)
    migrate(connection)
    connection.execute(
        "INSERT INTO job_runs (id, job_type, subject_type, subject_id, state, created_at, updated_at) VALUES ('job-1','collect','channel','s','queued',?,?)",
        (NOW, NOW),
    )
    connection.commit()
    connection.close()


def test_two_independent_connections_only_one_claims(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.sqlite3"
    _setup(path)
    barrier = threading.Barrier(2)
    results: list[object] = []

    def worker(worker_id: str) -> None:
        connection = connect(path)
        barrier.wait()
        try:
            results.append(claim_job(connection, "job-1", worker_id, NOW))
        finally:
            connection.close()

    threads = [threading.Thread(target=worker, args=(f"worker-{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sum(result is not None for result in results) == 1
    connection = connect(path)
    assert connection.execute("SELECT attempt_count, state FROM job_runs").fetchone() == (1, "claimed")
    assert connection.execute("SELECT COUNT(*) FROM job_events").fetchone() == (1,)


def test_active_caller_transaction_is_not_absorbed(tmp_path: Path) -> None:
    path = tmp_path / "transaction.sqlite3"
    _setup(path)
    connection = connect(path)
    connection.execute("CREATE TABLE caller_work (id INTEGER PRIMARY KEY)")
    connection.execute("INSERT INTO caller_work VALUES (1)")
    with pytest.raises(RuntimeError, match="active transaction"):
        claim_job(connection, "job-1", "worker", NOW)
    assert connection.in_transaction
    assert connection.execute("SELECT COUNT(*) FROM caller_work").fetchone() == (1,)
    assert connection.execute("SELECT state, attempt_count FROM job_runs").fetchone() == ("queued", 0)
    connection.rollback()


def test_claim_token_and_attempt_are_atomic(tmp_path: Path) -> None:
    path = tmp_path / "atomic.sqlite3"
    _setup(path)
    connection = connect(path)
    claimed = claim_job(connection, "job-1", "worker", NOW)
    assert claimed is not None
    assert claimed.attempt_count == 1
    row = connection.execute("SELECT claim_token, worker_id, claimed_at FROM job_runs").fetchone()
    assert row == (claimed.claim_token, "worker", NOW)


def test_claimed_job_can_transition_to_running(tmp_path: Path) -> None:
    connection = connect(tmp_path / "running.sqlite3")
    migrate(connection)
    connection.execute(
        "INSERT INTO job_runs (id, job_type, subject_type, subject_id, state, created_at, updated_at) VALUES ('job-1','collect','channel','s','queued',?,?)",
        (NOW, NOW),
    )
    connection.commit()
    assert claim_job(connection, "job-1", "worker", NOW)
    transition_job(connection, "job-1", "claimed", "running", "worker", now=NOW)
    assert connection.execute("SELECT state FROM job_runs").fetchone() == ("running",)
