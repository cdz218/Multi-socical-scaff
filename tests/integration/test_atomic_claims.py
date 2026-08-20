from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from multichannel.db import connect, migrate
from multichannel.jobs import ClaimedJob, InvalidTransition, claim_job, transition_job


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
    outcomes: list[ClaimedJob | None] = []
    errors: list[BaseException] = []

    def worker(worker_id: str) -> None:
        connection: sqlite3.Connection | None = None
        try:
            connection = connect(path)
            barrier.wait()
            outcomes.append(claim_job(connection, "job-1", worker_id, NOW))
        except BaseException as error:
            errors.append(error)
        finally:
            if connection is not None:
                connection.close()

    threads = [threading.Thread(target=worker, args=(f"worker-{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(outcomes) == 2
    assert not errors
    assert sum(isinstance(outcome, ClaimedJob) for outcome in outcomes) == 1
    assert outcomes.count(None) == 1
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
    claim = claim_job(connection, "job-1", "worker", NOW)
    assert claim is not None
    transition_job(connection, "job-1", "claimed", "running", "worker", claim_token=claim.claim_token, now=NOW)
    assert connection.execute("SELECT state FROM job_runs").fetchone() == ("running",)


def test_stale_claim_token_cannot_transition_reclaimed_job(tmp_path: Path) -> None:
    connection = connect(tmp_path / "stale-token.sqlite3")
    migrate(connection)
    connection.execute(
        "INSERT INTO job_runs (id, job_type, subject_type, subject_id, state, created_at, updated_at) VALUES ('job-1','collect','channel','s','queued',?,?)",
        (NOW, NOW),
    )
    connection.commit()
    first_claim = claim_job(connection, "job-1", "worker", NOW)
    assert first_claim is not None
    transition_job(
        connection,
        "job-1",
        "claimed",
        "queued",
        "worker",
        claim_token=first_claim.claim_token,
        now=NOW,
    )
    second_claim = claim_job(connection, "job-1", "worker", NOW)
    assert second_claim is not None
    assert second_claim.claim_token != first_claim.claim_token

    with pytest.raises(InvalidTransition, match="claim token"):
        transition_job(
            connection,
            "job-1",
            "claimed",
            "running",
            "worker",
            claim_token=first_claim.claim_token,
            now=NOW,
        )

    transition_job(
        connection,
        "job-1",
        "claimed",
        "running",
        "worker",
        claim_token=second_claim.claim_token,
        now=NOW,
    )
    assert connection.execute("SELECT state FROM job_runs WHERE id='job-1'").fetchone() == ("running",)


def test_concurrent_transitions_return_application_error_not_sqlite_lock(
    tmp_path: Path,
) -> None:
    path = tmp_path / "transition-race.sqlite3"
    _setup(path)
    connection = connect(path)
    claim = claim_job(connection, "job-1", "worker", NOW)
    assert claim is not None
    connection.close()
    read_barrier = threading.Barrier(2)
    outcomes: list[str] = []
    errors: list[BaseException] = []

    def worker(target_state: str) -> None:
        worker_connection = connect(path)
        trace_state = threading.local()
        trace_state.deferred_begin = False

        def synchronize_deferred_read(statement: str) -> None:
            if statement == "BEGIN":
                trace_state.deferred_begin = True
            elif (
                trace_state.deferred_begin
                and statement.startswith("SELECT state, worker_id, claim_token FROM job_runs")
            ):
                read_barrier.wait(timeout=5)

        worker_connection.set_trace_callback(
            synchronize_deferred_read
        )
        try:
            transition_job(worker_connection, "job-1", "claimed", target_state, "worker", claim_token=claim.claim_token, now=NOW)
            outcomes.append(target_state)
        except BaseException as error:
            errors.append(error)
        finally:
            worker_connection.close()

    threads = [
        threading.Thread(target=worker, args=("running",)),
        threading.Thread(target=worker, args=("cancelled",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(outcomes) == 1
    assert len(errors) == 1
    assert isinstance(errors[0], InvalidTransition)
    assert not isinstance(errors[0], sqlite3.OperationalError)
