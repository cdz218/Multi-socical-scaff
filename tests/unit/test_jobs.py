from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from multichannel.db import connect, migrate
from multichannel.jobs import (
    InvalidTransition,
    JobNotFound,
    KillSwitchEnabled,
    ReconciliationRequired,
    classify_failure,
    claim_job,
    record_event,
    requeue_job,
    set_kill_switch,
    transition_job,
)
from multichannel.jobs import _is_opaque_credential, _sanitize_identity


NOW = "2026-08-21T00:00:00Z"
SECRET = "sentinel-secret-token"
OPAQUE_CREDENTIAL = "Q7vN2xR9mK4pL8sT1wY6cD3fH5jU0bE2"
SENSITIVE_TEXTS = (
    "Basic dXNlcjpwYXNz",
    "Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature",
    "api_key=key-1234567890",
    "token: token-1234567890",
    "https://user:pass@example.test/path",
    "https://example.test/path?api_key=key-1234567890&view=summary",
    "ghp_abcdefghijklmnopqrstuvwxyz1234567890",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjMifQ",
)


def _connection(tmp_path: Path) -> sqlite3.Connection:
    connection = connect(tmp_path / "jobs.sqlite3")
    migrate(connection)
    return connection


def _insert_job(connection: sqlite3.Connection, state: str = "queued", job_id: str = "job-1") -> None:
    connection.execute(
        "INSERT INTO job_runs (id, job_type, subject_type, subject_id, state, created_at, updated_at) "
        "VALUES (?, 'collect', 'channel', 'subject', ?, ?, ?)",
        (job_id, state, NOW, NOW),
    )
    connection.commit()


@pytest.fixture(autouse=True)
def reset_kill_switch() -> None:
    set_kill_switch(False)
    yield
    set_kill_switch(False)


def test_exact_illegal_transition_copy_paste_case(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection)
    with pytest.raises(InvalidTransition):
        transition_job(connection, "job-1", "queued", "running", "w")


def test_generic_queued_to_claimed_transition_is_rejected_without_mutation(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection)

    with pytest.raises(InvalidTransition):
        transition_job(connection, "job-1", "queued", "claimed", "worker", now=NOW)

    assert connection.execute(
        "SELECT state, worker_id, claim_token, claimed_at, attempt_count FROM job_runs"
    ).fetchone() == ("queued", None, None, None, 0)
    assert connection.execute("SELECT COUNT(*) FROM job_events").fetchone() == (0,)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("queued", "cancelled"),
        ("claimed", "running"), ("claimed", "queued"), ("claimed", "cancelled"),
        ("running", "succeeded"), ("running", "failed"), ("running", "deferred"),
        ("running", "ambiguous"), ("running", "cancelled"),
        ("failed", "cancelled"),
        ("deferred", "cancelled"),
        ("ambiguous", "cancelled"),
    ],
)
def test_legal_transitions_emit_event(tmp_path: Path, old: str, new: str) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, old)
    transition_job(connection, "job-1", old, new, "worker", now=NOW)
    assert connection.execute("SELECT state FROM job_runs WHERE id='job-1'").fetchone() == (new,)
    assert connection.execute("SELECT from_state, to_state FROM job_events").fetchall() == [(old, new)]


@pytest.mark.parametrize("old", ["succeeded", "cancelled"])
def test_terminal_states_reject_transitions(tmp_path: Path, old: str) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, old)
    with pytest.raises(InvalidTransition):
        transition_job(connection, "job-1", old, "queued", "worker")


def test_expected_state_mismatch_does_not_mutate(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "running")
    with pytest.raises(InvalidTransition):
        transition_job(connection, "job-1", "queued", "failed", "worker")
    assert connection.execute("SELECT state FROM job_runs").fetchone() == ("running",)
    assert connection.execute("SELECT COUNT(*) FROM job_events").fetchone() == (0,)


def test_missing_job_behaviour(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    with pytest.raises(JobNotFound):
        transition_job(connection, "missing", "queued", "cancelled", "worker")
    assert claim_job(connection, "missing", "worker", NOW) is None
    with pytest.raises(JobNotFound):
        record_event(connection, "missing", "x", None, None, "worker", {})


def test_partial_failure_is_sanitized_and_canonical(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "running")
    error = RuntimeError(f"token={SECRET} https://user:pass@example.test/x?api_key={SECRET}\ntraceback")
    transition_job(
        connection, "job-1", "running", "failed", "worker", now=NOW, error=error,
        partial_state={"Password": SECRET, "stage": "upload", "nested": {"token": SECRET}},
    )
    row = connection.execute(
        "SELECT error_class, error_detail, partial_state_json FROM job_runs"
    ).fetchone()
    assert row[0] == "RuntimeError"
    assert SECRET not in row[1] and "traceback" not in row[1] and "user:pass" not in row[1]
    assert json.loads(row[2]) == {"nested": {"token": "[REDACTED]"}, "Password": "[REDACTED]", "stage": "upload"}
    detail = connection.execute("SELECT detail_json FROM job_events").fetchone()[0]
    assert SECRET not in detail
    assert "traceback" not in detail


def test_credential_shaped_values_are_sanitized_in_every_free_text_sink(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "queued")

    claim = claim_job(connection, "job-1", OPAQUE_CREDENTIAL, NOW)
    assert claim is not None
    transition_job(
        connection,
        "job-1",
        "claimed",
        "running",
        OPAQUE_CREDENTIAL,
        now=NOW,
    )
    error = RuntimeError(f"remote failure: {OPAQUE_CREDENTIAL}")
    transition_job(
        connection,
        "job-1",
        "running",
        "failed",
        OPAQUE_CREDENTIAL,
        now=NOW,
        error=error,
        partial_state={"stage": OPAQUE_CREDENTIAL},
    )
    record_event(
        connection,
        "job-1",
        "diagnostic",
        "failed",
        "failed",
        OPAQUE_CREDENTIAL,
        {"message": OPAQUE_CREDENTIAL},
        now=NOW,
    )
    assert requeue_job(
        connection,
        "job-1",
        OPAQUE_CREDENTIAL,
        OPAQUE_CREDENTIAL,
        "request-opaque-1",
        now=NOW,
    )

    persisted = "\n".join(
        str(value)
        for table in ("job_runs", "job_events", "requeue_requests")
        for row in connection.execute(f"SELECT * FROM {table}").fetchall()
        for value in row
    )
    assert OPAQUE_CREDENTIAL not in persisted


@pytest.mark.parametrize("secret", SENSITIVE_TEXTS)
def test_credential_patterns_are_redacted_from_error_and_event_detail(
    tmp_path: Path, secret: str
) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "running")
    transition_job(
        connection,
        "job-1",
        "running",
        "failed",
        "worker-01",
        now=NOW,
        error=RuntimeError(f"remote failure: {secret}"),
        partial_state={"stage": secret},
    )
    record_event(
        connection,
        "job-1",
        "diagnostic",
        "failed",
        "failed",
        "worker-01",
        {"message": secret},
        now=NOW,
    )
    persisted = "\n".join(
        str(value)
        for table in ("job_runs", "job_events")
        for row in connection.execute(f"SELECT * FROM {table}").fetchall()
        for value in row
    )
    assert secret not in persisted


def test_sanitizer_preserves_safe_identifiers_and_urls(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "queued")
    assert claim_job(connection, "job-1", "worker-01", NOW) is not None
    transition_job(connection, "job-1", "claimed", "running", "worker-01", now=NOW)
    record_event(
        connection,
        "job-1",
        "diagnostic",
        "running",
        "running",
        "worker-01",
        {"message": "retry upload", "url": "https://example.test/path?view=summary"},
        now=NOW,
    )
    row = connection.execute(
        "SELECT actor, detail_json FROM job_events WHERE detail_json LIKE '%retry upload%'"
    ).fetchone()
    assert row == (
        "worker-01",
        '{"message":"retry upload","url":"https://example.test/path?view=summary"}',
    )
    transition_job(connection, "job-1", "running", "failed", "worker-01", now=NOW)
    request_key = "123e4567-e89b-12d3-a456-426614174000"
    assert requeue_job(connection, "job-1", "operator-01", "retry upload", request_key, now=NOW)
    assert connection.execute(
        "SELECT request_key, operator, reason FROM requeue_requests"
    ).fetchone() == (request_key, "operator-01", "retry upload")


def test_json_rejects_nan_and_is_deterministic(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "running")
    with pytest.raises(ValueError):
        transition_job(connection, "job-1", "running", "failed", "worker", partial_state={"x": float("nan")})
    with pytest.raises(ValueError):
        record_event(connection, "job-1", "test", "running", "running", "worker", {"x": float("nan")})
    record_event(connection, "job-1", "test", "running", "running", "worker", {"b": 2, "a": 1}, now=NOW)
    assert connection.execute("SELECT detail_json FROM job_events").fetchone()[0] == '{"a":1,"b":2}'


def test_only_claim_owner_can_transition_claimed_or_running_job(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection)
    assert claim_job(connection, "job-1", "worker-a", NOW) is not None
    before = (
        connection.execute("SELECT state FROM job_runs WHERE id='job-1'").fetchone(),
        connection.execute("SELECT COUNT(*) FROM job_events").fetchone(),
    )

    with pytest.raises(InvalidTransition, match="owner"):
        transition_job(connection, "job-1", "claimed", "running", "intruder", now=NOW)

    assert (
        connection.execute("SELECT state FROM job_runs WHERE id='job-1'").fetchone(),
        connection.execute("SELECT COUNT(*) FROM job_events").fetchone(),
    ) == before
    transition_job(connection, "job-1", "claimed", "running", "worker-a", now=NOW)
    with pytest.raises(InvalidTransition, match="owner"):
        transition_job(connection, "job-1", "running", "failed", "intruder", now=NOW)
    transition_job(connection, "job-1", "running", "failed", "worker-a", now=NOW)


def test_standalone_event_cannot_forge_state_history_or_leak_structural_text(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "running")
    before = _counts(connection)

    with pytest.raises(InvalidTransition, match="durable state"):
        record_event(
            connection, "job-1", "diagnostic", "failed", "succeeded", "worker", {}, now=NOW
        )

    assert _counts(connection) == before
    record_event(
        connection,
        "job-1",
        "passwd=round-five-structural-secret",
        "running",
        "running",
        "worker",
        {},
        now=NOW,
    )
    event_type, from_state, to_state = connection.execute(
        "SELECT event_type, from_state, to_state FROM job_events"
    ).fetchone()
    assert "round-five-structural-secret" not in event_type
    assert (from_state, to_state) == ("running", "running")


def test_secret_aliases_and_private_key_pem_are_redacted_recursively(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "running")
    pem = "-----BEGIN PRIVATE KEY-----\nround-five-private-key\n-----END PRIVATE KEY-----"
    aliases = {
        "passwd": "round-five-passwd-secret",
        "private_key": pem,
        "client_secret": "round-five-client-secret",
        "access_key": "round-five-access-key",
        "nested": {"pass%77d": "round-five-encoded-passwd"},
        "message": pem,
    }
    transition_job(
        connection, "job-1", "running", "failed", "worker", partial_state=aliases, now=NOW
    )
    record_event(connection, "job-1", "diagnostic", "failed", "failed", "worker", aliases, now=NOW)
    persisted = "\n".join(
        str(value)
        for table in ("job_runs", "job_events")
        for row in connection.execute(f"SELECT * FROM {table}").fetchall()
        for value in row
    )
    for secret in (
        "round-five-passwd-secret",
        "round-five-private-key",
        "round-five-client-secret",
        "round-five-access-key",
        "round-five-encoded-passwd",
        "-----BEGIN PRIVATE KEY-----",
    ):
        assert secret not in persisted


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("key", ["token", "ordinary"])
def test_nonfinite_json_values_reject_before_redaction_or_mutation(
    tmp_path: Path, key: str, value: float
) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "running")
    before = _counts(connection)

    with pytest.raises(ValueError):
        transition_job(
            connection, "job-1", "running", "failed", "worker", partial_state={key: value}, now=NOW
        )
    with pytest.raises(ValueError):
        record_event(connection, "job-1", "diagnostic", "running", "running", "worker", {key: value}, now=NOW)

    assert _counts(connection) == before
    assert connection.execute("SELECT state FROM job_runs").fetchone() == ("running",)


def test_sensitive_requeue_reason_is_private_but_remains_idempotency_distinct(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "failed")
    first_reason = "token=round-five-first-sensitive-reason"
    second_reason = "token=round-five-second-sensitive-reason"

    assert requeue_job(connection, "job-1", "operator", first_reason, "request-1", now=NOW)
    counts = _counts(connection)
    assert not requeue_job(connection, "job-1", "operator", first_reason, "request-1", now=NOW)
    assert _counts(connection) == counts
    with pytest.raises(InvalidTransition, match="different requeue details"):
        requeue_job(connection, "job-1", "operator", second_reason, "request-1", now=NOW)

    stored_reason = connection.execute("SELECT reason FROM requeue_requests").fetchone()[0]
    assert stored_reason.startswith("reason-sha256:")
    assert first_reason not in stored_reason and second_reason not in stored_reason


@pytest.mark.parametrize(
    "url, secret",
    [
        ("https://example.test/access%5Ftoken%3Dround-five-path-secret", "round-five-path-secret"),
        ("https://example.test/path?access%5Ftoken=round-five-query-secret", "round-five-query-secret"),
        ("https://example.test/path#access%5Ftoken=round-five-fragment-secret", "round-five-fragment-secret"),
    ],
)
def test_percent_encoded_url_credentials_are_redacted_without_rewriting_safe_urls(
    tmp_path: Path, url: str, secret: str
) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "running")
    safe_url = "https://example.test/path%20space?view=summary%20today#tab%20one"
    transition_job(
        connection,
        "job-1",
        "running",
        "failed",
        "worker",
        error=RuntimeError(url),
        partial_state={"url": url},
        now=NOW,
    )
    record_event(
        connection,
        "job-1",
        "diagnostic",
        "failed",
        "failed",
        "worker",
        {"url": url, "safe_url": safe_url},
        now=NOW,
    )
    persisted = "\n".join(
        str(value)
        for table in ("job_runs", "job_events")
        for row in connection.execute(f"SELECT * FROM {table}").fetchall()
        for value in row
    )
    assert secret not in persisted
    assert "access%5Ftoken" not in persisted
    assert safe_url in persisted


def test_requeue_is_idempotent_and_conflicts_rejected(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "failed")
    assert requeue_job(connection, "job-1", "operator", "retry", "request-1", now=NOW)
    counts = [connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("job_runs", "job_events", "requeue_requests")]
    assert not requeue_job(connection, "job-1", "operator", "retry", "request-1", now=NOW)
    assert counts == [connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in ("job_runs", "job_events", "requeue_requests")]
    _insert_job(connection, "failed", "job-2")
    with pytest.raises(InvalidTransition):
        requeue_job(connection, "job-2", "other", "different", "request-1", now=NOW)


def test_credential_shaped_request_keys_are_distinct_and_idempotent(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "failed", "job-a")
    _insert_job(connection, "failed", "job-b")
    key_a = "Q7vN2xR9mK4pL8sT1wY6cD3fH5jU0bE2"
    key_b = "R8wO3yS0nL5qM9tU2xZ7dE4gF6kV1cH3"

    assert requeue_job(connection, "job-a", "operator-a", "retry", key_a, now=NOW)
    assert requeue_job(connection, "job-b", "operator-b", "retry", key_b, now=NOW)
    persisted_keys = [
        row[0]
        for row in connection.execute(
            "SELECT request_key FROM requeue_requests ORDER BY job_run_id"
        ).fetchall()
    ]
    assert len(set(persisted_keys)) == 2
    assert all(value.startswith("identity-sha256:") for value in persisted_keys)
    assert all(not _is_opaque_credential(value) for value in persisted_keys)
    persisted = "\n".join(
        str(value)
        for table in ("job_runs", "job_events", "requeue_requests")
        for row in connection.execute(f"SELECT * FROM {table}").fetchall()
        for value in row
    )
    assert key_a not in persisted
    assert key_b not in persisted

    assert not requeue_job(connection, "job-a", "operator-a", "retry", key_a, now=NOW)
    assert connection.execute(
        "SELECT request_key FROM requeue_requests WHERE job_run_id='job-a'"
    ).fetchone()[0] == persisted_keys[0]


def test_credential_identity_pseudonyms_are_stable_and_never_opaque(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "queued")
    credential = "Q7vN2xR9mK4pL8sT1wY6cD3fH5jU0bE2"

    claim = claim_job(connection, "job-1", credential, NOW)
    assert claim is not None
    assert claim.worker_id.startswith("identity-sha256:")
    assert not _is_opaque_credential(claim.worker_id)
    assert _sanitize_identity(credential) == claim.worker_id
    assert _sanitize_identity(claim.worker_id) != claim.worker_id
    transition_job(connection, "job-1", "claimed", "running", credential, now=NOW)
    transition_job(connection, "job-1", "running", "failed", credential, now=NOW)
    assert requeue_job(
        connection,
        "job-1",
        credential,
        "retry",
        "safe-identity-request",
        now=NOW,
    )
    actor_values = [row[0] for row in connection.execute("SELECT actor FROM job_events").fetchall()]
    assert all(value == claim.worker_id for value in actor_values)
    assert connection.execute("SELECT operator FROM requeue_requests").fetchone() == (claim.worker_id,)
    assert credential not in "\n".join(actor_values)
    assert all("Q7vN2xR9mK4pL8sT1wY6cD3fH5jU0bE2" not in str(row) for row in connection.execute("SELECT * FROM job_events"))


def test_reserved_identity_pseudonym_input_is_domain_hashed_and_raw_absent(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "failed", "job-raw")
    _insert_job(connection, "failed", "job-reserved")
    raw = OPAQUE_CREDENTIAL
    reserved = _sanitize_identity(raw)

    assert reserved.startswith("identity-sha256:")
    assert _sanitize_identity(reserved).startswith("identity-sha256:")
    assert _sanitize_identity(reserved) != reserved
    assert requeue_job(connection, "job-raw", "operator", "retry", raw, now=NOW)
    assert requeue_job(connection, "job-reserved", "operator", "retry", reserved, now=NOW)
    assert not requeue_job(connection, "job-reserved", "operator", "retry", reserved, now=NOW)
    record_event(connection, "job-raw", "identity-check", "queued", "queued", raw, {}, now=NOW)
    record_event(
        connection, "job-reserved", "identity-check", "queued", "queued", reserved, {}, now=NOW
    )
    keys = [
        row[0]
        for row in connection.execute(
            "SELECT request_key FROM requeue_requests ORDER BY job_run_id"
        ).fetchall()
    ]
    assert keys == [reserved, _sanitize_identity(reserved)]
    actors = [
        row[0]
        for row in connection.execute(
            "SELECT actor FROM job_events WHERE event_type='identity-check' ORDER BY job_run_id"
        ).fetchall()
    ]
    assert actors == [reserved, _sanitize_identity(reserved)]

    persisted = "\n".join(
        str(value)
        for table in ("job_runs", "job_events", "requeue_requests")
        for row in connection.execute(f"SELECT * FROM {table}").fetchall()
        for value in row
    )
    assert raw not in persisted


@pytest.mark.parametrize(
    "url",
    [
        "https://api.github.com/repos/example/ghp_abcdefghijklmnopqrstuvwxyz1234567890",
        f"https://example.test/path?view={OPAQUE_CREDENTIAL}",
        f"https://example.test/callback#access_token={OPAQUE_CREDENTIAL}",
    ],
)
def test_credentials_in_url_components_are_redacted(tmp_path: Path, url: str) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "running")
    transition_job(
        connection,
        "job-1",
        "running",
        "failed",
        "worker-01",
        now=NOW,
        error=RuntimeError(url),
        partial_state={"url": url},
    )
    record_event(connection, "job-1", "diagnostic", "failed", "failed", "worker-01", {"url": url}, now=NOW)

    persisted = "\n".join(
        str(value)
        for table in ("job_runs", "job_events")
        for row in connection.execute(f"SELECT * FROM {table}").fetchall()
        for value in row
    )
    assert url not in persisted
    assert "ghp_abcdefghijklmnopqrstuvwxyz1234567890" not in persisted
    assert OPAQUE_CREDENTIAL not in persisted


def test_ambiguous_requeue_requires_reconciliation(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "ambiguous")
    with pytest.raises(ReconciliationRequired):
        requeue_job(connection, "job-1", "operator", "retry", "request-1", now=NOW)
    assert requeue_job(connection, "job-1", "operator", "retry", "request-1", reconciled=True, now=NOW)


@pytest.mark.parametrize("state", ["failed", "deferred", "ambiguous"])
def test_direct_explicit_requeue_transitions_are_rejected_without_mutation(
    tmp_path: Path, state: str
) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, state)
    counts = _counts(connection)

    with pytest.raises(InvalidTransition, match="requeue_job"):
        transition_job(connection, "job-1", state, "queued", "worker", now=NOW)

    assert connection.execute("SELECT state FROM job_runs WHERE id='job-1'").fetchone() == (state,)
    assert _counts(connection) == counts


def test_claimed_job_can_still_recover_to_queued(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection, "claimed")

    transition_job(connection, "job-1", "claimed", "queued", "worker", now=NOW)

    assert connection.execute("SELECT state FROM job_runs WHERE id='job-1'").fetchone() == ("queued",)


@pytest.mark.parametrize(
    "invalid_now",
    ["not-a-timestamp", "2026-08-21T00:00:00+07:00", "2026-08-21T00:00:00", "2026-02-30T00:00:00Z"],
)
def test_invalid_timestamps_reject_before_any_job_event_or_request_mutation(
    tmp_path: Path, invalid_now: str
) -> None:
    connection = _connection(tmp_path)

    _insert_job(connection, "queued")
    claim_counts = _counts(connection)
    with pytest.raises(ValueError, match="UTC ISO-8601"):
        claim_job(connection, "job-1", "worker", invalid_now)
    assert _counts(connection) == claim_counts

    connection.execute("UPDATE job_runs SET state='running' WHERE id='job-1'")
    connection.commit()
    transition_counts = _counts(connection)
    with pytest.raises(ValueError, match="UTC ISO-8601"):
        transition_job(connection, "job-1", "running", "failed", "worker", now=invalid_now)
    assert _counts(connection) == transition_counts

    event_counts = _counts(connection)
    with pytest.raises(ValueError, match="UTC ISO-8601"):
        record_event(connection, "job-1", "test", "running", "running", "worker", {}, now=invalid_now)
    assert _counts(connection) == event_counts

    connection.execute("UPDATE job_runs SET state='failed' WHERE id='job-1'")
    connection.commit()
    requeue_counts = _counts(connection)
    with pytest.raises(ValueError, match="UTC ISO-8601"):
        requeue_job(connection, "job-1", "operator", "retry", "request-1", now=invalid_now)
    assert _counts(connection) == requeue_counts


def test_kill_switch_blocks_without_mutation_and_can_disable(tmp_path: Path) -> None:
    connection = _connection(tmp_path)
    _insert_job(connection)
    set_kill_switch(True)
    with pytest.raises(KillSwitchEnabled):
        claim_job(connection, "job-1", "worker", NOW)
    assert connection.execute("SELECT state, attempt_count FROM job_runs").fetchone() == ("queued", 0)
    set_kill_switch(False)
    assert claim_job(connection, "job-1", "worker", NOW) is not None


@pytest.mark.parametrize(
    ("status", "remote", "operation", "expected"),
    [(401, False, "write", "failed"), (403, False, "write", "failed"), (429, False, "write", "deferred"),
     (500, False, "read", "failed"), (500, False, "write", "failed"), (None, True, "write", "ambiguous"),
     (None, False, "local", "failed")],
)
def test_classify_failure_matrix(status: int | None, remote: bool, operation: str, expected: str) -> None:
    assert classify_failure(status, remote_side_effect=remote, operation=operation) == expected


def _counts(connection: sqlite3.Connection) -> tuple[int, int, int]:
    return tuple(
        int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        for table in ("job_runs", "job_events", "requeue_requests")
    )
