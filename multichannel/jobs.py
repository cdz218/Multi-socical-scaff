"""Atomic SQLite persistence for the foundation job lifecycle."""

from __future__ import annotations

from hashlib import sha256
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit, urlunsplit
from uuid import uuid4


@dataclass(frozen=True)
class ClaimedJob:
    id: str
    claim_token: str
    worker_id: str
    attempt_count: int


class InvalidTransition(RuntimeError):
    """Raised when a job state change is not permitted."""


class JobNotFound(RuntimeError):
    """Raised when an operation requires a job that does not exist."""


class ReconciliationRequired(RuntimeError):
    """Raised when an ambiguous job lacks reconciliation evidence."""


class KillSwitchEnabled(RuntimeError):
    """Raised before a new claim when the local worker stop is active."""


_KILL_SWITCH_ENABLED = False
_FAILURE_STATES = frozenset({"failed", "deferred", "ambiguous"})
_ALLOWED_TRANSITIONS = {
    "queued": frozenset({"cancelled"}),
    "claimed": frozenset({"running", "queued", "cancelled"}),
    "running": frozenset({"succeeded", "failed", "deferred", "ambiguous", "cancelled"}),
    "failed": frozenset({"cancelled"}),
    "deferred": frozenset({"cancelled"}),
    "ambiguous": frozenset({"cancelled"}),
    "succeeded": frozenset(),
    "cancelled": frozenset(),
}
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?key|authorization|bearer|client[_-]?secret|"
    r"credential|pass(?:word|wd)|private[_-]?key|secret|token)",
    re.I,
)
_URL = re.compile(r"https?://[^\s]+", re.I)
_BEARER = re.compile(r"\bbearer\s+[^\s]+", re.I)
_ASSIGNMENT = re.compile(
    r"\b(?:api[_-]?key|access[_-]?key|authorization|client[_-]?secret|credential|"
    r"pass(?:word|wd)|private[_-]?key|secret|token)\s*[=:]\s*[^\s&;,]+",
    re.I,
)
_BASIC = re.compile(r"\bbasic\s+[A-Za-z0-9+/]+=*", re.I)
_KNOWN_PREFIX = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]+|gh[pousr]_[A-Za-z0-9_-]+|github_pat_[A-Za-z0-9_]+|"
    r"xox[baprs]-[A-Za-z0-9-]+|AKIA[0-9A-Z]{16}|ya29\.[A-Za-z0-9_-]+|"
    r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+|(?:npm|pypi|hf|pat|lin_api)_[A-Za-z0-9_-]+|"
    r"(?:glpat|pplx)-[A-Za-z0-9_-]+)",
    re.I,
)
_OPAQUE = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z0-9_-]{24,}(?![A-Za-z0-9_-])")
_UUID = re.compile(r"^[0-9a-f]{8}[0-9a-f]{4}[1-5][0-9a-f]{3}[89ab][0-9a-f]{12}$", re.I)
_SECRET_QUERY_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?key|authorization|bearer|client[_-]?secret|"
    r"credential|pass(?:word|wd)|private[_-]?key|secret|token|auth)",
    re.I,
)
_PRIVATE_KEY_PEM = re.compile(r"-----BEGIN(?: [A-Z]+)? PRIVATE KEY-----", re.I)
_UTC_ISO_8601 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)
_MAX_ERROR_DETAIL = 512
_IDENTITY_PSEUDONYM_PREFIX = "identity-sha256:"
_IDENTITY_PSEUDONYM = re.compile(r"^identity-sha256:[0-9a-f]{64}$")
_REASON_PSEUDONYM_PREFIX = "reason-sha256:"


def set_kill_switch(enabled: bool) -> None:
    """Set the intentionally process-local guard used before job claims."""
    global _KILL_SWITCH_ENABLED
    _KILL_SWITCH_ENABLED = enabled


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _timestamp(now: str | None) -> str:
    if now is None:
        return _now()
    if not isinstance(now, str) or _UTC_ISO_8601.fullmatch(now) is None:
        raise ValueError("timestamps must be UTC ISO-8601 text ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{now[:-1]}+00:00")
    except ValueError as error:
        raise ValueError("timestamps must be UTC ISO-8601 text ending in Z") from error
    if parsed.tzinfo != timezone.utc:
        raise ValueError("timestamps must be UTC ISO-8601 text ending in Z")
    return now


def _reject_active_transaction(connection: sqlite3.Connection) -> None:
    if connection.in_transaction:
        raise RuntimeError("job operations require a connection without an active transaction")


def _decoded(value: str) -> str:
    """Decode URL escapes for inspection without normalizing safe output."""
    decoded = value
    for _ in range(2):
        next_value = unquote(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    return decoded


def _sanitize_url_component(value: str) -> str:
    result = _BASIC.sub("Basic [REDACTED]", value)
    result = _BEARER.sub("Bearer [REDACTED]", result)
    result = _ASSIGNMENT.sub("[REDACTED]", result)
    result = _KNOWN_PREFIX.sub("[REDACTED]", result)
    result = _OPAQUE.sub(
        lambda match: "[REDACTED]" if _is_opaque_credential(match.group(0)) else match.group(0),
        result,
    )
    return "[REDACTED]" if _looks_sensitive(_decoded(value)) else result


def _sanitize_url_pairs(component: str) -> str:
    if not component:
        return component
    safe_items: list[str] = []
    for item in component.split("&"):
        key, separator, raw_value = item.partition("=")
        decoded_key = _decoded(key)
        decoded_value = _decoded(raw_value)
        if _SECRET_QUERY_KEY.search(decoded_key) or _looks_sensitive(decoded_value):
            safe_items.append("[REDACTED]=[REDACTED]" if separator else "[REDACTED]")
        else:
            safe_items.append(_sanitize_url_component(item))
    return "&".join(safe_items)


def _sanitize_fragment(fragment: str) -> str:
    return _sanitize_url_pairs(fragment)


def _sanitize_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        if parts.username is not None or parts.password is not None:
            return "[REDACTED_URL]"
        safe_path = _sanitize_url_component(parts.path)
        safe_query = _sanitize_url_pairs(parts.query)
        safe_fragment = _sanitize_fragment(parts.fragment)
        return urlunsplit((parts.scheme, parts.netloc, safe_path, safe_query, safe_fragment))
    except ValueError:
        return "[REDACTED_URL]"


def _sanitize_text(value: str) -> str:
    # A first-line-only, bounded message prevents traceback and credential persistence.
    result = value.splitlines()[0] if value.splitlines() else ""
    urls: list[str] = []

    def redact_url(match: re.Match[str]) -> str:
        urls.append(_sanitize_url(match.group(0)))
        return f"\x00{len(urls) - 1}\x00"

    result = _URL.sub(redact_url, result)
    result = _BASIC.sub("Basic [REDACTED]", result)
    result = _BEARER.sub("Bearer [REDACTED]", result)
    result = _ASSIGNMENT.sub("[REDACTED]", result)
    result = _KNOWN_PREFIX.sub("[REDACTED]", result)
    result = _OPAQUE.sub(lambda match: "[REDACTED]" if _is_opaque_credential(match.group(0)) else match.group(0), result)
    for index, url in enumerate(urls):
        result = result.replace(f"\x00{index}\x00", url)
    return result[:_MAX_ERROR_DETAIL]


def _sanitize_identity(value: str, *, internal: bool = False) -> str:
    """Keep a secret-shaped identity usable for equality without storing it."""
    if internal and _IDENTITY_PSEUDONYM.fullmatch(value):
        return value
    if _IDENTITY_PSEUDONYM.fullmatch(value):
        return f"{_IDENTITY_PSEUDONYM_PREFIX}{sha256(value.encode()).hexdigest()}"
    sanitized = _sanitize_text(value)
    if _looks_sensitive(value) or sanitized != value:
        return f"{_IDENTITY_PSEUDONYM_PREFIX}{sha256(value.encode()).hexdigest()}"
    return sanitized


def _sanitize(value: Any, key: str | None = None) -> Any:
    if key is not None and _SECRET_KEY.search(_decoded(key)):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(item_key): _sanitize(item_value, str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        if _looks_sensitive(value):
            return "[REDACTED]"
        return _sanitize_text(value)
    return value


def _looks_sensitive(value: str) -> bool:
    lowered = value.lower()
    return (
        "sentinel-secret" in lowered
        or "bearer " in lowered
        or _BASIC.search(value) is not None
        or _SECRET_KEY.search(value) is not None
        or _KNOWN_PREFIX.search(value) is not None
        or _PRIVATE_KEY_PEM.search(value) is not None
        or _is_opaque_credential(value)
    )


def _is_opaque_credential(value: str) -> bool:
    compact = value.strip()
    if (
        len(compact) < 24
        or re.fullmatch(r"[A-Za-z0-9_-]+", compact) is None
        or _UUID.fullmatch(compact) is not None
    ):
        return False
    if not (re.search(r"[A-Z]", compact) and re.search(r"[a-z]", compact) and re.search(r"\d", compact)):
        return False
    distinct = len(set(compact))
    return distinct >= 12


def _validate_json_values(value: Any) -> None:
    """Reject non-finite JSON values before secret redaction can hide them."""
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("JSON values must be finite")
    elif isinstance(value, Mapping):
        for item in value.values():
            _validate_json_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_json_values(item)


def _json(value: Mapping[str, Any]) -> str:
    _validate_json_values(value)
    return json.dumps(_sanitize(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sanitize_reason(value: str) -> str:
    sanitized = _sanitize_text(value)
    if _looks_sensitive(value) or sanitized != value:
        return f"{_REASON_PSEUDONYM_PREFIX}{sha256(value.encode()).hexdigest()}"
    return sanitized


def _insert_event(
    connection: sqlite3.Connection,
    job_id: str,
    event_type: str,
    from_state: str | None,
    to_state: str | None,
    actor: str,
    detail: Mapping[str, Any],
    now: str,
) -> str:
    event_id = str(uuid4())
    connection.execute(
        "INSERT INTO job_events (id, job_run_id, event_type, from_state, to_state, actor, detail_json, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            event_id,
            job_id,
            _sanitize_text(event_type),
            _sanitize_text(from_state) if from_state is not None else None,
            _sanitize_text(to_state) if to_state is not None else None,
            _sanitize_identity(actor, internal=True),
            _json(detail),
            now,
        ),
    )
    return event_id


def record_event(
    connection: sqlite3.Connection,
    job_id: str,
    event_type: str,
    from_state: str | None,
    to_state: str | None,
    actor: str,
    detail: Mapping[str, Any],
    *,
    now: str | None = None,
) -> str:
    """Persist a standalone immutable event for an existing job."""
    _reject_active_transaction(connection)
    event_time = _timestamp(now)
    safe_actor = _sanitize_identity(actor)
    detail_json = _json(detail)
    safe_event_type = _sanitize_text(event_type)
    safe_from_state = _sanitize_text(from_state) if from_state is not None else None
    safe_to_state = _sanitize_text(to_state) if to_state is not None else None
    try:
        connection.execute("BEGIN")
        row = connection.execute("SELECT state FROM job_runs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        durable_state = str(row[0])
        if (
            (from_state is not None and from_state != durable_state)
            or (to_state is not None and to_state != durable_state)
        ):
            raise InvalidTransition("standalone event states must equal the durable state")
        event_id = _insert_event(
            connection,
            job_id,
            safe_event_type,
            safe_from_state,
            safe_to_state,
            safe_actor,
            json.loads(detail_json),
            event_time,
        )
        connection.commit()
        return event_id
    except Exception:
        connection.rollback()
        raise


def claim_job(connection: sqlite3.Connection, job_id: str, worker_id: str, now: str) -> ClaimedJob | None:
    """Claim a queued job with one compare-and-set update and its matching event."""
    if _KILL_SWITCH_ENABLED:
        raise KillSwitchEnabled("new job claims are disabled")
    claim_time = _timestamp(now)
    _reject_active_transaction(connection)
    safe_worker_id = _sanitize_identity(worker_id)
    claim_token = str(uuid4())
    try:
        connection.execute("BEGIN IMMEDIATE")
        cursor = connection.execute(
            "UPDATE job_runs SET state='claimed', worker_id=?, claim_token=?, claimed_at=?, "
            "updated_at=?, attempt_count=attempt_count+1 WHERE id=? AND state='queued'",
            (safe_worker_id, claim_token, claim_time, claim_time, job_id),
        )
        if cursor.rowcount != 1:
            connection.commit()
            return None
        row = connection.execute(
            "SELECT attempt_count FROM job_runs WHERE id=?", (job_id,)
        ).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        attempt_count = int(row[0])
        _insert_event(
            connection,
            job_id,
            "claimed",
            "queued",
            "claimed",
            safe_worker_id,
            {"attempt_count": attempt_count, "worker_id": safe_worker_id},
            claim_time,
        )
        connection.commit()
        return ClaimedJob(job_id, claim_token, safe_worker_id, attempt_count)
    except Exception:
        connection.rollback()
        raise


def transition_job(
    connection: sqlite3.Connection,
    job_id: str,
    expected_state: str,
    new_state: str,
    actor: str,
    *,
    now: str | None = None,
    error: BaseException | None = None,
    partial_state: Mapping[str, Any] | None = None,
) -> None:
    """Atomically compare-and-set a valid job state and write its event."""
    event_time = _timestamp(now)
    if expected_state in _FAILURE_STATES and new_state == "queued":
        raise InvalidTransition("failed, deferred, and ambiguous jobs must be requeued with requeue_job")
    if new_state not in _ALLOWED_TRANSITIONS.get(expected_state, frozenset()):
        raise InvalidTransition(f"{expected_state} -> {new_state} is not allowed")
    if error is not None and new_state not in _FAILURE_STATES:
        raise InvalidTransition("errors are only valid for failed, deferred, or ambiguous jobs")
    _reject_active_transaction(connection)
    safe_actor = _sanitize_identity(actor)
    # Error class names can carry credential-like suffixes; derive one safe value
    # and reuse it for both durable job state and the matching event.
    error_class = _sanitize_identity(error.__class__.__name__) if error is not None else None
    error_detail = _sanitize_text(str(error)) if error is not None else None
    partial_json = _json(partial_state) if partial_state is not None else None
    assignments = ["state=?", "updated_at=?"]
    values: list[Any] = [new_state, event_time]
    if new_state == "running":
        assignments.append("started_at=?")
        values.append(event_time)
    if new_state in {"succeeded", "failed", "cancelled"}:
        assignments.append("finished_at=?")
        values.append(event_time)
    if new_state == "queued":
        assignments.extend(
            [
                "worker_id=NULL", "claim_token=NULL", "claimed_at=NULL", "started_at=NULL",
                "finished_at=NULL", "next_eligible_at=NULL", "error_class=NULL", "error_detail=NULL",
                "partial_state_json=NULL",
            ]
        )
    elif new_state in _FAILURE_STATES:
        assignments.extend(["error_class=?", "error_detail=?", "partial_state_json=?"])
        values.extend([error_class, error_detail, partial_json])
    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT state, worker_id FROM job_runs WHERE id=?", (job_id,)
        ).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        if row[0] != expected_state:
            raise InvalidTransition(f"job {job_id} is not {expected_state}")
        if expected_state in {"claimed", "running"} and row[1] is not None and safe_actor != row[1]:
            raise InvalidTransition("only the assigned worker owner may transition this job")
        cursor = connection.execute(
            f"UPDATE job_runs SET {', '.join(assignments)} WHERE id=? AND state=?",
            (*values, job_id, expected_state),
        )
        if cursor.rowcount != 1:
            raise InvalidTransition(f"job {job_id} is not {expected_state}")
        detail: dict[str, Any] = {}
        if error is not None:
            detail["error_class"] = error_class
            detail["error_detail"] = error_detail
        if partial_state is not None:
            detail["partial_state"] = json.loads(partial_json) if partial_json is not None else None
        _insert_event(connection, job_id, "state_transition", expected_state, new_state, safe_actor, detail, event_time)
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def requeue_job(
    connection: sqlite3.Connection,
    job_id: str,
    operator: str,
    reason: str,
    request_key: str,
    *,
    reconciled: bool = False,
    now: str | None = None,
) -> bool:
    """Perform a single, explicitly requested requeue, keyed for idempotency."""
    _reject_active_transaction(connection)
    event_time = _timestamp(now)
    safe_operator = _sanitize_identity(operator)
    safe_reason = _sanitize_reason(reason)
    safe_request_key = _sanitize_identity(request_key)
    try:
        connection.execute("BEGIN IMMEDIATE")
        request = connection.execute(
            "SELECT job_run_id, operator, reason FROM requeue_requests WHERE request_key=?", (safe_request_key,)
        ).fetchone()
        if request is not None:
            request_job_id = request[0]
            request_operator = request[1]
            request_reason = request[2]
            if (request_job_id, request_operator, request_reason) == (
                job_id,
                safe_operator,
                safe_reason,
            ):
                connection.commit()
                return False
            raise InvalidTransition("request key was already used with different requeue details")
        row = connection.execute("SELECT state FROM job_runs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise JobNotFound(job_id)
        current_state = str(row[0])
        if current_state == "ambiguous" and not reconciled:
            raise ReconciliationRequired("ambiguous jobs require reconciliation before requeue")
        if current_state not in _FAILURE_STATES:
            raise InvalidTransition(f"cannot requeue a {current_state} job")
        connection.execute(
            "INSERT INTO requeue_requests (id, job_run_id, request_key, operator, reason, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid4()), job_id, safe_request_key, safe_operator, safe_reason, event_time),
        )
        cursor = connection.execute(
            "UPDATE job_runs SET state='queued', worker_id=NULL, claim_token=NULL, claimed_at=NULL, "
            "started_at=NULL, finished_at=NULL, next_eligible_at=NULL, error_class=NULL, error_detail=NULL, "
            "partial_state_json=NULL, updated_at=? WHERE id=? AND state=?",
            (event_time, job_id, current_state),
        )
        if cursor.rowcount != 1:
            raise InvalidTransition(f"job {job_id} changed before requeue")
        _insert_event(
            connection,
            job_id,
            "requeued",
            current_state,
            "queued",
            safe_operator,
            {"reason": safe_reason, "request_key": safe_request_key, "reconciled": reconciled},
            event_time,
        )
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise


def classify_failure(status_code: int | None, *, remote_side_effect: bool, operation: str) -> str:
    """Classify a completed failure without performing retries or remote I/O."""
    if status_code in {401, 403}:
        return "failed"
    if status_code == 429:
        return "deferred"
    if remote_side_effect:
        return "ambiguous"
    if status_code is not None and 500 <= status_code <= 599 and operation == "read":
        return "failed"
    return "failed"
