"""Small typed records used by the foundation persistence layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _id() -> str:
    return str(uuid4())


@dataclass(frozen=True)
class Channel:
    slug: str
    name: str
    id: str = field(default_factory=_id)
    locale: str = "en"
    enabled: int = 1
    created_at: str = field(default_factory=_now)


@dataclass(frozen=True)
class PlatformAccount:
    channel_id: str
    platform: str
    id: str = field(default_factory=_id)
    capability_state: str = "disabled"
    media_capability: str = "none"
    enabled: int = 0
    created_at: str = field(default_factory=_now)


@dataclass(frozen=True)
class JobRun:
    job_type: str
    subject_type: str
    subject_id: str
    id: str = field(default_factory=_id)
    state: str = "queued"
    attempt_count: int = 0
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)


@dataclass(frozen=True)
class JobEvent:
    job_run_id: str
    event_type: str
    actor: str
    detail_json: str
    id: str = field(default_factory=_id)
    created_at: str = field(default_factory=_now)


@dataclass(frozen=True)
class RequeueRequest:
    job_run_id: str
    request_key: str
    operator: str
    reason: str
    id: str = field(default_factory=_id)
    created_at: str = field(default_factory=_now)
