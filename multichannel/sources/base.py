"""Shared typed records and bounded source errors."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class SourceObservationInput:
    source_kind: Literal["github_repository", "github_release", "reddit_post", "reddit_comment"]
    source_id: str
    observed_at: datetime
    metrics: dict[str, int | float | None]
    raw_bytes: bytes
    incomplete_results: bool = False
    rate_limit_remaining: int | None = None


@dataclass(frozen=True)
class Capability:
    state: Literal["disabled", "credentials_blocked", "app_review_blocked", "ready_manual_finish", "ready_direct"]
    reason: str
    account_id: str | None
    media_capability: Literal["none", "public_url", "resumable"] = "none"


@dataclass(frozen=True)
class SourceError:
    outcome: Literal["deferred", "failed"]
    status_code: int | None
    endpoint_path: str
    detail: str


class SourceRequestError(RuntimeError):
    """Safe, typed failure for a read-only source request."""

    def __init__(self, error: SourceError) -> None:
        self.error = error
        super().__init__(f"{error.outcome} {error.endpoint_path}: {error.detail}")
