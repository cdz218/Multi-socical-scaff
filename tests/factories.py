"""Typed fixtures for SQLite persistence tests."""

from __future__ import annotations

from multichannel.models import Channel, JobEvent, JobRun, PlatformAccount, RequeueRequest


def channel() -> Channel:
    return Channel(slug="example-channel", name="Example Channel")


def platform_account(channel_id: str) -> PlatformAccount:
    return PlatformAccount(channel_id=channel_id, platform="youtube")


def job_run() -> JobRun:
    return JobRun(job_type="collect", subject_type="channel", subject_id="subject")


def job_event(job_run_id: str) -> JobEvent:
    return JobEvent(
        job_run_id=job_run_id,
        event_type="queued",
        actor="test",
        detail_json="{}",
    )


def requeue_request(job_run_id: str) -> RequeueRequest:
    return RequeueRequest(
        job_run_id=job_run_id,
        request_key="test-request",
        operator="test",
        reason="test fixture",
    )
