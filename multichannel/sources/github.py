"""Read-only GitHub search, enrichment, and repository-local captures."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx

from .base import Capability, SourceError, SourceObservationInput, SourceRequestError

_UTC = timezone.utc


def normalize_timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(_UTC).isoformat().replace("+00:00", "Z")


def _parse_datetime(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp is not text")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return parsed.astimezone(_UTC)


def _url(value: Any, *, api: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("URL is not text")
    parts = urlsplit(value)
    allowed = {"api.github.com"} if api else {"github.com", "www.github.com"}
    if parts.scheme != "https" or parts.hostname not in allowed or not parts.path:
        raise ValueError("URL is not a valid absolute GitHub URL")
    return value


def _int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected integer")
    return cast(int, value)


@dataclass(frozen=True)
class GitHubSearchResult:
    github_id: str
    owner: str
    name: str
    html_url: str
    api_url: str
    default_branch: str
    stars: int
    forks: int
    open_issues: int
    watchers: int
    pushed_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class GitHubRepository:
    github_id: str
    owner: str
    name: str
    canonical_url: str
    api_url: str
    default_branch: str
    description: str | None
    language: str | None
    license_spdx: str | None
    topics: tuple[str, ...]
    readme_url: str | None
    readme_ref: str | None
    readme_text: str | None
    readme_sha256: str | None
    created_at: str


@dataclass(frozen=True)
class GitHubRelease:
    github_release_id: str
    repository_github_id: str
    tag_name: str
    name: str | None
    body: str | None
    html_url: str
    published_at: str | None


@dataclass(frozen=True)
class GitHubEnrichment:
    repository: GitHubRepository
    release: GitHubRelease | None
    observations: tuple[SourceObservationInput, ...]
    parse_failures: tuple[str, ...] = ()


def _parse_search_row(row: Any) -> GitHubSearchResult:
    if not isinstance(row, dict):
        raise ValueError("search row is not an object")
    owner = row.get("owner")
    if not isinstance(owner, dict) or not isinstance(owner.get("login"), str):
        raise ValueError("search row owner is malformed")
    return GitHubSearchResult(
        github_id=str(_int(row["id"])), owner=owner["login"], name=str(row["name"]),
        html_url=_url(row["html_url"]), api_url=_url(row["url"], api=True),
        default_branch=str(row.get("default_branch") or "main"), stars=_int(row["stargazers_count"]),
        forks=_int(row["forks_count"]), open_issues=_int(row["open_issues_count"]),
        watchers=_int(row["watchers_count"]), pushed_at=_parse_datetime(row["pushed_at"]),
        updated_at=_parse_datetime(row["updated_at"]),
    )


def _rate_limit(response: httpx.Response) -> int | None:
    value = response.headers.get("X-RateLimit-Remaining")
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


class GitHubAdapter:
    def __init__(
        self,
        *,
        token: str | None = None,
        base_url: str = "https://api.github.com",
        client: httpx.Client | None = None,
        timeout: float = 10.0,
        search_limit: int = 20,
    ) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.search_limit = max(1, min(search_limit, 100))
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)
        self._owns_client = client is None
        self.last_search_observation: SourceObservationInput | None = None

    def close(self) -> None:
        """Close an internally created HTTP client; injected clients remain caller-owned."""
        if self._owns_client and not self._client.is_closed:
            self._client.close()

    def __enter__(self) -> GitHubAdapter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        # Cleanup is a backstop only; callers should use close() or a context manager.
        try:
            self.close()
        except Exception:
            pass

    def _capability_reason(self, authenticated: bool) -> str:
        if authenticated:
            return "GitHub token-authenticated read-only capability"
        return "GitHub unauthenticated read-only capability"

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        allowed_statuses: frozenset[int] = frozenset(),
    ) -> httpx.Response:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "multichannel-github-source/0.1",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            response = self._client.request(method, f"{self.base_url}{path}", params=params, headers=headers)
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            detail = str(error).splitlines()[0][:256]
            if self.token:
                detail = detail.replace(self.token, "[REDACTED]")
            detail = re.sub(r"(?i)(token|authorization|bearer)\s*[=:]\s*\S+", r"\1=[REDACTED]", detail)
            raise SourceRequestError(SourceError("failed", None, path, detail)) from None
        if response.status_code in allowed_statuses:
            return response
        if response.status_code in {429} or response.status_code >= 500:
            outcome: Literal["deferred", "failed"] = (
                "deferred" if response.status_code == 429 else "failed"
            )
            raise SourceRequestError(SourceError(outcome, response.status_code, path, "GitHub read request failed"))
        if response.status_code in {401, 403}:
            raise SourceRequestError(SourceError("failed", response.status_code, path, "GitHub credentials are blocked"))
        return response

    def preflight(self) -> Capability:
        try:
            response = self._request("GET", "/rate_limit")
        except SourceRequestError as error:
            if error.error.status_code in {401, 403}:
                return Capability("credentials_blocked", error.error.detail, None)
            raise
        account_id: str | None = None
        try:
            data = response.json()
            user = data.get("user") if isinstance(data, dict) else None
            if isinstance(user, dict) and isinstance(user.get("id"), int):
                account_id = str(user["id"])
        except (ValueError, TypeError):
            pass
        return Capability("ready_direct", self._capability_reason(bool(self.token)), account_id)

    def search(self, profile: str, observed_at: datetime) -> list[GitHubSearchResult]:
        response = self._request(
            "GET", "/search/repositories", params={"q": profile, "per_page": self.search_limit, "page": 1}
        )
        raw = response.content
        try:
            payload = response.json()
            items = payload["items"] if isinstance(payload, dict) else []
            incomplete = bool(payload.get("incomplete_results", False)) if isinstance(payload, dict) else False
        except (ValueError, TypeError, KeyError):
            raise SourceRequestError(SourceError("failed", response.status_code, "/search/repositories", "malformed search response")) from None
        results: list[GitHubSearchResult] = []
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        for item in items if isinstance(items, list) else []:
            try:
                result = _parse_search_row(item)
            except (KeyError, TypeError, ValueError):
                continue
            normalized_name = f"{result.owner.casefold()}/{result.name.casefold()}"
            if result.github_id in seen_ids or normalized_name in seen_names:
                continue
            seen_ids.add(result.github_id)
            seen_names.add(normalized_name)
            results.append(result)
        self.last_search_observation = SourceObservationInput(
            "github_repository", f"search:{profile}", observed_at, {"result_count": len(results)}, raw,
            incomplete, _rate_limit(response)
        )
        return results

    def enrich(self, result: GitHubSearchResult, observed_at: datetime) -> GitHubEnrichment:
        base = f"/repos/{quote(result.owner, safe='')}/{quote(result.name, safe='')}"
        repo_response = self._request("GET", base)
        readme_response = self._request("GET", f"{base}/readme")
        observations = [
            SourceObservationInput("github_repository", f"github:{result.github_id}:repository", observed_at, {"stars": result.stars, "forks": result.forks}, repo_response.content, rate_limit_remaining=_rate_limit(repo_response)),
        ]
        failures: list[str] = []
        try:
            repo_payload = repo_response.json()
            repository = self._repository_payload(repo_payload)
        except (ValueError, KeyError, TypeError):
            raise SourceRequestError(SourceError("failed", repo_response.status_code, base, "malformed repository response")) from None
        try:
            readme_payload = readme_response.json()
            content = base64.b64decode(readme_payload.get("content", ""), validate=True)
            repository = GitHubRepository(**{**repository.__dict__, "readme_url": _url(readme_payload["html_url"]) if readme_payload.get("html_url") else None, "readme_ref": readme_payload.get("sha"), "readme_text": content.decode("utf-8"), "readme_sha256": hashlib.sha256(content).hexdigest()})
        except (ValueError, KeyError, TypeError, UnicodeDecodeError, binascii.Error):
            failures.append("README response was malformed")
        observations.append(SourceObservationInput("github_repository", f"github:{result.github_id}:readme", observed_at, {}, readme_response.content, rate_limit_remaining=_rate_limit(readme_response)))
        release: GitHubRelease | None = None
        release_path = f"{base}/releases/latest"
        release_response = self._request("GET", release_path, allowed_statuses=frozenset({404}))
        if release_response.status_code != 404:
            observations.append(SourceObservationInput("github_release", f"github:{result.github_id}:release", observed_at, {}, release_response.content, rate_limit_remaining=_rate_limit(release_response)))
            try:
                release = self._release_payload(release_response.json(), result.github_id)
            except (ValueError, KeyError, TypeError):
                failures.append("latest release response was malformed")
        else:
            observations.append(SourceObservationInput("github_release", f"github:{result.github_id}:release", observed_at, {}, release_response.content, rate_limit_remaining=_rate_limit(release_response)))
        return GitHubEnrichment(repository, release, tuple(observations), tuple(failures))

    def _repository_payload(self, payload: Any) -> GitHubRepository:
        if not isinstance(payload, dict):
            raise ValueError("repository is not an object")
        license_data = payload.get("license")
        return GitHubRepository(
            github_id=str(_int(payload["id"])), owner=str(payload["owner"]["login"]), name=str(payload["name"]),
            canonical_url=_url(payload["html_url"]), api_url=_url(payload["url"], api=True),
            default_branch=str(payload.get("default_branch") or "main"), description=payload.get("description"),
            language=payload.get("language"), license_spdx=license_data.get("spdx_id") if isinstance(license_data, dict) else None,
            topics=tuple(sorted(str(topic) for topic in payload.get("topics", []) if isinstance(topic, str))),
            readme_url=None, readme_ref=None, readme_text=None, readme_sha256=None,
            created_at=normalize_timestamp(_parse_datetime(payload["created_at"])),
        )

    def _release_payload(self, payload: Any, repository_github_id: str) -> GitHubRelease:
        if not isinstance(payload, dict) or not isinstance(payload.get("id"), int):
            raise ValueError("release id is malformed")
        return GitHubRelease(str(payload["id"]), repository_github_id, str(payload["tag_name"]), payload.get("name"), payload.get("body"), _url(payload["html_url"]), normalize_timestamp(_parse_datetime(payload["published_at"])) if payload.get("published_at") else None)


def _capture_path(root: Path, observation: SourceObservationInput, raw_sha: str) -> Path:
    if observation.source_kind not in {
        "github_repository", "github_release", "reddit_post", "reddit_comment"
    }:
        raise ValueError("capture source kind is not approved")
    timestamp = normalize_timestamp(observation.observed_at).replace(":", "").replace("-", "")
    identity_hash = hashlib.sha256(observation.source_id.encode("utf-8")).hexdigest()
    return root / observation.source_kind / f"{identity_hash}-{timestamp}-{raw_sha}.json"


def _require_no_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"capture {label} must not be a symlink")


def _validated_capture_root(root: Path) -> Path:
    absolute = root.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        _require_no_symlink(current, "root")
    absolute.mkdir(parents=True, exist_ok=True)
    _require_no_symlink(absolute, "root")
    if not absolute.is_dir():
        raise ValueError("capture root is not a directory")
    return absolute.resolve()


def _stage_capture(root: Path, observation: SourceObservationInput) -> tuple[Path, Path, str]:
    root = _validated_capture_root(root)
    raw_sha = hashlib.sha256(observation.raw_bytes).hexdigest()
    target = _capture_path(root, observation, raw_sha)
    if not target.is_relative_to(root):
        raise ValueError("capture path escaped approved root")
    _require_no_symlink(target.parent, "source-kind directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    _require_no_symlink(target.parent, "source-kind directory")
    parent = target.parent.resolve()
    if not parent.is_relative_to(root):
        raise ValueError("capture path escaped approved root")
    _require_no_symlink(target, "destination")
    temporary = parent / f".{target.name}.{uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as staged:
            staged.write(observation.raw_bytes)
            staged.flush()
            os.fsync(staged.fileno())
    except Exception:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise
    return temporary, target, str(target.relative_to(root))


def _finalize_capture(temporary: Path, target: Path) -> bool:
    _require_no_symlink(target, "destination")
    if target.exists():
        temporary.unlink(missing_ok=True)
        return False
    os.replace(temporary, target)
    return True


def write_capture(root: Path, observation: SourceObservationInput) -> str:
    temporary, target, relative_path = _stage_capture(root, observation)
    try:
        _finalize_capture(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if not target.resolve().is_relative_to(target.parents[1].resolve()):
        raise ValueError("capture path escaped approved root")
    return relative_path


def persist_github_repository(connection: sqlite3.Connection, repository: GitHubRepository) -> str:
    repository_id = str(uuid4())
    connection.execute(
        """INSERT INTO github_repositories (id, github_id, owner, name, canonical_url, api_url, default_branch, description, language, license_spdx, topics_json, readme_url, readme_ref, readme_text, readme_sha256, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(github_id) DO UPDATE SET owner=excluded.owner, name=excluded.name, canonical_url=excluded.canonical_url, api_url=excluded.api_url, default_branch=excluded.default_branch, description=excluded.description, language=excluded.language, license_spdx=excluded.license_spdx, topics_json=excluded.topics_json, readme_url=excluded.readme_url, readme_ref=excluded.readme_ref, readme_text=excluded.readme_text, readme_sha256=excluded.readme_sha256""",
        (repository_id, repository.github_id, repository.owner, repository.name, repository.canonical_url, repository.api_url, repository.default_branch, repository.description, repository.language, repository.license_spdx, json.dumps(list(repository.topics), separators=(",", ":"), sort_keys=True), repository.readme_url, repository.readme_ref, repository.readme_text, repository.readme_sha256, repository.created_at),
    )
    row = connection.execute("SELECT id FROM github_repositories WHERE github_id = ?", (repository.github_id,)).fetchone()
    if row is None:
        raise RuntimeError("repository upsert did not return an id")
    connection.commit()
    return str(row[0])


def persist_observation(connection: sqlite3.Connection, observation: SourceObservationInput, capture_root: Path, *, repository_id: str | None = None, release_id: str | None = None) -> str:
    if (repository_id is None) == (release_id is None):
        raise sqlite3.IntegrityError("exactly one parent is required")
    if repository_id is not None:
        if observation.source_kind != "github_repository":
            raise sqlite3.IntegrityError("GitHub repository observations require a repository parent")
        if connection.execute("SELECT 1 FROM github_repositories WHERE id = ?", (repository_id,)).fetchone() is None:
            raise sqlite3.IntegrityError("unknown GitHub repository parent")
        source_identity = f"github_repository:{repository_id}"
    else:
        if observation.source_kind != "github_release":
            raise sqlite3.IntegrityError("GitHub release observations require a release parent")
        if connection.execute("SELECT 1 FROM github_releases WHERE id = ?", (release_id,)).fetchone() is None:
            raise sqlite3.IntegrityError("unknown GitHub release parent")
        source_identity = f"github_release:{release_id}"
    if connection.in_transaction:
        raise sqlite3.IntegrityError("observation persistence requires no active transaction")
    raw_sha = hashlib.sha256(observation.raw_bytes).hexdigest()
    persisted_observation = SourceObservationInput(
        observation.source_kind, source_identity, observation.observed_at, observation.metrics,
        observation.raw_bytes, observation.incomplete_results, observation.rate_limit_remaining,
    )
    temporary, target, relative_path = _stage_capture(capture_root, persisted_observation)
    observation_id = str(uuid4())
    created_capture = False
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO source_observations (id, source_kind, source_identity, observed_at, github_repository_id, github_release_id, metrics_json, raw_sha256, raw_path, incomplete_results, rate_limit_remaining, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (observation_id, persisted_observation.source_kind, source_identity, normalize_timestamp(observation.observed_at), repository_id, release_id, json.dumps(observation.metrics, sort_keys=True, separators=(",", ":")), raw_sha, relative_path, int(observation.incomplete_results), observation.rate_limit_remaining, normalize_timestamp(datetime.now(_UTC))),
        )
        created_capture = _finalize_capture(temporary, target)
        connection.commit()
    except Exception:
        connection.rollback()
        temporary.unlink(missing_ok=True)
        if created_capture:
            target.unlink(missing_ok=True)
        raise
    return observation_id


def persist_release(connection: sqlite3.Connection, release: GitHubRelease, repository_id: str) -> str:
    release_id = str(uuid4())
    connection.execute(
        """INSERT INTO github_releases (id, repository_id, github_release_id, tag_name, name, body, html_url, published_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(github_release_id) DO UPDATE SET repository_id=excluded.repository_id, tag_name=excluded.tag_name, name=excluded.name, body=excluded.body, html_url=excluded.html_url, published_at=excluded.published_at""",
        (release_id, repository_id, release.github_release_id, release.tag_name, release.name, release.body, release.html_url, release.published_at, normalize_timestamp(datetime.now(_UTC))),
    )
    row = connection.execute("SELECT id FROM github_releases WHERE github_release_id = ?", (release.github_release_id,)).fetchone()
    if row is None:
        raise RuntimeError("release upsert did not return an id")
    connection.commit()
    return str(row[0])
