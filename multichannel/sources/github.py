"""Read-only GitHub search, enrichment, and repository-local captures."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import sqlite3
import stat
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import SplitResult, quote, unquote, urlsplit
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


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty text")
    return value


def _canonical_github_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"(?:0|[1-9][0-9]*)", value):
        raise ValueError(f"{label} must be canonical decimal text")
    return value


def _path_component(value: Any, label: str) -> str:
    text = _nonempty_text(value, label)
    if any(character in text for character in "/\\?#"):
        raise ValueError(f"{label} is not a valid path component")
    return text


def _approved_url_parts(value: Any, *, api: bool = False) -> tuple[str, SplitResult]:
    if not isinstance(value, str):
        raise ValueError("URL is not text")
    parts = urlsplit(value)
    allowed = {"api.github.com"} if api else {"github.com", "www.github.com"}
    try:
        port = parts.port
    except ValueError as error:
        raise ValueError("URL has an invalid port") from error
    if (
        parts.scheme != "https"
        or parts.hostname is None
        or parts.hostname.casefold() not in allowed
        or parts.username is not None
        or parts.password is not None
        or port not in {None, 443}
        or parts.query
        or parts.fragment
        or not parts.path.startswith("/")
    ):
        raise ValueError("URL is not a valid absolute GitHub URL")
    return value, parts


def _url(value: Any, *, api: bool = False) -> str:
    return _approved_url_parts(value, api=api)[0]


def _repository_url_matches(url: str, owner: str, name: str, *, api: bool = False) -> bool:
    _, parts = _approved_url_parts(url, api=api)
    prefix = "/repos" if api else ""
    expected = f"{prefix}/{quote(owner, safe='')}/{quote(name, safe='')}"
    return parts.path.casefold() == expected.casefold()


def _readme_url_matches(url: str, owner: str, name: str) -> bool:
    _, parts = _approved_url_parts(url)
    prefix = f"/{quote(owner, safe='')}/{quote(name, safe='')}/"
    return parts.path.casefold().startswith(prefix.casefold())


def _release_url_matches(
    url: str, tag_name: str, *, owner: str | None = None, name: str | None = None
) -> bool:
    _, parts = _approved_url_parts(url)
    release_path = f"/releases/tag/{quote(tag_name, safe='')}"
    if owner is None or name is None:
        return parts.path.casefold().endswith(release_path.casefold())
    segments = parts.path.split("/")
    if len(segments) != 6 or segments[0] != "":
        return False
    if any(re.search(r"%(?![0-9A-Fa-f]{2})", segment) for segment in segments[1:]):
        return False
    decoded_segments = [unquote(segment) for segment in segments[1:]]
    if any("/" in segment for segment in decoded_segments[:2]):
        return False
    actual_owner, actual_name, releases, tag, actual_tag_name = decoded_segments
    return (
        actual_owner.casefold() == owner.casefold()
        and actual_name.casefold() == name.casefold()
        and releases == "releases"
        and tag == "tag"
        and actual_tag_name == tag_name
    )


def _utc_z_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{label} must be a UTC ISO-Z timestamp")
    parsed = _parse_datetime(value)
    if normalize_timestamp(parsed) != value:
        raise ValueError(f"{label} must be a UTC ISO-Z timestamp")
    return value


def _utc_datetime(value: Any, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{label} must be a UTC timestamp")
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

    def __post_init__(self) -> None:
        _canonical_github_id(self.github_id, "GitHub ID")
        owner = _path_component(self.owner, "repository owner")
        name = _path_component(self.name, "repository name")
        if not _repository_url_matches(_url(self.html_url), owner, name):
            raise ValueError("repository URL does not match owner and name")
        if not _repository_url_matches(_url(self.api_url, api=True), owner, name, api=True):
            raise ValueError("repository API URL does not match owner and name")
        _utc_datetime(self.pushed_at, "pushed timestamp")
        _utc_datetime(self.updated_at, "updated timestamp")


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

    def __post_init__(self) -> None:
        _canonical_github_id(self.github_id, "GitHub ID")
        owner = _path_component(self.owner, "repository owner")
        name = _path_component(self.name, "repository name")
        if not _repository_url_matches(_url(self.canonical_url), owner, name):
            raise ValueError("repository URL does not match owner and name")
        if not _repository_url_matches(_url(self.api_url, api=True), owner, name, api=True):
            raise ValueError("repository API URL does not match owner and name")
        if self.readme_url is not None and not _readme_url_matches(self.readme_url, owner, name):
            raise ValueError("README URL does not match repository")
        if self.readme_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.readme_sha256):
            raise ValueError("README hash must be lowercase SHA-256")
        _utc_z_timestamp(self.created_at, "created timestamp")


@dataclass(frozen=True)
class GitHubRelease:
    github_release_id: str
    repository_github_id: str
    tag_name: str
    name: str | None
    body: str | None
    html_url: str
    published_at: str | None

    def __post_init__(self) -> None:
        _canonical_github_id(self.github_release_id, "GitHub release ID")
        _canonical_github_id(self.repository_github_id, "repository GitHub ID")
        tag_name = _nonempty_text(self.tag_name, "release tag")
        if not _release_url_matches(self.html_url, tag_name):
            raise ValueError("release URL does not match release tag")
        if self.published_at is not None:
            _utc_z_timestamp(self.published_at, "published timestamp")


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


def _decode_readme_base64(payload: dict[str, Any]) -> bytes:
    if payload.get("encoding", "base64") != "base64":
        raise ValueError("README encoding is unsupported")
    encoded_content = payload.get("content")
    if not isinstance(encoded_content, str):
        raise ValueError("README content is not text")
    try:
        encoded = encoded_content.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("README content is not ASCII base64") from error
    # GitHub wraps README base64; accept only its documented ASCII whitespace.
    compact = (
        encoded.replace(b" ", b"")
        .replace(b"\t", b"")
        .replace(b"\r", b"")
        .replace(b"\n", b"")
    )
    return base64.b64decode(compact, validate=True)


def _validated_base_url(value: str) -> str:
    parts = urlsplit(value)
    try:
        port = parts.port
    except ValueError as error:
        raise ValueError("base URL has an invalid port") from error
    if (
        parts.scheme != "https"
        or parts.hostname is None
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or port not in {None, 443}
    ):
        raise ValueError("base URL must be an absolute HTTPS origin without credentials")
    decoded_path = unquote(parts.path)
    if any(segment in {".", ".."} for segment in decoded_path.split("/")):
        raise ValueError("base URL path prefix is not safe")
    return f"https://{parts.netloc}{parts.path.rstrip('/')}"


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
        self.base_url = _validated_base_url(base_url)
        self.search_limit = max(1, min(search_limit, 100))
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=False)
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
            response = self._client.request(
                method,
                f"{self.base_url}{path}",
                params=params,
                headers=headers,
                follow_redirects=False,
            )
        except httpx.RequestError:
            raise SourceRequestError(
                SourceError("failed", None, path, "GitHub network request failed")
            ) from None
        if 200 <= response.status_code < 300 or response.status_code in allowed_statuses:
            return response
        if response.status_code in {429} or response.status_code >= 500:
            outcome: Literal["deferred", "failed"] = (
                "deferred" if response.status_code == 429 else "failed"
            )
            raise SourceRequestError(SourceError(outcome, response.status_code, path, "GitHub read request failed"))
        if response.status_code in {401, 403}:
            raise SourceRequestError(SourceError("failed", response.status_code, path, "GitHub credentials are blocked"))
        raise SourceRequestError(SourceError("failed", response.status_code, path, "GitHub read request failed"))

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
            if not isinstance(payload, dict):
                raise ValueError("search response is not an object")
            items = payload["items"]
            incomplete = payload["incomplete_results"]
            if not isinstance(items, list) or not isinstance(incomplete, bool):
                raise ValueError("search response shape is malformed")
        except (ValueError, TypeError, KeyError):
            raise SourceRequestError(SourceError("failed", response.status_code, "/search/repositories", "malformed search response")) from None
        results: list[GitHubSearchResult] = []
        seen_ids: set[str] = set()
        seen_names: set[str] = set()
        for item in items:
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
        try:
            repo_payload = repo_response.json()
            repository = self._repository_payload(repo_payload)
        except (ValueError, KeyError, TypeError):
            raise SourceRequestError(SourceError("failed", repo_response.status_code, base, "malformed repository response")) from None
        if (
            repository.github_id != result.github_id
            or repository.owner.casefold() != result.owner.casefold()
            or repository.name.casefold() != result.name.casefold()
        ):
            raise SourceRequestError(
                SourceError("failed", repo_response.status_code, base, "repository response identity did not match request")
            )
        observations = [
            SourceObservationInput("github_repository", f"github:{result.github_id}:repository", observed_at, {"stars": result.stars, "forks": result.forks}, repo_response.content, rate_limit_remaining=_rate_limit(repo_response)),
        ]
        failures: list[str] = []
        readme_response = self._request("GET", f"{base}/readme", allowed_statuses=frozenset({404}))
        if readme_response.status_code != 404:
            try:
                readme_payload = readme_response.json()
                if not isinstance(readme_payload, dict):
                    raise ValueError("README is not an object")
                content = _decode_readme_base64(readme_payload)
                readme_url = readme_payload.get("html_url")
                repository = replace(
                    repository,
                    readme_url=_url(readme_url) if readme_url is not None else None,
                    readme_ref=readme_payload.get("sha"),
                    readme_text=content.decode("utf-8"),
                    readme_sha256=hashlib.sha256(content).hexdigest(),
                )
            except (ValueError, KeyError, TypeError, UnicodeDecodeError, binascii.Error):
                failures.append("README response was malformed")
        observations.append(SourceObservationInput("github_repository", f"github:{result.github_id}:readme", observed_at, {}, readme_response.content, rate_limit_remaining=_rate_limit(readme_response)))
        release: GitHubRelease | None = None
        release_path = f"{base}/releases/latest"
        release_response = self._request("GET", release_path, allowed_statuses=frozenset({404}))
        if release_response.status_code != 404:
            try:
                release = self._release_payload(
                    release_response.json(),
                    repository.github_id,
                    repository.owner,
                    repository.name,
                )
            except (ValueError, KeyError, TypeError):
                failures.append("latest release response was malformed")
            else:
                observations.append(SourceObservationInput("github_release", f"github:{result.github_id}:release", observed_at, {}, release_response.content, rate_limit_remaining=_rate_limit(release_response)))
        return GitHubEnrichment(repository, release, tuple(observations), tuple(failures))

    def _repository_payload(self, payload: Any) -> GitHubRepository:
        if not isinstance(payload, dict):
            raise ValueError("repository is not an object")
        owner_data = payload.get("owner")
        if not isinstance(owner_data, dict):
            raise ValueError("repository owner is malformed")
        owner = _path_component(owner_data.get("login"), "repository owner")
        name = _path_component(payload.get("name"), "repository name")
        license_data = payload.get("license")
        return GitHubRepository(
            github_id=str(_int(payload["id"])), owner=owner, name=name,
            canonical_url=_url(payload["html_url"]), api_url=_url(payload["url"], api=True),
            default_branch=str(payload.get("default_branch") or "main"), description=payload.get("description"),
            language=payload.get("language"), license_spdx=license_data.get("spdx_id") if isinstance(license_data, dict) else None,
            topics=tuple(sorted(str(topic) for topic in payload.get("topics", []) if isinstance(topic, str))),
            readme_url=None, readme_ref=None, readme_text=None, readme_sha256=None,
            created_at=normalize_timestamp(_parse_datetime(payload["created_at"])),
        )

    def _release_payload(
        self, payload: Any, repository_github_id: str, owner: str, name: str
    ) -> GitHubRelease:
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("id"), int)
            or not isinstance(payload.get("tag_name"), str)
        ):
            raise ValueError("release id is malformed")
        tag_name = payload["tag_name"]
        html_url = _url(payload["html_url"])
        if not _release_url_matches(html_url, tag_name, owner=owner, name=name):
            raise ValueError("release URL does not match repository")
        return GitHubRelease(
            str(payload["id"]), repository_github_id, tag_name, payload.get("name"),
            payload.get("body"), html_url,
            normalize_timestamp(_parse_datetime(payload["published_at"]))
            if payload.get("published_at")
            else None,
        )


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


def _validated_capture_target(root: Path, target: Path) -> Path:
    _require_no_symlink(target.parent, "source-kind directory")
    parent = target.parent.resolve()
    if not parent.is_relative_to(root):
        raise ValueError("capture path escaped approved root")
    _require_no_symlink(target, "destination")
    return parent


def _stage_capture(root: Path, observation: SourceObservationInput) -> tuple[Path, Path, Path, str]:
    root = _validated_capture_root(root)
    raw_sha = hashlib.sha256(observation.raw_bytes).hexdigest()
    target = _capture_path(root, observation, raw_sha)
    if not target.is_relative_to(root):
        raise ValueError("capture path escaped approved root")
    target.parent.mkdir(parents=True, exist_ok=True)
    parent = _validated_capture_target(root, target)
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
    return temporary, target, root, str(target.relative_to(root))


def _existing_capture_matches(target: Path, expected_sha: str) -> bool:
    _require_no_symlink(target, "destination")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(target, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("capture destination must be a regular file")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb") as existing:
            descriptor = -1
            for chunk in iter(lambda: existing.read(64 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha:
            raise ValueError("capture destination hash does not match expected raw bytes")
        return True
    finally:
        if descriptor != -1:
            os.close(descriptor)


def _capture_identity(path: Path) -> tuple[int, int]:
    status = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(status.st_mode):
        raise ValueError("capture destination must be a regular file")
    return status.st_dev, status.st_ino


def _remove_published_capture_if_owned(
    root: Path, target: Path, ownership: tuple[int, int]
) -> None:
    """Remove a published destination only while its inode remains ours."""
    try:
        _validated_capture_target(root, target)
        target_status = target.stat(follow_symlinks=False)
    except (FileNotFoundError, ValueError):
        return
    if stat.S_ISREG(target_status.st_mode) and (
        target_status.st_dev,
        target_status.st_ino,
    ) == ownership:
        target.unlink()


def _finalize_capture(
    root: Path, temporary: Path, target: Path, expected_sha: str
) -> tuple[int, int] | None:
    _validated_capture_target(root, target)
    try:
        # Linking publishes only if absent, so a pre-existing capture is never overwritten.
        os.link(temporary, target)
    except FileExistsError:
        _validated_capture_target(root, target)
        _existing_capture_matches(target, expected_sha)
        temporary.unlink(missing_ok=True)
        return None
    # The staging inode identifies this publication even if the path is replaced later.
    ownership = _capture_identity(temporary)
    try:
        _validated_capture_target(root, target)
        _existing_capture_matches(target, expected_sha)
    except Exception:
        _remove_published_capture_if_owned(root, target, ownership)
        raise
    temporary.unlink(missing_ok=True)
    return ownership


def write_capture(root: Path, observation: SourceObservationInput) -> str:
    temporary, target, capture_root, relative_path = _stage_capture(root, observation)
    try:
        _finalize_capture(capture_root, temporary, target, hashlib.sha256(observation.raw_bytes).hexdigest())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if not target.resolve().is_relative_to(capture_root):
        raise ValueError("capture path escaped approved root")
    return relative_path


def persist_github_repository(connection: sqlite3.Connection, repository: GitHubRepository) -> str:
    if connection.in_transaction:
        raise sqlite3.IntegrityError("repository persistence requires no active transaction")
    repository_id = str(uuid4())
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """INSERT INTO github_repositories (id, github_id, owner, name, canonical_url, api_url, default_branch, description, language, license_spdx, topics_json, readme_url, readme_ref, readme_text, readme_sha256, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(github_id) DO UPDATE SET owner=excluded.owner, name=excluded.name, canonical_url=excluded.canonical_url, api_url=excluded.api_url, default_branch=excluded.default_branch, description=excluded.description, language=excluded.language, license_spdx=excluded.license_spdx, topics_json=excluded.topics_json, readme_url=excluded.readme_url, readme_ref=excluded.readme_ref, readme_text=excluded.readme_text, readme_sha256=excluded.readme_sha256""",
            (repository_id, repository.github_id, repository.owner, repository.name, repository.canonical_url, repository.api_url, repository.default_branch, repository.description, repository.language, repository.license_spdx, json.dumps(list(repository.topics), separators=(",", ":"), sort_keys=True), repository.readme_url, repository.readme_ref, repository.readme_text, repository.readme_sha256, repository.created_at),
        )
        row = connection.execute(
            "SELECT id FROM github_repositories WHERE github_id = ?", (repository.github_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("repository upsert did not return an id")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return str(row[0])


def persist_observation(connection: sqlite3.Connection, observation: SourceObservationInput, capture_root: Path, *, repository_id: str | None = None, release_id: str | None = None) -> str:
    if connection.in_transaction:
        raise sqlite3.IntegrityError("observation persistence requires no active transaction")
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
    raw_sha = hashlib.sha256(observation.raw_bytes).hexdigest()
    persisted_observation = SourceObservationInput(
        observation.source_kind, source_identity, observation.observed_at, observation.metrics,
        observation.raw_bytes, observation.incomplete_results, observation.rate_limit_remaining,
    )
    temporary, target, validated_root, relative_path = _stage_capture(capture_root, persisted_observation)
    observation_id = str(uuid4())
    capture_ownership: tuple[int, int] | None = None
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO source_observations (id, source_kind, source_identity, observed_at, github_repository_id, github_release_id, metrics_json, raw_sha256, raw_path, incomplete_results, rate_limit_remaining, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (observation_id, persisted_observation.source_kind, source_identity, normalize_timestamp(observation.observed_at), repository_id, release_id, json.dumps(observation.metrics, sort_keys=True, separators=(",", ":")), raw_sha, relative_path, int(observation.incomplete_results), observation.rate_limit_remaining, normalize_timestamp(datetime.now(_UTC))),
        )
        capture_ownership = _finalize_capture(validated_root, temporary, target, raw_sha)
        connection.commit()
    except Exception:
        connection.rollback()
        temporary.unlink(missing_ok=True)
        if capture_ownership is not None:
            _remove_published_capture_if_owned(validated_root, target, capture_ownership)
        raise
    return observation_id


def persist_release(connection: sqlite3.Connection, release: GitHubRelease, repository_id: str) -> str:
    if connection.in_transaction:
        raise sqlite3.IntegrityError("release persistence requires no active transaction")
    release_id = str(uuid4())
    try:
        connection.execute("BEGIN IMMEDIATE")
        parent = connection.execute(
            "SELECT github_id, owner, name FROM github_repositories WHERE id = ?", (repository_id,)
        ).fetchone()
        if parent is None or parent[0] != release.repository_github_id:
            raise sqlite3.IntegrityError("release repository identity did not match parent")
        if not _release_url_matches(release.html_url, release.tag_name, owner=parent[1], name=parent[2]):
            raise sqlite3.IntegrityError("release URL did not match parent repository")
        existing = connection.execute(
            "SELECT id, repository_id FROM github_releases WHERE github_release_id = ?",
            (release.github_release_id,),
        ).fetchone()
        if existing is not None and existing[1] != repository_id:
            raise sqlite3.IntegrityError("release is already associated with another repository")
        connection.execute(
            """INSERT INTO github_releases (id, repository_id, github_release_id, tag_name, name, body, html_url, published_at, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(github_release_id) DO UPDATE SET tag_name=excluded.tag_name, name=excluded.name, body=excluded.body, html_url=excluded.html_url, published_at=excluded.published_at""",
            (release_id, repository_id, release.github_release_id, release.tag_name, release.name, release.body, release.html_url, release.published_at, normalize_timestamp(datetime.now(_UTC))),
        )
        row = connection.execute(
            "SELECT id FROM github_releases WHERE github_release_id = ?", (release.github_release_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("release upsert did not return an id")
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return str(row[0])
