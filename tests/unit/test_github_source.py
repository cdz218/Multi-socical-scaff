from __future__ import annotations

import base64
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import pytest_httpx

from multichannel.sources.base import SourceObservationInput, SourceRequestError
from multichannel.sources.github import (
    GitHubAdapter,
    GitHubRepository,
    GitHubSearchResult,
    normalize_timestamp,
    persist_github_repository,
    persist_observation,
)
from multichannel.db import connect, migrate


OBSERVED_AT = datetime(2026, 8, 21, 4, 5, 6, tzinfo=timezone.utc)
REPOSITORY_URL = "https://github.com/acme/widgets"
API_URL = "https://api.github.com/repos/acme/widgets"


def _search_row(
    *, github_id: int = 42, owner: str = "acme", name: str = "widgets"
) -> dict[str, object]:
    return {
        "id": github_id,
        "owner": {"login": owner},
        "name": name,
        "html_url": REPOSITORY_URL,
        "url": API_URL,
        "default_branch": "main",
        "stargazers_count": 12,
        "forks_count": 3,
        "open_issues_count": 4,
        "watchers_count": 5,
        "pushed_at": "2026-08-20T01:02:03Z",
        "updated_at": "2026-08-20T04:05:06+00:00",
    }


def _repository_payload() -> dict[str, object]:
    payload = _search_row()
    payload.update(
        {
            "description": None,
            "language": None,
            "license": None,
            "topics": ["widgets", "automation"],
            "created_at": "2025-01-01T00:00:00Z",
        }
    )
    return payload


def _result(
    *,
    github_id: str = "42",
    owner: str = "acme",
    name: str = "widgets",
    pushed_at: datetime = OBSERVED_AT,
    updated_at: datetime = OBSERVED_AT,
) -> GitHubSearchResult:
    return GitHubSearchResult(
        github_id=github_id, owner=owner, name=name, html_url=REPOSITORY_URL, api_url=API_URL,
        default_branch="main", stars=12, forks=3, open_issues=4, watchers=5,
        pushed_at=pushed_at, updated_at=updated_at,
    )


def test_search_parses_complete_rows_deduplicates_and_preserves_incomplete_results(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/search/repositories?q=python&per_page=25&page=1",
        json={"incomplete_results": True, "items": [_search_row(), _search_row()]},
        headers={"X-RateLimit-Remaining": "17"},
    )
    adapter = GitHubAdapter(search_limit=25)

    results = adapter.search("python", OBSERVED_AT)

    assert results == [
        _result(
            github_id="42",
            owner="acme",
            name="widgets",
            pushed_at=datetime(2026, 8, 20, 1, 2, 3, tzinfo=timezone.utc),
            updated_at=datetime(2026, 8, 20, 4, 5, 6, tzinfo=timezone.utc),
        )
    ]
    assert adapter.last_search_observation is not None
    assert adapter.last_search_observation.incomplete_results is True
    assert adapter.last_search_observation.rate_limit_remaining == 17


def test_search_skips_malformed_sibling_and_retains_valid_result(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/search/repositories?q=python&per_page=20&page=1",
        json={"incomplete_results": False, "items": [{"id": 1}, _search_row()]},
    )

    assert GitHubAdapter().search("python", OBSERVED_AT)[0].github_id == "42"


def test_search_deduplicates_distinct_ids_with_same_normalized_owner_and_name(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/search/repositories?q=python&per_page=20&page=1",
        json={
            "incomplete_results": False,
            "items": [
                _search_row(github_id=42, owner="Acme", name="Widgets"),
                _search_row(github_id=99, owner="acme", name="widgets"),
            ],
        },
    )

    results = GitHubAdapter().search("python", OBSERVED_AT)

    assert [result.github_id for result in results] == ["42"]


def test_enrich_uses_exact_paths_decodes_readme_and_allows_missing_release(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    readme = b"# Widgets\n"
    httpx_mock.add_response(url=API_URL, json=_repository_payload(), headers={"X-RateLimit-Remaining": "10"})
    httpx_mock.add_response(
        url=f"{API_URL}/readme",
        json={
            "content": base64.b64encode(readme).decode("ascii"),
            "url": f"{API_URL}/readme",
            "html_url": f"{REPOSITORY_URL}/blob/main/README.md",
            "sha": "abc123",
        },
    )
    httpx_mock.add_response(url=f"{API_URL}/releases/latest", status_code=404, json={"message": "Not Found"})
    result = _result(
        github_id="42",
        owner="acme",
        name="widgets",
        pushed_at=datetime(2026, 8, 20, 1, 2, 3, tzinfo=timezone.utc),
        updated_at=datetime(2026, 8, 20, 4, 5, 6, tzinfo=timezone.utc),
    )

    enrichment = GitHubAdapter().enrich(result, OBSERVED_AT)

    assert enrichment.repository.readme_text == "# Widgets\n"
    assert enrichment.repository.readme_sha256 == "91033039d0689273bec818c5191cee3317d30b523c1e737d34dd474e3ae76356"
    assert enrichment.release is None
    assert len(enrichment.observations) == 3
    assert enrichment.observations[0].rate_limit_remaining == 10


def test_enrich_keeps_nullable_release_fields_and_bounds_malformed_release(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    for url, payload in (
        (API_URL, _repository_payload()),
        (f"{API_URL}/readme", {"content": "", "url": f"{API_URL}/readme", "sha": "abc"}),
        (f"{API_URL}/releases/latest", {"id": "not-a-number"}),
    ):
        httpx_mock.add_response(url=url, json=payload)
    result = GitHubAdapter()
    search_result = _result(
        github_id="42", owner="acme", name="widgets", pushed_at=OBSERVED_AT, updated_at=OBSERVED_AT
    )

    enrichment = result.enrich(search_result, OBSERVED_AT)

    assert enrichment.release is None
    assert enrichment.parse_failures == ("latest release response was malformed",)


def test_preflight_and_http_error_classification(httpx_mock: pytest_httpx.HTTPXMock) -> None:
    httpx_mock.add_response(url="https://api.github.com/rate_limit", json={"resources": {}})
    assert GitHubAdapter().preflight().state == "ready_direct"
    assert "unauthenticated read-only" in GitHubAdapter()._capability_reason(False)

    for status_code, outcome in ((401, "credentials_blocked"), (403, "credentials_blocked")):
        httpx_mock.add_response(url="https://api.github.com/rate_limit", status_code=status_code)
        capability_or_error = GitHubAdapter().preflight()
        assert capability_or_error.state == outcome

    for status_code, outcome in ((429, "deferred"), (500, "failed")):
        httpx_mock.add_response(url="https://api.github.com/rate_limit", status_code=status_code)
        with pytest.raises(SourceRequestError) as raised:
            GitHubAdapter().preflight()
        assert raised.value.error.outcome == outcome

    for status_code, outcome in ((429, "deferred"), (500, "failed")):
        httpx_mock.add_response(
            url="https://api.github.com/search/repositories?q=python&per_page=20&page=1", status_code=status_code
        )
        with pytest.raises(SourceRequestError) as raised:
            GitHubAdapter().search("python", OBSERVED_AT)
        assert raised.value.error.outcome == outcome
        assert raised.value.error.status_code == status_code


def test_timeout_is_a_sanitized_failed_source_error(httpx_mock: pytest_httpx.HTTPXMock) -> None:
    token = "github_pat_this_must_not_be_persisted"
    httpx_mock.add_exception(
        httpx.ReadTimeout(f"token={token}"),
        url="https://api.github.com/search/repositories?q=python&per_page=20&page=1",
    )

    with pytest.raises(SourceRequestError) as raised:
        GitHubAdapter(token=token).search("python", OBSERVED_AT)

    assert raised.value.error.outcome == "failed"
    assert token not in raised.value.error.detail
    assert token not in str(raised.value)


def test_configured_authorization_is_sent_only_as_a_request_header(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    token = "github_pat_this_must_not_be_persisted"
    httpx_mock.add_response(url="https://api.github.com/rate_limit", json={"resources": {}})

    with GitHubAdapter(token=token) as adapter:
        assert adapter.preflight().state == "ready_direct"

    request = httpx_mock.get_requests()[0]
    assert request.headers["Authorization"] == f"Bearer {token}"
    assert token not in str(request.url)


def test_adapter_context_manager_closes_owned_client() -> None:
    adapter = GitHubAdapter()
    client = adapter._client

    with adapter:
        assert not client.is_closed

    assert client.is_closed


@pytest.mark.parametrize("value", ["2026-08-21T04:05:06Z", "2026-08-21T11:05:06+07:00"])
def test_normalize_timestamp_returns_canonical_utc_z(value: str) -> None:
    assert normalize_timestamp(datetime.fromisoformat(value.replace("Z", "+00:00"))) == "2026-08-21T04:05:06Z"


def test_invalid_repository_urls_are_rejected(httpx_mock: pytest_httpx.HTTPXMock) -> None:
    bad = _search_row()
    bad["html_url"] = "relative/path"
    httpx_mock.add_response(
        url="https://api.github.com/search/repositories?q=python&per_page=20&page=1",
        json={"incomplete_results": False, "items": [bad]},
    )

    assert GitHubAdapter().search("python", OBSERVED_AT) == []


def test_authorization_is_not_persisted_in_capture_or_observation(tmp_path: Path) -> None:
    token = "github_pat_this_must_not_be_persisted"
    repository = GitHubRepository(
        github_id="42", owner="acme", name="widgets", canonical_url=REPOSITORY_URL, api_url=API_URL,
        default_branch="main", description=None, language=None, license_spdx=None, topics=(),
        readme_url=None, readme_ref=None, readme_text=None, readme_sha256=None,
        created_at="2026-08-20T00:00:00Z",
    )
    connection = connect(tmp_path / "state.sqlite3")
    migrate(connection)
    repository_id = persist_github_repository(connection, repository)
    persist_observation(
        connection,
        SourceObservationInput("github_repository", token, OBSERVED_AT, {}, b'{"id":42}'),
        tmp_path / ".runtime" / "captures",
        repository_id=repository_id,
    )
    database_text = "\n".join(str(row) for row in connection.iterdump())
    capture_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / ".runtime" / "captures").rglob("*")
        if path.is_file()
    )
    assert token not in database_text
    assert token not in capture_text
