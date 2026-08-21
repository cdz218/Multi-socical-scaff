from __future__ import annotations

import base64
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
import pytest_httpx

from multichannel.sources.base import SourceObservationInput, SourceRequestError
from multichannel.sources.github import (
    GitHubAdapter,
    GitHubRelease,
    GitHubRepository,
    GitHubSearchResult,
    normalize_timestamp,
    persist_github_repository,
    persist_observation,
    write_capture,
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


def _release() -> GitHubRelease:
    return GitHubRelease(
        github_release_id="7",
        repository_github_id="42",
        tag_name="v1.0.0",
        name=None,
        body=None,
        html_url=f"{REPOSITORY_URL}/releases/tag/v1.0.0",
        published_at="2026-08-20T00:00:00Z",
    )


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
    assert len(enrichment.observations) == 2
    assert enrichment.observations[0].rate_limit_remaining == 10


def test_enrich_decodes_ascii_whitespace_wrapped_base64_readme(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    readme = b"# Wrapped Widgets\n"
    httpx_mock.add_response(url=API_URL, json=_repository_payload())
    httpx_mock.add_response(
        url=f"{API_URL}/readme",
        json={
            "encoding": "base64",
            "content": base64.encodebytes(readme).decode("ascii"),
            "html_url": f"{REPOSITORY_URL}/blob/main/README.md",
            "sha": "abc123",
        },
    )
    httpx_mock.add_response(url=f"{API_URL}/releases/latest", status_code=404)

    enrichment = GitHubAdapter().enrich(_result(), OBSERVED_AT)

    assert enrichment.repository.readme_text == "# Wrapped Widgets\n"
    assert enrichment.parse_failures == ()


@pytest.mark.parametrize(
    "readme_payload",
    [
        {"encoding": "utf-8", "content": "I2Jyb2tlbgo="},
        {"encoding": None, "content": "I2Jyb2tlbgo="},
        {"encoding": "base64", "content": "I2Jyb2tlbgo=!"},
        {"encoding": "base64", "content": "I2Jyb2tlbgo=\v"},
        {"encoding": "base64", "content": "I2Jyb2tlbgo=é"},
    ],
)
def test_enrich_rejects_unsupported_or_invalid_readme_base64(
    httpx_mock: pytest_httpx.HTTPXMock, readme_payload: dict[str, object]
) -> None:
    httpx_mock.add_response(url=API_URL, json=_repository_payload())
    httpx_mock.add_response(url=f"{API_URL}/readme", json=readme_payload)
    httpx_mock.add_response(url=f"{API_URL}/releases/latest", status_code=404)

    enrichment = GitHubAdapter().enrich(_result(), OBSERVED_AT)

    assert enrichment.repository.readme_text is None
    assert enrichment.repository.readme_sha256 is None
    assert enrichment.parse_failures == ("README response was malformed",)


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_enrich_allows_missing_readme_and_retains_its_repository_observation(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    httpx_mock.add_response(url=API_URL, json=_repository_payload())
    httpx_mock.add_response(url=f"{API_URL}/readme", status_code=404, json={"message": "Not Found"})
    httpx_mock.add_response(url=f"{API_URL}/releases/latest", status_code=404, json={"message": "Not Found"})

    enrichment = GitHubAdapter().enrich(_result(), OBSERVED_AT)

    assert enrichment.repository.readme_url is None
    assert enrichment.repository.readme_ref is None
    assert enrichment.repository.readme_text is None
    assert enrichment.repository.readme_sha256 is None
    assert enrichment.parse_failures == ()
    assert [observation.source_kind for observation in enrichment.observations] == [
        "github_repository",
        "github_repository",
    ]
    assert [request.url.path for request in httpx_mock.get_requests()] == [
        "/repos/acme/widgets",
        "/repos/acme/widgets/readme",
        "/repos/acme/widgets/releases/latest",
    ]


def test_enrich_rejects_non_404_readme_errors(httpx_mock: pytest_httpx.HTTPXMock) -> None:
    httpx_mock.add_response(url=API_URL, json=_repository_payload())
    httpx_mock.add_response(url=f"{API_URL}/readme", status_code=500)

    with pytest.raises(SourceRequestError) as raised:
        GitHubAdapter().enrich(_result(), OBSERVED_AT)

    assert raised.value.error.endpoint_path == "/repos/acme/widgets/readme"


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
    assert len(enrichment.observations) == 2


def test_enrich_exposes_a_release_observation_only_for_a_parsed_release(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    for url, payload in (
        (API_URL, _repository_payload()),
        (f"{API_URL}/readme", {"content": "", "url": f"{API_URL}/readme", "sha": "abc"}),
        (f"{API_URL}/releases/latest", {"id": 7, "tag_name": "v1.0.0", "html_url": f"{REPOSITORY_URL}/releases/tag/v1.0.0"}),
    ):
        httpx_mock.add_response(url=url, json=payload)

    enrichment = GitHubAdapter().enrich(_result(), OBSERVED_AT)

    assert enrichment.release is not None
    assert [observation.source_kind for observation in enrichment.observations] == [
        "github_repository", "github_repository", "github_release"
    ]


def test_enrich_accepts_a_repository_bound_url_with_an_encoded_release_tag(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    for url, payload in (
        (API_URL, _repository_payload()),
        (f"{API_URL}/readme", {"content": ""}),
        (
            f"{API_URL}/releases/latest",
            {
                "id": 7,
                "tag_name": "release/2026",
                "html_url": f"{REPOSITORY_URL}/releases/tag/release%2F2026",
            },
        ),
    ):
        httpx_mock.add_response(url=url, json=payload)

    enrichment = GitHubAdapter().enrich(_result(), OBSERVED_AT)

    assert enrichment.release is not None
    assert enrichment.release.html_url == f"{REPOSITORY_URL}/releases/tag/release%2F2026"


@pytest.mark.parametrize(
    "release_url",
    [
        "https://github.com/other/project/releases/tag/v1.0.0",
        "https://github.com/acme%2Fother/widgets/releases/tag/v1.0.0",
    ],
)
def test_enrich_rejects_release_url_not_bound_to_the_requested_repository(
    httpx_mock: pytest_httpx.HTTPXMock, release_url: str
) -> None:
    for url, payload in (
        (API_URL, _repository_payload()),
        (f"{API_URL}/readme", {"content": ""}),
        (
            f"{API_URL}/releases/latest",
            {"id": 7, "tag_name": "v1.0.0", "html_url": release_url},
        ),
    ):
        httpx_mock.add_response(url=url, json=payload)

    enrichment = GitHubAdapter().enrich(_result(), OBSERVED_AT)

    assert enrichment.release is None
    assert enrichment.parse_failures == ("latest release response was malformed",)
    assert [observation.source_kind for observation in enrichment.observations] == [
        "github_repository",
        "github_repository",
    ]


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


@pytest.mark.parametrize("status_code", [400, 404, 405, 409, 422])
def test_preflight_fails_closed_for_unhandled_client_errors(
    httpx_mock: pytest_httpx.HTTPXMock, status_code: int
) -> None:
    httpx_mock.add_response(url="https://api.github.com/rate_limit", status_code=status_code)

    with pytest.raises(SourceRequestError) as raised:
        GitHubAdapter().preflight()

    assert raised.value.error.outcome == "failed"
    assert raised.value.error.status_code == status_code
    assert raised.value.error.endpoint_path == "/rate_limit"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://attacker.invalid",
        "https://user:password@api.github.com",
        "https://api.github.com?credential=secret",
        "https://api.github.com#credential",
        "not-an-absolute-url",
    ],
)
def test_adapter_rejects_unsafe_base_url_before_creating_a_request(base_url: str) -> None:
    with pytest.raises(ValueError, match="base URL"):
        GitHubAdapter(token="synthetic-secret", base_url=base_url)


def test_redirects_fail_closed_without_cross_origin_authorization(
    httpx_mock: pytest_httpx.HTTPXMock,
) -> None:
    httpx_mock.add_response(
        url="https://api.github.com/rate_limit",
        status_code=302,
        headers={"Location": "https://attacker.invalid/rate_limit"},
    )

    with pytest.raises(SourceRequestError) as raised:
        GitHubAdapter(token="synthetic-secret").preflight()

    assert raised.value.error.status_code == 302
    assert len(httpx_mock.get_requests()) == 1
    assert httpx_mock.get_requests()[0].url.host == "api.github.com"


@pytest.mark.parametrize(
    "error_type",
    [httpx.RemoteProtocolError, httpx.ReadError, httpx.PoolTimeout],
)
def test_request_errors_are_sanitized_failed_source_errors(
    httpx_mock: pytest_httpx.HTTPXMock, error_type: type[httpx.RequestError]
) -> None:
    token = "github_pat_this_must_not_be_persisted"
    profile = "private-profile-credential"
    httpx_mock.add_exception(
        error_type(
            f"https://api.github.com/search/repositories?q={profile} authorization=Bearer {token}"
        ),
        url=f"https://api.github.com/search/repositories?q={profile}&per_page=20&page=1",
    )

    with pytest.raises(SourceRequestError) as raised:
        GitHubAdapter(token=token).search(profile, OBSERVED_AT)

    assert raised.value.error.outcome == "failed"
    assert raised.value.error.status_code is None
    assert raised.value.error.endpoint_path == "/search/repositories"
    assert token not in raised.value.error.detail
    assert profile not in raised.value.error.detail
    assert token not in str(raised.value)
    assert profile not in str(raised.value)
    assert raised.value.error.detail == "GitHub network request failed"


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


@pytest.mark.parametrize(
    "payload",
    [
        {"items": "not-a-list", "incomplete_results": False},
        {"items": [], "incomplete_results": "false"},
        {"incomplete_results": False},
        [],
    ],
)
def test_search_rejects_malformed_top_level_payload_without_recording_an_observation(
    httpx_mock: pytest_httpx.HTTPXMock, payload: object
) -> None:
    adapter = GitHubAdapter()
    sentinel = SourceObservationInput("github_repository", "existing", OBSERVED_AT, {}, b"existing")
    adapter.last_search_observation = sentinel
    httpx_mock.add_response(
        url="https://api.github.com/search/repositories?q=private-profile-credential&per_page=20&page=1",
        json=payload,
    )

    with pytest.raises(SourceRequestError) as raised:
        adapter.search("private-profile-credential", OBSERVED_AT)

    assert raised.value.error.detail == "malformed search response"
    assert "private-profile-credential" not in str(raised.value)
    assert adapter.last_search_observation is sentinel


@pytest.mark.parametrize(
    "field,value",
    [
        ("github_id", ""),
        ("owner", ""),
        ("name", "widgets/escape"),
        ("html_url", "https://github.com/acme/other"),
        ("api_url", "https://api.github.com/repos/acme/other"),
        ("pushed_at", datetime(2026, 8, 21, 4, 5, 6)),
    ],
)
def test_search_result_rejects_invalid_direct_construction(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        replace(_result(), **{field: value})


@pytest.mark.parametrize(
    "field,value",
    [("readme_sha256", "not-a-hash"), ("created_at", "2026-08-20T00:00:00+00:00")],
)
def test_repository_rejects_invalid_direct_construction(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        GitHubRepository(
            github_id="42", owner="acme", name="widgets", canonical_url=REPOSITORY_URL,
            api_url=API_URL, default_branch="main", description=None, language=None,
            license_spdx=None, topics=(), readme_url=None, readme_ref=None, readme_text=None,
            readme_sha256=value if field == "readme_sha256" else None,
            created_at=value if field == "created_at" else "2026-08-20T00:00:00Z",
        )


def test_release_rejects_invalid_direct_construction() -> None:
    with pytest.raises(ValueError):
        GitHubRelease("", "42", "v1.0.0", None, None, f"{REPOSITORY_URL}/releases/tag/v1", None)


@pytest.mark.parametrize("invalid_id", ["github-id", "+42", " 42", "42 ", "04"])
def test_models_require_canonical_decimal_github_ids(invalid_id: str) -> None:
    with pytest.raises(ValueError, match="canonical decimal"):
        replace(_result(), github_id=invalid_id)
    with pytest.raises(ValueError, match="canonical decimal"):
        GitHubRepository(
            github_id=invalid_id, owner="acme", name="widgets", canonical_url=REPOSITORY_URL,
            api_url=API_URL, default_branch="main", description=None, language=None,
            license_spdx=None, topics=(), readme_url=None, readme_ref=None, readme_text=None,
            readme_sha256=None, created_at="2026-08-20T00:00:00Z",
        )
    with pytest.raises(ValueError, match="canonical decimal"):
        GitHubRelease(
            invalid_id, "42", "v1.0.0", None, None,
            f"{REPOSITORY_URL}/releases/tag/v1.0.0", None,
        )
    with pytest.raises(ValueError, match="canonical decimal"):
        GitHubRelease(
            "7", invalid_id, "v1.0.0", None, None,
            f"{REPOSITORY_URL}/releases/tag/v1.0.0", None,
        )


def test_models_accept_canonical_decimal_github_ids() -> None:
    assert replace(_result(), github_id="0").github_id == "0"
    assert GitHubRepository(
        github_id="42", owner="acme", name="widgets", canonical_url=REPOSITORY_URL,
        api_url=API_URL, default_branch="main", description=None, language=None,
        license_spdx=None, topics=(), readme_url=None, readme_ref=None, readme_text=None,
        readme_sha256=None, created_at="2026-08-20T00:00:00Z",
    ).github_id == "42"
    assert _release().github_release_id == "7"


@pytest.mark.parametrize("mismatch", [("id", 99), ("owner", {"login": "other"})])
def test_enrich_rejects_repository_response_identity_mismatches(
    httpx_mock: pytest_httpx.HTTPXMock, mismatch: tuple[str, object]
) -> None:
    payload = _repository_payload()
    payload[mismatch[0]] = mismatch[1]
    if mismatch[0] == "owner":
        payload["html_url"] = "https://github.com/other/widgets"
        payload["url"] = "https://api.github.com/repos/other/widgets"
    httpx_mock.add_response(url=API_URL, json=payload)

    with pytest.raises(SourceRequestError) as raised:
        GitHubAdapter().enrich(_result(), OBSERVED_AT)

    assert raised.value.error.endpoint_path == "/repos/acme/widgets"
    assert raised.value.error.detail == "repository response identity did not match request"
    assert len(httpx_mock.get_requests()) == 1


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


def test_write_capture_rejects_substituted_existing_content_and_reuses_identical_content(
    tmp_path: Path,
) -> None:
    captures = tmp_path / ".runtime" / "captures"
    observation = SourceObservationInput(
        "github_repository", "github:42:repository", OBSERVED_AT, {}, b'{"id":42}'
    )
    relative_path = write_capture(captures, observation)
    capture = captures / relative_path

    assert write_capture(captures, observation) == relative_path
    assert capture.read_bytes() == observation.raw_bytes

    capture.write_bytes(b"substituted")
    with pytest.raises(ValueError, match="hash"):
        write_capture(captures, observation)
    assert capture.read_bytes() == b"substituted"
