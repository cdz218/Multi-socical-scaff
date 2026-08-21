from __future__ import annotations

import os

import pytest

from multichannel.sources.github import GitHubAdapter


pytestmark = pytest.mark.skipif(
    os.environ.get("MULTICHANNEL_CONTRACT_TESTS") != "1",
    reason="set MULTICHANNEL_CONTRACT_TESTS=1 to run the read-only GitHub contract check",
)


def test_github_rate_limit_contract() -> None:
    capability = GitHubAdapter(
        base_url=os.environ.get("MULTICHANNEL_GITHUB_API_URL", "https://api.github.com"),
        token=os.environ.get("GITHUB_TOKEN"),
    ).preflight()

    assert capability.state in {"ready_direct", "credentials_blocked"}
