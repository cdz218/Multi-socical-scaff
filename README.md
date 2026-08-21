# Multi-Channel

Multi-Channel is a repository-local Python pipeline foundation for evidence-based,
multi-channel social content. The implemented scope is the Milestone 0/1
foundation, not a complete content-generation or publishing pipeline.

## Current capabilities

- Locked Python 3.11-3.12 package.
- Repository-local Kokoro cache verification.
- SQLite migrations and an atomic job claim/lifecycle foundation.
- Sanitized job events and explicit, idempotent requeue requests.

The branch started from a 151-test verified baseline. GitHub and Reddit adapters,
plus downstream generation and publishing, remain planned.

## Prerequisites

- Python 3.11 or 3.12
- [uv](https://docs.astral.sh/uv/)
- FFmpeg and `ffprobe` for real Kokoro and media probes

## Setup

Use an installed Python 3.11 or 3.12 interpreter:

```sh
uv sync --python <interpreter> --locked
```

## Quality checks

```sh
uv run pytest -q
uv run ruff check multichannel tests scripts
uv run mypy multichannel scripts/bootstrap_kokoro.py
uv lock --check
```

## Safe Kokoro bootstrap

Run the repository bootstrap only when a verified local Kokoro cache is needed:

```sh
uv run python scripts/bootstrap_kokoro.py
```

The cache and generated output stay under `.runtime/`. The bootstrap verifies and
caches artifacts locally; it does not install a model package into the Python
environment.

## Repository layout

- `multichannel/` - package source, SQLite lifecycle, configuration, and migrations
- `scripts/` - local bootstrap tooling
- `tests/` - offline-capable unit and integration coverage
- `.runtime/` - ignored local cache and output

## Development workflow

Branch from `main`, make changes through pull requests, and keep `.runtime/`
artifacts and secrets out of commits.

Repository: https://github.com/cdz218/Multi-socical-scaff
