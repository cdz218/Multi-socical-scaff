"""Repository-local runtime configuration and safe diagnostics."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class ConfigurationError(ValueError):
    """Raised when configuration violates a local safety boundary."""


@dataclass(frozen=True)
class RuntimePaths:
    repository: Path
    runtime: Path

    @classmethod
    def from_repository(cls, repository: Path) -> "RuntimePaths":
        resolved_repository = repository.resolve()
        runtime = resolved_repository / ".runtime"
        resolved_runtime = runtime.resolve(strict=False)
        models = runtime / "models"
        if not resolved_runtime.is_relative_to(resolved_repository) or not models.resolve(
            strict=False
        ).is_relative_to(resolved_runtime):
            raise ConfigurationError("model cache path resolves outside repository-local .runtime")
        return cls(repository=resolved_repository, runtime=runtime)

    @property
    def models(self) -> Path:
        return self.runtime / "models"

    def resolve_model_cache(self, relative: str) -> Path:
        candidate = self.models / relative
        resolved = candidate.resolve(strict=False)
        models_root = self.models.resolve(strict=False)
        if not models_root.is_relative_to(self.runtime.resolve(strict=False)) or not resolved.is_relative_to(
            models_root
        ):
            raise ConfigurationError("model cache path resolves outside .runtime/models")
        return resolved


def diagnostic_report(environment: Mapping[str, str]) -> str:
    """Return structured diagnostics, exposing values only from the public allowlist."""
    public_keys = frozenset({"LANG", "LC_ALL", "LC_CTYPE", "SAFE", "TERM", "TZ"})
    redacted = {
        key: value if key in public_keys else "[REDACTED]"
        for key, value in sorted(environment.items())
    }
    return json.dumps({"environment": redacted}, sort_keys=True, separators=(",", ":"))


def find_interpreter(candidates: Sequence[Path]) -> Path:
    for candidate in candidates:
        try:
            result = subprocess.run(
                [str(candidate), "--version"], capture_output=True, check=False, text=True
            )
        except OSError:
            continue
        version = result.stdout.strip() or result.stderr.strip()
        if result.returncode == 0 and version.startswith("Python 3."):
            parts = version.removeprefix("Python ").split(".")
            if len(parts) >= 2 and parts[0] == "3" and 11 <= int(parts[1]) < 13:
                return candidate
    raise ConfigurationError("an interpreter satisfying >=3.11,<3.13 is required")
