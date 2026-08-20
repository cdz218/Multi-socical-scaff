from __future__ import annotations

import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path


def test_locked_kokoro_toolchain_is_importable(tmp_path: Path) -> None:
    lock = tomllib.loads((Path(__file__).parents[2] / "uv.lock").read_text(encoding="utf-8"))
    locked_versions = {
        package["name"]: package["version"]
        for package in lock["package"]
        if package["name"] in {"kokoro", "soundfile"}
    }
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                "import importlib.metadata as metadata, json, kokoro, soundfile; "
                "print(json.dumps({name: metadata.distribution(name).version "
                "for name in ('kokoro', 'soundfile')}))"
            ),
        ],
        capture_output=True,
        check=False,
        cwd=tmp_path,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == locked_versions == {"kokoro": "0.9.4", "soundfile": "0.14.0"}
