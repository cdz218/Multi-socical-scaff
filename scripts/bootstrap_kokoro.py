"""Download and verify the repository-local Kokoro cache."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import stat
import sys
import urllib.request
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, cast


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from multichannel.config import RuntimePaths  # noqa: E402


KOKORO_REPOSITORY = "hexgrad/Kokoro-82M"
KOKORO_REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"
KOKORO_VERSION = "0.9.4"
VOICE = "af_heart"
SAMPLE_RATE = 24000
ENGLISH_MODEL_PACKAGE = "en_core_web_sm"
ENGLISH_MODEL_VERSION = "3.8.0"
ENGLISH_MODEL_WHEEL = f"{ENGLISH_MODEL_PACKAGE}-{ENGLISH_MODEL_VERSION}-py3-none-any.whl"
ENGLISH_MODEL_ORIGIN = (
    "https://github.com/explosion/spacy-models/releases/download/"
    f"{ENGLISH_MODEL_PACKAGE}-{ENGLISH_MODEL_VERSION}/{ENGLISH_MODEL_WHEEL}"
)
ENGLISH_MODEL_WHEEL_SHA256 = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
APPROVED_ARTIFACT_SHA256 = {
    "model_metadata": "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
    "model_weights": "496dba118d1a58f5f3db2efc88dbdc216e0483fc89fe6e47ee1f2c53f18ad1e4",
    "language_voice": "0ab5709b8ffab19bfd849cd11d98f75b60af7733253ad0d67b12382a102cb4ff",
    "language_model": ENGLISH_MODEL_WHEEL_SHA256,
}


def required_artifacts() -> dict[str, str]:
    return {
        "model_metadata": "config.json",
        "model_weights": "kokoro-v1_0.pth",
        "language_voice": f"voices/{VOICE}.pt",
        "language_model": ENGLISH_MODEL_WHEEL,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_sha256(directory: Path) -> str:
    """Hash extracted model files in a stable order so cache tampering is detected."""
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def artifact_details(artifact: str) -> tuple[str, str]:
    if artifact == "language_model":
        return ENGLISH_MODEL_VERSION, ENGLISH_MODEL_ORIGIN
    return KOKORO_VERSION, KOKORO_REPOSITORY


def repository_root() -> Path:
    """Return the repository owning this script, independently of caller CWD."""
    return REPOSITORY_ROOT


def language_model_directory(root: Path) -> Path:
    return root / "language-model" / ENGLISH_MODEL_PACKAGE / (
        f"{ENGLISH_MODEL_PACKAGE}-{ENGLISH_MODEL_VERSION}"
    )


def artifact_manifest(root: Path, artifacts: Mapping[str, Path]) -> dict[str, Any]:
    expected = required_artifacts()
    manifest_artifacts: list[dict[str, Any]] = []
    for artifact, path in sorted(artifacts.items()):
        version, origin = artifact_details(artifact)
        item: dict[str, Any] = {
            "artifact": artifact,
            "name": expected[artifact],
            "path": str(path.relative_to(root)),
            "version": version,
            "origin": origin,
            "sha256": sha256(path),
        }
        if artifact != "language_model":
            item["revision"] = KOKORO_REVISION
        if artifact == "language_model":
            item["extracted_sha256"] = directory_sha256(language_model_directory(root))
        manifest_artifacts.append(item)
    return {
        "kokoro_version": KOKORO_VERSION,
        "artifacts": manifest_artifacts,
        "verification_wav_sha256": sha256(root / "verification.wav"),
    }


def _cache_path(root: Path, relative: str) -> Path:
    candidate = root / relative
    if not candidate.resolve(strict=False).is_relative_to(root.resolve()):
        raise RuntimeError("Kokoro verification manifest path escapes the cache")
    return candidate


def _verify_language_model(root: Path, item: Mapping[str, Any]) -> None:
    directory = language_model_directory(root)
    metadata = directory / "meta.json"
    if not directory.is_dir() or not metadata.is_file():
        raise RuntimeError("Kokoro cached language model is missing")
    try:
        model_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("Kokoro cached language model metadata is invalid") from error
    if (
        model_metadata.get("version") != ENGLISH_MODEL_VERSION
        or item.get("extracted_sha256") != directory_sha256(directory)
    ):
        raise RuntimeError("Kokoro cached language model does not match the verified wheel")


def verify_cache(root: Path) -> None:
    manifest_path = root / "verification.json"
    wav_path = root / "verification.wav"
    resolved_root = root.resolve()
    if wav_path.is_symlink() or not wav_path.resolve(strict=False).is_relative_to(resolved_root):
        raise RuntimeError("Kokoro verification WAV escapes the cache")
    if not manifest_path.is_file() or not wav_path.is_file():
        raise RuntimeError("Kokoro verification manifest or WAV is missing")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("Kokoro verification manifest is invalid") from error
    if not isinstance(manifest, dict):
        raise RuntimeError("Kokoro verification manifest is invalid")
    artifacts = manifest.get("artifacts")
    wav_digest = manifest.get("verification_wav_sha256")
    if (
        manifest.get("kokoro_version") != KOKORO_VERSION
        or not isinstance(artifacts, list)
    ):
        raise RuntimeError("Kokoro verification manifest is invalid")
    expected = required_artifacts()
    actual = {item.get("artifact"): item for item in artifacts if isinstance(item, dict)}
    if set(actual) != set(expected):
        raise RuntimeError("Kokoro required artifacts are missing or unexpected")
    for item in artifacts:
        if not isinstance(item, dict):
            raise RuntimeError("Kokoro verification manifest is invalid")
        artifact = item.get("artifact")
        path_name = item.get("path")
        if (
            not isinstance(artifact, str)
            or artifact not in expected
            or not isinstance(path_name, str)
        ):
            raise RuntimeError("Kokoro verification manifest is invalid")
        path = _cache_path(root, path_name)
        version, origin = artifact_details(artifact)
        if (
            item.get("name") != expected[artifact]
            or item.get("version") != version
            or item.get("origin") != origin
            or (artifact != "language_model" and item.get("revision") != KOKORO_REVISION)
            or not path.is_file()
            or sha256(path) != item.get("sha256")
            or sha256(path) != APPROVED_ARTIFACT_SHA256[artifact]
        ):
            raise RuntimeError("Kokoro verification manifest does not match approved cached artifacts")
        if artifact == "language_model":
            _verify_language_model(root, item)

    if not isinstance(wav_digest, str):
        raise RuntimeError("Kokoro verification manifest is missing the verification WAV digest")
    if sha256(wav_path) != wav_digest:
        raise RuntimeError("Kokoro verification WAV does not match the manifest")

    import soundfile as sf  # type: ignore[import-untyped]

    info = sf.info(wav_path)
    if info.samplerate != SAMPLE_RATE or info.channels != 1:
        raise RuntimeError("Kokoro verification WAV must be 24 kHz mono")
    if info.frames <= 0:
        raise RuntimeError("Kokoro verification WAV must contain audio frames")


def download_artifacts(root: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    artifacts: dict[str, Path] = {}
    for artifact, name in required_artifacts().items():
        if artifact == "language_model":
            continue
        downloaded = Path(
            hf_hub_download(
                repo_id=KOKORO_REPOSITORY,
                filename=name,
                revision=KOKORO_REVISION,
            )
        )
        if sha256(downloaded) != APPROVED_ARTIFACT_SHA256[artifact]:
            raise RuntimeError(f"Kokoro {artifact} digest is not approved")
        artifacts[artifact] = downloaded
    return artifacts


def download_language_model(root: Path) -> Path:
    """Cache the approved wheel without involving the Python package installer."""
    resolved_root = root.resolve(strict=False)
    if root.is_symlink():
        raise RuntimeError("Kokoro cache root must not be a symlink")
    downloads = root / "downloads"
    if downloads.is_symlink():
        raise RuntimeError("Kokoro downloads directory must not be a symlink")
    downloads.mkdir(parents=True, exist_ok=True)
    resolved_downloads = downloads.resolve(strict=False)
    if not resolved_downloads.is_relative_to(resolved_root) or not downloads.is_dir():
        raise RuntimeError("Kokoro downloads directory escapes the cache")
    destination = downloads / ENGLISH_MODEL_WHEEL
    if destination.is_symlink():
        raise RuntimeError("Kokoro language model wheel must not be a symlink")
    if destination.is_file():
        if sha256(destination) != ENGLISH_MODEL_WHEEL_SHA256:
            raise RuntimeError("Kokoro language model wheel digest is not approved")
        return destination
    temporary: Path | None = None
    output_fd: int | None = None
    try:
        for _ in range(5):
            candidate = downloads / f".{ENGLISH_MODEL_WHEEL}.{uuid.uuid4().hex}.partial"
            if not candidate.resolve(strict=False).is_relative_to(resolved_root):
                raise RuntimeError("Kokoro staging path escapes the cache")
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            try:
                output_fd = os.open(candidate, flags, 0o600)
            except FileExistsError:
                continue
            temporary = candidate
            break
        if output_fd is None or temporary is None:
            raise RuntimeError("could not create a private Kokoro staging file")
        with urllib.request.urlopen(ENGLISH_MODEL_ORIGIN) as response, os.fdopen(output_fd, "wb") as output:
            output_fd = None
            shutil.copyfileobj(response, output)
            output.flush()
            os.fsync(output.fileno())
        if sha256(temporary) != ENGLISH_MODEL_WHEEL_SHA256:
            raise RuntimeError("Kokoro language model wheel digest is not approved")
        if destination.is_symlink():
            raise RuntimeError("Kokoro language model wheel must not be a symlink")
        temporary.replace(destination)
    finally:
        if output_fd is not None:
            os.close(output_fd)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def _safe_archive_member(member: zipfile.ZipInfo, destination: Path) -> None:
    member_path = Path(member.filename)
    if (
        member_path.is_absolute()
        or ".." in member_path.parts
        or not (destination / member_path).resolve(strict=False).is_relative_to(destination.resolve())
        or stat.S_ISLNK(member.external_attr >> 16)
    ):
        raise RuntimeError("Kokoro language model archive member escapes the cache")


def _validated_language_model_directory(destination: Path) -> Path:
    model_directory = destination / ENGLISH_MODEL_PACKAGE / (
        f"{ENGLISH_MODEL_PACKAGE}-{ENGLISH_MODEL_VERSION}"
    )
    metadata = model_directory / "meta.json"
    config = model_directory / "config.cfg"
    if not model_directory.is_dir() or not metadata.is_file() or not config.is_file():
        raise RuntimeError("Kokoro language model wheel is missing its model data directory")
    try:
        version = json.loads(metadata.read_text(encoding="utf-8")).get("version")
    except json.JSONDecodeError as error:
        raise RuntimeError("Kokoro language model metadata is invalid") from error
    if version != ENGLISH_MODEL_VERSION:
        raise RuntimeError("Kokoro language model wheel has an unexpected version")
    return model_directory


def _extract_language_model(wheel: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    try:
        with zipfile.ZipFile(wheel) as archive:
            for member in archive.infolist():
                _safe_archive_member(member, destination)
            archive.extractall(destination)
        return _validated_language_model_directory(destination)
    except Exception:
        shutil.rmtree(destination)
        raise


def extract_language_model(wheel: Path, destination: Path) -> Path:
    """Atomically cache the validated spaCy model data directory from a wheel."""
    staging = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}")
    replacement = destination.with_name(f".{destination.name}-replaced-{uuid.uuid4().hex}")
    extracted = _extract_language_model(wheel, staging)
    try:
        if destination.exists():
            try:
                if directory_sha256(destination) == directory_sha256(staging):
                    shutil.rmtree(staging)
                    return _validated_language_model_directory(destination)
            except RuntimeError:
                pass
            destination.replace(replacement)
        staging.replace(destination)
    except Exception:
        if replacement.exists() and not destination.exists():
            replacement.replace(destination)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(replacement, ignore_errors=True)
    return destination / extracted.relative_to(staging)


class _SpacyUtilProxy:
    def __init__(self, util: Any) -> None:
        self._util = util

    def is_package(self, name: str) -> bool:
        return name == ENGLISH_MODEL_PACKAGE or self._util.is_package(name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._util, name)


class _SpacyCacheProxy:
    def __init__(self, spacy_module: Any, model_directory: Path) -> None:
        self._spacy_module = spacy_module
        self._model_directory = model_directory
        self.util = _SpacyUtilProxy(spacy_module.util)

    def load(self, name: str | Path, **kwargs: Any) -> Any:
        if name == ENGLISH_MODEL_PACKAGE:
            return self._spacy_module.load(self._model_directory, **kwargs)
        return self._spacy_module.load(name, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._spacy_module, name)


@contextmanager
def cached_english_model(model_directory: Path) -> Iterator[None]:
    """Redirect only Misaki's local spaCy reference for this bootstrap process."""
    if not model_directory.is_dir():
        raise RuntimeError("Kokoro cached language model is missing")
    misaki_en = cast(Any, importlib.import_module("misaki.en"))
    original_spacy = misaki_en.spacy
    misaki_en.spacy = _SpacyCacheProxy(original_spacy, model_directory)
    try:
        yield
    finally:
        misaki_en.spacy = original_spacy


def main() -> int:
    paths = RuntimePaths.from_repository(repository_root())
    root = paths.resolve_model_cache("kokoro")
    # Kokoro delegates downloads to Hugging Face; keep that cache repository-local.
    previous_environment = {
        key: os.environ.get(key)
        for key in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE")
    }
    try:
        os.environ["HF_HOME"] = str(root)
        os.environ["HF_HUB_CACHE"] = str(root / "hub")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(root / "hub")
        root.mkdir(parents=True, exist_ok=True)
        try:
            verify_cache(root)
        except RuntimeError:
            pass
        else:
            return 0

        from kokoro import KPipeline  # type: ignore[import-untyped]
        from kokoro.model import KModel  # type: ignore[import-untyped]
        import soundfile as sf

        artifacts = download_artifacts(root)
        language_wheel = download_language_model(root)
        language_directory = extract_language_model(language_wheel, root / "language-model")
        artifacts["language_model"] = language_wheel
        wav_path = root / "verification.wav"
        with cached_english_model(language_directory):
            approved_model = KModel(
                config=str(artifacts["model_metadata"]),
                model=str(artifacts["model_weights"]),
            )
            pipeline = KPipeline(lang_code="a", model=approved_model)
            samples = next(
                pipeline("Kokoro bootstrap verification.", voice=str(artifacts["language_voice"]))
            )[2]
        sf.write(wav_path, samples, SAMPLE_RATE)
        manifest_path = root / "verification.json"
        manifest_path.write_text(
            json.dumps(artifact_manifest(root, artifacts), sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        verify_cache(root)
        return 0
    finally:
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    raise SystemExit(main())
