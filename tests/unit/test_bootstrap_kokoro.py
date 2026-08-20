from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import zipfile
from pathlib import Path

import pytest
import soundfile as sf


def load_bootstrap_module() -> object:
    path = Path(__file__).parents[2] / "scripts" / "bootstrap_kokoro.py"
    spec = importlib.util.spec_from_file_location("bootstrap_kokoro", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_cache_accepts_actual_required_kokoro_artifact_metadata(tmp_path: Path) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / "kokoro"
    artifacts = _write_complete_cache(root, bootstrap)

    manifest = bootstrap.artifact_manifest(root, artifacts)
    (root / "verification.json").write_text(json.dumps(manifest), encoding="utf-8")

    bootstrap.verify_cache(root)

    assert {item["artifact"] for item in manifest["artifacts"]} == {
        "model_metadata",
        "model_weights",
        "language_voice",
        "language_model",
    }
    assert {
        item["origin"] for item in manifest["artifacts"] if item["artifact"] != "language_model"
    } == {"hexgrad/Kokoro-82M"}
    language_model = next(item for item in manifest["artifacts"] if item["artifact"] == "language_model")
    assert language_model["origin"] == bootstrap.ENGLISH_MODEL_ORIGIN
    assert all(item["version"] for item in manifest["artifacts"])


def test_verify_cache_rejects_missing_required_kokoro_model_metadata(tmp_path: Path) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / "kokoro"
    root.mkdir()
    (root / "verification.json").write_text(
        json.dumps({"kokoro_version": "0.9.4", "artifacts": []}), encoding="utf-8"
    )
    (root / "verification.wav").write_bytes(b"not-a-wav")

    with pytest.raises(RuntimeError, match="required"):
        bootstrap.verify_cache(root)


def test_extract_language_model_rejects_archive_path_escape(tmp_path: Path) -> None:
    bootstrap = load_bootstrap_module()
    wheel = tmp_path / bootstrap.ENGLISH_MODEL_WHEEL
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("../../escape", "not safe")

    with pytest.raises(RuntimeError, match="escapes"):
        bootstrap.extract_language_model(wheel, tmp_path / "language-model")

    assert not (tmp_path.parent / "escape").exists()


def test_english_language_model_artifact_is_pinned() -> None:
    bootstrap = load_bootstrap_module()

    artifact = bootstrap.required_artifacts()["language_model"]

    assert artifact == bootstrap.ENGLISH_MODEL_WHEEL
    assert bootstrap.ENGLISH_MODEL_VERSION == "3.8.0"
    assert bootstrap.ENGLISH_MODEL_ORIGIN.endswith(f"/{bootstrap.ENGLISH_MODEL_WHEEL}")
    assert bootstrap.ENGLISH_MODEL_WHEEL_SHA256 == (
        "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
    )


def test_misaki_uses_cached_english_model_without_package_installer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()
    model_path = _write_extracted_language_model(tmp_path, bootstrap)
    loads: list[Path] = []
    misaki_en = __import__("misaki.en", fromlist=["G2P"])

    monkeypatch.setattr(misaki_en.spacy, "load", lambda path, **kwargs: loads.append(Path(path)) or object())
    monkeypatch.setattr(
        misaki_en.spacy.cli,
        "download",
        lambda name: pytest.fail(f"package installer called for {name}"),
    )

    with bootstrap.cached_english_model(model_path):
        misaki_en.G2P()

    assert loads == [model_path]


def test_extract_language_model_resolves_real_wheel_data_directory_and_reuses_it(
    tmp_path: Path,
) -> None:
    bootstrap = load_bootstrap_module()
    wheel = tmp_path / bootstrap.ENGLISH_MODEL_WHEEL
    _write_language_model_wheel(wheel, bootstrap)
    destination = tmp_path / "language-model"

    extracted = bootstrap.extract_language_model(wheel, destination)
    reused = bootstrap.extract_language_model(wheel, destination)

    assert extracted == destination / bootstrap.ENGLISH_MODEL_PACKAGE / (
        f"{bootstrap.ENGLISH_MODEL_PACKAGE}-{bootstrap.ENGLISH_MODEL_VERSION}"
    )
    assert reused == extracted
    assert (extracted / "config.cfg").is_file()
    assert json.loads((extracted / "meta.json").read_text(encoding="utf-8"))["version"] == (
        bootstrap.ENGLISH_MODEL_VERSION
    )


def test_extract_language_model_replaces_tampered_data_directory_atomically(tmp_path: Path) -> None:
    bootstrap = load_bootstrap_module()
    wheel = tmp_path / bootstrap.ENGLISH_MODEL_WHEEL
    _write_language_model_wheel(wheel, bootstrap)
    destination = tmp_path / "language-model"
    extracted = bootstrap.extract_language_model(wheel, destination)
    (extracted / "config.cfg").write_text("tampered", encoding="utf-8")

    replaced = bootstrap.extract_language_model(wheel, destination)

    assert replaced == extracted
    assert (replaced / "config.cfg").read_text(encoding="utf-8") == "[nlp]\nlang = \"en\"\n"
    assert not list(tmp_path.glob(".language-model-*"))


def test_main_reuses_verified_cache_without_downloading_or_rewriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / ".runtime" / "models" / "kokoro"
    artifacts = _write_complete_cache(root, bootstrap)
    (root / "verification.json").write_text(
        json.dumps(bootstrap.artifact_manifest(root, artifacts)), encoding="utf-8"
    )
    manifest_before = (root / "verification.json").read_bytes()
    wav_before = (root / "verification.wav").read_bytes()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        bootstrap,
        "download_artifacts",
        lambda _: pytest.fail("model artifacts were downloaded"),
    )
    monkeypatch.setattr(
        bootstrap,
        "download_language_model",
        lambda _: pytest.fail("language wheel was downloaded"),
    )

    assert bootstrap.main() == 0

    assert (root / "verification.json").read_bytes() == manifest_before
    assert (root / "verification.wav").read_bytes() == wav_before


def test_verify_cache_rejects_altered_valid_verification_wav(tmp_path: Path) -> None:
    bootstrap = load_bootstrap_module()
    source_root = tmp_path / "verified-kokoro"
    artifacts = _write_complete_cache(source_root, bootstrap)
    (source_root / "verification.json").write_text(
        json.dumps(bootstrap.artifact_manifest(source_root, artifacts)), encoding="utf-8"
    )
    root = tmp_path / "altered-kokoro"
    shutil.copytree(source_root, root)
    sf.write(root / "verification.wav", [0.5, -0.5], bootstrap.SAMPLE_RATE)

    with pytest.raises(RuntimeError, match="WAV"):
        bootstrap.verify_cache(root)


def test_verify_cache_rejects_manifest_without_verification_wav_digest(tmp_path: Path) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / "kokoro"
    artifacts = _write_complete_cache(root, bootstrap)
    manifest = bootstrap.artifact_manifest(root, artifacts)
    del manifest["verification_wav_sha256"]
    (root / "verification.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="WAV digest"):
        bootstrap.verify_cache(root)


def test_verify_cache_rejects_zero_frame_verification_wav(tmp_path: Path) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / "kokoro"
    artifacts = _write_complete_cache(root, bootstrap)
    sf.write(root / "verification.wav", [], bootstrap.SAMPLE_RATE)
    (root / "verification.json").write_text(
        json.dumps(bootstrap.artifact_manifest(root, artifacts)), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="audio frames"):
        bootstrap.verify_cache(root)


def test_verify_cache_rejects_malformed_manifest_as_controlled_error(tmp_path: Path) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / "kokoro"
    _write_complete_cache(root, bootstrap)
    (root / "verification.json").write_text("{not-json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="manifest is invalid"):
        bootstrap.verify_cache(root)


def test_download_language_model_reuses_existing_pinned_wheel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bootstrap = load_bootstrap_module()
    wheel = tmp_path / "downloads" / bootstrap.ENGLISH_MODEL_WHEEL
    wheel.parent.mkdir(parents=True)
    wheel.write_bytes(b"pinned wheel")
    monkeypatch.setattr(bootstrap, "ENGLISH_MODEL_WHEEL_SHA256", bootstrap.sha256(wheel))
    monkeypatch.setattr(
        bootstrap.urllib.request,
        "urlopen",
        lambda _: pytest.fail("language wheel was downloaded"),
    )

    assert bootstrap.download_language_model(tmp_path) == wheel


def test_download_language_model_rejects_substituted_wheel_before_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()

    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", lambda _: io.BytesIO(b"substituted wheel"))

    with pytest.raises(RuntimeError, match="digest"):
        bootstrap.download_language_model(tmp_path)

    assert not (tmp_path / "downloads" / bootstrap.ENGLISH_MODEL_WHEEL).exists()


def test_verify_cache_rejects_missing_or_tampered_language_model(tmp_path: Path) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / "kokoro"
    artifacts = _write_complete_cache(root, bootstrap)
    manifest = bootstrap.artifact_manifest(root, artifacts)
    (root / "verification.json").write_text(json.dumps(manifest), encoding="utf-8")

    bootstrap.verify_cache(root)

    artifacts["language_model"].unlink()
    with pytest.raises(RuntimeError, match="cached artifacts"):
        bootstrap.verify_cache(root)

    artifacts["language_model"].write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="cached artifacts"):
        bootstrap.verify_cache(root)


@pytest.mark.parametrize("fail", [False, True])
def test_main_restores_hugging_face_environment_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail: bool
) -> None:
    bootstrap = load_bootstrap_module()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HOME", "previous-home")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "previous-hub")
    root = tmp_path / ".runtime" / "models" / "kokoro"
    if fail:
        monkeypatch.setattr(bootstrap, "download_artifacts", lambda _: (_ for _ in ()).throw(RuntimeError("failed")))
        with pytest.raises(RuntimeError, match="failed"):
            bootstrap.main()
    else:
        artifacts = _write_complete_cache(root, bootstrap)
        (root / "verification.json").write_text(
            json.dumps(bootstrap.artifact_manifest(root, artifacts)), encoding="utf-8"
        )
        assert bootstrap.main() == 0

    assert os.environ["HF_HOME"] == "previous-home"
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == "previous-hub"


def _write_complete_cache(root: Path, bootstrap: object) -> dict[str, Path]:
    contents = {
        "model_metadata": b"{}",
        "model_weights": b"model",
        "language_voice": b"voice",
        "language_model": b"wheel",
    }
    artifacts: dict[str, Path] = {}
    for artifact, content in contents.items():
        name = bootstrap.required_artifacts()[artifact]
        path = root / "cache" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        artifacts[artifact] = path
    _write_extracted_language_model(root, bootstrap)
    sf.write(root / "verification.wav", [0.0, 0.0], 24000)
    return artifacts


def _write_extracted_language_model(root: Path, bootstrap: object) -> Path:
    model_directory = (
        root
        / "language-model"
        / bootstrap.ENGLISH_MODEL_PACKAGE
        / f"{bootstrap.ENGLISH_MODEL_PACKAGE}-{bootstrap.ENGLISH_MODEL_VERSION}"
    )
    model_directory.mkdir(parents=True)
    (model_directory / "config.cfg").write_text("[nlp]\nlang = \"en\"\n", encoding="utf-8")
    (model_directory / "meta.json").write_text(
        json.dumps({"version": bootstrap.ENGLISH_MODEL_VERSION}), encoding="utf-8"
    )
    return model_directory


def _write_language_model_wheel(wheel: Path, bootstrap: object) -> None:
    model_root = f"{bootstrap.ENGLISH_MODEL_PACKAGE}/{bootstrap.ENGLISH_MODEL_PACKAGE}-{bootstrap.ENGLISH_MODEL_VERSION}"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(f"{bootstrap.ENGLISH_MODEL_PACKAGE}/__init__.py", "")
        archive.writestr(f"{model_root}/config.cfg", "[nlp]\nlang = \"en\"\n")
        archive.writestr(
            f"{model_root}/meta.json",
            json.dumps({"version": bootstrap.ENGLISH_MODEL_VERSION}),
        )
