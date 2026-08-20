from __future__ import annotations

import importlib.util
import io
import json
import os
import shutil
import sys
import types
import zipfile
from contextlib import nullcontext
from pathlib import Path

import pytest
import soundfile as sf

from multichannel.config import ConfigurationError


def load_bootstrap_module() -> object:
    path = Path(__file__).parents[2] / "scripts" / "bootstrap_kokoro.py"
    spec = importlib.util.spec_from_file_location("bootstrap_kokoro", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_cache_accepts_actual_required_kokoro_artifact_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / "kokoro"
    artifacts = _write_complete_cache(root, bootstrap)
    _approve_fixture_artifacts(monkeypatch, bootstrap, artifacts)

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


def test_main_uses_script_repository_cache_when_called_from_foreign_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()
    observed: list[Path] = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bootstrap, "verify_cache", lambda root: observed.append(root))

    assert bootstrap.main() == 0

    assert observed == [Path(bootstrap.__file__).resolve().parents[1] / ".runtime" / "models" / "kokoro"]


def test_main_rejects_runtime_symlink_before_external_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()
    repository = tmp_path / "repository"
    outside = tmp_path / "outside"
    repository.mkdir()
    outside.mkdir()
    (repository / ".runtime").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(bootstrap, "repository_root", lambda: repository)
    monkeypatch.setattr(bootstrap, "verify_cache", lambda _: pytest.fail("cache verification ran"))

    with pytest.raises(ConfigurationError, match="outside"):
        bootstrap.main()

    assert not list(outside.iterdir())


def test_verify_cache_rejects_self_consistent_fake_cache(tmp_path: Path) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / "kokoro"
    artifacts = _write_complete_cache(root, bootstrap)
    (root / "verification.json").write_text(
        json.dumps(bootstrap.artifact_manifest(root, artifacts)), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="approved"):
        bootstrap.verify_cache(root)


def test_verify_cache_rejects_fake_language_wheel_with_matching_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / "kokoro"
    artifacts = _write_complete_cache(root, bootstrap)
    approved = {artifact: bootstrap.sha256(path) for artifact, path in artifacts.items()}
    approved["language_model"] = "0" * 64
    monkeypatch.setattr(bootstrap, "APPROVED_ARTIFACT_SHA256", approved)
    (root / "verification.json").write_text(
        json.dumps(bootstrap.artifact_manifest(root, artifacts)), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="approved"):
        bootstrap.verify_cache(root)


def test_download_artifacts_uses_pinned_hugging_face_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()
    artifact = tmp_path / "downloaded"
    artifact.write_bytes(b"artifact")
    calls: list[dict[str, str]] = []

    def download(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(artifact)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(
        bootstrap,
        "APPROVED_ARTIFACT_SHA256",
        {name: bootstrap.sha256(artifact) for name in bootstrap.required_artifacts()},
    )

    bootstrap.download_artifacts(tmp_path)

    assert calls
    assert all(call["revision"] == bootstrap.KOKORO_REVISION for call in calls)


def test_main_injects_approved_local_kokoro_artifacts_without_internal_hub_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regenerate through real bootstrap wiring without allowing Kokoro fallbacks."""
    bootstrap = load_bootstrap_module()
    artifacts = {
        "model_metadata": tmp_path / "approved-config.json",
        "model_weights": tmp_path / "approved-model.pth",
        "language_voice": tmp_path / "approved-voice.pt",
    }
    for path in artifacts.values():
        path.write_bytes(b"approved")
    language_wheel = tmp_path / bootstrap.ENGLISH_MODEL_WHEEL
    language_wheel.write_bytes(b"approved wheel")
    language_directory = tmp_path / "language-model"
    language_directory.mkdir()
    calls: dict[str, object] = {}
    hub_calls: list[dict[str, object]] = []
    verify_calls = 0

    def fail_hub_download(**kwargs: object) -> str:
        hub_calls.append(kwargs)
        pytest.fail("Kokoro attempted an internal Hugging Face download")

    class FakeKModel:
        def __init__(self, *, config: str | None = None, model: str | None = None, **_: object) -> None:
            if config is None or model is None:
                fail_hub_download(repo_id="unexpected")
            calls["model"] = self
            calls["config"] = config
            calls["weights"] = model

    class FakeKPipeline:
        def __init__(self, *, lang_code: str, model: object = True, **_: object) -> None:
            if model is True:
                fail_hub_download(repo_id="unexpected")
            calls["pipeline_model"] = model
            calls["lang_code"] = lang_code

        def __call__(self, text: str, *, voice: str) -> object:
            if not voice.endswith(".pt"):
                fail_hub_download(repo_id="unexpected")
            calls["voice"] = voice
            return iter([(None, None, [0.0])])

    kokoro_module = types.ModuleType("kokoro")
    kokoro_model_module = types.ModuleType("kokoro.model")
    kokoro_module.KPipeline = FakeKPipeline
    kokoro_model_module.KModel = FakeKModel
    monkeypatch.setitem(sys.modules, "kokoro", kokoro_module)
    monkeypatch.setitem(sys.modules, "kokoro.model", kokoro_model_module)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fail_hub_download),
    )
    monkeypatch.setattr(bootstrap, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(bootstrap, "download_artifacts", lambda _: artifacts)
    monkeypatch.setattr(bootstrap, "download_language_model", lambda _: language_wheel)
    monkeypatch.setattr(bootstrap, "extract_language_model", lambda *_: language_directory)
    monkeypatch.setattr(bootstrap, "cached_english_model", lambda _: nullcontext())
    monkeypatch.setattr(bootstrap, "artifact_manifest", lambda *_: {})

    def verify_regeneration_path(_: Path) -> None:
        nonlocal verify_calls
        verify_calls += 1
        if verify_calls == 1:
            raise RuntimeError("cache requires regeneration")

    monkeypatch.setattr(bootstrap, "verify_cache", verify_regeneration_path)

    assert bootstrap.main() == 0

    assert calls["config"] == str(artifacts["model_metadata"])
    assert calls["weights"] == str(artifacts["model_weights"])
    assert calls["pipeline_model"] is calls["model"]
    assert calls["voice"] == str(artifacts["language_voice"])
    assert hub_calls == []


@pytest.mark.parametrize("artifact", ["model_metadata", "model_weights", "language_voice"])
def test_download_artifacts_rejects_non_language_digest_mismatch_before_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    bootstrap = load_bootstrap_module()
    downloaded = tmp_path / artifact
    downloaded.write_bytes(b"wrong artifact")
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda **_: str(downloaded)),
    )
    approved = {
        name: bootstrap.sha256(downloaded)
        for name in bootstrap.required_artifacts()
    }
    approved[artifact] = "0" * 64
    monkeypatch.setattr(bootstrap, "APPROVED_ARTIFACT_SHA256", approved)

    with pytest.raises(RuntimeError, match="approved"):
        bootstrap.download_artifacts(tmp_path)


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
    monkeypatch.setattr(bootstrap, "repository_root", lambda: tmp_path)
    _approve_fixture_artifacts(monkeypatch, bootstrap, artifacts)
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


def test_verify_cache_rejects_altered_valid_verification_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()
    source_root = tmp_path / "verified-kokoro"
    artifacts = _write_complete_cache(source_root, bootstrap)
    _approve_fixture_artifacts(monkeypatch, bootstrap, artifacts)
    (source_root / "verification.json").write_text(
        json.dumps(bootstrap.artifact_manifest(source_root, artifacts)), encoding="utf-8"
    )
    root = tmp_path / "altered-kokoro"
    shutil.copytree(source_root, root)
    sf.write(root / "verification.wav", [0.5, -0.5], bootstrap.SAMPLE_RATE)

    with pytest.raises(RuntimeError, match="WAV"):
        bootstrap.verify_cache(root)


def test_verify_cache_rejects_verification_wav_symlink_to_absolute_external_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / "kokoro"
    artifacts = _write_complete_cache(root, bootstrap)
    _approve_fixture_artifacts(monkeypatch, bootstrap, artifacts)
    external_wav = tmp_path / "outside.wav"
    external_wav.write_bytes((root / "verification.wav").read_bytes())
    (root / "verification.wav").unlink()
    (root / "verification.wav").symlink_to(external_wav)
    (root / "verification.json").write_text(
        json.dumps(bootstrap.artifact_manifest(root, artifacts)), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="WAV.*cache"):
        bootstrap.verify_cache(root)


def test_verify_cache_rejects_manifest_without_verification_wav_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / "kokoro"
    artifacts = _write_complete_cache(root, bootstrap)
    _approve_fixture_artifacts(monkeypatch, bootstrap, artifacts)
    manifest = bootstrap.artifact_manifest(root, artifacts)
    del manifest["verification_wav_sha256"]
    (root / "verification.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="WAV digest"):
        bootstrap.verify_cache(root)


def test_verify_cache_rejects_zero_frame_verification_wav(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / "kokoro"
    artifacts = _write_complete_cache(root, bootstrap)
    _approve_fixture_artifacts(monkeypatch, bootstrap, artifacts)
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


def test_download_language_model_does_not_follow_precreated_predictable_partial_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()
    expected = b"approved wheel"
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    predictable_partial = (downloads / bootstrap.ENGLISH_MODEL_WHEEL).with_suffix(".partial")
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside sentinel")
    predictable_partial.symlink_to(outside)
    monkeypatch.setattr(bootstrap, "ENGLISH_MODEL_WHEEL_SHA256", bootstrap.hashlib.sha256(expected).hexdigest())
    monkeypatch.setattr(bootstrap.urllib.request, "urlopen", lambda _: io.BytesIO(expected))

    wheel = bootstrap.download_language_model(tmp_path)

    assert wheel.read_bytes() == expected
    assert outside.read_bytes() == b"outside sentinel"


def test_verify_cache_rejects_missing_or_tampered_language_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / "kokoro"
    artifacts = _write_complete_cache(root, bootstrap)
    _approve_fixture_artifacts(monkeypatch, bootstrap, artifacts)
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
    monkeypatch.setattr(bootstrap, "repository_root", lambda: tmp_path)
    monkeypatch.setenv("HF_HOME", "previous-home")
    monkeypatch.setenv("HF_HUB_CACHE", "previous-effective-hub")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "previous-hub")
    root = tmp_path / ".runtime" / "models" / "kokoro"
    if fail:
        monkeypatch.setattr(bootstrap, "download_artifacts", lambda _: (_ for _ in ()).throw(RuntimeError("failed")))
        with pytest.raises(RuntimeError, match="failed"):
            bootstrap.main()
    else:
        artifacts = _write_complete_cache(root, bootstrap)
        _approve_fixture_artifacts(monkeypatch, bootstrap, artifacts)
        (root / "verification.json").write_text(
            json.dumps(bootstrap.artifact_manifest(root, artifacts)), encoding="utf-8"
        )
        assert bootstrap.main() == 0

    assert os.environ["HF_HOME"] == "previous-home"
    assert os.environ["HF_HUB_CACHE"] == "previous-effective-hub"
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == "previous-hub"


def test_main_sets_effective_hugging_face_hub_cache_for_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bootstrap = load_bootstrap_module()
    root = tmp_path / ".runtime" / "models" / "kokoro"
    monkeypatch.setattr(bootstrap, "repository_root", lambda: tmp_path)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "outside-hub"))
    monkeypatch.delitem(sys.modules, "huggingface_hub.constants", raising=False)
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)

    def inspect_effective_cache(_: Path) -> dict[str, Path]:
        from huggingface_hub import constants

        assert Path(constants.HF_HUB_CACHE) == root / "hub"
        raise RuntimeError("stop after effective cache inspection")

    monkeypatch.setattr(bootstrap, "download_artifacts", inspect_effective_cache)

    with pytest.raises(RuntimeError, match="stop after effective cache inspection"):
        bootstrap.main()


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


def _approve_fixture_artifacts(
    monkeypatch: pytest.MonkeyPatch, bootstrap: object, artifacts: dict[str, Path]
) -> None:
    monkeypatch.setattr(
        bootstrap,
        "APPROVED_ARTIFACT_SHA256",
        {artifact: bootstrap.sha256(path) for artifact, path in artifacts.items()},
    )


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
