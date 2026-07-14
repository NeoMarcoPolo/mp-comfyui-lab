from pathlib import Path

from pydantic import ValidationError
import pytest

from mp_comfyui_lab.config import LabConfig, LabConfigError
from mp_comfyui_lab.security import SafetyError


def test_vault_uri_resolves_under_configured_root(tmp_path):
    image = tmp_path / "posted" / "frame.png"
    image.parent.mkdir()
    image.write_bytes(b"png")
    config = LabConfig(
        experiments_root=tmp_path / "runs",
        vaults={"posted": tmp_path / "posted"},
    )
    assert config.resolve_fixture("vault://posted/frame.png") == image.resolve()


def test_vault_uri_cannot_escape_root(tmp_path):
    config = LabConfig(
        experiments_root=tmp_path / "runs",
        vaults={"posted": tmp_path},
    )
    with pytest.raises(SafetyError, match="escapes vault root"):
        config.resolve_fixture("vault://posted/../secret.txt")


def test_vault_uri_rejects_unknown_alias(tmp_path):
    config = LabConfig(experiments_root=tmp_path / "runs", vaults={})

    with pytest.raises(SafetyError, match="unknown vault alias"):
        config.resolve_fixture("vault://missing/frame.png")


@pytest.mark.parametrize("fixture", ["vault://posted", "vault://posted/"])
def test_vault_uri_requires_path_component(tmp_path, fixture):
    config = LabConfig(
        experiments_root=tmp_path / "runs",
        vaults={"posted": tmp_path},
    )

    with pytest.raises(SafetyError, match="missing path"):
        config.resolve_fixture(fixture)


def test_vault_uri_rejects_traversal_that_stays_under_root(tmp_path):
    image = tmp_path / "posted" / "frame.png"
    image.parent.mkdir()
    image.write_bytes(b"png")
    config = LabConfig(
        experiments_root=tmp_path / "runs",
        vaults={"posted": tmp_path / "posted"},
    )

    with pytest.raises(SafetyError, match="traversal"):
        config.resolve_fixture("vault://posted/nested/../frame.png")


def test_nonexistent_fixture_is_rejected(tmp_path):
    config = LabConfig(experiments_root=tmp_path / "runs", vaults={})

    with pytest.raises(SafetyError, match="does not exist"):
        config.resolve_fixture(tmp_path / "missing.png")


def test_relative_fixture_resolves_from_construction_directory(tmp_path, monkeypatch):
    fixture = tmp_path / "fixtures" / "frame.png"
    fixture.parent.mkdir()
    fixture.write_bytes(b"png")
    monkeypatch.chdir(tmp_path)
    config = LabConfig(experiments_root="runs", vaults={})

    assert config.resolve_fixture(Path("fixtures") / "frame.png") == fixture.resolve()


def test_absolute_fixture_resolves_directly(tmp_path):
    fixture = tmp_path / "frame.png"
    fixture.write_bytes(b"png")
    config = LabConfig(experiments_root=tmp_path / "runs", vaults={})

    assert config.resolve_fixture(fixture) == fixture.resolve()


def test_absent_config_uses_repository_defaults(tmp_path):
    config = LabConfig.load(tmp_path)

    assert config.experiments_root == (tmp_path / "experiments").resolve()
    assert config.vaults == {}


def test_load_without_base_defaults_to_current_working_directory(
    tmp_path,
    monkeypatch,
):
    fixture = tmp_path / "fixtures" / "frame.png"
    fixture.parent.mkdir()
    fixture.write_bytes(b"png")
    monkeypatch.chdir(tmp_path)

    config = LabConfig.load()

    assert config.experiments_root == (tmp_path / "experiments").resolve()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert config.resolve_fixture("fixtures/frame.png") == fixture.resolve()


def test_load_without_base_reads_local_config_from_current_working_directory(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "config.local.yaml").write_text(
        "experiments_root: local-runs\n"
        "vaults:\n"
        "  posted: fixtures/posted\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    config = LabConfig.load()

    assert config.experiments_root == (tmp_path / "local-runs").resolve()
    assert config.vaults == {"posted": (tmp_path / "fixtures/posted").resolve()}


def test_local_config_resolves_paths_from_repository_root(tmp_path):
    (tmp_path / "config.local.yaml").write_text(
        "experiments_root: local-runs\n"
        "vaults:\n"
        "  posted: fixtures/posted\n",
        encoding="utf-8",
    )

    config = LabConfig.load(tmp_path)

    assert config.experiments_root == (tmp_path / "local-runs").resolve()
    assert config.vaults == {"posted": (tmp_path / "fixtures/posted").resolve()}


def test_local_config_rejects_unknown_top_level_keys(tmp_path):
    (tmp_path / "config.local.yaml").write_text(
        "experiments_root: experiments\n"
        "vaults: {}\n"
        "unexpected: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unexpected"):
        LabConfig.load(tmp_path)


def test_local_config_rejects_credentials(tmp_path):
    (tmp_path / "config.local.yaml").write_text(
        "experiments_root: experiments\n"
        "vaults: {}\n"
        "api_key: do-not-store-this\n",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="api_key"):
        LabConfig.load(tmp_path)


def test_local_config_rejects_duplicate_root_ranking_block_with_context(tmp_path):
    (tmp_path / "config.local.yaml").write_text(
        "experiments_root: experiments\n"
        "ranking:\n"
        "  relevance: 0.2\n"
        "ranking:\n"
        "  relevance: 0.4\n",
        encoding="utf-8",
    )

    with pytest.raises(LabConfigError) as raised:
        LabConfig.load(tmp_path)

    message = str(raised.value)
    assert "config.local.yaml" in message
    assert "duplicate" in message
    assert "ranking" in message
    assert "<root>" in message
    assert "line 4" in message


def test_local_config_rejects_duplicate_nested_weight_with_context(tmp_path):
    (tmp_path / "config.local.yaml").write_text(
        "experiments_root: experiments\n"
        "ranking:\n"
        "  relevance: 0.2\n"
        "  relevance: 0.4\n",
        encoding="utf-8",
    )

    with pytest.raises(LabConfigError) as raised:
        LabConfig.load(tmp_path)

    message = str(raised.value)
    assert "config.local.yaml" in message
    assert "duplicate" in message
    assert "ranking.relevance" in message
    assert "line 4" in message


def test_local_config_accepts_valid_nested_ranking_yaml(tmp_path):
    (tmp_path / "config.local.yaml").write_text(
        "experiments_root: experiments\n"
        "ranking:\n"
        "  relevance: 0.4\n"
        "  novelty: 0.1\n",
        encoding="utf-8",
    )

    config = LabConfig.load(tmp_path)

    assert config.ranking.relevance == 0.4
    assert config.ranking.novelty == 0.1
