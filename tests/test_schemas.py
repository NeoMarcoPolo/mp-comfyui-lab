import json
import os
from pathlib import Path
import threading

import pytest

import mp_comfyui_lab.schemas as schema_module
from mp_comfyui_lab.cli.app import ExitCode, main
from mp_comfyui_lab.schemas import (
    SchemaCheckError,
    check_schemas,
    export_schemas,
    render_schemas,
)


SCHEMA_NAMES = {
    "brief.schema.json",
    "handoff.schema.json",
    "plan.schema.json",
}


def _single_document(capsys):
    captured = capsys.readouterr()
    assert captured.err == ""
    lines = captured.out.splitlines()
    assert len(lines) == 1
    return json.loads(lines[0])


def test_renderer_is_deterministic_sorted_json_with_trailing_newline() -> None:
    first = render_schemas()
    second = render_schemas()

    assert first == second
    assert set(first) == SCHEMA_NAMES
    for rendered in first.values():
        assert rendered.endswith("\n")
        assert not rendered.endswith("\n\n")
        parsed = json.loads(rendered)
        assert rendered == json.dumps(
            parsed,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        ) + "\n"


def test_handoff_schema_publishes_required_version_and_path_constraints() -> None:
    schema = json.loads(render_schemas()["handoff.schema.json"])

    assert "schema_version" in schema["required"]
    version = schema["properties"]["schema_version"]
    assert version["const"] == "1"
    assert "default" not in version
    assert "supported" in version["description"].lower()

    path = schema["$defs"]["HandoffArtifact"]["properties"]["path"]
    assert path["minLength"] == 1
    assert path["maxLength"] == 4096
    assert path["pattern"]
    assert "posix" in path["description"].lower()
    assert "relative" in path["description"].lower()


def test_export_and_check_detect_missing_and_stale_files_without_repair(
    tmp_path: Path,
) -> None:
    output = tmp_path / "schemas"
    written = export_schemas(output)

    assert {path.name for path in written} == SCHEMA_NAMES
    assert all(path.is_absolute() and path.is_file() for path in written)
    assert check_schemas(output) == written

    missing = output / "brief.schema.json"
    stale = output / "plan.schema.json"
    missing.unlink()
    stale.write_text("{}\n", encoding="utf-8")

    with pytest.raises(SchemaCheckError) as raised:
        check_schemas(output)

    assert raised.value.missing == (missing.resolve(),)
    assert raised.value.stale == (stale.resolve(),)
    assert not missing.exists()
    assert stale.read_text(encoding="utf-8") == "{}\n"


@pytest.mark.parametrize("failure_point", (1, 2, 3))
def test_export_restores_exact_prior_schema_set_after_publish_failure(
    tmp_path: Path,
    failure_point: int,
) -> None:
    output = tmp_path / "schemas"
    output.mkdir()
    prior = {
        "brief.schema.json": b'{"prior":"brief"}\n',
        "handoff.schema.json": None,
        "plan.schema.json": b'{"prior":"plan"}\n',
    }
    for name, content in prior.items():
        if content is not None:
            (output / name).write_bytes(content)

    replacements = 0

    def fail_once(source: Path, destination: Path) -> None:
        nonlocal replacements
        replacements += 1
        if replacements == failure_point:
            raise OSError(f"injected replacement failure {failure_point}")
        os.replace(source, destination)

    with pytest.raises(OSError, match="injected replacement failure"):
        export_schemas(output, _replace=fail_once)

    for name, content in prior.items():
        path = output / name
        if content is None:
            assert not path.exists()
        else:
            assert path.read_bytes() == content
    assert {path.name for path in output.iterdir()} == {
        name for name, content in prior.items() if content is not None
    }


def test_export_is_idempotent_and_leaves_only_the_complete_schema_set(
    tmp_path: Path,
) -> None:
    output = tmp_path / "schemas"

    first_paths = export_schemas(output)
    first_content = {path.name: path.read_bytes() for path in first_paths}
    second_paths = export_schemas(output)

    assert second_paths == first_paths
    assert {path.name: path.read_bytes() for path in second_paths} == first_content
    assert {path.name for path in output.iterdir()} == SCHEMA_NAMES


def test_check_waits_for_complete_schema_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "schemas"
    export_schemas(output)
    updated = {}
    for name, document in render_schemas().items():
        value = json.loads(document)
        value["x-transaction-test"] = True
        updated[name] = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        ) + "\n"
    monkeypatch.setattr(schema_module, "render_schemas", lambda: updated)

    first_published = threading.Event()
    release_writer = threading.Event()
    checker_started = threading.Event()
    checker_finished = threading.Event()
    errors: list[BaseException] = []
    replacements = 0

    def blocking_replace(source: Path, destination: Path) -> None:
        nonlocal replacements
        os.replace(source, destination)
        replacements += 1
        if replacements == 1:
            first_published.set()
            assert release_writer.wait(timeout=5)

    def write() -> None:
        try:
            export_schemas(output, _replace=blocking_replace)
        except BaseException as error:
            errors.append(error)

    def check() -> None:
        checker_started.set()
        try:
            check_schemas(output)
        except BaseException as error:
            errors.append(error)
        finally:
            checker_finished.set()

    writer = threading.Thread(target=write)
    reader = threading.Thread(target=check)
    writer.start()
    assert first_published.wait(timeout=5)
    reader.start()
    assert checker_started.wait(timeout=5)
    assert not checker_finished.wait(timeout=0.2)
    release_writer.set()
    writer.join(timeout=5)
    reader.join(timeout=5)

    assert not writer.is_alive()
    assert not reader.is_alive()
    assert errors == []
    assert check_schemas(output) == tuple(output.resolve() / name for name in updated)
    assert {path.name for path in output.iterdir()} == SCHEMA_NAMES


def test_committed_schemas_match_current_models() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    expected = render_schemas()

    assert {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted((repository_root / "schemas").glob("*.schema.json"))
    } == expected


def test_schema_cli_exports_checks_and_reports_staleness(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "generated"

    assert main(
        ["schemas", "export", "--output", str(output)],
        cwd=tmp_path,
    ) == ExitCode.OK
    exported = _single_document(capsys)
    assert exported["command"] == "schemas export"
    assert {Path(item["path"]).name for item in exported["artifacts"]} == (
        SCHEMA_NAMES
    )

    assert main(
        ["schemas", "check", "--output", str(output)],
        cwd=tmp_path,
    ) == ExitCode.OK
    checked = _single_document(capsys)
    assert checked["command"] == "schemas check"
    assert checked["data"]["status"] == "current"

    (output / "handoff.schema.json").write_text("{}\n", encoding="utf-8")
    assert main(
        ["schemas", "check", "--output", str(output)],
        cwd=tmp_path,
    ) == ExitCode.SCHEMA
    stale = _single_document(capsys)
    assert stale["command"] == "schemas check"
    assert stale["error"]["kind"] == "schema"
    assert "stale" in stale["error"]["message"]
    assert "handoff.schema.json" in stale["error"]["message"]
