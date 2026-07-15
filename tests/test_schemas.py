import json
import os
from pathlib import Path
import stat
import subprocess
import threading
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import mp_comfyui_lab.schemas as schema_module
from mp_comfyui_lab.cli.app import ExitCode, main
from mp_comfyui_lab.comparison.paths import path_component_is_link_or_junction
from mp_comfyui_lab.experiments.models import DimensionName, ExperimentBrief
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


def _brief_document(dimensions: object) -> dict[str, object]:
    return {
        "goal": "schema parity",
        "sources": {
            "queries": ["qwen"],
            "urls": [],
        },
        "fixtures": {
            "reference_images": [],
            "prompt_lines": ["one"],
        },
        "expectations": {
            "output_count": 1,
            "media_types": ["image/png"],
        },
        "dimensions": dimensions,
    }


def _dimensions_json_schema_accepts(schema: dict, value: object) -> bool:
    dimensions = schema["properties"]["dimensions"]
    names = set(schema["$defs"]["DimensionName"]["enum"])
    policies = set(schema["$defs"]["DimensionPolicy"]["enum"])
    if dimensions.get("type") == "object":
        return (
            isinstance(value, dict)
            and len(value) >= dimensions.get("minProperties", 0)
            and all(
                key in names and policy in policies
                for key, policy in value.items()
            )
        )
    if dimensions.get("type") == "array":
        return (
            isinstance(value, list)
            and len(value) >= dimensions.get("minItems", 0)
            and all(
                isinstance(item, dict)
                and set(item) == {"name", "policy"}
                and item["name"] in names
                and item["policy"] in policies
                for item in value
            )
        )
    return False


def test_brief_dimension_mapping_schema_matches_runtime_both_ways() -> None:
    schema = json.loads(render_schemas()["brief.schema.json"])
    runtime_valid = _brief_document(
        {"checkpoint": "approved", "sampler": "disabled"}
    )

    brief = ExperimentBrief.model_validate(runtime_valid)

    assert _dimensions_json_schema_accepts(schema, runtime_valid["dimensions"])
    assert [item.name.value for item in brief.dimensions] == [
        "checkpoint",
        "sampler",
    ]

    representative = {
        schema["$defs"]["DimensionName"]["enum"][-1]:
        schema["$defs"]["DimensionPolicy"]["enum"][0]
    }
    assert _dimensions_json_schema_accepts(schema, representative)
    representative_name, representative_policy = next(iter(representative.items()))
    representative_brief = ExperimentBrief.model_validate(
        _brief_document(representative)
    )
    assert (
        representative_brief.policy_for(DimensionName(representative_name)).value
        == representative_policy
    )


@pytest.mark.parametrize(
    "invalid",
    [
        [],
        [{"name": "checkpoint", "policy": "approved"}],
        {},
        {"unknown": "approved"},
        {"checkpoint": "unknown"},
    ],
)
def test_brief_dimension_mapping_schema_and_runtime_reject_same_invalid_shapes(
    invalid: object,
) -> None:
    schema = json.loads(render_schemas()["brief.schema.json"])

    assert not _dimensions_json_schema_accepts(schema, invalid)
    with pytest.raises(ValidationError):
        ExperimentBrief.model_validate(_brief_document(invalid))


def test_plan_schema_has_no_variant_count_ceiling() -> None:
    schema = json.loads(render_schemas()["plan.schema.json"])

    variants = schema["properties"]["variants"]
    assert variants["minItems"] == 1
    assert "maxItems" not in variants


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


@pytest.mark.parametrize("failed_child", ("stage", "backup"))
def test_export_cleans_transaction_root_when_child_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_child: str,
) -> None:
    output = tmp_path / "schemas"
    paths = export_schemas(output)
    prior = {path.name: path.read_bytes() for path in paths}
    original_mkdir = Path.mkdir

    def fail_child_creation(path: Path, *args, **kwargs) -> None:
        if path.name == failed_child:
            raise OSError(f"injected {failed_child} creation failure")
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fail_child_creation)

    with pytest.raises(OSError, match=f"injected {failed_child} creation failure"):
        export_schemas(output)

    assert {path.name for path in output.iterdir()} == set(prior)
    assert {
        path.name: path.read_bytes()
        for path in output.iterdir()
    } == prior


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


@pytest.mark.parametrize("operation", (export_schemas, check_schemas))
def test_schema_operations_reject_symlink_target_without_mutation(
    tmp_path: Path,
    operation,
) -> None:
    output = tmp_path / "schemas"
    output.mkdir()
    target = tmp_path / "outside-brief.json"
    target.write_bytes(b'{"outside":true}\n')
    linked = output / "brief.schema.json"
    try:
        linked.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("file symlinks are not supported")
    link_value = os.readlink(linked)

    with pytest.raises(
        schema_module.SchemaPublicationError,
        match="link|reparse",
    ):
        operation(output)

    assert linked.is_symlink()
    assert os.readlink(linked) == link_value
    assert target.read_bytes() == b'{"outside":true}\n'
    assert {path.name for path in output.iterdir()} == {"brief.schema.json"}


@pytest.mark.parametrize("operation", (export_schemas, check_schemas))
def test_schema_operations_reject_symlinked_output_root_without_mutation(
    tmp_path: Path,
    operation,
) -> None:
    target = tmp_path / "outside-schemas"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    linked = tmp_path / "schemas"
    try:
        linked.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not supported")
    link_value = os.readlink(linked)

    with pytest.raises(
        schema_module.SchemaPublicationError,
        match="link|reparse",
    ):
        operation(linked)

    assert linked.is_symlink()
    assert os.readlink(linked) == link_value
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert {path.name for path in target.iterdir()} == {"sentinel.txt"}


@pytest.mark.parametrize("operation", (export_schemas, check_schemas))
def test_schema_operations_reject_linked_output_ancestor_without_mutation(
    tmp_path: Path,
    operation,
) -> None:
    target = tmp_path / "outside-parent"
    target.mkdir()
    linked_parent = tmp_path / "linked-parent"
    try:
        linked_parent.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("directory symlinks are not supported")

    with pytest.raises(
        schema_module.SchemaPublicationError,
        match="link|reparse",
    ):
        operation(linked_parent / "schemas")

    assert linked_parent.is_symlink()
    assert list(target.iterdir()) == []


def test_path_helper_rejects_generic_windows_reparse_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "is_junction", None, raising=False)
    monkeypatch.setattr(os.path, "isjunction", None, raising=False)
    metadata = SimpleNamespace(
        st_mode=stat.S_IFREG,
        st_reparse_tag=0,
        st_file_attributes=0x400,
    )

    assert path_component_is_link_or_junction(tmp_path / "reparse", metadata)


@pytest.mark.parametrize("operation", (export_schemas, check_schemas))
def test_schema_operations_reject_non_regular_target_without_mutation(
    tmp_path: Path,
    operation,
) -> None:
    output = tmp_path / "schemas"
    output.mkdir()
    non_regular = output / "brief.schema.json"
    non_regular.mkdir()

    with pytest.raises(
        schema_module.SchemaPublicationError,
        match="regular file",
    ):
        operation(output)

    assert non_regular.is_dir()
    assert list(non_regular.iterdir()) == []
    assert {path.name for path in output.iterdir()} == {"brief.schema.json"}


@pytest.mark.parametrize("operation", (export_schemas, check_schemas))
def test_schema_operations_reject_windows_junction_output_root(
    tmp_path: Path,
    operation,
) -> None:
    if os.name != "nt":
        pytest.skip("directory junctions are Windows-specific")
    target = tmp_path / "junction-target"
    target.mkdir()
    sentinel = target / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    junction = tmp_path / "schemas-junction"
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip("directory junctions are not supported")
    try:
        with pytest.raises(
            schema_module.SchemaPublicationError,
            match="link|reparse",
        ):
            operation(junction)

        assert sentinel.read_text(encoding="utf-8") == "unchanged"
        assert {path.name for path in target.iterdir()} == {"sentinel.txt"}
    finally:
        junction.rmdir()


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
