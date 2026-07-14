import json
from pathlib import Path

import pytest

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
