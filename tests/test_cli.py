import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import mp_comfyui_lab
import pytest

from mp_comfyui_lab.cli import app
from mp_comfyui_lab.cli.app import ExitCode, main
from mp_comfyui_lab.discovery.query import expand_query
from mp_comfyui_lab.discovery.ranking import (
    Artifact,
    RankedWorkflow,
    RankingComponents,
)
from mp_comfyui_lab.discovery.service import (
    DerivationEvidence,
    DiscoveryNotFoundError,
    DiscoveryResult,
    InspectResult,
)
from mp_comfyui_lab.discovery import service as service_module
from mp_comfyui_lab.discovery.models import Engagement, WorkflowCandidate, WorkflowDetail
from mp_comfyui_lab.discovery.urls import UnsupportedSourceError
from mp_comfyui_lab.executors.base import ExecutorError


VERSION_DOCUMENT = (
    '{"schema_version":"1","ok":true,"command":"version",'
    '"data":{"version":"0.1.0"},"artifacts":[]}\n'
)


def _single_json_document(captured):
    documents = captured.out.splitlines()
    assert len(documents) == 1
    assert captured.err == ""
    return json.loads(documents[0])


def test_version_emits_one_json_document(capsys):
    result = main(["version"])

    captured = capsys.readouterr()
    assert result == ExitCode.OK
    assert captured.out == VERSION_DOCUMENT
    assert captured.err == ""


def test_missing_command_emits_structured_usage_error(capsys):
    result = main([])

    captured = capsys.readouterr()
    document = _single_json_document(captured)
    assert result == ExitCode.USAGE
    assert document["ok"] is False
    assert document["error"]["kind"] == "usage"


def test_root_help_emits_structured_success(capsys):
    result = main(["--help"])

    document = _single_json_document(capsys.readouterr())
    assert result == ExitCode.OK
    assert document["ok"] is True
    assert document["command"] == "help"
    assert "usage: mp-comfy-lab" in document["data"]["text"]


def test_version_help_emits_structured_success(capsys):
    result = main(["version", "--help"])

    document = _single_json_document(capsys.readouterr())
    assert result == ExitCode.OK
    assert document["ok"] is True
    assert document["command"] == "help"
    assert "usage: mp-comfy-lab version" in document["data"]["text"]


def test_unknown_command_preserves_argparse_diagnostic(capsys):
    result = main(["unknown"])

    document = _single_json_document(capsys.readouterr())
    assert result == ExitCode.USAGE
    assert document["error"]["message"] == (
        "argument command: invalid choice: 'unknown' "
        "(choose from version, search, inspect, similar)"
    )


def test_root_unknown_option_does_not_become_command(capsys):
    result = main(["--bogus"])

    document = _single_json_document(capsys.readouterr())
    assert result == ExitCode.USAGE
    assert document["command"] is None
    assert document["error"]["message"] == (
        "the following arguments are required: command"
    )


def test_entrypoint_exits_with_main_result(monkeypatch):
    monkeypatch.setattr(app, "main", lambda: ExitCode.SAFETY)

    with pytest.raises(SystemExit) as raised:
        app.entrypoint()

    assert raised.value.code == int(ExitCode.SAFETY)


def test_installed_launcher_emits_version_document():
    launcher_name = "mp-comfy-lab.exe" if sys.platform == "win32" else "mp-comfy-lab"
    launcher = Path(sys.executable).with_name(launcher_name)

    completed = subprocess.run(
        [launcher, "version"],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == VERSION_DOCUMENT
    assert completed.stderr == ""


def test_package_version_is_loaded_from_distribution_metadata(monkeypatch):
    def fake_version(distribution_name):
        assert distribution_name == "mp-comfyui-lab"
        return "9.9.9"

    with monkeypatch.context() as patch:
        patch.setattr(importlib.metadata, "version", fake_version)
        reloaded = importlib.reload(mp_comfyui_lab)
        assert reloaded.__version__ == "9.9.9"

    importlib.reload(mp_comfyui_lab)


def _workflow_candidate() -> WorkflowCandidate:
    return WorkflowCandidate(
        executor="runninghub",
        workflow_id="1980860923768426497",
        canonical_url="https://www.runninghub.ai/post/1980860923768426497",
        name="Next Scene",
        engagement=Engagement(uses=25_757, likes=512, collects=2_194),
    )


def _ranked_workflow() -> RankedWorkflow:
    return RankedWorkflow(
        candidate=_workflow_candidate(),
        components=RankingComponents(
            relevance=1.0,
            engagement=1.0,
            output_evidence=0.5,
            recency=0.5,
            creator_track_record=1.0,
            experimentability=0.5,
            novelty=1.0,
            penalty=0.0,
        ),
        score=0.86,
        explanation="Strongest evidence: relevance; 25,757 uses.",
    )


class FakeCliService:
    def __init__(self) -> None:
        self.calls = []
        self.error = None

    def _raise(self):
        if self.error is not None:
            raise self.error

    def search(self, query, **kwargs):
        self.calls.append(("search", query, kwargs))
        self._raise()
        return DiscoveryResult(
            experiment_id="search-fixedid",
            query_plan=expand_query(query),
            results=(_ranked_workflow(),),
            derivations=(),
            artifacts=(Artifact(kind="search", path="/tmp/search-results.json"),),
        )

    def inspect(self, url):
        self.calls.append(("inspect", url, {}))
        self._raise()
        candidate = _workflow_candidate()
        return InspectResult(
            experiment_id="inspect-fixedid",
            detail=WorkflowDetail(candidate=candidate),
            source_hash="a" * 64,
            capability_hash="b" * 64,
            roles=(),
            canonicalization_error=None,
            artifacts=(Artifact(kind="detail", path="/tmp/detail.json"),),
        )

    def similar(self, url, **kwargs):
        self.calls.append(("similar", url, kwargs))
        self._raise()
        return DiscoveryResult(
            experiment_id="similar-fixedid",
            query_plan=expand_query("next scene"),
            results=(_ranked_workflow(),),
            derivations=(DerivationEvidence(source="model", value="qwen"),),
            artifacts=(Artifact(kind="search", path="/tmp/search-results.json"),),
        )


def test_search_cli_emits_ranked_urls_components_expansions_and_artifacts(capsys):
    service = FakeCliService()

    code = main(
        [
            "search",
            "qwen next scene",
            "--sort",
            "REPUTATION",
            "--page",
            "2",
            "--size",
            "7",
            "--include-graphs",
        ],
        service=service,
    )

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.OK
    assert service.calls == [
        (
            "search",
            "qwen next scene",
            {"sort": "REPUTATION", "page": 2, "size": 7, "include_graphs": True},
        )
    ]
    assert payload["data"]["results"][0]["url"] == (
        "https://www.runninghub.ai/post/1980860923768426497"
    )
    assert set(payload["data"]["results"][0]["components"]) == {
        "relevance",
        "engagement",
        "output_evidence",
        "recency",
        "creator_track_record",
        "experimentability",
        "novelty",
        "penalty",
    }
    assert payload["data"]["expansions"][0]["concept"] == "next_scene"
    assert payload["artifacts"] == [
        {"kind": "search", "path": "/tmp/search-results.json"}
    ]


def test_cli_maps_incomplete_discovery_and_preserves_artifact_paths(capsys):
    service = FakeCliService()
    artifact = Artifact(kind="search", path="/tmp/partial-search-results.json")
    service.error = service_module.DiscoveryIncompleteError(
        "no provider search succeeded",
        experiment_id="search-fixedid",
        artifacts=(artifact,),
        diagnostics=(),
    )

    code = main(["search", "qwen"], service=service)

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.INCOMPLETE
    assert payload["error"]["kind"] == "incomplete"
    assert payload["artifacts"] == [
        {"kind": "search", "path": "/tmp/partial-search-results.json"}
    ]


def test_inspect_and_similar_cli_argument_wiring(capsys):
    service = FakeCliService()
    url = "https://www.runninghub.ai/post/1980860923768426497"

    inspect_code = main(["inspect", url], service=service)
    inspect_payload = _single_json_document(capsys.readouterr())
    similar_code = main(
        ["similar", url, "--page", "3", "--size", "9"],
        service=service,
    )
    similar_payload = _single_json_document(capsys.readouterr())

    assert inspect_code == similar_code == ExitCode.OK
    assert inspect_payload["data"]["url"] == url
    assert inspect_payload["data"]["source_hash"] == "a" * 64
    assert similar_payload["data"]["derivations"] == [
        {"source": "model", "value": "qwen"}
    ]
    assert service.calls == [
        ("inspect", url, {}),
        (
            "similar",
            url,
            {"sort": None, "page": 3, "size": 9, "include_graphs": False},
        ),
    ]


@pytest.mark.parametrize(
    ("error", "exit_code", "kind"),
    [
        (ExecutorError("provider failed"), ExitCode.PROVIDER, "provider"),
        (DiscoveryNotFoundError("missing"), ExitCode.NOT_FOUND, "not_found"),
        (
            UnsupportedSourceError("unsafe source"),
            ExitCode.SAFETY,
            "safety",
        ),
    ],
)
def test_discovery_cli_maps_expected_errors_without_traceback(
    capsys,
    error,
    exit_code,
    kind,
):
    service = FakeCliService()
    service.error = error

    code = main(["search", "qwen"], service=service)

    payload = _single_json_document(capsys.readouterr())
    assert code == exit_code
    assert payload["error"]["kind"] == kind
    assert "Traceback" not in json.dumps(payload)


@pytest.mark.parametrize(
    "arguments",
    [
        ["search", "qwen", "--page", "0"],
        ["search", "qwen", "--size", "101"],
        ["inspect", "https://www.runninghub.ai/post/1", "--sort", "HOTTEST"],
        ["similar", "https://www.runninghub.ai/post/1", "--include-graphs=true"],
    ],
)
def test_discovery_cli_rejects_invalid_usage_as_one_json_document(
    capsys,
    arguments,
):
    code = main(arguments, service=FakeCliService())

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.USAGE
    assert payload["error"]["kind"] == "usage"


def test_discovery_help_remains_structured_json(capsys):
    code = main(["search", "--help"], service=FakeCliService())

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.OK
    assert payload["command"] == "help"
    assert "search QUERY" in payload["data"]["text"]


def test_malformed_local_config_is_structured_usage_error(tmp_path, capsys):
    (tmp_path / "config.local.yaml").write_text(
        "experiments_root: [unterminated\n",
        encoding="utf-8",
    )

    code = main(["search", "qwen"], cwd=tmp_path)

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.USAGE
    assert payload["error"]["kind"] == "usage"
    assert "Traceback" not in json.dumps(payload)


def test_dependency_wiring_passes_cwd_and_injected_executor(
    tmp_path,
    capsys,
    monkeypatch,
):
    service = FakeCliService()
    executor = object()
    calls = []

    def make_service(root, *, executor=None):
        calls.append((root, executor))
        return service

    monkeypatch.setattr(app, "_make_service", make_service)

    code = main(["search", "qwen"], cwd=tmp_path, executor=executor)

    _single_json_document(capsys.readouterr())
    assert code == ExitCode.OK
    assert calls == [(tmp_path, executor)]
