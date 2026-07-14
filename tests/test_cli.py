import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import mp_comfyui_lab
import pytest

from mp_comfyui_lab.cli import app
from mp_comfyui_lab.cli.app import ExitCode, ReadOnlyPlanningBundleLoader, main
from mp_comfyui_lab.discovery.query import expand_query
from mp_comfyui_lab.discovery.ranking import (
    Artifact,
    RankedWorkflow,
    RankingComponents,
)
from mp_comfyui_lab.discovery.service import (
    DerivationEvidence,
    DiscoveryExecutionError,
    DiscoveryNotFoundError,
    DiscoveryResult,
    InspectResult,
)
from mp_comfyui_lab.discovery import service as service_module
from mp_comfyui_lab.discovery.models import Engagement, WorkflowCandidate, WorkflowDetail
from mp_comfyui_lab.discovery.urls import UnsupportedSourceError
from mp_comfyui_lab.executors.base import (
    ExecutorError,
    FetchedWorkflow,
    SearchPage,
    WorkflowRef,
)
from mp_comfyui_lab.evidence.models import (
    EvidenceGraph,
    EvidenceObservation,
    EvidenceSource,
    EvidenceSourceKind,
)
from mp_comfyui_lab.evidence.build import build_evidence_graph
from mp_comfyui_lab.experiments.models import (
    ExperimentPlan,
    PlanningSource,
    RunRecord,
    RunState,
    SourceBundle,
)
from mp_comfyui_lab.experiments.planning import recompute_exact_run_hash
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.security import sanitize_for_persistence
from mp_comfyui_lab.graph.hashing import capability_hash, source_graph_hash
from mp_comfyui_lab.graph.normalize import normalize_graph


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
        "(choose from version, search, inspect, similar, plan, run, resume, "
        "status, reconcile)"
    )


class FakeExperimentEngine:
    def __init__(self, record: RunRecord) -> None:
        self.record = record
        self.calls: list[tuple] = []

    def run(self, experiment_id, variant_id, *, approved_plan_hash):
        self.calls.append(
            ("run", experiment_id, variant_id, approved_plan_hash)
        )
        return self.record

    def resume(self, experiment_id, run_id):
        self.calls.append(("resume", experiment_id, run_id))
        return self.record

    def status(self, experiment_id, run_id):
        self.calls.append(("status", experiment_id, run_id))
        return self.record

    def reconcile(self, experiment_id, run_id, task_id):
        self.calls.append(("reconcile", experiment_id, run_id, task_id))
        return self.record


def test_run_cli_requires_and_passes_exact_plan_approval(capsys):
    record = RunRecord(
        run_id="run-cli",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.SUCCESS,
        provider_task_id="provider-task-1",
        expected_output_count=1,
        actual_output_count=1,
    )
    engine = FakeExperimentEngine(record)

    code = main(
        [
            "run",
            "plan-approved",
            "baseline",
            "--approve-plan-hash",
            "a" * 64,
        ],
        experiment_engine=engine,
    )

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.OK
    assert engine.calls == [
        ("run", "plan-approved", "baseline", "a" * 64)
    ]
    assert payload["command"] == "run"
    assert payload["data"]["run_id"] == "run-cli"
    assert payload["data"]["state"] == "SUCCESS"
    assert payload["data"]["provider_task_id"] == "provider-task-1"


def test_run_cli_rejects_missing_explicit_approval_hash(capsys):
    code = main(["run", "plan-approved", "baseline"])

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.USAGE
    assert payload["command"] == "run"
    assert payload["error"]["kind"] == "usage"
    assert "--approve-plan-hash" in payload["error"]["message"]


def test_status_cli_is_store_only_without_runninghub_construction(
    tmp_path,
    capsys,
    monkeypatch,
):
    store = ExperimentStore(tmp_path / "experiments")
    store.create("plan-approved")
    record = RunRecord(
        run_id="run-status",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.RUNNING,
        provider_task_id="provider-task-1",
        expected_output_count=2,
        actual_output_count=1,
        error={"kind": "prior_error"},
    )
    store.write_json_new(
        "plan-approved",
        "runs/run-status.json",
        record.model_dump(mode="json"),
    )

    def forbidden_executor():
        raise AssertionError("status constructed RunningHubExecutor")

    monkeypatch.setattr(app, "RunningHubExecutor", forbidden_executor)

    code = main(
        ["status", "plan-approved", "run-status"],
        cwd=tmp_path,
    )

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.OK
    assert payload["data"]["state"] == "RUNNING"
    assert payload["data"]["expected_output_count"] == 2
    assert payload["data"]["actual_output_count"] == 1
    assert payload["data"]["error"] == {"kind": "prior_error"}


@pytest.mark.parametrize(
    ("state", "expected_code"),
    [
        (RunState.INCOMPLETE, ExitCode.INCOMPLETE),
        (RunState.FAILED, ExitCode.EXECUTION),
        (RunState.CANCELLED, ExitCode.EXECUTION),
    ],
)
def test_execution_cli_maps_durable_state_to_exit_code(
    state,
    expected_code,
    capsys,
):
    record = RunRecord(
        run_id="run-cli-state",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=state,
        provider_task_id=(
            "provider-task-1"
            if state in {RunState.CANCELLED}
            else None
        ),
        expected_output_count=1,
        error={"kind": "state_result"},
    )
    engine = FakeExperimentEngine(record)

    code = main(
        [
            "run",
            "plan-approved",
            "baseline",
            "--approve-plan-hash",
            "a" * 64,
        ],
        experiment_engine=engine,
    )

    payload = _single_json_document(capsys.readouterr())
    assert code == expected_code
    assert payload["data"]["state"] == state.value


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


def test_cli_maps_terminal_discovery_to_execution_with_artifacts(capsys):
    service = FakeCliService()
    artifacts = (
        Artifact(kind="search", path="/tmp/failed-search-results.json"),
        Artifact(kind="attempts", path="/tmp/attempts.jsonl"),
    )
    service.error = DiscoveryExecutionError(
        "similar evidence cannot be represented",
        experiment_id="similar-fixedid",
        artifacts=artifacts,
        diagnostics=(),
    )

    code = main(
        ["similar", "https://www.runninghub.ai/post/1980860923768426497"],
        service=service,
    )

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.EXECUTION
    assert payload["error"]["kind"] == "execution"
    assert payload["artifacts"] == [
        {"kind": "search", "path": "/tmp/failed-search-results.json"},
        {"kind": "attempts", "path": "/tmp/attempts.jsonl"},
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
        {"source": "model", "value": "qwen", "omitted_count": 0}
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


class OfflinePlanningLoader:
    def __init__(self, bundle):
        self.bundle = bundle
        self.briefs = []

    def load_source_bundle(self, brief):
        self.briefs.append(brief)
        return self.bundle


class NoExecutorPath:
    def __getattr__(self, name):
        raise AssertionError(f"plan CLI touched executor attribute {name}")


def test_plan_cli_writes_dry_run_artifacts_and_never_touches_executor(
    tmp_path,
    capsys,
):
    graph = json.loads(
        (Path(__file__).parent / "fixtures" / "runninghub" / "qwen_graph.json").read_text(
            encoding="utf-8"
        )
    )
    source = PlanningSource(
                workflow_id="1980860923768426497",
                canonical_url="https://www.runninghub.ai/post/1980860923768426497",
                raw_graph=graph,
                normalized_graph=normalize_graph(graph),
                source_hash=source_graph_hash(graph),
                capability_hash=capability_hash(graph),
                evidence_ids=("evidence-1",),
            )
    bundle = SourceBundle(
        sources=(source,),
        evidence_graph=EvidenceGraph(
            sources=(EvidenceSource(source_id=source.workflow_id, kind=EvidenceSourceKind.CURRENT_NORMALIZED_GRAPH, canonical_url=source.canonical_url, source_hash=source.source_hash, capability_hash=source.capability_hash),),
            observations=(EvidenceObservation(source_id=source.workflow_id),),
        ),
    )
    loader = OfflinePlanningLoader(bundle)
    manifest = tmp_path / "plan.yaml"
    manifest.write_text(
        """goal: character variation
sources:
  queries: [qwen character]
  urls: ["https://www.runninghub.ai/post/1980860923768426497?token=secret"]
fixtures:
  reference_images: []
  prompt_lines: [wide shot, close portrait, night scene]
expectations:
  output_count: auto
  media_types: [image/png]
dimensions:
  checkpoint: approved
  lora_stack: approved
  prompt_grammar: approved
  sampler: approved
""",
        encoding="utf-8",
    )

    code = main(
        ["plan", str(manifest)],
        planning_loader=loader,
        executor=NoExecutorPath(),
        cwd=tmp_path,
    )

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.OK
    assert payload["command"] == "plan"
    assert payload["data"]["variant_ids"] == ["baseline"]
    assert loader.briefs[0].sources.urls == (
        "https://www.runninghub.ai/post/1980860923768426497",
    )
    artifact_paths = {item["path"] for item in payload["artifacts"]}
    assert any(path.endswith("brief.yaml") for path in artifact_paths)
    assert any("plans" in path for path in artifact_paths)
    persisted = "\n".join(Path(path).read_text(encoding="utf-8") for path in artifact_paths)
    assert "token=secret" not in persisted
    evidence_path = next(
        Path(path)
        for path in artifact_paths
        if path.replace("\\", "/").endswith("evidence/graph.json")
    )
    assert EvidenceGraph.model_validate_json(evidence_path.read_text(encoding="utf-8")).schema_version == "1"
    imported = build_evidence_graph((), prior_experiment_roots=(tmp_path / "experiments",))
    assert {source.source_id for source in imported.sources} == {source.workflow_id}
    assert not [warning for warning in imported.warnings if warning.code == "malformed_artifact"]


class ReadOnlyExecutor:
    name = "runninghub"

    def __init__(self, graph):
        self.graph = graph
        self.calls = []

    def search(self, request):
        self.calls.append(("search", request.query))
        return SearchPage(records=(_workflow_candidate(),), total=1, page=1, size=30, has_next=False)

    def fetch_workflow(self, ref):
        self.calls.append(("fetch_workflow", ref.workflow_id))
        return FetchedWorkflow(ref=ref, detail=WorkflowDetail(candidate=_workflow_candidate()), graph=self.graph)

    def inspect_capabilities(self, workflow):
        self.calls.append(("inspect_capabilities", workflow.ref.workflow_id))
        return None

    def __getattr__(self, name):
        raise AssertionError(f"read-only planning touched forbidden executor method {name}")


def test_default_read_only_planning_loader_uses_only_public_discovery_methods():
    graph = json.loads(
        (Path(__file__).parent / "fixtures" / "runninghub" / "qwen_graph.json").read_text(encoding="utf-8")
    )
    executor = ReadOnlyExecutor(graph)
    brief = app.ExperimentBrief.model_validate({
        "goal": "x",
        "sources": {"queries": ["next scene"], "urls": ["https://www.runninghub.ai/post/1980860923768426497"]},
        "fixtures": {"reference_images": [], "prompt_lines": ["one"]},
        "expectations": {"output_count": 1, "media_types": ["image/png"]},
        "dimensions": {"checkpoint": "approved"},
    })
    bundle = ReadOnlyPlanningBundleLoader(executor).load_source_bundle(brief)
    assert bundle.baseline_workflow_id == "1980860923768426497"
    call_names = [call[0] for call in executor.calls]
    assert set(call_names) == {"search", "fetch_workflow", "inspect_capabilities"}
    assert call_names[-2:] == ["fetch_workflow", "inspect_capabilities"]


def test_plan_main_constructs_only_read_only_default_executor(tmp_path, capsys, monkeypatch):
    graph = json.loads((Path(__file__).parent / "fixtures" / "runninghub" / "qwen_graph.json").read_text(encoding="utf-8"))
    graph[next(iter(graph))]["inputs"]["signed_input"] = "https://cdn.example/input.png?token=secret&revision=1"
    executor = ReadOnlyExecutor(graph)
    monkeypatch.setattr(app, "RunningHubExecutor", lambda: executor)
    manifest = tmp_path / "plan.yaml"
    manifest.write_text("""goal: x
sources:
  queries: [next scene]
  urls: []
fixtures:
  reference_images: []
  prompt_lines: [one]
expectations:
  output_count: 1
  media_types: [image/png]
dimensions:
  checkpoint: approved
""", encoding="utf-8")
    assert main(["plan", str(manifest)], cwd=tmp_path) == ExitCode.OK
    document = _single_json_document(capsys.readouterr())
    assert set(call[0] for call in executor.calls) == {"search", "fetch_workflow", "inspect_capabilities"}
    paths = [Path(item["path"]) for item in document["artifacts"]]
    source_snapshot = json.loads(next(path for path in paths if path.name == "source-bundle.json").read_text(encoding="utf-8"))
    assert "token=secret" not in json.dumps(source_snapshot)
    persisted_source = source_snapshot["sources"][0]
    assert source_graph_hash(persisted_source["raw_graph"]) == persisted_source["source_hash"]
    assert capability_hash(persisted_source["raw_graph"]) == persisted_source["capability_hash"]
    plan_path = next(path for path in paths if path.parent.name == "plans")
    persisted = json.loads(plan_path.read_text(encoding="utf-8"))
    assert sanitize_for_persistence(persisted) == persisted
    plan = ExperimentPlan.model_validate_json(plan_path.read_text(encoding="utf-8"), strict=True)
    assert plan.recompute_plan_hash() == plan.plan_hash == plan_path.stem
    assert recompute_exact_run_hash(plan.variants[0]) == plan.variants[0].exact_run_hash
    persisted["variants"][0]["materialized_graph"][next(iter(persisted["variants"][0]["materialized_graph"]))]["inputs"]["signed_input"] = "tampered"
    with pytest.raises(Exception):
        ExperimentPlan.model_validate(persisted)


class MultiWorkflowExecutor(ReadOnlyExecutor):
    def __init__(self, graphs, reverse=False):
        super().__init__(graphs["100"])
        self.graphs = graphs
        self.reverse = reverse

    def _candidate(self, workflow_id):
        return WorkflowCandidate(
            executor="runninghub",
            workflow_id=workflow_id,
            canonical_url=f"https://www.runninghub.ai/post/{workflow_id}",
            name=f"workflow-{workflow_id}",
        )

    def search(self, request):
        self.calls.append(("search", request.query))
        records = (self._candidate("100"), self._candidate("200"))
        return SearchPage(records=tuple(reversed(records)) if self.reverse else records, total=2, page=1, size=30, has_next=False)

    def fetch_workflow(self, ref):
        self.calls.append(("fetch_workflow", ref.workflow_id))
        candidate = self._candidate(ref.workflow_id)
        return FetchedWorkflow(ref=ref, detail=WorkflowDetail(candidate=candidate), graph=self.graphs[ref.workflow_id])


def test_default_plan_compiles_deterministic_donor_checkpoint_variant(tmp_path, capsys, monkeypatch):
    baseline = json.loads((Path(__file__).parent / "fixtures" / "runninghub" / "qwen_graph.json").read_text(encoding="utf-8"))
    donor = json.loads(json.dumps(baseline))
    checkpoint = next(node for node in donor.values() if node["class_type"] == "UNETLoader")
    checkpoint["inputs"]["unet_name"] = "donor.safetensors"
    manifest = tmp_path / "plan.yaml"
    manifest.write_text("""goal: x
sources:
  queries: [next scene]
  urls: [https://www.runninghub.ai/post/100]
fixtures:
  reference_images: []
  prompt_lines: [one]
expectations:
  output_count: 1
  media_types: [image/png]
dimensions:
  checkpoint: approved
""", encoding="utf-8")
    payloads = []
    for reverse in (False, True):
        executor = MultiWorkflowExecutor({"100": baseline, "200": donor}, reverse=reverse)
        monkeypatch.setattr(app, "RunningHubExecutor", lambda executor=executor: executor)
        assert main(["plan", str(manifest)], cwd=tmp_path) == ExitCode.OK
        document = _single_json_document(capsys.readouterr())
        plan_path = next(Path(item["path"]) for item in document["artifacts"] if Path(item["path"]).parent.name == "plans")
        plan = ExperimentPlan.model_validate_json(plan_path.read_text(encoding="utf-8"), strict=True)
        assert plan.capability_coverage == ("checkpoint",)
        variant = next(item for item in plan.variants if item.variant_id != "baseline")
        assert variant.evidence_ids == ("200",)
        assert variant.materialized_graph["106"]["inputs"]["unet_name"] == "donor.safetensors"
        assert {call[0] for call in executor.calls} == {"search", "fetch_workflow", "inspect_capabilities"}
        payloads.append(plan.approval_payload)
    assert payloads[0] == payloads[1]
