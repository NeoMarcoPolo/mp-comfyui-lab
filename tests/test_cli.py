import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

import mp_comfyui_lab
from mp_runninghub_api.errors import RunningHubApiError
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
from mp_comfyui_lab.security import SafetyError, sanitize_for_persistence
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
        "status, reconcile, compare, finalists, select, export, schemas)"
    )


class FakeExperimentEngine:
    def __init__(
        self,
        record: RunRecord,
        *,
        comparison_path: Path | None = None,
        finalists_path: Path | None = None,
        run_artifact_path: Path | None = None,
    ) -> None:
        self.record = record
        self.calls: list[tuple] = []
        self.comparison_path = comparison_path
        self.finalists_path = finalists_path
        self.run_path = run_artifact_path

    def run(self, experiment_id, variant_id, *, approved_plan_hash):
        self.calls.append(
            ("run", experiment_id, variant_id, approved_plan_hash)
        )
        return self.record

    def rerun(
        self,
        experiment_id,
        variant_id,
        *,
        approved_plan_hash,
        reason,
    ):
        self.calls.append(
            (
                "rerun",
                experiment_id,
                variant_id,
                approved_plan_hash,
                reason,
            )
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

    def review_artifact_paths(self, experiment_id):
        self.calls.append(("review_artifact_paths", experiment_id))
        return type(
            "ReviewPaths",
            (),
            {
                "comparison_path": self.comparison_path,
                "finalists_path": self.finalists_path,
            },
        )()

    def run_artifact_path(self, experiment_id, run_id):
        if self.run_path is not None:
            return self.run_path
        return (Path.cwd() / "runs" / f"{run_id}.json").resolve()


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
        ("run", "plan-approved", "baseline", "a" * 64),
        ("review_artifact_paths", "plan-approved"),
    ]
    assert payload["command"] == "run"
    assert payload["data"]["run_id"] == "run-cli"
    assert payload["data"]["state"] == "SUCCESS"
    assert payload["data"]["provider_task_id"] == "provider-task-1"


def test_run_cli_retains_falsey_injected_engine(capsys, monkeypatch):
    class FalseyExperimentEngine(FakeExperimentEngine):
        def __bool__(self) -> bool:
            return False

    record = RunRecord(
        run_id="run-falsey-engine",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.SUCCESS,
        provider_task_id="provider-task-falsey",
        expected_output_count=1,
        actual_output_count=1,
    )
    engine = FalseyExperimentEngine(record)
    monkeypatch.setattr(
        app,
        "_make_experiment_engine",
        lambda *_args, **_kwargs: pytest.fail(
            "falsey injected engine triggered production construction"
        ),
    )

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
    assert payload["data"]["run_id"] == "run-falsey-engine"
    assert engine.calls[0] == (
        "run",
        "plan-approved",
        "baseline",
        "a" * 64,
    )


def test_run_cli_rejects_missing_explicit_approval_hash(capsys):
    code = main(["run", "plan-approved", "baseline"])

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.USAGE
    assert payload["command"] == "run"
    assert payload["error"]["kind"] == "usage"
    assert "--approve-plan-hash" in payload["error"]["message"]


def test_run_cli_rerun_requires_reason_and_wires_original_variant(capsys):
    record = RunRecord(
        run_id="run-rerun",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.SUCCESS,
        provider_task_id="provider-task-2",
        expected_output_count=1,
        actual_output_count=1,
    )
    engine = FakeExperimentEngine(record)

    code = main(
        [
            "run",
            "plan-approved",
            "--rerun",
            "baseline",
            "--reason",
            "measure output variance",
            "--approve-plan-hash",
            "a" * 64,
        ],
        experiment_engine=engine,
    )

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.OK
    assert engine.calls == [
        (
            "rerun",
            "plan-approved",
            "baseline",
            "a" * 64,
            "measure output variance",
        ),
        ("review_artifact_paths", "plan-approved"),
    ]
    assert payload["data"]["run_id"] == "run-rerun"


@pytest.mark.parametrize(
    "arguments",
    [
        ["run", "plan-approved", "--rerun", "baseline"],
        [
            "run",
            "plan-approved",
            "baseline",
            "--rerun",
            "baseline",
            "--reason",
            "variance",
        ],
    ],
)
def test_run_cli_rejects_incomplete_or_ambiguous_rerun_usage(
    arguments,
    capsys,
):
    code = main([*arguments, "--approve-plan-hash", "a" * 64])

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.USAGE
    assert payload["error"]["kind"] == "usage"


def test_resume_cli_wires_existing_run_id(capsys):
    record = RunRecord(
        run_id="run-resume",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.FAILED,
        expected_output_count=1,
    )
    engine = FakeExperimentEngine(record)

    code = main(
        ["resume", "plan-approved", "run-resume"],
        experiment_engine=engine,
    )

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.EXECUTION
    assert engine.calls == [
        ("resume", "plan-approved", "run-resume"),
        ("review_artifact_paths", "plan-approved"),
    ]
    assert payload["data"]["run_id"] == "run-resume"
    assert payload["data"]["state"] == "FAILED"


def test_run_cli_preserves_ambiguous_incomplete_recovery_without_review(
    tmp_path: Path,
    capsys,
) -> None:
    run_path = (tmp_path / "store" / "runs" / "run-ambiguous.json").resolve()
    record = RunRecord(
        run_id="run-ambiguous",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.INCOMPLETE,
        expected_output_count=1,
        error={"kind": "ambiguous_submission"},
    )
    engine = FakeExperimentEngine(record, run_artifact_path=run_path)

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
    assert code == ExitCode.INCOMPLETE
    assert engine.calls == [
        ("run", "plan-approved", "baseline", "a" * 64),
    ]
    assert payload["data"]["run_id"] == "run-ambiguous"
    assert payload["data"]["state"] == "INCOMPLETE"
    assert payload["data"]["provider_task_id"] is None
    assert payload["data"]["recovery_disposition"] == "reconcile_submission"
    assert "comparison_path" not in payload["data"]
    assert "finalists_path" not in payload["data"]
    assert payload["artifacts"] == [{"kind": "run", "path": str(run_path)}]


def test_resume_cli_preserves_pending_provider_recovery_without_review(
    tmp_path: Path,
    capsys,
) -> None:
    run_path = (tmp_path / "store" / "runs" / "run-pending.json").resolve()
    record = RunRecord(
        run_id="run-pending",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.INCOMPLETE,
        provider_task_id="provider-task-pending",
        expected_output_count=1,
        error={"kind": "poll_error"},
    )
    engine = FakeExperimentEngine(record, run_artifact_path=run_path)

    code = main(
        ["resume", "plan-approved", "run-pending"],
        experiment_engine=engine,
    )

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.INCOMPLETE
    assert engine.calls == [("resume", "plan-approved", "run-pending")]
    assert payload["data"]["run_id"] == "run-pending"
    assert payload["data"]["provider_task_id"] == "provider-task-pending"
    assert payload["data"]["recovery_disposition"] == "resume_provider_task"
    assert "comparison_path" not in payload["data"]
    assert "finalists_path" not in payload["data"]
    assert payload["artifacts"] == [{"kind": "run", "path": str(run_path)}]


def test_run_cli_returns_absolute_comparison_and_finalist_paths(
    tmp_path: Path,
    capsys,
) -> None:
    comparison_path = (tmp_path / "facts.json").resolve()
    finalists_path = (tmp_path / "finalists.json").resolve()
    run_artifact_path = (
        tmp_path / "actual-store" / "plan-approved" / "runs" / "run-review-paths.json"
    ).resolve()
    record = RunRecord(
        run_id="run-review-paths",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.SUCCESS,
        provider_task_id="provider-task-paths",
        expected_output_count=1,
        actual_output_count=1,
    )
    engine = FakeExperimentEngine(
        record,
        comparison_path=comparison_path,
        finalists_path=finalists_path,
        run_artifact_path=run_artifact_path,
    )

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
    assert Path(payload["data"]["comparison_path"]) == comparison_path
    assert Path(payload["data"]["finalists_path"]) == finalists_path
    assert payload["artifacts"][-2:] == [
        {"kind": "comparison_facts", "path": str(comparison_path)},
        {"kind": "finalists", "path": str(finalists_path)},
    ]
    assert payload["artifacts"][0] == {
        "kind": "run",
        "path": str(run_artifact_path),
    }


def test_reconcile_cli_wires_operator_task_id(capsys):
    record = RunRecord(
        run_id="run-reconcile",
        experiment_id="plan-approved",
        variant_id="baseline",
        state=RunState.INCOMPLETE,
        expected_output_count=1,
        error={"kind": "ambiguous_submission"},
    )
    engine = FakeExperimentEngine(record)

    code = main(
        [
            "reconcile",
            "plan-approved",
            "run-reconcile",
            "--task-id",
            "operator-task-7",
        ],
        experiment_engine=engine,
    )

    payload = _single_json_document(capsys.readouterr())
    assert code == ExitCode.INCOMPLETE
    assert engine.calls == [
        (
            "reconcile",
            "plan-approved",
            "run-reconcile",
            "operator-task-7",
        )
    ]
    assert payload["data"]["run_id"] == "run-reconcile"


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


def test_default_cli_downloader_translates_runninghub_transport_error(
    tmp_path,
    monkeypatch,
):
    config = app.LabConfig.model_validate(
        {"experiments_root": tmp_path / "experiments"},
        context={"base_dir": tmp_path},
    )
    monkeypatch.setattr(app.LabConfig, "load", lambda _root: config)

    def failing_download(_url, _target, **_kwargs):
        raise RunningHubApiError(
            "download failed api_key=durable-secret "
            "https://cdn.example/output.png?sig=signed-secret"
        )

    monkeypatch.setattr(
        app,
        "_download_runninghub_output_http",
        failing_download,
    )
    engine = app._make_experiment_engine(tmp_path)

    with pytest.raises(RuntimeError) as raised:
        engine._downloader(
            "https://cdn.xiaoyaoyou.com/output.png?sig=live",
            tmp_path / "output.png",
        )

    assert type(raised.value).__name__ == "TransientOperationError"
    assert "durable-secret" not in str(raised.value)
    assert "signed-secret" not in str(raised.value)


def test_experiment_engine_retains_falsey_injected_output_downloader(
    tmp_path: Path,
) -> None:
    class FalseyDownloader:
        def __bool__(self) -> bool:
            return False

        def __call__(self, _url: str, target: Path) -> Path:
            return target

    downloader = FalseyDownloader()

    engine = app._make_experiment_engine(
        tmp_path,
        output_downloader=downloader,
    )

    assert engine._downloader is downloader


@pytest.mark.parametrize(
    "url",
    [
        "http://cdn.xiaoyaoyou.com/output.png",
        "https://user:pass@cdn.xiaoyaoyou.com/output.png",
        "https://127.0.0.1/output.png",
        "https://10.0.0.1/output.png",
        "https://169.254.1.1/output.png",
        "https://[::1]/output.png",
        "https://[fc00::1]/output.png",
        "https://2130706433/output.png",
        "https://xiaoyaoyou.com%2eevil.example/output.png",
    ],
)
def test_production_downloader_rejects_unsafe_url_before_http(
    tmp_path: Path,
    url: str,
) -> None:
    calls: list[str] = []

    class Session:
        @staticmethod
        def get(*_args, **_kwargs):
            calls.append("http")
            raise AssertionError("unsafe URL reached HTTP")

    with pytest.raises(SafetyError):
        app._download_runninghub_output(
            url,
            tmp_path / "output.png",
            session=Session(),
        )

    assert calls == []


def test_production_downloader_rejects_redirect_to_private_address(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    class Response:
        status_code = 302
        headers = {"Location": "https://127.0.0.1/private.png"}

        @staticmethod
        def close() -> None:
            return None

    class Session:
        @staticmethod
        def get(url, **kwargs):
            calls.append({"url": url, **kwargs})
            return Response()

    def public_resolver(*_args, **_kwargs):
        return [(2, 1, 6, "", ("93.184.216.34", 443))]

    with pytest.raises(SafetyError):
        app._download_runninghub_output(
            "https://cdn.xiaoyaoyou.com/output.png",
            tmp_path / "output.png",
            session=Session(),
            resolver=public_resolver,
        )

    assert len(calls) == 1
    assert calls[0]["allow_redirects"] is False


def test_production_downloader_rejects_private_dns_before_http(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class Session:
        @staticmethod
        def get(*_args, **_kwargs):
            calls.append("http")
            raise AssertionError("private DNS result reached HTTP")

    def private_resolver(*_args, **_kwargs):
        return [(2, 1, 6, "", ("10.0.0.7", 443))]

    with pytest.raises(SafetyError, match="non-public"):
        app._download_runninghub_output(
            "https://cdn.xiaoyaoyou.com/output.png",
            tmp_path / "output.png",
            session=Session(),
            resolver=private_resolver,
        )

    assert calls == []


def test_production_downloader_follows_safe_redirect_without_credentials(
    tmp_path: Path,
) -> None:
    content = b"safe cdn bytes"
    calls: list[dict[str, object]] = []

    class Cookies:
        clears = 0

        def clear(self) -> None:
            self.clears += 1

    class Response:
        def __init__(self, status_code, headers, chunks=()):
            self.status_code = status_code
            self.headers = headers
            self._chunks = chunks

        def iter_content(self, *, chunk_size):
            assert chunk_size > 0
            return iter(self._chunks)

        def close(self) -> None:
            return None

    responses = [
        Response(
            302,
            {"Location": "https://assets.runninghub.ai/final.png"},
        ),
        Response(200, {"Content-Length": str(len(content))}, (content,)),
    ]

    class Session:
        trust_env = True
        auth = ("user", "secret")
        cookies = Cookies()
        headers = {
            "Authorization": "Bearer session-secret",
            "Cookie": "session=session-secret",
            "X-API-Key": "session-secret",
        }

        def get(self, url, **kwargs):
            effective_headers = {**self.headers, **kwargs["headers"]}
            calls.append(
                {
                    "url": url,
                    "auth": self.auth,
                    **kwargs,
                    "headers": effective_headers,
                }
            )
            return responses.pop(0)

    def public_resolver(*_args, **_kwargs):
        return [(2, 1, 6, "", ("93.184.216.34", 443))]

    target = app._download_runninghub_output(
        "https://cdn.xiaoyaoyou.com/output.png",
        tmp_path / "output.png",
        session=Session(),
        resolver=public_resolver,
    )

    assert target.read_bytes() == content
    assert [item["url"] for item in calls] == [
        "https://cdn.xiaoyaoyou.com/output.png",
        "https://assets.runninghub.ai/final.png",
    ]
    assert all(item["auth"] is None for item in calls)
    assert all(
        not {"Authorization", "Cookie", "X-API-Key"}
        & set(item["headers"])
        for item in calls
    )
    assert all(item["allow_redirects"] is False for item in calls)
    assert Session.cookies.clears >= 2


@pytest.mark.parametrize("mode", ["header", "stream"])
def test_production_downloader_enforces_maximum_stream_bytes(
    tmp_path: Path,
    mode: str,
) -> None:
    class Response:
        status_code = 200
        headers = {"Content-Length": "5"} if mode == "header" else {}

        @staticmethod
        def iter_content(*, chunk_size):
            assert chunk_size > 0
            return iter((b"123", b"45"))

        @staticmethod
        def close() -> None:
            return None

    class Session:
        @staticmethod
        def get(_url, **_kwargs):
            return Response()

    def public_resolver(*_args, **_kwargs):
        return [(2, 1, 6, "", ("93.184.216.34", 443))]

    target = tmp_path / "output.png"
    with pytest.raises(SafetyError, match="byte limit"):
        app._download_runninghub_output(
            "https://cdn.xiaoyaoyou.com/output.png",
            target,
            session=Session(),
            resolver=public_resolver,
            max_bytes=4,
        )

    assert not target.exists() or target.stat().st_size <= 4


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


def test_discovery_cli_retains_falsey_injected_service(
    capsys,
    monkeypatch,
) -> None:
    class FalseyCliService(FakeCliService):
        def __bool__(self) -> bool:
            return False

    service = FalseyCliService()
    monkeypatch.setattr(
        app,
        "_make_service",
        lambda *_args, **_kwargs: pytest.fail(
            "falsey injected service triggered production construction"
        ),
    )

    code = main(["search", "qwen"], service=service)

    _single_json_document(capsys.readouterr())
    assert code == ExitCode.OK


class FalseyOfflinePlanningLoader:
    def __init__(self, bundle):
        self.bundle = bundle
        self.briefs = []

    def __bool__(self) -> bool:
        return False

    def load_source_bundle(self, brief):
        self.briefs.append(brief)
        return self.bundle


class NoExecutorPath:
    def __getattr__(self, name):
        raise AssertionError(f"plan CLI touched executor attribute {name}")


def test_plan_cli_retains_falsey_loader_and_never_touches_executor(
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
    loader = FalseyOfflinePlanningLoader(bundle)
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
