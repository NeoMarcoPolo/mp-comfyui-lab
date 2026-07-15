import json
from pathlib import Path

from mp_comfyui_lab.cli.app import ExitCode, main
from mp_comfyui_lab.discovery.models import WorkflowCandidate, WorkflowDetail
from mp_comfyui_lab.executors.base import (
    FetchedWorkflow,
    ProviderOutput,
    SearchPage,
    WorkflowRef,
)

from helpers import FakeExecutor, valid_media_bytes


WORKFLOW_ID = "1980860923768426497"
WORKFLOW_URL = f"https://www.runninghub.ai/post/{WORKFLOW_ID}"


def _invoke(capsys, arguments, **dependencies):
    code = main(arguments, **dependencies)
    captured = capsys.readouterr()
    assert code == ExitCode.OK, captured.out
    assert captured.err == ""
    documents = captured.out.splitlines()
    assert len(documents) == 1
    payload = json.loads(documents[0])
    assert payload["ok"] is True
    return payload


def test_noninteractive_agent_flow_reaches_self_contained_handoff(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "builtins.input",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("the noninteractive CLI must not prompt")
        ),
    )
    graph = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "runninghub"
            / "qwen_graph.json"
        ).read_text(encoding="utf-8")
    )
    graph_inputs = graph[next(iter(graph))]["inputs"]
    graph_inputs.update(
        {
            "openai_api_key": "e2e-openai-secret",
            "aws_secret_access_key": "e2e-aws-secret",
            "hf_token": "e2e-hf-secret",
            "token_count": 9,
            "tokenizer": "qwen-tokenizer",
            "secret_style": "garden",
            "prompt": "A secret garden",
        }
    )
    candidate = WorkflowCandidate(
        executor="fake",
        workflow_id=WORKFLOW_ID,
        canonical_url=WORKFLOW_URL,
        name="Qwen character workflow",
    )
    executor = FakeExecutor.success(
        outputs=(
            ProviderOutput(
                node_id="105",
                url="memory://result.png",
                media_type="image/png",
            ),
        )
    )
    executor.search_page = SearchPage(
        records=(candidate,),
        total=1,
        page=1,
        size=30,
        has_next=False,
    )
    executor.fetched_workflow = FetchedWorkflow(
        ref=WorkflowRef(
            executor="fake",
            workflow_id=WORKFLOW_ID,
            canonical_url=WORKFLOW_URL,
        ),
        detail=WorkflowDetail(candidate=candidate),
        graph=graph,
    )

    _invoke(
        capsys,
        ["search", "qwen character"],
        cwd=tmp_path,
        executor=executor,
    )
    _invoke(
        capsys,
        ["inspect", WORKFLOW_URL],
        cwd=tmp_path,
        executor=executor,
    )

    fixture = tmp_path / "portrait.png"
    fixture.write_bytes(valid_media_bytes("image/png"))
    manifest = tmp_path / "brief.yaml"
    manifest.write_text(
        f"""goal: preserve character identity in the next scene
sources:
  queries: []
  urls: [{WORKFLOW_URL}]
fixtures:
  reference_images:
    - uri: portrait.png
      binding_index: 0
  prompt_lines: [cinematic next scene]
expectations:
  output_count: 1
  media_types: [image/png]
dimensions:
  checkpoint: baseline_only
""",
        encoding="utf-8",
    )
    planned = _invoke(
        capsys,
        ["plan", str(manifest)],
        cwd=tmp_path,
        executor=executor,
    )
    experiment_id = planned["data"]["experiment_id"]
    plan_hash = planned["data"]["plan_hash"]
    variant_id = planned["data"]["variant_ids"][0]

    def download_output(_url: str, target: Path) -> Path:
        target.write_bytes(valid_media_bytes("image/png"))
        return target

    _invoke(
        capsys,
        [
            "run",
            experiment_id,
            "--approve-plan-hash",
            plan_hash,
        ],
        cwd=tmp_path,
        executor=executor,
        output_downloader=download_output,
    )
    compared = _invoke(
        capsys,
        ["compare", experiment_id],
        cwd=tmp_path,
    )
    gallery = Path(compared["data"]["gallery"])
    assert gallery.is_absolute()
    assert gallery.name == "index.html"
    assert gallery.is_file()

    finalists = _invoke(
        capsys,
        ["finalists", experiment_id],
        cwd=tmp_path,
    )
    assert finalists["data"]["requires_human_selection"] is True
    assert variant_id in finalists["data"]["variant_ids"]
    _invoke(
        capsys,
        ["select", experiment_id, variant_id],
        cwd=tmp_path,
    )
    exported = _invoke(
        capsys,
        ["export", experiment_id],
        cwd=tmp_path,
    )
    bundle = Path(exported["data"]["bundle"])
    assert bundle.is_absolute()
    assert bundle.is_dir()
    assert (bundle / "handoff.json").is_file()
    serialized_bundle = b"\n".join(
        path.read_bytes()
        for path in sorted(bundle.rglob("*"))
        if path.is_file()
    )
    for secret in (
        b"e2e-openai-secret",
        b"e2e-aws-secret",
        b"e2e-hf-secret",
    ):
        assert secret not in serialized_bundle
    selected_graph = json.loads(
        (bundle / "selected-workflow-api.json").read_text(encoding="utf-8")
    )
    selected_inputs = selected_graph[next(iter(graph))]["inputs"]
    assert selected_inputs["openai_api_key"] == "[REDACTED]"
    assert selected_inputs["aws_secret_access_key"] == "[REDACTED]"
    assert selected_inputs["hf_token"] == "[REDACTED]"
    assert selected_inputs["token_count"] == 9
    assert selected_inputs["tokenizer"] == "qwen-tokenizer"
    assert selected_inputs["secret_style"] == "garden"
    assert selected_inputs["prompt"] == "A secret garden"
