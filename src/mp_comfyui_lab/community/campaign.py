"""Build immutable ranked workflow campaign artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from mp_comfyui_lab.community.catalog import build_workflow_catalog
from mp_comfyui_lab.community.ranking import (
    EvidenceSnapshot,
    score_workflows,
    select_test_workflows,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _verify_graph_paths(catalog, graph_root: Path) -> None:
    missing = sorted(
        {
            relative_path
            for workflow in catalog
            for relative_path in (*workflow.api_graph_paths, *workflow.ui_graph_paths)
            if not (graph_root / relative_path).is_file()
        }
    )
    if missing:
        raise ValueError(f"Selected source graph paths are missing: {missing}")


def _summary_markdown(selection) -> str:
    lines = [
        "# Unstable Diffusion Discord Workflow Selection",
        "",
        f"Selected unique workflows: {len(selection.selected)}",
        f"Excluded unique workflows: {len(selection.excluded)}",
        "",
        "## Selected",
        "",
    ]
    for index, item in enumerate(selection.selected, start=1):
        identity = item.workflow.identity
        source = item.workflow.sources[0]
        lines.append(
            f"{index}. {identity.family} / {identity.task} / {identity.technique} "
            f"— score {item.score:.6f} — {source.message_url}"
        )
    lines.extend(["", "## Excluded", ""])
    for item in selection.excluded:
        identity = item.workflow.identity
        lines.append(
            f"- {identity.family} / {identity.task} / {identity.technique}: "
            f"{item.exclusion_reason}"
        )
    return "\n".join(lines) + "\n"


def run_campaign(
    manifest_paths: Sequence[Path],
    engagement_path: Path,
    graph_root: Path,
    output_dir: Path,
    as_of: datetime,
) -> dict[str, int]:
    """Create the complete deterministic ranking artifact set."""

    evidence_payload = json.loads(engagement_path.read_text(encoding="utf-8"))
    evidence = EvidenceSnapshot.model_validate(evidence_payload)
    if not evidence.complete:
        raise ValueError("Engagement evidence must be complete")
    catalog = build_workflow_catalog(manifest_paths, graph_root)
    _verify_graph_paths(catalog, graph_root)
    ranked = score_workflows(catalog, evidence, as_of)
    selection = select_test_workflows(ranked)
    eligible_families = {
        item.workflow.identity.family
        for item in (*selection.selected, *selection.excluded)
        if item.exclusion_reason not in {
            "obsolete_family",
            "not_approved_family",
            "lower_quality_duplicate",
        }
    }
    selected_families = {
        item.workflow.identity.family for item in selection.selected
    }
    if eligible_families - selected_families:
        raise ValueError(
            "Eligible families missing a selected winner: "
            + ", ".join(sorted(eligible_families - selected_families))
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "as_of": as_of.isoformat(),
        "engagement": {
            "path": str(engagement_path),
            "sha256": _sha256(engagement_path),
        },
        "manifests": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in manifest_paths
        ],
    }
    _write_json(
        output_dir / "source-evidence.json",
        {
            "schema_version": 1,
            "inputs": inputs,
            "evidence": evidence.model_dump(mode="json"),
        },
    )
    _write_json(
        output_dir / "workflow-catalog.json",
        {
            "schema_version": 1,
            "inputs": inputs,
            "workflow_count": len(catalog),
            "workflows": [item.model_dump(mode="json") for item in catalog],
        },
    )
    _write_json(
        output_dir / "ranked-workflows.json",
        {
            "schema_version": 1,
            "inputs": inputs,
            "ranked": [item.model_dump(mode="json") for item in ranked],
        },
    )
    _write_json(
        output_dir / "selected-test-manifest.json",
        {
            "schema_version": 1,
            "inputs": inputs,
            "selected_count": len(selection.selected),
            "excluded_count": len(selection.excluded),
            "selected": [item.model_dump(mode="json") for item in selection.selected],
            "excluded": [item.model_dump(mode="json") for item in selection.excluded],
        },
    )
    _write_text(output_dir / "selection-summary.md", _summary_markdown(selection))
    return {
        "catalog_count": len(catalog),
        "selected_count": len(selection.selected),
        "excluded_count": len(selection.excluded),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--engagement", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
    result = run_campaign(
        args.manifest,
        args.engagement,
        args.graph_root,
        args.output_dir,
        as_of,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
