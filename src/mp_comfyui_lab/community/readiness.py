"""Selected-only views over the existing dependency audit evidence."""

from __future__ import annotations

import copy
from typing import Any, Mapping


def _path(value: Any) -> str:
    return str(value or "").replace("\\", "/").casefold()


def _selected_graph_keys(selection: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    paths: set[str] = set()
    hashes: set[str] = set()
    for ranked in selection.get("selected", []):
        workflow = ranked.get("workflow", {})
        paths.update(
            _path(value)
            for value in (
                *workflow.get("api_graph_paths", []),
                *workflow.get("ui_graph_paths", []),
            )
        )
        hashes.update(
            str(value).casefold()
            for value in workflow.get("source_graph_hashes", [])
        )
    return paths, hashes


def _reference_matches(
    reference: Mapping[str, Any], paths: set[str], hashes: set[str]
) -> bool:
    graph_id = str(reference.get("graph_id") or "").casefold()
    relative_path = _path(reference.get("relative_path"))
    return graph_id in hashes or any(
        relative_path == selected
        or relative_path.endswith("/" + selected)
        or selected.endswith("/" + relative_path)
        for selected in paths
        if selected and relative_path
    )


def _scope_records(
    records: list[dict[str, Any]], paths: set[str], hashes: set[str]
) -> list[dict[str, Any]]:
    scoped = []
    for record in records:
        references = [
            reference
            for reference in record.get("workflow_references", [])
            if _reference_matches(reference, paths, hashes)
        ]
        if not references:
            continue
        copied = copy.deepcopy(record)
        copied["workflow_references"] = references
        scoped.append(copied)
    return scoped


def scope_dependency_inventory(
    inventory: Mapping[str, Any], selection: Mapping[str, Any]
) -> dict[str, Any]:
    """Filter a proven full-corpus dependency audit to selected graph evidence."""

    paths, hashes = _selected_graph_keys(selection)
    graphs = [
        copy.deepcopy(graph)
        for graph in inventory.get("graphs", [])
        if _reference_matches(graph, paths, hashes)
    ]
    models = _scope_records(list(inventory.get("models", [])), paths, hashes)
    nodes = _scope_records(list(inventory.get("nodes", [])), paths, hashes)
    frontend_only = _scope_records(
        list(inventory.get("frontend_only", [])), paths, hashes
    )
    subgraph_containers = _scope_records(
        list(inventory.get("subgraph_containers", [])), paths, hashes
    )
    return {
        "sources": copy.deepcopy(inventory.get("sources", {})),
        "selection_scope": {
            "graph_paths": sorted(paths),
            "graph_hashes": sorted(hashes),
        },
        "summary": {
            "graph_count": len(graphs),
            "missing_model_count": len(models),
            "missing_node_type_count": len(nodes),
            "frontend_only_type_count": len(frontend_only),
            "subgraph_container_type_count": len(subgraph_containers),
        },
        "models": models,
        "nodes": nodes,
        "frontend_only": frontend_only,
        "subgraph_containers": subgraph_containers,
        "graphs": graphs,
    }


def _workflow_scope(workflow: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    paths = {
        _path(value)
        for value in (
            *workflow.get("api_graph_paths", []),
            *workflow.get("ui_graph_paths", []),
        )
    }
    hashes = {
        str(value).casefold()
        for value in workflow.get("source_graph_hashes", [])
    }
    return paths, hashes


def _matching_records(
    records: list[dict[str, Any]], paths: set[str], hashes: set[str]
) -> list[dict[str, Any]]:
    found = []
    for record in records:
        references = record.get("workflow_references")
        if references is None:
            references = [record]
        if any(_reference_matches(reference, paths, hashes) for reference in references):
            found.append(record)
    return found


def build_selected_workflow_readiness(
    selection: Mapping[str, Any], scoped_inventory: Mapping[str, Any]
) -> dict[str, Any]:
    """Produce exact per-selected-workflow readiness from scoped audit records."""

    workflows = []
    for ranked in selection.get("selected", []):
        workflow = ranked.get("workflow", {})
        paths, hashes = _workflow_scope(workflow)
        graphs = _matching_records(
            list(scoped_inventory.get("graphs", [])), paths, hashes
        )
        models = _matching_records(
            list(scoped_inventory.get("models", [])), paths, hashes
        )
        nodes = _matching_records(
            list(scoped_inventory.get("nodes", [])), paths, hashes
        )
        missing_models = sorted(
            {
                str(filename)
                for graph in graphs
                for filename in graph.get("missing_model_filenames", [])
            }
            | {str(item["filename"]) for item in models},
            key=str.casefold,
        )
        missing_nodes = sorted(
            {
                str(node_type)
                for graph in graphs
                for node_type in graph.get("effective_missing_node_types", [])
            }
            | {str(item["node_type"]) for item in nodes},
            key=str.casefold,
        )
        missing_inputs = sorted(
            {
                str(input_name)
                for graph in graphs
                if graph.get("requires_user_inputs")
                for input_name in graph.get("referenced_user_inputs", [])
            },
            key=str.casefold,
        )
        blockers = []
        if not graphs:
            blockers.append("selected_graph_not_in_dependency_audit")
        if missing_models:
            blockers.append("missing_models")
        if missing_nodes:
            blockers.append("missing_custom_nodes")
        if missing_inputs:
            blockers.append("missing_inputs")
        restart_required = bool(missing_nodes)
        workflows.append(
            {
                "workflow_id": str(workflow.get("workflow_id")),
                "missing_models": missing_models,
                "missing_custom_nodes": missing_nodes,
                "missing_python_requirements": [],
                "missing_inputs": missing_inputs,
                "restart_required": restart_required,
                "live_node_validation_pending": restart_required,
                "render_ready_without_restart": not blockers,
                "blockers": blockers,
            }
        )
    return {
        "summary": {
            "selected_workflow_count": len(workflows),
            "ready_without_restart_count": sum(
                item["render_ready_without_restart"] for item in workflows
            ),
            "restart_required_count": sum(
                item["restart_required"] for item in workflows
            ),
            "blocked_count": sum(bool(item["blockers"]) for item in workflows),
        },
        "workflows": workflows,
    }
