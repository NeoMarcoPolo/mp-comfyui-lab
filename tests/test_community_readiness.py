from __future__ import annotations

from mp_comfyui_lab.community.readiness import (
    build_selected_workflow_readiness,
    scope_dependency_inventory,
)


def _selection() -> dict:
    return {
        "selected": [
            {
                "workflow": {
                    "workflow_id": "chosen",
                    "api_graph_paths": ["embedded/chosen-api.json"],
                    "ui_graph_paths": ["embedded/chosen-ui.json"],
                    "source_graph_hashes": ["chosen-hash"],
                }
            }
        ]
    }


def _reference(graph_id: str, path: str) -> dict:
    return {
        "source_kind": "embedded",
        "graph_id": graph_id,
        "graph_kind": "api",
        "filename": path.rsplit("/", 1)[-1],
        "relative_path": path,
        "message_url": "https://discord.com/channels/guild/channel/message",
    }


def test_scope_dependency_inventory_keeps_only_selected_graph_references() -> None:
    chosen = _reference("chosen-hash", "embedded/chosen-api.json")
    other = _reference("other-hash", "embedded/other-api.json")
    inventory = {
        "sources": {},
        "models": [
            {"filename": "chosen.safetensors", "source_status": "unresolved", "workflow_references": [chosen], "node_types": []},
            {"filename": "other.safetensors", "source_status": "unresolved", "workflow_references": [other], "node_types": []},
        ],
        "nodes": [
            {"node_type": "ChosenNode", "source_status": "manager_node_map", "workflow_references": [chosen]},
            {"node_type": "OtherNode", "source_status": "manager_node_map", "workflow_references": [other]},
        ],
        "frontend_only": [],
        "subgraph_containers": [],
        "graphs": [
            {**chosen, "effective_missing_node_types": ["ChosenNode"], "missing_model_filenames": ["chosen.safetensors"]},
            {**other, "effective_missing_node_types": ["OtherNode"], "missing_model_filenames": ["other.safetensors"]},
        ],
    }

    scoped = scope_dependency_inventory(inventory, _selection())

    assert [item["filename"] for item in scoped["models"]] == ["chosen.safetensors"]
    assert [item["node_type"] for item in scoped["nodes"]] == ["ChosenNode"]
    assert [item["graph_id"] for item in scoped["graphs"]] == ["chosen-hash"]


def test_readiness_reports_exact_missing_dependencies_and_restart_gate() -> None:
    chosen = _reference("chosen-hash", "embedded/chosen-api.json")
    scoped = {
        "models": [
            {"filename": "chosen.safetensors", "source_status": "embedded_exact_url", "workflow_references": [chosen]},
        ],
        "nodes": [
            {"node_type": "ChosenNode", "source_status": "manager_node_map", "workflow_references": [chosen]},
        ],
        "graphs": [
            {
                **chosen,
                "effective_missing_node_types": ["ChosenNode"],
                "missing_model_filenames": ["chosen.safetensors"],
                "requires_user_inputs": True,
                "referenced_user_inputs": ["reference.png"],
            }
        ],
    }

    readiness = build_selected_workflow_readiness(_selection(), scoped)

    assert readiness["summary"] == {
        "selected_workflow_count": 1,
        "ready_without_restart_count": 0,
        "restart_required_count": 1,
        "blocked_count": 1,
    }
    item = readiness["workflows"][0]
    assert item["missing_models"] == ["chosen.safetensors"]
    assert item["missing_custom_nodes"] == ["ChosenNode"]
    assert item["missing_inputs"] == ["reference.png"]
    assert item["restart_required"] is True
    assert item["render_ready_without_restart"] is False
