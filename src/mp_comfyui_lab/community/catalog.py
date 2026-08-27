"""Build conservative executable workflow identities from embedded graphs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from mp_comfyui_lab.community.models import (
    WorkflowIdentity,
    WorkflowRecord,
    WorkflowSource,
)
from mp_comfyui_lab.graph.hashing import (
    GraphHashError,
    capability_hash,
    source_graph_hash,
)
from mp_comfyui_lab.graph.normalize import GraphNormalizationError


_MODEL_EXTENSIONS = (
    ".safetensors",
    ".ckpt",
    ".pt",
    ".pth",
    ".bin",
    ".gguf",
)
def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _all_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _all_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)


def _evidence_text(image: Mapping[str, Any], graph: Mapping[str, Any] | None) -> str:
    values = [
        str(image.get("filename") or ""),
        str(image.get("message_content") or ""),
        *[str(item) for item in image.get("search_reasons") or []],
        *[str(item) for item in image.get("search_queries") or []],
    ]
    if graph is not None:
        values.extend(_all_strings(graph))
    return " ".join(values).casefold()


def _classify_family(evidence: str) -> tuple[str, str]:
    compact = _compact(evidence)
    if "scail2" in compact:
        return "scail2", "2"
    if "animate" in compact:
        return "animate", "current"
    if any(token in compact for token in ("wan22", "wanv22", "wanvideo22")):
        return "wan-2.2", "2.2"
    if any(token in compact for token in ("wan21", "wanv21", "wanvideo21")):
        return "wan-2.1", "2.1"
    if "krea2" in compact:
        return "krea-2", "2"
    if any(token in compact for token in ("zimageturbo", "zitturbo", "zit")):
        return "zit", "current"
    if "zimage" in compact:
        return "z-image", "current"
    if "flux2" in compact and "klein" in compact:
        return "flux-2-klein", "2"
    if "klein" in compact:
        return "flux-2-klein", "2"
    if "flux2" in compact:
        return "flux-2", "2"
    if "qwen" in compact and "edit" in compact:
        return "qwen-image-edit", "current"
    if "qwen" in compact:
        return "qwen-image", "current"
    if "minimax" in compact and any(token in compact for token in ("h3", "hailuo")):
        return "minimax-h3", "3"
    if "hunyuan" in compact:
        return "hunyuan", "unknown"
    if "pony" in compact:
        return "pony", "unknown"
    if "sdxl" in compact:
        return "sdxl", "1"
    return "unknown", "unknown"


def _classify_task(evidence: str, family: str) -> str:
    compact = _compact(evidence)
    if any(token in compact for token in ("referencetovideo", "r2v", "ref2video")):
        return "reference-to-video"
    if any(token in compact for token in ("imagetovideo", "i2v", "img2video")):
        return "image-to-video"
    if any(token in compact for token in ("texttovideo", "t2v", "txt2video")):
        return "text-to-video"
    if "edit" in compact and "image" in compact:
        return "image-edit"
    if any(token in compact for token in ("videocombine", "savevideo", "videooutput", "animate")):
        return "text-to-video"
    if family in {"wan-2.1", "wan-2.2", "minimax-h3"}:
        return "text-to-video"
    return "text-to-image"


def _classify_technique(evidence: str) -> str:
    compact = _compact(evidence)
    if any(token in compact for token in ("svipro2", "sviproii")):
        return "svi-pro-2"
    if "svipro" in compact:
        return "svi-pro"
    if "scail2" in compact:
        return "scail2"
    if "animate" in compact:
        return "animate"
    if any(token in compact for token in ("longvideo", "chunkedvideo", "multishot")):
        return "long-video"
    if any(token in compact for token in ("interpolation", "interpolate", "rife")):
        return "interpolation"
    return "base"


def _identity_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(graph))
    for node in normalized.values():
        if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
            continue
        node_class = _compact(str(node.get("class_type") or ""))
        for field_name in list(node["inputs"]):
            canonical = field_name.casefold()
            value = node["inputs"][field_name]
            if canonical in {"seed", "noise_seed", "random_seed"}:
                node["inputs"][field_name] = 0
            elif canonical in {"width", "height", "batch_size"}:
                node["inputs"][field_name] = 512 if canonical != "batch_size" else 1
            elif canonical == "filename_prefix":
                node["inputs"][field_name] = "output"
            elif (
                canonical in {"text", "prompt", "positive_prompt", "negative_prompt"}
                and isinstance(value, str)
                and any(token in node_class for token in ("textencode", "prompt", "clip"))
            ):
                node["inputs"][field_name] = "prompt"
    return normalized


def _model_stack(graph: Mapping[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    models: set[str] = set()
    loras: set[str] = set()
    for node in graph.values():
        if not isinstance(node, Mapping):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        for field_name, value in inputs.items():
            folded_field = str(field_name).casefold()
            values = [value]
            if isinstance(value, Mapping):
                values.extend(value.values())
            for candidate in values:
                if not isinstance(candidate, str) or not candidate.casefold().endswith(_MODEL_EXTENSIONS):
                    continue
                basename = candidate.replace("\\", "/").rsplit("/", 1)[-1]
                if "lora" in folded_field:
                    strength = ""
                    if isinstance(value, Mapping):
                        raw_strength = value.get("strength") or value.get("strength_model")
                        strength = f"@{raw_strength}" if raw_strength is not None else ""
                    loras.add(f"{basename}{strength}")
                else:
                    models.add(basename)
    return tuple(sorted(models, key=str.casefold)), tuple(sorted(loras, key=str.casefold))


def _sampler_signature(graph: Mapping[str, Any]) -> tuple[str, ...]:
    signatures = []
    for node in graph.values():
        if not isinstance(node, Mapping):
            continue
        class_type = str(node.get("class_type") or "")
        if "sampler" not in class_type.casefold() and "scheduler" not in class_type.casefold():
            continue
        inputs = node.get("inputs") if isinstance(node.get("inputs"), Mapping) else {}
        settings = {
            key: inputs[key]
            for key in ("sampler_name", "scheduler", "steps", "cfg", "denoise")
            if key in inputs and not isinstance(inputs[key], (dict, list))
        }
        signatures.append(
            f"{class_type}:{json.dumps(settings, sort_keys=True, separators=(',', ':'))}"
        )
    return tuple(sorted(signatures, key=str.casefold))


def _source(manifest: Mapping[str, Any], image: Mapping[str, Any]) -> WorkflowSource:
    return WorkflowSource(
        guild_id=str(manifest.get("guild_id") or "unknown"),
        channel_id=str(image["channel_id"]),
        message_id=str(image["message_id"]),
        message_url=str(image["message_url"]),
        author_id=str(image.get("author_id") or "unknown"),
        timestamp=image["timestamp"],
        attachment_id=str(image.get("attachment_id") or image["message_id"]),
        attachment_filename=str(image.get("filename") or "attachment"),
        attachment_sha256=str(image.get("sha256") or "unknown"),
    )


def _workflow_id(identity: WorkflowIdentity) -> str:
    return hashlib.sha256(identity.key.encode("utf-8")).hexdigest()[:24]


def _graph_hashes(
    graph: Mapping[str, Any], manifest_hash: str
) -> tuple[str, str, bool]:
    identity_graph = _identity_graph(graph)
    try:
        return capability_hash(identity_graph), source_graph_hash(graph), False
    except (GraphHashError, GraphNormalizationError):
        fallback = hashlib.sha256(
            json.dumps(
                identity_graph,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return f"fallback:{fallback}", manifest_hash, True


def build_workflow_catalog(
    manifest_paths: Sequence[Path], graph_root: Path
) -> list[WorkflowRecord]:
    """Merge manifests and collapse only confidently equivalent workflows."""

    candidates: list[dict[str, Any]] = []
    seen_attachments: set[str] = set()
    for manifest_path in manifest_paths:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for image in manifest.get("images", []):
            source = _source(manifest, image)
            if source.source_key in seen_attachments:
                continue
            seen_attachments.add(source.source_key)
            api_refs = [item for item in image.get("embedded_graphs", []) if item.get("graph_kind") == "api"]
            ui_refs = [item for item in image.get("embedded_graphs", []) if item.get("graph_kind") == "ui"]
            if api_refs:
                for api_ref in api_refs:
                    relative_path = str(api_ref["relative_path"])
                    graph = json.loads((graph_root / relative_path).read_text(encoding="utf-8"))
                    evidence = _evidence_text(image, graph)
                    family, version = _classify_family(evidence)
                    models, loras = _model_stack(graph)
                    capability_digest, source_digest, fallback_review = _graph_hashes(
                        graph,
                        str(api_ref.get("sha256_canonical_json") or "unknown"),
                    )
                    identity = WorkflowIdentity(
                        family=family,
                        version=version,
                        task=_classify_task(evidence, family),
                        technique=_classify_technique(evidence),
                        capability_hash=capability_digest,
                        core_models=models,
                        loras=loras,
                        sampler_signature=_sampler_signature(graph),
                    )
                    candidates.append(
                        {
                            "identity": identity,
                            "source": source,
                            "api_path": relative_path,
                            "ui_paths": [str(item["relative_path"]) for item in ui_refs],
                            "source_hash": source_digest,
                            "image_hash": source.attachment_sha256,
                            "review": len(api_refs) > 1 or fallback_review,
                        }
                    )
            else:
                evidence = _evidence_text(image, None)
                family, version = _classify_family(evidence)
                for ui_ref in ui_refs:
                    relative_path = str(ui_ref["relative_path"])
                    ui_hash = str(ui_ref.get("sha256_canonical_json") or hashlib.sha256(relative_path.encode()).hexdigest())
                    identity = WorkflowIdentity(
                        family=family,
                        version=version,
                        task=_classify_task(evidence, family),
                        technique=_classify_technique(evidence),
                        capability_hash=f"ui:{ui_hash}",
                        core_models=(),
                        loras=(),
                        sampler_signature=(),
                    )
                    candidates.append(
                        {
                            "identity": identity,
                            "source": source,
                            "api_path": None,
                            "ui_paths": [relative_path],
                            "source_hash": ui_hash,
                            "image_hash": source.attachment_sha256,
                            "review": True,
                        }
                    )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["identity"].key, []).append(candidate)

    records = []
    for identity_key in sorted(grouped):
        items = grouped[identity_key]
        identity = items[0]["identity"]
        sources = {item["source"].source_key: item["source"] for item in items}
        records.append(
            WorkflowRecord(
                workflow_id=_workflow_id(identity),
                identity=identity,
                sources=tuple(sources[key] for key in sorted(sources)),
                api_graph_paths=tuple(sorted({item["api_path"] for item in items if item["api_path"]})),
                ui_graph_paths=tuple(sorted({path for item in items for path in item["ui_paths"]})),
                source_graph_hashes=tuple(sorted({item["source_hash"] for item in items})),
                image_hashes=tuple(sorted({item["image_hash"] for item in items})),
                identity_review_required=any(item["review"] for item in items),
            )
        )
    return sorted(records, key=lambda record: record.workflow_id)
