from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import unicodedata
from typing import Any

from pydantic import ValidationError
import yaml

from mp_comfyui_lab.comparison.finalists import (
    FinalistSelectionError,
    FinalistsArtifact,
    HumanSelection,
    require_valid_selection,
)
from mp_comfyui_lab.comparison.metrics import ComparisonReport
from mp_comfyui_lab.comparison.paths import (
    hash_stored_artifact,
    path_component_is_link_or_junction,
    resolve_stored_artifact,
)
from mp_comfyui_lab.comparison.render import comparison_facts_sha256
from mp_comfyui_lab.comparison.render import _rename_directory_new
from mp_comfyui_lab.experiments.models import (
    ExperimentPlan,
    FixtureFingerprint,
    PlannedVariant,
    RunRecord,
)
from mp_comfyui_lab.experiments.planning import recompute_exact_run_hash
from mp_comfyui_lab.experiments.store import ExperimentStore
from mp_comfyui_lab.graph.hashing import capability_hash, source_graph_hash
from mp_comfyui_lab.json_types import thaw_json
from mp_comfyui_lab.security import SafetyError, safe_filename


MAX_EXPORT_SOURCE_BYTES = 512 * 1024 * 1024
_ASCII_COMPONENT = re.compile(r"[^a-z0-9]+")
_MODEL_FIELDS = frozenset(
    {"ckpt_name", "checkpoint", "model_name", "unet_name", "diffusion_model"}
)
_LORA_FIELDS = frozenset({"lora_name", "lora_file", "lora"})


class BundleExportError(ValueError):
    """Base class for deterministic handoff export failures."""


class SelectionRequiredError(BundleExportError):
    """Raised when export is not bound to a current explicit selection."""


class BundleIntegrityError(BundleExportError):
    """Raised when a durable source artifact is missing, unsafe, or changed."""


class BundleConflictError(BundleExportError):
    """Raised when a bundle path already contains different content."""


def suggested_workflow_name(
    *,
    base_model: str,
    workflow_type: str,
    description: str,
) -> str:
    components = tuple(
        _ascii_component(value, fallback=fallback)
        for value, fallback in (
            (_without_model_extension(base_model), "model"),
            (workflow_type, "workflow"),
            (description, "handoff"),
        )
    )
    return "_".join(components)[:160].rstrip("_")


class BundleExporter:
    def __init__(self, store: ExperimentStore) -> None:
        self._store = store

    def export(self, experiment_id: str) -> Path:
        with self._store.run_lock(experiment_id, "finalist-selection"):
            return self._export_locked(experiment_id)

    def _export_locked(self, experiment_id: str) -> Path:
        try:
            selection = require_valid_selection(self._store, experiment_id)
        except FinalistSelectionError as error:
            raise SelectionRequiredError(str(error)) from error

        root = self._store.root(experiment_id)
        finalists = self._load_finalists(root, experiment_id)
        plan = self._load_plan(root, selection)
        variant = self._load_variant(root, selection, plan)
        source = self._load_source(root, variant)
        comparison, comparison_bytes = self._load_comparison(
            root,
            finalists,
            selection,
        )
        run = self._load_selected_run(root, comparison, selection, variant)
        selection_bytes = self._read_stored_bytes(root, "selection.json")
        try:
            persisted_selection = HumanSelection.model_validate_json(
                selection_bytes,
                strict=True,
            )
        except ValidationError as error:
            raise BundleIntegrityError("selection artifact changed") from error
        if persisted_selection != selection:
            raise BundleIntegrityError("selection artifact changed")

        export_root = root / "export"
        self._require_safe_directory(export_root, context="export root")
        try:
            safe_filename(selection.selected_variant_id)
        except SafetyError as error:
            raise BundleIntegrityError("selected variant ID is unsafe") from error
        destination = export_root / selection.selected_variant_id
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{selection.selected_variant_id}.stage-",
                dir=export_root,
            )
        )
        try:
            self._stage_bundle(
                stage,
                experiment_id=experiment_id,
                selection=selection,
                plan=plan,
                variant=variant,
                source=source,
                run=run,
                root=root,
                comparison_bytes=comparison_bytes,
                selection_bytes=selection_bytes,
            )
            self._publish(stage, destination)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        return destination.resolve()

    def _stage_bundle(
        self,
        stage: Path,
        *,
        experiment_id: str,
        selection: HumanSelection,
        plan: ExperimentPlan,
        variant: PlannedVariant,
        source: dict[str, Any],
        run: RunRecord,
        root: Path,
        comparison_bytes: bytes,
        selection_bytes: bytes,
    ) -> None:
        (stage / "fixtures" / "files").mkdir(parents=True)
        (stage / "outputs").mkdir()
        selected_graph = thaw_json(variant.materialized_graph)
        source_graph = source["raw_graph"]
        self._write_json(stage / "source-workflow-api.json", source_graph)
        self._write_json(stage / "selected-workflow-api.json", selected_graph)
        self._write_json(
            stage / "semantic-patch.json",
            (
                {"mutations": []}
                if variant.semantic_patch is None
                else thaw_json(variant.semantic_patch)
            ),
        )
        model_stack = _model_stack(variant, source)
        self._write_json(stage / "model-stack.json", model_stack)

        fixture_entries = self._copy_fixtures(stage, root, run, variant)
        self._write_json(
            stage / "fixtures" / "manifest.json",
            {
                "schema_version": "1",
                "experiment_id": experiment_id,
                "variant_id": variant.variant_id,
                "run_id": run.run_id,
                "fixtures": fixture_entries,
            },
        )
        parameters = thaw_json(variant.execution_parameters)
        prompt_lines = parameters.get("prompt_lines")
        if not isinstance(prompt_lines, list) or any(
            not isinstance(item, str) for item in prompt_lines
        ):
            raise BundleIntegrityError(
                "selected variant prompt lines are missing or invalid"
            )
        self._write_json(
            stage / "prompts.json",
            {"prompt_lines": prompt_lines},
        )
        exported_outputs = self._copy_outputs(stage, root, run)
        self._write_json(
            stage / "execution.json",
            {
                "schema_version": "1",
                "experiment_id": experiment_id,
                "plan_hash": plan.plan_hash,
                "variant_id": variant.variant_id,
                "run": run.model_dump(mode="json"),
                "exported_outputs": exported_outputs,
            },
        )
        self._write_bytes(stage / "comparison.json", comparison_bytes)
        self._write_bytes(stage / "selection.json", selection_bytes)

        description = self._brief_goal(root, plan)
        workflow_name = suggested_workflow_name(
            base_model=model_stack["base_model"],
            workflow_type=_workflow_type(variant),
            description=description,
        )
        artifacts = tuple(
            {
                "path": path.relative_to(stage).as_posix(),
                "sha256": digest,
                "bytes": size,
            }
            for path, digest, size in self._artifact_hashes(stage)
        )
        self._write_json(
            stage / "handoff.json",
            {
                "schema_version": "1",
                "experiment_id": experiment_id,
                "variant_id": variant.variant_id,
                "suggested_workflow_name": workflow_name,
                "artifacts": artifacts,
                "instructions": [
                    (
                        "Upload or import selected-workflow-api.json to "
                        "RunningHub as a saved workflow."
                    ),
                    "Obtain the URL of the saved RunningHub workflow.",
                    (
                        "Invoke port-comfy-workflow with that URL through strip, "
                        "audit, register, smoke, and promote."
                    ),
                ],
            },
        )
        self._fsync_tree(stage)

    def _load_finalists(
        self,
        root: Path,
        experiment_id: str,
    ) -> FinalistsArtifact:
        try:
            finalists = FinalistsArtifact.model_validate_json(
                self._read_stored_bytes(root, "finalists.json"),
                strict=True,
            )
        except (ValidationError, ValueError) as error:
            raise BundleIntegrityError("finalists artifact is invalid") from error
        if finalists.experiment_id != experiment_id:
            raise BundleIntegrityError("finalists experiment identity changed")
        return finalists

    def _load_plan(
        self,
        root: Path,
        selection: HumanSelection,
    ) -> ExperimentPlan:
        try:
            plan = ExperimentPlan.model_validate_json(
                self._read_stored_bytes(
                    root,
                    f"plans/{selection.plan_hash}.json",
                ),
                strict=True,
            )
        except (ValidationError, ValueError) as error:
            raise BundleIntegrityError("approved plan artifact is invalid") from error
        if plan.plan_hash != selection.plan_hash:
            raise BundleIntegrityError("approved plan hash changed")
        return plan

    def _load_variant(
        self,
        root: Path,
        selection: HumanSelection,
        plan: ExperimentPlan,
    ) -> PlannedVariant:
        selected = tuple(
            item
            for item in plan.variants
            if item.variant_id == selection.selected_variant_id
        )
        if len(selected) != 1:
            raise BundleIntegrityError(
                "selected variant is not present exactly once in the approved plan"
            )
        try:
            artifact = PlannedVariant.model_validate_json(
                self._read_stored_bytes(
                    root,
                    f"variants/{selection.selected_variant_id}.json",
                ),
                strict=True,
            )
        except (ValidationError, ValueError) as error:
            raise BundleIntegrityError("selected variant artifact is invalid") from error
        if artifact != selected[0]:
            raise BundleIntegrityError(
                "selected variant artifact differs from the approved plan"
            )
        if recompute_exact_run_hash(artifact) != artifact.exact_run_hash:
            raise BundleIntegrityError("selected variant exact run hash changed")
        if (
            capability_hash(thaw_json(artifact.materialized_graph))
            != artifact.capability_hash
        ):
            raise BundleIntegrityError("selected workflow capability hash changed")
        return artifact

    def _load_source(
        self,
        root: Path,
        variant: PlannedVariant,
    ) -> dict[str, Any]:
        document = self._read_json(root, "sources/source-bundle.json")
        if not isinstance(document, Mapping) or not isinstance(
            document.get("sources"), list
        ):
            raise BundleIntegrityError("source bundle is invalid")
        matches = tuple(
            item
            for item in document["sources"]
            if isinstance(item, dict)
            and item.get("workflow_id") == variant.source_workflow_id
            and item.get("source_hash") == variant.source_hash
        )
        if len(matches) != 1:
            raise BundleIntegrityError(
                "selected workflow source is not present exactly once"
            )
        source = matches[0]
        graph = source.get("raw_graph")
        if not isinstance(graph, dict):
            raise BundleIntegrityError("source workflow API graph is invalid")
        try:
            actual_hash = source_graph_hash(graph)
        except ValueError as error:
            raise BundleIntegrityError("source workflow API graph is invalid") from error
        if actual_hash != variant.source_hash:
            raise BundleIntegrityError("source workflow hash changed")
        return source

    def _load_comparison(
        self,
        root: Path,
        finalists: FinalistsArtifact,
        selection: HumanSelection,
    ) -> tuple[ComparisonReport, bytes]:
        comparison_bytes = self._read_stored_bytes(root, "comparisons/facts.json")
        if hashlib.sha256(comparison_bytes).hexdigest() != (
            finalists.comparison_facts_sha256
        ):
            raise BundleIntegrityError("comparison artifact hash changed")
        try:
            comparison = ComparisonReport.model_validate_json(
                comparison_bytes,
                strict=True,
            )
        except ValidationError as error:
            raise BundleIntegrityError("comparison artifact is invalid") from error
        if comparison_facts_sha256(comparison) != finalists.comparison_facts_sha256:
            raise BundleIntegrityError("comparison artifact is not canonical")
        if selection.selected_variant_id not in comparison.variants:
            raise BundleIntegrityError(
                "selected variant is missing from comparison facts"
            )
        return comparison, comparison_bytes

    def _load_selected_run(
        self,
        root: Path,
        comparison: ComparisonReport,
        selection: HumanSelection,
        variant: PlannedVariant,
    ) -> RunRecord:
        run_id = comparison.variants[selection.selected_variant_id].run_id
        try:
            safe_filename(run_id)
            run = RunRecord.model_validate_json(
                self._read_stored_bytes(root, f"runs/{run_id}.json"),
                strict=True,
            )
        except (SafetyError, ValidationError, ValueError) as error:
            raise BundleIntegrityError("selected run artifact is invalid") from error
        if (
            run.run_id != run_id
            or run.experiment_id != selection.experiment_id
            or run.variant_id != variant.variant_id
            or run.plan_hash != selection.plan_hash
            or run.exact_run_hash != variant.exact_run_hash
        ):
            raise BundleIntegrityError("selected run provenance changed")
        snapshot = (
            None if run.variant_snapshot is None else thaw_json(run.variant_snapshot)
        )
        if snapshot != variant.model_dump(mode="json"):
            raise BundleIntegrityError("selected run variant snapshot changed")
        return run

    def _copy_fixtures(
        self,
        stage: Path,
        root: Path,
        run: RunRecord,
        variant: PlannedVariant,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        fixture_root = root / "runs" / run.run_id / "fixtures"
        if not variant.fixture_fingerprints:
            if fixture_root.exists():
                self._require_safe_directory(
                    fixture_root,
                    context="selected fixture snapshot directory",
                )
                if any(fixture_root.iterdir()):
                    raise BundleIntegrityError(
                        "selected run has unapproved fixture snapshots"
                    )
            return entries
        self._require_safe_directory(
            fixture_root,
            context="selected fixture snapshot directory",
        )
        for fixture in variant.fixture_fingerprints:
            source = self._fixture_snapshot(fixture_root, fixture)
            relative = Path("files", source.name)
            destination = stage / "fixtures" / relative
            self._copy_verified(
                source,
                destination,
                expected_sha256=fixture.sha256,
                expected_size=fixture.size,
                context="fixture snapshot",
            )
            entries.append(
                {
                    "uri": fixture.uri,
                    "binding_index": fixture.binding_index,
                    "media_type": fixture.media_type,
                    "path": relative.as_posix(),
                    "sha256": fixture.sha256,
                    "bytes": fixture.size,
                    "provenance_path": source.relative_to(root).as_posix(),
                }
            )
        return entries

    def _fixture_snapshot(
        self,
        fixture_root: Path,
        fixture: FixtureFingerprint,
    ) -> Path:
        prefix = f"{fixture.binding_index:04d}-{fixture.sha256[:16]}"
        try:
            paths = tuple(
                path
                for path in fixture_root.iterdir()
                if path.name.startswith(prefix)
            )
        except OSError as error:
            raise BundleIntegrityError(
                "fixture snapshot directory cannot be listed"
            ) from error
        if len(paths) != 1:
            raise BundleIntegrityError(
                "fixture snapshot is missing or ambiguous"
            )
        source = paths[0]
        try:
            safe_filename(source.name)
        except SafetyError as error:
            raise BundleIntegrityError("fixture snapshot name is unsafe") from error
        if path_component_is_link_or_junction(source):
            raise BundleIntegrityError("fixture snapshot is a symlink or junction")
        return source

    def _copy_outputs(
        self,
        stage: Path,
        root: Path,
        run: RunRecord,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for output in sorted(run.outputs, key=lambda item: item.ordinal):
            source, availability = resolve_stored_artifact(
                root,
                output.local_relative_path,
            )
            if source is None:
                raise BundleIntegrityError(
                    f"recorded output is {availability}"
                )
            suffix = source.suffix.casefold() or ".bin"
            destination_relative = Path("outputs", f"{output.ordinal:04d}{suffix}")
            self._copy_verified(
                source,
                stage / destination_relative,
                expected_sha256=output.content_sha256,
                expected_size=output.size,
                context="recorded output",
            )
            entries.append(
                {
                    "ordinal": output.ordinal,
                    "provider_order": output.provider_order,
                    "node_id": output.node_id,
                    "media_type": output.media_type,
                    "path": destination_relative.as_posix(),
                    "sha256": output.content_sha256,
                    "bytes": output.size,
                    "provenance_path": output.local_relative_path,
                }
            )
        return entries

    def _brief_goal(self, root: Path, plan: ExperimentPlan) -> str:
        payload = thaw_json(plan.approval_payload)
        if isinstance(payload, Mapping):
            brief = payload.get("brief")
            if isinstance(brief, Mapping):
                goal = brief.get("goal")
                if isinstance(goal, str) and goal.strip():
                    return goal
        try:
            brief_document = yaml.safe_load(
                self._read_stored_bytes(root, "brief.yaml").decode("utf-8")
            )
        except (UnicodeDecodeError, yaml.YAMLError, ValueError) as error:
            raise BundleIntegrityError("experiment brief is invalid") from error
        if not isinstance(brief_document, Mapping):
            raise BundleIntegrityError("experiment brief is invalid")
        goal = brief_document.get("goal")
        if not isinstance(goal, str) or not goal.strip():
            raise BundleIntegrityError("experiment brief goal is missing")
        return goal

    def _read_json(self, root: Path, relative_path: str) -> Any:
        try:
            return json.loads(self._read_stored_bytes(root, relative_path))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise BundleIntegrityError(
                f"stored JSON artifact is invalid: {relative_path}"
            ) from error

    @staticmethod
    def _read_stored_bytes(root: Path, relative_path: str) -> bytes:
        source, availability = resolve_stored_artifact(root, relative_path)
        if source is None:
            raise BundleIntegrityError(
                f"stored artifact is {availability}: {relative_path}"
            )
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(source, flags)
        except OSError as error:
            raise BundleIntegrityError(
                f"stored artifact cannot be read safely: {relative_path}"
            ) from error
        chunks: list[bytes] = []
        size = 0
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise BundleIntegrityError(
                    f"stored artifact is not a regular file: {relative_path}"
                )
            if before.st_size > MAX_EXPORT_SOURCE_BYTES:
                raise BundleIntegrityError(
                    f"stored artifact exceeds export limit: {relative_path}"
                )
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_EXPORT_SOURCE_BYTES:
                    raise BundleIntegrityError(
                        f"stored artifact exceeds export limit: {relative_path}"
                    )
                chunks.append(chunk)
            after = os.fstat(descriptor)
        except OSError as error:
            raise BundleIntegrityError(
                f"stored artifact cannot be read safely: {relative_path}"
            ) from error
        finally:
            os.close(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise BundleIntegrityError(
                f"stored artifact changed during export: {relative_path}"
            )
        return b"".join(chunks)

    @staticmethod
    def _copy_verified(
        source: Path,
        destination: Path,
        *,
        expected_sha256: str,
        expected_size: int,
        context: str,
    ) -> None:
        if expected_size > MAX_EXPORT_SOURCE_BYTES:
            raise BundleIntegrityError(f"{context} exceeds export limit")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(source, flags)
        except OSError as error:
            raise BundleIntegrityError(f"{context} cannot be opened safely") from error
        digest = hashlib.sha256()
        size = 0
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise BundleIntegrityError(f"{context} is not a regular file")
            with destination.open("xb") as output:
                while True:
                    chunk = os.read(descriptor, 1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_EXPORT_SOURCE_BYTES:
                        raise BundleIntegrityError(
                            f"{context} exceeds export limit"
                        )
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        if _stat_identity(before) != _stat_identity(after):
            raise BundleIntegrityError(f"{context} changed during export")
        if size != expected_size or digest.hexdigest() != expected_sha256:
            raise BundleIntegrityError(f"{context} hash or size changed")

    @staticmethod
    def _write_json(destination: Path, value: Any) -> None:
        content = (
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
        BundleExporter._write_bytes(destination, content)

    @staticmethod
    def _write_bytes(destination: Path, content: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with destination.open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError as error:
            raise BundleConflictError(
                "staged bundle contains duplicate artifact paths"
            ) from error

    @staticmethod
    def _artifact_hashes(stage: Path) -> tuple[tuple[Path, str, int], ...]:
        artifacts: list[tuple[Path, str, int]] = []
        for path in sorted(stage.rglob("*"), key=lambda item: item.as_posix()):
            if not path.is_file():
                continue
            digest, size = hash_stored_artifact(path)
            artifacts.append((path, digest, size))
        return tuple(artifacts)

    @staticmethod
    def _publish(stage: Path, destination: Path) -> None:
        if destination.exists() or destination.is_symlink():
            BundleExporter._verify_identical_bundle(stage, destination)
            return
        try:
            _rename_directory_new(stage, destination)
        except FileExistsError:
            BundleExporter._verify_identical_bundle(stage, destination)
        except OSError as error:
            if destination.exists() or destination.is_symlink():
                BundleExporter._verify_identical_bundle(stage, destination)
                return
            raise BundleConflictError("bundle could not be published atomically") from error
        except SafetyError as error:
            raise BundleConflictError("bundle could not be published atomically") from error
        BundleExporter._fsync_directory(destination.parent)

    @staticmethod
    def _verify_identical_bundle(expected: Path, actual: Path) -> None:
        if path_component_is_link_or_junction(actual) or not actual.is_dir():
            raise BundleConflictError(
                "bundle path already exists and is not a safe directory"
            )
        expected_entries = BundleExporter._tree_manifest(expected)
        actual_entries = BundleExporter._tree_manifest(actual)
        if expected_entries != actual_entries:
            raise BundleConflictError(
                "bundle already exists with different content"
            )

    @staticmethod
    def _tree_manifest(root: Path) -> tuple[tuple[str, str, int], ...]:
        entries: list[tuple[str, str, int]] = []
        try:
            paths = tuple(root.rglob("*"))
        except OSError as error:
            raise BundleConflictError("bundle tree cannot be read safely") from error
        for path in paths:
            if path_component_is_link_or_junction(path):
                raise BundleConflictError("bundle tree contains a symlink or junction")
            if path.is_dir():
                continue
            if not path.is_file():
                raise BundleConflictError("bundle tree contains a non-file entry")
            try:
                digest, size = hash_stored_artifact(path)
            except OSError as error:
                raise BundleConflictError(
                    "bundle tree changed while being verified"
                ) from error
            entries.append((path.relative_to(root).as_posix(), digest, size))
        return tuple(sorted(entries))

    @staticmethod
    def _require_safe_directory(path: Path, *, context: str) -> None:
        if path_component_is_link_or_junction(path) or not path.is_dir():
            raise BundleIntegrityError(f"{context} is not a safe local directory")

    @staticmethod
    def _fsync_tree(root: Path) -> None:
        directories = sorted(
            (path for path in root.rglob("*") if path.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        )
        for directory in directories:
            BundleExporter._fsync_directory(directory)
        BundleExporter._fsync_directory(root)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _model_stack(
    variant: PlannedVariant,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    graph = thaw_json(variant.materialized_graph)
    models: list[dict[str, str]] = []
    loras: list[dict[str, str]] = []
    for node_id, node in sorted(graph.items(), key=lambda item: _node_key(item[0])):
        if not isinstance(node, Mapping):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        for field_name, value in sorted(inputs.items()):
            if not isinstance(value, str) or not value.strip():
                continue
            entry = {
                "node_id": node_id,
                "class_type": str(node.get("class_type", "")),
                "field": field_name,
                "value": value,
            }
            canonical = field_name.casefold()
            if canonical in _MODEL_FIELDS:
                models.append(entry)
            elif canonical in _LORA_FIELDS:
                loras.append(entry)
    if not models:
        raise BundleIntegrityError("selected workflow has no identifiable base model")
    base_model = _without_model_extension(models[0]["value"])
    return {
        "schema_version": "1",
        "base_model": base_model,
        "models": models,
        "loras": loras,
        "source_workflow_id": variant.source_workflow_id,
        "source_url": source.get("canonical_url"),
        "source_hash": variant.source_hash,
        "selected_capability_hash": variant.capability_hash,
    }


def _workflow_type(variant: PlannedVariant) -> str:
    if variant.fixture_fingerprints:
        return "i2i"
    media_types = tuple(value.casefold() for value in variant.expected_media_types)
    if any(value.startswith("video/") for value in media_types):
        return "t2v"
    if any(value.startswith("audio/") for value in media_types):
        return "t2a"
    return "t2i"


def _without_model_extension(value: str) -> str:
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    lowered = name.casefold()
    for suffix in (".safetensors", ".checkpoint", ".ckpt", ".pt", ".pth"):
        if lowered.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _ascii_component(value: str, *, fallback: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").casefold()
    component = _ASCII_COMPONENT.sub("_", ascii_value).strip("_")
    return component or fallback


def _node_key(node_id: str) -> tuple[int, int | str, str]:
    if node_id.isascii() and node_id.isdecimal():
        return (0, int(node_id), node_id)
    return (1, node_id.casefold(), node_id)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
    )
