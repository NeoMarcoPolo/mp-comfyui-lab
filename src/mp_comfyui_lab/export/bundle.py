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
    ArtifactIntegrityError,
    CanonicalReview,
    HumanSelection,
    SelectionRequiredError,
    StaleSelectionError,
    load_canonical_review,
    require_valid_selection,
)
from mp_comfyui_lab.comparison.paths import (
    hash_stored_artifact,
    path_component_is_link_or_junction,
    resolve_stored_artifact,
)
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


class BundleIntegrityError(ArtifactIntegrityError):
    """Raised when a durable source artifact is missing, unsafe, or changed."""


class BundleConflictError(BundleExportError):
    """Raised when a bundle path already contains different content."""


def suggested_workflow_name(
    *,
    base_model: str,
    workflow_type: str,
    description: str,
) -> str:
    base_component, full_type, full_description = tuple(
        _ascii_component(value, fallback=fallback)
        for value, fallback in (
            (_without_model_extension(base_model), "model"),
            (workflow_type, "workflow"),
            (description, "handoff"),
        )
    )
    type_component = full_type[:24].strip("_") or "workflow"
    description_component = (
        full_description[:72].strip("_") or "handoff"
    )
    suffix = f"_{type_component}_{description_component}"
    full_name = f"{base_component}{suffix}"
    needs_hash = (
        len(full_name) > 160
        or type_component != full_type
        or description_component != full_description
    )
    if not needs_hash:
        return full_name
    hash_preimage = f"{base_component}\0{full_type}\0{full_description}"
    short_hash = hashlib.sha256(hash_preimage.encode("ascii")).hexdigest()[:10]
    description_budget = max(1, 72 - len(short_hash) - 1)
    description_component = (
        description_component[:description_budget].rstrip("_")
        or "handoff"
    )
    description_component = f"{description_component}_{short_hash}"
    suffix = f"_{type_component}_{description_component}"
    base_budget = max(1, 160 - len(suffix))
    base_component = base_component[:base_budget].rstrip("_") or "model"
    return f"{base_component}{suffix}"


class BundleExporter:
    def __init__(self, store: ExperimentStore) -> None:
        self._store = store

    def export(self, experiment_id: str) -> Path:
        with self._store.run_lock(experiment_id, "finalist-selection"):
            return self._export_locked(experiment_id)

    def _export_locked(self, experiment_id: str) -> Path:
        selection = require_valid_selection(self._store, experiment_id)
        review = load_canonical_review(self._store, experiment_id)
        root = review.root
        if (
            selection.plan_hash != review.plan.plan_hash
            or selection.finalists_artifact_sha256
            != hashlib.sha256(review.finalists_bytes).hexdigest()
            or selection.review_generation_id != review.manifest.generation_id
            or selection.review_manifest_sha256
            != hashlib.sha256(review.manifest_bytes).hexdigest()
        ):
            raise StaleSelectionError(
                "selection no longer matches the canonical review generation"
            )
        plan = review.plan
        variant = self._load_variant(root, selection, plan)
        source = self._load_source(root, variant, plan)
        proposal = next(
            (
                item
                for item in review.finalists.finalists
                if item.variant_id == selection.selected_variant_id
            ),
            None,
        )
        if proposal is None:
            raise StaleSelectionError(
                "selected variant is not a current canonical finalist"
            )
        run = next(
            (
                item
                for item in review.runs
                if item.run_id == proposal.run_evidence.run_id
            ),
            None,
        )
        if run is None:
            raise ArtifactIntegrityError(
                "selected finalist run evidence is missing"
            )
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
        variant_export_root = export_root / selection.selected_variant_id
        try:
            variant_export_root.mkdir(exist_ok=True)
        except OSError as error:
            raise BundleConflictError(
                "variant export root cannot be created safely"
            ) from error
        self._require_safe_directory(
            variant_export_root,
            context="variant export root",
        )
        export_generation_id = hashlib.sha256(selection_bytes).hexdigest()
        destination = variant_export_root / export_generation_id
        stage = Path(
            tempfile.mkdtemp(
                prefix=f".{export_generation_id}.stage-",
                dir=variant_export_root,
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
                review=review,
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
        review: CanonicalReview,
        selection_bytes: bytes,
    ) -> None:
        (stage / "fixtures" / "files").mkdir(parents=True)
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
        self._write_review_contract(
            stage,
            selection=selection,
            review=review,
        )
        exported_outputs = self._bind_exported_outputs(
            stage,
            root,
            run,
            review,
        )
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
        self._write_json(
            stage / "comparison.json",
            {
                "schema_version": "1",
                "generation_id": review.manifest.generation_id,
                "facts_sha256": review.manifest.comparison_facts_sha256,
                "facts": review.comparison.model_dump(mode="json"),
                "judgments_sha256": review.manifest.judgments_sha256,
                "judgments": (
                    None
                    if review.judgments is None
                    else review.judgments.model_dump(mode="json")
                ),
            },
        )
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
        plan: ExperimentPlan,
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
        identities = tuple(
            item
            for item in plan.source_provenance
            if item.workflow_id == variant.source_workflow_id
            and item.source_hash == variant.source_hash
        )
        if len(identities) != 1:
            raise BundleIntegrityError(
                "selected source lacks immutable approved provenance"
            )
        identity = identities[0]
        if identity.capability_hash != capability_hash(graph):
            raise BundleIntegrityError("source capability evidence changed")
        return {
            **source,
            **identity.model_dump(mode="json"),
            "raw_graph": graph,
        }

    def _copy_fixtures(
        self,
        stage: Path,
        root: Path,
        run: RunRecord,
        variant: PlannedVariant,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        fixture_relative = Path("runs", run.run_id, "fixtures")
        fixture_root = root / fixture_relative
        if not variant.fixture_fingerprints:
            if fixture_root.exists() or fixture_root.is_symlink():
                fixture_root = self._require_confined_directory(
                    root,
                    fixture_relative,
                    context="selected fixture snapshot directory",
                )
                if any(fixture_root.iterdir()):
                    raise BundleIntegrityError(
                        "selected run has unapproved fixture snapshots"
                    )
            return entries
        fixture_root = self._require_confined_directory(
            root,
            fixture_relative,
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

    def _bind_exported_outputs(
        self,
        stage: Path,
        root: Path,
        run: RunRecord,
        review: CanonicalReview,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        review_artifacts = {
            artifact.path: artifact for artifact in review.manifest.artifacts
        }
        for output in sorted(run.outputs, key=lambda item: item.ordinal):
            source, availability = resolve_stored_artifact(
                root,
                output.local_relative_path,
            )
            if source is None:
                raise BundleIntegrityError(
                    f"recorded output is {availability}"
                )
            manifest_entry = review_artifacts.get(output.local_relative_path)
            if (
                manifest_entry is None
                or manifest_entry.sha256 != output.content_sha256
                or manifest_entry.bytes != output.size
            ):
                raise BundleIntegrityError(
                    "selected output is not bound by the review manifest"
                )
            exported, availability = resolve_stored_artifact(
                stage,
                output.local_relative_path,
            )
            if exported is None:
                raise BundleIntegrityError(
                    f"exported selected output is {availability}"
                )
            try:
                source_digest, source_size = hash_stored_artifact(source)
                exported_digest, exported_size = hash_stored_artifact(exported)
            except OSError as error:
                raise BundleIntegrityError(
                    "selected output cannot be verified safely"
                ) from error
            if (
                source_digest != output.content_sha256
                or source_size != output.size
                or exported_digest != output.content_sha256
                or exported_size != output.size
            ):
                raise BundleIntegrityError(
                    "selected output differs from reviewed evidence"
                )
            entries.append(
                {
                    "ordinal": output.ordinal,
                    "provider_order": output.provider_order,
                    "node_id": output.node_id,
                    "media_type": output.media_type,
                    "path": output.local_relative_path,
                    "sha256": output.content_sha256,
                    "bytes": output.size,
                    "provenance_path": output.local_relative_path,
                }
            )
        return entries

    def _copy_review_artifacts(
        self,
        stage: Path,
        review: CanonicalReview,
    ) -> None:
        generation_root = review.manifest_path.parent
        for artifact in review.manifest.artifacts:
            source, availability = resolve_stored_artifact(
                generation_root,
                artifact.path,
            )
            if source is None:
                raise BundleIntegrityError(
                    f"review artifact is {availability}: {artifact.path}"
                )
            self._copy_verified(
                source,
                stage / artifact.path,
                expected_sha256=artifact.sha256,
                expected_size=artifact.bytes,
                context="review artifact",
            )

    def _write_review_contract(
        self,
        stage: Path,
        *,
        selection: HumanSelection,
        review: CanonicalReview,
    ) -> None:
        canonical_sha256 = hashlib.sha256(review.manifest_bytes).hexdigest()
        if canonical_sha256 != selection.review_manifest_sha256:
            raise StaleSelectionError(
                "selected review manifest hash changed during export"
            )
        canonical_relative = Path("review-manifest.json")
        self._write_bytes(
            stage / canonical_relative,
            review.manifest_bytes,
        )
        self._copy_review_artifacts(stage, review)
        review_root = (stage / canonical_relative).parent
        for artifact in review.manifest.artifacts:
            exported, availability = resolve_stored_artifact(
                review_root,
                artifact.path,
            )
            if exported is None:
                raise BundleIntegrityError(
                    "exported review manifest path is "
                    f"{availability}: {artifact.path}"
                )
            try:
                digest, size = hash_stored_artifact(exported)
            except OSError as error:
                raise BundleIntegrityError(
                    "exported review artifact cannot be verified"
                ) from error
            if digest != artifact.sha256 or size != artifact.bytes:
                raise BundleIntegrityError(
                    "exported review artifact differs from its manifest"
                )

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
    def _require_confined_directory(
        trusted_root: Path,
        relative_path: Path,
        *,
        context: str,
    ) -> Path:
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise BundleIntegrityError(f"{context} path is unsafe")
        current = trusted_root
        for component in relative_path.parts:
            current /= component
            try:
                metadata = current.lstat()
            except OSError as error:
                raise BundleIntegrityError(
                    f"{context} is not a safe local directory"
                ) from error
            if (
                path_component_is_link_or_junction(current, metadata)
                or not stat.S_ISDIR(metadata.st_mode)
            ):
                raise BundleIntegrityError(
                    f"{context} contains an unsafe path component"
                )
        try:
            trusted = trusted_root.resolve(strict=True)
            resolved = current.resolve(strict=True)
            resolved.relative_to(trusted)
        except (OSError, ValueError) as error:
            raise BundleIntegrityError(f"{context} escapes its trusted root") from error
        return resolved

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
