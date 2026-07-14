from __future__ import annotations

from collections.abc import Iterable
import json
import os
from pathlib import Path
import stat
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
)

from mp_comfyui_lab.security import sanitize_for_persistence

from .metrics import ComparisonReport


MAX_JUDGMENT_BYTES = 1_048_576
Score = Literal[1, 2, 3, 4, 5]
NonBlankReason = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=4096),
]


class JudgmentValidationError(ValueError):
    """Raised when an authored judgment is not bound to objective facts."""


class _JudgmentModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class VariantJudgment(_JudgmentModel):
    variant_id: Annotated[
        str,
        StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=256),
    ]
    identity_fidelity: Score
    prompt_adherence: Score
    visual_quality: Score
    creative_usefulness: Score
    reasons: tuple[NonBlankReason, ...] = Field(min_length=1, max_length=32)
    referenced_output_ids: tuple[
        Annotated[
            str,
            StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=512),
        ],
        ...,
    ] = Field(min_length=1, max_length=512)

    @field_validator(
        "identity_fidelity",
        "prompt_adherence",
        "visual_quality",
        "creative_usefulness",
        mode="before",
    )
    @classmethod
    def strict_score(cls, value: Any) -> Any:
        if type(value) is not int or value not in {1, 2, 3, 4, 5}:
            raise ValueError("score must be one of 1, 2, 3, 4, or 5")
        return value

    @field_validator("reasons", "referenced_output_ids", mode="before")
    @classmethod
    def require_json_arrays(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("field must be a JSON array")
        return tuple(value)

    @field_validator("referenced_output_ids")
    @classmethod
    def unique_output_references(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("referenced_output_ids must not contain duplicates")
        return values


class JudgmentSubmission(_JudgmentModel):
    schema_version: Literal["1"]
    experiment_id: Annotated[
        str,
        StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=96),
    ]
    facts_sha256: Annotated[
        str,
        StringConstraints(strict=True, pattern=r"^[0-9a-f]{64}$"),
    ]
    judgments: tuple[VariantJudgment, ...] = Field(min_length=1, max_length=512)

    @field_validator("judgments", mode="before")
    @classmethod
    def require_judgment_array(cls, value: Any) -> Any:
        if not isinstance(value, list):
            raise ValueError("judgments must be a JSON array")
        return tuple(value)


def load_and_validate_judgments(
    path: str | Path,
    report: ComparisonReport,
    *,
    expected_facts_sha256: str,
) -> JudgmentSubmission:
    document = _load_strict_json(Path(path))
    try:
        submission = JudgmentSubmission.model_validate(document)
    except ValidationError as error:
        raise JudgmentValidationError(
            f"judgments do not match the strict schema: {error}"
        ) from error
    if submission.experiment_id != report.experiment_id:
        raise JudgmentValidationError(
            "judgment experiment_id does not match comparison facts"
        )
    if submission.facts_sha256 != expected_facts_sha256:
        raise JudgmentValidationError(
            "judgment facts_sha256 does not match objective facts"
        )
    variant_ids = set(report.variants)
    seen_variants: set[str] = set()
    output_variants = {
        output.output_id: variant.variant_id
        for variant in report.variants.values()
        for output in variant.outputs
        if output.availability == "available"
        and output.artifact_relative_path is not None
    }
    for judgment in submission.judgments:
        if judgment.variant_id not in variant_ids:
            raise JudgmentValidationError(
                f"judgment references unknown variant_id {judgment.variant_id!r}"
            )
        if judgment.variant_id in seen_variants:
            raise JudgmentValidationError(
                f"duplicate judgment for variant_id {judgment.variant_id!r}"
            )
        seen_variants.add(judgment.variant_id)
        for output_id in judgment.referenced_output_ids:
            output_variant = output_variants.get(output_id)
            if output_variant is None:
                raise JudgmentValidationError(
                    f"judgment references unknown or unavailable output_id {output_id!r}"
                )
            if output_variant != judgment.variant_id:
                raise JudgmentValidationError(
                    "referenced output_id does not belong to the judged variant"
                )
    return submission.model_copy(
        update={
            "judgments": tuple(
                sorted(submission.judgments, key=lambda item: item.variant_id)
            )
        }
    )


def judgment_document_bytes(submission: JudgmentSubmission) -> bytes:
    payload = sanitize_for_persistence(submission.model_dump(mode="json"))
    return (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _load_strict_json(path: Path) -> Any:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise JudgmentValidationError("judgment input is not readable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise JudgmentValidationError("judgment input must be a regular file")
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            if (
                metadata.st_dev,
                metadata.st_ino,
                metadata.st_size,
                metadata.st_mtime_ns,
            ) != (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
            ):
                raise JudgmentValidationError(
                    "judgment input changed while opening"
                )
            if opened.st_size > MAX_JUDGMENT_BYTES:
                raise JudgmentValidationError(
                    "judgment input exceeds the size limit"
                )
            raw = handle.read(MAX_JUDGMENT_BYTES + 1)
            after = os.fstat(handle.fileno())
    except OSError as error:
        raise JudgmentValidationError("judgment input is not readable") from error
    if len(raw) > MAX_JUDGMENT_BYTES:
        raise JudgmentValidationError("judgment input exceeds the size limit")
    if (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(raw) != opened.st_size:
        raise JudgmentValidationError("judgment input changed while reading")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise JudgmentValidationError("judgment input must be UTF-8 JSON") from error
    try:
        return json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_nonfinite_number,
        )
    except JudgmentValidationError:
        raise
    except (json.JSONDecodeError, RecursionError) as error:
        raise JudgmentValidationError("judgment input must be valid JSON") from error


def _unique_object(items: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise JudgmentValidationError(
                f"duplicate JSON object key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite_number(value: str) -> Any:
    raise JudgmentValidationError(
        f"non-finite JSON number {value!r} is not allowed"
    )
