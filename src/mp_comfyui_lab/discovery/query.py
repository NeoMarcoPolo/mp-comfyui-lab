from __future__ import annotations

from importlib import resources
import re
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError
import yaml


MAX_QUERY_LENGTH = 512
MAX_CONCEPTS = 64
MAX_TERMS_PER_CONCEPT = 16
MAX_TERM_LENGTH = 128
MAX_EXPANDED_QUERIES = 64

BoundedQuery = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
    ),
]
ConceptName = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_]*$",
    ),
]
ExpansionTerm = Annotated[
    str,
    StringConstraints(
        strict=True,
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_TERM_LENGTH,
    ),
]


class QueryExpansionConfigError(ValueError):
    pass


class _FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
    )


class QueryConcept(_FrozenModel):
    concept: ConceptName
    terms: tuple[ExpansionTerm, ...] = Field(
        min_length=2,
        max_length=MAX_TERMS_PER_CONCEPT,
    )


class QueryExpansionMap(_FrozenModel):
    schema_version: int = Field(ge=1, le=1)
    concepts: tuple[QueryConcept, ...] = Field(
        min_length=1,
        max_length=MAX_CONCEPTS,
    )


class ExpansionRecord(_FrozenModel):
    concept: ConceptName
    source: ExpansionTerm
    expanded_terms: tuple[ExpansionTerm, ...] = Field(
        min_length=2,
        max_length=MAX_TERMS_PER_CONCEPT,
    )


class ExpandedQuery(_FrozenModel):
    original: BoundedQuery
    source_queries: tuple[BoundedQuery, ...] = Field(
        default=(),
        max_length=MAX_EXPANDED_QUERIES,
    )
    queries: tuple[BoundedQuery, ...] = Field(
        min_length=1,
        max_length=MAX_EXPANDED_QUERIES,
    )
    expansions: tuple[ExpansionRecord, ...] = Field(max_length=MAX_CONCEPTS)


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise QueryExpansionConfigError(f"duplicate YAML key: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _strict_yaml(document: str) -> Any:
    if not isinstance(document, str):
        raise QueryExpansionConfigError("query expansion YAML must be text")
    try:
        return yaml.load(document, Loader=_UniqueKeyLoader)
    except QueryExpansionConfigError:
        raise
    except yaml.YAMLError as error:
        raise QueryExpansionConfigError(f"malformed query expansion YAML: {error}") from error


def parse_query_expansions_yaml(document: str) -> QueryExpansionMap:
    payload = _strict_yaml(document)
    if not isinstance(payload, dict):
        raise QueryExpansionConfigError("query expansion document must be an object")
    concepts = payload.get("concepts")
    if not isinstance(concepts, list):
        raise QueryExpansionConfigError("concepts must be a list")

    normalized_concepts: list[dict[str, Any]] = []
    for index, value in enumerate(concepts):
        if not isinstance(value, dict):
            raise QueryExpansionConfigError(f"concepts[{index}] must be an object")
        terms = value.get("terms")
        if not isinstance(terms, list):
            raise QueryExpansionConfigError(f"concepts[{index}].terms must be a list")
        normalized_concepts.append({**value, "terms": tuple(terms)})

    normalized = {**payload, "concepts": tuple(normalized_concepts)}
    try:
        parsed = QueryExpansionMap.model_validate(normalized)
    except ValidationError as error:
        raise QueryExpansionConfigError(str(error)) from error

    concept_names: set[str] = set()
    term_owners: dict[str, str] = {}
    for concept in parsed.concepts:
        if concept.concept in concept_names:
            raise QueryExpansionConfigError(
                f"duplicate concept: {concept.concept}"
            )
        concept_names.add(concept.concept)
        local_terms: set[str] = set()
        for term in concept.terms:
            canonical = term.casefold()
            if canonical in local_terms:
                raise QueryExpansionConfigError(
                    f"duplicate term in concept {concept.concept}: {term}"
                )
            local_terms.add(canonical)
            owner = term_owners.get(canonical)
            if owner is not None:
                raise QueryExpansionConfigError(
                    f"ambiguous term {term!r} belongs to {owner} and {concept.concept}"
                )
            term_owners[canonical] = concept.concept
    return parsed


def load_query_expansions() -> QueryExpansionMap:
    try:
        document = (
            resources.files("mp_comfyui_lab.resources")
            .joinpath("query-expansions.yaml")
            .read_text(encoding="utf-8")
        )
    except (OSError, TypeError) as error:
        raise QueryExpansionConfigError(
            f"could not load query expansion resource: {error}"
        ) from error
    return parse_query_expansions_yaml(document)


def _term_pattern(term: str) -> re.Pattern[str]:
    pieces = re.findall(r"[^\W_]+", term, flags=re.UNICODE)
    is_ascii_phrase = bool(pieces) and all(piece.isascii() for piece in pieces)
    if is_ascii_phrase:
        body = r"[\W_]+".join(re.escape(piece) for piece in pieces)
        return re.compile(
            rf"(?<![A-Za-z0-9]){body}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
    return re.compile(re.escape(term), re.IGNORECASE)


def term_match(value: str, term: str) -> re.Match[str] | None:
    return _term_pattern(term).search(value)


def _unique(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        canonical = value.casefold()
        if canonical not in seen:
            seen.add(canonical)
            result.append(value)
    return tuple(result)


def expand_query(
    query: str,
    concepts: QueryExpansionMap | None = None,
) -> ExpandedQuery:
    base = ExpandedQuery(
        original=query,
        source_queries=(query,),
        queries=(query,),
        expansions=(),
    ).original

    expansion_map = load_query_expansions() if concepts is None else concepts
    queries = [base]
    records: list[ExpansionRecord] = []
    for concept in expansion_map.concepts:
        matched_term = next(
            (term for term in concept.terms if term_match(base, term) is not None),
            None,
        )
        if matched_term is None:
            continue
        records.append(
            ExpansionRecord(
                concept=concept.concept,
                source=matched_term,
                expanded_terms=concept.terms,
            )
        )
        pattern = _term_pattern(matched_term)
        for term in concept.terms:
            if term.casefold() == matched_term.casefold():
                continue
            expanded = pattern.sub(lambda _: term, base, count=1)
            if len(expanded) <= MAX_QUERY_LENGTH:
                queries.append(expanded)

    return ExpandedQuery(
        original=base,
        source_queries=(base,),
        queries=_unique(queries)[:MAX_EXPANDED_QUERIES],
        expansions=tuple(records),
    )
