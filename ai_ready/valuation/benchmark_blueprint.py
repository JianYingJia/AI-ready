"""Auditable construction of benchmark knowledge--capability blueprints.

The builder deliberately separates three concerns:

1. a frozen upper ontology defines comparable cross-benchmark dimensions;
2. humans or LLMs annotate public development items against that ontology;
3. deterministic consensus and aggregation create a question-free blueprint.

The module never reads hidden test items and does not call an LLM itself.  It
emits structured prompts so that any reviewed inference backend can be used,
then validates and aggregates the returned annotations reproducibly.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai_ready.utils.io import atomic_text, read_jsonl, write_json, write_jsonl

PUBLIC_SPLIT_ROLES = {"task_spec", "public_train", "public_dev", "public_validation"}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OntologyLabel(StrictModel):
    label_id: str = Field(min_length=1)
    description: str = Field(min_length=1)


class BenchmarkOntology(StrictModel):
    schema_version: str = "benchmark-upper-ontology-v1"
    ontology_id: str = Field(min_length=1)
    knowledge_domains: list[OntologyLabel] = Field(min_length=1)
    capabilities: list[OntologyLabel] = Field(min_length=1)
    problem_structures: list[OntologyLabel] = Field(min_length=1)
    response_types: list[OntologyLabel] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_labels(self) -> "BenchmarkOntology":
        for field in ("knowledge_domains", "capabilities", "problem_structures", "response_types"):
            labels = getattr(self, field)
            identifiers = [label.label_id for label in labels]
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"{field} contains duplicate label IDs")
        return self


class BenchmarkItem(StrictModel):
    benchmark_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    split_role: str
    question: str = Field(min_length=1)
    options: list[str] = Field(default_factory=list)
    answer: str | None = None
    rationale: str | None = None
    score_weight: float = Field(default=1.0, gt=0)

    @field_validator("split_role")
    @classmethod
    def public_split_only(cls, value: str) -> str:
        normalized = value.strip().casefold()
        if normalized not in PUBLIC_SPLIT_ROLES:
            raise ValueError(
                "Blueprint construction only accepts task specifications or public train/dev/validation items; "
                f"received split_role={value!r}"
            )
        return normalized


class ProposedTag(StrictModel):
    axis: str = Field(min_length=1)
    label_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    justification: str = Field(min_length=1)


class ItemAnnotation(StrictModel):
    annotation_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    annotator_id: str = Field(min_length=1)
    knowledge_domain_id: str = Field(min_length=1)
    knowledge_topic_ids: list[str] = Field(default_factory=list)
    capability_ids: list[str] = Field(min_length=1)
    problem_structure_ids: list[str] = Field(default_factory=list)
    response_type_id: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    proposed_tags: list[ProposedTag] = Field(default_factory=list)
    stage: str = "primary"
    source_annotation_ids: list[str] = Field(default_factory=list)

    @field_validator("knowledge_topic_ids", "capability_ids", "problem_structure_ids")
    @classmethod
    def unique_nonempty_tags(cls, values: list[str]) -> list[str]:
        normalized = [value.strip() for value in values]
        if any(not value for value in normalized):
            raise ValueError("Tag lists cannot contain empty IDs")
        if len(normalized) != len(set(normalized)):
            raise ValueError("Tag lists cannot contain duplicate IDs")
        return normalized


class ConsensusConfig(StrictModel):
    minimum_annotations: int = Field(default=2, ge=1)
    minimum_agreement: float = Field(default=2 / 3, gt=0.0, le=1.0)
    minimum_mean_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    require_structure_consensus: bool = False


class ConsensusAnnotation(StrictModel):
    item_id: str
    status: str
    knowledge_domain_id: str | None
    knowledge_topic_ids: list[str]
    capability_ids: list[str]
    problem_structure_ids: list[str]
    response_type_id: str | None
    mean_confidence: float
    agreement: dict[str, float]
    annotation_ids: list[str]
    review_reasons: list[str]
    proposed_tags: list[dict[str, Any]]


AnnotationInference = Callable[[str, dict[str, Any], int, float], Mapping[str, Any]]


class OllamaAnnotationInference:
    """Minimal JSON-only Ollama client used by the fully automated workflow."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", timeout_seconds: float = 180.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def __call__(self, model: str, prompt: dict[str, Any], seed: int, temperature: float) -> Mapping[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(
                {
                    "model": model,
                    "prompt": json.dumps(prompt, ensure_ascii=False, allow_nan=False),
                    "stream": False,
                    "format": "json",
                    "options": {"seed": seed, "temperature": temperature},
                },
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Ollama blueprint annotation failed for model {model}: {error}") from error
        raw = payload.get("response")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"Ollama returned no JSON response for model {model}")
        value = raw.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, flags=re.I | re.S)
        if fenced:
            value = fenced.group(1)
        result = json.loads(value)
        if not isinstance(result, dict):
            raise ValueError(f"Ollama annotation must be a JSON object for model {model}")
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ontology(path: str | Path) -> BenchmarkOntology:
    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return BenchmarkOntology.model_validate(payload)


def load_benchmark_items(path: str | Path) -> list[BenchmarkItem]:
    rows = [BenchmarkItem.model_validate(row) for row in read_jsonl(Path(path))]
    if not rows:
        raise ValueError("Benchmark item file is empty")
    item_ids = [row.item_id for row in rows]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("Benchmark item IDs must be unique")
    benchmark_ids = {row.benchmark_id for row in rows}
    if len(benchmark_ids) != 1:
        raise ValueError("One blueprint build can contain only one benchmark_id")
    return rows


def load_annotations(path: str | Path) -> list[ItemAnnotation]:
    rows = [ItemAnnotation.model_validate(row) for row in read_jsonl(Path(path))]
    annotation_ids = [row.annotation_id for row in rows]
    if len(annotation_ids) != len(set(annotation_ids)):
        raise ValueError("annotation_id values must be unique")
    return rows


def _ontology_ids(ontology: BenchmarkOntology) -> dict[str, set[str]]:
    return {
        "knowledge_domain": {row.label_id for row in ontology.knowledge_domains},
        "capability": {row.label_id for row in ontology.capabilities},
        "problem_structure": {row.label_id for row in ontology.problem_structures},
        "response_type": {row.label_id for row in ontology.response_types},
    }


def validate_annotations(
    items: list[BenchmarkItem], annotations: list[ItemAnnotation], ontology: BenchmarkOntology
) -> None:
    item_ids = {item.item_id for item in items}
    allowed = _ontology_ids(ontology)
    for row in annotations:
        if row.item_id not in item_ids:
            raise ValueError(f"Annotation references an unknown item_id: {row.item_id}")
        if row.knowledge_domain_id not in allowed["knowledge_domain"]:
            raise ValueError(f"Unknown knowledge_domain_id: {row.knowledge_domain_id}")
        unknown_capabilities = sorted(set(row.capability_ids) - allowed["capability"])
        if unknown_capabilities:
            raise ValueError(f"Unknown capability_ids: {unknown_capabilities}")
        unknown_structures = sorted(set(row.problem_structure_ids) - allowed["problem_structure"])
        if unknown_structures:
            raise ValueError(f"Unknown problem_structure_ids: {unknown_structures}")
        if row.response_type_id not in allowed["response_type"]:
            raise ValueError(f"Unknown response_type_id: {row.response_type_id}")


def _single_label_consensus(values: list[str]) -> tuple[str | None, float]:
    if not values:
        return None, 0.0
    counts = Counter(values)
    ranked = counts.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None, ranked[0][1] / len(values)
    return ranked[0][0], ranked[0][1] / len(values)


def _multi_label_consensus(values: list[list[str]], threshold: float) -> tuple[list[str], float]:
    if not values:
        return [], 0.0
    counts: Counter[str] = Counter()
    for labels in values:
        counts.update(set(labels))
    selected = sorted(label for label, count in counts.items() if count / len(values) >= threshold)
    if not selected:
        return [], max(counts.values(), default=0) / len(values)
    agreement = sum(counts[label] / len(values) for label in selected) / len(selected)
    return selected, agreement


def build_consensus_annotations(
    items: list[BenchmarkItem],
    annotations: list[ItemAnnotation],
    ontology: BenchmarkOntology,
    config: ConsensusConfig | None = None,
) -> list[ConsensusAnnotation]:
    config = config or ConsensusConfig()
    validate_annotations(items, annotations, ontology)
    grouped: dict[str, list[ItemAnnotation]] = defaultdict(list)
    for row in annotations:
        grouped[row.item_id].append(row)

    consensus_rows = []
    for item in items:
        rows = grouped.get(item.item_id, [])
        reasons = []
        domains, domain_agreement = _single_label_consensus([row.knowledge_domain_id for row in rows])
        response, response_agreement = _single_label_consensus([row.response_type_id for row in rows])
        capabilities, capability_agreement = _multi_label_consensus(
            [row.capability_ids for row in rows], config.minimum_agreement
        )
        structures, structure_agreement = _multi_label_consensus(
            [row.problem_structure_ids for row in rows], config.minimum_agreement
        )
        topics, _ = _multi_label_consensus([row.knowledge_topic_ids for row in rows], config.minimum_agreement)
        mean_confidence = sum(row.confidence for row in rows) / len(rows) if rows else 0.0
        proposed = [tag.model_dump(mode="json") for row in rows for tag in row.proposed_tags]

        if len(rows) < config.minimum_annotations:
            reasons.append("insufficient_annotations")
        if domain_agreement < config.minimum_agreement or domains is None:
            reasons.append("knowledge_domain_disagreement")
        if capability_agreement < config.minimum_agreement or not capabilities:
            reasons.append("capability_disagreement")
        if response_agreement < config.minimum_agreement or response is None:
            reasons.append("response_type_disagreement")
        if config.require_structure_consensus and structure_agreement < config.minimum_agreement:
            reasons.append("problem_structure_disagreement")
        if mean_confidence < config.minimum_mean_confidence:
            reasons.append("low_mean_confidence")
        if proposed:
            reasons.append("proposed_open_set_tags")

        consensus_rows.append(
            ConsensusAnnotation(
                item_id=item.item_id,
                status="review_required" if reasons else "accepted",
                knowledge_domain_id=domains,
                knowledge_topic_ids=topics,
                capability_ids=capabilities,
                problem_structure_ids=structures,
                response_type_id=response,
                mean_confidence=mean_confidence,
                agreement={
                    "knowledge_domain": domain_agreement,
                    "capabilities": capability_agreement,
                    "problem_structures": structure_agreement,
                    "response_type": response_agreement,
                },
                annotation_ids=sorted(row.annotation_id for row in rows),
                review_reasons=sorted(set(reasons)),
                proposed_tags=proposed,
            )
        )
    return consensus_rows


def _aggregate_axis(
    accepted: list[tuple[BenchmarkItem, ConsensusAnnotation]],
    labels: list[OntologyLabel],
    field: str,
) -> list[dict[str, Any]]:
    support = Counter()
    item_counts = Counter()
    for item, annotation in accepted:
        value = getattr(annotation, field)
        selected = [value] if isinstance(value, str) else list(value or [])
        if not selected:
            continue
        contribution = item.score_weight / len(selected)
        for label_id in selected:
            support[label_id] += contribution
            item_counts[label_id] += 1
    total = sum(support.values())
    return [
        {
            "label_id": label.label_id,
            "description": label.description,
            "weight": support[label.label_id] / total if total else 0.0,
            "support_weight": support[label.label_id],
            "item_count": item_counts[label.label_id],
        }
        for label in labels
        if item_counts[label.label_id] > 0
    ]


def build_blueprint_payload(
    items: list[BenchmarkItem],
    consensus: list[ConsensusAnnotation],
    ontology: BenchmarkOntology,
    config: ConsensusConfig,
) -> dict[str, Any]:
    consensus_by_id = {row.item_id: row for row in consensus}
    accepted = [
        (item, consensus_by_id[item.item_id]) for item in items if consensus_by_id[item.item_id].status == "accepted"
    ]
    capability_rows = _aggregate_axis(accepted, ontology.capabilities, "capability_ids")
    knowledge_rows = _aggregate_axis(accepted, ontology.knowledge_domains, "knowledge_domain_id")
    structure_rows = _aggregate_axis(accepted, ontology.problem_structures, "problem_structure_ids")
    response_rows = _aggregate_axis(accepted, ontology.response_types, "response_type_id")
    topic_counts = Counter(topic for _, row in accepted for topic in row.knowledge_topic_ids)

    return {
        "schema_version": "benchmark-knowledge-capability-blueprint-v2",
        "benchmark_id": items[0].benchmark_id,
        "test_items_included": False,
        "source": "Aggregated from task specifications or public train/dev/validation items only.",
        "ontology_id": ontology.ontology_id,
        "construction": {
            "method": "ontology_constrained_multi_annotator_consensus",
            "minimum_annotations": config.minimum_annotations,
            "minimum_agreement": config.minimum_agreement,
            "minimum_mean_confidence": config.minimum_mean_confidence,
            "accepted_item_count": len(accepted),
            "review_required_item_count": len(items) - len(accepted),
            "weight_policy": "normalize benchmark score_weight within each axis; split multi-label item weight equally",
        },
        # The legacy field keeps the result consumable by the existing proxy pipeline.
        "capabilities": [
            {
                "capability_id": row["label_id"],
                "description": row["description"],
                "weight": row["weight"],
                "item_count": row["item_count"],
            }
            for row in capability_rows
        ],
        "dimensions": {
            "knowledge_domains": knowledge_rows,
            "knowledge_topics": [
                {"topic_id": topic_id, "item_count": count}
                for topic_id, count in sorted(topic_counts.items(), key=lambda row: (-row[1], row[0]))
            ],
            "capabilities": capability_rows,
            "problem_structures": structure_rows,
            "response_types": response_rows,
        },
    }


def render_annotation_prompt(item: BenchmarkItem, ontology: BenchmarkOntology) -> dict[str, Any]:
    """Create a backend-neutral structured prompt for one public item."""

    label_block = {
        "knowledge_domains": [row.model_dump(mode="json") for row in ontology.knowledge_domains],
        "capabilities": [row.model_dump(mode="json") for row in ontology.capabilities],
        "problem_structures": [row.model_dump(mode="json") for row in ontology.problem_structures],
        "response_types": [row.model_dump(mode="json") for row in ontology.response_types],
    }
    output_schema = {
        "knowledge_domain_id": "one allowed knowledge-domain label_id",
        "knowledge_topic_ids": ["normalized domain-specific leaf topics"],
        "capability_ids": ["one or more allowed capability label_ids"],
        "problem_structure_ids": ["zero or more allowed problem-structure label_ids"],
        "response_type_id": "one allowed response-type label_id",
        "confidence": "number from 0 to 1",
        "rationale": "brief explanation of what a correct response must know and do",
        "proposed_tags": [
            {
                "axis": "axis name",
                "label_id": "candidate label",
                "description": "candidate definition",
                "justification": "why the fixed ontology is insufficient",
            }
        ],
    }
    content = {
        "question": item.question,
        "options": item.options,
        "answer": item.answer,
        "rationale": item.rationale,
        "response_format_hint": "Return JSON only.",
    }
    return {
        "schema_version": "benchmark-blueprint-annotation-prompt-v1",
        "benchmark_id": item.benchmark_id,
        "item_id": item.item_id,
        "split_role": item.split_role,
        "system": (
            "Classify the public benchmark item by the knowledge it requires, the operation a correct solver must "
            "perform, its problem structure, and its response form. Use only the supplied upper-ontology labels. "
            "Domain-specific leaf topics may be proposed, but every proposed tag requires justification. Do not "
            "infer capability from superficial keywords alone and abstain with low confidence when ambiguous."
        ),
        "allowed_ontology": label_block,
        "item": content,
        "output_schema": output_schema,
    }


def render_adjudication_prompt(
    item: BenchmarkItem,
    ontology: BenchmarkOntology,
    candidate_annotations: list[ItemAnnotation],
) -> dict[str, Any]:
    """Render a closed-set second-stage prompt for an automatically disputed item."""

    prompt = render_annotation_prompt(item, ontology)
    prompt["schema_version"] = "benchmark-blueprint-adjudication-prompt-v1"
    prompt["system"] = (
        "Act as an independent benchmark-taxonomy adjudicator. Re-evaluate the public item and the candidate "
        "annotations. Return the most defensible labels from the supplied upper ontology. Do not copy a candidate "
        "by majority alone, do not create a new upper-ontology label, and return proposed_tags as an empty list. "
        "Resolve superficial keyword matches by stating what a correct solver must actually know and do."
    )
    prompt["candidate_annotations"] = [
        {
            "annotation_id": row.annotation_id,
            "knowledge_domain_id": row.knowledge_domain_id,
            "knowledge_topic_ids": row.knowledge_topic_ids,
            "capability_ids": row.capability_ids,
            "problem_structure_ids": row.problem_structure_ids,
            "response_type_id": row.response_type_id,
            "confidence": row.confidence,
            "rationale": row.rationale,
            "proposed_tags": [tag.model_dump(mode="json") for tag in row.proposed_tags],
        }
        for row in candidate_annotations
    ]
    return prompt


def _stable_seed(base_seed: int, item_id: str, model: str, run_index: int, stage: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{item_id}|{model}|{run_index}|{stage}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def _annotation_from_inference(
    *,
    payload: Mapping[str, Any],
    item_id: str,
    model: str,
    seed: int,
    stage: str,
    source_annotation_ids: list[str] | None = None,
) -> ItemAnnotation:
    fields = {
        key: payload.get(key)
        for key in (
            "knowledge_domain_id",
            "knowledge_topic_ids",
            "capability_ids",
            "problem_structure_ids",
            "response_type_id",
            "confidence",
            "rationale",
            "proposed_tags",
        )
    }
    fields["knowledge_topic_ids"] = fields["knowledge_topic_ids"] or []
    fields["problem_structure_ids"] = fields["problem_structure_ids"] or []
    fields["proposed_tags"] = fields["proposed_tags"] or []
    return ItemAnnotation.model_validate(
        {
            "annotation_id": f"{stage}:{model}:{seed}:{item_id}",
            "item_id": item_id,
            "annotator_id": f"{model}:seed-{seed}",
            **fields,
            "stage": stage,
            "source_annotation_ids": source_annotation_ids or [],
        }
    )


def run_annotation_panel(
    *,
    items: list[BenchmarkItem],
    ontology: BenchmarkOntology,
    models: list[str],
    runs_per_model: int,
    inference: AnnotationInference,
    base_seed: int,
    temperature: float,
    stage: str = "primary",
    candidate_annotations: dict[str, list[ItemAnnotation]] | None = None,
) -> list[ItemAnnotation]:
    """Run a reproducible primary or adjudication panel over public items."""

    if not models or any(not model.strip() for model in models):
        raise ValueError("At least one non-empty annotation model is required")
    if runs_per_model <= 0:
        raise ValueError("runs_per_model must be positive")
    rows = []
    for item in items:
        candidates = (candidate_annotations or {}).get(item.item_id, [])
        prompt = (
            render_adjudication_prompt(item, ontology, candidates)
            if stage == "adjudication"
            else render_annotation_prompt(item, ontology)
        )
        source_ids = sorted(row.annotation_id for row in candidates)
        for model in models:
            for run_index in range(runs_per_model):
                seed = _stable_seed(base_seed, item.item_id, model, run_index, stage)
                payload = inference(model, prompt, seed, temperature)
                annotation = _annotation_from_inference(
                    payload=payload,
                    item_id=item.item_id,
                    model=model,
                    seed=seed,
                    stage=stage,
                    source_annotation_ids=source_ids,
                )
                if stage == "adjudication" and annotation.proposed_tags:
                    raise ValueError(
                        f"Automatic adjudicator {model} proposed an open-set tag for {item.item_id}; "
                        "closed-set adjudication is required"
                    )
                rows.append(annotation)
    validate_annotations(items, rows, ontology)
    return rows


def _replace_disputed_consensus(
    primary: list[ConsensusAnnotation], adjudicated: list[ConsensusAnnotation]
) -> list[ConsensusAnnotation]:
    adjudicated_by_id = {row.item_id: row for row in adjudicated}
    final = []
    unresolved = []
    for row in primary:
        if row.status == "accepted":
            final.append(row)
            continue
        replacement = adjudicated_by_id.get(row.item_id)
        if replacement is None or replacement.status != "accepted":
            unresolved.append(row.item_id)
            continue
        final.append(replacement)
    if unresolved:
        raise ValueError(
            "Automatic adjudication could not reach the frozen consensus threshold for items: "
            + ", ".join(sorted(unresolved))
        )
    return final


def run_automated_blueprint_pipeline(
    *,
    items_path: str | Path,
    ontology_path: str | Path,
    output_dir: str | Path,
    annotator_models: list[str],
    adjudicator_models: list[str] | None = None,
    runs_per_model: int = 3,
    adjudication_runs_per_model: int = 3,
    config: ConsensusConfig | None = None,
    base_seed: int = 17,
    temperature: float = 0.0,
    adjudication_temperature: float = 0.0,
    inference: AnnotationInference | None = None,
    ollama_url: str = "http://127.0.0.1:11434",
) -> dict[str, Any]:
    """Run primary LLM annotation, automatic dispute adjudication, and freezing."""

    config = config or ConsensusConfig(minimum_annotations=3)
    if config.minimum_annotations > len(annotator_models) * runs_per_model:
        raise ValueError("Primary panel is smaller than minimum_annotations")
    adjudicator_models = adjudicator_models or list(annotator_models)
    if config.minimum_annotations > len(adjudicator_models) * adjudication_runs_per_model:
        raise ValueError("Adjudication panel is smaller than minimum_annotations")
    inference = inference or OllamaAnnotationInference(base_url=ollama_url)
    items_path, ontology_path, output_dir = Path(items_path), Path(ontology_path), Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Blueprint output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    items = load_benchmark_items(items_path)
    ontology = load_ontology(ontology_path)

    primary_annotations = run_annotation_panel(
        items=items,
        ontology=ontology,
        models=annotator_models,
        runs_per_model=runs_per_model,
        inference=inference,
        base_seed=base_seed,
        temperature=temperature,
    )
    primary_consensus = build_consensus_annotations(items, primary_annotations, ontology, config)
    disputed_ids = {row.item_id for row in primary_consensus if row.status == "review_required"}
    disputed_items = [item for item in items if item.item_id in disputed_ids]
    primary_by_item: dict[str, list[ItemAnnotation]] = defaultdict(list)
    for row in primary_annotations:
        primary_by_item[row.item_id].append(row)

    adjudication_annotations: list[ItemAnnotation] = []
    adjudication_consensus: list[ConsensusAnnotation] = []
    if disputed_items:
        adjudication_annotations = run_annotation_panel(
            items=disputed_items,
            ontology=ontology,
            models=adjudicator_models,
            runs_per_model=adjudication_runs_per_model,
            inference=inference,
            base_seed=base_seed,
            temperature=adjudication_temperature,
            stage="adjudication",
            candidate_annotations=primary_by_item,
        )
        adjudication_consensus = build_consensus_annotations(disputed_items, adjudication_annotations, ontology, config)
    final_consensus = _replace_disputed_consensus(primary_consensus, adjudication_consensus)
    blueprint = build_blueprint_payload(items, final_consensus, ontology, config)
    blueprint["construction"].update(
        {
            "method": "ontology_constrained_llm_panel_with_automatic_adjudication",
            "annotator_models": annotator_models,
            "adjudicator_models": adjudicator_models,
            "runs_per_model": runs_per_model,
            "adjudication_runs_per_model": adjudication_runs_per_model,
            "automatically_adjudicated_item_count": len(disputed_items),
            "unresolved_item_count": 0,
        }
    )

    paths = {
        "blueprint": output_dir / "capability_blueprint.yaml",
        "primary_annotations": output_dir / "primary_annotations.jsonl",
        "primary_consensus": output_dir / "primary_consensus.jsonl",
        "adjudication_annotations": output_dir / "adjudication_annotations.jsonl",
        "automatic_adjudication_log": output_dir / "automatic_adjudication_log.jsonl",
        "final_consensus": output_dir / "item_consensus.jsonl",
    }
    atomic_text(paths["blueprint"], yaml.safe_dump(blueprint, allow_unicode=True, sort_keys=False, width=120))
    write_jsonl(paths["primary_annotations"], (row.model_dump(mode="json") for row in primary_annotations))
    write_jsonl(paths["primary_consensus"], (row.model_dump(mode="json") for row in primary_consensus))
    write_jsonl(
        paths["adjudication_annotations"],
        (row.model_dump(mode="json") for row in adjudication_annotations),
    )
    write_jsonl(
        paths["automatic_adjudication_log"],
        (row.model_dump(mode="json") for row in adjudication_consensus),
    )
    write_jsonl(paths["final_consensus"], (row.model_dump(mode="json") for row in final_consensus))
    manifest = {
        "schema_version": "automated-benchmark-blueprint-build-manifest-v1",
        "benchmark_id": items[0].benchmark_id,
        "hidden_test_items_accessed": False,
        "human_review_used": False,
        "inputs": {"items_sha256": _sha256(items_path), "ontology_sha256": _sha256(ontology_path)},
        "runtime": {
            "annotator_models": annotator_models,
            "adjudicator_models": adjudicator_models,
            "base_seed": base_seed,
            "temperature": temperature,
            "adjudication_temperature": adjudication_temperature,
            "ollama_url": ollama_url,
        },
        "counts": {
            "items": len(items),
            "primary_annotations": len(primary_annotations),
            "primary_accepted": len(items) - len(disputed_items),
            "automatically_adjudicated": len(disputed_items),
            "adjudication_annotations": len(adjudication_annotations),
            "final_accepted": len(final_consensus),
            "unresolved": 0,
        },
        "consensus_config": config.model_dump(mode="json"),
        "outputs": {name: {"uri": path.name, "sha256": _sha256(path)} for name, path in paths.items()},
    }
    manifest_path = output_dir / "build_manifest.json"
    write_json(manifest_path, manifest)
    return {**manifest, "output_dir": str(output_dir.resolve()), "manifest": str(manifest_path.resolve())}


def prepare_annotation_prompts(
    items_path: str | Path, ontology_path: str | Path, output_path: str | Path
) -> dict[str, Any]:
    items_path, ontology_path, output_path = Path(items_path), Path(ontology_path), Path(output_path)
    items = load_benchmark_items(items_path)
    ontology = load_ontology(ontology_path)
    write_jsonl(output_path, (render_annotation_prompt(item, ontology) for item in items))
    return {
        "benchmark_id": items[0].benchmark_id,
        "item_count": len(items),
        "output": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "hidden_test_items_accessed": False,
    }


def build_benchmark_blueprint(
    *,
    items_path: str | Path,
    ontology_path: str | Path,
    annotations_path: str | Path,
    output_dir: str | Path,
    config: ConsensusConfig | None = None,
) -> dict[str, Any]:
    config = config or ConsensusConfig()
    items_path, ontology_path, annotations_path = map(Path, (items_path, ontology_path, annotations_path))
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Blueprint output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    items = load_benchmark_items(items_path)
    ontology = load_ontology(ontology_path)
    annotations = load_annotations(annotations_path)
    consensus = build_consensus_annotations(items, annotations, ontology, config)
    blueprint = build_blueprint_payload(items, consensus, ontology, config)

    blueprint_path = output_dir / "capability_blueprint.yaml"
    consensus_path = output_dir / "item_consensus.jsonl"
    review_path = output_dir / "review_queue.jsonl"
    atomic_text(blueprint_path, yaml.safe_dump(blueprint, allow_unicode=True, sort_keys=False, width=120))
    write_jsonl(consensus_path, (row.model_dump(mode="json") for row in consensus))
    write_jsonl(
        review_path,
        (row.model_dump(mode="json") for row in consensus if row.status == "review_required"),
    )
    manifest = {
        "schema_version": "benchmark-blueprint-build-manifest-v1",
        "benchmark_id": items[0].benchmark_id,
        "hidden_test_items_accessed": False,
        "inputs": {
            "items_sha256": _sha256(items_path),
            "ontology_sha256": _sha256(ontology_path),
            "annotations_sha256": _sha256(annotations_path),
        },
        "outputs": {
            "blueprint": blueprint_path.name,
            "blueprint_sha256": _sha256(blueprint_path),
            "consensus": consensus_path.name,
            "consensus_sha256": _sha256(consensus_path),
            "review_queue": review_path.name,
            "review_queue_sha256": _sha256(review_path),
        },
        "counts": {
            "items": len(items),
            "annotations": len(annotations),
            "accepted": sum(row.status == "accepted" for row in consensus),
            "review_required": sum(row.status == "review_required" for row in consensus),
        },
        "consensus_config": config.model_dump(mode="json"),
    }
    manifest_path = output_dir / "build_manifest.json"
    write_json(manifest_path, manifest)
    return {**manifest, "output_dir": str(output_dir.resolve()), "manifest": str(manifest_path.resolve())}


__all__ = [
    "BenchmarkItem",
    "BenchmarkOntology",
    "ConsensusAnnotation",
    "ConsensusConfig",
    "ItemAnnotation",
    "OllamaAnnotationInference",
    "OntologyLabel",
    "ProposedTag",
    "build_benchmark_blueprint",
    "build_blueprint_payload",
    "build_consensus_annotations",
    "load_annotations",
    "load_benchmark_items",
    "load_ontology",
    "prepare_annotation_prompts",
    "render_annotation_prompt",
    "render_adjudication_prompt",
    "run_annotation_panel",
    "run_automated_blueprint_pipeline",
    "validate_annotations",
]
