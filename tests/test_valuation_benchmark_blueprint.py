import json

import pytest
import yaml

from ai_ready.valuation.benchmark_blueprint import (
    BenchmarkItem,
    BenchmarkOntology,
    ConsensusConfig,
    ItemAnnotation,
    build_benchmark_blueprint,
    build_consensus_annotations,
    render_annotation_prompt,
    run_automated_blueprint_pipeline,
)


def _ontology():
    return BenchmarkOntology.model_validate(
        {
            "ontology_id": "test-ontology",
            "knowledge_domains": [{"label_id": "medicine", "description": "Medicine"}],
            "capabilities": [
                {"label_id": "recall", "description": "Recall"},
                {"label_id": "conditional", "description": "Conditional reasoning"},
            ],
            "problem_structures": [
                {"label_id": "single_hop", "description": "Single hop"},
                {"label_id": "condition", "description": "Condition"},
            ],
            "response_types": [{"label_id": "multiple_choice", "description": "MCQ"}],
        }
    )


def _item(item_id="q1", score_weight=1.0):
    return BenchmarkItem(
        benchmark_id="public-dev-benchmark",
        item_id=item_id,
        split_role="public_dev",
        question="When is observation appropriate?",
        options=["A", "B"],
        answer="A",
        score_weight=score_weight,
    )


def _annotation(annotation_id, item_id="q1", capabilities=None, confidence=0.9, proposed_tags=None):
    return ItemAnnotation(
        annotation_id=annotation_id,
        item_id=item_id,
        annotator_id=annotation_id,
        knowledge_domain_id="medicine",
        knowledge_topic_ids=["treatment_indication"],
        capability_ids=capabilities or ["conditional"],
        problem_structure_ids=["condition"],
        response_type_id="multiple_choice",
        confidence=confidence,
        rationale="The solver must apply a treatment condition.",
        proposed_tags=proposed_tags or [],
    )


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def test_hidden_test_items_are_rejected():
    with pytest.raises(ValueError, match="public train/dev/validation"):
        BenchmarkItem(
            benchmark_id="b",
            item_id="q",
            split_role="hidden_test",
            question="Secret question",
        )


def test_consensus_accepts_stable_multi_axis_annotations():
    rows = build_consensus_annotations(
        [_item()],
        [_annotation("a1"), _annotation("a2"), _annotation("a3", capabilities=["conditional", "recall"])],
        _ontology(),
        ConsensusConfig(minimum_annotations=2, minimum_agreement=2 / 3),
    )
    assert rows[0].status == "accepted"
    assert rows[0].capability_ids == ["conditional"]
    assert rows[0].knowledge_topic_ids == ["treatment_indication"]
    assert rows[0].agreement["capabilities"] == pytest.approx(1.0)


def test_open_set_proposal_is_sent_to_review():
    proposal = {
        "axis": "capability",
        "label_id": "clinical_triage",
        "description": "Triage a patient",
        "justification": "The fixed ontology has no triage label.",
    }
    rows = build_consensus_annotations(
        [_item()],
        [_annotation("a1", proposed_tags=[proposal]), _annotation("a2")],
        _ontology(),
        ConsensusConfig(minimum_annotations=2),
    )
    assert rows[0].status == "review_required"
    assert "proposed_open_set_tags" in rows[0].review_reasons


def test_prompt_is_ontology_constrained_and_public():
    prompt = render_annotation_prompt(_item(), _ontology())
    assert prompt["split_role"] == "public_dev"
    assert prompt["item"]["question"] == "When is observation appropriate?"
    assert "allowed_ontology" in prompt
    assert "Return JSON only" in prompt["item"]["response_format_hint"]


def test_build_writes_question_free_blueprint_and_review_queue(tmp_path):
    items_path = tmp_path / "items.jsonl"
    annotations_path = tmp_path / "annotations.jsonl"
    ontology_path = tmp_path / "ontology.yaml"
    output_dir = tmp_path / "out"
    items = [_item("q1", 2.0), _item("q2", 1.0)]
    annotations = [
        _annotation("q1-a1", "q1"),
        _annotation("q1-a2", "q1"),
        _annotation("q2-a1", "q2", confidence=0.2),
    ]
    _write_jsonl(items_path, [row.model_dump(mode="json") for row in items])
    _write_jsonl(annotations_path, [row.model_dump(mode="json") for row in annotations])
    ontology_path.write_text(yaml.safe_dump(_ontology().model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    manifest = build_benchmark_blueprint(
        items_path=items_path,
        ontology_path=ontology_path,
        annotations_path=annotations_path,
        output_dir=output_dir,
        config=ConsensusConfig(minimum_annotations=2, minimum_mean_confidence=0.6),
    )

    blueprint_text = (output_dir / "capability_blueprint.yaml").read_text(encoding="utf-8")
    blueprint = yaml.safe_load(blueprint_text)
    review_rows = [json.loads(line) for line in (output_dir / "review_queue.jsonl").read_text().splitlines()]
    assert manifest["counts"] == {"items": 2, "annotations": 3, "accepted": 1, "review_required": 1}
    assert blueprint["test_items_included"] is False
    assert blueprint["capabilities"][0]["capability_id"] == "conditional"
    assert blueprint["capabilities"][0]["weight"] == pytest.approx(1.0)
    assert "When is observation appropriate?" not in blueprint_text
    assert review_rows[0]["item_id"] == "q2"


def test_automated_pipeline_uses_llm_adjudication_without_human_review(tmp_path):
    items_path = tmp_path / "items.jsonl"
    ontology_path = tmp_path / "ontology.yaml"
    output_dir = tmp_path / "automatic"
    items = [_item("q1"), _item("q2")]
    _write_jsonl(items_path, [row.model_dump(mode="json") for row in items])
    ontology_path.write_text(yaml.safe_dump(_ontology().model_dump(mode="json"), sort_keys=False), encoding="utf-8")

    def inference(model, prompt, seed, temperature):
        del seed, temperature
        item_id = prompt["item_id"]
        adjudicating = prompt["schema_version"] == "benchmark-blueprint-adjudication-prompt-v1"
        capability = "conditional"
        if item_id == "q2" and not adjudicating and model == "model-b":
            capability = "recall"
        return {
            "knowledge_domain_id": "medicine",
            "knowledge_topic_ids": ["treatment_indication"],
            "capability_ids": [capability],
            "problem_structure_ids": ["condition"],
            "response_type_id": "multiple_choice",
            "confidence": 0.9,
            "rationale": "The solver applies the relevant condition.",
            "proposed_tags": [],
        }

    manifest = run_automated_blueprint_pipeline(
        items_path=items_path,
        ontology_path=ontology_path,
        output_dir=output_dir,
        annotator_models=["model-a", "model-b"],
        adjudicator_models=["judge"],
        runs_per_model=1,
        adjudication_runs_per_model=2,
        config=ConsensusConfig(minimum_annotations=2, minimum_agreement=2 / 3),
        inference=inference,
    )

    final_rows = [json.loads(line) for line in (output_dir / "item_consensus.jsonl").read_text().splitlines()]
    assert manifest["human_review_used"] is False
    assert manifest["counts"]["automatically_adjudicated"] == 1
    assert manifest["counts"]["final_accepted"] == 2
    assert manifest["counts"]["unresolved"] == 0
    assert all(row["status"] == "accepted" for row in final_rows)
    assert (output_dir / "automatic_adjudication_log.jsonl").read_text(encoding="utf-8").strip()
