import pytest

from ai_ready.core.schema import EvidenceUnit, Provenance
from ai_ready.valuation.generic_knowledge import (
    ExtractionConfig,
    extract_generic_knowledge_units,
    to_semantic_units,
)


def _evidence():
    text = (
        "A supplier must submit the notice within 30 days after delivery. "
        "The deadline does not apply when the buyer waives written notice. "
        "Late submissions may incur a processing fee."
    )
    return EvidenceUnit(
        id="legal-1",
        document_id="contract-1",
        document_version_id="v1",
        text=text,
        raw_text=text,
        headings=["Notice requirements"],
        sources=[
            Provenance(
                document_id="contract-1",
                document_version_id="v1",
                source_uri="file:///contract.txt",
                source_sha256="a" * 64,
                parser="test",
                raw_text=text,
            )
        ],
    )


def _candidate_payload():
    return {
        "units": [
            {
                "unit_type": "rule",
                "claim": "A supplier must submit notice within 30 days after delivery unless waived by the buyer.",
                "conditions": ["after delivery"],
                "exceptions": ["the buyer waives written notice"],
                "temporal_scope": "within 30 days after delivery",
                "uncertainty": None,
                "entities": ["supplier", "buyer", "notice"],
                "relations": [{"subject": "supplier", "predicate": "must submit", "object": "notice"}],
                "evidence_quotes": [
                    "A supplier must submit the notice within 30 days after delivery.",
                    "The deadline does not apply when the buyer waives written notice.",
                ],
                "confidence": 0.9,
            },
            {
                "unit_type": "fact",
                "claim": "Late submissions may incur a processing fee.",
                "conditions": [],
                "exceptions": [],
                "temporal_scope": None,
                "uncertainty": "may",
                "entities": ["late submissions", "processing fee"],
                "relations": [],
                "evidence_quotes": ["Late submissions may incur a processing fee."],
                "confidence": 0.8,
            },
        ]
    }


def test_generic_extraction_is_evidence_bound_and_fully_automated():
    def inference(model, prompt, seed, temperature):
        del model, seed, temperature
        schema = prompt["schema_version"]
        if schema in {
            "generic-knowledge-extraction-prompt-v1",
            "generic-knowledge-consolidation-prompt-v1",
        }:
            return _candidate_payload()
        claim = prompt["candidate_unit"]["claim"]
        if "processing fee" in claim:
            return {
                "supported": False,
                "fields_consistent": False,
                "confidence": 0.9,
                "rationale": "The modal statement is not accepted by this verifier panel.",
            }
        if schema == "generic-knowledge-verification-prompt-v1":
            return {
                "supported": False,
                "fields_consistent": True,
                "confidence": 0.8,
                "rationale": "Primary verifiers disagree on the combined rule.",
            }
        return {
            "supported": True,
            "fields_consistent": True,
            "confidence": 0.95,
            "rationale": "Both the deadline and its exception are quoted exactly.",
        }

    units, dispositions = extract_generic_knowledge_units(
        [_evidence()],
        extraction_models=["extractor-a", "extractor-b"],
        consolidation_model="consolidator",
        verification_models=["verifier-a", "verifier-b"],
        adjudicator_models=["judge"],
        config=ExtractionConfig(
            extraction_runs_per_model=1,
            verification_runs_per_model=1,
            adjudication_runs_per_model=3,
            minimum_verifier_support=2 / 3,
        ),
        inference=inference,
    )

    assert len(units) == 2
    accepted = [unit for unit in units if unit.verification_status == "accepted"]
    rejected = [unit for unit in units if unit.verification_status == "rejected"]
    assert len(accepted) == 1
    assert len(rejected) == 1
    assert accepted[0].adjudicated is True
    assert accepted[0].conditions == ["after delivery"]
    assert accepted[0].exceptions == ["the buyer waives written notice"]
    assert accepted[0].evidence_offsets[0] == (0, 64)
    assert dispositions[0]["status"] == "accepted"
    semantic = to_semantic_units(units, batch_id="legal-batch")
    assert len(semantic) == 1
    assert semantic[0].activity_ids == ["unmapped"]
    assert semantic[0].group_id


def test_non_verbatim_evidence_is_rejected_before_verification():
    def inference(model, prompt, seed, temperature):
        del model, seed, temperature
        if prompt["schema_version"] == "generic-knowledge-extraction-prompt-v1":
            payload = _candidate_payload()
            payload["units"][0]["evidence_quotes"] = ["This quote does not exist in the source."]
            return payload
        return _candidate_payload()

    with pytest.raises(ValueError, match="not an exact source substring"):
        extract_generic_knowledge_units(
            [_evidence()],
            extraction_models=["extractor"],
            consolidation_model="consolidator",
            verification_models=["verifier"],
            config=ExtractionConfig(extraction_runs_per_model=1),
            inference=inference,
        )
