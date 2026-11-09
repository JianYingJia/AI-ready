"""The fixed task axis; capability labels remain a separate, extensible axis."""

TASK_CATALOG = {
    "factual_qa": {
        "name_zh": "事实问答",
        "knowledge_shape": ["explicit_fact", "rule", "numeric_condition"],
        "minimum_evidence_units": 1,
        "output_schema": {"question": "str", "answer": "str", "citations": "list[str]"},
        "difficulty_knobs": ["distractor", "output_facts", "implicitness"],
        "validator": "answer_entailment_and_citation",
        "invalid_when": "evidence is empty or only decorative",
    },
    "multi_turn": {
        "name_zh": "多轮对话",
        "knowledge_shape": ["conditional_rule", "procedure", "multiple_facts"],
        "minimum_evidence_units": 1,
        "output_schema": {"conversation": "list[turn]", "citations": "list[str]"},
        "difficulty_knobs": ["turns", "hidden_prerequisite", "state_carryover"],
        "validator": "turn_state_and_final_answer",
        "invalid_when": "no natural follow-up or state transition exists",
    },
    "summarization": {
        "name_zh": "摘要",
        "knowledge_shape": ["section", "table", "multi_fact_passage"],
        "minimum_evidence_units": 1,
        "output_schema": {"summary": "str", "required_facts": "list[str]", "citations": "list[str]"},
        "difficulty_knobs": ["evidence_span", "compression_ratio", "output_structure"],
        "validator": "fact_coverage_and_faithfulness",
        "invalid_when": "content is too short to require compression",
    },
    "judgment": {
        "name_zh": "判断",
        "knowledge_shape": ["condition", "exception", "compliance_boundary"],
        "minimum_evidence_units": 1,
        "output_schema": {"decision": "bool|enum", "explanation": "str", "citations": "list[str]"},
        "difficulty_knobs": ["conditions", "exceptions", "implicitness"],
        "validator": "decision_explanation_and_citation",
        "invalid_when": "there is no predicate, condition, or boundary to judge",
    },
    "comparison": {
        "name_zh": "对比",
        "knowledge_shape": ["two_entities", "two_versions", "two_documents"],
        "minimum_evidence_units": 1,
        "output_schema": {"dimensions": "list[str]", "differences": "list[object]", "citations": "list[str]"},
        "difficulty_knobs": ["comparison_dimensions", "document_count", "conflicting_rules"],
        "validator": "paired_fact_and_dimension_coverage",
        "invalid_when": "fewer than two comparable objects exist",
    },
    "extraction": {
        "name_zh": "抽取",
        "knowledge_shape": ["table", "entity", "numeric_field", "form"],
        "minimum_evidence_units": 1,
        "output_schema": {"records": "list[object]", "citations": "list[str]"},
        "difficulty_knobs": ["field_count", "normalization", "table_span"],
        "validator": "schema_and_exact_field_match",
        "invalid_when": "there are no extractable entities, fields, numbers, or tables",
    },
}

TASK_TYPES = tuple(TASK_CATALOG)


def catalog_rows() -> list[dict]:
    return [{"task_type": task_type, **definition} for task_type, definition in TASK_CATALOG.items()]
