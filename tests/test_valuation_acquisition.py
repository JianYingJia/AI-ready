import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ai_ready.valuation.acquisition import curate_medmcqa_cll_proxy


def _row(item_id: str, question: str, answer_index: int, options: list[str]) -> dict:
    return {
        "id": item_id,
        "question": question,
        "opa": options[0],
        "opb": options[1],
        "opc": options[2],
        "opd": options[3],
        "cop": answer_index,
        "choice_type": "single",
        "exp": "A source explanation.",
        "subject_name": "Medicine",
        "topic_name": None,
    }


def test_medmcqa_proxy_curation_locks_targeted_clean_deduplicated_items(tmp_path: Path):
    rows = [
        _row(
            "keep",
            "Which leukemia is commonly associated with autoimmune hemolytic anemia?",
            1,
            ["AML", "CLL", "CML", "ALL"],
        ),
        _row(
            "near-duplicate",
            "Which leukaemia is commonly associated with autoimmune haemolytic anaemia?",
            1,
            ["AML", "CLL", "CML", "ALL"],
        ),
        _row(
            "false-unit-match",
            "A patient has hemoglobin measured as 8 g/cll during a nutrition study.",
            0,
            ["Finding one", "Finding two", "Finding three", "Finding four"],
        ),
        _row(
            "bad-answer-token",
            "Which statement about CLL is correct?",
            0,
            ["A", "A complete medical option", "Another medical option", "A fourth option"],
        ),
        _row(
            "unrelated",
            "Which organism most often causes uncomplicated cystitis?",
            0,
            ["E. coli", "S. aureus", "Influenza virus", "Candida albicans"],
        ),
    ]
    rows[1]["exp"] = None
    upstream = tmp_path / "medmcqa.parquet"
    pq.write_table(pa.Table.from_pylist(rows), upstream)
    output = tmp_path / "locked.jsonl"

    stats = curate_medmcqa_cll_proxy(upstream, output)
    retained = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

    assert stats["item_count"] == 1
    assert retained[0]["source_item_id"] == "keep"
    assert retained[0]["answer_text"] == "CLL"
    assert stats["exclusion_counts"]["near_duplicate"] == 1
    assert stats["exclusion_counts"]["malformed_combination_answer"] == 1
    assert stats["exclusion_counts"]["not_cll_targeted"] == 2
