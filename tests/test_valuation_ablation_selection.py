import json
import sys
from pathlib import Path

import pytest
import yaml

from ai_ready.utils.io import write_jsonl
from ai_ready.valuation.ablation_selection import (
    FractionalBudgetPolicy,
    candidate_pool_digest,
    select_fractional_proxy,
)
from ai_ready.valuation.cli import main as valuation_main
from ai_ready.valuation.selection import (
    CoverageObligation,
    ProxySelectionConfig,
    ProxyUnit,
)


def _unit(identifier: str, *, tokens: int = 10) -> ProxyUnit:
    index = int(identifier[-1])
    return ProxyUnit(
        unit_id=identifier,
        batch_id="B01",
        batch_content_sha256="a" * 64,
        group_id=f"group-{identifier}",
        source_ids=[f"source-{index % 2}"],
        estimated_tokens=tokens,
        embedding=[float(index == 1), float(index != 1), index / 10],
        quality_bucket="clean" if index % 2 else "usable_noisy",
        benchmark_relevance=index / 4,
    )


def _units() -> list[ProxyUnit]:
    return [_unit(f"unit-{index}") for index in range(1, 5)]


def _obligations() -> list[CoverageObligation]:
    return [
        CoverageObligation(
            obligation_id=f"obligation-{index}",
            kind="rule",
            unit_ids=[f"unit-{index}"],
        )
        for index in range(1, 5)
    ]


def test_fractional_budget_produces_an_audited_strict_subset():
    result = select_fractional_proxy(
        _units(),
        _obligations(),
        ProxySelectionConfig(token_budget=999),
        FractionalBudgetPolicy(budget_fraction=0.5, require_strict_subset=True),
    )

    audit = result["fractional_budget"]
    assert audit["eligible_estimated_tokens"] == 40
    assert audit["derived_token_budget"] == 20
    assert audit["selected_group_count"] == 2
    assert audit["selected_group_fraction"] == pytest.approx(0.5)
    assert audit["strict_subset_observed"] is True
    assert result["selection_config"]["token_budget"] == 20
    assert len(result["ablation_selection_sha256"]) == 64


def test_full_candidate_comparator_requires_an_explicit_non_strict_policy():
    with pytest.raises(ValueError, match="requires budget_fraction < 1"):
        FractionalBudgetPolicy(budget_fraction=1.0, require_strict_subset=True)

    result = select_fractional_proxy(
        _units(),
        _obligations(),
        ProxySelectionConfig(token_budget=1),
        FractionalBudgetPolicy(budget_fraction=1.0, require_strict_subset=False),
    )
    assert result["fractional_budget"]["strict_subset_observed"] is False
    assert result["fractional_budget"]["selected_group_count"] == 4


def test_fractional_budget_rejects_a_batch_that_cannot_fit_one_dependency_group():
    with pytest.raises(ValueError, match="cannot fit any complete dependency group"):
        select_fractional_proxy(
            _units(),
            _obligations(),
            ProxySelectionConfig(token_budget=1),
            FractionalBudgetPolicy(budget_fraction=0.1, require_strict_subset=True),
        )


def test_candidate_pool_digest_is_order_invariant_and_content_sensitive():
    units, obligations = _units(), _obligations()
    expected = candidate_pool_digest(units, obligations)
    assert candidate_pool_digest(list(reversed(units)), list(reversed(obligations))) == expected

    changed = units.copy()
    changed[0] = changed[0].model_copy(update={"estimated_tokens": 11})
    assert candidate_pool_digest(changed, obligations) != expected


def test_maximum_selected_group_fraction_is_enforced():
    with pytest.raises(ValueError, match="exceeds the preregistered maximum"):
        select_fractional_proxy(
            _units(),
            _obligations(),
            ProxySelectionConfig(token_budget=1),
            FractionalBudgetPolicy(
                budget_fraction=0.5,
                require_strict_subset=True,
                maximum_selected_group_fraction=0.4,
            ),
        )


def test_fractional_selection_cli_writes_the_frozen_audit(tmp_path: Path, monkeypatch, capsys):
    units_path = tmp_path / "units.jsonl"
    obligations_path = tmp_path / "obligations.jsonl"
    config_path = tmp_path / "selector.yaml"
    output_path = tmp_path / "selection.json"
    write_jsonl(units_path, (row.model_dump(mode="json") for row in _units()))
    write_jsonl(obligations_path, (row.model_dump(mode="json") for row in _obligations()))
    config_path.write_text(
        yaml.safe_dump(ProxySelectionConfig(token_budget=999).model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai-ready-value",
            "select-ablation-fraction",
            "--units",
            str(units_path),
            "--obligations",
            str(obligations_path),
            "--config",
            str(config_path),
            "--budget-fraction",
            "0.5",
            "--output",
            str(output_path),
        ],
    )

    assert valuation_main() == 0
    stdout = json.loads(capsys.readouterr().out)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert stdout == saved
    assert saved["fractional_budget"]["strict_subset_observed"] is True
