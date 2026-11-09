"""Selection helpers used only by stage-level ablation experiments.

The production selector keeps its absolute token-budget contract.  Mechanism
experiments need comparable compression ratios across differently sized
batches, so this module derives an absolute budget from each frozen candidate
pool and records the derivation.  Keeping this wrapper separate prevents an
ablation-only policy from silently changing the formal production pipeline.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai_ready.valuation.selection import (
    CoverageObligation,
    ProxySelectionConfig,
    ProxyUnit,
    select_proxy,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FractionalBudgetPolicy(_StrictModel):
    """A batch-relative source budget frozen before selection."""

    budget_fraction: float = Field(gt=0, le=1)
    require_strict_subset: bool = True
    minimum_selected_groups: int = Field(default=1, ge=1)
    maximum_selected_group_fraction: float | None = Field(default=None, gt=0, le=1)

    @model_validator(mode="after")
    def validate_strict_policy(self):
        if self.require_strict_subset and self.budget_fraction >= 1:
            raise ValueError("A strict-subset policy requires budget_fraction < 1")
        return self


def _canonical_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _eligible_groups(units: list[ProxyUnit]) -> dict[str, list[ProxyUnit]]:
    groups: dict[str, list[ProxyUnit]] = defaultdict(list)
    for unit in units:
        groups[unit.group_id].append(unit)
    return {
        group_id: group
        for group_id, group in groups.items()
        if all(unit.eligible and unit.medically_valid for unit in group)
    }


def candidate_pool_digest(units: list[ProxyUnit], obligations: list[CoverageObligation]) -> str:
    """Bind a relative budget to the exact units and obligations it measured."""

    return _canonical_digest(
        {
            "units": [
                unit.model_dump(mode="json") for unit in sorted(units, key=lambda row: (row.group_id, row.unit_id))
            ],
            "obligations": [
                obligation.model_dump(mode="json")
                for obligation in sorted(obligations, key=lambda row: row.obligation_id)
            ],
        }
    )


def select_fractional_proxy(
    units: list[ProxyUnit],
    obligations: list[CoverageObligation],
    config: ProxySelectionConfig,
    policy: FractionalBudgetPolicy,
) -> dict:
    """Run an existing selector under an auditable per-batch relative budget."""

    if not units:
        raise ValueError("At least one proxy unit is required")
    eligible_groups = _eligible_groups(units)
    if not eligible_groups:
        raise ValueError("No eligible, medically valid dependency groups are available")
    group_costs = {
        group_id: sum(unit.estimated_tokens for unit in group) for group_id, group in eligible_groups.items()
    }
    candidate_tokens = sum(group_costs.values())
    derived_budget = math.floor(candidate_tokens * policy.budget_fraction)
    smallest_group = min(group_costs.values())
    if derived_budget < smallest_group:
        raise ValueError(
            "The fractional budget cannot fit any complete dependency group; "
            f"derived_budget={derived_budget}, smallest_group={smallest_group}"
        )

    # Revalidate instead of model_copy so future ProxySelectionConfig rules are
    # also enforced for the derived per-batch absolute budget.
    derived_config_payload = config.model_dump(mode="json")
    derived_config_payload["token_budget"] = derived_budget
    derived_config = ProxySelectionConfig.model_validate(derived_config_payload)
    result = select_proxy(units, obligations, derived_config)

    selected_group_count = len(result["selected_group_ids"])
    eligible_group_count = len(eligible_groups)
    selected_group_fraction = selected_group_count / eligible_group_count
    selected_token_fraction = result["used_tokens"] / candidate_tokens
    if selected_group_count < policy.minimum_selected_groups:
        raise ValueError(f"Selection produced {selected_group_count} groups; requires {policy.minimum_selected_groups}")
    if policy.require_strict_subset and selected_group_count >= eligible_group_count:
        raise ValueError("A strict-subset ablation selected every eligible dependency group")
    if (
        policy.maximum_selected_group_fraction is not None
        and selected_group_fraction > policy.maximum_selected_group_fraction
    ):
        raise ValueError(
            "Selected group fraction exceeds the preregistered maximum: "
            f"{selected_group_fraction:.6f} > {policy.maximum_selected_group_fraction:.6f}"
        )

    pool_sha256 = candidate_pool_digest(units, obligations)
    budget_audit = {
        "schema_version": "fractional-selection-budget-v1",
        "policy": policy.model_dump(mode="json"),
        "candidate_pool_sha256": pool_sha256,
        "eligible_group_count": eligible_group_count,
        "eligible_unit_count": sum(len(group) for group in eligible_groups.values()),
        "eligible_estimated_tokens": candidate_tokens,
        "smallest_eligible_group_tokens": smallest_group,
        "derived_token_budget": derived_budget,
        "selected_group_count": selected_group_count,
        "selected_group_fraction": selected_group_fraction,
        "selected_token_fraction": selected_token_fraction,
        "strict_subset_observed": selected_group_count < eligible_group_count,
    }
    result["fractional_budget"] = budget_audit
    result["ablation_selection_sha256"] = _canonical_digest(
        {
            "candidate_pool_sha256": pool_sha256,
            "budget_audit": budget_audit,
            "selection_config_sha256": result["selection_config_sha256"],
            "selected_group_ids": result["selected_group_ids"],
            "selected_unit_ids": result["selected_unit_ids"],
        }
    )
    return result
