"""Pure helpers for matched-exposure local confirmation experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TrainingExample:
    input_ids: torch.Tensor
    labels: torch.Tensor
    supervised_tokens: int


def content_static_score(components: dict[str, float], config: dict[str, Any]) -> float:
    weights = config.get(
        "content_static_weights",
        {"facility": 1.0, "obligation": 1.5, "quality_fidelity": 1.0, "source_coverage": 0.5},
    )
    denominator = sum(float(weight) for weight in weights.values())
    if denominator <= 0:
        raise ValueError("content_static_weights must have a positive sum")
    return sum(float(weights[name]) * float(components[name]) for name in weights) / denominator


def cap_supervised_tokens(examples: list[TrainingExample], target_tokens: int | None) -> list[TrainingExample]:
    if target_tokens is None:
        return examples
    if target_tokens <= 0:
        raise ValueError("matched_effective_supervised_tokens must be positive")
    capped: list[TrainingExample] = []
    remaining = target_tokens
    for example in examples:
        if remaining == 0:
            break
        if example.supervised_tokens <= remaining:
            capped.append(example)
            remaining -= example.supervised_tokens
            continue
        labels = example.labels.clone()
        supervised_positions = torch.nonzero(labels.ne(-100), as_tuple=False).flatten()
        labels[supervised_positions[remaining:]] = -100
        capped.append(TrainingExample(input_ids=example.input_ids, labels=labels, supervised_tokens=remaining))
        remaining = 0
    if remaining:
        raise ValueError(f"Selected examples provide {target_tokens - remaining}/{target_tokens} supervised tokens")
    return capped
