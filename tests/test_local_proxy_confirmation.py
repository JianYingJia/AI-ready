from __future__ import annotations

import torch

from ai_ready.valuation.local_confirmation import TrainingExample
from ai_ready.valuation.local_confirmation import (
    cap_supervised_tokens as _cap_supervised_tokens,
)
from ai_ready.valuation.local_confirmation import (
    content_static_score as _content_static_score,
)


def test_content_static_score_uses_frozen_cq_weights():
    score = _content_static_score(
        {"facility": 0.8, "obligation": 0.6, "quality_fidelity": 1.0, "source_coverage": 0.5},
        {},
    )
    assert score == (0.8 + 1.5 * 0.6 + 1.0 + 0.5 * 0.5) / 4.0


def test_cap_supervised_tokens_masks_only_excess_labels():
    first = TrainingExample(
        input_ids=torch.tensor([1, 2, 3, 4]),
        labels=torch.tensor([-100, 2, 3, 4]),
        supervised_tokens=3,
    )
    second = TrainingExample(
        input_ids=torch.tensor([5, 6, 7, 8]),
        labels=torch.tensor([-100, 6, 7, 8]),
        supervised_tokens=3,
    )
    capped = _cap_supervised_tokens([first, second], 4)
    assert [row.supervised_tokens for row in capped] == [3, 1]
    assert capped[1].labels.tolist() == [-100, 6, -100, -100]


def test_cap_supervised_tokens_rejects_insufficient_examples():
    example = TrainingExample(
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([-100, 2]),
        supervised_tokens=1,
    )
    try:
        _cap_supervised_tokens([example], 2)
    except ValueError as error:
        assert "provide 1/2" in str(error)
    else:
        raise AssertionError("Expected insufficient supervision to fail")
