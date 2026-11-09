from ai_ready.config import DifficultyWeights
from ai_ready.difficulty.model import build_task_route, document_difficulty


def test_document_difficulty_emits_six_separate_signals(unit_factory):
    simple = unit_factory("simple", "申请人提交材料。")
    complex_unit = unit_factory(
        "complex",
        "根据第12条，如果客户在2026年1月1日后申请，则必须提交三项材料；但公积金贷款另有规定，不得套用。",
        ["贷款", "例外规则"],
    )
    complex_unit.cross_references = ["第12条"]
    complex_unit.structure_labels = ["table", "list_item"]
    weights = DifficultyWeights()
    simple_profile = document_difficulty(simple, None, weights)
    complex_profile = document_difficulty(complex_unit, None, weights)
    assert complex_profile.score > simple_profile.score
    assert set(complex_profile.model_dump()) >= {
        "structure",
        "language",
        "knowledge_density",
        "dependency",
        "temporal",
        "ambiguity",
    }


def test_task_feasibility_and_task_difficulty_are_not_document_score(unit_factory):
    unit = unit_factory("a", "商业贷款最低金额为10000元。")
    factual = build_task_route("anchor", "topic", "factual_qa", [unit], {}, 0.9)
    comparison = build_task_route("anchor", "topic", "comparison", [unit], {}, 0.9)
    assert factual.feasible
    assert not comparison.feasible
    assert factual.predicted_score != 0.9
    assert factual.routing_priority < 1.0
    assert factual.empirical_score is None


def test_single_date_is_not_treated_as_cross_version_hardness(unit_factory):
    unit = unit_factory("date", "本规定自2026年1月1日起生效；如果申请人办理业务，则必须提交材料。")
    route = build_task_route("anchor", "topic", "judgment", [unit], {}, 0.2)
    assert route.feasible
    assert route.temporal_reasoning == 0.5
    assert "hard" not in route.feasible_levels
