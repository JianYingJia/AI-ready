from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_ready.config import L2Config, PipelineConfig, load_config


def test_default_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "configs/l1.yaml"
    monkeypatch.chdir(tmp_path)
    config = load_config(path)
    assert config.input_dir.is_dir()
    assert config.cleaning.recipe.is_file()


def test_invalid_chunk_budget():
    path = Path(__file__).resolve().parents[1] / "configs/l1.yaml"
    data = load_config(path).model_dump()
    data["parsing"]["max_tokens"] = 1000
    with pytest.raises(ValidationError, match="max_tokens"):
        PipelineConfig.model_validate(data)


def test_output_cannot_be_inside_input():
    path = Path(__file__).resolve().parents[1] / "configs/l1.yaml"
    data = load_config(path).model_dump()
    data["output_dir"] = data["input_dir"] / "generated"
    with pytest.raises(ValidationError, match="inside"):
        PipelineConfig.model_validate(data)


def test_unknown_config_key_rejected():
    path = Path(__file__).resolve().parents[1] / "configs/l1.yaml"
    data = load_config(path).model_dump()
    data["embeding"] = {}
    with pytest.raises(ValidationError, match="Extra inputs"):
        PipelineConfig.model_validate(data)


def test_l2_rejects_unknown_and_zero_weight_configs():
    with pytest.raises(ValidationError, match="Unknown priority_weights"):
        L2Config(priority_weights={"unknown": 1.0})
    with pytest.raises(ValidationError, match="difficulty weight"):
        L2Config(difficulty_weights={key: 0.0 for key in L2Config().difficulty_weights.model_dump()})
