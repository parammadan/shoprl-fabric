from pathlib import Path

import pytest

from shoprl.config import Config, load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_are_sane():
    c = Config()
    assert c.model.name == "Qwen/Qwen3-0.6B"
    assert c.rollout.num_samples >= 1
    assert c.rollout.engine in {"stub", "hf", "vllm"}


def test_load_dev_yaml():
    c = load_config(CONFIGS / "dev.yaml")
    assert c.experiment.name == "shoprl-dev"
    assert c.rollout.engine == "hf"
    assert c.rollout.max_new_tokens == 64


def test_load_stub_yaml():
    c = load_config(CONFIGS / "stub.yaml")
    assert c.rollout.engine == "stub"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_config(CONFIGS / "does-not-exist.yaml")


def test_bad_num_samples_rejected():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Config.model_validate({"rollout": {"num_samples": 0}})
