from shoprl.config import Config
from shoprl.rollout import RolloutGroup, StubRolloutEngine
from shoprl.rollout.factory import build_engine

PROMPTS = ["recommend earbuds", "recommend a bottle"]


def test_shape_one_group_per_prompt():
    eng = StubRolloutEngine(seed=0)
    groups = eng.generate(PROMPTS, num_samples=4)
    assert len(groups) == len(PROMPTS)
    assert all(isinstance(g, RolloutGroup) for g in groups)
    assert all(len(g) == 4 for g in groups)


def test_deterministic_given_seed():
    a = StubRolloutEngine(seed=0).generate(PROMPTS, num_samples=3)
    b = StubRolloutEngine(seed=0).generate(PROMPTS, num_samples=3)
    texts_a = [[c.text for c in g.completions] for g in a]
    texts_b = [[c.text for c in g.completions] for g in b]
    assert texts_a == texts_b


def test_prompt_preserved_on_completions():
    groups = StubRolloutEngine().generate(PROMPTS, num_samples=2)
    for g in groups:
        assert all(c.prompt == g.prompt for c in g.completions)


def test_factory_builds_stub_from_config():
    cfg = Config.model_validate({"rollout": {"engine": "stub"}})
    eng = build_engine(cfg)
    assert isinstance(eng, StubRolloutEngine)
