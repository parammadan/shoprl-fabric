from shoprl.data import generate_catalog, generate_prompts, satisfies
from shoprl.data.catalog import catalog_index
from shoprl.task import build_shortlist, build_task_prompt


def test_shortlist_has_mix_and_size():
    cat = generate_catalog(n=300, seed=0)
    ex = generate_prompts(cat, n=10, seed=0)[0]
    sl = build_shortlist(ex, cat, k=6, seed=0)
    assert 1 <= len(sl) <= 6
    assert len(set(sl)) == len(sl)  # unique
    idx = catalog_index(cat)
    sat = [s for s in sl if satisfies(idx[s], ex.constraints)]
    # Guaranteed at least one satisfying candidate exists in the shortlist.
    assert len(sat) >= 1


def test_shortlist_deterministic():
    cat = generate_catalog(n=300, seed=0)
    ex = generate_prompts(cat, n=10, seed=0)[0]
    assert build_shortlist(ex, cat, seed=0) == build_shortlist(ex, cat, seed=0)


def test_prompt_contains_skus_and_format():
    cat = generate_catalog(n=300, seed=0)
    idx = catalog_index(cat)
    ex = generate_prompts(cat, n=10, seed=0)[0]
    sl = build_shortlist(ex, cat, k=6, seed=0)
    prompt = build_task_prompt(ex, idx, sl)
    for sku in sl:
        assert sku in prompt
    assert "REC:" in prompt
    assert ex.prompt in prompt
