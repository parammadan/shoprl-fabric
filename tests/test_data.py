from shoprl.data import generate_catalog, generate_prompts, satisfies
from shoprl.data.catalog import catalog_index


def test_catalog_fields_and_uniqueness():
    cat = generate_catalog(n=300, seed=0)
    assert len(cat) == 300
    skus = [p.sku for p in cat]
    assert len(set(skus)) == 300  # unique SKUs
    for p in cat:
        assert p.sku.startswith("LAP-")
        assert p.price >= 300.0
        assert p.ram_gb in (8, 16, 32, 64)
        assert 2.0 <= p.weight_lbs <= 6.0
        assert 4 <= p.battery_hrs <= 20


def test_catalog_deterministic():
    a = generate_catalog(n=50, seed=0)
    b = generate_catalog(n=50, seed=0)
    assert [p.model_dump() for p in a] == [p.model_dump() for p in b]


def test_ground_truth_matches_predicate():
    cat = generate_catalog(n=300, seed=1)
    idx = catalog_index(cat)
    prompts = generate_prompts(cat, n=50, seed=1)
    for ex in prompts:
        # Every answer SKU truly satisfies the constraints...
        for sku in ex.answer_skus:
            assert satisfies(idx[sku], ex.constraints)
        # ...and no catalog product outside the answer set does.
        answer_set = set(ex.answer_skus)
        for p in cat:
            if p.sku not in answer_set:
                assert not satisfies(p, ex.constraints)


def test_prompts_nonempty_answers_by_default():
    cat = generate_catalog(n=300, seed=2)
    prompts = generate_prompts(cat, n=50, seed=2)
    assert all(len(ex.answer_skus) >= 1 for ex in prompts)
    assert all(ex.constraints for ex in prompts)  # at least one constraint
