import pytest

from shoprl.data.catalog import Product
from shoprl.reward import RewardContext

# A tiny hand-built catalog so every reward assertion is obvious by inspection.
#   LAP-0001: $900,  16GB, 3.0lbs, 12hrs  -> meets {max_price 1000, min_ram 16}
#   LAP-0002: $1500,  8GB, 5.0lbs,  6hrs  -> violates both
#   LAP-0003: $700,  32GB, 2.5lbs, 15hrs  -> meets both
_PRODUCTS = [
    Product(sku="LAP-0001", name="Acer UltraBook 14\"", price=900.0, ram_gb=16, weight_lbs=3.0, battery_hrs=12, brand="Acer"),
    Product(sku="LAP-0002", name="Dell ProBook 15\"", price=1500.0, ram_gb=8, weight_lbs=5.0, battery_hrs=6, brand="Dell"),
    Product(sku="LAP-0003", name="Apple ZenSlim 13\"", price=700.0, ram_gb=32, weight_lbs=2.5, battery_hrs=15, brand="Apple"),
]


@pytest.fixture
def catalog() -> dict[str, Product]:
    return {p.sku: p for p in _PRODUCTS}


@pytest.fixture
def ctx(catalog) -> RewardContext:
    return RewardContext(catalog=catalog, constraints={"max_price": 1000.0, "min_ram": 16})
