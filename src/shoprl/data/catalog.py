"""Laptop catalog generator.

The catalog is ground truth. Each product has a unique SKU (the key the reward
layer uses to detect hallucination) plus specs the prompts constrain over.
Specs are mildly correlated (more RAM / lighter / longer battery costs more) so
constraint trade-offs are non-trivial, but the numbers are synthetic.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

BRANDS = ["Acer", "Dell", "Lenovo", "HP", "Asus", "Apple", "Framework", "Razer"]
LINES = ["UltraBook", "ProBook", "ZenSlim", "AeroLite", "PowerLine", "NoteMax"]
RAM_OPTIONS = [8, 16, 32, 64]


class Product(BaseModel):
    sku: str
    name: str
    price: float
    ram_gb: int
    weight_lbs: float
    battery_hrs: int
    brand: str


def _make_product(i: int, rng: random.Random) -> Product:
    brand = rng.choice(BRANDS)
    line = rng.choice(LINES)
    screen = rng.choice([13, 14, 15, 16])
    ram_gb = rng.choice(RAM_OPTIONS)
    weight_lbs = round(rng.uniform(2.0, 6.0), 1)
    battery_hrs = rng.randint(4, 20)

    # Price is driven by specs plus noise: RAM and battery push it up, weight
    # pushes it down (lighter = premium). Keeps constraint satisfaction from
    # being trivially "cheapest = best".
    base = 350
    price = (
        base
        + ram_gb * 22
        + battery_hrs * 18
        + (6.0 - weight_lbs) * 90
        + rng.uniform(-120, 120)
    )
    price = round(max(300.0, price), 2)

    return Product(
        sku=f"LAP-{i:04d}",
        name=f"{brand} {line} {screen}\"",
        price=price,
        ram_gb=ram_gb,
        weight_lbs=weight_lbs,
        battery_hrs=battery_hrs,
        brand=brand,
    )


def generate_catalog(n: int = 300, seed: int = 0) -> list[Product]:
    """Deterministically generate `n` laptops."""
    rng = random.Random(f"catalog-{seed}")
    return [_make_product(i, rng) for i in range(1, n + 1)]


def catalog_index(products: list[Product]) -> dict[str, Product]:
    """SKU -> Product lookup — the reward layer's ground-truth map."""
    return {p.sku: p for p in products}


def write_jsonl(products: list[Product], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in products:
            f.write(json.dumps(p.model_dump()) + "\n")
    return path


def read_jsonl(path: str | Path) -> list[Product]:
    with Path(path).open("r") as f:
        return [Product.model_validate_json(line) for line in f if line.strip()]
