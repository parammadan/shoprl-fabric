"""Extract what a policy response *claims*, deterministically.

This is the bridge from free text to checkable facts. We pull:
  - every SKU the response cites (LAP-####), and
  - any specs it states for a SKU (price/RAM/weight/battery).

Nothing here judges correctness — it only reports claims. The reward functions
compare these claims against the catalog (ground truth).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Match any LAP-<digits> token as a *claim*; catalog membership (not the regex)
# decides validity. This way a garbled/invented id like "LAP-00106" is caught
# as a hallucinated SKU directly, instead of a greedy 4-digit match silently
# truncating it to a real one.
SKU_RE = re.compile(r"LAP-\d+", re.IGNORECASE)
PRICE_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")
RAM_RE = re.compile(r"(\d+)\s*GB", re.IGNORECASE)
WEIGHT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*lbs?", re.IGNORECASE)
BATTERY_RE = re.compile(r"(\d+)\s*(?:hrs?|hours?)", re.IGNORECASE)


@dataclass
class ParsedRec:
    """One recommendation claimed by the response (keyed by SKU)."""

    sku: str
    stated_price: float | None = None
    stated_ram: int | None = None
    stated_weight: float | None = None
    stated_battery: int | None = None

    @property
    def num_specs_stated(self) -> int:
        return sum(
            v is not None
            for v in (
                self.stated_price,
                self.stated_ram,
                self.stated_weight,
                self.stated_battery,
            )
        )


def _extract_specs(text: str) -> dict[str, float | int | None]:
    price = PRICE_RE.search(text)
    ram = RAM_RE.search(text)
    weight = WEIGHT_RE.search(text)
    battery = BATTERY_RE.search(text)
    return {
        "stated_price": float(price.group(1).replace(",", "")) if price else None,
        "stated_ram": int(ram.group(1)) if ram else None,
        "stated_weight": float(weight.group(1)) if weight else None,
        "stated_battery": int(battery.group(1)) if battery else None,
    }


def parse_response(response: str) -> list[ParsedRec]:
    """Return one ParsedRec per unique claimed SKU, in first-seen order.

    Specs are parsed line-by-line and attributed to the SKU(s) on that line. If
    a line uses `field | field | ... | reason` form, we skip the trailing reason
    field so prose digits (e.g. "great for 8 hours") aren't mistaken for specs.
    """
    recs: dict[str, ParsedRec] = {}
    order: list[str] = []

    for line in response.splitlines():
        skus = [m.upper() for m in SKU_RE.findall(line)]
        if not skus:
            continue

        # Isolate the structured portion from any free-text reason.
        spec_text = line
        if "|" in line:
            spec_text = "|".join(line.split("|")[:-1])
        specs = _extract_specs(spec_text)

        for sku in skus:
            if sku not in recs:
                recs[sku] = ParsedRec(sku=sku, **specs)  # type: ignore[arg-type]
                order.append(sku)
            else:
                # Backfill any spec this line supplies that we didn't have yet.
                existing = recs[sku]
                for field, value in specs.items():
                    if value is not None and getattr(existing, field) is None:
                        setattr(existing, field, value)

    return [recs[s] for s in order]
