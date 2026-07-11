"""CLI: materialize the synthetic dataset to JSONL.

    python -m shoprl.data.build --out data --n-catalog 300 --n-prompts 200

Output is reproducible from --seed, so we gitignore data/ rather than commit it.
"""
from __future__ import annotations

import argparse

from shoprl.data import catalog as catalog_mod
from shoprl.data import prompts as prompts_mod


def main() -> None:
    ap = argparse.ArgumentParser(prog="shoprl.data.build")
    ap.add_argument("--out", default="data", help="Output directory.")
    ap.add_argument("--n-catalog", type=int, default=300)
    ap.add_argument("--n-prompts", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    products = catalog_mod.generate_catalog(n=args.n_catalog, seed=args.seed)
    examples = prompts_mod.generate_prompts(products, n=args.n_prompts, seed=args.seed)

    cat_path = catalog_mod.write_jsonl(products, f"{args.out}/catalog.jsonl")
    prm_path = prompts_mod.write_jsonl(examples, f"{args.out}/prompts.jsonl")

    avg_answers = sum(len(e.answer_skus) for e in examples) / max(1, len(examples))
    print(f"wrote {len(products)} products -> {cat_path}")
    print(f"wrote {len(examples)} prompts  -> {prm_path}")
    print(f"avg ground-truth answers/prompt: {avg_answers:.1f}")


if __name__ == "__main__":
    main()
