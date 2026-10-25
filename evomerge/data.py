"""Loading fixed, reproducible problem subsets from Hindi GSM8K."""
import random

from .answers import extract_gold

# Hindi GSM8K is a row-by-row translation of openai/gsm8k. About 4% of the translated solutions lost
# their "####" answer marker, so reference answers are taken from the English original instead.
GOLD_SOURCE = {"bingbangboom/gsm8k-hindi": ("openai/gsm8k", "main", {"train_main": "train", "test_main": "test"})}


def load_problems(dataset, split, n, seed=0):
    """Return n problems as dicts {id, question, gold}, sampled deterministically from the split."""
    from datasets import load_dataset

    rows = load_dataset(dataset, split=split)
    golds = [extract_gold(r["answer"]) for r in rows]
    if dataset in GOLD_SOURCE:
        source, config, splits = GOLD_SOURCE[dataset]
        original = load_dataset(source, config, split=splits[split])
        if len(original) != len(rows):
            raise ValueError(f"{dataset}/{split} and {source}/{splits[split]} differ in length")
        golds = [extract_gold(r["answer"]) for r in original]

    problems = [
        {"id": f"{split}/{i}", "question": row["question"].strip(), "gold": gold}
        for i, (row, gold) in enumerate(zip(rows, golds))
        if gold is not None and row["question"].strip()
    ]
    rng = random.Random(seed)
    rng.shuffle(problems)
    return problems[:n] if n else problems


def load_validation(cfg):
    """Problems search_n .. search_n + val_n of the search split's shuffle: never seen by the search."""
    problems = load_problems(cfg.dataset, cfg.search_split, cfg.search_n + cfg.val_n, cfg.data_seed)
    return problems[cfg.search_n:]
