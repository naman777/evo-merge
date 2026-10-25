"""Score the parents, standard merges and evolved recipes on the held-out test set.

    python -m evomerge.evaluate --config configs/qwen_hindi_math.json --recipes runs/ga_seed0 runs/cmaes_seed0

Every candidate is written into the same live model, so all of them share one architecture config
(and RoPE base). Results go to results/<name>.json and a summary to results/summary.csv.
"""
import argparse
import csv
import json
import os

from .config import Config
from .data import load_problems
from .runtime import Evaluator, load_merge_setup

# name -> (merge method, weight for A, weight for B). Parents are "merges" with one weight set to 1.
BASELINES = {
    "parent_a": ("task_arithmetic", 1.0, 0.0),
    "parent_b": ("task_arithmetic", 0.0, 1.0),
    "average": ("task_arithmetic", 0.5, 0.5),
    "task_arithmetic": ("task_arithmetic", 1.0, 1.0),
    "ties": ("ties", 1.0, 1.0),
    "dare_ties": ("dare_ties", 1.0, 1.0),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--recipes", nargs="*", default=[], help="run directories holding best_recipe.json")
    parser.add_argument("--baselines", nargs="*", default=list(BASELINES), choices=list(BASELINES))
    parser.add_argument("--split", choices=["test", "search"], default="test")
    parser.add_argument("--out", default="results")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config, args.set)
    if args.split == "test":
        split, n, max_new_tokens = cfg.test_split, cfg.test_n, cfg.test_max_new_tokens
    else:
        split, n, max_new_tokens = cfg.search_split, cfg.search_n, cfg.max_new_tokens
    problems = load_problems(cfg.dataset, split, n, cfg.data_seed)
    model, tokenizer, merger = load_merge_setup(cfg)
    evaluator = Evaluator(model, tokenizer, problems, cfg.system_prompt, max_new_tokens, cfg.batch_size)
    os.makedirs(args.out, exist_ok=True)

    candidates = []
    for name in args.baselines:
        method, wa, wb = BASELINES[name]
        candidates.append((name, method, merger.uniform_genome([wa, wb])))
    for run_dir in args.recipes:
        with open(os.path.join(run_dir, "best_recipe.json"), encoding="utf-8") as f:
            recipe = json.load(f)
        candidates.append((os.path.basename(os.path.normpath(run_dir)), recipe["method"], recipe["genome"]))

    rows = []
    for name, method, genome in candidates:
        merger.method = method
        merger.apply(genome)
        accuracy, records, seconds = evaluator.evaluate(keep_outputs=True)
        with open(os.path.join(args.out, f"{name}.json"), "w", encoding="utf-8") as f:
            json.dump({"name": name, "method": method, "split": split, "n": len(records),
                       "max_new_tokens": max_new_tokens, "accuracy": accuracy, "records": records},
                      f, ensure_ascii=False, indent=1)
        rows.append({"name": name, "method": method, "split": split, "n": len(records),
                     "max_new_tokens": max_new_tokens, "accuracy": round(accuracy, 4), "seconds": round(seconds)})
        print(f"{name:20s} {method:16s} acc={accuracy:.3f}  ({seconds:.0f}s)", flush=True)

    summary = os.path.join(args.out, "summary.csv")
    write_header = not os.path.exists(summary)
    with open(summary, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) + ["config_from"])
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({**row, "config_from": cfg.config_from})


if __name__ == "__main__":
    main()
