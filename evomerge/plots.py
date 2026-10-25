"""Figures for the report.

    python -m evomerge.plots --runs runs/ga_seed0 runs/cmaes_seed0 runs/random_seed0 --results results
"""
import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def convergence(run_dirs, out):
    """Best-so-far search accuracy against the number of fitness evaluations."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for run_dir in run_dirs:
        evals = _read_jsonl(os.path.join(run_dir, "evals.jsonl"))
        best, xs, ys = 0.0, [], []
        for e in evals:
            best = max(best, e["fitness"])
            xs.append(e["eval"])
            ys.append(best)
        ax.plot(xs, ys, label=os.path.basename(os.path.normpath(run_dir)))
    ax.set_xlabel("fitness evaluations")
    ax.set_ylabel("best search-set accuracy")
    ax.set_title("Convergence")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "convergence.png"), dpi=150)
    plt.close(fig)


def recipe_profile(run_dir, out):
    """Evolved per-layer weight of each parent."""
    with open(os.path.join(run_dir, "best_recipe.json"), encoding="utf-8") as f:
        recipe = json.load(f)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for parent, weights in recipe["weights"].items():
        ax.plot(range(len(weights)), weights, marker="o", ms=3, label=parent.split("/")[-1])
    ax.set_xticks(range(len(recipe["groups"])))
    ax.set_xticklabels([g.replace("layer", "") for g in recipe["groups"]], fontsize=7)
    ax.set_xlabel("parameter group (embed, layers 0..L-1, final)")
    ax.set_ylabel("merge weight")
    name = os.path.basename(os.path.normpath(run_dir))
    ax.set_title(f"Evolved merge recipe ({name})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"recipe_{name}.png"), dpi=150)
    plt.close(fig)


def test_accuracy(results_dirs, out):
    rows = []
    for results_dir in results_dirs:
        path = os.path.join(results_dir, "summary.csv")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                rows += [r for r in csv.DictReader(f) if r["split"].startswith("test")]
    if not rows:
        return
    names = [r["name"] for r in rows]
    accs = [100 * float(r["accuracy"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(names, accs, color="#4C72B0")
    ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Hindi GSM8K test accuracy")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "test_accuracy.png"), dpi=150)
    plt.close(fig)


def generalization(search_dir, results_dirs, out):
    """Search-set accuracy against test accuracy for every candidate scored on both."""
    search, test = {}, {}
    for path, store in [(os.path.join(search_dir, "summary.csv"), search)] + [
            (os.path.join(d, "summary.csv"), test) for d in results_dirs]:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    store[r["name"]] = 100 * float(r["accuracy"])
    for run_dir in glob.glob("runs/*_seed0"):
        name = os.path.basename(run_dir)
        evals = _read_jsonl(os.path.join(run_dir, "evals.jsonl"))
        search[name] = 100 * max(e["fitness"] for e in evals)
    names = [n for n in test if n in search]
    if not names:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for n in names:
        evolved = n.endswith("_seed0")
        ax.scatter(search[n], test[n], color="#DD8452" if evolved else "#4C72B0", s=40)
        ax.annotate(n, (search[n], test[n]), fontsize=7, xytext=(4, 3), textcoords="offset points")
    lo = min(min(search[n] for n in names), min(test[n] for n in names)) - 3
    hi = max(max(search[n] for n in names), max(test[n] for n in names)) + 3
    ax.plot([lo, hi], [lo, hi], color="gray", lw=0.8, ls="--")
    ax.set_xlabel("search-set accuracy (%), 64 problems")
    ax.set_ylabel("test accuracy (%), 250 problems")
    ax.set_title("Search vs held-out test (orange = evolved)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "generalization.png"), dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=[])
    parser.add_argument("--results", nargs="*", default=["results"], help="dirs holding summary.csv")
    parser.add_argument("--search-results", default=None, help="dir with search-split summary.csv (pilot)")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.runs:
        convergence(args.runs, args.out)
        for run_dir in args.runs:
            recipe_profile(run_dir, args.out)
    test_accuracy(args.results, args.out)
    if args.search_results:
        generalization(args.search_results, args.results, args.out)
    print(f"figures written to {args.out}/")


if __name__ == "__main__":
    main()
