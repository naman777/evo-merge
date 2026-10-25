"""Pick each search's winner on a validation set instead of on its own (noisy) fitness.

    # score the top 10 distinct candidates of each run on validation (shard across GPUs if you like)
    python -m evomerge.rerank score --config configs/qwen_hindi_math.json \
        --runs runs/ga_seed0 runs/cmaes_seed0 runs/random_seed0 --shard 0 --num-shards 2
    # pick the best-on-validation candidate per run; writes results/rerank/<run>_reranked/best_recipe.json
    python -m evomerge.rerank pick --config configs/qwen_hindi_math.json --runs runs/ga_seed0 ...

The validation problems come from the search split but never from the 64 problems the search scored,
and use the test-time token budget. Scores go to results/rerank/val_scores_shard<i>.jsonl; scoring
skips candidates already in that file, so an interrupted run resumes.
"""
import argparse
import glob
import json
import os

from .config import Config
from .data import load_problems, load_validation
from .evaluate import BASELINES
from .merge import save_recipe


def top_candidates(run_dir, k):
    """The k best distinct genomes a run evaluated, best search fitness first (ties: earliest)."""
    with open(os.path.join(run_dir, "evals.jsonl"), encoding="utf-8") as f:
        evals = [json.loads(line) for line in f if line.strip()]
    evals.sort(key=lambda e: (-e["fitness"], e["eval"]))
    seen, picked = set(), []
    for e in evals:
        key = tuple(round(x, 6) for x in e["genome"])
        if key not in seen:
            seen.add(key)
            picked.append(e)
        if len(picked) == k:
            break
    return picked


def all_candidates(run_dirs, k, baselines):
    """(id, run, method, genome, search fitness) for every candidate, in a fixed order."""
    out = []
    for run_dir in run_dirs:
        run = os.path.basename(os.path.normpath(run_dir))
        with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
            method = json.load(f)["config"]["merge_method"]
        for e in top_candidates(run_dir, k):
            out.append((f"{run}/eval{e['eval']}", run, method, e["genome"], e["fitness"]))
    for name in baselines:
        out.append((f"baseline/{name}", "baseline", BASELINES[name][0], None, None))
    return out


def _read_scores(out_dir):
    scores = {}
    for path in glob.glob(os.path.join(out_dir, "val_scores_shard*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    scores[r["id"]] = r
    return scores


def score(cfg, args):
    from .runtime import Evaluator, load_merge_setup

    candidates = all_candidates(args.runs, args.top_k, args.baselines)
    mine = [c for i, c in enumerate(candidates) if i % args.num_shards == args.shard]
    done = _read_scores(args.out)
    todo = [c for c in mine if c[0] not in done]
    print(f"shard {args.shard}/{args.num_shards}: {len(mine)} candidates, {len(todo)} to score", flush=True)
    if not todo:
        return

    val = load_validation(cfg)
    search_ids = {p["id"] for p in load_problems(cfg.dataset, cfg.search_split, cfg.search_n, cfg.data_seed)}
    assert not search_ids & {p["id"] for p in val}, "validation overlaps the search set"
    model, tokenizer, merger = load_merge_setup(cfg)
    evaluator = Evaluator(model, tokenizer, val, cfg.system_prompt, cfg.test_max_new_tokens, cfg.batch_size)
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"val_scores_shard{args.shard}.jsonl")
    for cid, run, method, genome, search_fitness in todo:
        if genome is None:
            _, wa, wb = BASELINES[cid.split("/", 1)[1]]
            genome = merger.uniform_genome([wa, wb]).tolist()
        merger.method = method
        merger.apply(genome)
        accuracy, records, seconds = evaluator.evaluate()
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"id": cid, "run": run, "method": method, "search_fitness": search_fitness,
                                "val_accuracy": accuracy, "val_n": len(records), "genome": genome,
                                "correct": [r["id"] for r in records if r["correct"]]}) + "\n")
        print(f"{cid:28s} search={search_fitness if search_fitness is not None else '-':>8} "
              f"val={accuracy:.3f} ({seconds:.0f}s)", flush=True)


def pick(cfg, args):
    scores = _read_scores(args.out)
    rows = []
    for run_dir in args.runs:
        run = os.path.basename(os.path.normpath(run_dir))
        mine = [r for r in scores.values() if r["run"] == run]
        if not mine:
            raise SystemExit(f"no validation scores for {run}; run `score` first")
        mine.sort(key=lambda r: (-r["val_accuracy"], -r["search_fitness"]))
        best = mine[0]
        top_search = max(mine, key=lambda r: r["search_fitness"])
        out_dir = os.path.join(args.out, f"{run}_reranked")
        os.makedirs(out_dir, exist_ok=True)
        save_recipe(os.path.join(out_dir, "best_recipe.json"), best["genome"], len(best["genome"]) // 2 - 2,
                    [cfg.model_a, cfg.model_b], best["method"], cfg.density)
        rows.append((run, len(mine), best["id"], best["search_fitness"], best["val_accuracy"],
                     top_search["id"], top_search["val_accuracy"]))
    print(f"{'run':14s} {'scored':>6s}  {'picked by validation':26s} {'search':>6s} {'val':>6s}   "
          f"{'search winner':26s} {'its val':>7s}")
    for run, n, bid, bs, bv, tid, tv in rows:
        print(f"{run:14s} {n:6d}  {bid:26s} {bs:6.3f} {bv:6.3f}   {tid:26s} {tv:7.3f}")
    for r in sorted(scores.values(), key=lambda r: r["id"]):
        if r["run"] == "baseline":
            print(f"{r['id']:28s} val={r['val_accuracy']:.3f}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["score", "pick"])
    parser.add_argument("--config")
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--baselines", nargs="*", default=["average", "task_arithmetic"], choices=list(BASELINES))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--out", default="results/rerank")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config, args.set)
    (score if args.command == "score" else pick)(cfg, args)


if __name__ == "__main__":
    main()
