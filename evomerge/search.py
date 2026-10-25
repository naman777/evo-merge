"""Evolve a layer-wise merge recipe.

    python -m evomerge.search --config configs/qwen_hindi_math.json --optimizer ga --run ga_seed0

Writes to runs/<run>/: config.json, evals.jsonl (every fitness evaluation), generations.jsonl
(best/mean per generation), checkpoint.json (resumable state) and best_recipe.json.
Re-running the same command resumes from the last finished generation.
"""
import argparse
import json
import os
import time

import numpy as np

from .config import Config
from .data import load_problems
from .merge import save_recipe
from .optimizers import make_optimizer
from .runtime import Evaluator, load_merge_setup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument("--optimizer", choices=["ga", "cmaes", "random"], default="ga")
    parser.add_argument("--run", required=True, help="run name; output goes to runs/<run>/")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config values")
    parser.add_argument("--time-limit-hours", type=float, default=None,
                        help="stop cleanly after the generation that crosses this limit; re-run to resume")
    args = parser.parse_args(argv)
    started = time.time()

    cfg = Config.load(args.config, args.set)
    run_dir = os.path.join(args.runs_dir, args.run)
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_dir, "checkpoint.json")
    resuming = os.path.exists(checkpoint_path)
    if resuming:
        with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
            saved = json.load(f)
        # These do not change what the search does, so they may differ between sessions.
        free = {"max_evals", "test_max_new_tokens", "test_n", "test_split", "val_n"}
        same = {k: v for k, v in cfg.to_dict().items() if k not in free}
        before = {k: v for k, v in saved["config"].items() if k not in free}
        if saved["optimizer"] != args.optimizer or before != same:
            raise SystemExit(f"{run_dir} holds a run with a different optimizer or config; use another --run")
    else:
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump({"optimizer": args.optimizer, "config": cfg.to_dict()}, f, indent=2)

    if resuming:
        with open(checkpoint_path, encoding="utf-8") as f:
            ckpt = json.load(f)
        if ckpt["evals_done"] >= cfg.max_evals:
            print(f"{run_dir} already finished ({ckpt['evals_done']} evaluations); nothing to do")
            return

    problems = load_problems(cfg.dataset, cfg.search_split, cfg.search_n, cfg.data_seed)
    model, tokenizer, merger = load_merge_setup(cfg)
    evaluator = Evaluator(model, tokenizer, problems, cfg.system_prompt, cfg.max_new_tokens, cfg.batch_size)

    # Seed the search with both parents and their plain average.
    initial = [merger.uniform_genome([1.0, 0.0]), merger.uniform_genome([0.0, 1.0]),
               merger.uniform_genome([0.5, 0.5])]
    opt = make_optimizer(args.optimizer, merger.genome_size, cfg.weight_lo, cfg.weight_hi, cfg.seed,
                         initial, cfg.ga, cfg.cmaes)
    cache, evals_done = {}, 0
    if resuming:
        opt.load_state(ckpt["optimizer"])
        cache = {k: v for k, v in ckpt["cache"]}
        evals_done = ckpt["evals_done"]
        print(f"resumed at generation {opt.generation}, {evals_done} evaluations done", flush=True)

    while evals_done < cfg.max_evals:
        genomes = opt.ask()
        fitnesses = []
        for genome in genomes:
            key = _key(genome)
            if key not in cache:
                merger.apply(genome)
                accuracy, _, seconds = evaluator.evaluate()
                cache[key] = accuracy
                evals_done += 1
                _append(os.path.join(run_dir, "evals.jsonl"), {
                    "generation": opt.generation, "eval": evals_done, "fitness": accuracy,
                    "seconds": round(seconds, 1), "genome": np.round(genome, 6).tolist(),
                })
                print(f"[{args.optimizer}] gen {opt.generation} eval {evals_done}/{cfg.max_evals} "
                      f"acc={accuracy:.3f} ({seconds:.0f}s)", flush=True)
            fitnesses.append(cache[key])
        opt.tell(fitnesses)

        best_genome, best_fit = opt.best
        _append(os.path.join(run_dir, "generations.jsonl"), {
            "generation": opt.generation, "evals_done": evals_done, "best": best_fit,
            "gen_best": max(fitnesses), "gen_mean": float(np.mean(fitnesses)), "time": time.time(),
        })
        save_recipe(os.path.join(run_dir, "best_recipe.json"), best_genome, merger.num_layers,
                    [cfg.model_a, cfg.model_b], cfg.merge_method, cfg.density)
        with open(checkpoint_path + ".tmp", "w", encoding="utf-8") as f:
            json.dump({"optimizer": opt.state(), "cache": list(cache.items()), "evals_done": evals_done}, f)
        os.replace(checkpoint_path + ".tmp", checkpoint_path)
        print(f"[{args.optimizer}] generation {opt.generation} done: best={best_fit:.3f} "
              f"mean={np.mean(fitnesses):.3f}", flush=True)
        hours = (time.time() - started) / 3600
        if args.time_limit_hours and hours >= args.time_limit_hours and evals_done < cfg.max_evals:
            print(f"time limit reached after {hours:.1f} h at {evals_done}/{cfg.max_evals} evaluations; "
                  f"re-run the same command to resume", flush=True)
            return

    print(f"finished: best search accuracy {opt.best[1]:.3f}; recipe in {run_dir}/best_recipe.json")


def _key(genome):
    return ",".join(f"{x:.6f}" for x in np.asarray(genome))


def _append(path, record):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
