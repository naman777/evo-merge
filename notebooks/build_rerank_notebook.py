"""Generates kaggle_rerank.ipynb. Edit the cells here, then run: python notebooks/build_rerank_notebook.py"""
import json
import os

CELLS = [
    ("md", """# Re-ranking evolved recipes on a validation set (2x T4)

Inputs: the `evo-merge` code dataset (a version that has `evomerge/rerank.py`) and the output of the
`evo-merge-run` notebook (its finished `runs/`).

1. Each search's 10 best distinct candidates, plus the 50/50 average and task arithmetic, are scored on
   250 validation problems: training problems the search never saw, with the test-time budget of 1024 tokens.
2. Each search's winner is re-picked by validation score.
3. The re-picked winners are scored on the 250 held-out test problems.

Every step skips work already done, so if the run stops early, attach this notebook's own Output and run it again."""),
    ("code", """import glob, json, os, shutil, subprocess, time
T0 = time.time()
WORK = "/kaggle/working/evo-merge"
CONFIG = "configs/qwen_hindi_math.json"
RUNS = "runs/ga_seed0 runs/cmaes_seed0 runs/random_seed0"

if not os.path.exists(WORK):
    code = sorted(glob.glob("/kaggle/input/**/evomerge/rerank.py", recursive=True))
    assert code, "Attach a version of the evo-merge dataset that contains evomerge/rerank.py"
    shutil.copytree(os.path.dirname(os.path.dirname(code[0])), WORK,
                    ignore=shutil.ignore_patterns("runs", "results", "figures", "__pycache__", ".pytest_cache"))
for old in glob.glob("/kaggle/input/**/evo-merge/runs/*_seed0/evals.jsonl", recursive=True):
    run_dir = os.path.dirname(old)
    shutil.copytree(run_dir, f"{WORK}/runs/{os.path.basename(run_dir)}", dirs_exist_ok=True)
for old in glob.glob("/kaggle/input/**/evo-merge/results/rerank", recursive=True):  # resume an earlier re-rank
    shutil.copytree(old, f"{WORK}/results/rerank", dirs_exist_ok=True)

os.chdir(WORK)
print("runs:", sorted(os.listdir("runs")))
!pip install -q cma
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """def tail(path):
    try:
        lines = open(path, encoding="utf-8").read().strip().splitlines()
        return lines[-1] if lines else ""
    except FileNotFoundError:
        return ""

def run_parallel(jobs, report_every=900):
    \"\"\"jobs: list of (gpu, log file, shell command). Runs them at once and waits for all of them.\"\"\"
    procs = []
    for gpu, log, cmd in jobs:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_DISABLE_SYMLINKS_WARNING="1")
        print(f"[GPU {gpu}] {cmd}")
        procs.append((log, subprocess.Popen(cmd, shell=True, env=env, stdout=open(log, "a"),
                                            stderr=subprocess.STDOUT)))
    while any(p.poll() is None for _, p in procs):
        time.sleep(30)
        if int(time.time() - T0) % report_every < 30:
            for log, _ in procs:
                print(f"{(time.time() - T0) / 3600:5.2f} h | {log}: {tail(log)}", flush=True)
    for log, p in procs:
        print(f"{log}: exit code {p.returncode} | {tail(log)}")"""),
    ("code", """!python -m pytest -q"""),
    ("md", """## 1. Score candidates on validation (both GPUs)"""),
    ("code", """run_parallel([(s, f"rerank_shard{s}.log",
               f"python -m evomerge.rerank score --config {CONFIG} --runs {RUNS} --shard {s} --num-shards 2")
              for s in (0, 1)])"""),
    ("md", """## 2. Re-pick each search's winner by validation score"""),
    ("code", """!python -m evomerge.rerank pick --config {CONFIG} --runs {RUNS}"""),
    ("md", """## 3. Re-picked winners on the held-out test set"""),
    ("code", """jobs = []
for gpu, runs in [(0, ["ga_seed0", "cmaes_seed0"]), (1, ["random_seed0"])]:
    out = f"results/test_reranked_{gpu}"
    if not os.path.exists(f"{out}/summary.csv"):
        recipes = " ".join(f"results/rerank/{r}_reranked" for r in runs)
        jobs.append((gpu, f"test_reranked_{gpu}.log",
                     f"python -m evomerge.evaluate --config {CONFIG} --baselines --recipes {recipes} --out {out}"))
if jobs:
    run_parallel(jobs)
for gpu in (0, 1):
    path = f"results/test_reranked_{gpu}/summary.csv"
    print(open(path).read() if os.path.exists(path) else f"missing {path}")
done = all(os.path.exists(f"results/test_reranked_{g}/summary.csv") for g in (0, 1))
print("ALL DONE" if done else "NOT FINISHED: attach this version's Output as an input and run again")"""),
]


def cell(kind, text):
    lines = text.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": lines}


notebook = {
    "cells": [cell(k, t) for k, t in CELLS],
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                 "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_rerank.ipynb")
with open(path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print(f"wrote {path}")
