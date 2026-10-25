"""Generates kaggle_run.ipynb. Edit the cells here, then run: python notebooks/build_kaggle_notebook.py"""
import json
import os

CELLS = [
    ("md", """# Evolutionary Model Merging on Kaggle (2x T4)

**Setup (once):** upload `evo-merge-kaggle.zip` as a Kaggle Dataset, then in this notebook use
Add Input to attach it. Settings: Accelerator **GPU T4 x2**, Internet **on**.

**Run:** Save Version, then **Save & Run All (Commit)**. The run continues with the browser closed.
Every stage skips work that is already done and stops in time to save its outputs before Kaggle's
12-hour limit.

**If the run stops early** (the last cell says so): open this notebook, Add Input, then attach this
notebook's own latest **Output**, and Save & Run All again. The setup cell copies the previous `runs/`
and `results/` back, and the searches resume from their last finished generation."""),
    ("code", """import glob, json, os, shutil, subprocess, time
T0 = time.time()
SESSION_HOURS = 11.0  # Kaggle stops at 12 h; keep about an hour to save outputs
WORK = "/kaggle/working/evo-merge"
CONFIG = "configs/qwen_hindi_math.json"

if not os.path.exists(WORK):
    code = sorted(glob.glob("/kaggle/input/**/evomerge/__init__.py", recursive=True))
    assert code, "Attach the evo-merge dataset with Add Input"
    shutil.copytree(os.path.dirname(os.path.dirname(code[0])), WORK,
                    ignore=shutil.ignore_patterns("runs", "results", "figures"))

# Resume: bring back runs and results from an attached earlier output of this notebook.
for old in glob.glob("/kaggle/input/**/evo-merge/runs/*/checkpoint.json", recursive=True):
    run_dir = os.path.dirname(old)
    shutil.copytree(run_dir, f"{WORK}/runs/{os.path.basename(run_dir)}", dirs_exist_ok=True)
for old in glob.glob("/kaggle/input/**/evo-merge/results", recursive=True):
    shutil.copytree(old, f"{WORK}/results", dirs_exist_ok=True)

os.chdir(WORK)
print("code in", WORK, "| resumed runs:", sorted(os.listdir("runs")) if os.path.isdir("runs") else [])
!pip install -q cma
!nvidia-smi --query-gpu=name,memory.total --format=csv"""),
    ("code", """MAX_EVALS = json.load(open(CONFIG))["max_evals"]

def hours_left():
    return SESSION_HOURS - (time.time() - T0) / 3600

def finished(run):
    path = f"runs/{run}/checkpoint.json"
    return os.path.exists(path) and json.load(open(path))["evals_done"] >= MAX_EVALS

def tail(path):
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
        print(f"{log}: exit code {p.returncode} | {tail(log)}")

def search(gpu, optimizer):
    run = f"{optimizer}_seed0"
    return (gpu, f"{run}.log", f"python -m evomerge.search --config {CONFIG} --optimizer {optimizer} "
                               f"--run {run} --time-limit-hours {max(0.5, hours_left() - 0.75):.2f}")"""),
    ("code", """!python -m pytest -q"""),
    ("md", """## 1. Search: GA (GPU 0) and CMA-ES (GPU 1) in parallel
Same budget for both (`max_evals` fitness evaluations, each on 64 Hindi GSM8K training problems)."""),
    ("code", """jobs = [search(gpu, opt) for gpu, opt in [(0, "ga"), (1, "cmaes")] if not finished(f"{opt}_seed0")]
if jobs:
    run_parallel(jobs)
print({r: finished(r) for r in ["ga_seed0", "cmaes_seed0"]})"""),
    ("md", """## 2. Random search (GPU 0) and the baselines on the test set (GPU 1)
Random search is the control: same budget, no evolution. The baselines (both parents, the plain
average, task arithmetic, TIES, DARE-TIES) are scored on the 250 held-out test problems with up to
1024 new tokens (`test_max_new_tokens`); the search used 512, see docs/findings.md."""),
    ("code", """jobs = []
if finished("ga_seed0") and finished("cmaes_seed0") and hours_left() > 1:
    if not finished("random_seed0"):
        jobs.append(search(0, "random"))
    if not os.path.exists("results/test_baselines/summary.csv"):
        jobs.append((1, "test_baselines.log",
                     f"python -m evomerge.evaluate --config {CONFIG} --out results/test_baselines"))
    run_parallel(jobs)
else:
    print("waiting for stage 1 or out of time; re-run this notebook to continue")"""),
    ("md", """## 3. Evolved recipes on the test set"""),
    ("code", """runs = ["ga_seed0", "cmaes_seed0", "random_seed0"]
if all(finished(r) for r in runs) and not os.path.exists("results/test_recipes/summary.csv"):
    if hours_left() > 1:
        run_parallel([(0, "test_recipes.log", f"python -m evomerge.evaluate --config {CONFIG} "
                       f"--baselines --recipes {' '.join('runs/' + r for r in runs)} --out results/test_recipes")])
for path in ["results/test_baselines/summary.csv", "results/test_recipes/summary.csv"]:
    if os.path.exists(path):
        print(open(path).read())"""),
    ("md", """## 4. Figures"""),
    ("code", """done = [f"runs/{r}" for r in ["ga_seed0", "cmaes_seed0", "random_seed0"] if os.path.exists(f"runs/{r}/evals.jsonl")]
RUN_DIRS = " ".join(done)
!python -m evomerge.plots --runs {RUN_DIRS} --results results/test_baselines results/test_recipes
from IPython.display import Image, display
for f in sorted(os.listdir("figures")):
    display(Image(f"figures/{f}"))
complete = os.path.exists("results/test_recipes/summary.csv")
print("ALL DONE" if complete else "NOT FINISHED: attach this version's Output as an input and Save & Run All again")"""),
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
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_run.ipynb")
with open(path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print(f"wrote {path}")
