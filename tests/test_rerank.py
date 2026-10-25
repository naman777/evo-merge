import json

from evomerge.rerank import all_candidates, top_candidates


def make_run(tmp_path, name, fitnesses, genomes=None):
    run = tmp_path / name
    run.mkdir()
    (run / "config.json").write_text(json.dumps({"optimizer": "ga", "config": {"merge_method": "task_arithmetic"}}))
    with open(run / "evals.jsonl", "w") as f:
        for i, fit in enumerate(fitnesses):
            genome = genomes[i] if genomes else [float(i)] * 4
            f.write(json.dumps({"eval": i + 1, "fitness": fit, "genome": genome}) + "\n")
    return str(run)


def test_top_candidates_best_first_ties_earliest(tmp_path):
    run = make_run(tmp_path, "ga_seed0", [0.2, 0.3, 0.3, 0.1, 0.25])
    picked = top_candidates(run, 3)
    assert [e["eval"] for e in picked] == [2, 3, 5]


def test_top_candidates_skips_duplicate_genomes(tmp_path):
    genomes = [[0.5] * 4, [0.5] * 4, [0.1] * 4]
    run = make_run(tmp_path, "ga_seed0", [0.3, 0.3, 0.2], genomes)
    picked = top_candidates(run, 3)
    assert [e["eval"] for e in picked] == [1, 3]


def test_all_candidates_and_shards_cover_everything_once(tmp_path):
    runs = [make_run(tmp_path, n, [0.1 * i for i in range(12)]) for n in ["ga_seed0", "cmaes_seed0"]]
    cands = all_candidates(runs, 10, ["average", "task_arithmetic"])
    assert len(cands) == 22
    ids = [c[0] for c in cands]
    assert len(set(ids)) == 22 and "baseline/average" in ids and "ga_seed0/eval12" in ids
    shards = [[c for i, c in enumerate(cands) if i % 2 == s] for s in range(2)]
    assert sorted(c[0] for s in shards for c in s) == sorted(ids)
