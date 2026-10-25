"""Task-vector size per parameter group: is a pair of models mergeable at all?

    python -m evomerge.diagnose --config configs/qwen_hindi_math.json --out results/task_vectors.csv

Prints ||theta_k - base|| / ||base|| for each parent and group. Light fine-tunes (SFT, RL) change
a model by about 1%; merging works only between such parents. A parent with a relative task
vector near 1 has drifted into a different model, and mixing it in breaks the merge.
"""
import argparse
import csv
import os

from .config import Config
from .merge import group_names, load_state_dict, param_group


def task_vector_norms(base, parents, num_layers):
    sums = {}
    for name, tensor in base.items():
        if name == "lm_head.weight" and "model.embed_tokens.weight" in base:
            continue  # tied to the embeddings in these models
        if any(name not in p for p in parents):
            continue
        g = param_group(name, num_layers)
        b = tensor.float()
        s = sums.setdefault(g, [0.0] * (len(parents) + 1))
        s[0] += b.pow(2).sum().item()
        for k, p in enumerate(parents):
            s[k + 1] += (p[name].float() - b).pow(2).sum().item()
    return {g: [(s[k + 1] / s[0]) ** 0.5 for k in range(len(parents))] for g, s in sorted(sums.items())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config")
    parser.add_argument("--out", default="results/task_vectors.csv")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    cfg = Config.load(args.config, args.set)

    from transformers import AutoConfig

    num_layers = AutoConfig.from_pretrained(cfg.base_model).num_hidden_layers
    base = load_state_dict(cfg.base_model)
    parents = [load_state_dict(cfg.model_a), load_state_dict(cfg.model_b)]
    norms = task_vector_norms(base, parents, num_layers)
    names = group_names(num_layers)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group", cfg.model_a, cfg.model_b])
        for g, (a, b) in norms.items():
            writer.writerow([names[g], f"{a:.5f}", f"{b:.5f}"])
    print(f"{'group':8s} {'A':>8s} {'B':>8s}")
    for g, (a, b) in norms.items():
        print(f"{names[g]:8s} {a:8.4f} {b:8.4f}")
    print(f"written to {args.out}")


if __name__ == "__main__":
    main()
