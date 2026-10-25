"""Layer-wise model merging in parameter space.

Every parameter belongs to a group: the embeddings, one of the transformer layers, or the final
norm/head. A genome holds one weight per (parent, group), so the merged parameter is

    theta = base + sum_k  w[k, group] * f(theta_k - base)

where theta_k - base is parent k's "task vector" and f is the identity (task arithmetic), DARE
(random drop and rescale) or TIES trimming followed by sign election.
"""
import glob
import json
import os
import re

import numpy as np
import torch

METHODS = ("task_arithmetic", "ties", "dare_ties", "dare_linear")
CHUNK_ELEMENTS = 1 << 24  # merge large tensors in row chunks of about 16M elements
QUANTILE_SAMPLE = 1 << 21

_LAYER = re.compile(r"\.layers\.(\d+)\.")


def param_group(name, num_layers):
    """0 = embeddings, 1..L = transformer layers, L + 1 = final norm and lm_head."""
    match = _LAYER.search(name)
    if match:
        return int(match.group(1)) + 1
    if "embed_tokens" in name:
        return 0
    return num_layers + 1


def group_names(num_layers):
    return ["embed"] + [f"layer{i}" for i in range(num_layers)] + ["final"]


def load_state_dict(model_path):
    """Load a checkpoint's safetensors shards into a CPU state dict."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    if not os.path.isdir(model_path):
        model_path = snapshot_download(model_path, allow_patterns=["*.json", "*.safetensors"])
    state = {}
    for shard in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        state.update(load_file(shard, device="cpu"))
    if not state:
        raise FileNotFoundError(f"no safetensors weights in {model_path}")
    return state


class LayerwiseMerger:
    """Writes merges of a base model and K fine-tunes into a live model's parameters."""

    def __init__(self, model, base_state, parent_states, num_layers, method="task_arithmetic",
                 density=0.5, seed=0):
        if method not in METHODS:
            raise ValueError(f"unknown merge method {method!r}; expected one of {METHODS}")
        self.model = model
        self.base = base_state
        self.parents = parent_states
        self.num_layers = num_layers
        self.num_groups = num_layers + 2
        self.method = method
        self.density = density
        self.seed = seed
        self.params = [(name, p) for name, p in model.named_parameters()]
        for name, p in self.params:
            for label, state in [("base", base_state)] + [(f"parent {k}", s) for k, s in enumerate(parent_states)]:
                if name not in state:
                    raise KeyError(f"{name} missing from {label} checkpoint")
                if tuple(state[name].shape) != tuple(p.shape):
                    raise ValueError(f"{name}: {label} shape {tuple(state[name].shape)} != model {tuple(p.shape)}")
        self._thresholds = {}

    @property
    def genome_size(self):
        return len(self.parents) * self.num_groups

    def weights_from_genome(self, genome):
        genome = np.asarray(genome, dtype=np.float64)
        if genome.shape != (self.genome_size,):
            raise ValueError(f"genome has shape {genome.shape}, expected ({self.genome_size},)")
        return genome.reshape(len(self.parents), self.num_groups)

    def uniform_genome(self, weights):
        """Genome that gives parent k the same weight weights[k] in every group."""
        return np.repeat(np.asarray(weights, dtype=np.float64), self.num_groups)

    @torch.no_grad()
    def apply(self, genome):
        weights = self.weights_from_genome(genome)
        for index, (name, param) in enumerate(self.params):
            group = param_group(name, self.num_layers)
            w = [float(weights[k, group]) for k in range(len(self.parents))]
            self._merge_param(index, name, param, w)

    def _merge_param(self, index, name, param, w):
        device = param.device
        base = self.base[name]
        parents = [s[name] for s in self.parents]
        thresholds = self._ties_thresholds(name, base, parents) if self.method == "ties" else None
        flat = param.data.view(param.shape[0], -1) if param.dim() > 1 else param.data.view(-1, 1)
        rows = flat.shape[0]
        step = max(1, CHUNK_ELEMENTS // max(1, flat.shape[1]))
        for start in range(0, rows, step):
            stop = min(rows, start + step)
            b = _rows(base, start, stop).to(device, torch.float32)
            deltas = [_rows(t, start, stop).to(device, torch.float32) - b for t in parents]
            if self.method in ("dare_ties", "dare_linear"):
                deltas = [self._dare(d, index, k, start) for k, d in enumerate(deltas)]
            elif self.method == "ties":
                deltas = [torch.where(d.abs() >= thresholds[k], d, torch.zeros_like(d)) for k, d in enumerate(deltas)]
            weighted = [wk * d for wk, d in zip(w, deltas)]
            if self.method in ("ties", "dare_ties"):
                merged = _disjoint_sum(weighted)
            else:
                merged = torch.stack(weighted).sum(0)
            flat[start:stop].copy_((b + merged).view(stop - start, -1).to(param.dtype))

    def _dare(self, delta, index, k, start):
        gen = torch.Generator(device=delta.device)
        gen.manual_seed(hash((self.seed, index, k, start)) & 0x7FFFFFFF)
        keep = torch.rand(delta.shape, generator=gen, device=delta.device) < self.density
        return torch.where(keep, delta / self.density, torch.zeros_like(delta))

    def _ties_thresholds(self, name, base, parents):
        """Per-parent magnitude cut-off keeping the top `density` fraction of the task vector."""
        if name not in self._thresholds:
            values = []
            for k, t in enumerate(parents):
                delta = (t.float() - base.float()).abs().flatten()
                if delta.numel() > QUANTILE_SAMPLE:
                    gen = torch.Generator().manual_seed(self.seed * 1000 + k)
                    delta = delta[torch.randint(delta.numel(), (QUANTILE_SAMPLE,), generator=gen)]
                values.append(float(torch.quantile(delta, 1.0 - self.density)))
            self._thresholds[name] = values
        return self._thresholds[name]


def _rows(tensor, start, stop):
    if tensor.dim() > 1:
        return tensor[start:stop]
    return tensor.view(-1, 1)[start:stop]


def _disjoint_sum(weighted):
    """TIES sign election: keep each entry only from the parents that agree with the summed sign."""
    stacked = torch.stack(weighted)
    sign = torch.sign(stacked.sum(0))
    agree = torch.sign(stacked) == sign
    return torch.where(agree, stacked, torch.zeros_like(stacked)).sum(0)


def save_recipe(path, genome, num_layers, parent_names, method, density):
    weights = np.asarray(genome, dtype=np.float64).reshape(len(parent_names), num_layers + 2)
    recipe = {
        "method": method,
        "density": density,
        "parents": parent_names,
        "groups": group_names(num_layers),
        "weights": {p: weights[k].round(6).tolist() for k, p in enumerate(parent_names)},
        "genome": np.asarray(genome).round(6).tolist(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, indent=2)
