"""Experiment configuration, loaded from a JSON file with optional command-line overrides."""
import json
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # Models: two fine-tunes (A, B) of the same base model.
    base_model: str = "Qwen/Qwen2.5-1.5B"
    model_a: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_b: str = "sunblaze-ucb/Qwen2.5-1.5B-GRPO-MATH-1EPOCH"
    # Which parent's config (and so RoPE base) the merged model uses: "a" or "b".
    config_from: str = "a"
    # Optional explicit RoPE base for the merged model; overrides config_from's value.
    rope_theta: float | None = None
    dtype: str = "float16"
    device: str = "cuda"

    # Data: Hindi GSM8K. The search set and test set come from different splits.
    dataset: str = "bingbangboom/gsm8k-hindi"
    search_split: str = "train_main"
    search_n: int = 64
    test_split: str = "test_main"
    test_n: int = 250
    # Validation: the next val_n problems of the search split's shuffle, disjoint from the search set.
    val_n: int = 250
    data_seed: int = 0

    # Generation.
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}."
    max_new_tokens: int = 512  # search: rewards answers that finish within budget
    test_max_new_tokens: int = 1024  # held-out test: room for long (e.g. Hindi) reasoning
    batch_size: int = 32

    # Merging.
    merge_method: str = "task_arithmetic"  # task_arithmetic | ties | dare_ties | dare_linear
    density: float = 0.5  # fraction of each task vector kept by ties / dare
    weight_lo: float = 0.0
    weight_hi: float = 1.2

    # Search.
    seed: int = 0
    max_evals: int = 170
    ga: dict = field(default_factory=lambda: {
        "pop_size": 16,
        "elite": 2,
        "tournament_k": 3,
        "crossover_rate": 0.9,
        "blx_alpha": 0.3,
        "mutation_rate": 0.15,
        "mutation_sigma": 0.15,
        "sigma_decay": 0.95,
    })
    cmaes: dict = field(default_factory=lambda: {"pop_size": 16, "sigma0": 0.25})

    @classmethod
    def load(cls, path=None, overrides=None):
        data = {}
        if path:
            with open(path, encoding="utf-8") as f:
                data = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        for item in overrides or []:
            key, value = item.split("=", 1)
            data[key] = json.loads(value) if _looks_like_json(value) else value
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self):
        return asdict(self)


def _looks_like_json(value):
    try:
        json.loads(value)
        return True
    except ValueError:
        return False
