import numpy as np
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from evomerge import merge as merge_mod
from evomerge.merge import LayerwiseMerger, param_group

LAYERS = 3


def tiny_state(seed):
    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=LAYERS,
                      num_attention_heads=4, num_key_value_heads=2, tie_word_embeddings=True)
    model = Qwen2ForCausalLM(cfg)
    return model, {k: v.clone() for k, v in model.state_dict().items()}


@pytest.fixture()
def setup():
    model, base = tiny_state(0)
    _, a = tiny_state(1)
    _, b = tiny_state(2)
    return model, base, a, b


def params(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def test_param_groups():
    assert param_group("model.embed_tokens.weight", LAYERS) == 0
    assert param_group("model.layers.0.self_attn.q_proj.weight", LAYERS) == 1
    assert param_group("model.layers.2.mlp.up_proj.weight", LAYERS) == 3
    assert param_group("model.norm.weight", LAYERS) == LAYERS + 1


@pytest.mark.parametrize("weights,expect", [([0, 0], "base"), ([1, 0], "a"), ([0, 1], "b")])
def test_parent_genomes_reproduce_the_parents(setup, weights, expect):
    model, base, a, b = setup
    merger = LayerwiseMerger(model, base, [a, b], LAYERS)
    merger.apply(merger.uniform_genome(weights))
    target = {"base": base, "a": a, "b": b}[expect]
    for name, p in params(model).items():
        assert torch.allclose(p, target[name], atol=1e-6), name


def test_task_arithmetic_matches_formula_per_layer(setup):
    model, base, a, b = setup
    merger = LayerwiseMerger(model, base, [a, b], LAYERS)
    genome = np.random.default_rng(0).uniform(0, 1.2, merger.genome_size)
    merger.apply(genome)
    w = genome.reshape(2, LAYERS + 2)
    for name, p in params(model).items():
        g = param_group(name, LAYERS)
        expected = base[name] + w[0, g] * (a[name] - base[name]) + w[1, g] * (b[name] - base[name])
        assert torch.allclose(p, expected.float(), atol=1e-5), name


def test_chunked_merge_equals_unchunked(setup, monkeypatch):
    model, base, a, b = setup
    genome = np.linspace(0, 1.2, 2 * (LAYERS + 2))
    LayerwiseMerger(model, base, [a, b], LAYERS).apply(genome)
    whole = params(model)
    monkeypatch.setattr(merge_mod, "CHUNK_ELEMENTS", 50)
    LayerwiseMerger(model, base, [a, b], LAYERS).apply(genome)
    for name, p in params(model).items():
        assert torch.allclose(p, whole[name], atol=1e-6), name


def test_ties_keeps_only_sign_agreeing_entries(setup):
    model, base, a, b = setup
    merger = LayerwiseMerger(model, base, [a, b], LAYERS, method="ties", density=1.0)
    merger.apply(merger.uniform_genome([1.0, 1.0]))
    name = "model.layers.0.mlp.up_proj.weight"
    da, db = a[name] - base[name], b[name] - base[name]
    sign = torch.sign(da + db)
    expected = base[name] + torch.where(torch.sign(da) == sign, da, 0) + torch.where(torch.sign(db) == sign, db, 0)
    assert torch.allclose(dict(model.named_parameters())[name].detach(), expected, atol=1e-5)


def test_dare_is_deterministic_and_drops_entries(setup):
    model, base, a, b = setup
    merger = LayerwiseMerger(model, base, [a, b], LAYERS, method="dare_linear", density=0.3, seed=7)
    merger.apply(merger.uniform_genome([1.0, 0.0]))
    first = params(model)
    merger.apply(merger.uniform_genome([1.0, 0.0]))
    name = "model.layers.1.mlp.down_proj.weight"
    assert torch.equal(first[name], params(model)[name])
    kept = (first[name] != base[name]).float().mean().item()
    assert 0.15 < kept < 0.45


def test_shape_mismatch_is_rejected(setup):
    model, base, a, b = setup
    b = dict(b)
    b["model.norm.weight"] = torch.ones(5)
    with pytest.raises(ValueError):
        LayerwiseMerger(model, base, [a, b], LAYERS)
