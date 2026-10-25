import numpy as np
import pytest

from evomerge.optimizers import make_optimizer


def sphere(x):
    return -float(np.sum((x - 0.7) ** 2))


def rastrigin(x):
    z = 4 * (x - 0.7)
    return -float(10 * len(z) + np.sum(z**2 - 10 * np.cos(2 * np.pi * z)))


def run(name, fitness, dim=10, generations=60, seed=0):
    opt = make_optimizer(name, dim, 0.0, 1.2, seed, initial=None,
                         ga_params={"pop_size": 20, "elite": 2}, cmaes_params={"pop_size": 20, "sigma0": 0.3})
    for _ in range(generations):
        genomes = opt.ask()
        assert all(np.all((g >= 0.0) & (g <= 1.2)) for g in genomes)
        opt.tell([fitness(g) for g in genomes])
    return opt.best


@pytest.mark.parametrize("name", ["ga", "cmaes"])
def test_optimizers_solve_sphere(name):
    genome, fit = run(name, sphere)
    assert fit > -1e-2
    assert np.allclose(genome, 0.7, atol=0.06)


def test_ga_beats_random_search_on_rastrigin():
    ga = np.mean([run("ga", rastrigin, seed=s)[1] for s in range(3)])
    rnd = np.mean([run("random", rastrigin, seed=s)[1] for s in range(3)])
    assert ga > rnd


def test_ga_keeps_seeded_parents_and_elites():
    initial = [np.full(5, 1.0), np.zeros(5)]
    opt = make_optimizer("ga", 5, 0.0, 1.2, 0, initial, ga_params={"pop_size": 6, "elite": 2})
    first = opt.ask()
    assert np.array_equal(first[0], initial[0]) and np.array_equal(first[1], initial[1])
    opt.tell([10.0, 9.0, 1, 2, 3, 4])
    assert len(opt.ask()) == 4  # elites are carried over, not re-evaluated
    assert opt.best[1] == 10.0 and np.array_equal(opt.best[0], initial[0])


@pytest.mark.parametrize("name", ["ga", "cmaes", "random"])
def test_state_round_trip_resumes_identically(name):
    a = make_optimizer(name, 6, 0.0, 1.2, 3, None, ga_params={"pop_size": 8}, cmaes_params={"pop_size": 8})
    for _ in range(3):
        a.tell([sphere(g) for g in a.ask()])
    b = make_optimizer(name, 6, 0.0, 1.2, 99, None, ga_params={"pop_size": 8}, cmaes_params={"pop_size": 8})
    b.load_state(a.state())
    for _ in range(3):
        ga_, gb = a.ask(), b.ask()
        assert all(np.allclose(x, y) for x, y in zip(ga_, gb))
        fits = [sphere(g) for g in ga_]
        a.tell(fits)
        b.tell(fits)
    assert a.best[1] == b.best[1]
