"""Black-box optimizers with a shared ask/tell interface, all maximizing fitness.

    opt = make_optimizer(...)
    while not done:
        genomes = opt.ask()
        opt.tell([fitness(g) for g in genomes])

Each optimizer can serialize its state, so a search interrupted between generations (a Kaggle
session ending, say) resumes exactly where it stopped.
"""
import base64
import pickle

import numpy as np


class GeneticAlgorithm:
    """Real-coded generational GA.

    - Selection: k-way tournament.
    - Crossover: BLX-alpha blend crossover (child genes drawn from the parents' range widened by alpha).
    - Mutation: per-gene Gaussian noise with probability mutation_rate; sigma decays every generation.
    - Elitism: the best `elite` individuals survive unchanged and are not re-evaluated.
    - Bounds: genes are clipped to [lo, hi].
    """

    name = "ga"

    def __init__(self, dim, lo, hi, pop_size=16, elite=2, tournament_k=3, crossover_rate=0.9,
                 blx_alpha=0.3, mutation_rate=0.15, mutation_sigma=0.15, sigma_decay=0.95,
                 seed=0, initial=None):
        if not 0 <= elite < pop_size:
            raise ValueError("elite must be in [0, pop_size)")
        self.dim, self.lo, self.hi = dim, lo, hi
        self.pop_size, self.elite, self.tournament_k = pop_size, elite, tournament_k
        self.crossover_rate, self.blx_alpha = crossover_rate, blx_alpha
        self.mutation_rate, self.sigma, self.sigma_decay = mutation_rate, mutation_sigma, sigma_decay
        self.rng = np.random.default_rng(seed)
        self.generation = 0
        self.population = None  # evaluated individuals, shape (pop_size, dim)
        self.fitness = None
        self._pending = self._initial_population(initial)
        self._carried = None  # elites carried into the generation being evaluated

    def _initial_population(self, initial):
        """Seeded individuals (e.g. the two parents) first, the rest uniform at random."""
        seeds = [np.clip(np.asarray(g, dtype=np.float64), self.lo, self.hi) for g in (initial or [])]
        seeds = seeds[: self.pop_size]
        rest = self.rng.uniform(self.lo, self.hi, size=(self.pop_size - len(seeds), self.dim))
        return np.vstack(seeds + [rest]) if seeds else rest

    def ask(self):
        return [g.copy() for g in self._pending]

    def tell(self, fitnesses):
        fitnesses = np.asarray(fitnesses, dtype=np.float64)
        if len(fitnesses) != len(self._pending):
            raise ValueError(f"expected {len(self._pending)} fitness values, got {len(fitnesses)}")
        if self._carried is None:
            pop, fit = self._pending, fitnesses
        else:
            pop = np.vstack([self._carried[0], self._pending])
            fit = np.concatenate([self._carried[1], fitnesses])
        order = np.argsort(-fit, kind="stable")
        self.population, self.fitness = pop[order], fit[order]
        self.generation += 1
        self._breed()

    def _breed(self):
        self._carried = (self.population[: self.elite].copy(), self.fitness[: self.elite].copy())
        children = []
        while len(children) < self.pop_size - self.elite:
            a, b = self._tournament(), self._tournament()
            if self.rng.random() < self.crossover_rate:
                c1, c2 = self._blx(a, b), self._blx(a, b)
            else:
                c1, c2 = a.copy(), b.copy()
            children.extend([self._mutate(c1), self._mutate(c2)])
        self._pending = np.array(children[: self.pop_size - self.elite])
        self.sigma *= self.sigma_decay

    def _tournament(self):
        contenders = self.rng.choice(len(self.population), size=self.tournament_k, replace=False)
        return self.population[contenders.min()]  # population is sorted best-first

    def _blx(self, a, b):
        low, high = np.minimum(a, b), np.maximum(a, b)
        span = high - low
        child = self.rng.uniform(low - self.blx_alpha * span, high + self.blx_alpha * span)
        return np.clip(child, self.lo, self.hi)

    def _mutate(self, genome):
        mask = self.rng.random(self.dim) < self.mutation_rate
        genome = genome + mask * self.rng.normal(0.0, self.sigma, self.dim)
        return np.clip(genome, self.lo, self.hi)

    @property
    def best(self):
        return (self.population[0].copy(), float(self.fitness[0])) if self.population is not None else (None, None)

    def state(self):
        return _encode(self.__dict__)

    def load_state(self, blob):
        self.__dict__.update(_decode(blob))


class CMAES:
    """CMA-ES (via pycma), the optimizer used by the reference paper."""

    name = "cmaes"

    def __init__(self, dim, lo, hi, pop_size=16, sigma0=0.25, seed=0, initial=None):
        import cma

        x0 = np.mean(initial, axis=0) if initial else np.full(dim, (lo + hi) / 2)
        self.es = cma.CMAEvolutionStrategy(
            np.clip(x0, lo, hi), sigma0,
            {"bounds": [lo, hi], "popsize": pop_size, "seed": seed + 1, "verbose": -9},
        )
        self.generation = 0
        self._pending = None
        self._best = (None, None)

    def ask(self):
        self._pending = self.es.ask()
        return [np.asarray(g, dtype=np.float64) for g in self._pending]

    def tell(self, fitnesses):
        self.es.tell(self._pending, [-f for f in fitnesses])  # pycma minimizes
        i = int(np.argmax(fitnesses))
        if self._best[1] is None or fitnesses[i] > self._best[1]:
            self._best = (np.asarray(self._pending[i], dtype=np.float64), float(fitnesses[i]))
        self.generation += 1
        self._pending = None

    @property
    def best(self):
        return self._best

    def state(self):
        return _encode({"es": self.es, "generation": self.generation, "_best": self._best})

    def load_state(self, blob):
        self.__dict__.update(_decode(blob))


class RandomSearch:
    """Uniform random genomes: the control that shows the evolutionary operators matter."""

    name = "random"

    def __init__(self, dim, lo, hi, pop_size=16, seed=0, initial=None):
        self.dim, self.lo, self.hi, self.pop_size = dim, lo, hi, pop_size
        self.rng = np.random.default_rng(seed)
        self.generation = 0
        self._initial = [np.asarray(g, dtype=np.float64) for g in (initial or [])]
        self._pending = None
        self._best = (None, None)

    def ask(self):
        batch = self._initial[: self.pop_size] if self.generation == 0 else []
        extra = self.rng.uniform(self.lo, self.hi, size=(self.pop_size - len(batch), self.dim))
        self._pending = batch + list(extra)
        return [g.copy() for g in self._pending]

    def tell(self, fitnesses):
        i = int(np.argmax(fitnesses))
        if self._best[1] is None or fitnesses[i] > self._best[1]:
            self._best = (self._pending[i].copy(), float(fitnesses[i]))
        self.generation += 1

    @property
    def best(self):
        return self._best

    def state(self):
        return _encode(self.__dict__)

    def load_state(self, blob):
        self.__dict__.update(_decode(blob))


def make_optimizer(name, dim, lo, hi, seed, initial, ga_params=None, cmaes_params=None):
    if name == "ga":
        return GeneticAlgorithm(dim, lo, hi, seed=seed, initial=initial, **(ga_params or {}))
    if name == "cmaes":
        return CMAES(dim, lo, hi, seed=seed, initial=initial, **(cmaes_params or {}))
    if name == "random":
        pop = (ga_params or {}).get("pop_size", 16)
        return RandomSearch(dim, lo, hi, pop_size=pop, seed=seed, initial=initial)
    raise ValueError(f"unknown optimizer {name!r}")


def _encode(obj):
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def _decode(blob):
    return pickle.loads(base64.b64decode(blob))
