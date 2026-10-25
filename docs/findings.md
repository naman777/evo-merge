# Findings log

## 2026-10-24: Qwen2.5-Math-1.5B-Instruct cannot be merged with Qwen2.5-1.5B-Instruct

First choice for the math parent was the official `Qwen/Qwen2.5-Math-1.5B-Instruct`. A smoke run on
8 Hindi GSM8K training problems (greedy decoding, 384 new tokens, RTX 3050) showed that no merge
with it works.

**Each parent only works under its own RoPE base.** The Math model uses rope_theta = 10,000 while the
base and Instruct models use 1,000,000. A merged model has a single config.

| Merged model's config | Instruct alone | Math alone | 50/50 average |
|---|---|---|---|
| Instruct (theta 1e6) | 3/8 | 0/8 (off-topic text) | 0/8 (no answer) |
| Math (theta 1e4) | 0/8 | 1/8 | 0/8 |

**The Math model's task vector is as large as the model itself.** Relative size
||theta_k - base|| / ||base||, per parameter group:

| Group | Instruct | Math |
|---|---|---|
| embeddings | 0.028 | 1.08 |
| layers 0-27 | 0.002 - 0.012 | 0.53 - 1.41 |
| final norm | 0.000 | 0.32 |

Instruct is a light fine-tune (about 1% weight change). Qwen2.5-Math went through large-scale
continued pretraining, so it no longer shares a low-loss region with the base, and interpolation
breaks the model. Adding even a tenth of the Math task vector already hurts. Adding a quarter
produces repeated tokens ("The The The ...").

| Weights (Instruct, Math) | Accuracy, theta 1e6 | Accuracy, theta 1e4 |
|---|---|---|
| (1, 0.1) | 2/8 | 1/8 |
| (1, 0.25) | 0/8, degenerate | 0/8, degenerate |
| (1, 0.5) | 0/8, degenerate | 0/8, degenerate |
| (0.25, 1) | 0/8 | 0/8 |
| (0.5, 1) | 0/8 | 0/8 |

**Consequence.** Merging needs parents that are light fine-tunes of the same base. The math parent
was replaced by `sunblaze-ucb/Qwen2.5-1.5B-GRPO-MATH-1EPOCH`: GRPO reinforcement learning on MATH,
starting from `Qwen/Qwen2.5-1.5B`, with the same RoPE base (Zhao et al., "Learning to Reason without
External Rewards", arXiv 2505.19590). `python -m evomerge.diagnose` reproduces the table above
for any pair (`configs/qwen_official_math.json` holds the failing one).

This matches the paper's own setup: Akiba et al. merged fine-tunes of a shared Mistral-7B base.

## 2026-10-24: the replacement pair is mergeable

Relative task vectors (`results/task_vectors.csv`): Instruct 0.002-0.012 per layer (0.028 for the
embeddings), GRPO-MATH 0.0001-0.0005. RL fine-tuning moves the weights about 25x less than
instruction tuning, so the search bounds were widened to [0, 2], which lets the GA scale a task
vector up.

Smoke check on the same 8 problems (an indication only; real numbers come from the 64-problem search
set and the 250-problem test set):

| Model | Correct |
|---|---|
| Instruct (A) | 3/8 |
| GRPO-MATH (B) | 1/8 |
| 50/50 average | 2/8 |
| Task arithmetic (1, 1) | 3/8 |
| TIES (1, 1) | 3/8 |
| DARE-TIES (1, 1) | 3/8 |

Unlike the official Math model, every merge produces coherent step-by-step solutions. Each
evaluation of 8 problems takes about 35-40 s on an RTX 3050 (bf16).

## 2026-10-24: 64-problem pilot on the search set

Setup: `configs/qwen_hindi_math.json` (fp16, greedy decoding, at most 512 new tokens), the 64
`train_main` problems the search uses, RTX 3050. Each candidate took 4.5-6.5 minutes.
Raw outputs are in `results/pilot/`.

| Candidate | Correct | Accuracy |
|---|---|---|
| Instruct (A) | 9/64 | 14.1% |
| GRPO-MATH (B) | 12/64 | 18.8% |
| **50/50 average** | **16/64** | **25.0%** |
| Task arithmetic (1, 1) | 13/64 | 20.3% |
| TIES (1, 1), density 0.5 | 13/64 | 20.3% |
| DARE-TIES (1, 1), density 0.5 | 15/64 | 23.4% |

**Merging helps.** Every merge beats both parents, and the plain average beats the better parent by
6.2 points (4 problems). The parents are complementary: A alone solves 2 problems that B misses, B
alone solves 5 that A misses, and both solve 7. The average also solves 6 problems that neither
parent solves. At least one of the six candidates solves 22 of the 64 problems (34%), which suggests
there is room above the best fixed recipe for the GA to find. With 64 problems, one problem is worth
1.6 points, so differences of 1-2 problems are noise.

**The 512-token limit is the main source of errors.**

| Candidate | Hit the limit | Correct among those | Accuracy on the rest |
|---|---|---|---|
| Instruct (A) | 27/64 | 0 | 9/37 = 24% |
| GRPO-MATH (B) | 14/64 | 0 | 12/50 = 24% |
| 50/50 average | 10/64 | 0 | 16/54 = 30% |

Instruct often reasons in Hindi (41 of its 64 answers are mostly Devanagari), and Hindi needs
several times more tokens than English, so it is cut off most often. The GRPO parent and the merges
reason in English and finish more often. Part of the merge's gain is therefore concision: it reads
the Hindi question but answers in shorter English steps.

Implication: under a 512-token budget the GA's fitness rewards both correctness and finishing in
time. That is a legitimate objective for the search, but the held-out test comparison should use a
larger budget (e.g. 1024 tokens), so the parents are not penalized mainly for verbosity.

Format: answers use `\boxed{}` in 18/64 (A), 53/64 (B) and 54/64 (average) of cases. The rest
fall back to the last number in the text, which is how A's Hindi answers get scored.

Decision: the search keeps a 512-token budget (`max_new_tokens`). The held-out test evaluation uses
1024 tokens (new config option `test_max_new_tokens`). Every result file records the budget it was
scored with.

## 2026-10-25: full run on Kaggle (2x T4) and held-out test results

Kaggle notebook `evo-merge-run`: version 1 (24 Oct, 16:15 to about 02:50, about 10.5 h) ran all
three searches and the baseline tests. Version 2 (25 Oct, about 45 min) resumed from version 1's
output and tested the evolved recipes. Raw files are in `runs/*_seed0/`, `results/test_baselines/`,
`results/test_recipes/` and `logs/`; figures are in `figures/final/`.

Setup: 60-gene genome in [0, 2], task-arithmetic merge, fitness = accuracy on 64 Hindi GSM8K
training problems (512 new tokens), budget of 170 evaluations (CMA-ES and random search finish
their last generation, so 176). Population 16 for every optimizer. Each fitness evaluation took
about 90 s on a T4. Test: 250 `test_main` problems, 1024 new tokens.

### Searches

| Optimizer | Best search fitness | Reached at eval | Mean fitness, last 40 evals |
|---|---|---|---|
| GA (ours) | 32.8% | 36 | 29.8% |
| CMA-ES | 39.1% | 139 | 31.3% |
| Random search | 37.5% | 160 | 26.3% |

Population diversity (mean per-gene standard deviation within a generation):

| Generation | 0 | 3 | 6 | 8 | 9 | 11 |
|---|---|---|---|---|---|---|
| GA | 0.57 | 0.36 | 0.27 | 0.20 | 0.05 | 0.04 |
| CMA-ES | 0.23 | 0.20 | 0.19 | 0.19 | 0.18 | (10: 0.18) |

- **The GA exploits but converges prematurely.** Its generation mean rose from 25.4% to 30.5% (random
  search stayed at about 26%), but diversity collapsed at generation 9 and the best never improved
  after evaluation 36. Likely causes: mutation sigma decaying 5% per generation (0.15 to about 0.085),
  tournament size 3 with only 16 individuals, and elitism of 2. Fixing this is a natural follow-up
  (no sigma decay, a larger population, or a restart when diversity drops).
- **CMA-ES keeps its step size** (std about 0.18 throughout) and kept improving until the end, which is
  why the paper uses it.
- GA and random search share their first generation (same seed), so their curves match up to
  evaluation 16. This makes the comparison fair.

### Held-out test (250 problems)

| Candidate | Test accuracy | Search-set accuracy (64) |
|---|---|---|
| Instruct (A) | 18.0% | 14.1% |
| GRPO-MATH (B) | 22.0% | 18.8% |
| **50/50 average** | **26.0%** | 25.0% |
| **Task arithmetic (1, 1)** | **26.0%** | 20.3% |
| TIES | 24.4% | 20.3% |
| DARE-TIES | 24.0% | 23.4% |
| GA, evolved | 22.4% | 32.8% |
| CMA-ES, evolved | 21.2% | 39.1% |
| Random search, best | 21.6% | 37.5% |

**Main result: the evolved recipes do not beat simple merging on unseen problems.** All three score
below the plain 50/50 average and task arithmetic, and the higher a recipe's search score, the larger
its drop on the test set (GA -10.4, random -15.9, CMA-ES -17.9 points). The fixed merges barely
move between the two sets. See `figures/final/generalization.png`.

**Cause: overfitting a noisy fitness function (the winner's curse).** On 64 problems the standard
error of one accuracy is about 5.4 points, and each search compares about 170 candidates. The best of
170 noisy scores is mostly the candidate whose noise happened to be largest, not the best recipe. A
search with 60 free weights and only 64 problems has far more freedom than its fitness signal
supports.

**How certain (exact McNemar test, paired on the same 250 problems):**

| Comparison | Only the first right | Only the second right | p |
|---|---|---|---|
| average vs Instruct (A) | 35 | 15 | 0.007 |
| average vs GRPO-MATH (B) | 25 | 15 | 0.15 |
| average vs GA | 26 | 17 | 0.22 |
| average vs CMA-ES | 25 | 13 | 0.07 |
| average vs random | 29 | 18 | 0.14 |
| GA vs CMA-ES | 20 | 17 | 0.74 |

Only "merging beats the Instruct parent" is significant at 5%. The evolved recipes being worse than
the average is consistent across all three (none is better), but no single gap is significant. GA vs
CMA-ES on the test set is a tie.

Truncation does not explain the gap: with 1024 tokens only 2-8 of 250 answers hit the limit for
every merge (47 for Instruct alone), and nearly all merged answers end in `\boxed{}`.

### What this means for the report

1. Merging two light fine-tunes works: every merge beats both parents on the test set (+4 points
   over the better parent for the average and task arithmetic).
2. Evolutionary search finds recipes that look much better on its fitness set, but with a 64-problem
   fitness they do not generalize. The bottleneck is the size of the fitness set, not the optimizer.
3. The GA's design matters: diversity collapse, compared with CMA-ES adapting its step size.

Possible follow-ups, roughly cheapest first:
- re-rank each search's top 10 candidates on a separate validation set before picking the winner
- a larger fitness set (e.g. 256 problems) with fewer evaluations
- a GA without sigma decay, or with restarts
- a lower-dimensional genome (e.g. 2 x 4 layer blocks instead of 2 x 30)

## 2026-10-25: re-ranking on a validation set

Kaggle notebook `evo-merge-rerank` (about 3.5 h on 2x T4; `evomerge/rerank.py`). Each search's 10 best
distinct candidates by search fitness, plus the 50/50 average and task arithmetic, were scored on a
**validation set**: 250 `train_main` problems disjoint from the 64-problem search set, with the
test-time budget of 1024 tokens. Each search's winner was re-picked by validation score and then
tested on the same 250 test problems as before. Files: `results/rerank/` and `results/test_reranked_*/`.

### Validation scores (250 problems)

| Candidates | Search fitness | Validation |
|---|---|---|
| GA top 10 | 31.2-32.8% | 20.8-26.8% |
| CMA-ES top 10 | 35.9-39.1% | 19.6-25.2% |
| Random top 10 | 32.8-37.5% | 20.0-27.2% |
| **All 30, mean** | 34.0% | **23.9%** |
| 50/50 average | (25.0%) | 24.0% |
| Task arithmetic | (20.3%) | 23.6% |

**Search fitness carries no information about validation accuracy.** Across the 30 candidates the
correlation between search score and validation score is r = -0.05. The candidates' mean validation
accuracy (23.9%) equals the plain average's (24.0%). Among the top candidates, the differences the
searches "found" were noise from the 64-problem fitness, not better recipes.

### Re-picked winners on the test set

| Search | Original pick: test | Re-picked by validation | Its validation | Its test |
|---|---|---|---|---|
| GA | 22.4% (eval 36) | eval 154 | 26.8% | 20.8% |
| CMA-ES | 21.2% (eval 139) | eval 41 | 25.2% | **24.0%** |
| Random | 21.6% (eval 160) | eval 97 | 27.2% | 20.0% |
| *50/50 average* | | | *24.0%* | ***26.0%*** |
| *Task arithmetic* | | | *23.6%* | ***26.0%*** |

Re-ranking helped CMA-ES (+2.8 points) and hurt GA (-1.6) and random search (-1.6). No re-picked
winner reaches the plain average. Picking the best of 10 on a 250-problem validation set is a milder
version of the same winner's curse: the standard error is about 2.7 points, so the top validation
scores (26.8%, 27.2%) were partly luck too, and they fell back to 20-24% on test.

### Conclusion of the experiment

- **Merging two light fine-tunes helps.** The plain average and task arithmetic beat both parents
  (26.0% vs 22.0% and 18.0% on test; significant against the Instruct parent, p = 0.007).
- **Per-layer evolutionary search did not find anything better than uniform merging** in this setting
  (1.5B models, 60 weights, fitness from 64 problems, 170 evaluations). The apparent search gains of
  +8 to +14 points were entirely noise, and selecting on a separate validation set confirms it: search
  fitness does not predict held-out accuracy (r = -0.05), and the evolved recipes score the same as
  uniform merging on average.
- **Lesson:** with a noisy fitness, an evolutionary optimizer optimizes the noise. Merge-recipe search
  needs a fitness set large enough that its noise (about 5.4 points at n=64) is smaller than the
  effects being searched for. Here the true differences between recipes appear to be only a few points.
- **GA vs CMA-ES:** CMA-ES explores better (it keeps its step size; the GA collapsed at generation 9).
  With a fitness this noisy, neither optimizer's final recipe generalized, and on test they tie.

Natural next experiments if there were more compute: a fitness set of 500+ problems (fewer
evaluations, each about 8x slower), a lower-dimensional genome (e.g. 2 x 4 layer blocks), or the
paper's data-flow-space merging.
