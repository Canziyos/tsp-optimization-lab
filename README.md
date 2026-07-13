# TSP Optimization Lab

A reproducible comparison of classical and nature-inspired solvers for symmetric
travelling-salesperson problems. The current experiment uses the TSPLIB Berlin52
instance and one shared implementation of TSPLIB `EUC_2D` distance semantics.

## Implemented solvers

| Solver | Role | Stochastic |
|---|---|---|
| Nearest neighbour | Fast constructive baseline | No |
| 2-opt | Best-improvement local search, initialized by nearest neighbour | No |
| Genetic algorithm | Population-based evolutionary solver | Yes |
| Ant colony optimization | Rank-based pheromone search | Yes |
| Hybrid GA + 2-opt | Memetic search with periodic local refinement | Yes |

All solvers use the same result model, evaluator and benchmark rather than
separate experiment scripts.

## Verified Berlin52 results

- Reference optimum retained from the original experiments: **7542**
- Nearest-neighbour baseline: **8980**
- Nearest-neighbour + 2-opt: **7842** after 11 improving moves
- Genetic algorithm, seed 2: **7905** after 1000 generations
- Genetic algorithm with `--target 8000`: **7998** at generation 300
- Ant colony optimization, seed 2: **7662** at iteration 86
- Hybrid GA + 2-opt, seeds 1 and 2: **7542**, the reference optimum

The default 1000-generation comparison produced GA distances of 8291, 8251 and
7905 for seeds 0, 1 and 2. The deterministic 2-opt baseline therefore beats all
three standalone GA runs while taking only a small fraction of their runtime.
Default ACO runs reached 7717, 7727 and 7662 for seeds 0, 1 and 2. These pure
swarm runs beat the deterministic 2-opt baseline without applying local search.
Default hybrid runs reached 7762, 7542 and 7542 for seeds 0, 1 and 2. The hybrid
starts from the deterministic 2-opt tour, periodically refines its strongest GA
offspring, and retains the standalone GA unchanged as a fair control.

The old repository reported the reference tour as `7544.37` because it summed
continuous Euclidean edges. TSPLIB rounds every edge before summing it.

## Install and run

```bash
python -m pip install -e .
python -m tsp_optimization_lab solve nearest
python -m tsp_optimization_lab solve two-opt
python -m tsp_optimization_lab solve ga --seed 2
python -m tsp_optimization_lab solve aco --seed 2
python -m tsp_optimization_lab solve hybrid --seed 2
```

The installed command offers the same interface:

```bash
tsp-lab solve ga --generations 500 --seed 7
```

Solvers write nothing unless an output is requested:

```bash
python -m tsp_optimization_lab solve ga --csv artifacts/ga-history.csv
python -m pip install -e ".[plot]"
python -m tsp_optimization_lab solve ga --plot artifacts/ga-convergence.png
```

## Benchmark

Compare deterministic baselines once and run GA, ACO and the hybrid across seeds:

```bash
python -m tsp_optimization_lab benchmark --seeds 0 1 2
python -m tsp_optimization_lab benchmark --seeds 0 1 2 \
  --csv artifacts/comparison.csv
```

Every record uses the same instance and distance matrix and reports algorithm,
seed, best distance, best step and wall-clock runtime. Runtime is descriptive;
algorithms do not yet share an equal evaluation budget.

## Test

```bash
python -m unittest discover -s tests -v
```

Tests cover parsing, known-optimum distance, tour and operator invariants,
determinism, 2-opt monotonicity, benchmarking, and competitive Berlin52 GA and
ACO and hybrid results. CI runs on Linux and Windows with Python 3.10 and 3.12.

## Structure

```text
src/tsp_optimization_lab/
  models.py       shared history and result contract
  tsplib.py       parser and EUC_2D distance matrix
  tours.py        tour validation and evaluation
  nearest.py      nearest-neighbour baseline
  two_opt.py      best-improvement local search
  genetic.py      genetic algorithm
  operators.py    GA selection, crossover and mutation
  ant_colony.py   pheromone construction, evaporation and ranked deposits
  hybrid.py       GA orchestration with periodic 2-opt refinement
  benchmark.py    multi-solver timing and quality runs
  cli_options.py  solver arguments and typed configuration construction
  reporting.py    opt-in CSV and convergence plots
  cli.py          solve and benchmark commands
```

Berlin52 originates from
[TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/), maintained by
Gerhard Reinelt.
