# Berlin52 Genetic Algorithm

A compact, reproducible genetic algorithm for the symmetric travelling-salesperson
problem, implemented with NumPy and tested on the TSPLIB Berlin52 instance.

The project uses tournament selection, order crossover, inversion mutation and
elitist population replacement. Distances follow the TSPLIB `EUC_2D` rule:
each edge is rounded to the nearest integer before the tour is summed.

## Result

The reference tour retained from the original experiments evaluates to **7542**,
the published optimum for Berlin52. Earlier repository code reported `7544.37`
because it summed continuous Euclidean edges instead of applying TSPLIB rounding.
The cleaned default solver independently reaches **7905** with seed 2; using
`--target 8000` reaches 7998 after 300 generations in the verified run.

## Run

```bash
python -m pip install -e .
python -m berlin52_ga
```

The default run uses the historically strongest configuration:

```text
population=200, generations=1000, crossover=0.80, mutation=0.05
tournament=4, win_probability=0.75, elites=10, seed=2
```

Override any setting from the command line:

```bash
python -m berlin52_ga --generations 200 --population 100 --seed 7
```

The solver writes nothing unless explicitly requested:

```bash
python -m berlin52_ga --csv artifacts/history.csv
python -m pip install -e ".[plot]"
python -m berlin52_ga --plot artifacts/convergence.png
```

## Test

```bash
python -m unittest discover -s tests -v
```

Tests cover TSPLIB parsing and distance semantics, the known optimum tour,
operator invariants, configuration validation, deterministic runs and early stop.
CI runs on Linux and Windows with Python 3.10 and 3.12.

## Structure

```text
src/berlin52_ga/
  tsplib.py      TSPLIB parser and EUC_2D distances
  tours.py       tour validation, creation and evaluation
  operators.py   selection, crossover and mutation
  solver.py      configuration and genetic-algorithm loop
  reporting.py   opt-in CSV and convergence plots
  cli.py         command-line interface
src/berlin52_ga/data/
  berlin52.tsp    packaged TSPLIB instance
tests/
```

## Scope

This repository deliberately focuses on one complete experiment. The previous
decision-tree, regression and neural-network exercises remain preserved on the
`archive/pre-berlin52-rescue` branch and in Git history. Reinforcement learning
belongs to a separate project and is not claimed here.

Berlin52 originates from
[TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/), maintained by
Gerhard Reinelt.
