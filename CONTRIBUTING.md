# Contributing

Thanks for your interest in Gist Graph. This is a small research project, so the bar for a good
contribution is mostly: it is correct, it is tested, and it is honest about what it shows.

## Getting set up

```bash
git clone https://github.com/dhrrishitvdeka/gistgraph.git
cd gistgraph
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
ruff check . && ruff format --check .
```

The tests use a tiny randomly initialised model and need no GPU and no downloads. Training or
evaluating with the real model (`Qwen/Qwen2.5-0.5B-Instruct`) needs a CUDA GPU.

## Making a change

1. Open an issue first for anything larger than a bug fix, so the direction can be agreed.
2. Branch from `main`. Keep one logical change per commit.
3. Write tests with the change. Shapes, masking and gradient flow are checked for every module, and
   structural properties are checked directly (for example, routing is sparse and message passing is
   permutation-equivariant). New modules should follow that pattern.
4. Run `pytest` and `ruff check . && ruff format --check .` before you push.
5. Update the design note in `docs/design/` if you change how something works or why. Each note
   explains the design, the maths and the reasons for the choices, and should stay accurate.

### Commit messages

[Conventional Commits](https://www.conventionalcommits.org/) with an imperative subject of at most 72
characters: `feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `perf:`, `chore:`, `ci:`. Explain *why* in
the body when it is not obvious.

## Experiments and results

- Experiments are configs in `configs/experiments/` and run through `python -m gistgraph`. Do not add
  results that cannot be regenerated from a config, a seed and a command.
- Do not commit `runs/`, `cache/`, model weights or datasets.
- Report negative and null results. A comparison whose confidence interval includes zero is a
  result, and the results page should say so plainly.
- State what a number does not cover (seeds, subset sizes, context cuts). `docs/results/results.md`
  has a standing list of caveats; extend it rather than hiding a limitation.

## Reporting bugs

Use the bug report template. Include the config (`config.resolved.yaml` is written next to every
run), the command, the seed, and the full error.

## Code style

Python 3.10+, `ruff` for linting and formatting (line length 100), type hints on public functions,
and short docstrings that explain *why* when the code does not make it obvious.

## Licence

By contributing you agree that your contribution is licensed under the Apache License 2.0, the same
as the project.
