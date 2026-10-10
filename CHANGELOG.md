# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
[Semantic Versioning](https://semver.org/) once it reaches 1.0.

## [Unreleased]

### Added
- Config fingerprinting: training refuses to resume into an `out_dir` trained with a different
  config unless `--force` is passed.
- Guard that skips optimizer steps with a non-finite loss or gradient.
- Exact RNG state restore on resume.
- Seeded random-graph control.
- `py.typed` marker.
- CI: Python 3.12, a Windows job, a wheel build job and Dependabot updates.
- Pre-commit hygiene hooks (whitespace, end of file, YAML/TOML checks, large files, merge
  conflicts).

### Changed
- Teacher-cache key includes dtype, top-k and prompt; cache loading is faster.
- Dev split is disjoint from the eval split.
- Stricter config validation: enums, types and circular `base` references.
- Removed the dead fields `data.n_train` and `compressor.writer.kind`.
- LLMLingua-2 is loaded once per run.
- `run_experiments.py` exits non-zero when any run fails.
- Raised floors to `torch>=2.6`; pinned `ruff==0.6.9` in the dev extra.

### Fixed
- Edge penalty under the Gumbel relaxation.
- Checkpoint and cache writes are atomic.

### Security
- Checkpoints are loaded with `weights_only=True`.
- CI actions pinned to commit SHAs, with read-only default permissions.

## [0.1.0] - 2026-10-06

First public release (pre-release: no compressor has been trained yet).

### Added
- Config system: typed YAML schema with inheritance, dotted overrides and strict validation.
- Evaluation harness with SQuAD-style EM and F1, achieved-ratio reporting and retention tables.
- Non-learned baselines: truncation, extractive (lead, TextRank, BM25) and LLMLingua-2.
- Flat-slot Perceiver-style compressor trained by distillation from the full-context LLM, with a
  reconstruction warm-up, a teacher cache and resumable training.
- Routed sparse writing, hard-concrete node gates and an augmented-Lagrangian rate controller.
- Entmax edge induction, relational message passing and a random-walk positional bias, with
  no-edge and random-graph controls.
- Streaming memory with a mass-weighted merge gate.
- Paired-bootstrap analysis for four hypotheses, an edge-alignment probe, figures and a results
  builder.
- Design notes for every milestone in `docs/design/`.

### Status
- No compressor has been fully trained yet, so no performance claim is made for the method.
- Non-learned baselines are evaluated on HotpotQA only (partial); see `docs/results/results.md`.

[Unreleased]: https://github.com/dhrrishitvdeka/gistgraph/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dhrrishitvdeka/gistgraph/releases/tag/v0.1.0
